from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cmg.constants import INDEX_TO_PHASE
from cmg.data import sequence_collate_fn
from cmg.data.tactile import compute_measured_force_curve, ema_filter, load_tactile_array
from cmg.evaluation import prepare_evaluation_context, resolve_path
from cmg.training import move_to_device

try:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
except ImportError as exc:  # pragma: no cover
    raise RuntimeError('matplotlib is required for visualize_policy_force.py. Please install project dependencies.') from exc


PHASE_COLORS = {
    'Water': '#2a6f97',
    'Interface': '#f4a261',
    'Air': '#8ab17d',
}
PRED_COLOR = '#c1121f'
TARGET_COLOR = '#2b2d42'
REFERENCE_COLOR = '#6d597a'
MEASURED_COLOR = '#264653'
EMA_COLOR = '#457b9d'


def resolve_data_path(project_root: Path, relative_path: str) -> Path:
    path = Path(relative_path)
    if path.is_absolute():
        return path
    if path.parts and path.parts[0] == 'data':
        return project_root / path
    return project_root / 'data' / path


def finite_or_nan(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float('nan')
    return result if np.isfinite(result) else float('nan')


def collect_event_times(sample: pd.Series) -> list[tuple[str, float]]:
    events: list[tuple[str, float]] = []
    for field, label in [
        ('t_contact_all', 'contact'),
        ('t_grasp_stable', 'stable'),
        ('t_if_enter', 'if_enter'),
        ('t_if_exit', 'if_exit'),
        ('t_end', 'end'),
    ]:
        value = pd.to_numeric(sample.get(field), errors='coerce')
        if not pd.isna(value):
            events.append((label, float(value)))
    return events


def build_loader(context: dict[str, Any], batch_size: int, num_workers: int) -> DataLoader:
    return DataLoader(
        context['dataset'],
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        collate_fn=sequence_collate_fn,
    )


def restrict_context_dataset(context: dict[str, Any], requested: list[str] | None, limit: int | None) -> list[str]:
    dataset = context['dataset']
    available = [str(record['sample_id']) for record in dataset.sample_records]
    if requested:
        missing = [sample_id for sample_id in requested if sample_id not in set(available)]
        if missing:
            raise ValueError(f'Sample ids are not present in this evaluated subset: {missing}')
        selected = [str(sample_id) for sample_id in requested]
    else:
        selected = available
    if limit is not None:
        selected = selected[: max(0, int(limit))]
    selected_set = set(selected)
    dataset.sample_records = [
        record for record in dataset.sample_records
        if str(record['sample_id']) in selected_set
    ]
    dataset.samples = dataset.samples.loc[dataset.samples['sample_id'].astype(str).isin(selected_set)].copy()
    dataset.windows = dataset.windows.loc[dataset.windows['sample_id'].astype(str).isin(selected_set)].copy()
    dataset.windows_by_sample = {
        sample_id: dataset.windows_by_sample[sample_id]
        for sample_id in selected
        if sample_id in dataset.windows_by_sample
    }
    return selected


def collect_policy_rows(context: dict[str, Any], *, batch_size: int, num_workers: int) -> list[dict[str, Any]]:
    loader = build_loader(context, batch_size=batch_size, num_workers=num_workers)
    model = context['model']
    device = context['device']
    dataset = context['dataset']
    model.eval()
    rows: list[dict[str, Any]] = []

    with torch.no_grad():
        for batch in loader:
            batch = move_to_device(batch, device)
            outputs = model(batch)

            force_pred = outputs['force_pred'].reshape_as(batch['expert_forces']).detach().cpu()
            force_base = outputs['force_base'].reshape_as(batch['expert_forces']).detach().cpu()
            force_base_learned = outputs.get('force_base_learned')
            if force_base_learned is not None:
                force_base_learned = force_base_learned.reshape_as(batch['expert_forces']).detach().cpu()
            force_delta = outputs['force_interface_delta'].reshape_as(batch['expert_forces']).detach().cpu()
            force_delta_raw = outputs.get('force_interface_delta_raw')
            if force_delta_raw is not None:
                force_delta_raw = force_delta_raw.reshape_as(batch['expert_forces']).detach().cpu()
            residual_diagnostics = {}
            for key in (
                'force_interface_delta_pos_raw',
                'force_interface_delta_neg_raw',
                'force_interface_delta_pos_magnitude',
                'force_interface_delta_neg_magnitude',
                'residual_direction_logit_neg',
                'residual_direction_logit_pos',
                'residual_direction_prob_neg',
                'residual_direction_prob_pos',
            ):
                value = outputs.get(key)
                if value is not None:
                    residual_diagnostics[key] = value.reshape_as(batch['expert_forces']).detach().cpu()
            gate = outputs['interface_gate'].reshape_as(batch['expert_forces']).detach().cpu()
            raw_gate = outputs.get('raw_interface_gate')
            if raw_gate is not None:
                raw_gate = raw_gate.reshape_as(batch['expert_forces']).detach().cpu()
            medium_probs = outputs['medium_probs'].detach().cpu()
            expert = batch['expert_forces'].detach().cpu()
            control_target = batch['control_force_targets'].detach().cpu()
            reference = batch['reference_forces'].detach().cpu()
            delta_target = batch['delta_force_targets'].detach().cpu()
            phase_labels = batch['phase_labels'].detach().cpu()
            stable_masks = batch['stable_masks'].detach().cpu()
            has_expert = batch['has_expert'].detach().cpu()
            has_control = batch['has_control_target'].detach().cpu()
            window_mask = batch['window_mask'].detach().cpu()

            for batch_index, sample_id in enumerate(batch['sample_ids']):
                sample_windows = dataset.windows_by_sample[str(sample_id)]
                valid_count = int(window_mask[batch_index].sum().item())
                for window_index in range(valid_count):
                    window = sample_windows[window_index]
                    phase_index = int(phase_labels[batch_index, window_index].item())
                    phase_label = INDEX_TO_PHASE.get(phase_index, str(window.get('phase_label', 'unknown')))
                    pred_value = finite_or_nan(force_pred[batch_index, window_index].item())
                    target_value = finite_or_nan(control_target[batch_index, window_index].item())
                    expert_value = finite_or_nan(expert[batch_index, window_index].item())
                    reference_value = finite_or_nan(reference[batch_index, window_index].item())
                    delta_value = finite_or_nan(force_delta[batch_index, window_index].item())
                    raw_delta_value = (
                        finite_or_nan(force_delta_raw[batch_index, window_index].item())
                        if force_delta_raw is not None
                        else float('nan')
                    )
                    residual_diagnostic_values = {
                        key: finite_or_nan(value[batch_index, window_index].item())
                        for key, value in residual_diagnostics.items()
                    }
                    gate_value = finite_or_nan(gate[batch_index, window_index].item())
                    raw_gate_value = (
                        finite_or_nan(raw_gate[batch_index, window_index].item())
                        if raw_gate is not None
                        else float('nan')
                    )
                    delta_target_value = finite_or_nan(delta_target[batch_index, window_index].item())
                    rows.append(
                        {
                            'sample_id': str(sample_id),
                            'object_id': str(batch['object_ids'][batch_index]),
                            'window_id': str(window['window_id']),
                            'window_index': window_index,
                            'window_start': float(window['window_start']),
                            'window_end': float(window['window_end']),
                            'window_center': float(window['window_center']),
                            'phase_label': phase_label,
                            'is_stable_mask': int(bool(stable_masks[batch_index, window_index].item())),
                            'has_expert': int(bool(has_expert[batch_index, window_index].item())),
                            'has_control_target': int(bool(has_control[batch_index, window_index].item())),
                            'force_pred': pred_value,
                            'force_base': finite_or_nan(force_base[batch_index, window_index].item()),
                            'force_base_learned': (
                                finite_or_nan(force_base_learned[batch_index, window_index].item())
                                if force_base_learned is not None
                                else float('nan')
                            ),
                            'force_interface_delta': delta_value,
                            'force_interface_delta_raw': raw_delta_value,
                            'force_interface_delta_pos_raw': residual_diagnostic_values.get(
                                'force_interface_delta_pos_raw', float('nan')
                            ),
                            'force_interface_delta_neg_raw': residual_diagnostic_values.get(
                                'force_interface_delta_neg_raw', float('nan')
                            ),
                            'force_interface_delta_pos_magnitude': residual_diagnostic_values.get(
                                'force_interface_delta_pos_magnitude', float('nan')
                            ),
                            'force_interface_delta_neg_magnitude': residual_diagnostic_values.get(
                                'force_interface_delta_neg_magnitude', float('nan')
                            ),
                            'residual_direction_logit_neg': residual_diagnostic_values.get(
                                'residual_direction_logit_neg', float('nan')
                            ),
                            'residual_direction_logit_pos': residual_diagnostic_values.get(
                                'residual_direction_logit_pos', float('nan')
                            ),
                            'residual_direction_prob_neg': residual_diagnostic_values.get(
                                'residual_direction_prob_neg', float('nan')
                            ),
                            'residual_direction_prob_pos': residual_diagnostic_values.get(
                                'residual_direction_prob_pos', float('nan')
                            ),
                            'interface_gate': gate_value,
                            'raw_interface_gate': raw_gate_value,
                            'medium_prob_water': finite_or_nan(medium_probs[batch_index, window_index, 0].item()),
                            'medium_prob_interface': finite_or_nan(medium_probs[batch_index, window_index, 1].item()),
                            'medium_prob_air': finite_or_nan(medium_probs[batch_index, window_index, 2].item()),
                            'gated_residual': gate_value * delta_value if np.isfinite(gate_value) and np.isfinite(delta_value) else float('nan'),
                            'expert_force': expert_value,
                            'control_force_target': target_value,
                            'reference_force': reference_value,
                            'delta_force_target': delta_target_value,
                            'prediction_error': pred_value - target_value if np.isfinite(pred_value) and np.isfinite(target_value) else float('nan'),
                            'absolute_error': abs(pred_value - target_value) if np.isfinite(pred_value) and np.isfinite(target_value) else float('nan'),
                        }
                    )
    return rows


def summarize_group(rows: pd.DataFrame, group_keys: list[str]) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame()
    valid = rows.loc[
        rows['has_control_target'].astype(int).eq(1)
        & np.isfinite(rows['prediction_error'].astype(float))
    ].copy()
    if valid.empty:
        return pd.DataFrame(columns=[*group_keys, 'window_count', 'mae', 'rmse', 'bias', 'hit_rate_100', 'hit_rate_200', 'hit_rate_300'])

    records: list[dict[str, Any]] = []
    for key_values, group in valid.groupby(group_keys, dropna=False):
        if not isinstance(key_values, tuple):
            key_values = (key_values,)
        error = group['prediction_error'].to_numpy(dtype=np.float64)
        absolute = np.abs(error)
        record = {key: value for key, value in zip(group_keys, key_values)}
        record.update(
            {
                'window_count': int(len(group)),
                'mae': float(np.mean(absolute)),
                'rmse': float(np.sqrt(np.mean(error ** 2))),
                'bias': float(np.mean(error)),
                'hit_rate_100': float(np.mean(absolute <= 100.0)),
                'hit_rate_200': float(np.mean(absolute <= 200.0)),
                'hit_rate_300': float(np.mean(absolute <= 300.0)),
                'gate_mean': float(np.nanmean(group['interface_gate'].to_numpy(dtype=np.float64))),
                'gated_residual_abs_mean': float(np.nanmean(np.abs(group['gated_residual'].to_numpy(dtype=np.float64)))),
            }
        )
        records.append(record)
    return pd.DataFrame(records)


def choose_render_sample_ids(
    selected_sample_ids: list[str],
    sample_summary: pd.DataFrame,
    *,
    limit: int | None,
    selection: str,
) -> list[str]:
    if limit is None:
        return selected_sample_ids
    count = max(0, int(limit))
    if count == 0 or not selected_sample_ids:
        return []
    if sample_summary.empty or selection == 'first':
        return selected_sample_ids[:count]

    summary = sample_summary.copy()
    summary['sample_id'] = summary['sample_id'].astype(str)
    summary = summary.loc[summary['sample_id'].isin(set(selected_sample_ids))]
    if summary.empty:
        return selected_sample_ids[:count]

    if selection == 'best':
        ordered = summary.sort_values(['mae', 'sample_id'], ascending=[True, True])
        return ordered['sample_id'].astype(str).head(count).tolist()
    if selection == 'worst':
        ordered = summary.sort_values(['mae', 'sample_id'], ascending=[False, True])
        return ordered['sample_id'].astype(str).head(count).tolist()
    if selection == 'mixed':
        best = summary.sort_values(['mae', 'sample_id'], ascending=[True, True])['sample_id'].astype(str).tolist()
        worst = summary.sort_values(['mae', 'sample_id'], ascending=[False, True])['sample_id'].astype(str).tolist()
        picked: list[str] = []
        for candidate in [*(best[: max(1, count // 2)]), *(worst[: count])]:
            if candidate not in picked:
                picked.append(candidate)
            if len(picked) >= count:
                break
        return picked
    raise ValueError(f'Unsupported selection: {selection!r}')


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text('', encoding='utf-8')
        return
    with path.open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def compute_y_limits(series_list: list[np.ndarray]) -> tuple[float, float]:
    finite = [series[np.isfinite(series)] for series in series_list if series.size]
    finite = [series for series in finite if series.size]
    if not finite:
        return -1.0, 1.0
    values = np.concatenate(finite)
    low = float(np.percentile(values, 1.0)) if values.size >= 32 else float(values.min())
    high = float(np.percentile(values, 99.0)) if values.size >= 32 else float(values.max())
    if high <= low:
        high = low + 1.0
    pad = 0.10 * (high - low)
    return low - pad, high + pad


def plot_sample(
    *,
    project_root: Path,
    data_config: dict[str, Any],
    sample: pd.Series,
    sample_rows: pd.DataFrame,
    output_path: Path,
    title_suffix: str,
) -> dict[str, Any]:
    tactile_path = resolve_data_path(project_root, str(sample['tactile_path']))
    tactile_dt = float(data_config['tactile_dt'])
    sync_offset = finite_or_nan(sample.get('sync_offset_sec', 0.0))
    if not np.isfinite(sync_offset):
        sync_offset = 0.0
    tactile = load_tactile_array(tactile_path)
    tactile_times = sync_offset + np.arange(tactile.shape[0], dtype=np.float32) * tactile_dt
    measured_force = compute_measured_force_curve(tactile, normal_sign_table=data_config['normal_sign_table'])
    measured_force_low = ema_filter(measured_force, alpha=float(data_config['ema_alpha_expert']))

    rows = sample_rows.sort_values('window_center').reset_index(drop=True)
    centers = rows['window_center'].to_numpy(dtype=np.float32)
    pred = rows['force_pred'].to_numpy(dtype=np.float32)
    control_target = rows['control_force_target'].to_numpy(dtype=np.float32)
    expert_force = rows['expert_force'].to_numpy(dtype=np.float32)
    reference = rows['reference_force'].to_numpy(dtype=np.float32)
    force_base = rows['force_base'].to_numpy(dtype=np.float32)
    force_base_learned = rows['force_base_learned'].to_numpy(dtype=np.float32)
    gate = rows['interface_gate'].to_numpy(dtype=np.float32)
    raw_gate = rows['raw_interface_gate'].to_numpy(dtype=np.float32)
    medium_prob_water = rows['medium_prob_water'].to_numpy(dtype=np.float32)
    medium_prob_interface = rows['medium_prob_interface'].to_numpy(dtype=np.float32)
    medium_prob_air = rows['medium_prob_air'].to_numpy(dtype=np.float32)
    delta = rows['force_interface_delta'].to_numpy(dtype=np.float32)
    delta_target = rows['delta_force_target'].to_numpy(dtype=np.float32)
    gated_residual = rows['gated_residual'].to_numpy(dtype=np.float32)
    prediction_error = rows['prediction_error'].to_numpy(dtype=np.float32)

    y_min, y_max = compute_y_limits([measured_force, measured_force_low, pred, control_target, expert_force, reference, force_base])
    fig, axes = plt.subplots(
        nrows=4,
        ncols=1,
        figsize=(18, 12),
        sharex=True,
        gridspec_kw={'height_ratios': [3.4, 1.25, 1.25, 1.1], 'hspace': 0.12},
    )
    force_ax, residual_ax, gate_ax, error_ax = axes

    for _, row in rows.iterrows():
        color = PHASE_COLORS.get(str(row['phase_label']), '#999999')
        for ax in axes:
            ax.axvspan(float(row['window_start']), float(row['window_end']), color=color, alpha=0.10, linewidth=0)

    force_ax.plot(tactile_times, measured_force, color=MEASURED_COLOR, linewidth=0.9, alpha=0.45, label='raw measured force')
    force_ax.plot(tactile_times, measured_force_low, color=EMA_COLOR, linewidth=1.3, alpha=0.85, label='EMA measured force')
    force_ax.plot(centers, pred, color=PRED_COLOR, linewidth=1.8, marker='o', markersize=3.5, label='policy force_pred')
    force_ax.plot(centers, control_target, color=TARGET_COLOR, linewidth=1.4, linestyle='--', marker='x', markersize=4, label='control target')
    force_ax.plot(centers, force_base, color=REFERENCE_COLOR, linewidth=1.3, linestyle='-.', label='force_base')
    if np.isfinite(reference).any():
        force_ax.plot(centers, reference, color='#9a8c98', linewidth=1.0, linestyle=':', label='reference_force')
    if np.isfinite(force_base_learned).any():
        force_ax.plot(centers, force_base_learned, color='#adb5bd', linewidth=0.9, linestyle=':', label='learned base')
    force_ax.set_ylabel('Force')
    force_ax.set_ylim(y_min, y_max)
    force_ax.grid(alpha=0.25)
    force_ax.legend(loc='upper right', fontsize=9)

    residual_ax.axhline(0.0, color='#555555', linewidth=0.8, alpha=0.7)
    residual_ax.plot(centers, delta, color='#f77f00', linewidth=1.2, marker='.', label='pred delta')
    residual_ax.plot(centers, gated_residual, color='#bc6c25', linewidth=1.5, marker='o', markersize=3, label='gate * delta')
    residual_ax.plot(centers, delta_target, color='#343a40', linewidth=1.0, linestyle='--', label='target delta')
    residual_ax.set_ylabel('Residual')
    residual_ax.grid(alpha=0.25)
    residual_ax.legend(loc='upper right', fontsize=9, ncol=3)

    gate_ax.plot(centers, gate, color='#9d4edd', linewidth=1.5, marker='.', label='interface_gate')
    if np.isfinite(raw_gate).any():
        gate_ax.plot(centers, raw_gate, color='#5a189a', linewidth=1.0, linestyle='--', alpha=0.75, label='raw interface prob')
    gate_ax.plot(centers, medium_prob_water, color=PHASE_COLORS['Water'], linewidth=0.9, alpha=0.65, label='p(Water)')
    gate_ax.plot(centers, medium_prob_interface, color=PHASE_COLORS['Interface'], linewidth=0.9, alpha=0.75, label='p(Interface)')
    gate_ax.plot(centers, medium_prob_air, color=PHASE_COLORS['Air'], linewidth=0.9, alpha=0.65, label='p(Air)')
    gate_ax.set_ylabel('Gate')
    gate_ax.set_ylim(-0.05, 1.05)
    gate_ax.grid(alpha=0.25)
    gate_ax.legend(loc='upper right', fontsize=8, ncol=5)

    error_ax.axhline(0.0, color='#555555', linewidth=0.8, alpha=0.7)
    error_ax.plot(centers, prediction_error, color='#d00000', linewidth=1.1, marker='.', label='pred - target')
    error_ax.set_ylabel('Error / Phase')
    for _, row in rows.iterrows():
        color = PHASE_COLORS.get(str(row['phase_label']), '#999999')
        start = float(row['window_start'])
        end = float(row['window_end'])
        error_ax.axvspan(start, end, ymin=0.02, ymax=0.18, color=color, alpha=0.70, linewidth=0)
        if int(row['is_stable_mask']) == 1:
            error_ax.plot(
                [start, end],
                [0.96, 0.96],
                color=color,
                linewidth=2.0,
                transform=error_ax.get_xaxis_transform(),
                clip_on=False,
            )
    error_ax.grid(alpha=0.18)
    error_ax.set_xlabel('Time (s)')

    for label, time_sec in collect_event_times(sample):
        for ax in axes:
            ax.axvline(time_sec, color='#444444', linestyle=':', linewidth=0.9)
        force_ax.text(time_sec, y_max, label, rotation=90, va='top', ha='right', fontsize=8)

    phase_handles = [Patch(facecolor=color, alpha=0.65, label=phase) for phase, color in PHASE_COLORS.items()]
    phase_handles.append(Line2D([0], [0], color='#444444', linestyle=':', label='events'))
    error_ax.legend(handles=phase_handles, loc='upper right', fontsize=9, ncol=4)

    title = (
        f'{sample["sample_id"]} | object={sample["object_id"]} | trial={sample["trial_result"]} | '
        f'{title_suffix}'
    )
    fig.suptitle(title, fontsize=13)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches='tight')
    plt.close(fig)

    valid_error = rows['prediction_error'].to_numpy(dtype=np.float32)
    valid_error = valid_error[np.isfinite(valid_error)]
    return {
        'sample_id': str(sample['sample_id']),
        'object_id': str(sample['object_id']),
        'output_path': str(output_path),
        'window_count': int(len(rows)),
        'prediction_mae': float(np.mean(np.abs(valid_error))) if valid_error.size else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--project-root', default='.')
    parser.add_argument('--stage', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--subset', default='val', choices=['train', 'val', 'test'])
    parser.add_argument('--sample-id', nargs='*', default=None)
    parser.add_argument('--limit', type=int, default=3)
    parser.add_argument('--selection', default='first', choices=['first', 'best', 'worst', 'mixed'])
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--output-dir', default=None)
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    stage_path = resolve_path(project_root, args.stage)
    checkpoint_path = resolve_path(project_root, args.checkpoint)
    context = prepare_evaluation_context(
        project_root=project_root,
        stage=stage_path,
        checkpoint=checkpoint_path,
        subset=args.subset,
        only_stable=False,
    )
    selected_sample_ids = restrict_context_dataset(context, args.sample_id, None)

    rows = collect_policy_rows(context, batch_size=int(args.batch_size), num_workers=int(args.num_workers))
    if args.output_dir:
        output_dir = resolve_path(project_root, args.output_dir)
    else:
        output_dir = (
            project_root
            / 'evals'
            / 'policy_force_curves'
            / str(context['stage_config']['name'])
            / f'{checkpoint_path.stem}__{args.subset}'
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    rows_path = output_dir / 'policy_force_windows.csv'
    write_rows(rows_path, rows)
    rows_df = pd.DataFrame(rows)
    sample_summary = summarize_group(rows_df, ['sample_id', 'object_id'])
    phase_summary = summarize_group(rows_df, ['phase_label'])
    sample_phase_summary = summarize_group(rows_df, ['sample_id', 'object_id', 'phase_label'])
    sample_summary_path = output_dir / 'policy_force_sample_summary.csv'
    phase_summary_path = output_dir / 'policy_force_phase_summary.csv'
    sample_phase_summary_path = output_dir / 'policy_force_sample_phase_summary.csv'
    sample_summary.to_csv(sample_summary_path, index=False, encoding='utf-8-sig')
    phase_summary.to_csv(phase_summary_path, index=False, encoding='utf-8-sig')
    sample_phase_summary.to_csv(sample_phase_summary_path, index=False, encoding='utf-8-sig')

    render_sample_ids = choose_render_sample_ids(
        selected_sample_ids,
        sample_summary,
        limit=args.limit,
        selection=str(args.selection),
    )
    samples = pd.read_csv(project_root / 'data' / 'processed' / 'samples.csv')

    render_results: list[dict[str, Any]] = []
    for sample_id in render_sample_ids:
        sample_match = samples.loc[samples['sample_id'].astype(str) == str(sample_id)]
        if sample_match.empty:
            raise ValueError(f'Could not find sample metadata for {sample_id}')
        sample_rows = rows_df.loc[rows_df['sample_id'].astype(str) == str(sample_id)].copy()
        result = plot_sample(
            project_root=project_root,
            data_config=context['data_config'],
            sample=sample_match.iloc[0],
            sample_rows=sample_rows,
            output_path=output_dir / f'{sample_id}.png',
            title_suffix=f'stage={context["stage_config"]["name"]} | checkpoint={checkpoint_path.name}',
        )
        render_results.append(result)

    summary = {
        'stage_name': context['stage_config']['name'],
        'stage_path': str(stage_path),
        'checkpoint_path': str(checkpoint_path),
        'subset': args.subset,
        'rows_path': str(rows_path),
        'sample_summary_path': str(sample_summary_path),
        'phase_summary_path': str(phase_summary_path),
        'sample_phase_summary_path': str(sample_phase_summary_path),
        'output_dir': str(output_dir),
        'selection': args.selection,
        'rendered': render_results,
    }
    summary_path = output_dir / 'policy_force_curve_summary.json'
    with summary_path.open('w', encoding='utf-8') as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(
        {
            'output_dir': str(output_dir),
            'summary_path': str(summary_path),
            'rows_path': str(rows_path),
            'sample_summary_path': str(sample_summary_path),
            'phase_summary_path': str(phase_summary_path),
            'rendered_count': len(render_results),
            'sample_ids': render_sample_ids,
        }
    )


if __name__ == '__main__':
    main()
