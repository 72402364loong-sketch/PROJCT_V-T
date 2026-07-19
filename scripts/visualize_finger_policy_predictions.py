from __future__ import annotations

import argparse
import csv
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
from cmg.config import deep_update, load_yaml
from cmg.data import sequence_collate_fn
from cmg.data.tactile import build_tactile_time_axis, compute_clean_force_curve, load_tactile_array
from cmg.evaluation import prepare_evaluation_context, resolve_path
from cmg.training import move_to_device

try:
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
except ImportError as exc:  # pragma: no cover
    raise RuntimeError('matplotlib is required for visualize_finger_policy_predictions.py.') from exc


PHASE_COLORS = {
    'Water': '#2a6f97',
    'Interface': '#f4a261',
    'Air': '#8ab17d',
}
MODEL_COLORS = ['#0077b6', '#d62828', '#2a9d8f', '#7b2cbf', '#e76f51']
TARGET_COLOR = '#111111'
REFERENCE_COLOR = '#7a7a7a'
RAW_FORCE_COLOR = '#8d6e63'
FINGER_NAMES = ['finger0', 'finger1', 'finger2']


def parse_model_spec(raw: list[str]) -> dict[str, str]:
    if len(raw) != 3:
        raise ValueError('Each --model requires: LABEL STAGE_YAML CHECKPOINT.')
    return {'label': raw[0], 'stage': raw[1], 'checkpoint': raw[2]}


def restrict_dataset(context: dict[str, Any], sample_ids: list[str]) -> None:
    dataset = context['dataset']
    selected = set(sample_ids)
    dataset.sample_records = [
        record for record in dataset.sample_records
        if str(record['sample_id']) in selected
    ]
    dataset.samples = dataset.samples.loc[dataset.samples['sample_id'].astype(str).isin(selected)].copy()
    dataset.windows = dataset.windows.loc[dataset.windows['sample_id'].astype(str).isin(selected)].copy()
    dataset.windows_by_sample = {
        sample_id: dataset.windows_by_sample[sample_id]
        for sample_id in sample_ids
        if sample_id in dataset.windows_by_sample
    }


def select_sample_ids(
    context: dict[str, Any],
    requested: list[str] | None,
    limit: int | None,
) -> list[str]:
    available = [str(record['sample_id']) for record in context['dataset'].sample_records]
    if requested:
        known = set(available)
        missing = [sample_id for sample_id in requested if sample_id not in known]
        if missing:
            raise ValueError(f'Sample ids are not present in this subset: {missing}')
        selected = [str(sample_id) for sample_id in requested]
    else:
        selected = available
    if limit is not None:
        selected = selected[: max(0, int(limit))]
    if not selected:
        raise ValueError('No samples selected.')
    return selected


def finite_float(value: Any) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return float('nan')
    return parsed if np.isfinite(parsed) else float('nan')


def parse_optional_float(value: Any) -> float | None:
    parsed = finite_float(value)
    return float(parsed) if np.isfinite(parsed) else None


def resolve_data_path(project_root: Path, raw_path: str | Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    if path.parts and path.parts[0] == 'data':
        return project_root / path
    return project_root / 'data' / path


def load_data_config(project_root: Path, stage_path: str) -> dict[str, Any]:
    data_config = load_yaml(project_root / 'configs' / 'data' / 'default.yaml')
    stage = load_yaml(resolve_path(project_root, stage_path))
    return deep_update(data_config, stage.get('data', {}))


def collect_raw_force_curves(
    *,
    project_root: Path,
    data_config: dict[str, Any],
    samples: pd.DataFrame,
    sample_ids: list[str],
) -> dict[str, dict[str, np.ndarray]]:
    curves: dict[str, dict[str, np.ndarray]] = {}
    tactile_dt = float(data_config['tactile_dt'])
    alpha = float(data_config.get('ema_alpha_expert', 0.1))
    smoothing_mode = str(data_config.get('expert_force_smoothing', 'zero_phase_ema'))
    baseline_mode = str(data_config.get('expert_force_baseline_mode', 'pre_contact_median'))
    baseline_window_sec = float(data_config.get('expert_force_baseline_window_sec', 0.5))

    for sample_id in sample_ids:
        sample_rows = samples.loc[samples['sample_id'].astype(str) == str(sample_id)]
        if sample_rows.empty:
            continue
        sample = sample_rows.iloc[0]
        tactile = load_tactile_array(resolve_data_path(project_root, str(sample['tactile_path'])))
        sync_offset_sec = parse_optional_float(sample.get('sync_offset_sec')) or 0.0
        contact_time = parse_optional_float(sample.get('t_contact_all'))
        time_axis = build_tactile_time_axis(tactile.shape[0], dt=tactile_dt, offset_sec=sync_offset_sec)
        force_curve = compute_clean_force_curve(
            tactile,
            dt=tactile_dt,
            sync_offset_sec=sync_offset_sec,
            contact_time=contact_time,
            alpha=alpha,
            normal_sign_table=data_config['normal_sign_table'],
            smoothing_mode=smoothing_mode,
            baseline_mode=baseline_mode,
            baseline_window_sec=baseline_window_sec,
            force_target_mode='per_finger_z_abs_mean',
        )
        keep = time_axis >= 0.0
        end_time = parse_optional_float(sample.get('t_end'))
        if end_time is not None:
            keep &= time_axis <= float(end_time)
        curves[str(sample_id)] = {
            'time': time_axis[keep].astype(np.float32),
            'force': force_curve[keep].astype(np.float32),
        }
    return curves


def collect_rows_for_model(
    *,
    project_root: Path,
    spec: dict[str, str],
    subset: str,
    sample_ids: list[str],
    batch_size: int,
    num_workers: int,
) -> list[dict[str, Any]]:
    context = prepare_evaluation_context(
        project_root=project_root,
        stage=spec['stage'],
        checkpoint=spec['checkpoint'],
        subset=subset,
        num_workers=num_workers,
    )
    restrict_dataset(context, sample_ids)
    loader = DataLoader(
        context['dataset'],
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        collate_fn=sequence_collate_fn,
    )
    model = context['model']
    device = context['device']
    dataset = context['dataset']
    model.eval()

    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for batch in loader:
            batch = move_to_device(batch, device)
            outputs = model(batch)

            pred = outputs['finger_force_pred'].reshape_as(batch['finger_control_force_targets']).detach().cpu()
            base = outputs['finger_force_base'].reshape_as(batch['finger_reference_forces']).detach().cpu()
            delta = outputs['finger_force_interface_delta'].reshape_as(batch['finger_delta_force_targets']).detach().cpu()
            target = batch['finger_control_force_targets'].detach().cpu()
            reference = batch['finger_reference_forces'].detach().cpu()
            target_delta = batch['finger_delta_force_targets'].detach().cpu()
            has_target = batch['has_finger_control_target'].detach().cpu().bool()
            has_reference = batch['has_finger_reference'].detach().cpu().bool()
            phase_labels = batch['phase_labels'].detach().cpu()
            stable_masks = batch['stable_masks'].detach().cpu().bool()
            window_mask = batch['window_mask'].detach().cpu().bool()
            gate = outputs['interface_gate'].reshape_as(batch['phase_labels']).detach().cpu()

            for batch_index, sample_id in enumerate(batch['sample_ids']):
                sample_id = str(sample_id)
                sample_windows = dataset.windows_by_sample[sample_id]
                valid_count = int(window_mask[batch_index].sum().item())
                for window_index in range(valid_count):
                    window = sample_windows[window_index]
                    phase_index = int(phase_labels[batch_index, window_index].item())
                    phase_label = INDEX_TO_PHASE.get(phase_index, str(window.get('phase_label', 'unknown')))
                    gate_value = finite_float(gate[batch_index, window_index].item())
                    for finger_index, finger_name in enumerate(FINGER_NAMES):
                        pred_value = finite_float(pred[batch_index, window_index, finger_index].item())
                        target_value = finite_float(target[batch_index, window_index, finger_index].item())
                        reference_value = finite_float(reference[batch_index, window_index, finger_index].item())
                        delta_value = finite_float(delta[batch_index, window_index, finger_index].item())
                        target_delta_value = finite_float(target_delta[batch_index, window_index, finger_index].item())
                        rows.append(
                            {
                                'model': spec['label'],
                                'sample_id': sample_id,
                                'object_id': str(batch['object_ids'][batch_index]),
                                'window_id': str(window['window_id']),
                                'window_index': int(window_index),
                                'window_start': finite_float(window['window_start']),
                                'window_end': finite_float(window['window_end']),
                                'window_center': finite_float(window['window_center']),
                                'policy_timestamp': finite_float(window.get('policy_timestamp', window['window_center'])),
                                'phase_label': phase_label,
                                'is_stable_mask': int(bool(stable_masks[batch_index, window_index].item())),
                                'interface_gate': gate_value,
                                'finger': finger_name,
                                'finger_index': int(finger_index),
                                'has_target': int(bool(has_target[batch_index, window_index, finger_index].item())),
                                'has_reference': int(bool(has_reference[batch_index, window_index, finger_index].item())),
                                'target': target_value,
                                'reference': reference_value,
                                'base': finite_float(base[batch_index, window_index, finger_index].item()),
                                'pred': pred_value,
                                'target_delta': target_delta_value,
                                'pred_delta': delta_value,
                                'gated_delta': gate_value * delta_value if np.isfinite(gate_value) and np.isfinite(delta_value) else float('nan'),
                                'abs_error': abs(pred_value - target_value) if np.isfinite(pred_value) and np.isfinite(target_value) else float('nan'),
                                'abs_delta_error': abs(delta_value - target_delta_value) if np.isfinite(delta_value) and np.isfinite(target_delta_value) else float('nan'),
                            }
                        )
    return rows


def summarize_rows(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame()
    summary_rows: list[dict[str, Any]] = []
    group_cols = ['sample_id', 'object_id', 'model', 'finger']
    for keys, group in rows.groupby(group_cols, sort=True):
        sample_id, object_id, model, finger = keys
        target_mask = group['has_target'].astype(bool) & np.isfinite(group['abs_error'])
        interface_mask = target_mask & (group['phase_label'].astype(str) == 'Interface')
        stable_mask = group['is_stable_mask'].astype(bool) & np.isfinite(group['gated_delta'])
        summary_rows.append(
            {
                'sample_id': sample_id,
                'object_id': object_id,
                'model': model,
                'finger': finger,
                'overall_mae': float(group.loc[target_mask, 'abs_error'].mean()) if target_mask.any() else float('nan'),
                'interface_mae': float(group.loc[interface_mask, 'abs_error'].mean()) if interface_mask.any() else float('nan'),
                'delta_interface_mae': float(group.loc[interface_mask, 'abs_delta_error'].mean()) if interface_mask.any() else float('nan'),
                'stable_gated_delta_abs_mean': float(group.loc[stable_mask, 'gated_delta'].abs().mean()) if stable_mask.any() else float('nan'),
                'pred_delta_abs_interface_mean': float(group.loc[interface_mask, 'pred_delta'].abs().mean()) if interface_mask.any() else float('nan'),
                'target_delta_abs_interface_mean': float(group.loc[interface_mask, 'target_delta'].abs().mean()) if interface_mask.any() else float('nan'),
                'interface_window_count': int(interface_mask.sum()),
            }
        )
    return pd.DataFrame(summary_rows)


def phase_spans(sample_rows: pd.DataFrame) -> pd.DataFrame:
    return (
        sample_rows[['window_start', 'window_end', 'phase_label']]
        .drop_duplicates()
        .sort_values('window_start')
        .reset_index(drop=True)
    )


def event_times(samples: pd.DataFrame, sample_id: str) -> list[tuple[str, float]]:
    rows = samples.loc[samples['sample_id'].astype(str) == str(sample_id)]
    if rows.empty:
        return []
    row = rows.iloc[0]
    events = []
    for key, label in [
        ('t_contact_all', 'contact'),
        ('t_grasp_stable', 'stable'),
        ('t_if_enter', 'if_enter'),
        ('t_if_exit', 'if_exit'),
        ('t_end', 'end'),
    ]:
        value = finite_float(row.get(key))
        if np.isfinite(value):
            events.append((label, value))
    return events


def shade_phases(ax: Any, spans: pd.DataFrame) -> None:
    for _, span in spans.iterrows():
        color = PHASE_COLORS.get(str(span['phase_label']), '#cccccc')
        ax.axvspan(float(span['window_start']), float(span['window_end']), color=color, alpha=0.11, linewidth=0)


def plot_line(ax: Any, x: np.ndarray, y: np.ndarray, *, label: str, color: str, linestyle: str, linewidth: float) -> None:
    finite = np.isfinite(x) & np.isfinite(y)
    if finite.any():
        ax.plot(x[finite], y[finite], label=label, color=color, linestyle=linestyle, linewidth=linewidth, marker='o', markersize=2.4)


def render_sample(
    *,
    rows: pd.DataFrame,
    samples: pd.DataFrame,
    sample_id: str,
    model_labels: list[str],
    model_colors: dict[str, str],
    raw_force_curves: dict[str, dict[str, np.ndarray]] | None,
    output_path: Path,
) -> dict[str, Any]:
    sample_rows = rows.loc[rows['sample_id'].astype(str) == str(sample_id)].copy()
    if sample_rows.empty:
        raise ValueError(f'No prediction rows for sample {sample_id}.')

    spans = phase_spans(sample_rows)
    object_id = str(sample_rows['object_id'].iloc[0])
    fig = plt.figure(figsize=(18, 12))
    grid = fig.add_gridspec(nrows=4, ncols=2, height_ratios=[1.0, 1.0, 1.0, 0.52], hspace=0.22, wspace=0.12)
    axes: list[Any] = []
    delta_axes: list[Any] = []
    for finger_index, finger_name in enumerate(FINGER_NAMES):
        force_ax = fig.add_subplot(grid[finger_index, 0])
        delta_ax = fig.add_subplot(grid[finger_index, 1], sharex=force_ax)
        axes.extend([force_ax, delta_ax])
        delta_axes.append(delta_ax)
        shade_phases(force_ax, spans)
        shade_phases(delta_ax, spans)

        finger_rows = sample_rows.loc[sample_rows['finger'] == finger_name]
        raw_curve = raw_force_curves.get(sample_id) if raw_force_curves else None
        if raw_curve is not None:
            raw_time = raw_curve['time']
            raw_force = raw_curve['force']
            if raw_force.ndim == 2 and finger_index < raw_force.shape[1]:
                raw_finger_force = raw_force[:, finger_index]
                finite_raw = np.isfinite(raw_time) & np.isfinite(raw_finger_force)
                if finite_raw.any():
                    force_ax.plot(
                        raw_time[finite_raw],
                        raw_finger_force[finite_raw],
                        label='raw force',
                        color=RAW_FORCE_COLOR,
                        linestyle=':',
                        linewidth=1.0,
                        alpha=0.82,
                        zorder=2,
                    )
        target_rows = (
            finger_rows.loc[finger_rows['model'] == model_labels[0]]
            .sort_values('policy_timestamp')
            .reset_index(drop=True)
        )
        x = target_rows['policy_timestamp'].to_numpy(dtype=float)
        target_mask = target_rows['has_target'].to_numpy(dtype=bool)
        reference_mask = target_rows['has_reference'].to_numpy(dtype=bool)
        plot_line(
            force_ax,
            x[target_mask],
            target_rows.loc[target_mask, 'target'].to_numpy(dtype=float),
            label='target',
            color=TARGET_COLOR,
            linestyle='-',
            linewidth=1.8,
        )
        plot_line(
            force_ax,
            x[reference_mask],
            target_rows.loc[reference_mask, 'reference'].to_numpy(dtype=float),
            label='reference',
            color=REFERENCE_COLOR,
            linestyle='--',
            linewidth=1.25,
        )
        plot_line(
            delta_ax,
            x[target_mask],
            target_rows.loc[target_mask, 'target_delta'].to_numpy(dtype=float),
            label='target delta',
            color=TARGET_COLOR,
            linestyle='-',
            linewidth=1.55,
        )

        for model_label in model_labels:
            model_rows = (
                finger_rows.loc[finger_rows['model'] == model_label]
                .sort_values('policy_timestamp')
                .reset_index(drop=True)
            )
            if model_rows.empty:
                continue
            mx = model_rows['policy_timestamp'].to_numpy(dtype=float)
            color = model_colors[model_label]
            plot_line(
                force_ax,
                mx,
                model_rows['pred'].to_numpy(dtype=float),
                label=f'{model_label} pred',
                color=color,
                linestyle='-',
                linewidth=1.25,
            )
            plot_line(
                delta_ax,
                mx,
                model_rows['pred_delta'].to_numpy(dtype=float),
                label=f'{model_label} delta',
                color=color,
                linestyle='-',
                linewidth=1.15,
            )

        force_ax.set_ylabel(f'{finger_name}\nforce')
        delta_ax.set_ylabel(f'{finger_name}\ndelta')
        force_ax.grid(alpha=0.22)
        delta_ax.grid(alpha=0.22)
        if finger_index == 0:
            force_ax.set_title('Per-finger force: target / reference / prediction')
            delta_ax.set_title('Per-finger residual: delta target / predicted delta')
        if finger_index < 2:
            force_ax.tick_params(labelbottom=False)
            delta_ax.tick_params(labelbottom=False)

    gate_ax = fig.add_subplot(grid[3, :], sharex=axes[0])
    shade_phases(gate_ax, spans)
    gate_ax.set_ylim(-0.05, 1.05)
    gate_ax.set_yticks([0.0, 0.5, 1.0])
    gate_ax.set_ylabel('gate')
    gate_ax.set_xlabel('time (s)')
    for model_label in model_labels:
        model_rows = (
            sample_rows.loc[sample_rows['model'] == model_label]
            [['policy_timestamp', 'interface_gate']]
            .drop_duplicates()
            .sort_values('policy_timestamp')
        )
        plot_line(
            gate_ax,
            model_rows['policy_timestamp'].to_numpy(dtype=float),
            model_rows['interface_gate'].to_numpy(dtype=float),
            label=f'{model_label} gate',
            color=model_colors[model_label],
            linestyle='-',
            linewidth=1.1,
        )
    gate_ax.grid(alpha=0.22)

    for label, time_sec in event_times(samples, sample_id):
        for ax in axes + [gate_ax]:
            ax.axvline(time_sec, color='#6d597a', linestyle=':', linewidth=0.9)
        gate_ax.text(time_sec, 1.02, label, rotation=90, va='bottom', ha='right', fontsize=8)

    legend_handles = [
        Line2D([0], [0], color=RAW_FORCE_COLOR, linestyle=':', linewidth=1.0, label='raw force'),
        Line2D([0], [0], color=TARGET_COLOR, linewidth=1.8, label='target'),
        Line2D([0], [0], color=REFERENCE_COLOR, linestyle='--', linewidth=1.25, label='reference'),
    ]
    legend_handles.extend(
        Line2D([0], [0], color=model_colors[label], linewidth=1.3, label=label)
        for label in model_labels
    )
    legend_handles.extend(
        Patch(facecolor=color, alpha=0.18, label=phase)
        for phase, color in PHASE_COLORS.items()
    )
    fig.legend(handles=legend_handles, loc='upper center', ncol=min(8, len(legend_handles)), frameon=False)
    fig.suptitle(f'{sample_id} | object={object_id} | per-finger policy prediction comparison', fontsize=13, y=0.985)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=170, bbox_inches='tight')
    plt.close(fig)
    return {'sample_id': sample_id, 'object_id': object_id, 'output_path': str(output_path)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--project-root', default='.')
    parser.add_argument('--subset', default='val', choices=['train', 'val', 'test'])
    parser.add_argument('--model', nargs=3, action='append', metavar=('LABEL', 'STAGE_YAML', 'CHECKPOINT'))
    parser.add_argument('--sample-id', nargs='*', default=None)
    parser.add_argument('--limit', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--include-raw-force', action='store_true')
    parser.add_argument('--output-dir', default='data/processed/debug/finger_policy_predictions')
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    if args.model:
        specs = [parse_model_spec(raw) for raw in args.model]
    else:
        specs = [
            {
                'label': 'stage38c',
                'stage': 'configs/stages/stage38c_policy_fold1_per_finger_reference_delta_scale20_long.yaml',
                'checkpoint': 'runs/stage38c_policy_fold1_per_finger_reference_delta_scale20_long/checkpoints/best.pt',
            }
        ]

    first_context = prepare_evaluation_context(
        project_root=project_root,
        stage=specs[0]['stage'],
        checkpoint=specs[0]['checkpoint'],
        subset=args.subset,
        num_workers=args.num_workers,
    )
    sample_ids = select_sample_ids(first_context, args.sample_id, args.limit)
    model_labels = [spec['label'] for spec in specs]
    model_colors = {
        label: MODEL_COLORS[index % len(MODEL_COLORS)]
        for index, label in enumerate(model_labels)
    }

    rows: list[dict[str, Any]] = []
    for spec in specs:
        rows.extend(
            collect_rows_for_model(
                project_root=project_root,
                spec=spec,
                subset=args.subset,
                sample_ids=sample_ids,
                batch_size=int(args.batch_size),
                num_workers=int(args.num_workers),
            )
        )
    frame = pd.DataFrame(rows)
    output_dir = resolve_path(project_root, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = output_dir / f'{args.subset}_finger_policy_prediction_rows.csv'
    summary_path = output_dir / f'{args.subset}_finger_policy_prediction_summary.csv'
    frame.to_csv(rows_path, index=False)
    summarize_rows(frame).to_csv(summary_path, index=False)

    samples = pd.read_csv(project_root / 'data' / 'processed' / 'samples.csv')
    raw_force_curves = None
    if args.include_raw_force:
        raw_force_curves = collect_raw_force_curves(
            project_root=project_root,
            data_config=load_data_config(project_root, specs[0]['stage']),
            samples=samples,
            sample_ids=sample_ids,
        )
    rendered = []
    for sample_id in sample_ids:
        output_path = output_dir / f'{sample_id}_finger_policy_predictions.png'
        rendered.append(
            render_sample(
                rows=frame,
                samples=samples,
                sample_id=sample_id,
                model_labels=model_labels,
                model_colors=model_colors,
                raw_force_curves=raw_force_curves,
                output_path=output_path,
            )
        )
        print(f'{sample_id} -> {output_path}')

    print(
        {
            'sample_ids': sample_ids,
            'rows_path': str(rows_path),
            'summary_path': str(summary_path),
            'output_dir': str(output_dir),
            'rendered': rendered,
        }
    )


if __name__ == '__main__':
    main()
