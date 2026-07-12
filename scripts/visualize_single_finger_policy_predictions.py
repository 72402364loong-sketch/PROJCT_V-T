from __future__ import annotations

import argparse
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
from cmg.data.tactile import compute_per_finger_abs_mean_force_curve, load_tactile_array
from cmg.evaluation import prepare_evaluation_context, resolve_path
from cmg.training import move_to_device

try:
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
except ImportError as exc:  # pragma: no cover
    raise RuntimeError('matplotlib is required for visualize_single_finger_policy_predictions.py.') from exc


PHASE_COLORS = {
    'Water': '#2a6f97',
    'Interface': '#f4a261',
    'Air': '#8ab17d',
}
FINGER_NAMES = ['finger0', 'finger1', 'finger2']
RAW_COLOR = '#9a9a9a'
TARGET_COLOR = '#111111'
REFERENCE_COLOR = '#6d597a'
PRED_COLOR = '#c1121f'
GATE_COLOR = '#2a9d8f'


def finite_float(value: Any) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return float('nan')
    return parsed if np.isfinite(parsed) else float('nan')


def resolve_data_path(project_root: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    if path.parts and path.parts[0] == 'data':
        return project_root / path
    return project_root / 'data' / path


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


def select_sample_ids(context: dict[str, Any], requested: list[str] | None, limit: int | None) -> list[str]:
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


def collect_prediction_rows(
    context: dict[str, Any],
    sample_ids: list[str],
    *,
    batch_size: int,
    num_workers: int,
) -> pd.DataFrame:
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
            target = batch['finger_control_force_targets'].detach().cpu()
            reference = batch['finger_reference_forces'].detach().cpu()
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
                    for finger_index, finger_name in enumerate(FINGER_NAMES):
                        rows.append(
                            {
                                'sample_id': sample_id,
                                'object_id': str(batch['object_ids'][batch_index]),
                                'window_id': str(window['window_id']),
                                'window_index': int(window_index),
                                'window_start': finite_float(window['window_start']),
                                'window_end': finite_float(window['window_end']),
                                'window_center': finite_float(window['window_center']),
                                'phase_label': phase_label,
                                'is_stable_mask': int(bool(stable_masks[batch_index, window_index].item())),
                                'interface_gate': finite_float(gate[batch_index, window_index].item()),
                                'finger': finger_name,
                                'finger_index': int(finger_index),
                                'has_target': int(bool(has_target[batch_index, window_index, finger_index].item())),
                                'has_reference': int(bool(has_reference[batch_index, window_index, finger_index].item())),
                                'target': finite_float(target[batch_index, window_index, finger_index].item()),
                                'reference': finite_float(reference[batch_index, window_index, finger_index].item()),
                                'pred': finite_float(pred[batch_index, window_index, finger_index].item()),
                            }
                        )
    return pd.DataFrame(rows)


def load_raw_per_finger_curve(
    *,
    project_root: Path,
    data_config: dict[str, Any],
    sample: pd.Series,
) -> tuple[np.ndarray, np.ndarray]:
    tactile_path = resolve_data_path(project_root, str(sample['tactile_path']))
    tactile = load_tactile_array(tactile_path)
    tactile_dt = float(data_config['tactile_dt'])
    sync_offset_sec = finite_float(sample.get('sync_offset_sec'))
    if not np.isfinite(sync_offset_sec):
        sync_offset_sec = 0.0
    contact_time = finite_float(sample.get('t_contact_all'))
    if not np.isfinite(contact_time):
        contact_time = None
    raw_curve = compute_per_finger_abs_mean_force_curve(
        tactile,
        dt=tactile_dt,
        sync_offset_sec=sync_offset_sec,
        contact_time=contact_time,
        alpha=None,
        smoothing_mode='none',
        baseline_mode=str(data_config.get('expert_force_baseline_mode', 'pre_contact_median')),
        baseline_window_sec=float(data_config.get('expert_force_baseline_window_sec', 0.5)),
    )
    time_axis = sync_offset_sec + np.arange(raw_curve.shape[0], dtype=np.float32) * tactile_dt
    return time_axis, raw_curve


def event_times(sample: pd.Series) -> list[tuple[str, float]]:
    events = []
    for key, label in [
        ('t_contact_all', 'contact'),
        ('t_grasp_stable', 'stable'),
        ('t_if_enter', 'if_enter'),
        ('t_if_exit', 'if_exit'),
        ('t_end', 'end'),
    ]:
        value = finite_float(sample.get(key))
        if np.isfinite(value):
            events.append((label, value))
    return events


def sample_end_time(sample: pd.Series, fallback: float) -> float:
    end_time = finite_float(sample.get('t_end'))
    if np.isfinite(end_time):
        return max(0.0, end_time)
    return max(0.0, float(fallback))


def shade_phases(ax: Any, spans: pd.DataFrame, *, alpha: float = 0.11) -> None:
    for _, span in spans.iterrows():
        color = PHASE_COLORS.get(str(span['phase_label']), '#cccccc')
        ax.axvspan(float(span['window_start']), float(span['window_end']), color=color, alpha=alpha, linewidth=0)


def plot_series(
    ax: Any,
    x: np.ndarray,
    y: np.ndarray,
    *,
    label: str,
    color: str,
    linestyle: str = '-',
    linewidth: float = 1.4,
    marker: str | None = 'o',
    markersize: float = 3.0,
) -> None:
    finite = np.isfinite(x) & np.isfinite(y)
    if not finite.any():
        return
    ax.plot(
        x[finite],
        y[finite],
        label=label,
        color=color,
        linestyle=linestyle,
        linewidth=linewidth,
        marker=marker,
        markersize=markersize,
    )


def render_finger_plot(
    *,
    project_root: Path,
    data_config: dict[str, Any],
    samples: pd.DataFrame,
    rows: pd.DataFrame,
    sample_id: str,
    finger_index: int,
    model_label: str,
    output_path: Path,
) -> dict[str, Any]:
    sample_rows = samples.loc[samples['sample_id'].astype(str) == str(sample_id)]
    if sample_rows.empty:
        raise ValueError(f'Unknown sample_id: {sample_id}')
    sample = sample_rows.iloc[0]
    finger_name = FINGER_NAMES[finger_index]
    frame = rows.loc[(rows['sample_id'] == sample_id) & (rows['finger_index'] == finger_index)].copy()
    if frame.empty:
        raise ValueError(f'No prediction rows for {sample_id} {finger_name}.')
    frame = frame.sort_values('window_center').reset_index(drop=True)

    time_axis, raw_curve = load_raw_per_finger_curve(
        project_root=project_root,
        data_config=data_config,
        sample=sample,
    )
    spans = frame[['window_start', 'window_end', 'phase_label']].drop_duplicates().sort_values('window_start')

    fig = plt.figure(figsize=(13, 7.2))
    grid = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[3.2, 0.85], hspace=0.12)
    force_ax = fig.add_subplot(grid[0, 0])
    gate_ax = fig.add_subplot(grid[1, 0], sharex=force_ax)
    shade_phases(force_ax, spans)
    shade_phases(gate_ax, spans, alpha=0.16)

    fallback_end = float(frame['window_end'].max()) if len(frame) else float(time_axis[-1])
    end_time = sample_end_time(sample, fallback=fallback_end)
    raw_mask = (time_axis >= 0.0) & (time_axis <= end_time)
    plot_series(
        force_ax,
        time_axis[raw_mask],
        raw_curve[raw_mask, finger_index],
        label='raw per-finger force',
        color=RAW_COLOR,
        linestyle='-',
        linewidth=0.85,
        marker=None,
    )
    target_mask = frame['has_target'].astype(bool).to_numpy()
    reference_mask = frame['has_reference'].astype(bool).to_numpy()
    x = frame['window_center'].to_numpy(dtype=float)
    plot_series(
        force_ax,
        x[target_mask],
        frame.loc[target_mask, 'target'].to_numpy(dtype=float),
        label='target',
        color=TARGET_COLOR,
        linewidth=1.8,
    )
    plot_series(
        force_ax,
        x[reference_mask],
        frame.loc[reference_mask, 'reference'].to_numpy(dtype=float),
        label='reference',
        color=REFERENCE_COLOR,
        linestyle='--',
        linewidth=1.4,
    )
    plot_series(
        force_ax,
        x,
        frame['pred'].to_numpy(dtype=float),
        label=f'{model_label} pred',
        color=PRED_COLOR,
        linewidth=1.55,
    )
    plot_series(
        gate_ax,
        x,
        frame['interface_gate'].to_numpy(dtype=float),
        label='interface gate',
        color=GATE_COLOR,
        linewidth=1.25,
    )

    for label, time_sec in event_times(sample):
        force_ax.axvline(time_sec, color='#6d597a', linestyle=':', linewidth=0.9)
        gate_ax.axvline(time_sec, color='#6d597a', linestyle=':', linewidth=0.9)
        gate_ax.text(time_sec, 1.04, label, rotation=90, va='bottom', ha='right', fontsize=8)

    force_ax.set_ylabel(f'{finger_name} force')
    force_ax.grid(alpha=0.24)
    force_ax.tick_params(labelbottom=False)
    force_ax.set_xlim(0.0, end_time)
    gate_ax.set_ylabel('gate')
    gate_ax.set_xlabel('time (s)')
    gate_ax.set_ylim(-0.05, 1.05)
    gate_ax.set_yticks([0.0, 0.5, 1.0])
    gate_ax.grid(alpha=0.2)

    handles = [
        Line2D([0], [0], color=RAW_COLOR, linewidth=1.1, label='raw per-finger force'),
        Line2D([0], [0], color=TARGET_COLOR, linewidth=1.8, label='target'),
        Line2D([0], [0], color=REFERENCE_COLOR, linestyle='--', linewidth=1.4, label='reference'),
        Line2D([0], [0], color=PRED_COLOR, linewidth=1.55, label=f'{model_label} pred'),
        Line2D([0], [0], color=GATE_COLOR, linewidth=1.25, label='interface gate'),
    ]
    handles.extend(Patch(facecolor=color, alpha=0.2, label=phase) for phase, color in PHASE_COLORS.items())
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=4, frameon=False)
    fig.suptitle(
        f'{sample_id} | object={sample.get("object_id")} | {finger_name} | single-finger policy prediction',
        y=0.905,
        fontsize=13,
    )
    fig.subplots_adjust(top=0.82)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches='tight')
    plt.close(fig)
    return {'sample_id': sample_id, 'finger': finger_name, 'output_path': str(output_path)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--project-root', default='.')
    parser.add_argument('--stage', default='configs/stages/stage38c_policy_fold1_per_finger_reference_delta_scale20_long.yaml')
    parser.add_argument('--checkpoint', default='runs/stage38c_policy_fold1_per_finger_reference_delta_scale20_long/checkpoints/best.pt')
    parser.add_argument('--model-label', default='stage38c')
    parser.add_argument('--subset', default='val', choices=['train', 'val', 'test'])
    parser.add_argument('--sample-id', nargs='*', default=None)
    parser.add_argument('--limit', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--output-dir', default='data/processed/debug/single_finger_policy_predictions')
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    context = prepare_evaluation_context(
        project_root=project_root,
        stage=args.stage,
        checkpoint=args.checkpoint,
        subset=args.subset,
        num_workers=int(args.num_workers),
    )
    sample_ids = select_sample_ids(context, args.sample_id, args.limit)
    rows = collect_prediction_rows(
        context,
        sample_ids,
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
    )
    output_dir = resolve_path(project_root, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = output_dir / f'{args.subset}_single_finger_prediction_rows.csv'
    rows.to_csv(rows_path, index=False)

    samples = pd.read_csv(project_root / 'data' / 'processed' / 'samples.csv')
    rendered = []
    for sample_id in sample_ids:
        for finger_index, finger_name in enumerate(FINGER_NAMES):
            output_path = output_dir / f'{sample_id}_{finger_name}_{args.model_label}_single_finger.png'
            rendered.append(
                render_finger_plot(
                    project_root=project_root,
                    data_config=context['data_config'],
                    samples=samples,
                    rows=rows,
                    sample_id=sample_id,
                    finger_index=finger_index,
                    model_label=args.model_label,
                    output_path=output_path,
                )
            )
            print(f'{sample_id} {finger_name} -> {output_path}')

    print({'sample_ids': sample_ids, 'rows_path': str(rows_path), 'output_dir': str(output_dir), 'rendered': rendered})


if __name__ == '__main__':
    main()
