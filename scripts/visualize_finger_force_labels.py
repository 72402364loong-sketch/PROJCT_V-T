from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cmg.config import deep_update, load_yaml
from cmg.data.tactile import (
    compute_clean_force_curve,
    compute_measured_force_curve,
    ema_filter,
    load_tactile_array,
)

try:
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover
    raise RuntimeError('matplotlib is required for visualize_finger_force_labels.py.') from exc


PHASE_COLORS = {
    'Water': '#2a6f97',
    'Interface': '#f4a261',
    'Air': '#8ab17d',
}
FINGER_COLORS = ['#0077b6', '#d62828', '#2a9d8f']
EVENT_FIELDS = [
    ('t_contact_all', 'contact'),
    ('t_grasp_stable', 'stable'),
    ('t_if_enter', 'if_enter'),
    ('t_if_exit', 'if_exit'),
    ('t_end', 'end'),
]


def resolve_data_path(project_root: Path, relative_path: str) -> Path:
    path = Path(relative_path)
    if path.is_absolute():
        return path
    if path.parts and path.parts[0] == 'data':
        return project_root / path
    return project_root / 'data' / path


def load_data_config(project_root: Path, stage_path: str | None) -> dict[str, Any]:
    data_config = load_yaml(project_root / 'configs' / 'data' / 'default.yaml')
    if stage_path:
        stage = load_yaml(project_root / stage_path)
        data_config = deep_update(data_config, stage.get('data', {}))
    return data_config


def parse_optional_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if np.isnan(parsed):
        return None
    return parsed


def collect_event_times(sample: pd.Series) -> list[tuple[str, float]]:
    events: list[tuple[str, float]] = []
    for field, label in EVENT_FIELDS:
        value = parse_optional_float(sample.get(field))
        if value is not None:
            events.append((label, value))
    return events


def resolve_reference_window_indices(windows: pd.DataFrame, reference_count: int) -> list[int]:
    first_interface_index = len(windows)
    for index, row in windows.reset_index(drop=True).iterrows():
        if str(row['phase_label']) == 'Interface':
            first_interface_index = int(index)
            break

    before_interface = windows.reset_index(drop=True).iloc[:first_interface_index]
    stable_water = [
        int(index)
        for index, row in before_interface.iterrows()
        if bool(row['is_stable_mask']) and str(row['phase_label']) == 'Water'
    ]
    if stable_water:
        return stable_water[-reference_count:]

    stable = [
        int(index)
        for index, row in before_interface.iterrows()
        if bool(row['is_stable_mask'])
    ]
    return stable[-reference_count:]


def window_vector_mean(
    curve: np.ndarray,
    start_idx: int,
    end_idx: int,
    sample_mask: np.ndarray | None = None,
) -> np.ndarray:
    segment = curve[start_idx:end_idx]
    if sample_mask is not None:
        segment = segment[sample_mask]
    if segment.size == 0:
        return np.full((curve.shape[1],), float('nan'), dtype=np.float32)
    return np.mean(segment, axis=0).astype(np.float32)


def aggregate_reference(values: list[np.ndarray], statistic: str) -> tuple[np.ndarray, bool]:
    finite = [
        np.asarray(value, dtype=np.float32)
        for value in values
        if np.asarray(value).ndim == 1 and bool(np.all(np.isfinite(value)))
    ]
    if not finite:
        return np.full((3,), float('nan'), dtype=np.float32), False
    stacked = np.stack(finite, axis=0)
    if str(statistic).strip().lower() == 'median':
        return np.median(stacked, axis=0).astype(np.float32), True
    return np.mean(stacked, axis=0).astype(np.float32), True


def window_interval_mask(
    time_axis: np.ndarray,
    start_idx: int,
    end_idx: int,
    *,
    interval_start: float,
    interval_end: float,
) -> np.ndarray:
    window_times = time_axis[start_idx:end_idx]
    if window_times.size == 0:
        return np.zeros(0, dtype=bool)
    return (window_times >= interval_start) & (window_times < interval_end)


def build_window_targets(
    sample: pd.Series,
    windows: pd.DataFrame,
    data_config: dict[str, Any],
    curve: np.ndarray,
    time_axis: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[int], np.ndarray | None]:
    windows = windows.reset_index(drop=True)
    reference_count = int(data_config.get('reference_force_window_count', 3))
    reference_indices = resolve_reference_window_indices(windows, reference_count)
    reference_values = [
        window_vector_mean(
            curve,
            int(windows.iloc[index]['tactile_start_idx']),
            int(windows.iloc[index]['tactile_end_idx']),
        )
        for index in reference_indices
    ]
    reference, has_reference = aggregate_reference(
        reference_values,
        str(data_config.get('reference_force_statistic', 'mean')),
    )

    window_means = np.full((len(windows), 3), float('nan'), dtype=np.float32)
    targets = np.full((len(windows), 3), float('nan'), dtype=np.float32)
    if_enter = parse_optional_float(sample.get('t_if_enter'))
    if_exit = parse_optional_float(sample.get('t_if_exit'))
    margin = float(data_config.get('expert_force_interface_margin_sec', 0.0))
    use_local_reference = (
        str(data_config.get('expert_force_mode', 'measured_force')).strip().lower() == 'local_reference_delta'
        and has_reference
        and if_enter is not None
        and if_exit is not None
        and if_exit > if_enter
    )

    for index, row in windows.iterrows():
        start_idx = int(row['tactile_start_idx'])
        end_idx = int(row['tactile_end_idx'])
        window_means[index] = window_vector_mean(curve, start_idx, end_idx)
        is_interface = str(row['phase_label']) == 'Interface'
        if use_local_reference:
            overlap = window_interval_mask(
                time_axis,
                start_idx,
                end_idx,
                interval_start=float(if_enter) - margin,
                interval_end=float(if_exit) + margin,
            )
            if is_interface:
                target = window_vector_mean(curve, start_idx, end_idx, sample_mask=overlap if overlap.any() else None)
                if np.all(np.isfinite(target)):
                    targets[index] = target
            elif index in reference_indices or overlap.any():
                targets[index] = reference
        else:
            targets[index] = window_means[index]

    return window_means, targets, reference_indices, reference if has_reference else None


def render_sample(
    project_root: Path,
    data_config: dict[str, Any],
    sample: pd.Series,
    windows: pd.DataFrame,
    output_path: Path,
) -> dict[str, Any]:
    tactile_path = resolve_data_path(project_root, str(sample['tactile_path']))
    tactile = load_tactile_array(tactile_path)
    tactile_dt = float(data_config['tactile_dt'])
    sync_offset_sec = parse_optional_float(sample.get('sync_offset_sec')) or 0.0
    contact_time = parse_optional_float(sample.get('t_contact_all'))
    time_axis = sync_offset_sec + np.arange(tactile.shape[0], dtype=np.float32) * tactile_dt

    smoothing_mode = str(data_config.get('expert_force_smoothing', 'zero_phase_ema'))
    baseline_mode = str(data_config.get('expert_force_baseline_mode', 'pre_contact_median'))
    baseline_window_sec = float(data_config.get('expert_force_baseline_window_sec', 0.5))
    alpha = float(data_config.get('ema_alpha_expert', 0.1))
    finger_curve = compute_clean_force_curve(
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
    signed_sum = compute_measured_force_curve(
        tactile,
        normal_sign_table=data_config['normal_sign_table'],
        alpha=None,
    )
    signed_sum_low = ema_filter(signed_sum, alpha=alpha)
    window_means, targets, reference_indices, reference = build_window_targets(
        sample,
        windows,
        data_config,
        finger_curve,
        time_axis,
    )

    fig = plt.figure(figsize=(18, 10))
    grid = fig.add_gridspec(nrows=3, ncols=1, height_ratios=[2.6, 1.6, 1.1], hspace=0.24)
    finger_ax = fig.add_subplot(grid[0, 0])
    signed_ax = fig.add_subplot(grid[1, 0], sharex=finger_ax)
    window_ax = fig.add_subplot(grid[2, 0], sharex=finger_ax)

    for finger_index, color in enumerate(FINGER_COLORS):
        label = f'finger{finger_index}'
        finger_ax.plot(time_axis, finger_curve[:, finger_index], color=color, linewidth=1.4, label=label)
        centers = windows['window_center'].to_numpy(dtype=np.float32)
        valid_target = np.isfinite(targets[:, finger_index])
        if np.any(valid_target):
            finger_ax.scatter(
                centers[valid_target],
                targets[valid_target, finger_index],
                color=color,
                s=18,
                marker='o',
                edgecolor='white',
                linewidth=0.4,
                zorder=4,
            )
        valid_mean = np.isfinite(window_means[:, finger_index])
        if np.any(valid_mean):
            finger_ax.scatter(
                centers[valid_mean],
                window_means[valid_mean, finger_index],
                color=color,
                s=12,
                marker='x',
                alpha=0.45,
                zorder=3,
            )
    if reference is not None:
        for finger_index, color in enumerate(FINGER_COLORS):
            finger_ax.axhline(float(reference[finger_index]), color=color, linestyle=':', linewidth=0.9, alpha=0.7)

    finger_ax.set_ylabel('Per-finger force')
    finger_ax.set_title('per_finger_z_abs_mean: mean(abs(z - pre_contact_baseline)) over 4 sensors')
    finger_ax.grid(alpha=0.25)
    finger_ax.legend(loc='upper right')

    signed_ax.plot(time_axis, signed_sum, color='#264653', linewidth=1.0, label='legacy signed-sum')
    signed_ax.plot(time_axis, signed_sum_low, color='#e76f51', linewidth=1.0, linestyle='--', label='EMA legacy signed-sum')
    signed_ax.set_ylabel('Legacy scalar')
    signed_ax.grid(alpha=0.25)
    signed_ax.legend(loc='upper right')

    windows = windows.reset_index(drop=True)
    window_ax.set_ylim(0, 1)
    window_ax.set_yticks([0.25, 0.78], labels=['phase', 'reference'])
    window_ax.grid(alpha=0.15)
    for index, row in windows.iterrows():
        color = PHASE_COLORS.get(str(row['phase_label']), '#999999')
        start = float(row['window_start'])
        end = float(row['window_end'])
        window_ax.axvspan(start, end, ymin=0.05, ymax=0.5, color=color, alpha=0.35)
        if index in reference_indices:
            window_ax.axvspan(start, end, ymin=0.62, ymax=0.95, color='#6d597a', alpha=0.55)
        window_ax.axvline(start, color='#cccccc', linewidth=0.4)
    if len(windows):
        window_ax.axvline(float(windows.iloc[-1]['window_end']), color='#cccccc', linewidth=0.4)

    for label, time_sec in collect_event_times(sample):
        for ax in (finger_ax, signed_ax, window_ax):
            ax.axvline(time_sec, color='#6d597a', linestyle=':', linewidth=1.0)
        finger_ax.text(time_sec, finger_ax.get_ylim()[1], label, rotation=90, va='top', ha='right', fontsize=8)

    window_ax.set_xlabel('Time (s)')
    title = (
        f'{sample["sample_id"]} | object={sample["object_id"]} | trial={sample["trial_result"]} | '
        f'smoothing={smoothing_mode} | baseline={baseline_mode}/{baseline_window_sec:.2f}s'
    )
    fig.suptitle(title, fontsize=13)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches='tight')
    plt.close(fig)
    return {
        'sample_id': str(sample['sample_id']),
        'output_path': str(output_path),
        'reference_indices': reference_indices,
        'has_reference': reference is not None,
    }


def resolve_sample_ids(samples: pd.DataFrame, args: argparse.Namespace) -> list[str]:
    selected = samples.copy()
    if args.trial_result is not None:
        selected = selected.loc[selected['trial_result'].astype(str).str.lower() == args.trial_result]
    if args.all:
        sample_ids = selected['sample_id'].astype(str).tolist()
    else:
        requested = args.sample_id or []
        if not requested:
            raise ValueError('Please provide --sample-id <id> [more ids ...] or use --all.')
        known = set(samples['sample_id'].astype(str))
        missing = [sample_id for sample_id in requested if sample_id not in known]
        if missing:
            raise ValueError(f'Unknown sample_id values: {missing}')
        selected_ids = set(selected['sample_id'].astype(str))
        sample_ids = [sample_id for sample_id in requested if sample_id in selected_ids]
    if args.limit is not None:
        sample_ids = sample_ids[: max(0, int(args.limit))]
    return sample_ids


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--project-root', default='.')
    parser.add_argument('--stage', default=None, help='Optional stage yaml; its data block overrides default data config.')
    parser.add_argument('--sample-id', nargs='*', default=None)
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--trial-result', choices=['stable', 'fail'], default='stable')
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--output', default=None, help='Single-sample output path. Only valid for exactly one sample.')
    parser.add_argument('--output-dir', default='data/processed/debug/finger_force_labels')
    args = parser.parse_args()

    if args.all and args.sample_id:
        parser.error('--all and --sample-id cannot be used together.')

    project_root = Path(args.project_root).resolve()
    data_config = load_data_config(project_root, args.stage)
    samples = pd.read_csv(project_root / 'data' / 'processed' / 'samples.csv')
    windows = pd.read_csv(project_root / 'data' / 'processed' / 'windows.csv')

    sample_ids = resolve_sample_ids(samples, args)
    if not sample_ids:
        raise ValueError('No samples matched the current selection.')
    if args.output and len(sample_ids) != 1:
        raise ValueError('--output is only valid when rendering exactly one sample.')

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = project_root / output_dir

    results: list[dict[str, Any]] = []
    for index, sample_id in enumerate(sample_ids, start=1):
        sample_rows = samples.loc[samples['sample_id'] == sample_id]
        if sample_rows.empty:
            raise ValueError(f'Unknown sample_id: {sample_id}')
        sample = sample_rows.iloc[0]
        sample_windows = windows.loc[windows['sample_id'] == sample_id].sort_values('window_start').reset_index(drop=True)
        if sample_windows.empty:
            raise ValueError(f'No windows found for sample_id: {sample_id}')

        if args.output:
            output_path = Path(args.output)
            if not output_path.is_absolute():
                output_path = project_root / output_path
        else:
            output_path = output_dir / f'{sample_id}_finger_force_labels.png'

        result = render_sample(project_root, data_config, sample, sample_windows, output_path)
        results.append(result)
        print(f'[{index}/{len(sample_ids)}] {sample_id} -> {output_path}')

    if len(results) == 1:
        print(results[0])
    else:
        print({'count': len(results), 'output_dir': str(output_dir)})


if __name__ == '__main__':
    main()
