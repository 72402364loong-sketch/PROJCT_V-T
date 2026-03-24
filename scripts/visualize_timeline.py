from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cmg.config import load_yaml
from cmg.data.sidecar import load_window_sidecar
from cmg.data.tactile import compute_measured_force_curve, ema_filter, load_tactile_array
from cmg.data.video import get_video_metadata, load_frame_at_index

try:
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover
    raise RuntimeError('matplotlib is required for visualize_timeline.py. Please install project dependencies.') from exc


PHASE_COLORS = {
    'Water': '#2a6f97',
    'Interface': '#f4a261',
    'Air': '#8ab17d',
}
EVENT_FIELDS = [
    ('t_contact_all', 'contact'),
    ('t_grasp_stable', 'stable'),
    ('t_if_enter', 'if_enter'),
    ('t_if_exit', 'if_exit'),
    ('t_end', 'end'),
]
FORCE_VIEW_LOW_PERCENTILE = 1.0
FORCE_VIEW_HIGH_PERCENTILE = 99.0
FORCE_VIEW_TRIGGER_RATIO = 1.8
FORCE_VIEW_PADDING = 0.08


def resolve_data_path(project_root: Path, relative_path: str) -> Path:
    path = Path(relative_path)
    if path.is_absolute():
        return path
    if path.parts and path.parts[0] == 'data':
        return project_root / path
    return project_root / 'data' / path


def pick_audit_window(windows: pd.DataFrame) -> pd.Series:
    interface_windows = windows.loc[windows['phase_label'] == 'Interface']
    if not interface_windows.empty:
        return interface_windows.iloc[len(interface_windows) // 2]
    return windows.iloc[len(windows) // 2]


def collect_event_times(sample: pd.Series) -> list[tuple[str, float]]:
    events: list[tuple[str, float]] = []
    for field, label in EVENT_FIELDS:
        value = pd.to_numeric(sample.get(field), errors='coerce')
        if not pd.isna(value):
            events.append((label, float(value)))
    return events


def build_preview_frames(video_path: Path, sample: pd.Series, image_size: int, roi: dict[str, int] | None) -> list[tuple[str, float, np.ndarray]]:
    metadata = get_video_metadata(video_path)
    fps = float(metadata['fps'])
    frame_count = int(metadata['frame_count'])
    previews: list[tuple[str, float, np.ndarray]] = []
    for label, time_sec in collect_event_times(sample):
        frame_index = min(max(int(round(time_sec * fps)), 0), max(0, frame_count - 1))
        frame = load_frame_at_index(video_path, frame_index, image_size=image_size, roi=roi)
        previews.append((label, time_sec, frame))
    return previews


def compute_force_view(measured_force: np.ndarray, measured_force_low: np.ndarray) -> dict[str, object]:
    series_list = [
        np.asarray(measured_force, dtype=np.float32),
        np.asarray(measured_force_low, dtype=np.float32),
    ]
    finite_arrays = [series[np.isfinite(series)] for series in series_list if series.size]
    finite_arrays = [series for series in finite_arrays if series.size]
    if not finite_arrays:
        return {
            'y_min': -1.0,
            'y_max': 1.0,
            'span': 2.0,
            'use_robust_view': False,
            'high_clip_mask': np.zeros_like(series_list[0], dtype=bool),
            'low_clip_mask': np.zeros_like(series_list[0], dtype=bool),
            'note': '',
        }

    finite = np.concatenate(finite_arrays)
    full_low = float(np.min(finite))
    full_high = float(np.max(finite))
    full_span = max(full_high - full_low, 1.0)

    if finite.size >= 32:
        robust_low = float(np.percentile(finite, FORCE_VIEW_LOW_PERCENTILE))
        robust_high = float(np.percentile(finite, FORCE_VIEW_HIGH_PERCENTILE))
    else:
        robust_low = full_low
        robust_high = full_high
    robust_span = max(robust_high - robust_low, 1.0)

    preliminary_low = robust_low - FORCE_VIEW_PADDING * robust_span
    preliminary_high = robust_high + FORCE_VIEW_PADDING * robust_span
    high_clip_mask = np.isfinite(measured_force) & (measured_force > preliminary_high)
    low_clip_mask = np.isfinite(measured_force) & (measured_force < preliminary_low)
    use_robust_view = (
        finite.size >= 32
        and full_span > robust_span * FORCE_VIEW_TRIGGER_RATIO
        and (bool(np.any(high_clip_mask)) or bool(np.any(low_clip_mask)))
    )

    if use_robust_view:
        span = robust_span
        y_min = preliminary_low
        y_max = preliminary_high
        note = (
            f'robust y-scale {FORCE_VIEW_LOW_PERCENTILE:.0f}-{FORCE_VIEW_HIGH_PERCENTILE:.0f} pct '
            f'| clipped {int(np.count_nonzero(high_clip_mask))} high / {int(np.count_nonzero(low_clip_mask))} low spikes'
        )
    else:
        span = full_span
        y_min = full_low - FORCE_VIEW_PADDING * full_span
        y_max = full_high + FORCE_VIEW_PADDING * full_span
        high_clip_mask = np.zeros_like(measured_force, dtype=bool)
        low_clip_mask = np.zeros_like(measured_force, dtype=bool)
        note = ''

    return {
        'y_min': y_min,
        'y_max': y_max,
        'span': span,
        'use_robust_view': use_robust_view,
        'high_clip_mask': high_clip_mask,
        'low_clip_mask': low_clip_mask,
        'note': note,
    }


def render_sample_timeline(
    project_root: Path,
    data_config: dict,
    sample: pd.Series,
    sample_windows: pd.DataFrame,
    *,
    output_path: Path,
) -> dict[str, object]:
    video_path = resolve_data_path(project_root, str(sample['video_path']))
    tactile_path = resolve_data_path(project_root, str(sample['tactile_path']))
    tactile_dt = float(data_config['tactile_dt'])
    tactile_array = load_tactile_array(tactile_path)
    tactile_times = float(sample['sync_offset_sec']) + np.arange(tactile_array.shape[0], dtype=np.float32) * tactile_dt
    measured_force = compute_measured_force_curve(tactile_array, normal_sign_table=data_config['normal_sign_table'])
    measured_force_low = ema_filter(measured_force, alpha=float(data_config['ema_alpha_expert']))
    force_view = compute_force_view(measured_force, measured_force_low)

    metadata = get_video_metadata(video_path)
    audit_window = pick_audit_window(sample_windows)
    sidecar = None
    if isinstance(audit_window.get('sidecar_cache_path'), str) and audit_window['sidecar_cache_path']:
        sidecar = load_window_sidecar(project_root, audit_window['sidecar_cache_path'])

    previews = build_preview_frames(
        video_path,
        sample,
        image_size=int(data_config.get('image_size', 224)),
        roi=data_config.get('roi'),
    )

    nrows = 4 if previews else 3
    height_ratios = [2.2, 2.4, 1.3, 1.3] if previews else [2.4, 1.3, 1.3]
    fig = plt.figure(figsize=(18, 11))
    outer = fig.add_gridspec(nrows=nrows, ncols=1, height_ratios=height_ratios, hspace=0.28)

    row_offset = 0
    if previews:
        preview_grid = outer[0].subgridspec(1, len(previews))
        for index, (label, time_sec, frame) in enumerate(previews):
            ax = fig.add_subplot(preview_grid[0, index])
            ax.imshow(frame)
            ax.set_title(f'{label}\n{time_sec:.3f}s', fontsize=10)
            ax.axis('off')
        row_offset = 1

    force_ax = fig.add_subplot(outer[row_offset, 0])
    window_ax = fig.add_subplot(outer[row_offset + 1, 0], sharex=force_ax)
    align_ax = fig.add_subplot(outer[row_offset + 2, 0], sharex=force_ax)

    force_ax.plot(tactile_times, measured_force, color='#264653', linewidth=1.2, label='F_meas')
    force_ax.plot(tactile_times, measured_force_low, color='#e76f51', linewidth=1.0, linestyle='--', label='EMA(F_meas)')
    force_ax.set_ylabel('Force')
    force_ax.set_ylim(float(force_view['y_min']), float(force_view['y_max']))
    force_ax.legend(loc='upper right')
    force_ax.grid(alpha=0.25)

    if bool(force_view['use_robust_view']):
        note = str(force_view['note'])
        force_ax.text(
            0.01,
            0.98,
            note,
            transform=force_ax.transAxes,
            ha='left',
            va='top',
            fontsize=8,
            bbox={'boxstyle': 'round,pad=0.25', 'facecolor': 'white', 'edgecolor': '#d0d0d0', 'alpha': 0.85},
        )
        span = float(force_view['span'])
        high_marker_y = float(force_view['y_max']) - 0.03 * span
        low_marker_y = float(force_view['y_min']) + 0.03 * span
        high_clip_mask = np.asarray(force_view['high_clip_mask'], dtype=bool)
        low_clip_mask = np.asarray(force_view['low_clip_mask'], dtype=bool)
        if np.any(high_clip_mask):
            force_ax.scatter(
                tactile_times[high_clip_mask],
                np.full(int(np.count_nonzero(high_clip_mask)), high_marker_y, dtype=np.float32),
                marker='^',
                s=22,
                color='#c1121f',
                alpha=0.9,
                label='_nolegend_',
                zorder=4,
            )
        if np.any(low_clip_mask):
            force_ax.scatter(
                tactile_times[low_clip_mask],
                np.full(int(np.count_nonzero(low_clip_mask)), low_marker_y, dtype=np.float32),
                marker='v',
                s=22,
                color='#c1121f',
                alpha=0.9,
                label='_nolegend_',
                zorder=4,
            )

    event_label_y = float(force_view['y_max']) - 0.01 * float(force_view['span'])
    for label, time_sec in collect_event_times(sample):
        force_ax.axvline(time_sec, color='#6d597a', linestyle=':', linewidth=1.0)
        force_ax.text(time_sec, event_label_y, label, rotation=90, va='top', ha='right', fontsize=8)

    window_ax.set_ylim(0, 1)
    window_ax.set_yticks([0.25, 0.8], labels=['phase', 'stable'])
    window_ax.grid(alpha=0.15)
    for _, row in sample_windows.iterrows():
        color = PHASE_COLORS.get(str(row['phase_label']), '#999999')
        start = float(row['window_start'])
        end = float(row['window_end'])
        window_ax.axvspan(start, end, ymin=0.05, ymax=0.55, color=color, alpha=0.35)
        if int(row['is_stable_mask']) == 1:
            window_ax.axvspan(start, end, ymin=0.65, ymax=0.95, color=color, alpha=0.85)
        window_ax.axvline(start, color='#cccccc', linewidth=0.4)
    window_ax.axvline(float(sample_windows.iloc[-1]['window_end']), color='#cccccc', linewidth=0.4)
    for label, time_sec in collect_event_times(sample):
        window_ax.axvline(time_sec, color='#6d597a', linestyle=':', linewidth=1.0)
    window_ax.set_ylabel('Windows')

    align_ax.grid(alpha=0.15)
    if sidecar is not None:
        fps = float(metadata['fps'])
        video_times_all = sidecar['video_frame_times_all']
        sampled_indices = sidecar['video_default_sampled_indices']
        sampled_mask = sidecar['video_default_sampled_valid_mask'].astype(bool)
        sampled_times = sampled_indices[sampled_mask].astype(np.float32) / fps if sampled_indices.size else np.zeros(0, dtype=np.float32)
        tactile_raw_times = sidecar['tactile_raw_times'].astype(np.float32)
        tactile_resampled_times = sidecar['tactile_resample_target_times'].astype(np.float32)
        align_ax.eventplot(
            [video_times_all, sampled_times, tactile_raw_times, tactile_resampled_times],
            lineoffsets=[3, 2, 1, 0],
            linelengths=0.7,
            colors=['#577590', '#277da1', '#f8961e', '#f94144'],
        )
        align_ax.set_yticks([3, 2, 1, 0], labels=['video_all', 'video_sampled', 'tactile_raw', 'tactile_resampled'])
        align_ax.set_title(f'Audit Window: {audit_window["window_id"]}', fontsize=10)
    else:
        align_ax.text(0.5, 0.5, 'No sidecar cache found for audit window.', transform=align_ax.transAxes, ha='center', va='center')
        align_ax.set_yticks([])

    for label, time_sec in collect_event_times(sample):
        align_ax.axvline(time_sec, color='#6d597a', linestyle=':', linewidth=1.0)
    align_ax.set_xlabel('Time (s)')

    title = (
        f'{sample["sample_id"]} | object={sample["object_id"]} | trial={sample["trial_result"]} | '
        f'sync_offset={float(sample["sync_offset_sec"]):.3f}s | '
        f'video_dur={float(metadata["duration"]):.3f}s | tactile_end={float(tactile_times[-1]) if tactile_times.size else 0.0:.3f}s'
    )
    fig.suptitle(title, fontsize=13)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches='tight')
    plt.close(fig)
    return {
        'sample_id': str(sample['sample_id']),
        'output_path': str(output_path),
        'audit_window': str(audit_window['window_id']),
        'robust_force_view': bool(force_view['use_robust_view']),
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
        missing = [sample_id for sample_id in requested if sample_id not in set(samples['sample_id'].astype(str))]
        if missing:
            raise ValueError(f'Unknown sample_id values: {missing}')
        sample_ids = [sample_id for sample_id in requested if sample_id in set(selected['sample_id'].astype(str))]

    if args.limit is not None:
        sample_ids = sample_ids[: max(0, int(args.limit))]
    return sample_ids


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--project-root', default='.')
    parser.add_argument('--sample-id', nargs='*', default=None)
    parser.add_argument('--all', action='store_true', help='Render all samples into the default timeline directory.')
    parser.add_argument('--trial-result', choices=['stable', 'fail'], default=None, help='Optionally filter the selected samples by trial_result.')
    parser.add_argument('--limit', type=int, default=None, help='Limit the number of rendered samples after filtering.')
    parser.add_argument('--output', default=None, help='Single-sample output path. Only valid when rendering exactly one sample.')
    args = parser.parse_args()

    if args.all and args.sample_id:
        parser.error('--all and --sample-id cannot be used together.')

    project_root = Path(args.project_root).resolve()
    data_config = load_yaml(project_root / 'configs' / 'data' / 'default.yaml')
    samples = pd.read_csv(project_root / 'data' / 'processed' / 'samples.csv')
    windows = pd.read_csv(project_root / 'data' / 'processed' / 'windows.csv')

    sample_ids = resolve_sample_ids(samples, args)
    if not sample_ids:
        raise ValueError('No samples matched the current selection.')

    results: list[dict[str, object]] = []
    for index, sample_id in enumerate(sample_ids, start=1):
        sample_rows = samples.loc[samples['sample_id'] == sample_id]
        if sample_rows.empty:
            raise ValueError(f'Unknown sample_id: {sample_id}')
        sample = sample_rows.iloc[0]
        sample_windows = windows.loc[windows['sample_id'] == sample_id].sort_values('window_start').reset_index(drop=True)
        if sample_windows.empty:
            raise ValueError(f'No windows found for sample_id: {sample_id}')

        if len(sample_ids) == 1 and args.output:
            output_path = Path(args.output)
            if not output_path.is_absolute():
                output_path = project_root / output_path
        else:
            output_path = project_root / 'data' / 'processed' / 'debug' / 'timelines' / f'{sample_id}.png'

        result = render_sample_timeline(
            project_root,
            data_config,
            sample,
            sample_windows,
            output_path=output_path,
        )
        results.append(result)
        print(f'[{index}/{len(sample_ids)}] {sample_id} -> {output_path}')

    if len(results) == 1:
        print(results[0])
    else:
        robust_count = sum(1 for result in results if bool(result['robust_force_view']))
        print(
            {
                'count': len(results),
                'output_dir': str((project_root / 'data' / 'processed' / 'debug' / 'timelines').resolve()),
                'robust_force_view_count': robust_count,
                'trial_result_filter': args.trial_result,
            }
        )


if __name__ == '__main__':
    main()
