from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from .annotations import normalize_object_attributes, normalize_sample_events
from .sidecar import write_window_sidecar
from .tactile import compute_resample_mapping, load_tactile_array
from .video import build_frame_time_array, build_window_frame_indices, get_video_metadata, sample_frame_indices_with_mapping
from .windowing import determine_tail_type, label_window


def _resolve_data_path(project_root: Path, relative_path: str) -> Path:
    relative = Path(relative_path)
    if relative.is_absolute():
        return relative
    if relative.parts and relative.parts[0] == 'data':
        return project_root / relative
    return project_root / 'data' / relative


def _round_time(value: float) -> float:
    return round(float(value), 3)


def _resolve_interface_bounds(sample: dict[str, object]) -> tuple[float, float]:
    t_end = float(sample['t_end'])
    has_if_enter = not pd.isna(sample['t_if_enter'])
    has_if_exit = not pd.isna(sample['t_if_exit'])
    if has_if_enter and has_if_exit:
        return float(sample['t_if_enter']), float(sample['t_if_exit'])
    if has_if_enter and not has_if_exit:
        return float(sample['t_if_enter']), t_end
    return t_end, t_end


def _project_target_times(raw_times: np.ndarray, mapping: dict[str, np.ndarray]) -> np.ndarray:
    if raw_times.size == 0:
        return np.full(mapping['target_positions'].shape, np.nan, dtype=np.float32)
    if raw_times.size == 1:
        return np.full(mapping['target_positions'].shape, float(raw_times[0]), dtype=np.float32)
    return np.interp(
        mapping['target_positions'],
        mapping['source_positions'],
        raw_times.astype(np.float32),
    ).astype(np.float32)


def build_samples_table(project_root: str | Path) -> pd.DataFrame:
    root = Path(project_root)
    objects = normalize_object_attributes(root / 'data' / 'annotations' / 'object_attributes.csv')
    samples = normalize_sample_events(root / 'data' / 'annotations' / 'sample_events.csv')
    merged = samples.merge(objects, on='object_id', how='left', validate='m:1')
    ordered_columns = [
        'sample_id',
        'video_path',
        'tactile_path',
        'object_id',
        'object_name',
        'object_alias',
        'object_pool',
        'fragility',
        'geometry',
        'surface',
        'water_condition',
        'lift_speed',
        'placement_variant',
        'trial_id',
        'trial_result',
        't_start',
        't_contact_all',
        't_grasp_stable',
        't_if_enter',
        't_if_exit',
        't_end',
        'notes',
        'object_notes',
        'sync_offset_sec',
        'sync_audit_status',
        'sync_audit_note',
    ]
    return merged[ordered_columns].sort_values('sample_id').reset_index(drop=True)


def build_windows_table(
    samples: pd.DataFrame,
    project_root: str | Path,
    *,
    video_fps: float,
    tactile_dt: float,
    window_size_sec: float,
    window_stride_sec: float,
    interface_overlap_sec: float,
    stable_margin_sec: float,
    short_tail_min_sec: float,
    num_frames_per_window: int | None = None,
    tactile_points_per_window: int | None = None,
    write_sidecar_cache: bool = True,
) -> pd.DataFrame:
    root = Path(project_root)
    expected_video_frames = int(round(window_size_sec * video_fps))
    expected_tactile_points = int(round(window_size_sec / tactile_dt))
    rows: list[dict[str, Any]] = []
    for sample in samples.to_dict('records'):
        video_path = _resolve_data_path(root, sample['video_path'])
        tactile_path = _resolve_data_path(root, sample['tactile_path'])
        tactile_array = load_tactile_array(tactile_path)
        tactile_count = int(tactile_array.shape[0])
        tactile_end = _round_time(float(sample['sync_offset_sec']) + tactile_count * tactile_dt)
        metadata = get_video_metadata(video_path)
        video_duration = _round_time(float(metadata['duration']))
        frame_count = int(metadata['frame_count'])
        effective_end = _round_time(min(float(sample['t_end']), tactile_end, video_duration))
        t_if_enter, t_if_exit = _resolve_interface_bounds(sample)
        tail_type = determine_tail_type(
            t_effective_end=effective_end,
            t_if_exit=t_if_exit,
            full_tail_sec=window_size_sec,
            short_tail_min_sec=short_tail_min_sec,
        )
        max_start = effective_end - window_size_sec
        if max_start < 0:
            continue
        window_starts = np.arange(float(sample['t_start']), max_start + 1e-6, window_stride_sec)
        tactile_times_full = float(sample['sync_offset_sec']) + np.arange(tactile_count, dtype=np.float32) * tactile_dt
        for window_index, window_start in enumerate(window_starts):
            window_start = _round_time(window_start)
            window_end = _round_time(window_start + window_size_sec)
            window_id = f"{sample['sample_id']}_W{window_index:03d}"
            label = label_window(
                window_start=window_start,
                window_end=window_end,
                t_if_enter=t_if_enter,
                t_if_exit=t_if_exit,
                overlap_threshold=interface_overlap_sec,
                stable_margin=stable_margin_sec,
            )
            frame_indices = build_window_frame_indices(
                frame_count=frame_count,
                fps=float(metadata['fps'] or video_fps),
                start_time=window_start,
                end_time=window_end,
            )
            frame_times = build_frame_time_array(frame_indices, float(metadata['fps'] or video_fps))
            sampled_frame_indices: list[int] = []
            sampled_frame_mask: list[bool] = []
            sampled_frame_positions: list[float] = []
            if num_frames_per_window is not None:
                sampled_frame_indices, sampled_frame_mask, sampled_frame_positions = sample_frame_indices_with_mapping(
                    frame_indices,
                    num_frames_per_window,
                )
            tactile_start_idx = int(np.searchsorted(tactile_times_full, window_start, side='left'))
            tactile_end_idx = int(np.searchsorted(tactile_times_full, window_end, side='left'))
            tactile_points = max(0, tactile_end_idx - tactile_start_idx)
            tactile_indices = np.arange(tactile_start_idx, tactile_end_idx, dtype=np.int32)
            tactile_times_window = tactile_times_full[tactile_start_idx:tactile_end_idx]
            if tactile_points_per_window is not None:
                tactile_mapping = compute_resample_mapping(tactile_points, tactile_points_per_window)
                tactile_target_times = _project_target_times(tactile_times_window, tactile_mapping)
            else:
                tactile_mapping = {
                    'source_positions': np.zeros(0, dtype=np.float32),
                    'target_positions': np.zeros(0, dtype=np.float32),
                    'left_indices': np.zeros(0, dtype=np.int32),
                    'right_indices': np.zeros(0, dtype=np.int32),
                    'left_weights': np.zeros(0, dtype=np.float32),
                    'right_weights': np.zeros(0, dtype=np.float32),
                    'valid_mask': np.zeros(0, dtype=bool),
                }
                tactile_target_times = np.zeros(0, dtype=np.float32)

            sidecar_cache_path = ''
            if write_sidecar_cache:
                sidecar_cache_path = write_window_sidecar(
                    root,
                    sample_id=str(sample['sample_id']),
                    window_id=window_id,
                    payload={
                        'window_start': np.asarray([window_start], dtype=np.float32),
                        'window_end': np.asarray([window_end], dtype=np.float32),
                        'window_center': np.asarray([window_start + 0.5 * window_size_sec], dtype=np.float32),
                        'sync_offset_sec': np.asarray([float(sample['sync_offset_sec'])], dtype=np.float32),
                        'video_frame_indices_all': np.asarray(frame_indices, dtype=np.int32),
                        'video_frame_times_all': frame_times.astype(np.float32),
                        'video_default_sampled_indices': np.asarray(sampled_frame_indices, dtype=np.int32),
                        'video_default_sampled_valid_mask': np.asarray(sampled_frame_mask, dtype=bool),
                        'video_default_sampled_source_positions': np.asarray(sampled_frame_positions, dtype=np.float32),
                        'tactile_raw_indices': tactile_indices.astype(np.int32),
                        'tactile_raw_times': tactile_times_window.astype(np.float32),
                        'tactile_resample_source_positions': tactile_mapping['source_positions'].astype(np.float32),
                        'tactile_resample_target_positions': tactile_mapping['target_positions'].astype(np.float32),
                        'tactile_resample_left_indices': tactile_mapping['left_indices'].astype(np.int32),
                        'tactile_resample_right_indices': tactile_mapping['right_indices'].astype(np.int32),
                        'tactile_resample_left_weights': tactile_mapping['left_weights'].astype(np.float32),
                        'tactile_resample_right_weights': tactile_mapping['right_weights'].astype(np.float32),
                        'tactile_resample_valid_mask': tactile_mapping['valid_mask'].astype(bool),
                        'tactile_resample_target_times': tactile_target_times.astype(np.float32),
                    },
                )

            rows.append(
                {
                    'window_id': window_id,
                    'sample_id': sample['sample_id'],
                    'object_id': sample['object_id'],
                    'object_pool': sample['object_pool'],
                    'window_start': window_start,
                    'window_end': window_end,
                    'window_center': _round_time(window_start + 0.5 * window_size_sec),
                    'phase_label': label.phase_label,
                    'is_stable_mask': int(label.is_stable_mask),
                    'stable_phase': label.stable_phase or '',
                    'tail_type': tail_type,
                    'valid_ratio_video': round(len(frame_indices) / max(1, expected_video_frames), 6),
                    'valid_ratio_tactile': round(tactile_points / max(1, expected_tactile_points), 6),
                    'num_video_frames': len(frame_indices),
                    'num_tactile_points': tactile_points,
                    'video_frame_indices_json': json.dumps(frame_indices, ensure_ascii=False),
                    'tactile_start_idx': tactile_start_idx,
                    'tactile_end_idx': tactile_end_idx,
                    'water_condition': sample['water_condition'],
                    'lift_speed': sample['lift_speed'],
                    'placement_variant': sample['placement_variant'],
                    'trial_result': sample['trial_result'],
                    'sync_offset_sec': _round_time(sample['sync_offset_sec']),
                    'sidecar_cache_path': sidecar_cache_path,
                }
            )
    return pd.DataFrame(rows).sort_values(['sample_id', 'window_start']).reset_index(drop=True)


def write_preprocessed_outputs(project_root: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    root = Path(project_root)
    processed_root = root / 'data' / 'processed'
    stats_root = processed_root / 'stats'
    processed_root.mkdir(parents=True, exist_ok=True)
    stats_root.mkdir(parents=True, exist_ok=True)

    data_config_path = root / 'configs' / 'data' / 'default.yaml'
    data_config = {}
    if data_config_path.exists():
        with data_config_path.open('r', encoding='utf-8') as handle:
            data_config = yaml.safe_load(handle) or {}

    samples = build_samples_table(root)
    windows = build_windows_table(
        samples=samples,
        project_root=root,
        video_fps=float(data_config.get('video_fps', 30.0)),
        tactile_dt=float(data_config.get('tactile_dt', 0.039)),
        window_size_sec=float(data_config.get('window_size_sec', 1.0)),
        window_stride_sec=float(data_config.get('window_stride_sec', 0.25)),
        interface_overlap_sec=float(data_config.get('interface_overlap_sec', 0.25)),
        stable_margin_sec=float(data_config.get('stable_margin_sec', 0.25)),
        short_tail_min_sec=float(data_config.get('short_tail_min_sec', 0.5)),
        num_frames_per_window=int(data_config.get('num_frames_per_window')) if data_config.get('num_frames_per_window') is not None else None,
        tactile_points_per_window=int(data_config.get('tactile_points_per_window')) if data_config.get('tactile_points_per_window') is not None else None,
        write_sidecar_cache=bool(data_config.get('write_sidecar_cache', True)),
    )
    samples.to_csv(processed_root / 'samples.csv', index=False, encoding='utf-8-sig')
    windows.to_csv(processed_root / 'windows.csv', index=False, encoding='utf-8-sig')

    summary = {
        'num_samples': int(len(samples)),
        'num_windows': int(len(windows)),
        'phase_counts': windows['phase_label'].value_counts().to_dict(),
        'trial_result_counts': samples['trial_result'].value_counts().to_dict(),
        'object_pool_counts': samples['object_pool'].value_counts().to_dict(),
    }
    with (stats_root / 'dataset_summary.json').open('w', encoding='utf-8') as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    return samples, windows
