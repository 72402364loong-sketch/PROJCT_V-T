from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from cmg.constants import DIRECTION_TO_INDEX, MEDIUM_TO_PHASE

from .annotations import normalize_object_attributes, normalize_sample_events
from .sidecar import write_window_sidecar
from .splits import derive_split_labels, resolve_split_config
from .tactile import compute_resample_mapping, load_tactile_array
from .video import build_frame_time_array, build_window_frame_indices, get_video_metadata, sample_frame_indices_with_mapping
from .windowing import compute_phase_label_at_timestamp, determine_tail_type, label_window


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_data_path(project_root: Path, relative_path: str) -> Path:
    relative = Path(relative_path)
    if relative.is_absolute():
        return relative
    if relative.parts and relative.parts[0] == 'data':
        return project_root / relative
    return project_root / 'data' / relative


def _round_time(value: float) -> float:
    return round(float(value), 6)


def _resolve_project_path(project_root: Path, raw_path: str | Path) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else project_root / path


def _git_revision(project_root: Path) -> str:
    try:
        result = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return 'unknown'


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


def build_samples_table(
    project_root: str | Path,
    *,
    split_path: str | Path | None = None,
    split_version: str | None = None,
    reference_duration_sec: float = 0.75,
    reference_statistic: str = 'median',
    reference_source: str = 'raw_tactile',
    w2a_force_baseline_mode: str = 'pre_contact_median',
    a2w_force_baseline_mode: str = 'none',
) -> pd.DataFrame:
    root = Path(project_root)
    objects = normalize_object_attributes(root / 'data' / 'annotations' / 'object_attributes.csv')
    samples = normalize_sample_events(root / 'data' / 'annotations' / 'sample_events.csv')
    merged = samples.merge(objects, on='object_id', how='left', validate='m:1')
    merged['direction_index'] = merged['direction'].map(DIRECTION_TO_INDEX)
    if merged['direction_index'].isna().any():
        invalid = sorted(merged.loc[merged['direction_index'].isna(), 'direction'].astype(str).unique().tolist())
        raise ValueError(f'Unsupported direction values while building samples: {invalid}')
    merged['direction_index'] = merged['direction_index'].astype(int)

    if split_path is None:
        merged['split'] = 'unassigned'
        resolved_split_version = split_version or ''
    else:
        resolved_split_path = _resolve_project_path(root, split_path)
        split_config = resolve_split_config(resolved_split_path)
        resolved_split_version = split_version or str(split_config.get('name', resolved_split_path.stem))
        if split_version and split_version != split_config.get('name'):
            raise ValueError(
                f'split_version={split_version!r} does not match split name={split_config.get("name")!r}.'
            )
        merged['split'] = derive_split_labels(merged, resolved_split_path)
    merged['split_version'] = resolved_split_version

    reference_end = pd.to_numeric(merged['t_if_enter'], errors='coerce')
    reference_start = (reference_end - float(reference_duration_sec)).clip(
        lower=pd.to_numeric(merged['t_start'], errors='coerce')
    )
    merged['reference_start_time'] = reference_start.round(6)
    merged['reference_end_time'] = reference_end.round(6)
    merged['reference_duration_sec'] = float(reference_duration_sec)
    merged['reference_statistic'] = str(reference_statistic)
    merged['reference_source'] = str(reference_source)
    merged['has_contact_event'] = merged['t_contact_all'].notna().astype(int)
    merged['force_baseline_mode'] = np.where(
        merged['direction'].eq('W2A') & merged['t_contact_all'].notna(),
        str(w2a_force_baseline_mode),
        str(a2w_force_baseline_mode),
    )
    merged['has_reference_candidate'] = (
        merged['trial_result'].eq('stable')
        & merged['reference_start_time'].notna()
        & merged['reference_end_time'].notna()
        & merged['reference_start_time'].lt(merged['reference_end_time'])
    ).astype(int)
    merged['policy_supervision_eligible'] = merged['trial_result'].eq('stable').astype(int)
    ordered_columns = [
        'sample_id',
        'video_path',
        'tactile_path',
        'object_id',
        'physical_object_uid',
        'object_name',
        'object_alias',
        'object_pool',
        'fragility',
        'geometry',
        'surface',
        'direction',
        'direction_index',
        'source_medium',
        'target_medium',
        'reference_medium',
        'split',
        'split_version',
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
        'reference_start_time',
        'reference_end_time',
        'reference_duration_sec',
        'reference_statistic',
        'reference_source',
        'has_contact_event',
        'force_baseline_mode',
        'has_reference_candidate',
        'policy_supervision_eligible',
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
    policy_timestamp_anchor: str = 'window_center',
    causal_only: bool = False,
    sidecar_cache_relative_root: str | Path = Path('processed') / 'cache',
    target_aggregation_sec: float | None = None,
    reference_duration_sec: float | None = None,
    phase_label_mode: str = 'window_overlap',
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
        source_phase = MEDIUM_TO_PHASE[str(sample['source_medium'])]
        target_phase = MEDIUM_TO_PHASE[str(sample['target_medium'])]
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
        anchor = str(policy_timestamp_anchor).strip().lower()
        if anchor not in {'window_start', 'window_center', 'window_end'}:
            raise ValueError(f'Unsupported policy_timestamp_anchor={policy_timestamp_anchor!r}.')
        normalized_phase_label_mode = str(phase_label_mode).strip().lower()
        if normalized_phase_label_mode not in {'window_overlap', 'policy_timestamp'}:
            raise ValueError(f'Unsupported phase_label_mode={phase_label_mode!r}.')
        for window_index, window_start in enumerate(window_starts):
            window_start = _round_time(window_start)
            window_end = _round_time(window_start + window_size_sec)
            window_center = _round_time(window_start + 0.5 * window_size_sec)
            policy_timestamp = {
                'window_start': window_start,
                'window_center': window_center,
                'window_end': window_end,
            }[anchor]
            target_start = policy_timestamp if target_aggregation_sec is None else policy_timestamp - float(target_aggregation_sec)
            reference_start = float(sample['reference_start_time']) if not pd.isna(sample['reference_start_time']) else float('nan')
            reference_end = float(sample['reference_end_time']) if not pd.isna(sample['reference_end_time']) else float('nan')
            window_id = f"{sample['sample_id']}_W{window_index:03d}"
            label = label_window(
                window_start=window_start,
                window_end=window_end,
                t_if_enter=t_if_enter,
                t_if_exit=t_if_exit,
                overlap_threshold=interface_overlap_sec,
                stable_margin=stable_margin_sec,
                source_phase=source_phase,
                target_phase=target_phase,
            )
            semantic_phase_label = compute_phase_label_at_timestamp(
                policy_timestamp,
                t_if_enter,
                t_if_exit,
                source_phase=source_phase,
                target_phase=target_phase,
            )
            phase_label = semantic_phase_label if normalized_phase_label_mode == 'policy_timestamp' else label.phase_label
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
            if causal_only:
                if frame_times.size and float(np.max(frame_times)) > policy_timestamp + 1e-6:
                    raise ValueError(f'Future video frame detected in {window_id}.')
                if tactile_times_window.size and float(np.max(tactile_times_window)) > policy_timestamp + 1e-6:
                    raise ValueError(f'Future tactile sample detected in {window_id}.')
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
                        'window_center': np.asarray([window_center], dtype=np.float32),
                        'policy_timestamp': np.asarray([policy_timestamp], dtype=np.float32),
                        'causal_only': np.asarray([causal_only], dtype=bool),
                        'target_interval_start': np.asarray([target_start], dtype=np.float32),
                        'target_interval_end': np.asarray([policy_timestamp], dtype=np.float32),
                        'reference_interval_start': np.asarray([reference_start], dtype=np.float32),
                        'reference_interval_end': np.asarray([reference_end], dtype=np.float32),
                        'reference_eligible': np.asarray([bool(sample['has_reference_candidate'])], dtype=bool),
                        'policy_supervision_eligible': np.asarray([bool(sample['policy_supervision_eligible'])], dtype=bool),
                        'direction': np.asarray([str(sample['direction'])]),
                        'direction_index': np.asarray([int(sample['direction_index'])], dtype=np.int64),
                        'source_medium': np.asarray([str(sample['source_medium'])]),
                        'target_medium': np.asarray([str(sample['target_medium'])]),
                        'reference_medium': np.asarray([str(sample['reference_medium'])]),
                        'split': np.asarray([str(sample['split'])]),
                        'split_version': np.asarray([str(sample['split_version'])]),
                        'physical_object_uid': np.asarray([str(sample['physical_object_uid'])]),
                        'phase_label': np.asarray([phase_label]),
                        'semantic_phase_label': np.asarray([semantic_phase_label]),
                        'window_overlap_phase_label': np.asarray([label.phase_label]),
                        'context_stable_mask': np.asarray([label.is_stable_mask], dtype=bool),
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
                    cache_relative_root=sidecar_cache_relative_root,
                )

            rows.append(
                {
                    'window_id': window_id,
                    'sample_id': sample['sample_id'],
                    'object_id': sample['object_id'],
                    'physical_object_uid': sample['physical_object_uid'],
                    'object_pool': sample['object_pool'],
                    'direction': sample['direction'],
                    'direction_index': int(sample['direction_index']),
                    'source_medium': sample['source_medium'],
                    'target_medium': sample['target_medium'],
                    'reference_medium': sample['reference_medium'],
                    'source_phase': source_phase,
                    'target_phase': target_phase,
                    'split': sample['split'],
                    'split_version': sample['split_version'],
                    'window_start': window_start,
                    'window_end': window_end,
                    'window_center': window_center,
                    'policy_timestamp': policy_timestamp,
                    'phase_label': phase_label,
                    'semantic_phase_label': semantic_phase_label,
                    'window_overlap_phase_label': label.phase_label,
                    'is_stable_mask': int(label.is_stable_mask),
                    'context_stable_mask': int(label.is_stable_mask),
                    'stable_phase': label.stable_phase or '',
                    'reference_interval_start': reference_start,
                    'reference_interval_end': reference_end,
                    'reference_eligible': int(sample['has_reference_candidate']),
                    'policy_supervision_eligible': int(sample['policy_supervision_eligible']),
                    'has_contact_event': int(sample['has_contact_event']),
                    'force_baseline_mode': sample['force_baseline_mode'],
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


def write_preprocessed_outputs(
    project_root: str | Path,
    *,
    config_path: str | Path = 'configs/data/default.yaml',
    output_dir: str | Path | None = None,
    generation_command: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    root = Path(project_root)
    data_config_path = _resolve_project_path(root, config_path)
    data_config = {}
    if data_config_path.exists():
        with data_config_path.open('r', encoding='utf-8') as handle:
            data_config = yaml.safe_load(handle) or {}

    configured_output = output_dir or data_config.get('processed_dir', 'data/processed')
    processed_root = _resolve_project_path(root, configured_output)
    stats_root = processed_root / 'stats'
    processed_root.mkdir(parents=True, exist_ok=True)
    stats_root.mkdir(parents=True, exist_ok=True)

    reference_config = data_config.get('reference', {})
    force_baseline_config = data_config.get('force_baseline', {})
    samples = build_samples_table(
        root,
        split_path=data_config.get('split_path'),
        split_version=data_config.get('split_version'),
        reference_duration_sec=float(reference_config.get('duration_sec', 0.75)),
        reference_statistic=str(reference_config.get('statistic', 'median')),
        reference_source=str(reference_config.get('source', 'raw_tactile')),
        w2a_force_baseline_mode=str(force_baseline_config.get('W2A', 'pre_contact_median')),
        a2w_force_baseline_mode=str(force_baseline_config.get('A2W', 'none')),
    )
    policy_stride_sec = float(data_config.get('policy_stride_sec', data_config.get('window_stride_sec', 0.25)))
    policy_rate_hz = float(data_config.get('policy_rate_hz', 1.0 / policy_stride_sec))
    policy_timestamp_anchor = str(data_config.get('policy_timestamp_anchor', 'window_center'))
    causal_only = bool(data_config.get('causal_only', False))
    phase_label_mode = str(data_config.get('phase_label_mode', 'window_overlap'))
    if policy_stride_sec <= 0.0 or policy_rate_hz <= 0.0:
        raise ValueError('policy_stride_sec and policy_rate_hz must be positive.')
    if not np.isclose(policy_rate_hz * policy_stride_sec, 1.0, atol=1e-6):
        raise ValueError('policy_rate_hz and policy_stride_sec are inconsistent.')
    if causal_only and policy_timestamp_anchor.strip().lower() != 'window_end':
        raise ValueError("causal-v2 preprocessing requires policy_timestamp_anchor='window_end'.")
    tactile_context_sec = float(data_config.get('tactile_context_sec', data_config.get('window_size_sec', 1.0)))
    visual_context_sec = float(data_config.get('visual_context_sec', data_config.get('window_size_sec', 1.0)))
    if causal_only and (
        not np.isclose(tactile_context_sec, float(data_config.get('window_size_sec', 1.0)))
        or not np.isclose(visual_context_sec, float(data_config.get('window_size_sec', 1.0)))
    ):
        raise ValueError('Current causal-v2 builder requires tactile/visual context to equal window_size_sec.')
    try:
        processed_relative = processed_root.relative_to(root / 'data')
        sidecar_cache_relative_root = processed_relative / 'cache'
    except ValueError as exc:
        raise ValueError('processed_dir must be inside the project data directory so sidecars remain portable.') from exc
    windows = build_windows_table(
        samples=samples,
        project_root=root,
        video_fps=float(data_config.get('video_fps', 30.0)),
        tactile_dt=float(data_config.get('tactile_dt', 0.039)),
        window_size_sec=float(data_config.get('window_size_sec', 1.0)),
        window_stride_sec=policy_stride_sec,
        interface_overlap_sec=float(data_config.get('interface_overlap_sec', 0.25)),
        stable_margin_sec=float(data_config.get('stable_margin_sec', 0.25)),
        short_tail_min_sec=float(data_config.get('short_tail_min_sec', 0.5)),
        num_frames_per_window=int(data_config.get('num_frames_per_window')) if data_config.get('num_frames_per_window') is not None else None,
        tactile_points_per_window=int(data_config.get('tactile_points_per_window')) if data_config.get('tactile_points_per_window') is not None else None,
        write_sidecar_cache=bool(data_config.get('write_sidecar_cache', True)),
        policy_timestamp_anchor=policy_timestamp_anchor,
        causal_only=causal_only,
        sidecar_cache_relative_root=sidecar_cache_relative_root,
        target_aggregation_sec=data_config.get('target', {}).get('aggregation_sec'),
        reference_duration_sec=reference_config.get('duration_sec'),
        phase_label_mode=phase_label_mode,
    )
    samples_output_path = processed_root / 'samples.csv'
    windows_output_path = processed_root / 'windows.csv'
    samples.to_csv(samples_output_path, index=False, encoding='utf-8-sig')
    windows.to_csv(windows_output_path, index=False, encoding='utf-8-sig')

    summary = {
        'num_samples': int(len(samples)),
        'num_windows': int(len(windows)),
        'phase_counts': windows['phase_label'].value_counts().to_dict(),
        'trial_result_counts': samples['trial_result'].value_counts().to_dict(),
        'object_pool_counts': samples['object_pool'].value_counts().to_dict(),
        'direction_counts': samples['direction'].value_counts().to_dict(),
        'split_counts': samples['split'].value_counts().to_dict(),
        'direction_trial_result_counts': (
            samples.groupby(['split', 'direction', 'trial_result']).size().reset_index(name='count').to_dict('records')
        ),
        'window_direction_phase_counts': (
            windows.groupby(['split', 'direction', 'phase_label']).size().reset_index(name='count').to_dict('records')
        ),
        'reference_eligibility_counts': (
            samples.groupby(['direction', 'has_reference_candidate']).size().reset_index(name='count').to_dict('records')
        ),
    }
    with (stats_root / 'dataset_summary.json').open('w', encoding='utf-8') as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    manifest = {
        'schema_version': str(data_config.get('schema_version', 'causal-v2' if causal_only else 'legacy-v1')),
        'dataset_version': str(data_config.get('dataset_version', 'legacy')),
        'annotations_sha256': _file_sha256(root / 'data' / 'annotations' / 'sample_events.csv'),
        'object_attributes_sha256': _file_sha256(root / 'data' / 'annotations' / 'object_attributes.csv'),
        'config_sha256': _file_sha256(data_config_path),
        'split_sha256': (
            _file_sha256(_resolve_project_path(root, data_config['split_path']))
            if data_config.get('split_path')
            else None
        ),
        'samples_sha256': _file_sha256(samples_output_path),
        'windows_sha256': _file_sha256(windows_output_path),
        'policy_rate_hz': policy_rate_hz,
        'policy_stride_sec': policy_stride_sec,
        'policy_timestamp_anchor': policy_timestamp_anchor,
        'causal_only': causal_only,
        'phase_label_mode': phase_label_mode,
        'phase_label_semantics': (
            'instantaneous phase at policy_timestamp'
            if phase_label_mode.strip().lower() == 'policy_timestamp'
            else 'legacy window-overlap phase'
        ),
        'stable_mask_semantics': 'full input context outside interface plus stable margin',
        'tactile_context_sec': tactile_context_sec,
        'visual_context_sec': visual_context_sec,
        'target': data_config.get('target', {}),
        'reference': reference_config,
        'reference_definition': 'source_medium_pre_interface_fixed_window',
        'force_baseline': force_baseline_config,
        'direction_values': list(DIRECTION_TO_INDEX),
        'direction_to_index': DIRECTION_TO_INDEX,
        'raw_sensor_rates_hz': {
            'video': float(data_config.get('video_fps', 30.0)),
            'tactile': float(1.0 / float(data_config.get('tactile_dt', 0.039))),
        },
        'split_path': data_config.get('split_path'),
        'split_version': data_config.get('split_version'),
        'generator': str(Path(__file__).relative_to(root)) if Path(__file__).is_relative_to(root) else str(Path(__file__)),
        'git_commit': _git_revision(root),
        'generated_at_utc': datetime.now(timezone.utc).isoformat(),
        'generation_command': generation_command or '',
        'num_samples': int(len(samples)),
        'num_windows': int(len(windows)),
        'phase_counts': summary['phase_counts'],
        'direction_counts': summary['direction_counts'],
        'split_counts': summary['split_counts'],
        'direction_trial_result_counts': summary['direction_trial_result_counts'],
        'window_direction_phase_counts': summary['window_direction_phase_counts'],
        'reference_eligibility_counts': summary['reference_eligibility_counts'],
    }
    with (processed_root / 'manifest.yaml').open('w', encoding='utf-8') as handle:
        yaml.safe_dump(manifest, handle, allow_unicode=True, sort_keys=False)
    return samples, windows
