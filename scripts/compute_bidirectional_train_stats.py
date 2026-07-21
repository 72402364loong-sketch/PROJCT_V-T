from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cmg.data.tactile import (
    build_tactile_time_axis,
    compute_clean_force_curve,
    load_tactile_array,
    normalize_normal_sign_table,
    select_tactile_channels,
    split_ac_dc,
    tactile_channel_indices_for_axes,
)


class RunningVectorStats:
    def __init__(self, width: int) -> None:
        self.width = int(width)
        self.count = 0
        self.total = np.zeros(self.width, dtype=np.float64)
        self.total_sq = np.zeros(self.width, dtype=np.float64)
        self.minimum = np.full(self.width, np.inf, dtype=np.float64)
        self.maximum = np.full(self.width, -np.inf, dtype=np.float64)

    def update(self, values: np.ndarray) -> None:
        array = np.asarray(values, dtype=np.float64).reshape(-1, self.width)
        valid_rows = np.all(np.isfinite(array), axis=1)
        array = array[valid_rows]
        if not len(array):
            return
        self.count += int(len(array))
        self.total += array.sum(axis=0)
        self.total_sq += np.square(array).sum(axis=0)
        self.minimum = np.minimum(self.minimum, array.min(axis=0))
        self.maximum = np.maximum(self.maximum, array.max(axis=0))

    def finalize(self) -> dict[str, Any]:
        if self.count == 0:
            return {
                'count': 0,
                'mean': [0.0] * self.width,
                'std': [1.0] * self.width,
                'min': [0.0] * self.width,
                'max': [0.0] * self.width,
            }
        mean = self.total / self.count
        variance = np.maximum(self.total_sq / self.count - np.square(mean), 0.0)
        return {
            'count': self.count,
            'mean': mean.astype(np.float32).tolist(),
            'std': np.sqrt(variance).astype(np.float32).tolist(),
            'min': self.minimum.astype(np.float32).tolist(),
            'max': self.maximum.astype(np.float32).tolist(),
        }


def resolve_project_path(root: Path, raw_path: str | Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    if path.parts and path.parts[0] == 'data':
        return root / path
    return root / 'data' / path


def compute_target_vector(
    force_curve: np.ndarray,
    time_axis: np.ndarray,
    policy_timestamp: float,
    *,
    aggregation_sec: float,
    min_samples: int,
) -> np.ndarray:
    mask = (time_axis >= policy_timestamp - aggregation_sec) & (time_axis <= policy_timestamp)
    selected = force_curve[mask]
    if len(selected) >= min_samples:
        return np.median(selected, axis=0).astype(np.float32)
    causal_indices = np.flatnonzero(time_axis <= policy_timestamp)
    if causal_indices.size:
        return np.asarray(force_curve[int(causal_indices[-1])], dtype=np.float32)
    return np.full((force_curve.shape[1],), float('nan'), dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--project-root', default='.')
    parser.add_argument('--config', default='configs/data/policy_20hz_bidirectional_v4.yaml')
    parser.add_argument(
        '--output',
        default='data/processed/policy_20hz_bidirectional_v1_fixed_test/stats/bidirectional_train_statistics_v1.json',
    )
    args = parser.parse_args()

    root = Path(args.project_root).resolve()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = root / config_path
    with config_path.open('r', encoding='utf-8') as handle:
        config = yaml.safe_load(handle) or {}
    manifest_path = resolve_project_path(root, config['processed_dir']) / 'manifest.yaml'
    with manifest_path.open('r', encoding='utf-8') as handle:
        manifest = yaml.safe_load(handle) or {}

    samples = pd.read_csv(resolve_project_path(root, config['samples_path']))
    train = samples.loc[samples['split'].eq('train') & samples['trial_result'].eq('stable')].copy()
    train_ids = set(train['sample_id'].astype(str))
    window_chunks: list[pd.DataFrame] = []
    for chunk in pd.read_csv(
        resolve_project_path(root, config['windows_path']),
        usecols=['sample_id', 'phase_label', 'policy_timestamp'],
        chunksize=50000,
    ):
        selected = chunk.loc[chunk['sample_id'].astype(str).isin(train_ids) & chunk['phase_label'].eq('Interface')]
        if not selected.empty:
            window_chunks.append(selected)
    interface_windows = pd.concat(window_chunks, ignore_index=True)
    interface_times_by_sample = {
        str(sample_id): group['policy_timestamp'].astype(float).to_numpy()
        for sample_id, group in interface_windows.groupby('sample_id')
    }

    axes = tuple(str(axis).strip().lower() for axis in config.get('tactile_input_axes', ['x', 'y', 'z']))
    channel_indices = tactile_channel_indices_for_axes(axes)
    tactile_width = int(len(channel_indices))
    sign_table = normalize_normal_sign_table(config['normal_sign_table'])
    alpha = float(config.get('ema_alpha_acdc', 0.1))
    tactile_dt = float(config['tactile_dt'])
    target_config = config['target']

    tactile_stats = {
        scope: {
            'high': RunningVectorStats(tactile_width),
            'low': RunningVectorStats(tactile_width),
        }
        for scope in ['all', 'W2A', 'A2W']
    }
    reference_stats = {scope: RunningVectorStats(3) for scope in ['all', 'W2A', 'A2W']}
    target_stats = {scope: RunningVectorStats(3) for scope in ['all', 'W2A', 'A2W']}
    delta_stats = {scope: RunningVectorStats(3) for scope in ['all', 'W2A', 'A2W']}

    for sample in train.to_dict('records'):
        direction = str(sample['direction'])
        tactile_array = load_tactile_array(resolve_project_path(root, sample['tactile_path']))
        high, low = split_ac_dc(tactile_array, alpha=alpha)
        high = select_tactile_channels(high, channel_indices)
        low = select_tactile_channels(low, channel_indices)
        for scope in ['all', direction]:
            tactile_stats[scope]['high'].update(high)
            tactile_stats[scope]['low'].update(low)

        baseline_mode = str(sample['force_baseline_mode'])
        contact_time = None if pd.isna(sample['t_contact_all']) else float(sample['t_contact_all'])
        force_curve = compute_clean_force_curve(
            tactile_array,
            dt=tactile_dt,
            sync_offset_sec=float(sample['sync_offset_sec']),
            contact_time=contact_time,
            alpha=float(config.get('ema_alpha_expert', 0.1)),
            normal_sign_table=sign_table,
            smoothing_mode='none',
            baseline_mode=baseline_mode,
            baseline_window_sec=float(config.get('expert_force_baseline_window_sec', 0.5)),
            force_target_mode='per_finger_z_abs_mean',
        )
        time_axis = build_tactile_time_axis(
            len(force_curve),
            dt=tactile_dt,
            offset_sec=float(sample['sync_offset_sec']),
        )
        reference_mask = (
            (time_axis >= float(sample['reference_start_time']))
            & (time_axis < float(sample['reference_end_time']))
        )
        if not reference_mask.any():
            raise RuntimeError(f'{sample["sample_id"]} has no tactile points in its reference interval.')
        reference = np.median(force_curve[reference_mask], axis=0).astype(np.float32)
        for scope in ['all', direction]:
            reference_stats[scope].update(reference)

        policy_times = interface_times_by_sample.get(str(sample['sample_id']), np.zeros(0, dtype=np.float32))
        targets = np.stack([
            compute_target_vector(
                force_curve,
                time_axis,
                float(timestamp),
                aggregation_sec=float(target_config['aggregation_sec']),
                min_samples=int(target_config['min_samples']),
            )
            for timestamp in policy_times
        ], axis=0) if len(policy_times) else np.zeros((0, 3), dtype=np.float32)
        deltas = targets - reference.reshape(1, -1)
        for scope in ['all', direction]:
            target_stats[scope].update(targets)
            delta_stats[scope].update(deltas)

    finalized_reference = {scope: stats.finalize() for scope, stats in reference_stats.items()}
    shortcut_gap: dict[str, list[float]] = {}
    w2a_mean = np.asarray(finalized_reference['W2A']['mean'], dtype=np.float64)
    a2w_mean = np.asarray(finalized_reference['A2W']['mean'], dtype=np.float64)
    pooled_std = np.sqrt(
        0.5 * np.square(finalized_reference['W2A']['std'])
        + 0.5 * np.square(finalized_reference['A2W']['std'])
    )
    shortcut_gap['reference_mean_gap_a2w_minus_w2a'] = (a2w_mean - w2a_mean).astype(np.float32).tolist()
    shortcut_gap['reference_standardized_gap'] = (
        (a2w_mean - w2a_mean) / np.maximum(pooled_std, 1e-6)
    ).astype(np.float32).tolist()

    report = {
        'dataset_version': config['dataset_version'],
        'schema_version': config['schema_version'],
        'split_version': config['split_version'],
        'annotations_sha256': manifest.get('annotations_sha256'),
        'samples_sha256': manifest.get('samples_sha256'),
        'generated_at_utc': datetime.now(timezone.utc).isoformat(),
        'normalization_scope': {'split': 'train', 'trial_results': ['stable']},
        'sample_count': int(len(train)),
        'direction_sample_counts': {
            str(key): int(value) for key, value in train['direction'].value_counts().to_dict().items()
        },
        'object_ids': sorted(train['object_id'].astype(str).unique().tolist()),
        'tactile_input_axes': list(axes),
        'tactile': {
            scope: {kind: stats.finalize() for kind, stats in blocks.items()}
            for scope, blocks in tactile_stats.items()
        },
        'reference_force_per_finger': finalized_reference,
        'interface_target_force_per_finger': {
            scope: stats.finalize() for scope, stats in target_stats.items()
        },
        'interface_delta_force_per_finger': {
            scope: stats.finalize() for scope, stats in delta_stats.items()
        },
        'direction_shortcut_diagnostics': shortcut_gap,
    }

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = root / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w', encoding='utf-8') as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(json.dumps({
        'output_path': str(output_path),
        'sample_count': report['sample_count'],
        'direction_sample_counts': report['direction_sample_counts'],
        'reference_standardized_gap': shortcut_gap['reference_standardized_gap'],
        'interface_delta_counts': {
            scope: report['interface_delta_force_per_finger'][scope]['count']
            for scope in ['W2A', 'A2W']
        },
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
