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

from cmg.constants import DIRECTION_TO_INDEX, DIRECTION_TO_MEDIA
from cmg.data.sidecar import load_window_sidecar, resolve_window_sidecar_path


SAMPLE_REQUIRED_COLUMNS = {
    'sample_id', 'object_id', 'physical_object_uid', 'direction', 'direction_index',
    'source_medium', 'target_medium', 'reference_medium', 'split', 'split_version',
    'trial_result', 'reference_start_time', 'reference_end_time',
    'has_reference_candidate', 'policy_supervision_eligible', 'force_baseline_mode',
}
WINDOW_REQUIRED_COLUMNS = {
    'window_id', 'sample_id', 'physical_object_uid', 'direction', 'direction_index',
    'source_medium', 'target_medium', 'reference_medium', 'source_phase', 'target_phase',
    'split', 'split_version', 'phase_label', 'stable_phase',
    'reference_interval_start', 'reference_interval_end', 'reference_eligible',
    'policy_supervision_eligible', 'sidecar_cache_path',
}
SIDECAR_REQUIRED_KEYS = {
    'direction', 'direction_index', 'source_medium', 'target_medium', 'reference_medium',
    'split', 'split_version', 'physical_object_uid', 'reference_eligible',
    'policy_supervision_eligible', 'phase_label', 'reference_interval_start',
    'reference_interval_end', 'policy_timestamp', 'video_frame_times_all',
    'tactile_raw_times',
}


def add_issue(issues: list[dict[str, str]], field: str, message: str) -> None:
    issues.append({'field': field, 'message': message})


def resolve_project_path(root: Path, raw_path: str | Path) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else root / path


def validate_outputs(project_root: Path, config_path: Path) -> dict[str, Any]:
    with config_path.open('r', encoding='utf-8') as handle:
        config = yaml.safe_load(handle) or {}
    processed_root = resolve_project_path(project_root, config['processed_dir'])
    samples = pd.read_csv(processed_root / 'samples.csv')
    windows = pd.read_csv(processed_root / 'windows.csv')
    with (processed_root / 'manifest.yaml').open('r', encoding='utf-8') as handle:
        manifest = yaml.safe_load(handle) or {}

    issues: list[dict[str, str]] = []
    missing_sample_columns = sorted(SAMPLE_REQUIRED_COLUMNS - set(samples.columns))
    missing_window_columns = sorted(WINDOW_REQUIRED_COLUMNS - set(windows.columns))
    if missing_sample_columns:
        add_issue(issues, 'samples.schema', f'Missing columns: {missing_sample_columns}')
    if missing_window_columns:
        add_issue(issues, 'windows.schema', f'Missing columns: {missing_window_columns}')
    if issues:
        return {'error_count': len(issues), 'issues': issues}

    expected_split_version = str(config['split_version'])
    if set(samples['split_version'].astype(str)) != {expected_split_version}:
        add_issue(issues, 'samples.split_version', 'Samples contain an unexpected split version.')
    if set(windows['split_version'].astype(str)) != {expected_split_version}:
        add_issue(issues, 'windows.split_version', 'Windows contain an unexpected split version.')
    if set(samples['direction']) != set(DIRECTION_TO_INDEX):
        add_issue(issues, 'samples.direction', f'Unexpected direction values: {sorted(set(samples["direction"]))}')

    for direction, media in DIRECTION_TO_MEDIA.items():
        mask = samples['direction'].eq(direction)
        if not samples.loc[mask, 'direction_index'].eq(DIRECTION_TO_INDEX[direction]).all():
            add_issue(issues, 'samples.direction_index', f'{direction} has an invalid direction_index.')
        for field, expected in media.items():
            if not samples.loc[mask, field].eq(expected).all():
                add_issue(issues, f'samples.{field}', f'{direction} does not consistently use {expected}.')

    leakage = samples.groupby('physical_object_uid')['split'].nunique()
    if (leakage > 1).any():
        add_issue(issues, 'samples.split', f'Physical object leakage: {sorted(leakage.loc[leakage > 1].index.tolist())}')
    fixed_test = samples.loc[samples['object_id'].isin({'OBJ004', 'OBJ007'}), 'split']
    if not fixed_test.eq('test').all():
        add_issue(issues, 'samples.split', 'OBJ004 and OBJ007 must remain in test.')

    fail_samples = samples['trial_result'].ne('stable')
    if samples.loc[fail_samples, 'has_reference_candidate'].astype(bool).any():
        add_issue(issues, 'samples.has_reference_candidate', 'Fail samples must not be reference eligible.')
    if samples.loc[fail_samples, 'policy_supervision_eligible'].astype(bool).any():
        add_issue(issues, 'samples.policy_supervision_eligible', 'Fail samples must not be policy eligible.')
    eligible = samples['has_reference_candidate'].astype(bool)
    reference_duration = samples.loc[eligible, 'reference_end_time'] - samples.loc[eligible, 'reference_start_time']
    if not reference_duration.gt(0).all() or not reference_duration.le(float(config['reference']['duration_sec']) + 1e-6).all():
        add_issue(issues, 'samples.reference_interval', 'Eligible reference intervals must be in (0, configured duration].')

    propagated_columns = [
        'physical_object_uid', 'direction', 'direction_index', 'source_medium', 'target_medium',
        'reference_medium', 'split', 'split_version', 'policy_supervision_eligible',
    ]
    expected = windows[['sample_id', *propagated_columns]].merge(
        samples[['sample_id', *propagated_columns]],
        on='sample_id',
        suffixes=('_window', '_sample'),
        validate='m:1',
    )
    for column in propagated_columns:
        if not expected[f'{column}_window'].astype(str).eq(expected[f'{column}_sample'].astype(str)).all():
            add_issue(issues, f'windows.{column}', 'Window values do not match their parent sample.')

    fail_sample_ids = set(samples.loc[fail_samples, 'sample_id'].astype(str))
    fail_windows = windows['sample_id'].astype(str).isin(fail_sample_ids)
    if windows.loc[fail_windows, 'reference_eligible'].astype(bool).any():
        add_issue(issues, 'windows.reference_eligible', 'Fail windows must not be reference eligible.')
    if windows.loc[fail_windows, 'policy_supervision_eligible'].astype(bool).any():
        add_issue(issues, 'windows.policy_supervision_eligible', 'Fail windows must not be policy eligible.')

    phase_rank = np.full(len(windows), -1, dtype=np.int8)
    phase_orders = {
        'W2A': {'Water': 0, 'Interface': 1, 'Air': 2},
        'A2W': {'Air': 0, 'Interface': 1, 'Water': 2},
    }
    for direction, order in phase_orders.items():
        direction_mask = windows['direction'].eq(direction)
        mapped = windows.loc[direction_mask, 'phase_label'].map(order)
        if mapped.isna().any():
            add_issue(issues, 'windows.phase_label', f'{direction} contains an unsupported phase.')
        phase_rank[np.flatnonzero(direction_mask.to_numpy())] = mapped.fillna(-1).astype(int).to_numpy()
    ordered = windows.assign(_phase_rank=phase_rank).sort_values(['sample_id', 'policy_timestamp'])
    reversals = ordered.groupby('sample_id')['_phase_rank'].diff().lt(0)
    if reversals.any():
        bad_samples = sorted(ordered.loc[reversals, 'sample_id'].astype(str).unique().tolist())
        add_issue(issues, 'windows.phase_order', f'Non-monotonic direction-aware phase order: {bad_samples[:20]}')

    sidecar_paths = windows['sidecar_cache_path'].astype(str)
    missing_sidecars = [
        value for value in sidecar_paths
        if not resolve_window_sidecar_path(project_root, value).exists()
    ]
    if missing_sidecars:
        add_issue(issues, 'sidecar.exists', f'Missing {len(missing_sidecars)} sidecars.')

    sampled_indices = np.linspace(0, max(0, len(windows) - 1), num=min(12, len(windows)), dtype=int)
    for index in sampled_indices:
        window = windows.iloc[int(index)]
        payload = load_window_sidecar(project_root, window['sidecar_cache_path'])
        missing_keys = sorted(SIDECAR_REQUIRED_KEYS - set(payload))
        if missing_keys:
            add_issue(issues, 'sidecar.schema', f'{window["window_id"]} missing keys: {missing_keys}')
            continue
        if payload['direction'].item() != window['direction']:
            add_issue(issues, 'sidecar.direction', f'{window["window_id"]} direction mismatch.')
        policy_timestamp = float(payload['policy_timestamp'].item())
        for time_key in ['video_frame_times_all', 'tactile_raw_times']:
            values = payload[time_key]
            if values.size and float(np.max(values)) > policy_timestamp + 1e-6:
                add_issue(issues, f'sidecar.{time_key}', f'{window["window_id"]} contains future data.')

    if int(manifest.get('num_samples', -1)) != len(samples):
        add_issue(issues, 'manifest.num_samples', 'Manifest sample count does not match samples.csv.')
    if int(manifest.get('num_windows', -1)) != len(windows):
        add_issue(issues, 'manifest.num_windows', 'Manifest window count does not match windows.csv.')
    if manifest.get('direction_to_index') != DIRECTION_TO_INDEX:
        add_issue(issues, 'manifest.direction_to_index', 'Manifest direction mapping is incorrect.')

    return {
        'dataset_version': config.get('dataset_version'),
        'schema_version': config.get('schema_version'),
        'generated_at_utc': datetime.now(timezone.utc).isoformat(),
        'processed_dir': str(processed_root),
        'sample_count': int(len(samples)),
        'window_count': int(len(windows)),
        'sidecar_count': int(len(sidecar_paths) - len(missing_sidecars)),
        'direction_counts': {str(key): int(value) for key, value in samples['direction'].value_counts().to_dict().items()},
        'split_counts': {str(key): int(value) for key, value in samples['split'].value_counts().to_dict().items()},
        'reference_candidate_counts': {
            str(key): int(value)
            for key, value in samples.groupby('direction')['has_reference_candidate'].sum().to_dict().items()
        },
        'error_count': len(issues),
        'issues': issues,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--project-root', default='.')
    parser.add_argument('--config', default='configs/data/policy_20hz_bidirectional_v4.yaml')
    parser.add_argument('--output', default=None)
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    config_path = resolve_project_path(project_root, args.config)
    report = validate_outputs(project_root, config_path)
    output_path = (
        resolve_project_path(project_root, args.output)
        if args.output
        else resolve_project_path(project_root, 'data/processed/policy_20hz_bidirectional_v1_fixed_test/stats/phase_b_validation.json')
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w', encoding='utf-8') as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(json.dumps({'output_path': str(output_path), **report}, ensure_ascii=False, indent=2))
    if report['error_count']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
