from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


LEGACY_SAMPLE_AUDIT_COLUMNS = {'notes', 'sync_audit_note'}
LEGACY_WINDOW_CONTRACT_COLUMNS = [
    'window_id',
    'sample_id',
    'window_start',
    'window_end',
    'policy_timestamp',
    'phase_label',
    'is_stable_mask',
    'tactile_start_idx',
    'tactile_end_idx',
]


def resolve_path(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def resolve_data_path(root: Path, value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    if path.parts and path.parts[0] == 'data':
        return root / path
    return root / 'data' / path


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def normalized_frame(frame: pd.DataFrame, *, sort_columns: list[str], columns: list[str]) -> pd.DataFrame:
    return frame.loc[:, columns].sort_values(sort_columns).reset_index(drop=True)


def frame_difference_count(left: pd.DataFrame, right: pd.DataFrame) -> int:
    if left.shape != right.shape or list(left.columns) != list(right.columns):
        return max(len(left), len(right))
    differences = 0
    for column in left.columns:
        left_values = left[column]
        right_values = right[column]
        both_missing = left_values.isna() & right_values.isna()
        equal = left_values.eq(right_values) | both_missing
        differences += int((~equal).sum())
    return differences


def add_issue(issues: list[dict[str, str]], field: str, message: str) -> None:
    issues.append({'field': field, 'message': message})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--project-root', default='.')
    parser.add_argument('--formal-dir', default='data/processed/policy_20hz_bidirectional_v1_fixed_test')
    parser.add_argument('--rebuild-dir', default='data/processed/phase_e_bidirectional_rebuild_audit')
    parser.add_argument('--legacy-dir', default='data/processed/policy_20hz_causal_v3')
    parser.add_argument('--split', default='data/splits/split_unseen_fixed_test_obj004_obj007_v1.yaml')
    parser.add_argument(
        '--output',
        default='data/processed/policy_20hz_bidirectional_v1_fixed_test/stats/phase_e_release_validation.json',
    )
    args = parser.parse_args()

    root = Path(args.project_root).resolve()
    formal_dir = resolve_path(root, args.formal_dir)
    rebuild_dir = resolve_path(root, args.rebuild_dir)
    legacy_dir = resolve_path(root, args.legacy_dir)
    split_path = resolve_path(root, args.split)
    issues: list[dict[str, str]] = []

    required_paths = [
        formal_dir / 'samples.csv', formal_dir / 'windows.csv', formal_dir / 'manifest.yaml',
        rebuild_dir / 'samples.csv', rebuild_dir / 'windows.csv', rebuild_dir / 'manifest.yaml',
        legacy_dir / 'samples.csv', legacy_dir / 'windows.csv', legacy_dir / 'manifest.yaml',
        split_path,
    ]
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise FileNotFoundError('Phase E inputs are missing: ' + ', '.join(missing))

    formal_samples = pd.read_csv(formal_dir / 'samples.csv')
    formal_windows = pd.read_csv(formal_dir / 'windows.csv')
    rebuild_samples = pd.read_csv(rebuild_dir / 'samples.csv')
    rebuild_windows = pd.read_csv(rebuild_dir / 'windows.csv')
    legacy_samples = pd.read_csv(legacy_dir / 'samples.csv')
    legacy_windows = pd.read_csv(legacy_dir / 'windows.csv')

    w2a_samples = formal_samples.loc[formal_samples['direction'].eq('W2A')].copy()
    w2a_windows = formal_windows.loc[formal_windows['direction'].eq('W2A')].copy()
    legacy_sample_columns = [
        column for column in legacy_samples.columns
        if column in w2a_samples.columns and column not in LEGACY_SAMPLE_AUDIT_COLUMNS
    ]
    legacy_sample_left = normalized_frame(
        legacy_samples,
        sort_columns=['sample_id'],
        columns=legacy_sample_columns,
    )
    legacy_sample_right = normalized_frame(
        w2a_samples,
        sort_columns=['sample_id'],
        columns=legacy_sample_columns,
    )
    legacy_sample_differences = frame_difference_count(legacy_sample_left, legacy_sample_right)
    if legacy_sample_differences:
        add_issue(issues, 'legacy.samples', f'{legacy_sample_differences} W2A sample values changed.')

    legacy_window_left = normalized_frame(
        legacy_windows,
        sort_columns=['window_id'],
        columns=LEGACY_WINDOW_CONTRACT_COLUMNS,
    )
    legacy_window_right = normalized_frame(
        w2a_windows,
        sort_columns=['window_id'],
        columns=LEGACY_WINDOW_CONTRACT_COLUMNS,
    )
    legacy_window_differences = frame_difference_count(legacy_window_left, legacy_window_right)
    if legacy_window_differences:
        add_issue(issues, 'legacy.windows', f'{legacy_window_differences} W2A window values changed.')

    sample_columns = list(formal_samples.columns)
    rebuild_sample_differences = frame_difference_count(
        normalized_frame(formal_samples, sort_columns=['sample_id'], columns=sample_columns),
        normalized_frame(rebuild_samples, sort_columns=['sample_id'], columns=sample_columns),
    )
    if rebuild_sample_differences:
        add_issue(issues, 'rebuild.samples', f'{rebuild_sample_differences} rebuilt sample values differ.')

    rebuild_window_columns = [column for column in formal_windows.columns if column != 'sidecar_cache_path']
    rebuild_window_differences = frame_difference_count(
        normalized_frame(formal_windows, sort_columns=['window_id'], columns=rebuild_window_columns),
        normalized_frame(rebuild_windows, sort_columns=['window_id'], columns=rebuild_window_columns),
    )
    if rebuild_window_differences:
        add_issue(issues, 'rebuild.windows', f'{rebuild_window_differences} rebuilt window values differ.')

    missing_sidecars = 0
    for relative_path in rebuild_windows['sidecar_cache_path'].astype(str):
        sidecar_path = resolve_data_path(root, relative_path)
        if not sidecar_path.exists():
            missing_sidecars += 1
    if missing_sidecars:
        add_issue(issues, 'rebuild.sidecars', f'{missing_sidecars} rebuilt sidecars are missing.')

    with split_path.open('r', encoding='utf-8') as handle:
        split = yaml.safe_load(handle) or {}
    expected_by_object: dict[str, str] = {}
    for subset in ['train', 'val', 'test']:
        for object_id in split.get(f'{subset}_object_ids', []):
            expected_by_object[str(object_id)] = subset
    actual_object_splits = {
        str(object_id): str(group['split'].iloc[0])
        for object_id, group in formal_samples.groupby('object_id')
    }
    split_mismatches = {
        object_id: {'expected': expected_by_object.get(object_id, 'excluded'), 'actual': actual}
        for object_id, actual in actual_object_splits.items()
        if actual != expected_by_object.get(object_id, 'excluded')
    }
    if split_mismatches:
        add_issue(issues, 'split.object_assignments', json.dumps(split_mismatches, ensure_ascii=False))

    uid_split_counts = formal_samples.groupby('physical_object_uid')['split'].nunique()
    leaking_uids = sorted(uid_split_counts.loc[uid_split_counts.gt(1)].index.astype(str).tolist())
    if leaking_uids:
        add_issue(issues, 'split.physical_object_uid', f'Leakage detected: {leaking_uids}.')
    required_test = sorted(str(value) for value in split.get('required_test_object_ids', []))
    missing_required_test = [value for value in required_test if actual_object_splits.get(value) != 'test']
    if missing_required_test:
        add_issue(issues, 'split.required_test', f'Not in Test: {missing_required_test}.')

    direction_counts_per_object = formal_samples.groupby('object_id')['direction'].nunique()
    incomplete_directions = sorted(direction_counts_per_object.loc[direction_counts_per_object.ne(2)].index.tolist())
    if incomplete_directions:
        add_issue(issues, 'split.direction_coverage', f'Objects without both directions: {incomplete_directions}.')

    window_parent = formal_samples.set_index('sample_id')[['split', 'direction', 'physical_object_uid']]
    for column in ['split', 'direction', 'physical_object_uid']:
        expected = formal_windows['sample_id'].map(window_parent[column])
        mismatch_count = int((formal_windows[column].astype(str) != expected.astype(str)).sum())
        if mismatch_count:
            add_issue(issues, f'windows.{column}', f'{mismatch_count} rows disagree with parent samples.')

    fail_samples = formal_samples['trial_result'].astype(str).str.lower().ne('stable')
    fail_reference_count = int(formal_samples.loc[fail_samples, 'has_reference_candidate'].astype(bool).sum())
    fail_policy_count = int(formal_samples.loc[fail_samples, 'policy_supervision_eligible'].astype(bool).sum())
    if fail_reference_count or fail_policy_count:
        add_issue(
            issues,
            'fail.supervision',
            f'fail_reference={fail_reference_count}, fail_policy={fail_policy_count}.',
        )

    prior_reports = {
        'phase_a': root / 'data/processed/stats/annotation_validation_bidirectional_v1.json',
        'phase_b': formal_dir / 'stats/phase_b_validation.json',
        'phase_c': formal_dir / 'stats/phase_c_dataset_smoke.json',
        'phase_d': formal_dir / 'stats/phase_d_sampler_smoke.json',
        'phase_e_rebuild_dataset': formal_dir / 'stats/phase_e_rebuild_dataset_smoke.json',
    }
    prior_report_status: dict[str, dict[str, Any]] = {}
    for name, path in prior_reports.items():
        with path.open('r', encoding='utf-8') as handle:
            payload = json.load(handle)
        error_count = int(payload.get('error_count', 0))
        warning_count = int(payload.get('warning_count', 0))
        prior_report_status[name] = {
            'path': str(path),
            'sha256': file_sha256(path),
            'error_count': error_count,
            'warning_count': warning_count,
        }
        if error_count or warning_count:
            add_issue(issues, f'reports.{name}', f'error={error_count}, warning={warning_count}.')

    report = {
        'release': 'bidirectional_v1__bidirectional-causal-v4',
        'generated_at_utc': datetime.now(timezone.utc).isoformat(),
        'legacy_regression': {
            'baseline': 'policy_20hz_causal_v3',
            'sample_count': int(len(legacy_samples)),
            'w2a_sample_count': int(len(w2a_samples)),
            'sample_value_difference_count': legacy_sample_differences,
            'audited_annotation_columns_excluded': sorted(LEGACY_SAMPLE_AUDIT_COLUMNS),
            'window_count': int(len(legacy_windows)),
            'w2a_window_count': int(len(w2a_windows)),
            'window_value_difference_count': legacy_window_differences,
            'phase_counts': {
                str(key): int(value) for key, value in w2a_windows['phase_label'].value_counts().to_dict().items()
            },
        },
        'isolated_rebuild': {
            'sample_count': int(len(rebuild_samples)),
            'window_count': int(len(rebuild_windows)),
            'sidecar_count': int(len(rebuild_windows) - missing_sidecars),
            'sample_value_difference_count': rebuild_sample_differences,
            'window_value_difference_count_excluding_sidecar_path': rebuild_window_differences,
            'missing_sidecar_count': missing_sidecars,
        },
        'split_leakage': {
            'required_test_object_ids': required_test,
            'required_test_satisfied': not missing_required_test,
            'physical_uid_leakage_count': len(leaking_uids),
            'object_assignment_mismatch_count': len(split_mismatches),
            'objects_with_both_directions': int(direction_counts_per_object.eq(2).sum()),
        },
        'prior_report_status': prior_report_status,
        'artifact_sha256': {
            'formal_samples': file_sha256(formal_dir / 'samples.csv'),
            'formal_windows': file_sha256(formal_dir / 'windows.csv'),
            'legacy_samples': file_sha256(legacy_dir / 'samples.csv'),
            'legacy_windows': file_sha256(legacy_dir / 'windows.csv'),
            'rebuild_samples': file_sha256(rebuild_dir / 'samples.csv'),
            'rebuild_windows': file_sha256(rebuild_dir / 'windows.csv'),
        },
        'error_count': len(issues),
        'issues': issues,
    }

    output_path = resolve_path(root, args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w', encoding='utf-8') as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(json.dumps({'output_path': str(output_path), **report}, ensure_ascii=False, indent=2))
    if issues:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
