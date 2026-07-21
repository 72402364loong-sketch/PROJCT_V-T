from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


def resolve_path(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def json_payload(path: Path) -> dict[str, Any]:
    with path.open('r', encoding='utf-8') as handle:
        return json.load(handle)


def sidecar_index_sha256(windows_path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    count = 0
    for chunk in pd.read_csv(windows_path, usecols=['window_id', 'sidecar_cache_path'], chunksize=50000):
        for window_id, sidecar_path in chunk.itertuples(index=False, name=None):
            digest.update(f'{window_id}|{sidecar_path}\n'.encode('utf-8'))
            count += 1
    return digest.hexdigest(), count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--project-root', default='.')
    parser.add_argument('--formal-dir', default='data/processed/policy_20hz_bidirectional_v1_fixed_test')
    parser.add_argument(
        '--output',
        default='data/processed/policy_20hz_bidirectional_v1_fixed_test/stats/release_manifest_bidirectional_v1.json',
    )
    args = parser.parse_args()

    root = Path(args.project_root).resolve()
    formal_dir = resolve_path(root, args.formal_dir)
    manifest_path = formal_dir / 'manifest.yaml'
    with manifest_path.open('r', encoding='utf-8') as handle:
        processed_manifest = yaml.safe_load(handle) or {}
    if processed_manifest.get('release_status') != 'frozen':
        raise RuntimeError('Processed manifest must declare release_status=frozen before release freezing.')

    report_paths = {
        'phase_a_annotations': root / 'data/processed/stats/annotation_validation_bidirectional_v1.json',
        'phase_b_preprocess': formal_dir / 'stats/phase_b_validation.json',
        'phase_c_dataset': formal_dir / 'stats/phase_c_dataset_smoke.json',
        'phase_d_sampler': formal_dir / 'stats/phase_d_sampler_smoke.json',
        'phase_d_direction_stats': formal_dir / 'stats/bidirectional_train_statistics_v1.json',
        'phase_e_release': formal_dir / 'stats/phase_e_release_validation.json',
        'phase_e_a2w_spotcheck': formal_dir / 'stats/phase_e_a2w_phase_reference_spotcheck.json',
        'phase_e_rebuild_dataset': formal_dir / 'stats/phase_e_rebuild_dataset_smoke.json',
    }
    report_status: dict[str, dict[str, Any]] = {}
    for name, path in report_paths.items():
        payload = json_payload(path)
        error_count = int(payload.get('error_count', 0))
        warning_count = int(payload.get('warning_count', 0))
        if error_count or warning_count:
            raise RuntimeError(f'{name} is not clean: error={error_count}, warning={warning_count}.')
        report_status[name] = {
            'path': str(path.relative_to(root)),
            'sha256': file_sha256(path),
            'error_count': error_count,
            'warning_count': warning_count,
        }

    phase_e = json_payload(report_paths['phase_e_release'])
    phase_c = json_payload(report_paths['phase_c_dataset'])
    phase_d = json_payload(report_paths['phase_d_sampler'])
    tactile_cache_path = Path(phase_d['tactile_stats_cache_path'])
    if not tactile_cache_path.is_absolute():
        tactile_cache_path = root / tactile_cache_path

    artifacts = {
        'processed_manifest': manifest_path,
        'samples': formal_dir / 'samples.csv',
        'windows': formal_dir / 'windows.csv',
        'physical_stats': formal_dir / 'stats/physical_attribute_norm_stats.json',
        'tactile_stats': tactile_cache_path,
        'direction_stats': formal_dir / 'stats/bidirectional_train_statistics_v1.json',
        'a2w_spotcheck_image': formal_dir / 'stats/phase_e_a2w_phase_reference_spotcheck.png',
        'data_config': root / 'configs/data/policy_20hz_bidirectional_v4.yaml',
        'train_config': root / 'configs/train/bidirectional_v1.yaml',
        'split': root / 'data/splits/split_unseen_fixed_test_obj004_obj007_v1.yaml',
        'annotations': root / 'data/annotations/sample_events.csv',
        'object_attributes': root / 'data/annotations/object_attributes.csv',
        'physical_attributes': root / 'data/annotations/object_physical_attributes.csv',
        'legacy_manifest': root / 'data/processed/policy_20hz_causal_v3/manifest.yaml',
        'legacy_samples': root / 'data/processed/policy_20hz_causal_v3/samples.csv',
        'legacy_windows': root / 'data/processed/policy_20hz_causal_v3/windows.csv',
        'upgrade_documentation': root / '双向数据管线改造说明.md',
    }
    source_paths = {
        'annotations': root / 'src/cmg/data/annotations.py',
        'preprocess': root / 'src/cmg/data/preprocess.py',
        'windowing': root / 'src/cmg/data/windowing.py',
        'splits': root / 'src/cmg/data/splits.py',
        'dataset': root / 'src/cmg/data/dataset.py',
        'sampler': root / 'src/cmg/data/sampler.py',
        'train_entrypoint': root / 'scripts/train.py',
        'phase_e_validator': root / 'scripts/validate_phase_e_release.py',
        'phase_e_visualizer': root / 'scripts/visualize_a2w_phase_reference_spotcheck.py',
        'dataset_smoke': root / 'scripts/smoke_bidirectional_dataset.py',
        'release_freezer': root / 'scripts/freeze_bidirectional_release.py',
        'release_verifier': root / 'scripts/verify_frozen_bidirectional_release.py',
    }
    missing = [str(path) for path in [*artifacts.values(), *source_paths.values()] if not path.exists()]
    if missing:
        raise FileNotFoundError('Release artifacts are missing: ' + ', '.join(missing))

    artifact_hashes = {
        name: {
            'path': str(path.relative_to(root)),
            'sha256': file_sha256(path),
            'size_bytes': int(path.stat().st_size),
        }
        for name, path in artifacts.items()
    }
    source_hashes = {
        name: {'path': str(path.relative_to(root)), 'sha256': file_sha256(path)}
        for name, path in source_paths.items()
    }
    with artifacts['split'].open('r', encoding='utf-8') as handle:
        split_config = yaml.safe_load(handle) or {}
    sidecar_hash, sidecar_count = sidecar_index_sha256(formal_dir / 'windows.csv')

    release_manifest = {
        'release_id': 'bidirectional_v1__bidirectional-causal-v4__split_unseen_fixed_test_obj004_obj007_v1',
        'release_status': 'frozen',
        'frozen_at_utc': datetime.now(timezone.utc).isoformat(),
        'dataset_version': processed_manifest['dataset_version'],
        'schema_version': processed_manifest['schema_version'],
        'split_version': processed_manifest['split_version'],
        'contract': {
            'direction_values': ['W2A', 'A2W'],
            'reference_definition': 'source_medium_pre_interface_fixed_window',
            'reference_duration_sec': 0.75,
            'reference_statistic': 'median',
            'policy_allowed_trial_results': ['stable'],
            'reference_allowed_trial_results': ['stable'],
            'a2w_force_baseline': 'none',
            'w2a_force_baseline': 'pre_contact_median',
            'required_test_object_ids': ['OBJ004', 'OBJ007'],
            'train_object_ids': list(split_config['train_object_ids']),
            'val_object_ids': list(split_config['val_object_ids']),
            'test_object_ids': list(split_config['test_object_ids']),
            'excluded_object_ids': [],
        },
        'counts': {
            'samples': 564,
            'windows': 273850,
            'sidecars': sidecar_count,
            'train_stable_samples': int(phase_c['subsets']['train']['sample_count']),
            'val_stable_samples': int(phase_c['subsets']['val']['sample_count']),
            'test_stable_samples': int(phase_c['subsets']['test']['sample_count']),
        },
        'release_evidence': {
            'legacy_w2a_sample_difference_count': int(
                phase_e['legacy_regression']['sample_value_difference_count']
            ),
            'legacy_w2a_window_difference_count': int(
                phase_e['legacy_regression']['window_value_difference_count']
            ),
            'isolated_rebuild_sample_difference_count': int(
                phase_e['isolated_rebuild']['sample_value_difference_count']
            ),
            'isolated_rebuild_window_difference_count': int(
                phase_e['isolated_rebuild']['window_value_difference_count_excluding_sidecar_path']
            ),
            'isolated_rebuild_missing_sidecar_count': int(
                phase_e['isolated_rebuild']['missing_sidecar_count']
            ),
            'physical_uid_leakage_count': int(phase_e['split_leakage']['physical_uid_leakage_count']),
        },
        'sidecar_index': {
            'count': sidecar_count,
            'sha256_of_window_id_and_relative_path': sidecar_hash,
        },
        'artifacts': artifact_hashes,
        'source_code': source_hashes,
        'validation_reports': report_status,
    }

    output_path = resolve_path(root, args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w', encoding='utf-8') as handle:
        json.dump(release_manifest, handle, ensure_ascii=False, indent=2)
    release_sha256 = file_sha256(output_path)
    checksum_path = output_path.with_suffix('.sha256')
    with checksum_path.open('w', encoding='utf-8') as handle:
        handle.write(f'{release_sha256}  {output_path.name}\n')

    print(json.dumps({
        'output_path': str(output_path),
        'checksum_path': str(checksum_path),
        'release_sha256': release_sha256,
        'release_status': 'frozen',
        'counts': release_manifest['counts'],
        'release_evidence': release_manifest['release_evidence'],
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
