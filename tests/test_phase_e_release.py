from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

from scripts.visualize_a2w_phase_reference_spotcheck import compressed_phases


ROOT = Path(__file__).resolve().parents[1]
FORMAL = ROOT / 'data' / 'processed' / 'policy_20hz_bidirectional_v1_fixed_test'


def _json(path: Path) -> dict:
    with path.open('r', encoding='utf-8') as handle:
        return json.load(handle)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def test_compressed_phases_preserves_transition_order() -> None:
    assert compressed_phases(['Air', 'Air', 'Interface', 'Interface', 'Water']) == [
        'Air', 'Interface', 'Water'
    ]


def test_phase_e_release_validation_is_clean() -> None:
    report = _json(FORMAL / 'stats' / 'phase_e_release_validation.json')
    assert report['error_count'] == 0
    assert report['legacy_regression']['sample_value_difference_count'] == 0
    assert report['legacy_regression']['window_value_difference_count'] == 0
    assert report['isolated_rebuild']['missing_sidecar_count'] == 0
    assert report['split_leakage']['required_test_satisfied']
    assert report['split_leakage']['physical_uid_leakage_count'] == 0


def test_a2w_spotcheck_covers_train_val_test_and_obj007() -> None:
    report = _json(FORMAL / 'stats' / 'phase_e_a2w_phase_reference_spotcheck.json')
    assert report['error_count'] == 0
    assert {sample['split'] for sample in report['samples']} == {'train', 'val', 'test'}
    assert 'OBJ007' in {sample['object_id'] for sample in report['samples']}
    assert all(all(sample['checks'].values()) for sample in report['samples'])


def test_release_manifest_is_frozen_and_checksum_matches() -> None:
    with (FORMAL / 'manifest.yaml').open('r', encoding='utf-8') as handle:
        processed_manifest = yaml.safe_load(handle)
    assert processed_manifest['release_status'] == 'frozen'
    assert processed_manifest['release_phase'] == 'Phase E'

    release_path = FORMAL / 'stats' / 'release_manifest_bidirectional_v1.json'
    checksum_path = release_path.with_suffix('.sha256')
    release = _json(release_path)
    expected_checksum = checksum_path.read_text(encoding='utf-8').split()[0]
    assert release['release_status'] == 'frozen'
    assert release['counts']['samples'] == 564
    assert release['counts']['windows'] == release['counts']['sidecars'] == 273850
    assert not any(release['release_evidence'].values())
    assert _sha256(release_path) == expected_checksum
