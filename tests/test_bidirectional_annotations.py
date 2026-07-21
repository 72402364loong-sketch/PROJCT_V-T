from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from cmg.constants import DIRECTION_TO_INDEX, DIRECTION_TO_MEDIA, MEDIUM_TO_PHASE
from cmg.data.annotations import (
    SAMPLE_EVENT_COLUMNS,
    normalize_object_attributes,
    normalize_sample_events,
    read_csv_with_fallback,
)
from scripts.validate_annotations import validate_sample_event_schema, validate_sample_events


ROOT = Path(__file__).resolve().parents[1]
SAMPLE_EVENTS_PATH = ROOT / 'data' / 'annotations' / 'sample_events.csv'


def _valid_bidirectional_pair() -> pd.DataFrame:
    samples = normalize_sample_events(SAMPLE_EVENTS_PATH)
    pair = samples.loc[
        samples['object_id'].eq('OBJ001')
        & samples['trial_result'].eq('stable')
        & samples['sync_audit_status'].eq('verified_zero')
    ].groupby('direction', group_keys=False).head(1)
    assert set(pair['direction']) == {'W2A', 'A2W'}
    return pair.reset_index(drop=True)


def test_direction_and_medium_contract_is_frozen() -> None:
    assert DIRECTION_TO_INDEX == {'W2A': 0, 'A2W': 1}
    assert MEDIUM_TO_PHASE == {'water': 'Water', 'air': 'Air'}
    assert DIRECTION_TO_MEDIA['W2A'] == {
        'source_medium': 'water',
        'target_medium': 'air',
        'reference_medium': 'water',
    }
    assert DIRECTION_TO_MEDIA['A2W'] == {
        'source_medium': 'air',
        'target_medium': 'water',
        'reference_medium': 'air',
    }


def test_formal_event_table_has_frozen_schema_and_preserves_bidirectional_fields() -> None:
    raw = read_csv_with_fallback(SAMPLE_EVENTS_PATH)
    assert tuple(raw.columns) == SAMPLE_EVENT_COLUMNS
    assert validate_sample_event_schema(raw) == []

    normalized = normalize_sample_events(SAMPLE_EVENTS_PATH)
    assert len(normalized) == 564
    assert normalized['direction'].value_counts().to_dict() == {'W2A': 287, 'A2W': 277}
    for column in [
        'physical_object_uid',
        'source_medium',
        'target_medium',
        'reference_medium',
        'direction',
    ]:
        assert column in normalized.columns


def test_blank_sync_fields_normalize_to_verified_zero() -> None:
    raw = read_csv_with_fallback(SAMPLE_EVENTS_PATH).head(1).copy()
    raw.loc[:, 'sync_offset_sec'] = float('nan')
    raw.loc[:, 'sync_audit_status'] = pd.NA
    with TemporaryDirectory() as temp_dir:
        path = Path(temp_dir) / 'sample_events.csv'
        raw.to_csv(path, index=False)
        normalized = normalize_sample_events(path)

    assert float(normalized.loc[0, 'sync_offset_sec']) == 0.0
    assert normalized.loc[0, 'sync_audit_status'] == 'verified_zero'


def test_a2w_missing_contact_labels_are_valid() -> None:
    pair = _valid_bidirectional_pair()
    a2w = pair.loc[pair['direction'].eq('A2W')].iloc[0]
    assert pd.isna(a2w['t_contact_all'])
    assert pd.isna(a2w['t_grasp_stable'])

    issues = validate_sample_events(ROOT, pair, {'OBJ001'}, check_files=False)
    assert [issue for issue in issues if issue['level'] == 'error'] == []


def test_direction_medium_mismatch_is_rejected() -> None:
    pair = _valid_bidirectional_pair()
    pair.loc[pair['direction'].eq('A2W'), 'target_medium'] = 'air'
    issues = validate_sample_events(ROOT, pair, {'OBJ001'}, check_files=False)
    errors = [issue for issue in issues if issue['level'] == 'error']
    assert any(issue['field'] == 'target_medium' and issue['row_id'].startswith('S') for issue in errors)


def test_physical_uid_mismatch_across_directions_is_rejected() -> None:
    pair = _valid_bidirectional_pair()
    pair.loc[pair['direction'].eq('A2W'), 'physical_object_uid'] = 'VT_DIFFERENT_OBJECT'
    issues = validate_sample_events(ROOT, pair, {'OBJ001'}, check_files=False)
    errors = [issue for issue in issues if issue['level'] == 'error']
    assert any(issue['field'] == 'physical_object_uid' for issue in errors)


def test_missing_direction_column_is_a_schema_error() -> None:
    raw = read_csv_with_fallback(SAMPLE_EVENTS_PATH).drop(columns=['direction'])
    issues = validate_sample_event_schema(raw)
    assert any(issue['level'] == 'error' and issue['field'] == 'direction' for issue in issues)


def test_formal_event_table_has_no_contract_issues() -> None:
    samples = normalize_sample_events(SAMPLE_EVENTS_PATH)
    objects = normalize_object_attributes(ROOT / 'data' / 'annotations' / 'object_attributes.csv')
    issues = validate_sample_events(ROOT, samples, set(objects['object_id']), check_files=False)
    assert issues == []
