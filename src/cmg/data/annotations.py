from __future__ import annotations

from pathlib import Path

import pandas as pd

from cmg.constants import infer_object_pool

CSV_ENCODINGS = ('utf-8-sig', 'utf-8', 'gb18030')

SAMPLE_EVENT_COLUMNS = (
    'sample_id',
    'video_path',
    'tactile_path',
    'object_id',
    'physical_object_uid',
    'source_medium',
    'target_medium',
    'reference_medium',
    'direction',
    'water_condition',
    'lift_speed',
    'placement_variant',
    'trial_result',
    't_start',
    't_contact_all',
    't_grasp_stable',
    't_if_enter',
    't_if_exit',
    't_end',
    'notes',
    'sync_offset_sec',
    'sync_audit_status',
    'sync_audit_note',
)

NORMALIZED_SAMPLE_EVENT_COLUMNS = (
    *SAMPLE_EVENT_COLUMNS[:13],
    'trial_id',
    *SAMPLE_EVENT_COLUMNS[13:],
)


def read_csv_with_fallback(path: str | Path) -> pd.DataFrame:
    last_error: UnicodeDecodeError | None = None
    for encoding in CSV_ENCODINGS:
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError as exc:
            last_error = exc
    if last_error is not None:
        raise last_error
    return pd.read_csv(path)


def _require_columns(frame: pd.DataFrame, columns: tuple[str, ...], *, table: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f'{table} is missing required columns: {missing}')


def _normalize_string_column(series: pd.Series, *, case: str | None = None) -> pd.Series:
    normalized = series.astype('string').str.strip()
    if case == 'lower':
        normalized = normalized.str.lower()
    elif case == 'upper':
        normalized = normalized.str.upper()
    return normalized



def _normalize_category(value: str | float | None) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip().lower()
    return (
        text.replace('_', '-')
        .replace('bowl like', 'bowl-like')
        .replace('cup like', 'cup-like')
        .replace('constricted opening', 'constricted-opening')
    )



def normalize_object_attributes(path: str | Path) -> pd.DataFrame:
    frame = read_csv_with_fallback(path)
    rename_map: dict[str, str] = {}
    for column in frame.columns:
        text = str(column).strip()
        if not text or text.lower().startswith('unnamed'):
            rename_map[column] = 'object_alias'
    frame = frame.rename(columns=rename_map)
    if 'object_alias' not in frame.columns:
        frame['object_alias'] = pd.NA
    if 'object_pool' not in frame.columns:
        frame['object_pool'] = frame['object_id'].map(infer_object_pool)
    frame['object_pool'] = frame['object_pool'].fillna(frame['object_id'].map(infer_object_pool))
    for column in ['fragility', 'geometry', 'surface']:
        frame[column] = frame[column].map(_normalize_category)
    if 'notes' not in frame.columns:
        frame['notes'] = pd.NA
    frame['object_alias'] = frame['object_alias'].fillna(frame['object_name'])
    frame = frame.rename(columns={'notes': 'object_notes'})
    ordered_columns = [
        'object_id',
        'object_name',
        'object_alias',
        'object_pool',
        'fragility',
        'geometry',
        'surface',
        'object_notes',
    ]
    return frame[ordered_columns].sort_values('object_id').reset_index(drop=True)



def normalize_sample_events(path: str | Path) -> pd.DataFrame:
    frame = read_csv_with_fallback(path)
    _require_columns(frame, SAMPLE_EVENT_COLUMNS, table='sample_events')
    frame = frame.copy()

    frame['sync_offset_sec'] = pd.to_numeric(frame['sync_offset_sec'], errors='coerce').astype('float64')
    frame['sync_offset_sec'] = frame['sync_offset_sec'].where(frame['sync_offset_sec'].notna(), 0.0).round(3)
    frame['sync_audit_status'] = _normalize_string_column(frame['sync_audit_status'], case='lower')
    frame['sync_audit_status'] = frame['sync_audit_status'].fillna('verified_zero')
    frame['sync_audit_note'] = frame['sync_audit_note'].fillna('')

    frame['sample_id'] = _normalize_string_column(frame['sample_id'], case='upper')
    frame['object_id'] = _normalize_string_column(frame['object_id'], case='upper')
    frame['physical_object_uid'] = _normalize_string_column(frame['physical_object_uid'])
    for column in ['video_path', 'tactile_path']:
        frame[column] = _normalize_string_column(frame[column])
    for column in [
        'source_medium',
        'target_medium',
        'reference_medium',
        'water_condition',
        'lift_speed',
        'placement_variant',
        'trial_result',
    ]:
        frame[column] = _normalize_string_column(frame[column], case='lower')
    frame['direction'] = _normalize_string_column(frame['direction'], case='upper')

    frame['trial_id'] = (frame.groupby(['object_id', 'direction']).cumcount() + 1).astype(int)

    numeric_columns = ['t_start', 't_contact_all', 't_grasp_stable', 't_if_enter', 't_if_exit', 't_end']
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors='coerce').round(3)

    return frame[list(NORMALIZED_SAMPLE_EVENT_COLUMNS)].sort_values('sample_id').reset_index(drop=True)

