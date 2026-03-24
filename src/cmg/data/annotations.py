from __future__ import annotations

from pathlib import Path

import pandas as pd

from cmg.constants import infer_object_pool



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
    frame = pd.read_csv(path)
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
    frame = pd.read_csv(path)
    for missing in ['trial_id', 'notes', 'sync_offset_sec', 'sync_audit_status', 'sync_audit_note']:
        if missing not in frame.columns:
            frame[missing] = pd.NA

    frame['sync_offset_sec'] = pd.to_numeric(frame['sync_offset_sec'], errors='coerce').astype('float64')
    frame['sync_offset_sec'] = frame['sync_offset_sec'].where(frame['sync_offset_sec'].notna(), 0.0).round(3)
    frame['sync_audit_status'] = frame['sync_audit_status'].fillna('default_zero')
    frame['sync_audit_note'] = frame['sync_audit_note'].fillna('')

    trial_id_series = pd.to_numeric(frame['trial_id'], errors='coerce').astype('float64')
    default_trial_ids = (frame.groupby('object_id').cumcount() + 1).astype('float64')
    frame['trial_id'] = trial_id_series.where(trial_id_series.notna(), default_trial_ids).astype(int)

    frame['placement_variant'] = frame['placement_variant'].astype(str).str.strip().str.lower()
    allowed_variants = {'normal', 'rotate', 'inverse'}
    invalid_variants = sorted(set(frame['placement_variant']) - allowed_variants)
    if invalid_variants:
        raise ValueError(f'Unsupported placement_variant values: {invalid_variants}')

    numeric_columns = ['t_start', 't_contact_all', 't_grasp_stable', 't_if_enter', 't_if_exit', 't_end']
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors='coerce').round(3)

    ordered_columns = [
        'sample_id',
        'video_path',
        'tactile_path',
        'object_id',
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
        'sync_offset_sec',
        'sync_audit_status',
        'sync_audit_note',
    ]
    return frame[ordered_columns].sort_values('sample_id').reset_index(drop=True)
