from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cmg.constants import FRAGILITY_TO_INDEX, GEOMETRY_TO_INDEX, SURFACE_TO_INDEX
from cmg.data.annotations import read_csv_with_fallback
from cmg.data.video import get_video_metadata

ALLOWED_TRIAL_RESULTS = {'stable', 'partial_slip', 'fail'}
ALLOWED_WATER_CONDITIONS = {'clear', 'turbid'}
ALLOWED_LIFT_SPEEDS = {'normal', 'fast'}
ALLOWED_PLACEMENT_VARIANTS = {'normal', 'rotate', 'inverse'}
TIME_TOLERANCE_SEC = 1e-3
FAIL_END_MARGIN_SEC = 0.25


def normalize_category(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip().lower()
    return (
        text.replace('_', '-')
        .replace('bowl like', 'bowl-like')
        .replace('cup like', 'cup-like')
        .replace('constricted opening', 'constricted-opening')
    )


def add_issue(issues: list[dict[str, Any]], *, level: str, table: str, row_id: str, field: str, message: str) -> None:
    issues.append({
        'level': level,
        'table': table,
        'row_id': row_id,
        'field': field,
        'message': message,
    })


def validate_object_attributes(frame: pd.DataFrame) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    issues: list[dict[str, Any]] = []
    frame = frame.copy()
    rename_map: dict[str, str] = {}
    for column in frame.columns:
        text = str(column).strip()
        if not text or text.lower().startswith('unnamed'):
            rename_map[column] = 'object_alias'
    frame = frame.rename(columns=rename_map)
    if frame['object_id'].duplicated().any():
        for object_id in frame.loc[frame['object_id'].duplicated(), 'object_id'].tolist():
            add_issue(issues, level='error', table='object_attributes', row_id=str(object_id), field='object_id', message='Duplicate object_id.')

    for _, row in frame.iterrows():
        object_id = str(row['object_id'])
        fragility = normalize_category(row.get('fragility'))
        geometry = normalize_category(row.get('geometry'))
        surface = normalize_category(row.get('surface'))
        if fragility not in FRAGILITY_TO_INDEX:
            add_issue(issues, level='error', table='object_attributes', row_id=object_id, field='fragility', message=f'Unsupported fragility value: {row.get("fragility")!r}.')
        if geometry not in GEOMETRY_TO_INDEX:
            add_issue(issues, level='error', table='object_attributes', row_id=object_id, field='geometry', message=f'Unsupported geometry value: {row.get("geometry")!r}.')
        if surface not in SURFACE_TO_INDEX:
            add_issue(issues, level='error', table='object_attributes', row_id=object_id, field='surface', message=f'Unsupported surface value: {row.get("surface")!r}.')

    frame['fragility'] = frame['fragility'].map(normalize_category)
    frame['geometry'] = frame['geometry'].map(normalize_category)
    frame['surface'] = frame['surface'].map(normalize_category)
    return issues, frame


def resolve_data_path(project_root: Path, relative_path: str) -> Path:
    path = Path(relative_path)
    if path.is_absolute():
        return path
    if path.parts and path.parts[0] == 'data':
        return project_root / path
    return project_root / 'data' / path


def as_float(row: pd.Series, field: str) -> float | None:
    value = pd.to_numeric(row.get(field), errors='coerce')
    if pd.isna(value):
        return None
    return float(value)


def validate_time_order(issues: list[dict[str, Any]], sample_id: str, named_times: list[tuple[str, float | None]]) -> None:
    previous_name: str | None = None
    previous_value: float | None = None
    for name, value in named_times:
        if value is None:
            continue
        if previous_value is not None and value + TIME_TOLERANCE_SEC < previous_value:
            add_issue(
                issues,
                level='error',
                table='sample_events',
                row_id=sample_id,
                field=name,
                message=f'Time order violated: {name}={value:.3f} < {previous_name}={previous_value:.3f}.',
            )
        previous_name = name
        previous_value = value


def validate_sample_events(project_root: Path, frame: pd.DataFrame, object_ids: set[str]) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    if frame['sample_id'].duplicated().any():
        for sample_id in frame.loc[frame['sample_id'].duplicated(), 'sample_id'].tolist():
            add_issue(issues, level='error', table='sample_events', row_id=str(sample_id), field='sample_id', message='Duplicate sample_id.')

    for _, row in frame.iterrows():
        sample_id = str(row['sample_id'])
        trial_result = str(row.get('trial_result', '')).strip().lower()
        water_condition = str(row.get('water_condition', '')).strip().lower()
        lift_speed = str(row.get('lift_speed', '')).strip().lower()
        placement_variant = str(row.get('placement_variant', '')).strip().lower()
        object_id = str(row.get('object_id', '')).strip()

        if object_id not in object_ids:
            add_issue(issues, level='error', table='sample_events', row_id=sample_id, field='object_id', message=f'Unknown object_id: {object_id!r}.')
        if trial_result not in ALLOWED_TRIAL_RESULTS:
            add_issue(issues, level='error', table='sample_events', row_id=sample_id, field='trial_result', message=f'Unsupported trial_result: {trial_result!r}.')
        if water_condition not in ALLOWED_WATER_CONDITIONS:
            add_issue(issues, level='error', table='sample_events', row_id=sample_id, field='water_condition', message=f'Unsupported water_condition: {water_condition!r}.')
        if lift_speed not in ALLOWED_LIFT_SPEEDS:
            add_issue(issues, level='error', table='sample_events', row_id=sample_id, field='lift_speed', message=f'Unsupported lift_speed: {lift_speed!r}.')
        if placement_variant not in ALLOWED_PLACEMENT_VARIANTS:
            add_issue(issues, level='error', table='sample_events', row_id=sample_id, field='placement_variant', message=f'Unsupported placement_variant: {placement_variant!r}.')

        for path_field in ['video_path', 'tactile_path']:
            raw_path = str(row.get(path_field, '')).strip()
            if not raw_path:
                add_issue(issues, level='error', table='sample_events', row_id=sample_id, field=path_field, message='Path is missing.')
                continue
            if Path(raw_path).is_absolute():
                add_issue(issues, level='warning', table='sample_events', row_id=sample_id, field=path_field, message='Path should be relative, but is absolute.')
            resolved = resolve_data_path(project_root, raw_path)
            if not resolved.exists():
                add_issue(issues, level='error', table='sample_events', row_id=sample_id, field=path_field, message=f'Path does not exist: {resolved}.')

        t_start = as_float(row, 't_start')
        t_contact_all = as_float(row, 't_contact_all')
        t_grasp_stable = as_float(row, 't_grasp_stable')
        t_if_enter = as_float(row, 't_if_enter')
        t_if_exit = as_float(row, 't_if_exit')
        t_end = as_float(row, 't_end')

        if t_start is None:
            add_issue(issues, level='error', table='sample_events', row_id=sample_id, field='t_start', message='t_start is required.')
        elif abs(t_start) > TIME_TOLERANCE_SEC:
            add_issue(issues, level='error', table='sample_events', row_id=sample_id, field='t_start', message=f't_start must be 0, got {t_start:.3f}.')
        if t_if_enter is None:
            add_issue(issues, level='error', table='sample_events', row_id=sample_id, field='t_if_enter', message='t_if_enter is required.')
        if t_end is None:
            add_issue(issues, level='error', table='sample_events', row_id=sample_id, field='t_end', message='t_end is required.')
        if trial_result == 'stable' and t_if_exit is None:
            add_issue(issues, level='error', table='sample_events', row_id=sample_id, field='t_if_exit', message='Stable trials must provide t_if_exit.')

        validate_time_order(
            issues,
            sample_id,
            [
                ('t_start', t_start),
                ('t_contact_all', t_contact_all),
                ('t_grasp_stable', t_grasp_stable),
                ('t_if_enter', t_if_enter),
                ('t_if_exit', t_if_exit),
                ('t_end', t_end),
            ],
        )

        if trial_result == 'stable' and t_if_exit is not None and t_end is not None and t_end <= t_if_exit + TIME_TOLERANCE_SEC:
            add_issue(issues, level='error', table='sample_events', row_id=sample_id, field='t_end', message='Stable trials should end after t_if_exit.')
        if trial_result == 'fail' and t_if_exit is None:
            add_issue(issues, level='warning', table='sample_events', row_id=sample_id, field='t_if_exit', message='Fail trial has no t_if_exit; this is allowed but should be audit-confirmed.')

        video_path = str(row.get('video_path', '')).strip()
        if video_path and t_end is not None:
            resolved_video = resolve_data_path(project_root, video_path)
            if resolved_video.exists():
                metadata = get_video_metadata(resolved_video)
                video_duration = float(metadata['duration'])
                if t_end > video_duration + TIME_TOLERANCE_SEC:
                    add_issue(issues, level='error', table='sample_events', row_id=sample_id, field='t_end', message=f't_end={t_end:.3f} exceeds video duration {video_duration:.3f}.')
                if trial_result == 'fail' and t_end >= video_duration - FAIL_END_MARGIN_SEC:
                    add_issue(issues, level='warning', table='sample_events', row_id=sample_id, field='t_end', message=f'Fail trial t_end={t_end:.3f} is very close to video end {video_duration:.3f}; verify it is the failure time, not the video end time.')

    return issues


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--project-root', default='.')
    parser.add_argument('--output', default='data/processed/stats/annotation_validation.json')
    parser.add_argument('--strict-warnings', action='store_true')
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    object_path = project_root / 'data' / 'annotations' / 'object_attributes.csv'
    sample_path = project_root / 'data' / 'annotations' / 'sample_events.csv'

    objects_raw = read_csv_with_fallback(object_path)
    samples_raw = read_csv_with_fallback(sample_path)
    object_issues, objects = validate_object_attributes(objects_raw)
    sample_issues = validate_sample_events(project_root, samples_raw, set(objects['object_id'].tolist()))
    all_issues = object_issues + sample_issues
    errors = [issue for issue in all_issues if issue['level'] == 'error']
    warnings = [issue for issue in all_issues if issue['level'] == 'warning']

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = project_root / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        'object_count': int(len(objects_raw)),
        'sample_count': int(len(samples_raw)),
        'error_count': int(len(errors)),
        'warning_count': int(len(warnings)),
        'issues': all_issues,
    }
    with output_path.open('w', encoding='utf-8') as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)

    print({
        'output_path': str(output_path),
        'error_count': len(errors),
        'warning_count': len(warnings),
    })
    if errors or (warnings and args.strict_warnings):
        raise SystemExit(1)


if __name__ == '__main__':
    main()

