from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cmg.constants import (
    DIRECTION_TO_MEDIA,
    FRAGILITY_TO_INDEX,
    GEOMETRY_TO_INDEX,
    SURFACE_TO_INDEX,
)
from cmg.data.annotations import SAMPLE_EVENT_COLUMNS, normalize_sample_events, read_csv_with_fallback
from cmg.data.video import get_video_metadata

ALLOWED_TRIAL_RESULTS = {'stable', 'fail'}
ALLOWED_WATER_CONDITIONS = {'clear', 'turbid'}
ALLOWED_LIFT_SPEEDS = {'normal', 'fast'}
ALLOWED_PLACEMENT_VARIANTS = {'normal', 'rotate', 'inverse'}
ALLOWED_SYNC_AUDIT_STATUSES = {'verified_zero', 'manually_corrected'}
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


def validate_sample_event_schema(frame: pd.DataFrame) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    missing = [column for column in SAMPLE_EVENT_COLUMNS if column not in frame.columns]
    extra = [column for column in frame.columns if column not in SAMPLE_EVENT_COLUMNS]
    for column in missing:
        add_issue(
            issues,
            level='error',
            table='sample_events',
            row_id='<schema>',
            field=column,
            message='Required bidirectional schema column is missing.',
        )
    for column in extra:
        add_issue(
            issues,
            level='warning',
            table='sample_events',
            row_id='<schema>',
            field=column,
            message='Column is not part of the frozen 23-column raw annotation schema.',
        )
    return issues


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


def validate_sample_events(
    project_root: Path,
    frame: pd.DataFrame,
    object_ids: set[str],
    *,
    check_files: bool = True,
) -> list[dict[str, Any]]:
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
        physical_object_uid = str(row.get('physical_object_uid', '')).strip()
        direction = str(row.get('direction', '')).strip().upper()
        source_medium = str(row.get('source_medium', '')).strip().lower()
        target_medium = str(row.get('target_medium', '')).strip().lower()
        reference_medium = str(row.get('reference_medium', '')).strip().lower()
        sync_audit_status = str(row.get('sync_audit_status', '')).strip().lower()
        sync_audit_note = str(row.get('sync_audit_note', '')).strip()
        annotation_note = str(row.get('notes', '')).strip().lower()

        if object_id not in object_ids:
            add_issue(issues, level='error', table='sample_events', row_id=sample_id, field='object_id', message=f'Unknown object_id: {object_id!r}.')
        if not physical_object_uid or physical_object_uid.lower() in {'nan', '<na>'}:
            add_issue(issues, level='error', table='sample_events', row_id=sample_id, field='physical_object_uid', message='physical_object_uid is required.')
        if trial_result not in ALLOWED_TRIAL_RESULTS:
            add_issue(issues, level='error', table='sample_events', row_id=sample_id, field='trial_result', message=f'Unsupported trial_result: {trial_result!r}.')
        if water_condition not in ALLOWED_WATER_CONDITIONS:
            add_issue(issues, level='error', table='sample_events', row_id=sample_id, field='water_condition', message=f'Unsupported water_condition: {water_condition!r}.')
        if lift_speed not in ALLOWED_LIFT_SPEEDS:
            add_issue(issues, level='error', table='sample_events', row_id=sample_id, field='lift_speed', message=f'Unsupported lift_speed: {lift_speed!r}.')
        if placement_variant not in ALLOWED_PLACEMENT_VARIANTS:
            add_issue(issues, level='error', table='sample_events', row_id=sample_id, field='placement_variant', message=f'Unsupported placement_variant: {placement_variant!r}.')

        if direction not in DIRECTION_TO_MEDIA:
            add_issue(issues, level='error', table='sample_events', row_id=sample_id, field='direction', message=f'Unsupported direction: {direction!r}.')
        else:
            actual_media = {
                'source_medium': source_medium,
                'target_medium': target_medium,
                'reference_medium': reference_medium,
            }
            for field, expected in DIRECTION_TO_MEDIA[direction].items():
                if actual_media[field] != expected:
                    add_issue(
                        issues,
                        level='error',
                        table='sample_events',
                        row_id=sample_id,
                        field=field,
                        message=f'{direction} requires {field}={expected!r}, got {actual_media[field]!r}.',
                    )

        if sync_audit_status not in ALLOWED_SYNC_AUDIT_STATUSES:
            add_issue(
                issues,
                level='error',
                table='sample_events',
                row_id=sample_id,
                field='sync_audit_status',
                message=f'Unsupported sync_audit_status: {sync_audit_status!r}.',
            )

        sync_offset_sec = as_float(row, 'sync_offset_sec')
        if sync_offset_sec is None:
            add_issue(issues, level='error', table='sample_events', row_id=sample_id, field='sync_offset_sec', message='sync_offset_sec is required after normalization.')
        elif sync_audit_status == 'verified_zero' and abs(sync_offset_sec) > TIME_TOLERANCE_SEC:
            add_issue(issues, level='error', table='sample_events', row_id=sample_id, field='sync_offset_sec', message='verified_zero requires sync_offset_sec=0.')
        elif sync_audit_status == 'manually_corrected':
            if abs(sync_offset_sec) <= TIME_TOLERANCE_SEC:
                add_issue(issues, level='error', table='sample_events', row_id=sample_id, field='sync_offset_sec', message='manually_corrected requires a non-zero offset.')
            if not sync_audit_note or sync_audit_note.lower() in {'nan', '<na>'}:
                add_issue(issues, level='warning', table='sample_events', row_id=sample_id, field='sync_audit_note', message='manually_corrected should include an audit note for traceability.')

        if direction == 'A2W':
            if sync_offset_sec is not None and abs(sync_offset_sec) > TIME_TOLERANCE_SEC:
                add_issue(issues, level='error', table='sample_events', row_id=sample_id, field='sync_offset_sec', message='A2W hardware synchronization requires sync_offset_sec=0.')
            if sync_audit_status != 'verified_zero':
                add_issue(issues, level='error', table='sample_events', row_id=sample_id, field='sync_audit_status', message='A2W hardware synchronization requires verified_zero.')

        for path_field in ['video_path', 'tactile_path']:
            raw_path = str(row.get(path_field, '')).strip()
            if not raw_path or raw_path.lower() in {'nan', '<na>'}:
                add_issue(issues, level='error', table='sample_events', row_id=sample_id, field=path_field, message='Path is missing.')
                continue
            if Path(raw_path).is_absolute():
                add_issue(issues, level='warning', table='sample_events', row_id=sample_id, field=path_field, message='Path should be relative, but is absolute.')
            if check_files:
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
        if t_end is None:
            add_issue(issues, level='error', table='sample_events', row_id=sample_id, field='t_end', message='t_end is required.')
        if trial_result == 'stable' and t_if_enter is None:
            add_issue(issues, level='error', table='sample_events', row_id=sample_id, field='t_if_enter', message='Stable trials must provide t_if_enter.')
        if trial_result == 'stable' and t_if_exit is None:
            add_issue(issues, level='error', table='sample_events', row_id=sample_id, field='t_if_exit', message='Stable trials must provide t_if_exit.')

        if direction == 'W2A':
            if t_contact_all is None:
                add_issue(issues, level='error', table='sample_events', row_id=sample_id, field='t_contact_all', message='W2A requires t_contact_all.')
            if t_grasp_stable is None:
                add_issue(issues, level='error', table='sample_events', row_id=sample_id, field='t_grasp_stable', message='W2A requires t_grasp_stable.')
        elif direction == 'A2W':
            if t_contact_all is not None:
                add_issue(issues, level='error', table='sample_events', row_id=sample_id, field='t_contact_all', message='A2W starts with the object already grasped; t_contact_all must be empty.')
            if t_grasp_stable is not None:
                add_issue(issues, level='error', table='sample_events', row_id=sample_id, field='t_grasp_stable', message='A2W starts with the object already grasped; t_grasp_stable must be empty.')

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

        video_path = str(row.get('video_path', '')).strip()
        if check_files and video_path and t_end is not None:
            resolved_video = resolve_data_path(project_root, video_path)
            if resolved_video.exists():
                metadata = get_video_metadata(resolved_video)
                video_duration = float(metadata['duration'])
                if t_end > video_duration + TIME_TOLERANCE_SEC:
                    add_issue(issues, level='error', table='sample_events', row_id=sample_id, field='t_end', message=f't_end={t_end:.3f} exceeds video duration {video_duration:.3f}.')
                if (
                    trial_result == 'fail'
                    and t_end >= video_duration - FAIL_END_MARGIN_SEC
                    and annotation_note != 'failure_time_unresolved_video_end'
                ):
                    add_issue(issues, level='warning', table='sample_events', row_id=sample_id, field='t_end', message=f'Fail trial t_end={t_end:.3f} is very close to video end {video_duration:.3f}; verify it is the failure time, not the video end time.')

    object_uid_counts = frame.groupby('object_id', dropna=False)['physical_object_uid'].nunique(dropna=True)
    for object_id, uid_count in object_uid_counts.items():
        if int(uid_count) != 1:
            add_issue(
                issues,
                level='error',
                table='sample_events',
                row_id=str(object_id),
                field='physical_object_uid',
                message=f'object_id maps to {int(uid_count)} physical_object_uid values; expected exactly one.',
            )

    uid_object_counts = frame.groupby('physical_object_uid', dropna=False)['object_id'].nunique(dropna=True)
    for physical_object_uid, object_count in uid_object_counts.items():
        if int(object_count) != 1:
            add_issue(
                issues,
                level='error',
                table='sample_events',
                row_id=str(physical_object_uid),
                field='object_id',
                message=f'physical_object_uid maps to {int(object_count)} object_id values; expected exactly one.',
            )

    for physical_object_uid, group in frame.groupby('physical_object_uid', dropna=False):
        directions = set(group['direction'].astype(str).str.strip().str.upper())
        if directions != set(DIRECTION_TO_MEDIA):
            add_issue(
                issues,
                level='error',
                table='sample_events',
                row_id=str(physical_object_uid),
                field='direction',
                message=f'Physical object must contain both directions, got {sorted(directions)}.',
            )

    return issues


def _count_records(frame: pd.DataFrame, columns: list[str]) -> list[dict[str, Any]]:
    counts = frame.groupby(columns, dropna=False).size().reset_index(name='count')
    return counts.to_dict('records')


def build_annotation_summary(frame: pd.DataFrame) -> dict[str, Any]:
    missing_event_counts: dict[str, dict[str, int]] = {}
    for direction, group in frame.groupby('direction'):
        missing_event_counts[str(direction)] = {
            field: int(group[field].isna().sum())
            for field in ['t_contact_all', 't_grasp_stable', 't_if_enter', 't_if_exit']
        }

    stable = frame['trial_result'].eq('stable')
    reference_candidate = stable & frame['t_if_enter'].notna()
    return {
        'sample_count': int(len(frame)),
        'object_count': int(frame['object_id'].nunique()),
        'physical_object_count': int(frame['physical_object_uid'].nunique()),
        'direction_counts': {str(key): int(value) for key, value in frame['direction'].value_counts().to_dict().items()},
        'trial_result_counts': {str(key): int(value) for key, value in frame['trial_result'].value_counts().to_dict().items()},
        'direction_trial_result_counts': _count_records(frame, ['direction', 'trial_result']),
        'direction_object_counts': _count_records(frame, ['direction', 'object_id']),
        'direction_media_counts': _count_records(
            frame,
            ['direction', 'source_medium', 'target_medium', 'reference_medium'],
        ),
        'sync_audit_counts': _count_records(frame, ['direction', 'sync_audit_status']),
        'missing_event_counts_by_direction': missing_event_counts,
        'reference_candidate_counts': {
            str(direction): int(reference_candidate.loc[group.index].sum())
            for direction, group in frame.groupby('direction')
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--project-root', default='.')
    parser.add_argument('--output', default='data/processed/stats/annotation_validation_bidirectional_v1.json')
    parser.add_argument('--skip-file-checks', action='store_true')
    parser.add_argument('--strict-warnings', action='store_true')
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    object_path = project_root / 'data' / 'annotations' / 'object_attributes.csv'
    sample_path = project_root / 'data' / 'annotations' / 'sample_events.csv'

    objects_raw = read_csv_with_fallback(object_path)
    samples_raw = read_csv_with_fallback(sample_path)
    object_issues, objects = validate_object_attributes(objects_raw)
    schema_issues = validate_sample_event_schema(samples_raw)
    if any(issue['level'] == 'error' for issue in schema_issues):
        samples = samples_raw
        sample_issues: list[dict[str, Any]] = []
        summary: dict[str, Any] = {'sample_count': int(len(samples_raw))}
    else:
        samples = normalize_sample_events(sample_path)
        sample_issues = validate_sample_events(
            project_root,
            samples,
            set(objects['object_id'].tolist()),
            check_files=not args.skip_file_checks,
        )
        summary = build_annotation_summary(samples)
    all_issues = object_issues + schema_issues + sample_issues
    errors = [issue for issue in all_issues if issue['level'] == 'error']
    warnings = [issue for issue in all_issues if issue['level'] == 'warning']

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = project_root / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        'dataset_version': 'bidirectional_v1',
        'schema_version': 'bidirectional-causal-v4',
        'generated_at_utc': datetime.now(timezone.utc).isoformat(),
        'sample_events_path': str(sample_path),
        'file_checks_enabled': not args.skip_file_checks,
        'object_count': int(len(objects_raw)),
        'sample_count': int(len(samples_raw)),
        'error_count': int(len(errors)),
        'warning_count': int(len(warnings)),
        'summary': summary,
        'issues': all_issues,
    }
    with output_path.open('w', encoding='utf-8') as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)

    print(json.dumps({
        'output_path': str(output_path),
        'error_count': len(errors),
        'warning_count': len(warnings),
        'summary': summary,
    }, ensure_ascii=False, indent=2))
    if errors or (warnings and args.strict_warnings):
        raise SystemExit(1)


if __name__ == '__main__':
    main()

