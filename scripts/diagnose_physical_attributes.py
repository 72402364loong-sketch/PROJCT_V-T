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

from cmg.data.dataset import PHYSICAL_CATEGORICAL_ATTRIBUTES, PHYSICAL_REQUIRED_COLUMNS
from cmg.data.splits import resolve_sample_ids


def resolve_path(project_root: Path, raw_path: str | Path) -> Path:
    path = Path(raw_path)
    if not path.is_absolute():
        path = project_root / path
    return path


def numeric_summary(values: pd.Series) -> dict[str, Any]:
    values = pd.to_numeric(values, errors='coerce').dropna().astype(float)
    if values.empty:
        return {
            'count': 0,
            'mean': None,
            'std': None,
            'min': None,
            'q25': None,
            'median': None,
            'q75': None,
            'max': None,
        }
    return {
        'count': int(values.shape[0]),
        'mean': float(values.mean()),
        'std': float(values.std(ddof=0)),
        'min': float(values.min()),
        'q25': float(values.quantile(0.25)),
        'median': float(values.quantile(0.50)),
        'q75': float(values.quantile(0.75)),
        'max': float(values.max()),
    }


def value_counts(series: pd.Series) -> dict[str, int]:
    counts = series.fillna('unknown').astype(str).value_counts(dropna=False).to_dict()
    return {str(key): int(value) for key, value in counts.items()}


def infer_transition_counts(windows: pd.DataFrame) -> dict[str, int]:
    if windows.empty:
        return {}
    required = {'sample_id', 'phase_label', 'window_start'}
    missing = required - set(windows.columns)
    if missing:
        raise RuntimeError(f'Window table is missing required columns for transition diagnostics: {sorted(missing)}')
    transitions: list[str] = []
    ordered = windows.sort_values(['sample_id', 'window_start'])
    for _, group in ordered.groupby('sample_id', sort=True):
        phases = group['phase_label'].fillna('').astype(str).tolist()
        stable_media = [phase for phase in phases if phase in ('Water', 'Air')]
        if not stable_media:
            transitions.append('unknown')
            continue
        transitions.append(f'{stable_media[0]}_to_{stable_media[-1]}')
    return {str(key): int(value) for key, value in pd.Series(transitions).value_counts().to_dict().items()}


def window_summary(samples: pd.DataFrame, windows: pd.DataFrame, sample_ids: set[str]) -> dict[str, Any]:
    subset_samples = samples.loc[samples['sample_id'].astype(str).isin(sample_ids)].copy()
    subset_windows = windows.loc[windows['sample_id'].astype(str).isin(sample_ids)].copy()
    stable_mask = subset_windows['is_stable_mask'].astype(str).str.lower().isin({'true', '1', '1.0'})
    interface_count = int((subset_windows['phase_label'].astype(str) == 'Interface').sum())
    return {
        'window_count': int(subset_windows.shape[0]),
        'phase_counts': value_counts(subset_windows['phase_label']),
        'stable_window_count': int(stable_mask.sum()),
        'stable_phase_counts': value_counts(subset_windows.loc[stable_mask, 'stable_phase']),
        'interface_window_count': interface_count,
        'interface_window_ratio': float(interface_count / subset_windows.shape[0]) if subset_windows.shape[0] else None,
        'transition_counts': infer_transition_counts(subset_windows),
        'sample_placement_variant_counts': value_counts(subset_samples['placement_variant']),
        'stable_window_placement_variant_counts': value_counts(subset_windows.loc[stable_mask, 'placement_variant']),
    }


def load_physical_table(path: Path) -> pd.DataFrame:
    table = pd.read_csv(path)
    missing = [column for column in PHYSICAL_REQUIRED_COLUMNS if column not in table.columns]
    if missing:
        raise RuntimeError(f'Physical attribute table is missing required columns: {missing}')
    if table['object_id'].duplicated().any():
        duplicated = sorted(table.loc[table['object_id'].duplicated(), 'object_id'].astype(str).unique().tolist())
        raise RuntimeError(f'Physical attribute table contains duplicated object_id values: {duplicated}')
    table = table.copy()
    table['object_id'] = table['object_id'].astype(str)
    for column in ('dry_mass_g', 'full_water_mass_g', 'water_capacity_g', 'capacity_ratio'):
        table[column] = pd.to_numeric(table[column], errors='coerce')
    table['is_open_container'] = pd.to_numeric(table['is_open_container'], errors='coerce').fillna(0).astype(int)
    table['capacity_valid_mask'] = pd.to_numeric(table['capacity_valid_mask'], errors='coerce').fillna(0).astype(int)
    for column in PHYSICAL_CATEGORICAL_ATTRIBUTES:
        table[column] = table[column].fillna('unknown').astype(str).str.strip()
    return table


def table_summary(table: pd.DataFrame) -> dict[str, Any]:
    valid_capacity = table['capacity_valid_mask'] == 1
    return {
        'object_count': int(table['object_id'].nunique()),
        'open_container_count': int((table['is_open_container'] == 1).sum()),
        'non_open_count': int((table['is_open_container'] == 0).sum()),
        'dry_mass_g': numeric_summary(table['dry_mass_g']),
        'water_capacity_g_valid': numeric_summary(table.loc[valid_capacity, 'water_capacity_g']),
        'capacity_ratio_valid': numeric_summary(table.loc[valid_capacity, 'capacity_ratio']),
        'class_counts': {
            column: table[column].value_counts(dropna=False).to_dict()
            for column in PHYSICAL_CATEGORICAL_ATTRIBUTES
        },
    }


def subset_summary(
    samples: pd.DataFrame,
    windows: pd.DataFrame,
    table: pd.DataFrame,
    split_path: Path,
    subset: str,
) -> dict[str, Any]:
    sample_ids = set(resolve_sample_ids(samples, split_path, subset=subset))
    subset_samples = samples.loc[samples['sample_id'].isin(sample_ids)].copy()
    subset_objects = sorted(subset_samples['object_id'].astype(str).unique().tolist())
    subset_table = table.loc[table['object_id'].isin(subset_objects)].copy()
    missing = sorted(set(subset_objects) - set(table['object_id'].astype(str).tolist()))
    valid_capacity = subset_table['capacity_valid_mask'] == 1
    return {
        'sample_count': int(subset_samples.shape[0]),
        'object_count': int(len(subset_objects)),
        'objects': subset_objects,
        'missing_physical_attribute_objects': missing,
        'open_container_objects': int((subset_table['is_open_container'] == 1).sum()),
        'non_open_objects': int((subset_table['is_open_container'] == 0).sum()),
        'dry_mass_g': numeric_summary(subset_table['dry_mass_g']),
        'water_capacity_g_valid': numeric_summary(subset_table.loc[valid_capacity, 'water_capacity_g']),
        'capacity_ratio_valid': numeric_summary(subset_table.loc[valid_capacity, 'capacity_ratio']),
        'class_counts': {
            column: subset_table[column].value_counts(dropna=False).to_dict()
            for column in PHYSICAL_CATEGORICAL_ATTRIBUTES
        },
        'window_summary': window_summary(samples, windows, sample_ids),
    }


def split_balance_flags(split_summary: dict[str, Any]) -> list[dict[str, Any]]:
    flags: list[dict[str, Any]] = []
    for subset, summary in split_summary.items():
        object_count = int(summary['object_count'])
        if object_count == 0:
            flags.append({'subset': subset, 'severity': 'high', 'message': 'Subset has no objects.'})
            continue
        if int(summary['open_container_objects']) == 0:
            flags.append({'subset': subset, 'severity': 'medium', 'message': 'Subset has no open-container objects.'})
        if int(summary['non_open_objects']) == 0:
            flags.append({'subset': subset, 'severity': 'medium', 'message': 'Subset has no non-open-container objects.'})
        if int(summary['window_summary']['interface_window_count']) < 150:
            flags.append({'subset': subset, 'severity': 'medium', 'message': 'Subset has fewer than 150 Interface windows.'})
        if summary['capacity_ratio_valid']['count'] > 0 and split_summary['train']['capacity_ratio_valid']['count'] > 0:
            train_max = split_summary['train']['capacity_ratio_valid']['max']
            subset_max = summary['capacity_ratio_valid']['max']
            if subset != 'train' and train_max is not None and subset_max is not None and float(subset_max) > float(train_max) * 1.5:
                flags.append(
                    {
                        'subset': subset,
                        'severity': 'high',
                        'message': 'Subset contains capacity_ratio values far above the train range.',
                        'train_max': train_max,
                        'subset_max': subset_max,
                    }
                )
    return flags


def split_integrity(split_summary: dict[str, Any]) -> dict[str, Any]:
    object_sets = {subset: set(summary['objects']) for subset, summary in split_summary.items()}
    return {
        'train_val_object_overlap': sorted(object_sets['train'] & object_sets['val']),
        'train_test_object_overlap': sorted(object_sets['train'] & object_sets['test']),
        'val_test_object_overlap': sorted(object_sets['val'] & object_sets['test']),
    }


def run_diagnostics(project_root: Path, split_path: Path, physical_table_path: Path, norm_stats_path: Path) -> dict[str, Any]:
    samples = pd.read_csv(project_root / 'data' / 'processed' / 'samples.csv')
    samples['sample_id'] = samples['sample_id'].astype(str)
    windows = pd.read_csv(project_root / 'data' / 'processed' / 'windows.csv')
    windows['sample_id'] = windows['sample_id'].astype(str)
    table = load_physical_table(physical_table_path)
    all_sample_objects = set(samples['object_id'].astype(str).unique().tolist())
    covered_objects = set(table['object_id'].astype(str).tolist())
    split_summary = {
        subset: subset_summary(samples, windows, table, split_path, subset)
        for subset in ('train', 'val', 'test')
    }
    return {
        'physical_attribute_table': str(physical_table_path.resolve()),
        'split_path': str(split_path.resolve()),
        'norm_stats_path': str(norm_stats_path.resolve()),
        'norm_stats_exists': bool(norm_stats_path.exists()),
        'table_summary': table_summary(table),
        'processed_sample_object_count': int(len(all_sample_objects)),
        'processed_sample_objects_missing_physical_attributes': sorted(all_sample_objects - covered_objects),
        'physical_attribute_objects_not_in_processed_samples': sorted(covered_objects - all_sample_objects),
        'split_summary': split_summary,
        'split_integrity': split_integrity(split_summary),
        'balance_flags': split_balance_flags(split_summary),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--project-root', default='.')
    parser.add_argument('--split', default='data/splits/split_unseen_fold1_v1.yaml')
    parser.add_argument('--physical-table', default='data/annotations/object_physical_attributes.csv')
    parser.add_argument('--norm-stats', default='data/processed/stats/physical_attribute_norm_stats.json')
    parser.add_argument('--output', default=None)
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    split_path = resolve_path(project_root, args.split)
    physical_table_path = resolve_path(project_root, args.physical_table)
    norm_stats_path = resolve_path(project_root, args.norm_stats)
    payload = run_diagnostics(project_root, split_path, physical_table_path, norm_stats_path)

    if args.output:
        output_path = resolve_path(project_root, args.output)
    else:
        output_path = project_root / 'evals' / 'physical_attributes' / split_path.stem / 'physical_attribute_split_diagnostics.json'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w', encoding='utf-8') as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    print(json.dumps({'output_path': str(output_path), 'balance_flags': payload['balance_flags']}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
