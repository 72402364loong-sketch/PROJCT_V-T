from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cmg.config import deep_update, load_yaml
from cmg.data import CrossMediumSequenceDataset
from cmg.evaluation import resolve_path


def load_dataset_from_stage(
    *,
    project_root: Path,
    stage_path: Path,
    subset: str,
    only_stable: bool,
) -> tuple[CrossMediumSequenceDataset, dict[str, Any]]:
    data_config = load_yaml(project_root / 'configs' / 'data' / 'default.yaml')
    train_config = load_yaml(project_root / 'configs' / 'train' / 'base.yaml')
    stage_config = load_yaml(stage_path)
    data_config = deep_update(data_config, stage_config.get('data', {}))
    train_config = deep_update(train_config, stage_config.get('train', {}))

    allowed_trial_results = ['stable'] if only_stable else train_config.get('default_allowed_trial_results', {}).get(subset)
    dataset = CrossMediumSequenceDataset(
        project_root=project_root,
        split_path=project_root / stage_config['split'],
        subset=subset,
        num_frames_per_window=int(data_config['num_frames_per_window']),
        image_size=int(data_config['image_size']),
        roi=data_config.get('roi'),
        tactile_dt=float(data_config['tactile_dt']),
        acdc_alpha=float(data_config['ema_alpha_acdc']),
        expert_alpha=float(data_config['ema_alpha_expert']),
        normal_sign_table=data_config['normal_sign_table'],
        clip_mean=data_config.get('clip_mean'),
        clip_std=data_config.get('clip_std'),
        tactile_points_per_window=int(data_config.get('tactile_points_per_window')) if data_config.get('tactile_points_per_window') is not None else None,
        standardize_tactile=bool(data_config.get('standardize_tactile', False)),
        min_valid_ratio_video=float(data_config.get('min_valid_ratio_video', 0.0)),
        min_valid_ratio_tactile=float(data_config.get('min_valid_ratio_tactile', 0.0)),
        normalization_subset='train',
        normalization_trial_results=train_config.get('default_allowed_trial_results', {}).get('train'),
        tail_mode=stage_config.get('tail_mode', data_config.get('valid_tail_mode', 'all_valid')),
        allowed_trial_results=allowed_trial_results,
        visual_feature_cache_dir=data_config.get('visual_feature_cache_dir'),
        reference_force_window_count=int(data_config.get('reference_force_window_count', 3)),
        reference_force_statistic=str(data_config.get('reference_force_statistic', 'mean')),
        expert_force_mode=str(data_config.get('expert_force_mode', 'measured_force')),
        expert_force_smoothing=str(data_config.get('expert_force_smoothing', 'ema')),
        expert_force_baseline_mode=str(data_config.get('expert_force_baseline_mode', 'none')),
        expert_force_baseline_window_sec=float(data_config.get('expert_force_baseline_window_sec', 0.5)),
        expert_force_interface_margin_sec=float(data_config.get('expert_force_interface_margin_sec', 0.0)),
    )
    return dataset, stage_config


def summarize_sample_reference_force(dataset: CrossMediumSequenceDataset) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for index, sample in enumerate(dataset.sample_records):
        item = dataset[index]
        sample_id = str(sample['sample_id'])
        windows = dataset.windows_by_sample.get(sample_id, [])
        reference_window_indices = dataset._resolve_reference_window_indices(windows)
        reference_values = np.asarray(
            [
                float(value)
                for value, flag in zip(item['reference_forces'], item['has_reference'])
                if bool(flag) and np.isfinite(value)
            ],
            dtype=np.float32,
        )
        reference_force = float(np.median(reference_values)) if reference_values.size > 0 else float('nan')
        rows.append(
            {
                'sample_id': sample_id,
                'object_id': str(sample['object_id']),
                'object_name': str(sample['object_name']),
                'object_pool': str(sample['object_pool']),
                'fragility': str(sample['fragility']),
                'geometry': str(sample['geometry']),
                'surface': str(sample['surface']),
                'water_condition': str(sample['water_condition']),
                'lift_speed': str(sample['lift_speed']),
                'placement_variant': str(sample['placement_variant']),
                'trial_result': str(sample['trial_result']),
                'interface_window_count': int(sample.get('interface_window_count', 0)),
                'interface_expert_count': int(sample.get('interface_expert_count', 0)),
                'reference_window_count_used': int(len(reference_window_indices)),
                'reference_window_indices': ','.join(str(value) for value in reference_window_indices),
                'reference_force': reference_force,
                'has_reference_sample': int(np.isfinite(reference_force)),
            }
        )
    return pd.DataFrame(rows)


def build_group_summary(frame: pd.DataFrame, group_by: list[str]) -> pd.DataFrame:
    valid = frame.loc[frame['has_reference_sample'] == 1].copy()
    if valid.empty:
        return pd.DataFrame(columns=group_by + ['sample_count', 'object_count', 'reference_force_mean'])

    grouped = (
        valid.groupby(group_by, dropna=False)
        .agg(
            sample_count=('sample_id', 'count'),
            object_count=('object_id', 'nunique'),
            reference_force_mean=('reference_force', 'mean'),
            reference_force_std=('reference_force', 'std'),
            reference_force_median=('reference_force', 'median'),
            reference_force_min=('reference_force', 'min'),
            reference_force_max=('reference_force', 'max'),
        )
        .reset_index()
    )
    grouped['reference_force_std'] = grouped['reference_force_std'].fillna(0.0)
    grouped['reference_force_range'] = grouped['reference_force_max'] - grouped['reference_force_min']
    return grouped.sort_values(group_by).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--project-root', default='.')
    parser.add_argument('--stage', required=True)
    parser.add_argument('--subset', default='train', choices=['train', 'val', 'test'])
    parser.add_argument('--only-stable', action='store_true')
    parser.add_argument('--output-dir', default=None)
    parser.add_argument(
        '--group-by',
        nargs='+',
        default=['object_id', 'water_condition', 'lift_speed', 'placement_variant'],
    )
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    stage_path = resolve_path(project_root, args.stage)
    dataset, stage_config = load_dataset_from_stage(
        project_root=project_root,
        stage_path=stage_path,
        subset=args.subset,
        only_stable=bool(args.only_stable),
    )

    if args.output_dir:
        output_dir = resolve_path(project_root, args.output_dir)
    else:
        suffix = '__stable' if args.only_stable else ''
        output_dir = project_root / 'evals' / 'reference_force' / str(stage_config['name']) / f'{args.subset}{suffix}'
    output_dir.mkdir(parents=True, exist_ok=True)

    sample_frame = summarize_sample_reference_force(dataset)
    group_frame = build_group_summary(sample_frame, list(args.group_by))
    object_frame = build_group_summary(sample_frame, ['object_id'])

    sample_path = output_dir / 'reference_force_samples.csv'
    group_path = output_dir / 'reference_force_group_summary.csv'
    object_path = output_dir / 'reference_force_object_summary.csv'
    summary_path = output_dir / 'reference_force_summary.json'

    sample_frame.to_csv(sample_path, index=False, encoding='utf-8-sig')
    group_frame.to_csv(group_path, index=False, encoding='utf-8-sig')
    object_frame.to_csv(object_path, index=False, encoding='utf-8-sig')

    summary = {
        'stage_name': str(stage_config['name']),
        'subset': args.subset,
        'only_stable': bool(args.only_stable),
        'sample_count': int(len(sample_frame)),
        'reference_sample_count': int(sample_frame['has_reference_sample'].sum()) if not sample_frame.empty else 0,
        'group_by': list(args.group_by),
        'sample_path': str(sample_path),
        'group_path': str(group_path),
        'object_path': str(object_path),
    }
    with summary_path.open('w', encoding='utf-8') as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    print(
        {
            'output_dir': str(output_dir),
            'sample_path': str(sample_path),
            'group_path': str(group_path),
            'object_path': str(object_path),
            'summary_path': str(summary_path),
            'reference_sample_count': summary['reference_sample_count'],
        }
    )


if __name__ == '__main__':
    main()
