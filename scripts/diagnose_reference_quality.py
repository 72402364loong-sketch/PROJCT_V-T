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
from cmg.data.tactile import compute_clean_force_curve, compute_expert_force_curve, load_tactile_array
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
        tactile_input_axes=data_config.get('tactile_input_axes'),
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
        soft_gate_pre_sec=float(data_config.get('soft_gate_pre_sec', 0.0)),
        soft_gate_post_sec=float(data_config.get('soft_gate_post_sec', 0.0)),
        soft_gate_ramp=str(data_config.get('soft_gate_ramp', 'linear')),
        interface_context_manifest=data_config.get('interface_context_manifest'),
    )
    return dataset, stage_config


def build_reference_curve(dataset: CrossMediumSequenceDataset, sample: dict[str, Any]) -> np.ndarray | None:
    tactile_array = load_tactile_array(dataset._resolve_data_path(sample['tactile_path']))
    expert_curve = compute_expert_force_curve(
        tactile_array=tactile_array,
        dt=dataset.tactile_dt,
        trial_result=sample['trial_result'],
        contact_time=sample.get('t_contact_all'),
        alpha=dataset.expert_alpha,
        normal_sign_table=dataset.normal_sign_table,
    )
    if expert_curve is None:
        return None
    if dataset.expert_force_mode != 'local_reference_delta':
        return expert_curve

    sync_offset_sec = dataset._parse_optional_float(sample.get('sync_offset_sec')) or 0.0
    contact_time = dataset._parse_optional_float(sample.get('t_contact_all'))
    return compute_clean_force_curve(
        tactile_array,
        dt=dataset.tactile_dt,
        sync_offset_sec=sync_offset_sec,
        contact_time=contact_time,
        alpha=dataset.expert_alpha,
        normal_sign_table=dataset.normal_sign_table,
        smoothing_mode=dataset.expert_force_smoothing,
        baseline_mode=dataset.expert_force_baseline_mode,
        baseline_window_sec=dataset.expert_force_baseline_window_sec,
    )


def summarize_samples(
    dataset: CrossMediumSequenceDataset,
    *,
    std_threshold: float,
    range_threshold: float,
    min_window_count: int,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    bad_cases: list[dict[str, Any]] = []

    for sample in dataset.sample_records:
        sample_id = str(sample['sample_id'])
        windows = dataset.windows_by_sample.get(sample_id, [])
        reference_indices = dataset._resolve_reference_window_indices(windows)
        reference_curve = build_reference_curve(dataset, sample)
        values: list[float] = []
        for window_index in reference_indices:
            window = windows[window_index]
            if reference_curve is None:
                values.append(float('nan'))
                continue
            values.append(
                dataset._compute_window_force_value(
                    reference_curve,
                    int(window['tactile_start_idx']),
                    int(window['tactile_end_idx']),
                )
            )

        finite_values = np.asarray([value for value in values if np.isfinite(value)], dtype=np.float32)
        reference_force, has_reference = dataset._aggregate_reference_force(values)
        reference_std = float(finite_values.std()) if finite_values.size > 1 else 0.0
        reference_min = float(finite_values.min()) if finite_values.size else float('nan')
        reference_max = float(finite_values.max()) if finite_values.size else float('nan')
        reference_range = float(reference_max - reference_min) if finite_values.size else float('nan')
        nonstable_reference_count = sum(
            1
            for window_index in reference_indices
            if window_index < len(windows) and not bool(windows[window_index].get('is_stable_mask', False))
        )
        reasons: list[str] = []
        if not has_reference:
            reasons.append('missing_reference')
        if len(reference_indices) < min_window_count:
            reasons.append('too_few_reference_windows')
        if reference_std >= std_threshold:
            reasons.append('high_reference_std')
        if np.isfinite(reference_range) and reference_range >= range_threshold:
            reasons.append('high_reference_range')
        if nonstable_reference_count > 0:
            reasons.append('nonstable_reference_window')

        row = {
            'sample_id': sample_id,
            'object_id': str(sample['object_id']),
            'object_name': str(sample.get('object_name', '')),
            'object_pool': str(sample.get('object_pool', '')),
            'water_condition': str(sample.get('water_condition', '')),
            'lift_speed': str(sample.get('lift_speed', '')),
            'placement_variant': str(sample.get('placement_variant', '')),
            'trial_result': str(sample.get('trial_result', '')),
            't_grasp_stable': dataset._parse_optional_float(sample.get('t_grasp_stable')),
            't_if_enter': dataset._parse_optional_float(sample.get('t_if_enter')),
            't_if_exit': dataset._parse_optional_float(sample.get('t_if_exit')),
            'reference_force': float(reference_force),
            'has_reference': int(bool(has_reference)),
            'reference_window_count': int(len(reference_indices)),
            'reference_window_indices': json.dumps([int(index) for index in reference_indices]),
            'reference_window_values_json': json.dumps([float(value) for value in values]),
            'reference_force_std': reference_std,
            'reference_force_min': reference_min,
            'reference_force_max': reference_max,
            'reference_force_range': reference_range,
            'nonstable_reference_window_count': int(nonstable_reference_count),
            'bad_reference_reason': ';'.join(reasons),
        }
        rows.append(row)
        if reasons:
            bad_cases.append(row)

    return pd.DataFrame(rows), bad_cases


def build_group_summary(frame: pd.DataFrame, group_by: list[str]) -> pd.DataFrame:
    valid = frame.loc[frame['has_reference'] == 1].copy()
    if valid.empty:
        return pd.DataFrame(columns=group_by + ['sample_count'])
    grouped = (
        valid.groupby(group_by, dropna=False)
        .agg(
            sample_count=('sample_id', 'count'),
            reference_force_mean=('reference_force', 'mean'),
            reference_force_std=('reference_force', 'std'),
            reference_force_median=('reference_force', 'median'),
            reference_force_min=('reference_force', 'min'),
            reference_force_max=('reference_force', 'max'),
            bad_reference_count=('bad_reference_reason', lambda values: int(sum(bool(value) for value in values))),
        )
        .reset_index()
    )
    grouped['reference_force_std'] = grouped['reference_force_std'].fillna(0.0)
    grouped['reference_force_range'] = grouped['reference_force_max'] - grouped['reference_force_min']
    return grouped.sort_values(group_by).reset_index(drop=True)


def maybe_write_distribution_plot(frame: pd.DataFrame, output_path: Path) -> bool:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    valid = frame.loc[frame['has_reference'] == 1, 'reference_force'].astype(float)
    if valid.empty:
        return False
    plt.figure(figsize=(8, 4))
    plt.hist(valid, bins=32)
    plt.xlabel('reference_force')
    plt.ylabel('sample_count')
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--project-root', default='.')
    parser.add_argument('--stage', required=True)
    parser.add_argument('--subset', default='train', choices=['train', 'val', 'test'])
    parser.add_argument('--only-stable', action='store_true')
    parser.add_argument('--output-dir', default=None)
    parser.add_argument('--std-threshold', type=float, default=100.0)
    parser.add_argument('--range-threshold', type=float, default=200.0)
    parser.add_argument('--min-window-count', type=int, default=1)
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
        output_dir = project_root / 'evals' / 'reference_quality' / str(stage_config['name']) / f'{args.subset}{suffix}'
    output_dir.mkdir(parents=True, exist_ok=True)

    sample_frame, bad_cases = summarize_samples(
        dataset,
        std_threshold=float(args.std_threshold),
        range_threshold=float(args.range_threshold),
        min_window_count=int(args.min_window_count),
    )
    group_frame = build_group_summary(sample_frame, list(args.group_by))
    object_frame = build_group_summary(sample_frame, ['object_id'])

    sample_path = output_dir / 'reference_quality_samples.csv'
    group_path = output_dir / 'reference_quality_group_summary.csv'
    object_path = output_dir / 'reference_quality_object_summary.csv'
    bad_cases_path = output_dir / 'bad_reference_cases.json'
    plot_path = output_dir / 'reference_force_distribution.png'
    summary_path = output_dir / 'reference_quality_summary.json'

    sample_frame.to_csv(sample_path, index=False, encoding='utf-8-sig')
    group_frame.to_csv(group_path, index=False, encoding='utf-8-sig')
    object_frame.to_csv(object_path, index=False, encoding='utf-8-sig')
    with bad_cases_path.open('w', encoding='utf-8') as handle:
        json.dump(bad_cases, handle, ensure_ascii=False, indent=2)

    wrote_plot = maybe_write_distribution_plot(sample_frame, plot_path)
    summary = {
        'stage_name': str(stage_config['name']),
        'subset': args.subset,
        'only_stable': bool(args.only_stable),
        'sample_count': int(len(sample_frame)),
        'reference_sample_count': int(sample_frame['has_reference'].sum()) if not sample_frame.empty else 0,
        'bad_reference_count': int(len(bad_cases)),
        'std_threshold': float(args.std_threshold),
        'range_threshold': float(args.range_threshold),
        'min_window_count': int(args.min_window_count),
        'sample_path': str(sample_path),
        'group_path': str(group_path),
        'object_path': str(object_path),
        'bad_cases_path': str(bad_cases_path),
        'distribution_plot_path': str(plot_path) if wrote_plot else None,
    }
    with summary_path.open('w', encoding='utf-8') as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    print(
        {
            'output_dir': str(output_dir),
            'summary_path': str(summary_path),
            'sample_path': str(sample_path),
            'group_path': str(group_path),
            'object_path': str(object_path),
            'bad_cases_path': str(bad_cases_path),
            'bad_reference_count': summary['bad_reference_count'],
            'distribution_plot_path': summary['distribution_plot_path'],
        }
    )


if __name__ == '__main__':
    main()
