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


def load_dataset_from_stage(project_root: Path, stage_path: Path, subset: str) -> tuple[CrossMediumSequenceDataset, dict[str, Any]]:
    data_config = load_yaml(project_root / 'configs' / 'data' / 'default.yaml')
    train_config = load_yaml(project_root / 'configs' / 'train' / 'base.yaml')
    stage_config = load_yaml(stage_path)
    data_config = deep_update(data_config, stage_config.get('data', {}))
    train_config = deep_update(train_config, stage_config.get('train', {}))

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
        allowed_trial_results=train_config.get('default_allowed_trial_results', {}).get(subset),
        visual_feature_cache_dir=data_config.get('visual_feature_cache_dir'),
        samples_path=data_config.get('samples_path', 'data/processed/samples.csv'),
        windows_path=data_config.get('windows_path', 'data/processed/windows.csv'),
        reference_force_window_count=int(data_config.get('reference_force_window_count', 3)),
        reference_force_statistic=str(data_config.get('reference', {}).get('statistic', data_config.get('reference_force_statistic', 'mean'))),
        reference_force_duration_sec=data_config.get('reference', {}).get('duration_sec'),
        reference_force_source=str(data_config.get('reference', {}).get('source', 'window_targets')),
        target_aggregation=str(data_config.get('target', {}).get('aggregation', 'window_mean')),
        target_aggregation_sec=data_config.get('target', {}).get('aggregation_sec'),
        target_min_samples=int(data_config.get('target', {}).get('min_samples', 1)),
        target_fallback=str(data_config.get('target', {}).get('fallback', 'latest_causal')),
        expert_force_mode=str(data_config.get('expert_force_mode', 'measured_force')),
        expert_force_smoothing=str(data_config.get('expert_force_smoothing', 'ema')),
        expert_force_baseline_mode=str(data_config.get('expert_force_baseline_mode', 'none')),
        expert_force_baseline_window_sec=float(data_config.get('expert_force_baseline_window_sec', 0.5)),
        expert_force_interface_margin_sec=float(data_config.get('expert_force_interface_margin_sec', 0.0)),
        soft_gate_pre_sec=float(data_config.get('soft_gate_pre_sec', 0.0)),
        soft_gate_post_sec=float(data_config.get('soft_gate_post_sec', 0.0)),
        soft_gate_ramp=str(data_config.get('soft_gate_ramp', 'linear')),
        interface_context_manifest=data_config.get('interface_context_manifest'),
        attribute_taxonomy=str(data_config.get('attribute_taxonomy', 'legacy')),
        physical_attribute_table=data_config.get('physical_attribute_table'),
        physical_attribute_norm_stats_path=data_config.get('physical_attribute_norm_stats_path'),
    )
    return dataset, stage_config


def build_curves(dataset: CrossMediumSequenceDataset, sample: dict[str, Any]) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    tactile_array = load_tactile_array(dataset._resolve_data_path(sample['tactile_path']))
    expert_curve = compute_expert_force_curve(
        tactile_array=tactile_array,
        dt=dataset.tactile_dt,
        trial_result=sample['trial_result'],
        contact_time=sample.get('t_contact_all'),
        alpha=dataset.expert_alpha,
        normal_sign_table=dataset.normal_sign_table,
    )
    clean_force_curve = None
    tactile_time_axis = None
    if dataset.expert_force_mode == 'local_reference_delta':
        sync_offset_sec = dataset._parse_optional_float(sample.get('sync_offset_sec')) or 0.0
        contact_time = dataset._parse_optional_float(sample.get('t_contact_all'))
        clean_force_curve = compute_clean_force_curve(
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
        tactile_time_axis = np.arange(tactile_array.shape[0], dtype=np.float32) * dataset.tactile_dt - sync_offset_sec
    return expert_curve, clean_force_curve, tactile_time_axis


def summarize_subset(dataset: CrossMediumSequenceDataset, subset: str, large_delta_threshold: float) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for sample in dataset.sample_records:
        sample_id = str(sample['sample_id'])
        object_id = str(sample['object_id'])
        windows = dataset.windows_by_sample.get(sample_id, [])
        expert_curve, clean_force_curve, tactile_time_axis = build_curves(dataset, sample)
        reference_curve = clean_force_curve if clean_force_curve is not None else expert_curve
        sample_reference_force, has_sample_reference, reference_window_indices = dataset._compute_sample_reference_force(
            windows,
            reference_curve,
        )

        interface_interval_start = None
        interface_interval_end = None
        t_if_enter = dataset._parse_optional_float(sample.get('t_if_enter'))
        t_if_exit = dataset._parse_optional_float(sample.get('t_if_exit'))
        if t_if_enter is not None and t_if_exit is not None and t_if_exit > t_if_enter:
            interface_interval_start = float(t_if_enter) - dataset.expert_force_interface_margin_sec
            interface_interval_end = float(t_if_exit) + dataset.expert_force_interface_margin_sec

        use_local_reference_targets = (
            dataset.expert_force_mode == 'local_reference_delta'
            and clean_force_curve is not None
            and tactile_time_axis is not None
            and has_sample_reference
            and interface_interval_start is not None
            and interface_interval_end is not None
        )

        context_meta = dataset.interface_context_by_sample.get(sample_id, {})
        for window_index, window in enumerate(windows):
            phase_label = str(window['phase_label'])
            is_interface_window = phase_label == 'Interface'
            start_idx = int(window['tactile_start_idx'])
            end_idx = int(window['tactile_end_idx'])

            expert_force = float('nan')
            if expert_curve is not None:
                expert_force = dataset._compute_window_force_value(expert_curve, start_idx, end_idx)

            target_force = float('nan')
            delta_force_target = float('nan')
            has_delta_target = False
            if use_local_reference_targets and is_interface_window:
                local_overlap_mask = dataset._window_interval_mask(
                    tactile_time_axis,
                    start_idx,
                    end_idx,
                    interval_start=float(interface_interval_start),
                    interval_end=float(interface_interval_end),
                )
                target_force = dataset._compute_window_force_value(
                    clean_force_curve,
                    start_idx,
                    end_idx,
                    sample_mask=local_overlap_mask if bool(local_overlap_mask.any()) else None,
                )
                if not np.isfinite(target_force):
                    target_force = dataset._compute_window_force_value(clean_force_curve, start_idx, end_idx)
                if np.isfinite(target_force):
                    delta_force_target = float(target_force - sample_reference_force)
                    has_delta_target = True
            elif is_interface_window and has_sample_reference and np.isfinite(expert_force):
                target_force = float(expert_force)
                delta_force_target = float(expert_force - sample_reference_force)
                has_delta_target = True

            if not has_delta_target:
                continue

            target_sign = 'pos' if delta_force_target > 0.0 else 'neg' if delta_force_target < 0.0 else 'zero'
            is_large_delta = bool(abs(delta_force_target) >= large_delta_threshold)
            rows.append(
                {
                    'subset': subset,
                    'sample_id': sample_id,
                    'object_id': object_id,
                    'object_name': str(sample.get('object_name', '')),
                    'water_condition': str(sample.get('water_condition', '')),
                    'lift_speed': str(sample.get('lift_speed', '')),
                    'placement_variant': str(sample.get('placement_variant', '')),
                    'trial_result': str(sample.get('trial_result', '')),
                    'window_index': int(window_index),
                    'window_id': str(window.get('window_id', '')),
                    'window_center': float(window.get('window_center', float('nan'))),
                    'phase_label': phase_label,
                    'expert_force': float(expert_force),
                    'control_force_target': float(target_force),
                    'reference_force': float(sample_reference_force),
                    'delta_force_target': float(delta_force_target),
                    'abs_delta_force_target': float(abs(delta_force_target)),
                    'delta_target_sign': target_sign,
                    'is_large_delta': int(is_large_delta),
                    'is_large_pos_delta': int(is_large_delta and delta_force_target > 0.0),
                    'is_large_neg_delta': int(is_large_delta and delta_force_target < 0.0),
                    'has_interface_context': int(bool(context_meta.get('has_context', False))),
                    'selected_context_count': int(context_meta.get('selected_context_count', 0) or 0),
                    'stable_context_count': int(context_meta.get('stable_context_count', 0) or 0),
                    'used_fallback_context': int(bool(context_meta.get('used_fallback_context', False))),
                    'reference_window_count': int(len(reference_window_indices)),
                    'reference_window_indices_json': json.dumps([int(index) for index in reference_window_indices]),
                }
            )
    return pd.DataFrame(rows)


def grouped_summary(frame: pd.DataFrame, group_by: list[str]) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    grouped = (
        frame.groupby(group_by, dropna=False)
        .agg(
            interface_target_count=('delta_force_target', 'count'),
            delta_mean=('delta_force_target', 'mean'),
            delta_min=('delta_force_target', 'min'),
            delta_max=('delta_force_target', 'max'),
            abs_delta_mean=('abs_delta_force_target', 'mean'),
            large_count=('is_large_delta', 'sum'),
            large_pos_count=('is_large_pos_delta', 'sum'),
            large_neg_count=('is_large_neg_delta', 'sum'),
            context_window_mean=('selected_context_count', 'mean'),
            context_sample_rate=('has_interface_context', 'mean'),
            reference_force_mean=('reference_force', 'mean'),
            reference_force_min=('reference_force', 'min'),
            reference_force_max=('reference_force', 'max'),
        )
        .reset_index()
    )
    return grouped.sort_values(['large_neg_count', 'large_pos_count', 'interface_target_count'], ascending=[False, False, False])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--project-root', default='.')
    parser.add_argument('--stage', required=True)
    parser.add_argument('--subset', default='val', choices=['train', 'val', 'test', 'all'])
    parser.add_argument('--output-dir', default=None)
    parser.add_argument('--large-delta-threshold', type=float, default=100.0)
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    stage_path = resolve_path(project_root, args.stage)
    subsets = ['train', 'val', 'test'] if args.subset == 'all' else [args.subset]
    subset_frames: list[pd.DataFrame] = []
    stage_config = None
    for subset in subsets:
        dataset, loaded_stage_config = load_dataset_from_stage(project_root, stage_path, subset)
        stage_config = loaded_stage_config
        subset_frames.append(summarize_subset(dataset, subset, float(args.large_delta_threshold)))

    frame = pd.concat(subset_frames, ignore_index=True) if subset_frames else pd.DataFrame()
    stage_name = str(stage_config['name'] if stage_config is not None else stage_path.stem)
    output_dir = resolve_path(project_root, args.output_dir) if args.output_dir else (
        project_root / 'evals' / 'delta_targets' / stage_name / str(args.subset)
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    windows_path = output_dir / 'delta_target_windows.csv'
    sample_path = output_dir / 'delta_target_sample_summary.csv'
    object_path = output_dir / 'delta_target_object_summary.csv'
    subset_path = output_dir / 'delta_target_subset_summary.csv'
    summary_path = output_dir / 'delta_target_summary.json'

    sample_summary = grouped_summary(frame, ['subset', 'sample_id', 'object_id', 'object_name'])
    object_summary = grouped_summary(frame, ['subset', 'object_id', 'object_name'])
    subset_summary = grouped_summary(frame, ['subset'])

    frame.to_csv(windows_path, index=False, encoding='utf-8-sig')
    sample_summary.to_csv(sample_path, index=False, encoding='utf-8-sig')
    object_summary.to_csv(object_path, index=False, encoding='utf-8-sig')
    subset_summary.to_csv(subset_path, index=False, encoding='utf-8-sig')

    summary = {
        'stage_name': stage_name,
        'subset': args.subset,
        'large_delta_threshold': float(args.large_delta_threshold),
        'interface_target_count': int(len(frame)),
        'large_delta_count': int(frame['is_large_delta'].sum()) if not frame.empty else 0,
        'large_pos_count': int(frame['is_large_pos_delta'].sum()) if not frame.empty else 0,
        'large_neg_count': int(frame['is_large_neg_delta'].sum()) if not frame.empty else 0,
        'large_neg_sample_count': int(frame.loc[frame['is_large_neg_delta'] == 1, 'sample_id'].nunique()) if not frame.empty else 0,
        'large_neg_object_count': int(frame.loc[frame['is_large_neg_delta'] == 1, 'object_id'].nunique()) if not frame.empty else 0,
        'context_sample_rate': float(frame.groupby('sample_id')['has_interface_context'].max().mean()) if not frame.empty else 0.0,
        'windows_path': str(windows_path),
        'sample_path': str(sample_path),
        'object_path': str(object_path),
        'subset_path': str(subset_path),
    }
    with summary_path.open('w', encoding='utf-8') as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    print(summary)


if __name__ == '__main__':
    main()
