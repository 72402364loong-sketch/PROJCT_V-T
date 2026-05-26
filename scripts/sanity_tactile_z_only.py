from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cmg.config import deep_update, load_yaml, sync_tactile_model_config
from cmg.data import CrossMediumSequenceDataset, sequence_collate_fn
from cmg.losses import compute_losses
from cmg.models import CrossMediumSystem
from cmg.training import build_phase_class_weights, move_to_device


def resolve_path(project_root: Path, raw_path: str | Path) -> Path:
    path = Path(raw_path)
    if not path.is_absolute():
        path = project_root / path
    return path


def load_stage_configs(project_root: Path, stage_path: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    data_config = load_yaml(project_root / 'configs' / 'data' / 'default.yaml')
    model_config = load_yaml(project_root / 'configs' / 'model' / 'default.yaml')
    train_config = load_yaml(project_root / 'configs' / 'train' / 'base.yaml')
    stage_config = load_yaml(stage_path)
    data_config = deep_update(data_config, stage_config.get('data', {}))
    model_config = deep_update(model_config, stage_config.get('model', {}))
    train_config = deep_update(train_config, stage_config.get('train', {}))
    train_config = deep_update(train_config, {'run_name': stage_config['name']})
    return data_config, model_config, train_config, stage_config


def make_dataset(
    project_root: Path,
    split_path: Path,
    subset: str,
    data_config: dict[str, Any],
    train_config: dict[str, Any],
    stage_config: dict[str, Any],
) -> CrossMediumSequenceDataset:
    return CrossMediumSequenceDataset(
        project_root=project_root,
        split_path=split_path,
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
        attribute_taxonomy=str(data_config.get('attribute_taxonomy', 'legacy')),
        physical_attribute_table=data_config.get('physical_attribute_table'),
        physical_attribute_norm_stats_path=data_config.get('physical_attribute_norm_stats_path'),
    )


def array_equal_exact(left: list[Any], right: list[Any]) -> bool:
    left_array = np.asarray(left)
    right_array = np.asarray(right)
    return bool(np.array_equal(left_array, right_array, equal_nan=True))


def finite_max_abs_diff(left: list[Any], right: list[Any]) -> float:
    left_array = np.asarray(left, dtype=np.float64)
    right_array = np.asarray(right, dtype=np.float64)
    finite = np.isfinite(left_array) & np.isfinite(right_array)
    if not finite.any():
        return 0.0
    return float(np.max(np.abs(left_array[finite] - right_array[finite])))


def dataset_signature(dataset: CrossMediumSequenceDataset) -> dict[str, Any]:
    sample_ids = [str(sample['sample_id']) for sample in dataset.sample_records]
    window_ids_by_sample = {
        sample_id: [str(window['window_id']) for window in dataset.windows_by_sample[sample_id]]
        for sample_id in sample_ids
    }
    return {
        'sample_count': int(len(sample_ids)),
        'window_count': int(sum(len(value) for value in window_ids_by_sample.values())),
        'sample_ids': sample_ids,
        'window_ids_by_sample': window_ids_by_sample,
    }


def compare_label_chains(
    baseline_dataset: CrossMediumSequenceDataset,
    z_only_dataset: CrossMediumSequenceDataset,
    *,
    max_samples: int | None,
) -> dict[str, Any]:
    baseline_sig = dataset_signature(baseline_dataset)
    z_only_sig = dataset_signature(z_only_dataset)
    if baseline_sig != z_only_sig:
        raise AssertionError('Sample/window signatures differ between baseline and z-only datasets.')

    sample_count = len(baseline_dataset) if max_samples is None else min(len(baseline_dataset), max_samples)
    keys = ['reference_forces', 'delta_force_targets', 'phase_labels']
    exact_counts = {key: 0 for key in keys}
    max_diffs = {key: 0.0 for key in keys}
    first_window_reference_pair: dict[str, Any] | None = None
    for index in range(sample_count):
        baseline_item = baseline_dataset[index]
        z_only_item = z_only_dataset[index]
        sample_id = str(baseline_item['sample_id'])
        windows = baseline_dataset.windows_by_sample[sample_id]
        for key in keys:
            if not array_equal_exact(baseline_item[key], z_only_item[key]):
                raise AssertionError(f'{key} differs for sample {sample_id}.')
            exact_counts[key] += len(baseline_item[key])
            max_diffs[key] = max(max_diffs[key], finite_max_abs_diff(baseline_item[key], z_only_item[key]))
        if first_window_reference_pair is None and windows:
            first_window_reference_pair = {
                'sample_id': sample_id,
                'window_id': str(windows[0]['window_id']),
                'baseline_reference_force': float(baseline_item['reference_forces'][0]),
                'z_only_reference_force': float(z_only_item['reference_forces'][0]),
            }

    return {
        'sample_count_checked': int(sample_count),
        'window_count_checked': int(sum(len(baseline_dataset[index]['phase_labels']) for index in range(sample_count))),
        'exact_match_counts': exact_counts,
        'finite_max_abs_diff': max_diffs,
        'first_window_reference_pair': first_window_reference_pair,
        'sample_count': baseline_sig['sample_count'],
        'window_count': baseline_sig['window_count'],
    }


def build_axis_datasets(
    project_root: Path,
    stage_path: Path,
    subset: str,
    *,
    max_samples: int | None = None,
) -> tuple[CrossMediumSequenceDataset, CrossMediumSequenceDataset, dict[str, Any], dict[str, Any], dict[str, Any]]:
    data_config, model_config, train_config, stage_config = load_stage_configs(project_root, stage_path)
    baseline_data_config = deep_update(data_config, {'tactile_input_axes': ['x', 'y', 'z']})
    z_only_data_config = deep_update(data_config, {'tactile_input_axes': ['z']})
    baseline_model_config = sync_tactile_model_config(baseline_data_config, model_config)
    z_only_model_config = sync_tactile_model_config(z_only_data_config, model_config)
    split_path = project_root / stage_config['split']
    baseline_dataset = make_dataset(project_root, split_path, subset, baseline_data_config, train_config, stage_config)
    z_only_dataset = make_dataset(project_root, split_path, subset, z_only_data_config, train_config, stage_config)
    _ = max_samples
    return baseline_dataset, z_only_dataset, baseline_model_config, z_only_model_config, train_config


def run_sanity(project_root: Path, stage_path: Path, subset: str, max_samples: int | None) -> dict[str, Any]:
    baseline_dataset, z_only_dataset, baseline_model_config, z_only_model_config, train_config = build_axis_datasets(
        project_root,
        stage_path,
        subset,
        max_samples=max_samples,
    )
    baseline_batch = next(iter(DataLoader(baseline_dataset, batch_size=int(train_config['batch_size']), shuffle=False, collate_fn=sequence_collate_fn)))
    z_only_batch = next(iter(DataLoader(z_only_dataset, batch_size=int(train_config['batch_size']), shuffle=False, collate_fn=sequence_collate_fn)))
    z_only_model = CrossMediumSystem(z_only_model_config)

    if baseline_batch['tactile_high'].shape[-1] != 36 or baseline_batch['tactile_low'].shape[-1] != 36:
        raise AssertionError('Baseline batch is not 36-channel tactile input.')
    if z_only_batch['tactile_high'].shape[-1] != 12 or z_only_batch['tactile_low'].shape[-1] != 12:
        raise AssertionError('z-only batch is not 12-channel tactile input.')
    if z_only_model.tactile.input_dim != 12 or z_only_model.tactile.axis_dim != 1:
        raise AssertionError('z-only model tactile metadata is not input_dim=12, axis_dim=1.')

    label_summary = compare_label_chains(
        baseline_dataset,
        z_only_dataset,
        max_samples=max_samples,
    )
    return {
        'baseline_tactile_high_shape': list(baseline_batch['tactile_high'].shape),
        'baseline_tactile_low_shape': list(baseline_batch['tactile_low'].shape),
        'z_only_tactile_high_shape': list(z_only_batch['tactile_high'].shape),
        'z_only_tactile_low_shape': list(z_only_batch['tactile_low'].shape),
        'z_only_model_tactile': {
            'input_dim': int(z_only_model.tactile.input_dim),
            'axis_dim': int(z_only_model.tactile.axis_dim),
            'num_taxels': int(z_only_model.tactile.num_taxels),
        },
        **label_summary,
    }


def summarize_overfit_step(outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor], loss_value: float) -> dict[str, Any]:
    force_pred = outputs['force_pred'].reshape_as(batch['control_force_targets'])
    force_delta = outputs['force_interface_delta'].reshape_as(batch['delta_force_targets'])
    control_mask = batch['has_control_target'] & batch['window_mask']
    delta_mask = batch['delta_supervision_masks'] & batch['window_mask']
    if not control_mask.any():
        raise AssertionError('Tiny overfit batch has no control targets.')
    control_abs = torch.abs(force_pred[control_mask] - batch['control_force_targets'][control_mask])
    summary = {
        'loss_total': float(loss_value),
        'control_mae': float(control_abs.mean().item()),
        'force_pred_mean': float(force_pred[control_mask].mean().item()),
        'force_target_mean': float(batch['control_force_targets'][control_mask].mean().item()),
    }
    if delta_mask.any():
        delta_abs = torch.abs(force_delta[delta_mask])
        target_delta_abs = torch.abs(batch['delta_force_targets'][delta_mask])
        delta_error = torch.abs(force_delta[delta_mask] - batch['delta_force_targets'][delta_mask])
        summary.update(
            {
                'interface_delta_abs_mean': float(delta_abs.mean().item()),
                'target_delta_abs_mean': float(target_delta_abs.mean().item()),
                'interface_delta_mae': float(delta_error.mean().item()),
            }
        )
    return summary


def run_tiny_overfit(project_root: Path, stage_path: Path, subset: str, steps: int, lr: float) -> dict[str, Any]:
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    data_config, model_config, train_config, stage_config = load_stage_configs(project_root, stage_path)
    data_config = deep_update(data_config, {'tactile_input_axes': ['z']})
    model_config = sync_tactile_model_config(data_config, model_config)
    train_config = deep_update(train_config, {'device': 'cpu', 'amp_enabled': False})

    dataset = make_dataset(project_root, project_root / stage_config['split'], subset, data_config, train_config, stage_config)
    batch = next(iter(DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=sequence_collate_fn)))
    if batch['tactile_high'].shape[-1] != 12 or batch['tactile_low'].shape[-1] != 12:
        raise AssertionError('Tiny overfit batch is not z-only.')

    device = torch.device('cpu')
    batch = move_to_device(batch, device)
    model = CrossMediumSystem(model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=0.0)
    phase_class_weights = build_phase_class_weights(train_config, device)
    checkpoints = {0, max(1, steps // 2), steps}
    records: list[dict[str, Any]] = []

    for step in range(steps + 1):
        model.train()
        outputs = model(batch)
        losses = compute_losses(
            outputs,
            batch,
            loss_weights=stage_config.get('loss_weights', {}),
            attribute_loss_config=stage_config.get('attribute_loss'),
            policy_loss_config=stage_config.get('policy_loss'),
            temperature_clip=float(model_config['losses']['temperature_clip']),
            temperature_inv=float(model_config['losses']['temperature_inv']),
            phase_class_weights=phase_class_weights,
        )
        if step in checkpoints:
            record = {'step': int(step)}
            record.update(summarize_overfit_step(outputs, batch, float(losses['total'].item())))
            records.append(record)
        if step == steps:
            break
        optimizer.zero_grad(set_to_none=True)
        losses['total'].backward()
        optimizer.step()

    return {
        'sample_ids': batch['sample_ids'],
        'object_ids': batch['object_ids'],
        'tactile_high_shape': list(batch['tactile_high'].shape),
        'tactile_low_shape': list(batch['tactile_low'].shape),
        'model_tactile': {
            'input_dim': int(model.tactile.input_dim),
            'axis_dim': int(model.tactile.axis_dim),
            'num_taxels': int(model.tactile.num_taxels),
        },
        'steps': int(steps),
        'learning_rate': float(lr),
        'records': records,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--project-root', default='.')
    parser.add_argument('--stage', required=True)
    parser.add_argument('--subset', default='train', choices=['train', 'val', 'test'])
    parser.add_argument('--max-samples', type=int, default=0)
    parser.add_argument('--tiny-overfit', action='store_true')
    parser.add_argument('--steps', type=int, default=30)
    parser.add_argument('--lr', type=float, default=1e-3)
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    stage_path = resolve_path(project_root, args.stage)
    max_samples = None if int(args.max_samples) <= 0 else int(args.max_samples)
    payload = {
        'stage': str(stage_path),
        'subset': args.subset,
        'sanity': run_sanity(project_root, stage_path, args.subset, max_samples=max_samples),
    }
    if args.tiny_overfit:
        payload['tiny_overfit'] = run_tiny_overfit(
            project_root,
            stage_path,
            args.subset,
            steps=int(args.steps),
            lr=float(args.lr),
        )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
