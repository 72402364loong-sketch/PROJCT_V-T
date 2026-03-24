from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cmg.config import deep_update, load_yaml
from cmg.data import CrossMediumSequenceDataset
from cmg.data.sidecar import load_window_sidecar
from cmg.data.tactile import compute_measured_force_curve, load_tactile_array, resample_tactile_window, split_ac_dc, standardize_tactile_window
from cmg.data.video import load_window_frames, sample_frame_indices
from cmg.models import CrossMediumSystem
from cmg.online import OnlineInferenceStub, OnlineJSONLLogger
from cmg.training import load_checkpoint_context, load_model_weights


def resolve_path(project_root: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    if not path.is_absolute():
        path = project_root / path
    return path


def resolve_data_path(project_root: Path, relative_path: str) -> Path:
    path = Path(relative_path)
    if path.is_absolute():
        return path
    if path.parts and path.parts[0] == 'data':
        return project_root / path
    return project_root / 'data' / path


def load_configs(project_root: Path, stage_path: Path, checkpoint_context: dict) -> tuple[dict, dict, dict, dict]:
    data_config = load_yaml(project_root / 'configs' / 'data' / 'default.yaml')
    model_config = load_yaml(project_root / 'configs' / 'model' / 'default.yaml')
    train_config = load_yaml(project_root / 'configs' / 'train' / 'base.yaml')
    stage_config = load_yaml(stage_path)
    data_config = deep_update(data_config, stage_config.get('data', {}))
    model_config = deep_update(model_config, stage_config.get('model', {}))
    train_config = deep_update(train_config, stage_config.get('train', {}))

    archived = checkpoint_context.get('run_config')
    if isinstance(archived, dict):
        archived_stage = archived.get('stage') if isinstance(archived.get('stage'), dict) else {}
        archived_stage_name = archived_stage.get('name')
        current_stage_name = stage_config.get('name')
        if archived_stage_name and current_stage_name and str(archived_stage_name) != str(current_stage_name):
            raise ValueError(
                f'Checkpoint stage {archived_stage_name!r} does not match online stub stage {current_stage_name!r}.'
            )
        if isinstance(archived.get('data'), dict):
            data_config = archived['data']
        elif isinstance(archived_stage.get('data'), dict):
            data_config = deep_update(data_config, archived_stage['data'])
        if isinstance(archived.get('model'), dict):
            model_config = archived['model']
        if isinstance(archived.get('train'), dict):
            train_config = archived['train']
    return data_config, model_config, train_config, stage_config


def make_stats_dataset(project_root: Path, split_path: Path, data_config: dict, train_config: dict, stage_config: dict) -> CrossMediumSequenceDataset:
    return CrossMediumSequenceDataset(
        project_root=project_root,
        split_path=split_path,
        subset='train',
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
        allowed_trial_results=train_config.get('default_allowed_trial_results', {}).get('train'),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--project-root', default='.')
    parser.add_argument('--stage', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--sample-id', required=True)
    parser.add_argument('--output', default=None)
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    stage_path = resolve_path(project_root, args.stage)
    checkpoint_path = resolve_path(project_root, args.checkpoint)
    checkpoint_context = load_checkpoint_context(checkpoint_path)
    data_config, model_config, train_config, stage_config = load_configs(project_root, stage_path, checkpoint_context)
    split_path = project_root / stage_config['split']

    samples = pd.read_csv(project_root / 'data' / 'processed' / 'samples.csv')
    windows = pd.read_csv(project_root / 'data' / 'processed' / 'windows.csv')
    sample_rows = samples.loc[samples['sample_id'] == args.sample_id]
    if sample_rows.empty:
        raise ValueError(f'Unknown sample_id: {args.sample_id}')
    sample = sample_rows.iloc[0]
    sample_windows = windows.loc[windows['sample_id'] == args.sample_id].sort_values('window_start').reset_index(drop=True)
    if sample_windows.empty:
        raise ValueError(f'No windows found for sample_id: {args.sample_id}')

    stats_dataset = make_stats_dataset(project_root, split_path, data_config, train_config, stage_config)

    device = torch.device(train_config.get('device', 'cuda') if torch.cuda.is_available() else 'cpu')
    model = CrossMediumSystem(model_config).to(device)
    load_model_weights(model, checkpoint_path, strict=True)
    stub = OnlineInferenceStub(model, device=device)

    if args.output:
        output_path = resolve_path(project_root, args.output)
    else:
        output_path = project_root / 'data' / 'processed' / 'debug' / 'online_stub' / f'{args.sample_id}__{checkpoint_path.stem}.jsonl'
    logger = OnlineJSONLLogger(output_path)

    video_path = resolve_data_path(project_root, str(sample['video_path']))
    tactile_path = resolve_data_path(project_root, str(sample['tactile_path']))
    tactile_array = load_tactile_array(tactile_path)
    tactile_high_full, tactile_low_full = split_ac_dc(tactile_array, alpha=float(data_config['ema_alpha_acdc']))
    measured_force_curve = compute_measured_force_curve(tactile_array, normal_sign_table=data_config['normal_sign_table'])

    clip_mean = None if data_config.get('clip_mean') is None else torch.tensor(data_config['clip_mean'], dtype=torch.float32).view(1, 3, 1, 1)
    clip_std = None if data_config.get('clip_std') is None else torch.tensor(data_config['clip_std'], dtype=torch.float32).view(1, 3, 1, 1)

    for _, window in sample_windows.iterrows():
        if isinstance(window.get('sidecar_cache_path'), str) and window['sidecar_cache_path']:
            sidecar = load_window_sidecar(project_root, window['sidecar_cache_path'])
            all_frame_indices = [int(value) for value in sidecar['video_frame_indices_all'].tolist()]
        else:
            all_frame_indices = json.loads(window['video_frame_indices_json'])
        sampled_indices, frame_mask = sample_frame_indices(all_frame_indices, int(data_config['num_frames_per_window']))
        frame_array = load_window_frames(
            video_path=video_path,
            frame_indices=sampled_indices,
            image_size=int(data_config['image_size']),
            roi=data_config.get('roi'),
        )
        frame_tensor = torch.from_numpy(frame_array).permute(0, 3, 1, 2)
        if clip_mean is not None and clip_std is not None:
            frame_tensor = (frame_tensor - clip_mean) / clip_std

        start_idx = int(window['tactile_start_idx'])
        end_idx = int(window['tactile_end_idx'])
        tactile_high = tactile_high_full[start_idx:end_idx]
        tactile_low = tactile_low_full[start_idx:end_idx]
        tactile_mask = np.ones(tactile_high.shape[0], dtype=bool)
        target_points = data_config.get('tactile_points_per_window')
        if target_points is not None:
            tactile_high, high_mask = resample_tactile_window(tactile_high, int(target_points))
            tactile_low, low_mask = resample_tactile_window(tactile_low, int(target_points))
            tactile_mask = high_mask & low_mask
        if bool(data_config.get('standardize_tactile', False)) and stats_dataset.tactile_high_mean is not None and stats_dataset.tactile_low_mean is not None:
            tactile_high = standardize_tactile_window(tactile_high, stats_dataset.tactile_high_mean, stats_dataset.tactile_high_std)
            tactile_low = standardize_tactile_window(tactile_low, stats_dataset.tactile_low_mean, stats_dataset.tactile_low_std)

        outputs = stub.step(
            video=frame_tensor.unsqueeze(0),
            frame_mask=torch.tensor(frame_mask, dtype=torch.bool).unsqueeze(0),
            tactile_high=torch.from_numpy(tactile_high.astype(np.float32)).unsqueeze(0),
            tactile_low=torch.from_numpy(tactile_low.astype(np.float32)).unsqueeze(0),
            tactile_mask=torch.from_numpy(tactile_mask.astype(bool)).unsqueeze(0),
        )

        measured_force = float(np.mean(measured_force_curve[start_idx:end_idx])) if end_idx > start_idx else math.nan
        medium_probs = outputs['medium_probs'][0].tolist()
        logger.write(
            {
                'sample_id': args.sample_id,
                'object_id': str(sample['object_id']),
                'trial_result': str(sample['trial_result']),
                'water_condition': str(sample['water_condition']),
                'lift_speed': str(sample['lift_speed']),
                'placement_variant': str(sample['placement_variant']),
                'window_id': str(window['window_id']),
                'window_start': float(window['window_start']),
                'window_end': float(window['window_end']),
                'window_center': float(window['window_center']),
                'F_des': float(outputs['force_pred'][0].item()),
                'F_meas': measured_force,
                'p_medium': medium_probs,
                'medium_hidden_norm': float(outputs['medium_hidden'].norm().item()),
            }
        )

    print({'output_path': str(output_path), 'steps': int(len(sample_windows))})


if __name__ == '__main__':
    main()

