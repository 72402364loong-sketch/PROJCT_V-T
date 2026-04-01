from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from .config import deep_update, load_yaml
from .data import CrossMediumSequenceDataset, sequence_collate_fn
from .models import CrossMediumSystem
from .training import build_phase_class_weights, load_checkpoint_context, load_model_weights, run_model_epoch



def resolve_path(project_root: Path, raw_path: str | Path) -> Path:
    path = Path(raw_path)
    if not path.is_absolute():
        path = project_root / path
    return path



def apply_archived_run_config(
    data_config: dict[str, Any],
    model_config: dict[str, Any],
    train_config: dict[str, Any],
    stage_config: dict[str, Any],
    checkpoint_context: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    archived = checkpoint_context.get('run_config')
    if not isinstance(archived, dict):
        return data_config, model_config, train_config

    archived_stage = archived.get('stage') if isinstance(archived.get('stage'), dict) else {}
    archived_stage_name = archived_stage.get('name')
    current_stage_name = stage_config.get('name')
    if archived_stage_name and current_stage_name and str(archived_stage_name) != str(current_stage_name):
        raise ValueError(
            f'Checkpoint stage {archived_stage_name!r} does not match evaluation stage {current_stage_name!r}.'
        )

    if isinstance(archived.get('data'), dict):
        data_config = archived['data']
    elif isinstance(archived_stage.get('data'), dict):
        data_config = deep_update(data_config, archived_stage['data'])
    if isinstance(archived.get('model'), dict):
        model_config = archived['model']
    if isinstance(archived.get('train'), dict):
        train_config = archived['train']
    return data_config, model_config, train_config



def prepare_evaluation_context(
    *,
    project_root: str | Path,
    stage: str | Path,
    checkpoint: str | Path,
    subset: str = 'test',
    only_stable: bool = False,
) -> dict[str, Any]:
    project_root = Path(project_root).resolve()
    stage_path = resolve_path(project_root, stage)
    checkpoint_path = resolve_path(project_root, checkpoint)
    checkpoint_context = load_checkpoint_context(checkpoint_path)

    data_config = load_yaml(project_root / 'configs' / 'data' / 'default.yaml')
    model_config = load_yaml(project_root / 'configs' / 'model' / 'default.yaml')
    train_config = load_yaml(project_root / 'configs' / 'train' / 'base.yaml')
    stage_config = load_yaml(stage_path)
    data_config = deep_update(data_config, stage_config.get('data', {}))
    model_config = deep_update(model_config, stage_config.get('model', {}))
    train_config = deep_update(train_config, stage_config.get('train', {}))
    data_config, model_config, train_config = apply_archived_run_config(
        data_config,
        model_config,
        train_config,
        stage_config,
        checkpoint_context,
    )
    if data_config.get('visual_feature_cache_dir') and (
        float(stage_config.get('loss_weights', {}).get('clip', 0.0)) > 0.0
        or float(stage_config.get('loss_weights', {}).get('inv', 0.0)) > 0.0
    ):
        raise ValueError('visual feature cache only supports stages with clip/inv loss weights disabled.')
    split_path = project_root / stage_config['split']

    allowed_trial_results = ['stable'] if only_stable else None
    dataset = CrossMediumSequenceDataset(
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
        standardize_tactile=bool(data_config.get('standardize_tactile', False)),
        min_valid_ratio_video=float(data_config.get('min_valid_ratio_video', 0.0)),
        min_valid_ratio_tactile=float(data_config.get('min_valid_ratio_tactile', 0.0)),
        normalization_subset='train',
        normalization_trial_results=train_config.get('default_allowed_trial_results', {}).get('train'),
        tail_mode=stage_config.get('tail_mode', data_config.get('valid_tail_mode', 'all_valid')),
        allowed_trial_results=allowed_trial_results,
        visual_feature_cache_dir=data_config.get('visual_feature_cache_dir'),
        reference_force_window_count=int(data_config.get('reference_force_window_count', 3)),
    )

    loader_kwargs: dict[str, Any] = {
        'batch_size': int(train_config['batch_size']),
        'shuffle': False,
        'num_workers': int(train_config['num_workers']),
        'pin_memory': bool(train_config.get('pin_memory', False)),
        'collate_fn': sequence_collate_fn,
    }
    if int(train_config['num_workers']) > 0 and bool(train_config.get('persistent_workers', False)):
        loader_kwargs['persistent_workers'] = True
    loader = DataLoader(dataset, **loader_kwargs)

    device = torch.device(train_config.get('device', 'cuda') if torch.cuda.is_available() else 'cpu')
    model = CrossMediumSystem(model_config).to(device)
    load_info = load_model_weights(model, checkpoint_path, strict=True)
    phase_class_weights = build_phase_class_weights(train_config, device)
    return {
        'project_root': project_root,
        'stage_path': stage_path,
        'checkpoint_path': checkpoint_path,
        'subset': subset,
        'only_stable': bool(only_stable),
        'stage_config': stage_config,
        'data_config': data_config,
        'model_config': model_config,
        'train_config': train_config,
        'dataset': dataset,
        'loader': loader,
        'device': device,
        'model': model,
        'phase_class_weights': phase_class_weights,
        'checkpoint_context': checkpoint_context,
        'load_info': load_info,
    }



def evaluate_checkpoint(
    *,
    project_root: str | Path,
    stage: str | Path,
    checkpoint: str | Path,
    subset: str = 'test',
    only_stable: bool = False,
) -> dict[str, Any]:
    context = prepare_evaluation_context(
        project_root=project_root,
        stage=stage,
        checkpoint=checkpoint,
        subset=subset,
        only_stable=only_stable,
    )
    metrics = run_model_epoch(
        context['model'],
        context['loader'],
        device=context['device'],
        training=False,
        stage_config=context['stage_config'],
        model_config=context['model_config'],
        phase_class_weights=context['phase_class_weights'],
        amp_enabled=bool(context['train_config'].get('amp_enabled', False)) and context['device'].type == 'cuda',
    )
    return {
        'stage_name': context['stage_config'].get('name'),
        'stage_path': str(context['stage_path'].resolve()),
        'subset': subset,
        'only_stable': bool(only_stable),
        'checkpoint_path': str(context['checkpoint_path'].resolve()),
        'checkpoint_run_config_path': context['load_info'].get('run_config_path'),
        'metrics': metrics,
    }
