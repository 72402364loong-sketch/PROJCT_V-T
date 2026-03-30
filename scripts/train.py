from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cmg.config import deep_update, load_yaml
from cmg.data import (
    CrossMediumSequenceDataset,
    InterfaceAwareObjectBatchSampler,
    InterfaceAwareSequenceSampler,
    ObjectAwareBatchSampler,
    sequence_collate_fn,
)
from cmg.models import CrossMediumSystem
from cmg.training import Trainer, load_checkpoint_context, load_model_weights


def load_current_configs(project_root: Path, stage_path: Path) -> tuple[dict, dict, dict, dict]:
    data_config = load_yaml(project_root / 'configs' / 'data' / 'default.yaml')
    model_config = load_yaml(project_root / 'configs' / 'model' / 'default.yaml')
    train_config = load_yaml(project_root / 'configs' / 'train' / 'base.yaml')
    stage_config = load_yaml(stage_path)
    data_config = deep_update(data_config, stage_config.get('data', {}))
    model_config = deep_update(model_config, stage_config.get('model', {}))
    train_config = deep_update(train_config, stage_config.get('train', {}))
    train_config = deep_update(train_config, {'run_name': stage_config['name']})
    return data_config, model_config, train_config, stage_config


RUNTIME_DATA_OVERRIDE_KEYS = {'visual_feature_cache_dir'}
RUNTIME_TRAIN_OVERRIDE_KEYS = {'num_workers', 'pin_memory', 'persistent_workers', 'val_every_n_epochs'}


def apply_resume_config(
    data_config: dict,
    model_config: dict,
    train_config: dict,
    stage_config: dict,
    resume_context: dict,
) -> tuple[dict, dict, dict, dict]:
    archived = resume_context.get('run_config')
    if not isinstance(archived, dict):
        return data_config, model_config, train_config, stage_config

    archived_stage = archived.get('stage') if isinstance(archived.get('stage'), dict) else {}
    archived_stage_name = archived_stage.get('name')
    current_stage_name = stage_config.get('name')
    if archived_stage_name and current_stage_name and str(archived_stage_name) != str(current_stage_name):
        raise ValueError(
            f'--resume only supports the same stage. Checkpoint stage {archived_stage_name!r} does not match current stage {current_stage_name!r}.'
        )

    if isinstance(archived.get('data'), dict):
        data_config = archived['data']
    elif isinstance(archived_stage.get('data'), dict):
        data_config = deep_update(data_config, archived_stage['data'])
    if isinstance(archived.get('model'), dict):
        model_config = archived['model']
    if isinstance(archived.get('train'), dict):
        train_config = archived['train']
    if archived_stage:
        stage_config = archived_stage
    train_config = deep_update(train_config, {'run_name': stage_config['name']})
    return data_config, model_config, train_config, stage_config


def apply_runtime_overrides(
    data_config: dict,
    train_config: dict,
    current_data_config: dict,
    current_train_config: dict,
) -> tuple[dict, dict]:
    for key in RUNTIME_DATA_OVERRIDE_KEYS:
        if key in current_data_config:
            data_config[key] = current_data_config[key]
    for key in RUNTIME_TRAIN_OVERRIDE_KEYS:
        if key in current_train_config:
            train_config[key] = current_train_config[key]
    return data_config, train_config


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_dataset(project_root: Path, split_path: Path, subset: str, data_config: dict, train_config: dict, stage_config: dict) -> CrossMediumSequenceDataset:
    allowed_trial_results = train_config.get('default_allowed_trial_results', {}).get(subset)
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
        standardize_tactile=bool(data_config.get('standardize_tactile', False)),
        min_valid_ratio_video=float(data_config.get('min_valid_ratio_video', 0.0)),
        min_valid_ratio_tactile=float(data_config.get('min_valid_ratio_tactile', 0.0)),
        normalization_subset='train',
        normalization_trial_results=train_config.get('default_allowed_trial_results', {}).get('train'),
        tail_mode=stage_config.get('tail_mode', data_config.get('valid_tail_mode', 'all_valid')),
        allowed_trial_results=allowed_trial_results,
        visual_feature_cache_dir=data_config.get('visual_feature_cache_dir'),
    )


def make_train_loader(train_dataset: CrossMediumSequenceDataset, train_config: dict) -> DataLoader:
    batch_size = int(train_config['batch_size'])
    sampling_mode = str(train_config.get('sampling_mode', '') or '').strip().lower()
    interface_alpha = float(train_config.get('interface_sampler_alpha', 1.0))
    min_interface_expert_windows = int(train_config.get('interface_sampler_min_expert_windows', 1))
    common_kwargs = {
        'num_workers': int(train_config['num_workers']),
        'collate_fn': sequence_collate_fn,
        'pin_memory': bool(train_config.get('pin_memory', False)),
    }
    if int(train_config['num_workers']) > 0 and bool(train_config.get('persistent_workers', False)):
        common_kwargs['persistent_workers'] = True

    if sampling_mode == 'interface_object_aware':
        batch_sampler = InterfaceAwareObjectBatchSampler(
            train_dataset,
            batch_size=batch_size,
            samples_per_object=int(train_config.get('samples_per_object', 2)),
            alpha=interface_alpha,
            min_interface_expert_windows=min_interface_expert_windows,
            seed=int(train_config.get('seed', 42)),
        )
        return DataLoader(train_dataset, batch_sampler=batch_sampler, **common_kwargs)

    if sampling_mode == 'interface_aware':
        sampler = InterfaceAwareSequenceSampler(
            train_dataset,
            alpha=interface_alpha,
            min_interface_expert_windows=min_interface_expert_windows,
            seed=int(train_config.get('seed', 42)),
        )
        return DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=False,
            sampler=sampler,
            **common_kwargs,
        )

    if sampling_mode == 'object_aware' or bool(train_config.get('use_object_aware_sampler', False)):
        batch_sampler = ObjectAwareBatchSampler(
            train_dataset,
            batch_size=batch_size,
            samples_per_object=int(train_config.get('samples_per_object', 2)),
            seed=int(train_config.get('seed', 42)),
        )
        return DataLoader(train_dataset, batch_sampler=batch_sampler, **common_kwargs)

    return DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        **common_kwargs,
    )


def resolve_optional_path(project_root: Path, raw_path: str | None) -> Path | None:
    if not raw_path:
        return None
    path = Path(raw_path)
    if not path.is_absolute():
        path = project_root / path
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--project-root', default='.')
    parser.add_argument('--stage', required=True)
    parser.add_argument('--resume', default=None, help='Resume the same run/stage from a checkpoint under runs/<stage>/checkpoints/.')
    parser.add_argument('--init-from', default=None, help='Initialize model weights from another stage/checkpoint without restoring optimizer, scheduler, scaler, or epoch.')
    args = parser.parse_args()

    if args.resume and args.init_from:
        parser.error('--resume and --init-from are mutually exclusive. Use --resume for same-stage continuation, or --init-from for cross-stage initialization.')

    project_root = Path(args.project_root).resolve()
    stage_path = Path(args.stage)
    if not stage_path.is_absolute():
        stage_path = project_root / stage_path

    resume_path = resolve_optional_path(project_root, args.resume)
    init_path = resolve_optional_path(project_root, args.init_from)
    resume_context = load_checkpoint_context(resume_path) if resume_path is not None else None

    data_config, model_config, train_config, stage_config = load_current_configs(project_root, stage_path)
    current_data_config = data_config.copy()
    current_train_config = train_config.copy()
    if resume_context is not None:
        data_config, model_config, train_config, stage_config = apply_resume_config(
            data_config,
            model_config,
            train_config,
            stage_config,
            resume_context,
        )
        data_config, train_config = apply_runtime_overrides(
            data_config,
            train_config,
            current_data_config,
            current_train_config,
        )

    if data_config.get('visual_feature_cache_dir') and (
        float(stage_config.get('loss_weights', {}).get('clip', 0.0)) > 0.0
        or float(stage_config.get('loss_weights', {}).get('inv', 0.0)) > 0.0
    ):
        raise ValueError('visual feature cache only supports stages with clip/inv loss weights disabled.')

    set_global_seed(int(train_config.get('seed', 42)))
    split_path = project_root / stage_config['split']

    train_dataset = make_dataset(project_root, split_path, 'train', data_config, train_config, stage_config)
    val_dataset = make_dataset(project_root, split_path, 'val', data_config, train_config, stage_config)

    train_loader = make_train_loader(train_dataset, train_config)
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(train_config['batch_size']),
        shuffle=False,
        num_workers=int(train_config['num_workers']),
        pin_memory=bool(train_config.get('pin_memory', False)),
        persistent_workers=int(train_config['num_workers']) > 0 and bool(train_config.get('persistent_workers', False)),
        collate_fn=sequence_collate_fn,
    )

    model = CrossMediumSystem(model_config)
    if init_path is not None:
        init_info = load_model_weights(model, init_path, strict=True, allow_lora_injection=True)
        print({'initialized_from': str(init_path), **init_info})

    if resume_context is not None and resume_context.get('run_dir'):
        run_dir = Path(str(resume_context['run_dir']))
    else:
        run_dir = project_root / 'runs' / stage_config['name']

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=train_config,
        stage_config=stage_config,
        model_config=model_config,
        run_dir=run_dir,
        data_config=data_config,
    )
    if resume_path is not None:
        resume_info = trainer.load_checkpoint(resume_path)
        print({'resumed_from': str(resume_path), 'metrics': resume_info})
    metrics = trainer.fit()
    print(metrics)


if __name__ == '__main__':
    main()




