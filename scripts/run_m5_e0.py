from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cmg.config import resolve_project_path, resolve_training_configs, sha256_file, sync_tactile_model_config
from cmg.data import load_sample_visual_feature_cache, sequence_collate_fn, write_sample_visual_feature_cache
from cmg.data.splits import resolve_sample_ids
from cmg.data.video import load_frames_by_indices, sample_frame_indices
from cmg.evaluation import prepare_evaluation_context
from cmg.models import CrossMediumSystem
from cmg.training import load_model_weights, move_to_device, run_model_epoch


DIRECTION_PREFIXES = (
    'direction_embedding.*',
    'medium_direction_adapter.*',
    'policy_direction_adapter.*',
)
CORE_OUTPUT_KEYS = (
    'medium_logits',
    'force_pred',
    'force_base',
    'force_interface_delta',
    'finger_force_pred',
    'finger_force_base',
    'finger_force_interface_delta',
)


def resolve_path(project_root: Path, raw_path: str | Path) -> Path:
    return resolve_project_path(project_root, raw_path)


def json_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def aggregate_file_sha256(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.name.encode('utf-8'))
        digest.update(sha256_file(path).encode('ascii'))
    return digest.hexdigest()


def video_path_for_sample(project_root: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    if path.parts and path.parts[0] == 'data':
        return project_root / path
    return project_root / 'data' / path


def stable_val_rows(
    project_root: Path,
    data_config: dict[str, Any],
    stage_config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    samples_path = resolve_path(project_root, data_config['samples_path'])
    windows_path = resolve_path(project_root, data_config['windows_path'])
    samples = pd.read_csv(samples_path)
    windows = pd.read_csv(windows_path)
    val_ids = set(resolve_sample_ids(samples, resolve_path(project_root, stage_config['split']), subset='val'))
    stable_mask = samples['trial_result'].astype(str).str.strip().str.lower().eq('stable')
    selected_samples = samples.loc[samples['sample_id'].isin(val_ids) & stable_mask].copy()
    selected_ids = set(selected_samples['sample_id'].astype(str))
    selected_windows = windows.loc[windows['sample_id'].astype(str).isin(selected_ids)].copy()
    if selected_samples.empty or selected_windows.empty:
        raise RuntimeError('M5-0 E0 stable-only Val cohort is empty.')
    if 'split' in selected_samples.columns and not selected_samples['split'].astype(str).eq('val').all():
        raise RuntimeError('E0 cohort includes a non-Val sample.')
    if not selected_samples['trial_result'].astype(str).str.lower().eq('stable').all():
        raise RuntimeError('E0 cohort includes a failed sample.')
    return selected_samples.sort_values('sample_id'), selected_windows


def load_visual_model(
    model_config: dict[str, Any],
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[CrossMediumSystem, dict[str, Any]]:
    construction_config = copy.deepcopy(model_config)
    construction_config['visual']['pretrained'] = False
    construction_config['visual']['pretrained_tag'] = None
    model = CrossMediumSystem(construction_config)
    load_info = load_model_weights(
        model,
        checkpoint_path,
        strict=True,
        allow_lora_injection=True,
        allowed_missing_prefixes=DIRECTION_PREFIXES,
    )
    if load_info['target_tensor_count'] - load_info['loaded_tensor_count'] != 9:
        raise RuntimeError(f'Expected exactly 9 direction-only missing tensors, got {load_info["missing_keys"]}.')
    model.eval()
    model.visual_encoder.to(device).eval()
    return model, load_info


def encode_unique_frame_windows(
    visual_encoder: torch.nn.Module,
    frame_map: dict[int, np.ndarray],
    window_specs: list[tuple[str, list[int], list[bool]]],
    *,
    device: torch.device,
    clip_mean: torch.Tensor | None,
    clip_std: torch.Tensor | None,
    frames_per_batch: int,
    pool_windows_per_batch: int,
) -> np.ndarray:
    encode_started = time.time()
    unique_indices = sorted({index for _, indices, _ in window_specs for index in indices})
    index_to_position = {frame_index: position for position, frame_index in enumerate(unique_indices)}
    token_chunks: list[torch.Tensor] = []
    with torch.inference_mode():
        for start in range(0, len(unique_indices), frames_per_batch):
            indices = unique_indices[start : start + frames_per_batch]
            array = np.stack([frame_map[index] for index in indices], axis=0)
            frames = torch.from_numpy(array).permute(0, 3, 1, 2)
            if clip_mean is not None and clip_std is not None:
                frames = (frames - clip_mean) / clip_std
            frames = frames.to(device)
            if visual_encoder.backbone_kind == 'open_clip':
                features = visual_encoder._forward_open_clip_frames(frames)
            else:
                features = visual_encoder.backbone(frames)
                if visual_encoder.adapter is not None:
                    features = visual_encoder.adapter(features)
            token_chunks.append(visual_encoder.token_projection(features).cpu())
            if start == 0 or start + frames_per_batch >= len(unique_indices):
                print(
                    json.dumps(
                        {
                            'cache_stage': 'backbone',
                            'encoded_frames': min(start + frames_per_batch, len(unique_indices)),
                            'total_frames': len(unique_indices),
                            'elapsed_sec': round(time.time() - encode_started, 1),
                        }
                    ),
                    flush=True,
                )
    unique_tokens = torch.cat(token_chunks, dim=0)

    pooled_chunks: list[torch.Tensor] = []
    with torch.inference_mode():
        for start in range(0, len(window_specs), pool_windows_per_batch):
            specs = window_specs[start : start + pool_windows_per_batch]
            positions = torch.tensor(
                [[index_to_position[index] for index in indices] for _, indices, _ in specs],
                dtype=torch.long,
            )
            tokens = unique_tokens[positions].to(device)
            masks = torch.tensor([mask for _, _, mask in specs], dtype=torch.bool, device=device)
            pooled_chunks.append(visual_encoder.temporal_pool(tokens, masks).cpu())
    return torch.cat(pooled_chunks, dim=0).numpy().astype(np.float32)


def direct_window_features(
    visual_encoder: torch.nn.Module,
    frame_map: dict[int, np.ndarray],
    window_specs: list[tuple[str, list[int], list[bool]]],
    *,
    device: torch.device,
    clip_mean: torch.Tensor | None,
    clip_std: torch.Tensor | None,
) -> np.ndarray:
    videos: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []
    for _, indices, mask in window_specs:
        frames = torch.from_numpy(np.stack([frame_map[index] for index in indices], axis=0)).permute(0, 3, 1, 2)
        if clip_mean is not None and clip_std is not None:
            frames = (frames - clip_mean) / clip_std
        videos.append(frames)
        masks.append(torch.tensor(mask, dtype=torch.bool))
    with torch.inference_mode():
        features, _ = visual_encoder(torch.stack(videos).to(device), torch.stack(masks).to(device))
    return features.cpu().numpy().astype(np.float32)


def build_visual_cache(
    project_root: Path,
    stage_path: Path,
    checkpoint_path: Path,
    *,
    device: torch.device,
    frames_per_batch: int,
    overwrite: bool,
) -> dict[str, Any]:
    data_config, model_config, _, stage_config, config_sources = resolve_training_configs(project_root, stage_path)
    model_config = sync_tactile_model_config(data_config, model_config)
    cache_dir = resolve_path(project_root, data_config['visual_feature_cache_dir'])
    cache_dir.mkdir(parents=True, exist_ok=True)
    samples, windows = stable_val_rows(project_root, data_config, stage_config)
    model, load_info = load_visual_model(model_config, checkpoint_path, device)
    visual = model.visual_encoder
    clip_mean = None
    clip_std = None
    if data_config.get('clip_mean') is not None and data_config.get('clip_std') is not None:
        clip_mean = torch.tensor(data_config['clip_mean'], dtype=torch.float32).view(1, 3, 1, 1)
        clip_std = torch.tensor(data_config['clip_std'], dtype=torch.float32).view(1, 3, 1, 1)

    processed = 0
    reused = 0
    equivalence_max_abs_difference = 0.0
    started = time.time()
    for sample_number, sample in enumerate(samples.to_dict('records'), start=1):
        sample_id = str(sample['sample_id'])
        sample_windows = windows.loc[windows['sample_id'].astype(str).eq(sample_id)].sort_values('window_start')
        specs: list[tuple[str, list[int], list[bool]]] = []
        requested_indices: list[int] = []
        for window in sample_windows.to_dict('records'):
            sampled, mask = sample_frame_indices(json.loads(window['video_frame_indices_json']), int(data_config['num_frames_per_window']))
            specs.append((str(window['window_id']), sampled, mask))
            requested_indices.extend(sampled)
        expected_ids = np.asarray([window_id for window_id, _, _ in specs], dtype=np.str_)
        cache_path = cache_dir / f'{sample_id}.npz'
        if cache_path.exists() and not overwrite:
            try:
                existing = load_sample_visual_feature_cache(cache_path)
                valid = (
                    np.array_equal(existing['window_ids'], expected_ids)
                    and existing['visual_features'].shape == (len(specs), int(model_config['visual']['token_dim']))
                    and np.isfinite(existing['visual_features']).all()
                )
            except Exception:
                valid = False
            if valid:
                if sample_number == 1:
                    audit_specs = specs[:2]
                    audit_indices = [index for _, indices, _ in audit_specs for index in indices]
                    audit_frame_map = load_frames_by_indices(
                        video_path_for_sample(project_root, str(sample['video_path'])),
                        audit_indices,
                        image_size=int(data_config['image_size']),
                        roi=data_config.get('roi'),
                    )
                    direct = direct_window_features(
                        visual,
                        audit_frame_map,
                        audit_specs,
                        device=device,
                        clip_mean=clip_mean,
                        clip_std=clip_std,
                    )
                    equivalence_max_abs_difference = float(
                        np.max(np.abs(direct - existing['visual_features'][: len(audit_specs)]))
                    )
                    if equivalence_max_abs_difference > 2e-4:
                        raise RuntimeError(
                            f'Cached visual features differ from direct VisualEncoder output by '
                            f'{equivalence_max_abs_difference:.8g}.'
                        )
                    print(
                        json.dumps(
                            {
                                'cache_stage': 'direct_equivalence_audit',
                                'sample_id': sample_id,
                                'window_count': len(audit_specs),
                                'max_abs_difference': equivalence_max_abs_difference,
                            }
                        ),
                        flush=True,
                    )
                reused += 1
                print(json.dumps({'cache': 'reused', 'sample': sample_number, 'total': len(samples), 'sample_id': sample_id}), flush=True)
                continue

        frame_map = load_frames_by_indices(
            video_path_for_sample(project_root, str(sample['video_path'])),
            requested_indices,
            image_size=int(data_config['image_size']),
            roi=data_config.get('roi'),
        )
        print(
            json.dumps(
                {
                    'cache_stage': 'decoded',
                    'sample': sample_number,
                    'sample_id': sample_id,
                    'unique_frames': len(frame_map),
                    'elapsed_sec': round(time.time() - started, 1),
                }
            ),
            flush=True,
        )
        features = encode_unique_frame_windows(
            visual,
            frame_map,
            specs,
            device=device,
            clip_mean=clip_mean,
            clip_std=clip_std,
            frames_per_batch=max(1, frames_per_batch),
            pool_windows_per_batch=1024,
        )
        if sample_number == 1:
            direct = direct_window_features(
                visual,
                frame_map,
                specs[:2],
                device=device,
                clip_mean=clip_mean,
                clip_std=clip_std,
            )
            equivalence_max_abs_difference = float(np.max(np.abs(direct - features[:2])))
            if equivalence_max_abs_difference > 2e-4:
                raise RuntimeError(
                    f'Unique-frame cache optimization differs from direct VisualEncoder output by '
                    f'{equivalence_max_abs_difference:.8g}.'
                )
        write_sample_visual_feature_cache(project_root, cache_dir, sample_id, expected_ids.tolist(), features)
        processed += 1
        print(
            json.dumps(
                {
                    'cache': 'written',
                    'sample': sample_number,
                    'total': len(samples),
                    'sample_id': sample_id,
                    'direction': sample['direction'],
                    'windows': len(specs),
                    'unique_frames': len(frame_map),
                    'elapsed_sec': round(time.time() - started, 1),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    cache_files = [cache_dir / f'{sample_id}.npz' for sample_id in samples['sample_id'].astype(str)]
    missing = [str(path) for path in cache_files if not path.exists()]
    if missing:
        raise RuntimeError(f'M5-0 visual cache is incomplete: {missing[:3]}')
    manifest = {
        'phase': 'Model Phase M5-0 E0',
        'artifact': 'canonical_checkpoint_visual_feature_cache',
        'cache_dir': str(cache_dir.resolve()),
        'subset': 'val',
        'only_stable': True,
        'sample_count': int(len(samples)),
        'window_count': int(len(windows)),
        'direction_sample_counts': {str(k): int(v) for k, v in samples.groupby('direction').size().items()},
        'direction_window_counts': {str(k): int(v) for k, v in windows.groupby('direction').size().items()},
        'feature_dim': int(model_config['visual']['token_dim']),
        'num_frames_per_window': int(data_config['num_frames_per_window']),
        'image_size': int(data_config['image_size']),
        'roi': data_config.get('roi'),
        'encoding': {
            'dtype': 'float32',
            'amp_enabled': False,
            'unique_frame_backbone_reuse': True,
            'direct_equivalence_max_abs_difference': equivalence_max_abs_difference,
            'direct_equivalence_tolerance': 2e-4,
        },
        'checkpoint': {
            'path': str(checkpoint_path.resolve()),
            'sha256': load_info['checkpoint_sha256'],
            'loaded_tensor_count': load_info['loaded_tensor_count'],
            'target_tensor_count': load_info['target_tensor_count'],
            'missing_keys': load_info['missing_keys'],
        },
        'sources': config_sources,
        'samples_path': str(resolve_path(project_root, data_config['samples_path'])),
        'samples_sha256': sha256_file(resolve_path(project_root, data_config['samples_path'])),
        'windows_path': str(resolve_path(project_root, data_config['windows_path'])),
        'windows_sha256': sha256_file(resolve_path(project_root, data_config['windows_path'])),
        'split_path': str(resolve_path(project_root, stage_config['split'])),
        'split_sha256': sha256_file(resolve_path(project_root, stage_config['split'])),
        'cache_file_count': len(cache_files),
        'cache_files_aggregate_sha256': aggregate_file_sha256(cache_files),
        'processed_sample_count': processed,
        'reused_sample_count': reused,
        'elapsed_sec': round(time.time() - started, 3),
    }
    json_write(cache_dir / 'manifest.json', manifest)
    del model
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return manifest


class DirectionSubset(Dataset):
    def __init__(self, base: Dataset, indices: list[int]) -> None:
        self.base = base
        self.indices = indices
        self.physical_attribute_stats = getattr(base, 'physical_attribute_stats', None)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.base[self.indices[index]]


def make_loader(dataset: Dataset, train_config: dict[str, Any]) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=int(train_config['batch_size']),
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        collate_fn=sequence_collate_fn,
    )


def direction_zero_effect(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> dict[str, Any]:
    batch = move_to_device(next(iter(loader)), device)
    with torch.inference_mode():
        w2a_batch = dict(batch)
        a2w_batch = dict(batch)
        w2a_batch['direction_ids'] = torch.zeros_like(batch['direction_ids'])
        a2w_batch['direction_ids'] = torch.ones_like(batch['direction_ids'])
        w2a = model(w2a_batch)
        a2w = model(a2w_batch)
    differences = {
        key: float((w2a[key] - a2w[key]).abs().max().item())
        for key in CORE_OUTPUT_KEYS
        if key in w2a and key in a2w
    }
    maximum = max(differences.values(), default=0.0)
    if maximum != 0.0:
        raise RuntimeError(f'E0 direction adapters have a non-zero initial effect: {differences}')
    return {'passed': True, 'max_abs_difference': maximum, 'by_output': differences}


def reference_only_baseline(loader: DataLoader) -> dict[str, Any]:
    totals = {key: {'abs': 0.0, 'signed': 0.0, 'count': 0} for key in ('w2a', 'a2w')}
    for batch in loader:
        target = batch['finger_control_force_targets']
        reference = batch['finger_reference_forces']
        mask = (
            batch['window_mask'].unsqueeze(-1)
            & batch['delta_supervision_masks'].unsqueeze(-1)
            & batch['has_finger_control_target']
            & batch['has_finger_reference']
            & torch.isfinite(target)
            & torch.isfinite(reference)
        )
        for direction, direction_index in (('w2a', 0), ('a2w', 1)):
            direction_mask = mask & (batch['direction_ids'] == direction_index)[:, None, None]
            error = reference[direction_mask] - target[direction_mask]
            totals[direction]['abs'] += float(error.abs().sum().item())
            totals[direction]['signed'] += float(error.sum().item())
            totals[direction]['count'] += int(error.numel())
    by_direction = {
        direction: {
            'finger_control_interface_mae': values['abs'] / max(1, values['count']),
            'finger_control_interface_bias': values['signed'] / max(1, values['count']),
            'finger_control_interface_count': values['count'],
        }
        for direction, values in totals.items()
    }
    present = [values['finger_control_interface_mae'] for values in by_direction.values() if values['finger_control_interface_count']]
    return {
        'definition': 'finger_reference_forces used directly as the control-force prediction on delta-supervised interface observations',
        'by_direction': by_direction,
        'finger_control_interface_mae_macro_direction': sum(present) / max(1, len(present)),
        'direction_macro_complete': len(present) == 2,
    }


def evaluate_e0(
    project_root: Path,
    stage_path: Path,
    checkpoint_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    context = prepare_evaluation_context(
        project_root=project_root,
        stage=stage_path,
        checkpoint=checkpoint_path,
        subset='val',
        only_stable=True,
        num_workers=0,
        use_archived_run_config=False,
        allow_lora_injection=True,
        allowed_missing_prefixes=DIRECTION_PREFIXES,
        initialize_pretrained_backbone=False,
    )
    dataset = context['dataset']
    records = dataset.sample_records
    direction_indices = {
        direction: [index for index, record in enumerate(records) if str(record['direction']).upper() == direction]
        for direction in ('W2A', 'A2W')
    }
    if not all(direction_indices.values()):
        raise RuntimeError(f'E0 requires both directions in Val: {direction_indices}')
    loaders = {
        'bidirectional': context['loader'],
        'w2a': make_loader(DirectionSubset(dataset, direction_indices['W2A']), context['train_config']),
        'a2w': make_loader(DirectionSubset(dataset, direction_indices['A2W']), context['train_config']),
    }
    zero_effect = direction_zero_effect(context['model'], loaders['bidirectional'], context['device'])
    reference = reference_only_baseline(loaders['bidirectional'])

    provenance = {
        'phase': 'Model Phase M5-0 E0',
        'stage_name': context['stage_config']['name'],
        'stage_path': str(stage_path.resolve()),
        'subset': 'val',
        'only_stable': True,
        'use_archived_run_config': False,
        'test_evaluated': False,
        'checkpoint': {
            'path': str(checkpoint_path.resolve()),
            'sha256': context['load_info']['checkpoint_sha256'],
            'loaded_tensor_count': context['load_info']['loaded_tensor_count'],
            'target_tensor_count': context['load_info']['target_tensor_count'],
            'missing_keys': context['load_info']['missing_keys'],
        },
        'sources': resolve_training_configs(project_root, stage_path)[4],
        'data': {
            'schema_version': context['data_config'].get('schema_version'),
            'dataset_version': context['data_config'].get('dataset_version'),
            'split_version': context['data_config'].get('split_version'),
            'samples_path': str(resolve_path(project_root, context['data_config']['samples_path'])),
            'samples_sha256': sha256_file(resolve_path(project_root, context['data_config']['samples_path'])),
            'windows_path': str(resolve_path(project_root, context['data_config']['windows_path'])),
            'windows_sha256': sha256_file(resolve_path(project_root, context['data_config']['windows_path'])),
            'split_path': str(resolve_path(project_root, context['stage_config']['split'])),
            'split_sha256': sha256_file(resolve_path(project_root, context['stage_config']['split'])),
            'visual_cache_manifest_path': str(resolve_path(project_root, context['data_config']['visual_feature_cache_dir']) / 'manifest.json'),
            'visual_cache_manifest_sha256': sha256_file(resolve_path(project_root, context['data_config']['visual_feature_cache_dir']) / 'manifest.json'),
        },
        'cohort': {
            'sample_count': len(records),
            'window_count': int(len(dataset.windows)),
            'direction_sample_counts': {direction: len(indices) for direction, indices in direction_indices.items()},
            'direction_window_counts': {
                str(key): int(value) for key, value in dataset.windows.groupby('direction').size().items()
            },
            'object_ids': sorted(dataset.samples['object_id'].astype(str).unique().tolist()),
        },
        'initialization_audit': {
            'direction_zero_effect': zero_effect,
            'allowed_missing_prefixes': list(DIRECTION_PREFIXES),
            'disallowed_missing_keys': context['load_info']['disallowed_missing_keys'],
            'disallowed_unexpected_keys': context['load_info']['disallowed_unexpected_keys'],
        },
        'reference_only_baseline': reference,
    }

    reports: dict[str, Any] = {}
    for name in ('w2a', 'a2w', 'bidirectional'):
        print(json.dumps({'evaluation': 'started', 'cohort': name, 'sample_count': len(loaders[name].dataset)}), flush=True)
        started = time.time()
        metrics = run_model_epoch(
            context['model'],
            loaders[name],
            device=context['device'],
            training=False,
            stage_config=context['stage_config'],
            model_config=context['model_config'],
            phase_class_weights=context['phase_class_weights'],
            amp_enabled=False,
            loss_reduction_config=context['train_config'].get('loss_reduction'),
        )
        report = {
            **provenance,
            'evaluation_cohort': name,
            'evaluation_sample_count': len(loaders[name].dataset),
            'elapsed_sec': round(time.time() - started, 3),
            'metrics': metrics,
        }
        report_path = output_dir / f'e0_{name}_val.json'
        json_write(report_path, report)
        reports[name] = {'path': str(report_path.resolve()), 'sha256': sha256_file(report_path), 'metrics': metrics}
        print(
            json.dumps(
                {
                    'evaluation': 'completed',
                    'cohort': name,
                    'elapsed_sec': report['elapsed_sec'],
                    'medium_f1_interface': metrics['medium_f1_interface'],
                    'finger_control_interface_mae': metrics['finger_control_interface_mae'],
                }
            ),
            flush=True,
        )

    bidirectional = reports['bidirectional']['metrics']
    policy_baseline = float(bidirectional['finger_control_interface_mae_w2a'])
    medium_baseline = float(bidirectional['medium_f1_interface_w2a'])
    summary = {
        **provenance,
        'status': 'passed',
        'reports': {
            name: {'path': value['path'], 'sha256': value['sha256']}
            for name, value in reports.items()
        },
        'key_metrics': {
            'w2a': {
                'medium_f1_interface': bidirectional['medium_f1_interface_w2a'],
                'finger_control_interface_mae': bidirectional['finger_control_interface_mae_w2a'],
                'finger_delta_interface_mae': bidirectional['finger_delta_interface_mae_w2a'],
                'sample_count': bidirectional['sample_count_w2a'],
            },
            'a2w_zero_shot': {
                'medium_f1_interface': bidirectional['medium_f1_interface_a2w'],
                'finger_control_interface_mae': bidirectional['finger_control_interface_mae_a2w'],
                'finger_delta_interface_mae': bidirectional['finger_delta_interface_mae_a2w'],
                'sample_count': bidirectional['sample_count_a2w'],
            },
            'direction_macro': {
                'medium_f1_interface': bidirectional['medium_f1_interface_macro_direction'],
                'finger_control_interface_mae': bidirectional['finger_control_interface_mae_macro_direction'],
                'finger_delta_interface_mae': bidirectional['finger_delta_interface_mae_macro_direction'],
                'complete': bidirectional['direction_macro_complete'],
            },
        },
        'proposed_w2a_retention_guards': {
            'policy': {
                'metric': 'finger_control_interface_mae_w2a',
                'mode': 'min',
                'baseline_value': policy_baseline,
                'max_relative_degradation': 0.05,
                'threshold': policy_baseline * 1.05,
            },
            'medium': {
                'metric': 'medium_f1_interface_w2a',
                'mode': 'max',
                'baseline_value': medium_baseline,
                'max_relative_degradation': 0.05,
                'threshold': medium_baseline * 0.95,
            },
        },
    }
    summary_path = output_dir / 'e0_summary.json'
    json_write(summary_path, summary)
    summary['summary_path'] = str(summary_path.resolve())
    summary['summary_sha256'] = sha256_file(summary_path)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description='Run Model Phase M5-0 E0 on stable-only bidirectional Val.')
    parser.add_argument('--project-root', default='.')
    parser.add_argument('--stage', default='configs/stages/stage39e0_warmstart_baseline.yaml')
    parser.add_argument('--checkpoint', default='runs/stage38j_f20_v3_causal/checkpoints/best.pt')
    parser.add_argument('--output-dir', default='evals/stage39e0_warmstart_baseline')
    parser.add_argument('--device', default=None)
    parser.add_argument('--frames-per-batch', type=int, default=64)
    parser.add_argument('--overwrite-cache', action='store_true')
    parser.add_argument('--skip-cache', action='store_true')
    parser.add_argument('--cache-only', action='store_true')
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    stage_path = resolve_path(project_root, args.stage)
    checkpoint_path = resolve_path(project_root, args.checkpoint)
    output_dir = resolve_path(project_root, args.output_dir)
    device = torch.device(args.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)
    if not args.skip_cache:
        manifest = build_visual_cache(
            project_root,
            stage_path,
            checkpoint_path,
            device=device,
            frames_per_batch=args.frames_per_batch,
            overwrite=args.overwrite_cache,
        )
        print(json.dumps({'cache_manifest': str((Path(manifest['cache_dir']) / 'manifest.json').resolve())}), flush=True)
        if args.cache_only:
            return
    summary = evaluate_e0(project_root, stage_path, checkpoint_path, output_dir)
    stats_path = project_root / 'data/processed/stats/model_phase_m5_e0_validation.json'
    json_write(stats_path, summary)
    print(json.dumps({'status': summary['status'], 'summary_path': summary['summary_path'], 'stats_path': str(stats_path.resolve())}), flush=True)


if __name__ == '__main__':
    main()
