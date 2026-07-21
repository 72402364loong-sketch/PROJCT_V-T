from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import torch
import yaml
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from cmg.config import sha256_file
from cmg.direction_metrics import DirectionMetricAccumulator
from cmg.attribute_metrics import (
    AttributeAggregationAccumulator,
    AttributeWindowMetricAccumulator,
    attribute_class_counts,
    flatten_aggregation_metrics,
    flatten_window_metrics,
)
from cmg.losses import compute_losses
from cmg.physical_metrics import PhysicalAttributeMetricAccumulator, flatten_physical_metrics

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:  # pragma: no cover
    SummaryWriter = None


def move_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def accuracy_from_confusion(confusion: torch.Tensor) -> float:
    total = confusion.sum().item()
    if total == 0:
        return 0.0
    return float(torch.diag(confusion).sum().item() / total)


def class_f1_from_confusion(confusion: torch.Tensor, index: int) -> float:
    tp = confusion[index, index].item()
    fp = confusion[:, index].sum().item() - tp
    fn = confusion[index, :].sum().item() - tp
    denom = (2 * tp + fp + fn)
    return 0.0 if denom == 0 else float((2 * tp) / denom)


def macro_f1_from_confusion(confusion: torch.Tensor) -> float:
    scores = [class_f1_from_confusion(confusion, index) for index in range(confusion.shape[0])]
    return float(sum(scores) / len(scores))


def resolve_attr_metric_tasks(model_config: dict[str, Any]) -> list[str]:
    attributes_config = model_config.get('attributes', {}) if isinstance(model_config.get('attributes', {}), dict) else {}
    raw_tasks = attributes_config.get('metric_tasks', ['fragility', 'geometry', 'surface'])
    if isinstance(raw_tasks, str):
        raw_tasks = [raw_tasks]
    tasks = [str(task).strip().lower() for task in raw_tasks if str(task).strip()]
    valid_tasks = [task for task in tasks if task in {'fragility', 'geometry', 'surface'}]
    return valid_tasks or ['fragility', 'geometry', 'surface']


def average_selected_attr_f1(scores: dict[str, float], selected_tasks: list[str]) -> float:
    selected = [float(scores[task]) for task in selected_tasks if task in scores]
    return float(sum(selected) / max(1, len(selected)))


def build_scheduler(
    optimizer: AdamW,
    total_steps: int,
    warmup_ratio: float,
    min_lr_scale: float,
) -> LambdaLR:
    warmup_steps = int(round(total_steps * warmup_ratio))
    warmup_steps = min(max(0, warmup_steps), total_steps)

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        if total_steps <= warmup_steps:
            return min_lr_scale
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_scale + (1.0 - min_lr_scale) * cosine

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


def build_phase_class_weights(config: dict[str, Any], device: torch.device) -> torch.Tensor:
    return torch.tensor(
        [
            float(config['phase_class_weights']['Water']),
            float(config['phase_class_weights']['Interface']),
            float(config['phase_class_weights']['Air']),
        ],
        dtype=torch.float32,
        device=device,
    )


def load_checkpoint_context(path: str | Path) -> dict[str, Any]:
    checkpoint_path = Path(path).resolve()
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    run_dir_value = checkpoint.get('run_dir') if isinstance(checkpoint, dict) else None
    run_dir = Path(run_dir_value) if run_dir_value else None
    run_config_path_value = checkpoint.get('run_config_path') if isinstance(checkpoint, dict) else None

    run_config_candidates: list[Path] = []
    if run_config_path_value:
        run_config_candidates.append(Path(run_config_path_value))
    if run_dir is not None:
        run_config_candidates.append(run_dir / 'run_config.yaml')
    if checkpoint_path.parent.name == 'checkpoints':
        run_config_candidates.append(checkpoint_path.parent.parent / 'run_config.yaml')

    run_config_path = None
    for candidate in run_config_candidates:
        if candidate.exists():
            run_config_path = candidate
            break

    run_config = None
    if run_config_path is not None:
        with run_config_path.open('r', encoding='utf-8') as handle:
            run_config = yaml.safe_load(handle) or {}

    return {
        'checkpoint_path': str(checkpoint_path),
        'checkpoint_sha256': sha256_file(checkpoint_path),
        'checkpoint': checkpoint,
        'stage_name': checkpoint.get('stage_name') if isinstance(checkpoint, dict) else None,
        'run_dir': str(run_dir.resolve()) if run_dir is not None else None,
        'run_config_path': str(run_config_path.resolve()) if run_config_path is not None else None,
        'run_config': run_config,
    }


def remap_open_clip_lora_keys(
    state_dict: dict[str, torch.Tensor],
    target_state_keys: set[str],
) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    remapped: dict[str, torch.Tensor] = {}
    remap_log: dict[str, str] = {}
    for key, value in state_dict.items():
        candidate = key
        if '.attn.' in key and '.attn.base_attn.' not in key:
            maybe = key.replace('.attn.', '.attn.base_attn.')
            if maybe in target_state_keys:
                candidate = maybe
        remapped[candidate] = value
        if candidate != key:
            remap_log[key] = candidate
    return remapped, remap_log



def is_allowed_lora_injection_mismatch(key: str) -> bool:
    return (
        'lora_' in key
        or key.startswith('policy_head.input_layer.')
        or key.startswith('policy_head.hidden_layer.')
        or key.startswith('policy_head.output_layer.')
        or key.startswith('policy_head.interface_output_layer.')
        or key.startswith('policy_head.base_')
        or key.startswith('policy_head.residual_')
        or key.startswith('policy_head.finger_')
        or key.startswith('content_encoder.conv.')
    )


def key_matches_prefix(key: str, prefix: str) -> bool:
    normalized = str(prefix).strip()
    if normalized.endswith('*'):
        normalized = normalized[:-1]
    if not normalized:
        return False
    return key == normalized.rstrip('.') or key.startswith(normalized)


def state_dict_module_report(
    target_state: dict[str, torch.Tensor],
    loaded_keys: list[str],
) -> dict[str, dict[str, int]]:
    report: dict[str, dict[str, int]] = {}
    loaded = set(loaded_keys)
    for key, value in target_state.items():
        module_name = key.split('.', 1)[0]
        module = report.setdefault(
            module_name,
            {
                'target_tensor_count': 0,
                'target_value_count': 0,
                'loaded_tensor_count': 0,
                'loaded_value_count': 0,
            },
        )
        value_count = int(value.numel())
        module['target_tensor_count'] += 1
        module['target_value_count'] += value_count
        if key in loaded:
            module['loaded_tensor_count'] += 1
            module['loaded_value_count'] += value_count
    return report



def load_model_weights(
    model: nn.Module,
    path: str | Path,
    *,
    strict: bool = True,
    allow_lora_injection: bool = False,
    allowed_missing_prefixes: list[str] | tuple[str, ...] | None = None,
) -> dict[str, Any]:
    context = load_checkpoint_context(path)
    checkpoint = context['checkpoint']
    state_dict = checkpoint['model'] if isinstance(checkpoint, dict) and 'model' in checkpoint else checkpoint
    if not isinstance(state_dict, dict) or not all(isinstance(key, str) for key in state_dict):
        raise RuntimeError(f'Checkpoint {context["checkpoint_path"]} does not contain a valid model state_dict.')
    target_state = model.state_dict()
    remapped_keys: dict[str, str] = {}
    if allow_lora_injection:
        state_dict, remapped_keys = remap_open_clip_lora_keys(state_dict, set(target_state.keys()))

    shape_mismatches = [
        {
            'key': key,
            'checkpoint_shape': list(value.shape),
            'model_shape': list(target_state[key].shape),
        }
        for key, value in state_dict.items()
        if key in target_state and tuple(value.shape) != tuple(target_state[key].shape)
    ]
    if shape_mismatches:
        details = {
            'checkpoint_path': context['checkpoint_path'],
            'shape_mismatches': shape_mismatches,
            'remapped_keys': remapped_keys,
        }
        raise RuntimeError(
            'Checkpoint initialization encountered tensor shape mismatches: '
            + json.dumps(details, ensure_ascii=False)
        )

    missing_keys = sorted(set(target_state) - set(state_dict))
    unexpected_keys = sorted(set(state_dict) - set(target_state))
    explicit_missing_prefixes = tuple(str(prefix) for prefix in (allowed_missing_prefixes or ()))

    def allowed_missing(key: str) -> bool:
        return (
            any(key_matches_prefix(key, prefix) for prefix in explicit_missing_prefixes)
            or (allow_lora_injection and is_allowed_lora_injection_mismatch(key))
        )

    def allowed_unexpected(key: str) -> bool:
        return allow_lora_injection and is_allowed_lora_injection_mismatch(key)

    disallowed_missing = [key for key in missing_keys if not allowed_missing(key)]
    disallowed_unexpected = [key for key in unexpected_keys if not allowed_unexpected(key)]
    if (strict and (disallowed_missing or disallowed_unexpected)) or (
        not strict and explicit_missing_prefixes and disallowed_missing
    ):
        details = {
            'checkpoint_path': context['checkpoint_path'],
            'missing_keys': missing_keys,
            'unexpected_keys': unexpected_keys,
            'allowed_missing_prefixes': list(explicit_missing_prefixes),
            'disallowed_missing_keys': disallowed_missing,
            'disallowed_unexpected_keys': disallowed_unexpected,
            'remapped_keys': remapped_keys,
        }
        raise RuntimeError(
            'Checkpoint initialization encountered keys beyond the explicitly allowed architecture delta: '
            + json.dumps(details, ensure_ascii=False)
        )

    compatible_state = {key: value for key, value in state_dict.items() if key in target_state}
    incompatible = model.load_state_dict(compatible_state, strict=False)
    loaded_keys = sorted(compatible_state)
    if sorted(incompatible.missing_keys) != missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            'Internal checkpoint compatibility audit disagrees with torch.load_state_dict: '
            + json.dumps(
                {
                    'audited_missing_keys': missing_keys,
                    'torch_missing_keys': sorted(incompatible.missing_keys),
                    'torch_unexpected_keys': sorted(incompatible.unexpected_keys),
                },
                ensure_ascii=False,
            )
        )
    return {
        'checkpoint_path': context['checkpoint_path'],
        'checkpoint_sha256': context['checkpoint_sha256'],
        'missing_keys': missing_keys,
        'unexpected_keys': unexpected_keys,
        'allowed_missing_prefixes': list(explicit_missing_prefixes),
        'disallowed_missing_keys': disallowed_missing,
        'disallowed_unexpected_keys': disallowed_unexpected,
        'remapped_keys': remapped_keys,
        'allow_lora_injection': allow_lora_injection,
        'loaded_tensor_count': len(loaded_keys),
        'loaded_value_count': int(sum(target_state[key].numel() for key in loaded_keys)),
        'target_tensor_count': len(target_state),
        'target_value_count': int(sum(value.numel() for value in target_state.values())),
        'module_load_report': state_dict_module_report(target_state, loaded_keys),
        'stage_name': context['stage_name'],
        'run_dir': context['run_dir'],
        'run_config_path': context['run_config_path'],
    }

def resolve_module(root: nn.Module, module_name: str) -> nn.Module | None:
    current: nn.Module | None = root
    for part in module_name.split('.'):
        if current is None or not hasattr(current, part):
            return None
        current = getattr(current, part)
    return current


def freeze_modules(
    model: nn.Module,
    module_names: list[str],
    *,
    strict: bool = True,
    lock_eval: bool = True,
    train_mode_module_names: list[str] | tuple[str, ...] | None = None,
) -> dict[str, Any]:
    normalized_names = list(dict.fromkeys(str(name).strip() for name in module_names if str(name).strip()))
    normalized_train_mode_names = list(
        dict.fromkeys(str(name).strip() for name in (train_mode_module_names or ()) if str(name).strip())
    )
    resolved = {name: resolve_module(model, name) for name in normalized_names}
    missing_names = [name for name, module in resolved.items() if module is None]
    invalid_train_mode_names = [
        name for name in normalized_train_mode_names if name not in resolved or resolved[name] is None
    ]
    if strict and missing_names:
        raise ValueError(f'Unknown freeze_modules paths: {missing_names}.')
    if invalid_train_mode_names:
        raise ValueError(
            'frozen train-mode modules must also be listed in freeze_modules: '
            f'{invalid_train_mode_names}.'
        )

    frozen_parameter_ids: set[int] = set()
    for name, module in resolved.items():
        if module is None:
            continue
        for parameter in module.parameters():
            parameter.requires_grad = False
            frozen_parameter_ids.add(id(parameter))
        if lock_eval and name not in normalized_train_mode_names:
            module.eval()

    locked_names = [
        name
        for name, module in resolved.items()
        if module is not None and lock_eval and name not in normalized_train_mode_names
    ]
    setattr(model, '_cmg_frozen_eval_module_names', tuple(locked_names))
    setattr(model, '_cmg_frozen_train_mode_module_names', tuple(normalized_train_mode_names))
    parameter_names = [
        name
        for name, parameter in model.named_parameters()
        if id(parameter) in frozen_parameter_ids
    ]
    return {
        'requested_module_names': normalized_names,
        'resolved_module_names': [name for name, module in resolved.items() if module is not None],
        'missing_module_names': missing_names,
        'lock_eval': bool(lock_eval),
        'eval_locked_module_names': locked_names,
        'train_mode_module_names': normalized_train_mode_names,
        'parameter_names': parameter_names,
        'parameter_tensor_count': len(parameter_names),
        'parameter_value_count': int(
            sum(parameter.numel() for parameter in model.parameters() if id(parameter) in frozen_parameter_ids)
        ),
    }


def enforce_frozen_modules_eval(model: nn.Module) -> None:
    for module_name in getattr(model, '_cmg_frozen_eval_module_names', ()):
        module = resolve_module(model, str(module_name))
        if module is None:
            raise RuntimeError(f'Frozen eval-lock module disappeared after setup: {module_name!r}.')
        module.eval()
    if model.training:
        for module_name in getattr(model, '_cmg_frozen_train_mode_module_names', ()):
            module = resolve_module(model, str(module_name))
            if module is None:
                raise RuntimeError(f'Frozen train-mode module disappeared after setup: {module_name!r}.')
            module.train()


def audit_model_parameters(model: nn.Module) -> dict[str, Any]:
    trainable_names: list[str] = []
    frozen_names: list[str] = []
    trainable_values = 0
    frozen_values = 0
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            trainable_names.append(name)
            trainable_values += int(parameter.numel())
        else:
            frozen_names.append(name)
            frozen_values += int(parameter.numel())
    total_values = trainable_values + frozen_values
    return {
        'trainable_parameter_names': trainable_names,
        'frozen_parameter_names': frozen_names,
        'trainable_tensor_count': len(trainable_names),
        'frozen_tensor_count': len(frozen_names),
        'trainable_value_count': trainable_values,
        'frozen_value_count': frozen_values,
        'total_value_count': total_values,
        'trainable_fraction': float(trainable_values / total_values) if total_values else 0.0,
    }


def build_optimizer(model: nn.Module, config: dict[str, Any]) -> AdamW:
    base_lr = float(config.get('base_learning_rate', config.get('learning_rate', 1e-4)))
    lora_lr = float(config.get('lora_learning_rate', base_lr))
    weight_decay = float(config['weight_decay'])
    raw_multipliers = config.get('module_learning_rate_multipliers', {})
    module_learning_rate_multipliers = {
        str(prefix): float(multiplier)
        for prefix, multiplier in raw_multipliers.items()
    } if isinstance(raw_multipliers, dict) else {}

    base_params = []
    scaled_base_groups: dict[str, dict[str, Any]] = {}
    lora_params = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if 'lora_' in name:
            lora_params.append(parameter)
        else:
            matched_prefix = None
            for prefix in module_learning_rate_multipliers:
                if name.startswith(prefix) and (matched_prefix is None or len(prefix) > len(matched_prefix)):
                    matched_prefix = prefix
            if matched_prefix is None:
                base_params.append(parameter)
                continue
            multiplier = module_learning_rate_multipliers[matched_prefix]
            group_name = 'base_' + matched_prefix.replace('.', '_')
            group = scaled_base_groups.setdefault(
                group_name,
                {
                    'params': [],
                    'lr': base_lr * multiplier,
                    'name': group_name,
                },
            )
            group['params'].append(parameter)

    param_groups: list[dict[str, Any]] = []
    if base_params:
        param_groups.append({'params': base_params, 'lr': base_lr, 'name': 'base'})
    for group_name in sorted(scaled_base_groups):
        param_groups.append(scaled_base_groups[group_name])
    if lora_params:
        param_groups.append({'params': lora_params, 'lr': lora_lr, 'name': 'lora'})
    if not param_groups:
        raise RuntimeError('No trainable parameters found for optimizer setup.')
    return AdamW(param_groups, weight_decay=weight_decay)


def audit_optimizer_groups(model: nn.Module, optimizer: AdamW) -> list[dict[str, Any]]:
    parameter_names = {id(parameter): name for name, parameter in model.named_parameters()}
    reports: list[dict[str, Any]] = []
    for index, group in enumerate(optimizer.param_groups):
        names = [parameter_names.get(id(parameter), '<unnamed>') for parameter in group['params']]
        reports.append(
            {
                'index': index,
                'name': str(group.get('name', f'group{index}')),
                'learning_rate': float(group['lr']),
                'parameter_names': names,
                'parameter_tensor_count': len(names),
                'parameter_value_count': int(sum(parameter.numel() for parameter in group['params'])),
            }
        )
    return reports


def numeric_value(metrics: dict[str, Any], metric_name: str, epoch: int) -> float:
    if metric_name == 'epoch':
        return float(epoch)
    value = metrics.get(metric_name)
    if value is None:
        raise KeyError(f'Metric {metric_name!r} is not available for tie-break comparison.')
    return float(value)


def compare_metric(current: float, best: float, mode: str, eps: float = 1e-12) -> int:
    if mode == 'max':
        if current > best + eps:
            return 1
        if current < best - eps:
            return -1
        return 0
    if current < best - eps:
        return 1
    if current > best + eps:
        return -1
    return 0


def evaluate_w2a_retention_guard(
    metrics: dict[str, Any],
    config: dict[str, Any] | None,
) -> dict[str, Any]:
    guard = config if isinstance(config, dict) else {}
    enabled = bool(guard.get('enabled', False))
    if not enabled:
        return {'enabled': False, 'passed': True}
    metric_name = str(guard.get('metric', '')).strip()
    if not metric_name:
        raise RuntimeError('Enabled W2A retention guard requires a metric name.')
    if not metric_name.endswith('_w2a'):
        raise RuntimeError('W2A retention guard metric must be an explicit *_w2a metric.')
    if metric_name not in metrics:
        raise RuntimeError(f'W2A retention guard metric {metric_name!r} is missing from validation metrics.')
    default_count_metrics = {
        'medium_accuracy_w2a': 'medium_window_count_w2a',
        'medium_macro_f1_w2a': 'medium_window_count_w2a',
        'medium_f1_water_w2a': 'medium_window_count_w2a',
        'medium_f1_interface_w2a': 'medium_window_count_w2a',
        'medium_f1_air_w2a': 'medium_window_count_w2a',
        'finger_control_interface_mae_w2a': 'finger_control_interface_count_w2a',
        'finger_delta_interface_mae_w2a': 'finger_delta_interface_count_w2a',
        'finger_large_delta_wrong_sign_rate_w2a': 'finger_large_delta_count_w2a',
        'stable_leakage_mean_w2a': 'stable_leakage_count_w2a',
        'stable_leakage_p95_w2a': 'stable_leakage_count_w2a',
    }
    count_metric = str(guard.get('count_metric', default_count_metrics.get(metric_name, ''))).strip()
    if count_metric:
        if count_metric not in metrics:
            raise RuntimeError(f'W2A retention guard count metric {count_metric!r} is missing.')
        if int(metrics[count_metric]) <= 0:
            raise RuntimeError(f'W2A retention guard metric {metric_name!r} has no valid W2A observations.')
    if 'baseline_value' not in guard:
        raise RuntimeError('Enabled W2A retention guard requires E0 baseline_value.')
    current = float(metrics[metric_name])
    baseline = float(guard['baseline_value'])
    if not math.isfinite(current) or not math.isfinite(baseline):
        raise RuntimeError('W2A retention guard current and baseline values must be finite.')
    if baseline < 0.0:
        raise RuntimeError('W2A retention guard baseline_value must be non-negative.')
    mode = str(guard.get('mode', 'min')).strip().lower()
    if mode not in {'min', 'max'}:
        raise RuntimeError("W2A retention guard mode must be 'min' or 'max'.")
    tolerance = float(guard.get('max_relative_degradation', 0.05))
    absolute_tolerance = float(guard.get('absolute_tolerance', 0.0))
    if tolerance < 0.0 or absolute_tolerance < 0.0:
        raise RuntimeError('W2A retention guard tolerances must be non-negative.')
    if mode == 'min':
        threshold = baseline * (1.0 + tolerance) + absolute_tolerance
        passed = current <= threshold + 1e-12
        degradation = current - baseline
    else:
        threshold = baseline * (1.0 - tolerance) - absolute_tolerance
        passed = current >= threshold - 1e-12
        degradation = baseline - current
    relative_degradation = degradation / abs(baseline) if baseline != 0.0 else None
    return {
        'enabled': True,
        'passed': bool(passed),
        'metric': metric_name,
        'mode': mode,
        'current_value': current,
        'baseline_value': baseline,
        'threshold': threshold,
        'max_relative_degradation': tolerance,
        'absolute_tolerance': absolute_tolerance,
        'relative_degradation': relative_degradation,
    }


def collect_sampler_epoch_metrics(loader: DataLoader) -> dict[str, Any]:
    """Expose direction-aware sampler coverage without coupling Trainer to its class."""
    batch_sampler = getattr(loader, 'batch_sampler', None)
    summary = getattr(batch_sampler, 'last_epoch_summary', None)
    if not isinstance(summary, dict) or not summary:
        return {}
    return {
        'sampler_balance_mode': str(summary.get('balance_mode', '')),
        'sampler_batch_count': int(summary.get('batch_count', 0)),
        'sampler_total_draws': int(summary.get('total_draws', 0)),
        'sampler_unique_draws': int(summary.get('unique_draws', 0)),
        'sampler_repeated_draws': int(summary.get('repeated_draws', 0)),
        'sampler_repeat_rate': float(summary.get('repeat_rate', 0.0)),
        'sampler_direction_draw_counts': dict(summary.get('direction_draw_counts', {})),
        'sampler_object_draw_counts': dict(summary.get('object_draw_counts', {})),
    }


def run_model_epoch(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    training: bool,
    stage_config: dict[str, Any],
    model_config: dict[str, Any],
    phase_class_weights: torch.Tensor,
    amp_enabled: bool = False,
    optimizer: AdamW | None = None,
    scheduler: LambdaLR | None = None,
    scaler: torch.amp.GradScaler | None = None,
    grad_clip_norm: float = 0.0,
    loss_reduction_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if training and (optimizer is None or scheduler is None or scaler is None):
        raise RuntimeError('Training epoch execution requires optimizer, scheduler, and GradScaler.')

    mode = 'train' if training else 'eval'
    model.train(training)
    enforce_frozen_modules_eval(model)
    loss_sums = {
        'total': 0.0,
        'clip': 0.0,
        'inv': 0.0,
        'med': 0.0,
        'med_original': 0.0,
        'med_class_balanced': 0.0,
        'med_class_balanced_w2a': 0.0,
        'med_class_balanced_a2w': 0.0,
        'attr': 0.0,
        'attr_sample': 0.0,
        'attr_window': 0.0,
        'physical': 0.0,
        'physical_dry_mass': 0.0,
        'physical_capacity_ratio': 0.0,
        'physical_is_open_container': 0.0,
        'pol': 0.0,
        'med_w2a': 0.0,
        'med_a2w': 0.0,
        'pol_w2a': 0.0,
        'pol_a2w': 0.0,
    }
    phase_confusion = torch.zeros(3, 3, dtype=torch.long)
    attr_metric_tasks = resolve_attr_metric_tasks(model_config)
    attr_classes = attribute_class_counts(model_config)
    attr_window_metrics = AttributeWindowMetricAccumulator(attr_classes, attr_metric_tasks)
    attr_aggregation_metrics = AttributeAggregationAccumulator(attr_classes, attr_metric_tasks)
    physical_stats = getattr(loader.dataset, 'physical_attribute_stats', None)
    physical_metrics = PhysicalAttributeMetricAccumulator(physical_stats)
    overall_abs = 0.0
    overall_sq = 0.0
    overall_count = 0
    stable_abs = 0.0
    stable_sq = 0.0
    stable_count = 0
    interface_abs = 0.0
    interface_sq = 0.0
    interface_count = 0
    interface_signed_sum = 0.0
    interface_hits_100 = 0
    interface_hits_200 = 0
    interface_hits_300 = 0
    control_overall_abs = 0.0
    control_overall_sq = 0.0
    control_overall_count = 0
    control_reference_abs = 0.0
    control_reference_sq = 0.0
    control_reference_count = 0
    control_interface_abs = 0.0
    control_interface_sq = 0.0
    control_interface_count = 0
    control_interface_signed_sum = 0.0
    control_interface_hits_100 = 0
    control_interface_hits_200 = 0
    control_interface_hits_300 = 0
    delta_interface_abs = 0.0
    delta_interface_sq = 0.0
    delta_interface_count = 0
    delta_interface_signed_sum = 0.0
    finger_overall_abs = 0.0
    finger_overall_sq = 0.0
    finger_overall_count = 0
    finger_stable_abs = 0.0
    finger_stable_sq = 0.0
    finger_stable_count = 0
    finger_interface_abs = 0.0
    finger_interface_sq = 0.0
    finger_interface_count = 0
    finger_control_overall_abs = 0.0
    finger_control_overall_sq = 0.0
    finger_control_overall_count = 0
    finger_control_reference_abs = 0.0
    finger_control_reference_sq = 0.0
    finger_control_reference_count = 0
    finger_control_interface_abs = 0.0
    finger_control_interface_sq = 0.0
    finger_control_interface_count = 0
    finger_delta_interface_abs = 0.0
    finger_delta_interface_sq = 0.0
    finger_delta_interface_count = 0
    finger_delta_interface_signed_sum = 0.0
    finger_large_delta_interface_abs = 0.0
    finger_large_delta_interface_sq = 0.0
    finger_large_delta_interface_count = 0
    finger_large_delta_interface_signed_sum = 0.0
    finger_large_delta_interface_wrong_sign = 0
    finger_large_delta_interface_pred_sum = 0.0
    finger_large_delta_interface_target_sum = 0.0
    finger_large_delta_interface_pred_abs_sum = 0.0
    finger_large_delta_interface_target_abs_sum = 0.0
    finger_large_delta_pos_count = 0
    finger_large_delta_pos_wrong_sign = 0
    finger_large_delta_pos_abs = 0.0
    finger_large_delta_neg_count = 0
    finger_large_delta_neg_wrong_sign = 0
    finger_large_delta_neg_abs = 0.0
    finger_control_interface_abs_by_finger = torch.zeros(3, dtype=torch.float64)
    finger_control_interface_count_by_finger = torch.zeros(3, dtype=torch.long)
    large_delta_interface_abs = 0.0
    large_delta_interface_sq = 0.0
    large_delta_interface_count = 0
    large_delta_interface_signed_sum = 0.0
    large_delta_interface_hits_100 = 0
    large_delta_interface_hits_200 = 0
    large_delta_interface_hits_300 = 0
    large_delta_interface_wrong_sign = 0
    large_delta_interface_pred_sum = 0.0
    large_delta_interface_target_sum = 0.0
    large_delta_pos_abs = 0.0
    large_delta_pos_sq = 0.0
    large_delta_pos_count = 0
    large_delta_pos_signed_sum = 0.0
    large_delta_pos_wrong_sign = 0
    large_delta_pos_pred_sum = 0.0
    large_delta_pos_target_sum = 0.0
    large_delta_neg_abs = 0.0
    large_delta_neg_sq = 0.0
    large_delta_neg_count = 0
    large_delta_neg_signed_sum = 0.0
    large_delta_neg_wrong_sign = 0
    large_delta_neg_pred_sum = 0.0
    large_delta_neg_target_sum = 0.0
    direction_gate_large_delta_correct = 0
    direction_gate_large_delta_count = 0
    direction_gate_large_delta_margin_sum = 0.0
    direction_gate_pos_correct = 0
    direction_gate_pos_count = 0
    direction_gate_pos_margin_sum = 0.0
    direction_gate_pos_prob_sum = 0.0
    direction_gate_neg_correct = 0
    direction_gate_neg_count = 0
    direction_gate_neg_margin_sum = 0.0
    direction_gate_neg_prob_sum = 0.0
    direction_magnitude_pos_abs = 0.0
    direction_magnitude_pos_count = 0
    direction_magnitude_pos_active_sum = 0.0
    direction_magnitude_pos_target_sum = 0.0
    direction_magnitude_pos_opposite_sum = 0.0
    direction_magnitude_neg_abs = 0.0
    direction_magnitude_neg_count = 0
    direction_magnitude_neg_active_sum = 0.0
    direction_magnitude_neg_target_sum = 0.0
    direction_magnitude_neg_opposite_sum = 0.0
    stable_leakage_values: list[torch.Tensor] = []
    gate_abs_sum = 0.0
    gate_count = 0
    gate_stable_sum = 0.0
    gate_stable_count = 0
    gate_interface_miss_sum = 0.0
    gate_interface_count = 0
    gate_smoothness_sum = 0.0
    gate_smoothness_count = 0
    policy_loss_config = stage_config.get('policy_loss', {})
    policy_loss_config = policy_loss_config if isinstance(policy_loss_config, dict) else {}
    large_delta_threshold = float(policy_loss_config.get('large_delta_threshold', 100.0))
    finger_large_delta_threshold = float(
        policy_loss_config.get(
            'finger_large_delta_threshold',
            policy_loss_config.get('residual_large_delta_threshold', large_delta_threshold),
        )
    )
    direction_metrics = DirectionMetricAccumulator(
        finger_large_delta_threshold=finger_large_delta_threshold,
    )
    delta_normalization_scale = float(policy_loss_config.get('delta_normalization_scale', 100.0))
    direction_magnitude_scale = float(
        policy_loss_config.get(
            'sign_magnitude_scale',
            policy_loss_config.get('direction_magnitude_scale', delta_normalization_scale),
        )
    )
    direction_magnitude_pos_scale = float(
        policy_loss_config.get(
            'sign_magnitude_pos_scale',
            policy_loss_config.get('direction_magnitude_pos_scale', direction_magnitude_scale),
        )
    )
    direction_magnitude_neg_scale = float(
        policy_loss_config.get(
            'sign_magnitude_neg_scale',
            policy_loss_config.get('direction_magnitude_neg_scale', direction_magnitude_scale),
        )
    )
    iterator = tqdm(loader, desc=mode, leave=False)
    for batch in iterator:
        batch = move_to_device(batch, device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with torch.amp.autocast('cuda', enabled=amp_enabled):
                outputs = model(batch)
                losses = compute_losses(
                    outputs,
                    batch,
                    loss_weights=stage_config.get('loss_weights', {}),
                    attribute_loss_config=stage_config.get('attribute_loss'),
                    physical_loss_config=stage_config.get('physical_loss'),
                    policy_loss_config=stage_config.get('policy_loss'),
                    temperature_clip=float(model_config['losses']['temperature_clip']),
                    temperature_inv=float(model_config['losses']['temperature_inv']),
                    phase_class_weights=phase_class_weights,
                    loss_reduction_config=loss_reduction_config,
                )
            if training:
                previous_scale = float(scaler.get_scale())
                scaler.scale(losses['total']).backward()
                if grad_clip_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()
                if (not amp_enabled) or float(scaler.get_scale()) >= previous_scale:
                    scheduler.step()
        for key in loss_sums:
            loss_sums[key] += float(losses[key].item())

        window_mask = batch['window_mask']
        flat_valid = window_mask.reshape(-1)
        phase_targets = batch['phase_labels'].reshape(-1)[flat_valid].detach().cpu()
        phase_preds = outputs['medium_logits'].reshape(-1, 3)[flat_valid].argmax(dim=-1).detach().cpu()
        for pred, target in zip(phase_preds, phase_targets):
            phase_confusion[int(target), int(pred)] += 1

        attr_window_metrics.update(outputs, batch)
        attr_aggregation_metrics.update(outputs, batch)
        physical_metrics.update(outputs, batch)
        if 'direction_ids' in batch:
            direction_metrics.update(outputs, batch)

        force_pred = outputs['force_pred'].reshape_as(batch['expert_forces'])
        force_delta = outputs.get('force_interface_delta')
        interface_gate = outputs.get('interface_gate')
        direction_prob_pos = outputs.get('residual_direction_prob_pos')
        direction_prob_neg = outputs.get('residual_direction_prob_neg')
        if direction_prob_pos is not None and direction_prob_neg is not None:
            direction_prob_pos = direction_prob_pos.reshape_as(batch['expert_forces'])
            direction_prob_neg = direction_prob_neg.reshape_as(batch['expert_forces'])
        direction_pos_magnitude = outputs.get('force_interface_delta_pos_magnitude')
        direction_neg_magnitude = outputs.get('force_interface_delta_neg_magnitude')
        if direction_pos_magnitude is not None and direction_neg_magnitude is not None:
            direction_pos_magnitude = direction_pos_magnitude.reshape_as(batch['expert_forces'])
            direction_neg_magnitude = direction_neg_magnitude.reshape_as(batch['expert_forces'])
        if force_delta is not None and interface_gate is not None:
            force_delta = force_delta.reshape_as(batch['expert_forces'])
            interface_gate = interface_gate.reshape_as(batch['expert_forces'])
            gated_residual = interface_gate * force_delta
            stable_valid_mask = batch['stable_masks'] & window_mask
            if stable_valid_mask.any():
                stable_leakage_values.append(gated_residual[stable_valid_mask].abs().detach().cpu())
            if 'soft_gate_targets' in batch:
                gate_target = batch['soft_gate_targets']
                valid_gate = window_mask
                gate_abs_sum += float((interface_gate[valid_gate] - gate_target[valid_gate]).abs().sum().item())
                gate_count += int(valid_gate.sum().item())
            stable_gate_mask = batch['stable_masks'] & window_mask
            if stable_gate_mask.any():
                gate_stable_sum += float(interface_gate[stable_gate_mask].sum().item())
                gate_stable_count += int(stable_gate_mask.sum().item())
            interface_gate_mask = (batch['phase_labels'] == 1) & window_mask
            if interface_gate_mask.any():
                gate_interface_miss_sum += float((1.0 - interface_gate[interface_gate_mask]).abs().sum().item())
                gate_interface_count += int(interface_gate_mask.sum().item())
            if interface_gate.shape[1] > 1:
                pair_mask = window_mask[:, 1:] & window_mask[:, :-1]
                if pair_mask.any():
                    gate_smoothness_sum += float((interface_gate[:, 1:] - interface_gate[:, :-1]).abs()[pair_mask].sum().item())
                    gate_smoothness_count += int(pair_mask.sum().item())
        expert_mask = batch['has_expert'] & window_mask
        if expert_mask.any():
            pred = force_pred[expert_mask]
            target = batch['expert_forces'][expert_mask]
            abs_error = torch.abs(pred - target)
            sq_error = (pred - target) ** 2
            overall_abs += float(abs_error.sum().item())
            overall_sq += float(sq_error.sum().item())
            overall_count += int(abs_error.numel())
            stable_expert_mask = expert_mask & batch['stable_masks']
            if stable_expert_mask.any():
                pred_stable = force_pred[stable_expert_mask]
                target_stable = batch['expert_forces'][stable_expert_mask]
                stable_abs += float(torch.abs(pred_stable - target_stable).sum().item())
                stable_sq += float(((pred_stable - target_stable) ** 2).sum().item())
                stable_count += int(pred_stable.numel())
            interface_mask = expert_mask & (batch['phase_labels'] == 1)
            if interface_mask.any():
                pred_interface = force_pred[interface_mask]
                target_interface = batch['expert_forces'][interface_mask]
                interface_error = pred_interface - target_interface
                abs_interface_error = torch.abs(interface_error)
                interface_abs += float(abs_interface_error.sum().item())
                interface_sq += float((interface_error ** 2).sum().item())
                interface_count += int(pred_interface.numel())
                interface_signed_sum += float(interface_error.sum().item())
                interface_hits_100 += int((abs_interface_error <= 100.0).sum().item())
                interface_hits_200 += int((abs_interface_error <= 200.0).sum().item())
                interface_hits_300 += int((abs_interface_error <= 300.0).sum().item())

        control_mask = batch['has_control_target'] & window_mask if 'has_control_target' in batch else expert_mask
        if control_mask.any():
            pred_control = force_pred[control_mask]
            target_control = batch['control_force_targets'][control_mask] if 'control_force_targets' in batch else batch['expert_forces'][control_mask]
            control_abs_error = torch.abs(pred_control - target_control)
            control_sq_error = (pred_control - target_control) ** 2
            control_overall_abs += float(control_abs_error.sum().item())
            control_overall_sq += float(control_sq_error.sum().item())
            control_overall_count += int(control_abs_error.numel())

            reference_control_mask = control_mask & batch['reference_supervision_masks'] if 'reference_supervision_masks' in batch else control_mask.new_zeros(control_mask.shape)
            if reference_control_mask.any():
                pred_reference = force_pred[reference_control_mask]
                target_reference = batch['control_force_targets'][reference_control_mask] if 'control_force_targets' in batch else batch['expert_forces'][reference_control_mask]
                control_reference_abs += float(torch.abs(pred_reference - target_reference).sum().item())
                control_reference_sq += float(((pred_reference - target_reference) ** 2).sum().item())
                control_reference_count += int(pred_reference.numel())

            interface_control_mask = control_mask & batch['delta_supervision_masks'] if 'delta_supervision_masks' in batch else (control_mask & (batch['phase_labels'] == 1))
            if interface_control_mask.any():
                pred_control_interface = force_pred[interface_control_mask]
                target_control_interface = batch['control_force_targets'][interface_control_mask] if 'control_force_targets' in batch else batch['expert_forces'][interface_control_mask]
                control_interface_error = pred_control_interface - target_control_interface
                abs_control_interface_error = torch.abs(control_interface_error)
                control_interface_abs += float(abs_control_interface_error.sum().item())
                control_interface_sq += float((control_interface_error ** 2).sum().item())
                control_interface_count += int(pred_control_interface.numel())
                control_interface_signed_sum += float(control_interface_error.sum().item())
                control_interface_hits_100 += int((abs_control_interface_error <= 100.0).sum().item())
                control_interface_hits_200 += int((abs_control_interface_error <= 200.0).sum().item())
                control_interface_hits_300 += int((abs_control_interface_error <= 300.0).sum().item())
                if force_delta is not None and 'delta_force_targets' in batch:
                    pred_delta_interface = force_delta[interface_control_mask]
                    target_delta_interface = batch['delta_force_targets'][interface_control_mask]
                    delta_error = pred_delta_interface - target_delta_interface
                    abs_delta_error = torch.abs(delta_error)
                    delta_interface_abs += float(abs_delta_error.sum().item())
                    delta_interface_sq += float((delta_error ** 2).sum().item())
                    delta_interface_count += int(delta_error.numel())
                    delta_interface_signed_sum += float(delta_error.sum().item())

                    large_delta_mask = torch.abs(target_delta_interface) >= large_delta_threshold
                    if large_delta_mask.any():
                        large_delta_pred = pred_delta_interface[large_delta_mask]
                        large_delta_target = target_delta_interface[large_delta_mask]
                        large_delta_error = delta_error[large_delta_mask]
                        abs_large_delta_error = torch.abs(large_delta_error)
                        large_delta_interface_abs += float(abs_large_delta_error.sum().item())
                        large_delta_interface_sq += float((large_delta_error ** 2).sum().item())
                        large_delta_interface_count += int(large_delta_error.numel())
                        large_delta_interface_signed_sum += float(large_delta_error.sum().item())
                        large_delta_interface_hits_100 += int((abs_large_delta_error <= 100.0).sum().item())
                        large_delta_interface_hits_200 += int((abs_large_delta_error <= 200.0).sum().item())
                        large_delta_interface_hits_300 += int((abs_large_delta_error <= 300.0).sum().item())
                        large_delta_interface_wrong_sign += int((large_delta_pred * large_delta_target < 0.0).sum().item())
                        large_delta_interface_pred_sum += float(large_delta_pred.sum().item())
                        large_delta_interface_target_sum += float(large_delta_target.sum().item())
                        if direction_prob_pos is not None and direction_prob_neg is not None:
                            direction_pos_interface = direction_prob_pos[interface_control_mask]
                            direction_neg_interface = direction_prob_neg[interface_control_mask]
                            large_delta_direction_pos = direction_pos_interface[large_delta_mask]
                            large_delta_direction_neg = direction_neg_interface[large_delta_mask]
                            direction_target_pos = large_delta_target > 0.0
                            direction_pred_pos = large_delta_direction_pos >= large_delta_direction_neg
                            direction_gate_large_delta_correct += int((direction_pred_pos == direction_target_pos).sum().item())
                            direction_gate_large_delta_count += int(large_delta_target.numel())
                            direction_margin = torch.where(
                                direction_target_pos,
                                large_delta_direction_pos - large_delta_direction_neg,
                                large_delta_direction_neg - large_delta_direction_pos,
                            )
                            direction_gate_large_delta_margin_sum += float(direction_margin.sum().item())

                            if direction_target_pos.any():
                                pos_direction_pred = direction_pred_pos[direction_target_pos]
                                pos_direction_margin = direction_margin[direction_target_pos]
                                pos_direction_prob = large_delta_direction_pos[direction_target_pos]
                                direction_gate_pos_correct += int(pos_direction_pred.sum().item())
                                direction_gate_pos_count += int(direction_target_pos.sum().item())
                                direction_gate_pos_margin_sum += float(pos_direction_margin.sum().item())
                                direction_gate_pos_prob_sum += float(pos_direction_prob.sum().item())

                            direction_target_neg = large_delta_target < 0.0
                            if direction_target_neg.any():
                                neg_direction_pred_is_neg = ~direction_pred_pos[direction_target_neg]
                                neg_direction_margin = direction_margin[direction_target_neg]
                                neg_direction_prob = large_delta_direction_neg[direction_target_neg]
                                direction_gate_neg_correct += int(neg_direction_pred_is_neg.sum().item())
                                direction_gate_neg_count += int(direction_target_neg.sum().item())
                                direction_gate_neg_margin_sum += float(neg_direction_margin.sum().item())
                                direction_gate_neg_prob_sum += float(neg_direction_prob.sum().item())

                        if direction_pos_magnitude is not None and direction_neg_magnitude is not None:
                            magnitude_pos_interface = direction_pos_magnitude[interface_control_mask]
                            magnitude_neg_interface = direction_neg_magnitude[interface_control_mask]
                            large_delta_magnitude_pos = magnitude_pos_interface[large_delta_mask]
                            large_delta_magnitude_neg = magnitude_neg_interface[large_delta_mask]

                            magnitude_target_pos = large_delta_target > 0.0
                            if magnitude_target_pos.any():
                                pos_active = large_delta_magnitude_pos[magnitude_target_pos]
                                pos_target_magnitude = (
                                    torch.abs(large_delta_target[magnitude_target_pos])
                                    / direction_magnitude_pos_scale
                                )
                                pos_opposite = large_delta_magnitude_neg[magnitude_target_pos]
                                direction_magnitude_pos_abs += float(
                                    torch.abs(pos_active - pos_target_magnitude).sum().item()
                                )
                                direction_magnitude_pos_count += int(magnitude_target_pos.sum().item())
                                direction_magnitude_pos_active_sum += float(pos_active.sum().item())
                                direction_magnitude_pos_target_sum += float(pos_target_magnitude.sum().item())
                                direction_magnitude_pos_opposite_sum += float(pos_opposite.sum().item())

                            magnitude_target_neg = large_delta_target < 0.0
                            if magnitude_target_neg.any():
                                neg_active = large_delta_magnitude_neg[magnitude_target_neg]
                                neg_target_magnitude = (
                                    torch.abs(large_delta_target[magnitude_target_neg])
                                    / direction_magnitude_neg_scale
                                )
                                neg_opposite = large_delta_magnitude_pos[magnitude_target_neg]
                                direction_magnitude_neg_abs += float(
                                    torch.abs(neg_active - neg_target_magnitude).sum().item()
                                )
                                direction_magnitude_neg_count += int(magnitude_target_neg.sum().item())
                                direction_magnitude_neg_active_sum += float(neg_active.sum().item())
                                direction_magnitude_neg_target_sum += float(neg_target_magnitude.sum().item())
                                direction_magnitude_neg_opposite_sum += float(neg_opposite.sum().item())

                        pos_large_delta_mask = large_delta_mask & (target_delta_interface > 0.0)
                        if pos_large_delta_mask.any():
                            pos_pred = pred_delta_interface[pos_large_delta_mask]
                            pos_target = target_delta_interface[pos_large_delta_mask]
                            pos_error = delta_error[pos_large_delta_mask]
                            abs_pos_error = torch.abs(pos_error)
                            large_delta_pos_abs += float(abs_pos_error.sum().item())
                            large_delta_pos_sq += float((pos_error ** 2).sum().item())
                            large_delta_pos_count += int(pos_error.numel())
                            large_delta_pos_signed_sum += float(pos_error.sum().item())
                            large_delta_pos_wrong_sign += int((pos_pred * pos_target < 0.0).sum().item())
                            large_delta_pos_pred_sum += float(pos_pred.sum().item())
                            large_delta_pos_target_sum += float(pos_target.sum().item())

                        neg_large_delta_mask = large_delta_mask & (target_delta_interface < 0.0)
                        if neg_large_delta_mask.any():
                            neg_pred = pred_delta_interface[neg_large_delta_mask]
                            neg_target = target_delta_interface[neg_large_delta_mask]
                            neg_error = delta_error[neg_large_delta_mask]
                            abs_neg_error = torch.abs(neg_error)
                            large_delta_neg_abs += float(abs_neg_error.sum().item())
                            large_delta_neg_sq += float((neg_error ** 2).sum().item())
                            large_delta_neg_count += int(neg_error.numel())
                            large_delta_neg_signed_sum += float(neg_error.sum().item())
                            large_delta_neg_wrong_sign += int((neg_pred * neg_target < 0.0).sum().item())
                            large_delta_neg_pred_sum += float(neg_pred.sum().item())
                            large_delta_neg_target_sum += float(neg_target.sum().item())

        if 'finger_force_pred' in outputs and 'finger_expert_forces' in batch:
            finger_pred = outputs['finger_force_pred'].reshape_as(batch['finger_expert_forces'])
            finger_window_mask = window_mask.unsqueeze(-1).expand_as(batch['finger_expert_forces'])
            finger_stable_mask = batch['stable_masks'].unsqueeze(-1).expand_as(batch['finger_expert_forces']) & finger_window_mask
            finger_interface_mask = (batch['phase_labels'] == 1).unsqueeze(-1).expand_as(batch['finger_expert_forces']) & finger_window_mask
            finger_expert_mask = batch['has_finger_expert'] & finger_window_mask

            if finger_expert_mask.any():
                finger_target = batch['finger_expert_forces']
                finger_error = finger_pred[finger_expert_mask] - finger_target[finger_expert_mask]
                finger_abs_error = finger_error.abs()
                finger_overall_abs += float(finger_abs_error.sum().item())
                finger_overall_sq += float((finger_error ** 2).sum().item())
                finger_overall_count += int(finger_abs_error.numel())

                finger_stable_expert_mask = finger_expert_mask & finger_stable_mask
                if finger_stable_expert_mask.any():
                    stable_error = finger_pred[finger_stable_expert_mask] - finger_target[finger_stable_expert_mask]
                    finger_stable_abs += float(stable_error.abs().sum().item())
                    finger_stable_sq += float((stable_error ** 2).sum().item())
                    finger_stable_count += int(stable_error.numel())

                finger_interface_expert_mask = finger_expert_mask & finger_interface_mask
                if finger_interface_expert_mask.any():
                    interface_error = finger_pred[finger_interface_expert_mask] - finger_target[finger_interface_expert_mask]
                    finger_interface_abs += float(interface_error.abs().sum().item())
                    finger_interface_sq += float((interface_error ** 2).sum().item())
                    finger_interface_count += int(interface_error.numel())

            finger_control_mask = batch['has_finger_control_target'] & finger_window_mask
            if finger_control_mask.any():
                finger_control_target = batch['finger_control_force_targets']
                control_error_all = finger_pred[finger_control_mask] - finger_control_target[finger_control_mask]
                control_abs_all = control_error_all.abs()
                finger_control_overall_abs += float(control_abs_all.sum().item())
                finger_control_overall_sq += float((control_error_all ** 2).sum().item())
                finger_control_overall_count += int(control_abs_all.numel())

                finger_reference_control_mask = finger_control_mask & batch['has_finger_reference']
                if finger_reference_control_mask.any():
                    reference_error = finger_pred[finger_reference_control_mask] - finger_control_target[finger_reference_control_mask]
                    finger_control_reference_abs += float(reference_error.abs().sum().item())
                    finger_control_reference_sq += float((reference_error ** 2).sum().item())
                    finger_control_reference_count += int(reference_error.numel())

                finger_interface_control_mask = finger_control_mask & finger_interface_mask
                if finger_interface_control_mask.any():
                    interface_error = finger_pred[finger_interface_control_mask] - finger_control_target[finger_interface_control_mask]
                    finger_control_interface_abs += float(interface_error.abs().sum().item())
                    finger_control_interface_sq += float((interface_error ** 2).sum().item())
                    finger_control_interface_count += int(interface_error.numel())

                    per_finger_abs = (finger_pred - finger_control_target).abs()
                    for finger_index in range(min(per_finger_abs.shape[-1], finger_control_interface_abs_by_finger.numel())):
                        mask_i = finger_interface_control_mask[..., finger_index]
                        if mask_i.any():
                            finger_control_interface_abs_by_finger[finger_index] += float(per_finger_abs[..., finger_index][mask_i].sum().item())
                            finger_control_interface_count_by_finger[finger_index] += int(mask_i.sum().item())

                    if 'finger_force_interface_delta' in outputs and 'finger_delta_force_targets' in batch:
                        finger_delta = outputs['finger_force_interface_delta'].reshape_as(batch['finger_delta_force_targets'])
                        finger_delta_pred_interface = finger_delta[finger_interface_control_mask]
                        finger_delta_target_interface = batch['finger_delta_force_targets'][finger_interface_control_mask]
                        delta_error = finger_delta_pred_interface - finger_delta_target_interface
                        finger_delta_interface_abs += float(delta_error.abs().sum().item())
                        finger_delta_interface_sq += float((delta_error ** 2).sum().item())
                        finger_delta_interface_count += int(delta_error.numel())
                        finger_delta_interface_signed_sum += float(delta_error.sum().item())

                        finger_large_delta_mask = torch.abs(finger_delta_target_interface) >= finger_large_delta_threshold
                        if finger_large_delta_mask.any():
                            large_pred = finger_delta_pred_interface[finger_large_delta_mask]
                            large_target = finger_delta_target_interface[finger_large_delta_mask]
                            large_error = delta_error[finger_large_delta_mask]
                            abs_large_error = large_error.abs()
                            finger_large_delta_interface_abs += float(abs_large_error.sum().item())
                            finger_large_delta_interface_sq += float((large_error ** 2).sum().item())
                            finger_large_delta_interface_count += int(large_error.numel())
                            finger_large_delta_interface_signed_sum += float(large_error.sum().item())
                            finger_large_delta_interface_wrong_sign += int((large_pred * large_target < 0.0).sum().item())
                            finger_large_delta_interface_pred_sum += float(large_pred.sum().item())
                            finger_large_delta_interface_target_sum += float(large_target.sum().item())
                            finger_large_delta_interface_pred_abs_sum += float(large_pred.abs().sum().item())
                            finger_large_delta_interface_target_abs_sum += float(large_target.abs().sum().item())

                            pos_mask = large_target > 0.0
                            if pos_mask.any():
                                pos_pred = large_pred[pos_mask]
                                pos_target = large_target[pos_mask]
                                pos_error = large_error[pos_mask]
                                finger_large_delta_pos_count += int(pos_error.numel())
                                finger_large_delta_pos_abs += float(pos_error.abs().sum().item())
                                finger_large_delta_pos_wrong_sign += int((pos_pred * pos_target < 0.0).sum().item())

                            neg_mask = large_target < 0.0
                            if neg_mask.any():
                                neg_pred = large_pred[neg_mask]
                                neg_target = large_target[neg_mask]
                                neg_error = large_error[neg_mask]
                                finger_large_delta_neg_count += int(neg_error.numel())
                                finger_large_delta_neg_abs += float(neg_error.abs().sum().item())
                                finger_large_delta_neg_wrong_sign += int((neg_pred * neg_target < 0.0).sum().item())

    denom = max(1, len(loader))
    if stable_leakage_values:
        stable_leakage = torch.cat(stable_leakage_values)
        stable_leakage_mean = float(stable_leakage.mean().item())
        stable_leakage_p95 = float(torch.quantile(stable_leakage, 0.95).item())
    else:
        stable_leakage_mean = 0.0
        stable_leakage_p95 = 0.0
    attribute_window_diagnostics = attr_window_metrics.finalize()
    attribute_aggregation = attr_aggregation_metrics.finalize()
    physical_attribute_metrics = physical_metrics.finalize()
    all_window_attr = attribute_window_diagnostics['all_windows']
    stable_water_air_sample_attr = attribute_aggregation['sample_aggregation']['stable_water_air']['mean_logits']
    model_pool_sample_attr = attribute_aggregation['sample_model_pool']
    if int(model_pool_sample_attr.get('count', 0)) > 0:
        primary_attr = model_pool_sample_attr
        primary_attr_source = 'sample_model_pool'
    elif int(stable_water_air_sample_attr.get('count', 0)) > 0:
        primary_attr = stable_water_air_sample_attr
        primary_attr_source = 'sample_stable_water_air_mean_logits'
    else:
        primary_attr = all_window_attr
        primary_attr_source = 'window_all'

    frag_macro_f1 = float(all_window_attr['fragility_macro_f1'])
    geom_macro_f1 = float(all_window_attr['geometry_macro_f1'])
    surf_macro_f1 = float(all_window_attr['surface_macro_f1'])
    attr_f1_scores = {
        'fragility': frag_macro_f1,
        'geometry': geom_macro_f1,
        'surface': surf_macro_f1,
    }
    attr_macro_f1_avg = average_selected_attr_f1(attr_f1_scores, attr_metric_tasks)
    attr_all_macro_f1_avg = (frag_macro_f1 + geom_macro_f1 + surf_macro_f1) / 3.0
    medium_macro_f1 = macro_f1_from_confusion(phase_confusion)
    attr_primary_macro_f1 = float(primary_attr['attr_macro_f1_avg'])
    metrics: dict[str, Any] = {
        'loss_total': loss_sums['total'] / denom,
        'loss_clip': loss_sums['clip'] / denom,
        'loss_inv': loss_sums['inv'] / denom,
        'loss_med': loss_sums['med'] / denom,
        'loss_med_original': loss_sums['med_original'] / denom,
        'loss_med_class_balanced': loss_sums['med_class_balanced'] / denom,
        'loss_med_class_balanced_w2a': loss_sums['med_class_balanced_w2a'] / denom,
        'loss_med_class_balanced_a2w': loss_sums['med_class_balanced_a2w'] / denom,
        'loss_attr': loss_sums['attr'] / denom,
        'loss_attr_sample': loss_sums['attr_sample'] / denom,
        'loss_attr_window': loss_sums['attr_window'] / denom,
        'loss_physical': loss_sums['physical'] / denom,
        'loss_physical_dry_mass': loss_sums['physical_dry_mass'] / denom,
        'loss_physical_capacity_ratio': loss_sums['physical_capacity_ratio'] / denom,
        'loss_physical_is_open_container': loss_sums['physical_is_open_container'] / denom,
        'loss_pol': loss_sums['pol'] / denom,
        'loss_med_w2a': loss_sums['med_w2a'] / denom,
        'loss_med_a2w': loss_sums['med_a2w'] / denom,
        'loss_pol_w2a': loss_sums['pol_w2a'] / denom,
        'loss_pol_a2w': loss_sums['pol_a2w'] / denom,
        'contrastive_loss_sum': (loss_sums['clip'] + loss_sums['inv']) / denom,
        'medium_accuracy': accuracy_from_confusion(phase_confusion),
        'medium_macro_f1': medium_macro_f1,
        'medium_f1_water': class_f1_from_confusion(phase_confusion, 0),
        'medium_f1_interface': class_f1_from_confusion(phase_confusion, 1),
        'medium_f1_air': class_f1_from_confusion(phase_confusion, 2),
        'fragility_accuracy': all_window_attr['fragility_accuracy'],
        'geometry_accuracy': all_window_attr['geometry_accuracy'],
        'surface_accuracy': all_window_attr['surface_accuracy'],
        'fragility_macro_f1': frag_macro_f1,
        'geometry_macro_f1': geom_macro_f1,
        'surface_macro_f1': surf_macro_f1,
        'attr_macro_f1_avg': attr_macro_f1_avg,
        'attr_all_macro_f1_avg': attr_all_macro_f1_avg,
        'attr_primary_macro_f1': attr_primary_macro_f1,
        'attr_primary_source': primary_attr_source,
        'attr_metric_tasks': attr_metric_tasks,
        'combined_attr_accuracy': all_window_attr['combined_attr_accuracy'],
        'overall_mae': overall_abs / max(1, overall_count),
        'overall_mse': overall_sq / max(1, overall_count),
        'stable_mae': stable_abs / max(1, stable_count),
        'stable_mse': stable_sq / max(1, stable_count),
        'interface_mae': interface_abs / max(1, interface_count),
        'interface_mse': interface_sq / max(1, interface_count),
        'interface_bias': interface_signed_sum / max(1, interface_count),
        'interface_hit_rate_100': interface_hits_100 / max(1, interface_count),
        'interface_hit_rate_200': interface_hits_200 / max(1, interface_count),
        'interface_hit_rate_300': interface_hits_300 / max(1, interface_count),
        'control_overall_mae': control_overall_abs / max(1, control_overall_count),
        'control_overall_mse': control_overall_sq / max(1, control_overall_count),
        'control_reference_mae': control_reference_abs / max(1, control_reference_count),
        'control_reference_mse': control_reference_sq / max(1, control_reference_count),
        'control_interface_mae': control_interface_abs / max(1, control_interface_count),
        'control_interface_mse': control_interface_sq / max(1, control_interface_count),
        'control_interface_bias': control_interface_signed_sum / max(1, control_interface_count),
        'control_interface_hit_rate_100': control_interface_hits_100 / max(1, control_interface_count),
        'control_interface_hit_rate_200': control_interface_hits_200 / max(1, control_interface_count),
        'control_interface_hit_rate_300': control_interface_hits_300 / max(1, control_interface_count),
        'delta_interface_mae': delta_interface_abs / max(1, delta_interface_count),
        'delta_interface_mse': delta_interface_sq / max(1, delta_interface_count),
        'delta_interface_bias': delta_interface_signed_sum / max(1, delta_interface_count),
        'finger_overall_mae': finger_overall_abs / max(1, finger_overall_count),
        'finger_overall_mse': finger_overall_sq / max(1, finger_overall_count),
        'finger_stable_mae': finger_stable_abs / max(1, finger_stable_count),
        'finger_stable_mse': finger_stable_sq / max(1, finger_stable_count),
        'finger_interface_mae': finger_interface_abs / max(1, finger_interface_count),
        'finger_interface_mse': finger_interface_sq / max(1, finger_interface_count),
        'finger_control_overall_mae': finger_control_overall_abs / max(1, finger_control_overall_count),
        'finger_control_overall_mse': finger_control_overall_sq / max(1, finger_control_overall_count),
        'finger_control_reference_mae': finger_control_reference_abs / max(1, finger_control_reference_count),
        'finger_control_reference_mse': finger_control_reference_sq / max(1, finger_control_reference_count),
        'finger_control_interface_mae': finger_control_interface_abs / max(1, finger_control_interface_count),
        'finger_control_interface_mse': finger_control_interface_sq / max(1, finger_control_interface_count),
        'finger_delta_interface_mae': finger_delta_interface_abs / max(1, finger_delta_interface_count),
        'finger_delta_interface_mse': finger_delta_interface_sq / max(1, finger_delta_interface_count),
        'finger_delta_interface_bias': finger_delta_interface_signed_sum / max(1, finger_delta_interface_count),
        'finger0_control_interface_mae': float(finger_control_interface_abs_by_finger[0].item()) / max(1, int(finger_control_interface_count_by_finger[0].item())),
        'finger1_control_interface_mae': float(finger_control_interface_abs_by_finger[1].item()) / max(1, int(finger_control_interface_count_by_finger[1].item())),
        'finger2_control_interface_mae': float(finger_control_interface_abs_by_finger[2].item()) / max(1, int(finger_control_interface_count_by_finger[2].item())),
        'finger_large_delta_threshold': finger_large_delta_threshold,
        'finger_large_delta_interface_count': finger_large_delta_interface_count,
        'finger_large_delta_interface_mae': finger_large_delta_interface_abs / max(1, finger_large_delta_interface_count),
        'finger_large_delta_interface_mse': finger_large_delta_interface_sq / max(1, finger_large_delta_interface_count),
        'finger_large_delta_interface_bias': finger_large_delta_interface_signed_sum / max(1, finger_large_delta_interface_count),
        'finger_large_delta_wrong_sign_rate': finger_large_delta_interface_wrong_sign / max(1, finger_large_delta_interface_count),
        'finger_large_delta_pred_mean': finger_large_delta_interface_pred_sum / max(1, finger_large_delta_interface_count),
        'finger_large_delta_target_mean': finger_large_delta_interface_target_sum / max(1, finger_large_delta_interface_count),
        'finger_large_delta_pred_abs_mean': finger_large_delta_interface_pred_abs_sum / max(1, finger_large_delta_interface_count),
        'finger_large_delta_target_abs_mean': finger_large_delta_interface_target_abs_sum / max(1, finger_large_delta_interface_count),
        'finger_large_delta_pos_count': finger_large_delta_pos_count,
        'finger_large_delta_pos_mae': finger_large_delta_pos_abs / max(1, finger_large_delta_pos_count),
        'finger_large_delta_pos_wrong_sign_rate': finger_large_delta_pos_wrong_sign / max(1, finger_large_delta_pos_count),
        'finger_large_delta_neg_count': finger_large_delta_neg_count,
        'finger_large_delta_neg_mae': finger_large_delta_neg_abs / max(1, finger_large_delta_neg_count),
        'finger_large_delta_neg_wrong_sign_rate': finger_large_delta_neg_wrong_sign / max(1, finger_large_delta_neg_count),
        'large_delta_threshold': large_delta_threshold,
        'large_delta_interface_count': large_delta_interface_count,
        'large_delta_interface_mae': large_delta_interface_abs / max(1, large_delta_interface_count),
        'large_delta_interface_mse': large_delta_interface_sq / max(1, large_delta_interface_count),
        'large_delta_interface_bias': large_delta_interface_signed_sum / max(1, large_delta_interface_count),
        'large_delta_interface_hit_rate_100': large_delta_interface_hits_100 / max(1, large_delta_interface_count),
        'large_delta_interface_hit_rate_200': large_delta_interface_hits_200 / max(1, large_delta_interface_count),
        'large_delta_interface_hit_rate_300': large_delta_interface_hits_300 / max(1, large_delta_interface_count),
        'large_delta_wrong_sign_rate': large_delta_interface_wrong_sign / max(1, large_delta_interface_count),
        'large_delta_pred_mean': large_delta_interface_pred_sum / max(1, large_delta_interface_count),
        'large_delta_target_mean': large_delta_interface_target_sum / max(1, large_delta_interface_count),
        'large_delta_pos_count': large_delta_pos_count,
        'large_delta_pos_mae': large_delta_pos_abs / max(1, large_delta_pos_count),
        'large_delta_pos_mse': large_delta_pos_sq / max(1, large_delta_pos_count),
        'large_delta_pos_bias': large_delta_pos_signed_sum / max(1, large_delta_pos_count),
        'large_delta_pos_wrong_sign_rate': large_delta_pos_wrong_sign / max(1, large_delta_pos_count),
        'large_delta_pos_pred_mean': large_delta_pos_pred_sum / max(1, large_delta_pos_count),
        'large_delta_pos_target_mean': large_delta_pos_target_sum / max(1, large_delta_pos_count),
        'large_delta_neg_count': large_delta_neg_count,
        'large_delta_neg_mae': large_delta_neg_abs / max(1, large_delta_neg_count),
        'large_delta_neg_mse': large_delta_neg_sq / max(1, large_delta_neg_count),
        'large_delta_neg_bias': large_delta_neg_signed_sum / max(1, large_delta_neg_count),
        'large_delta_neg_wrong_sign_rate': large_delta_neg_wrong_sign / max(1, large_delta_neg_count),
        'large_delta_neg_pred_mean': large_delta_neg_pred_sum / max(1, large_delta_neg_count),
        'large_delta_neg_target_mean': large_delta_neg_target_sum / max(1, large_delta_neg_count),
        'direction_gate_large_delta_acc': direction_gate_large_delta_correct / max(1, direction_gate_large_delta_count),
        'direction_gate_large_delta_margin': direction_gate_large_delta_margin_sum / max(1, direction_gate_large_delta_count),
        'direction_gate_pos_acc': direction_gate_pos_correct / max(1, direction_gate_pos_count),
        'direction_gate_pos_margin': direction_gate_pos_margin_sum / max(1, direction_gate_pos_count),
        'direction_gate_pos_prob_mean': direction_gate_pos_prob_sum / max(1, direction_gate_pos_count),
        'direction_gate_neg_acc': direction_gate_neg_correct / max(1, direction_gate_neg_count),
        'direction_gate_neg_margin': direction_gate_neg_margin_sum / max(1, direction_gate_neg_count),
        'direction_gate_neg_prob_mean': direction_gate_neg_prob_sum / max(1, direction_gate_neg_count),
        'direction_magnitude_scale': direction_magnitude_scale,
        'direction_magnitude_pos_scale': direction_magnitude_pos_scale,
        'direction_magnitude_neg_scale': direction_magnitude_neg_scale,
        'direction_magnitude_pos_mae': direction_magnitude_pos_abs / max(1, direction_magnitude_pos_count),
        'direction_magnitude_pos_active_mean': direction_magnitude_pos_active_sum / max(1, direction_magnitude_pos_count),
        'direction_magnitude_pos_target_mean': direction_magnitude_pos_target_sum / max(1, direction_magnitude_pos_count),
        'direction_magnitude_pos_opposite_mean': direction_magnitude_pos_opposite_sum / max(1, direction_magnitude_pos_count),
        'direction_magnitude_neg_mae': direction_magnitude_neg_abs / max(1, direction_magnitude_neg_count),
        'direction_magnitude_neg_active_mean': direction_magnitude_neg_active_sum / max(1, direction_magnitude_neg_count),
        'direction_magnitude_neg_target_mean': direction_magnitude_neg_target_sum / max(1, direction_magnitude_neg_count),
        'direction_magnitude_neg_opposite_mean': direction_magnitude_neg_opposite_sum / max(1, direction_magnitude_neg_count),
        'stable_leakage_mean': stable_leakage_mean,
        'stable_leakage_p95': stable_leakage_p95,
        'gate_mae': gate_abs_sum / max(1, gate_count),
        'gate_false_positive_stable': gate_stable_sum / max(1, gate_stable_count),
        'gate_false_negative_interface': gate_interface_miss_sum / max(1, gate_interface_count),
        'gate_smoothness': gate_smoothness_sum / max(1, gate_smoothness_count),
        'joint_score': 0.6 * medium_macro_f1 + 0.4 * attr_primary_macro_f1,
        'medium_confusion': phase_confusion.tolist(),
        'fragility_confusion': attr_window_metrics.confusions['all_windows']['fragility'].tolist(),
        'geometry_confusion': attr_window_metrics.confusions['all_windows']['geometry'].tolist(),
        'surface_confusion': attr_window_metrics.confusions['all_windows']['surface'].tolist(),
        'attribute_window_diagnostics': attribute_window_diagnostics,
        'attribute_sample_aggregation': attribute_aggregation['sample_aggregation'],
        'attribute_object_aggregation': attribute_aggregation['object_aggregation'],
        'attribute_sample_model_pool': attribute_aggregation['sample_model_pool'],
        'attribute_object_model_pool': attribute_aggregation['object_model_pool'],
        'physical_attribute_metrics': physical_attribute_metrics,
    }
    large_delta_pos_present = large_delta_pos_count > 0
    large_delta_neg_present = large_delta_neg_count > 0
    large_delta_balance_denominator = max(1, int(large_delta_pos_present) + int(large_delta_neg_present))
    metrics['large_delta_balanced_mae'] = (
        (metrics['large_delta_pos_mae'] if large_delta_pos_present else 0.0)
        + (metrics['large_delta_neg_mae'] if large_delta_neg_present else 0.0)
    ) / large_delta_balance_denominator
    metrics['large_delta_balanced_wrong_sign_rate'] = (
        (metrics['large_delta_pos_wrong_sign_rate'] if large_delta_pos_present else 0.0)
        + (metrics['large_delta_neg_wrong_sign_rate'] if large_delta_neg_present else 0.0)
    ) / large_delta_balance_denominator
    metrics['large_delta_pos_neg_mae_gap'] = (
        abs(metrics['large_delta_pos_mae'] - metrics['large_delta_neg_mae'])
        if large_delta_pos_present and large_delta_neg_present
        else 0.0
    )
    metrics['large_delta_pos_neg_wrong_sign_gap'] = (
        abs(metrics['large_delta_pos_wrong_sign_rate'] - metrics['large_delta_neg_wrong_sign_rate'])
        if large_delta_pos_present and large_delta_neg_present
        else 0.0
    )
    metrics['large_delta_balanced_score'] = (
        metrics['large_delta_balanced_mae'] + 100.0 * metrics['large_delta_balanced_wrong_sign_rate']
    )
    finger_large_delta_pos_present = finger_large_delta_pos_count > 0
    finger_large_delta_neg_present = finger_large_delta_neg_count > 0
    finger_large_delta_balance_denominator = max(
        1,
        int(finger_large_delta_pos_present) + int(finger_large_delta_neg_present),
    )
    metrics['finger_large_delta_balanced_mae'] = (
        (metrics['finger_large_delta_pos_mae'] if finger_large_delta_pos_present else 0.0)
        + (metrics['finger_large_delta_neg_mae'] if finger_large_delta_neg_present else 0.0)
    ) / finger_large_delta_balance_denominator
    metrics['finger_large_delta_balanced_wrong_sign_rate'] = (
        (metrics['finger_large_delta_pos_wrong_sign_rate'] if finger_large_delta_pos_present else 0.0)
        + (metrics['finger_large_delta_neg_wrong_sign_rate'] if finger_large_delta_neg_present else 0.0)
    ) / finger_large_delta_balance_denominator
    metrics['finger_large_delta_pos_neg_mae_gap'] = (
        abs(metrics['finger_large_delta_pos_mae'] - metrics['finger_large_delta_neg_mae'])
        if finger_large_delta_pos_present and finger_large_delta_neg_present
        else 0.0
    )
    metrics['finger_large_delta_pos_neg_wrong_sign_gap'] = (
        abs(metrics['finger_large_delta_pos_wrong_sign_rate'] - metrics['finger_large_delta_neg_wrong_sign_rate'])
        if finger_large_delta_pos_present and finger_large_delta_neg_present
        else 0.0
    )
    metrics['finger_large_delta_balanced_score'] = (
        metrics['finger_large_delta_balanced_mae']
        + 100.0 * metrics['finger_large_delta_balanced_wrong_sign_rate']
    )
    metrics.update(flatten_window_metrics(attribute_window_diagnostics))
    metrics.update(flatten_aggregation_metrics(attribute_aggregation))
    metrics.update(flatten_physical_metrics(physical_attribute_metrics))
    metrics.update(direction_metrics.finalize())
    return metrics


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: dict[str, Any],
        stage_config: dict[str, Any],
        model_config: dict[str, Any],
        run_dir: str | Path,
        data_config: dict[str, Any] | None = None,
        config_sources: dict[str, Any] | None = None,
        initialization_info: dict[str, Any] | None = None,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.stage_config = stage_config
        self.model_config = model_config
        self.data_config = data_config or {}
        self.config_sources = config_sources or {}
        self.initialization_info = initialization_info
        self.run_dir = Path(run_dir)
        self.device = torch.device(
            config.get('device', 'cuda') if torch.cuda.is_available() else 'cpu'
        )
        self.model.to(self.device)
        self.freeze_report = freeze_modules(
            self.model,
            stage_config.get('freeze_modules', []),
            strict=True,
            lock_eval=bool(stage_config.get('lock_frozen_modules_eval', True)),
            train_mode_module_names=stage_config.get('frozen_train_mode_modules', []),
        )
        self.optimizer = build_optimizer(self.model, config)
        self.model_parameter_audit = audit_model_parameters(self.model)
        self.optimizer_group_audit = audit_optimizer_groups(self.model, self.optimizer)
        total_steps = max(1, int(config['epochs']) * max(1, len(train_loader)))
        self.scheduler = build_scheduler(
            self.optimizer,
            total_steps=total_steps,
            warmup_ratio=float(config.get('warmup_ratio', 0.0)),
            min_lr_scale=float(config.get('min_lr_scale', 0.1)),
        )
        self.amp_enabled = bool(config.get('amp_enabled', False)) and self.device.type == 'cuda'
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.amp_enabled)
        self.grad_clip_norm = float(config.get('grad_clip_norm', 0.0))
        self.early_stopping_patience = int(config.get('early_stopping_patience', 0))
        self.val_every_n_epochs = max(1, int(config.get('val_every_n_epochs', 1)))
        self.selection_metric = str(stage_config.get('selection_metric', config.get('selection_metric', 'interface_mae')))
        self.maximize_metric = bool(stage_config.get('maximize_metric', config.get('maximize_metric', False)))
        self.tie_breakers = list(stage_config.get('tie_breakers', []))
        self.w2a_retention_guard = stage_config.get(
            'w2a_retention_guard',
            config.get('w2a_retention_guard', {'enabled': False}),
        )
        if not isinstance(self.w2a_retention_guard, dict):
            raise RuntimeError('w2a_retention_guard must be a mapping.')
        self.checkpoint_prefix = str(stage_config.get('checkpoint_prefix', f"stage{stage_config.get('stage_index', 'x')}"))
        self.best_checkpoint_name = str(
            stage_config.get('best_checkpoint_name', f'{self.checkpoint_prefix}_best_{self.selection_metric}.pt')
        )
        self.stage_name = str(stage_config.get('name', self.checkpoint_prefix))
        self.start_epoch = 1

        self.phase_class_weights = build_phase_class_weights(config, self.device)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = self.run_dir / 'checkpoints'
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.generic_latest_path = self.checkpoint_dir / 'latest.pt'
        self.generic_best_path = self.checkpoint_dir / 'best.pt'
        self.named_latest_path = self.checkpoint_dir / f'{self.checkpoint_prefix}_latest.pt'
        self.named_best_path = self.checkpoint_dir / self.best_checkpoint_name
        self.generic_diagnostic_best_path = self.checkpoint_dir / 'diagnostic_best.pt'
        self.named_diagnostic_best_path = self.checkpoint_dir / f'{self.checkpoint_prefix}_diagnostic_best.pt'
        self.best_metrics_path = self.checkpoint_dir / f'{self.checkpoint_prefix}_best_metrics.json'
        self.generic_best_metrics_path = self.checkpoint_dir / 'best_metrics.json'
        self.metrics_jsonl_path = self.run_dir / 'metrics.jsonl'
        self.metrics_csv_path = self.run_dir / 'metrics.csv'
        self.parameter_audit_path = self.run_dir / 'parameter_audit.json'
        parameter_audit_payload = {
            'freeze': self.freeze_report,
            'model': self.model_parameter_audit,
            'optimizer_groups': self.optimizer_group_audit,
        }
        with self.parameter_audit_path.open('w', encoding='utf-8') as handle:
            json.dump(parameter_audit_payload, handle, ensure_ascii=False, indent=2)
        self.parameter_audit_record = {
            'path': str(self.parameter_audit_path.resolve()),
            'sha256': sha256_file(self.parameter_audit_path),
            'trainable_tensor_count': self.model_parameter_audit['trainable_tensor_count'],
            'frozen_tensor_count': self.model_parameter_audit['frozen_tensor_count'],
            'trainable_value_count': self.model_parameter_audit['trainable_value_count'],
            'frozen_value_count': self.model_parameter_audit['frozen_value_count'],
            'trainable_fraction': self.model_parameter_audit['trainable_fraction'],
        }
        with (self.run_dir / 'run_config.yaml').open('w', encoding='utf-8') as handle:
            run_config_payload = {
                'train': config,
                'stage': stage_config,
                'model': model_config,
                'config_sources': self.config_sources,
                'initialization': self.initialization_info,
                'parameter_audit': self.parameter_audit_record,
                'freeze': {
                    key: value
                    for key, value in self.freeze_report.items()
                    if key != 'parameter_names'
                },
                'optimizer_groups': [
                    {
                        key: value
                        for key, value in group.items()
                        if key != 'parameter_names'
                    }
                    for group in self.optimizer_group_audit
                ],
            }
            if self.data_config:
                run_config_payload['data'] = self.data_config
            yaml.safe_dump(
                run_config_payload,
                handle,
                allow_unicode=True,
                sort_keys=False,
            )
        self.writer = None
        if bool(config.get('log_tensorboard', False)) and SummaryWriter is not None:
            self.writer = SummaryWriter(log_dir=str(self.run_dir / 'tensorboard'))

    def _selection_value(self, metrics: dict[str, Any]) -> float:
        completion_key = f'{self.selection_metric}_complete'
        if self.selection_metric.endswith('_macro_direction') and not bool(metrics.get(completion_key, False)):
            raise RuntimeError(
                f'Selection metric {self.selection_metric!r} is incomplete because validation did not '
                'contain valid W2A and A2W supervision.'
            )
        if self.selection_metric == 'joint_score':
            value = float(metrics['joint_score'])
        else:
            value = float(metrics[self.selection_metric])
        if not math.isfinite(value):
            raise RuntimeError(f'Selection metric {self.selection_metric!r} must be finite, got {value}.')
        return value

    def _apply_w2a_guard(self, metrics: dict[str, Any]) -> dict[str, Any]:
        report = evaluate_w2a_retention_guard(metrics, self.w2a_retention_guard)
        metrics['w2a_guard'] = report
        metrics['w2a_guard_enabled'] = bool(report['enabled'])
        metrics['w2a_guard_passed'] = bool(report['passed'])
        if report['enabled']:
            metrics['w2a_guard_current_value'] = float(report['current_value'])
            metrics['w2a_guard_baseline_value'] = float(report['baseline_value'])
            metrics['w2a_guard_threshold'] = float(report['threshold'])
            if report['relative_degradation'] is not None:
                metrics['w2a_guard_relative_degradation'] = float(report['relative_degradation'])
        return report

    def _should_run_validation(self, epoch: int) -> bool:
        if self.val_every_n_epochs <= 1:
            return True
        final_epoch = int(self.config['epochs'])
        return epoch == self.start_epoch or epoch % self.val_every_n_epochs == 0 or epoch == final_epoch

    def _log_scalars(self, metrics: dict[str, Any], epoch: int) -> None:
        if self.writer is None:
            return
        for key, value in metrics.items():
            if isinstance(value, bool):
                continue
            if isinstance(value, (int, float)):
                self.writer.add_scalar(key, float(value), epoch)

    def _write_best_record(self, epoch: int, metrics: dict[str, Any], selection_value: float) -> None:
        payload = {
            'epoch': epoch,
            'selection_metric': self.selection_metric,
            'selection_value': selection_value,
            'metrics': metrics,
        }
        for path in [self.best_metrics_path, self.generic_best_metrics_path]:
            with path.open('w', encoding='utf-8') as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2)

    def _load_existing_best(self) -> dict[str, Any]:
        default = -math.inf if self.maximize_metric else math.inf
        if not self.best_metrics_path.exists():
            return {'selection_value': default, 'metrics': {}, 'epoch': None}
        with self.best_metrics_path.open('r', encoding='utf-8') as handle:
            payload = json.load(handle)
        return {
            'selection_value': float(payload.get('selection_value', default)),
            'metrics': payload.get('metrics', {}),
            'epoch': payload.get('epoch'),
        }

    def _is_better(self, current_metrics: dict[str, Any], current_epoch: int, best_record: dict[str, Any]) -> bool:
        best_epoch = best_record.get('epoch')
        if best_epoch is None:
            return True
        current_selection = float(current_metrics['selection_value'])
        best_selection = float(best_record['selection_value'])
        primary_mode = 'max' if self.maximize_metric else 'min'
        selection_compare = compare_metric(current_selection, best_selection, mode=primary_mode)
        if selection_compare != 0:
            return selection_compare > 0

        best_metrics = best_record.get('metrics', {})
        for tie_breaker in self.tie_breakers:
            metric_name = str(tie_breaker['metric'])
            mode = str(tie_breaker.get('mode', 'min'))
            current_value = numeric_value(current_metrics, metric_name, current_epoch)
            best_value = numeric_value(best_metrics, metric_name, int(best_epoch))
            result = compare_metric(current_value, best_value, mode=mode)
            if result != 0:
                return result > 0
        return False

    def _serialize_metrics_row(self, metrics: dict[str, Any]) -> dict[str, Any]:
        row: dict[str, Any] = {}
        for key, value in metrics.items():
            if isinstance(value, (int, float, str)):
                row[key] = value
            else:
                row[key] = json.dumps(value, ensure_ascii=False)
        return row

    def _append_metrics_csv(self, metrics: dict[str, Any]) -> None:
        row = self._serialize_metrics_row(metrics)
        file_exists = self.metrics_csv_path.exists()
        with self.metrics_csv_path.open('a', encoding='utf-8', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)

    def load_checkpoint(self, path: str | Path) -> dict[str, Any]:
        checkpoint_path = Path(path)
        expected_parent = self.checkpoint_dir.resolve()
        if checkpoint_path.resolve().parent != expected_parent:
            raise ValueError(
                f'--resume only supports checkpoints from the same run/stage. '
                f'Expected a checkpoint under {expected_parent}, got {checkpoint_path.resolve().parent}.'
            )
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        checkpoint_stage = checkpoint.get('stage_name')
        if checkpoint_stage is not None and str(checkpoint_stage) != self.stage_name:
            raise ValueError(
                f'Checkpoint stage {checkpoint_stage!r} does not match current stage {self.stage_name!r}. '
                'Use --init-from for cross-stage weight initialization.'
            )
        checkpoint_run_dir = checkpoint.get('run_dir')
        if checkpoint_run_dir is not None and Path(checkpoint_run_dir).resolve() != self.run_dir.resolve():
            raise ValueError(
                f'Checkpoint run_dir {checkpoint_run_dir!r} does not match current run_dir {str(self.run_dir.resolve())!r}. '
                'Use --init-from for cross-stage weight initialization.'
            )
        self.model.load_state_dict(checkpoint['model'])
        if 'optimizer' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
        if 'scheduler' in checkpoint and checkpoint['scheduler'] is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler'])
        if 'scaler' in checkpoint and checkpoint['scaler'] is not None:
            self.scaler.load_state_dict(checkpoint['scaler'])
        self.start_epoch = int(checkpoint.get('epoch', 0)) + 1
        return checkpoint.get('metrics', {})

    def fit(self) -> dict[str, Any]:
        best_record = self._load_existing_best()
        diagnostic_record = {'selection_value': -math.inf if self.maximize_metric else math.inf, 'metrics': {}, 'epoch': None}
        best_metrics = best_record.get('metrics', {})
        latest_val_metrics = best_record.get('metrics', {}).copy() if isinstance(best_record.get('metrics'), dict) else {}
        stale_epochs = 0
        try:
            for epoch in range(self.start_epoch, int(self.config['epochs']) + 1):
                train_metrics = self.run_epoch(self.train_loader, training=True)
                train_metrics.update(collect_sampler_epoch_metrics(self.train_loader))
                validation_ran = self._should_run_validation(epoch)
                val_metrics = None
                if validation_ran:
                    val_metrics = self.run_epoch(self.val_loader, training=False)
                    val_metrics['selection_value'] = self._selection_value(val_metrics)
                    self._apply_w2a_guard(val_metrics)
                    latest_val_metrics = val_metrics.copy()
                metrics = {f'train_{k}': v for k, v in train_metrics.items()}
                metrics.update({f'val_{k}': v for k, v in latest_val_metrics.items()})
                metrics['epoch'] = epoch
                metrics['learning_rate'] = float(self.optimizer.param_groups[0]['lr'])
                for group_index, group in enumerate(self.optimizer.param_groups):
                    group_name = str(group.get('name', f'group{group_index}'))
                    metrics[f'learning_rate_{group_name}'] = float(group['lr'])
                with self.metrics_jsonl_path.open('a', encoding='utf-8') as handle:
                    handle.write(json.dumps(metrics, ensure_ascii=False) + '\n')
                self._append_metrics_csv(metrics)
                self._log_scalars(metrics, epoch)
                latest_metrics_for_checkpoint = val_metrics if val_metrics is not None else latest_val_metrics
                self.save_checkpoint(self.generic_latest_path, epoch, latest_metrics_for_checkpoint)
                self.save_checkpoint(self.named_latest_path, epoch, latest_metrics_for_checkpoint)
                if val_metrics is not None:
                    if self._is_better(val_metrics, epoch, diagnostic_record):
                        diagnostic_record = {
                            'selection_value': float(val_metrics['selection_value']),
                            'metrics': val_metrics,
                            'epoch': epoch,
                        }
                        self.save_checkpoint(self.generic_diagnostic_best_path, epoch, val_metrics)
                        self.save_checkpoint(self.named_diagnostic_best_path, epoch, val_metrics)
                    checkpoint_eligible = bool(val_metrics.get('w2a_guard_passed', True))
                    if checkpoint_eligible and self._is_better(val_metrics, epoch, best_record):
                        best_record = {
                            'selection_value': float(val_metrics['selection_value']),
                            'metrics': val_metrics,
                            'epoch': epoch,
                        }
                        best_metrics = val_metrics
                        self.save_checkpoint(self.generic_best_path, epoch, val_metrics)
                        self.save_checkpoint(self.named_best_path, epoch, val_metrics)
                        self._write_best_record(epoch, val_metrics, float(val_metrics['selection_value']))
                        stale_epochs = 0
                    else:
                        stale_epochs += 1
                    if self.early_stopping_patience > 0 and stale_epochs >= self.early_stopping_patience:
                        break
        finally:
            if self.writer is not None:
                self.writer.close()
        return best_metrics

    def save_checkpoint(self, path: Path, epoch: int, metrics: dict[str, Any]) -> None:
        torch.save(
            {
                'epoch': epoch,
                'model': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'scheduler': self.scheduler.state_dict(),
                'scaler': self.scaler.state_dict(),
                'metrics': metrics,
                'stage_name': self.stage_name,
                'run_dir': str(self.run_dir.resolve()),
                'run_config_path': str((self.run_dir / 'run_config.yaml').resolve()),
                'checkpoint_prefix': self.checkpoint_prefix,
                'config_sources': self.config_sources,
                'initialization': self.initialization_info,
                'parameter_audit': self.parameter_audit_record,
            },
            path,
        )

    def run_epoch(self, loader: DataLoader, training: bool) -> dict[str, Any]:
        return run_model_epoch(
            self.model,
            loader,
            device=self.device,
            training=training,
            stage_config=self.stage_config,
            model_config=self.model_config,
            phase_class_weights=self.phase_class_weights,
            amp_enabled=self.amp_enabled,
            optimizer=self.optimizer if training else None,
            scheduler=self.scheduler if training else None,
            scaler=self.scaler if training else None,
            grad_clip_norm=self.grad_clip_norm,
            loss_reduction_config=self.config.get('loss_reduction'),
        )
