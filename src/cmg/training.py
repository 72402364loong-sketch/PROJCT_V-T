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

from cmg.losses import compute_losses

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
    checkpoint_path = Path(path)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
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
    )



def load_model_weights(
    model: nn.Module,
    path: str | Path,
    *,
    strict: bool = True,
    allow_lora_injection: bool = False,
) -> dict[str, Any]:
    context = load_checkpoint_context(path)
    checkpoint = context['checkpoint']
    state_dict = checkpoint['model'] if isinstance(checkpoint, dict) and 'model' in checkpoint else checkpoint
    remapped_keys: dict[str, str] = {}
    if allow_lora_injection:
        state_dict, remapped_keys = remap_open_clip_lora_keys(state_dict, set(model.state_dict().keys()))
        strict = False
    incompatible = model.load_state_dict(state_dict, strict=strict)
    missing_keys = list(incompatible.missing_keys)
    unexpected_keys = list(incompatible.unexpected_keys)
    if allow_lora_injection:
        disallowed_missing = [key for key in missing_keys if not is_allowed_lora_injection_mismatch(key)]
        disallowed_unexpected = [key for key in unexpected_keys if not is_allowed_lora_injection_mismatch(key)]
        if disallowed_missing or disallowed_unexpected:
            details = {
                'missing_keys': missing_keys,
                'unexpected_keys': unexpected_keys,
                'remapped_keys': remapped_keys,
            }
            raise RuntimeError(
                'Checkpoint initialization encountered mismatched keys beyond the expected stage-to-stage architecture delta: '
                + json.dumps(details, ensure_ascii=False)
            )
    return {
        'checkpoint_path': context['checkpoint_path'],
        'missing_keys': missing_keys,
        'unexpected_keys': unexpected_keys,
        'remapped_keys': remapped_keys,
        'allow_lora_injection': allow_lora_injection,
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


def freeze_modules(model: nn.Module, module_names: list[str]) -> None:
    for module_name in module_names:
        module = resolve_module(model, module_name)
        if module is None:
            continue
        for parameter in module.parameters():
            parameter.requires_grad = False


def build_optimizer(model: nn.Module, config: dict[str, Any]) -> AdamW:
    base_lr = float(config.get('base_learning_rate', config.get('learning_rate', 1e-4)))
    lora_lr = float(config.get('lora_learning_rate', base_lr))
    weight_decay = float(config['weight_decay'])

    base_params = []
    lora_params = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if 'lora_' in name:
            lora_params.append(parameter)
        else:
            base_params.append(parameter)

    param_groups: list[dict[str, Any]] = []
    if base_params:
        param_groups.append({'params': base_params, 'lr': base_lr, 'name': 'base'})
    if lora_params:
        param_groups.append({'params': lora_params, 'lr': lora_lr, 'name': 'lora'})
    if not param_groups:
        raise RuntimeError('No trainable parameters found for optimizer setup.')
    return AdamW(param_groups, weight_decay=weight_decay)


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
) -> dict[str, Any]:
    if training and (optimizer is None or scheduler is None or scaler is None):
        raise RuntimeError('Training epoch execution requires optimizer, scheduler, and GradScaler.')

    mode = 'train' if training else 'eval'
    model.train(training)
    loss_sums = {'total': 0.0, 'clip': 0.0, 'inv': 0.0, 'med': 0.0, 'attr': 0.0, 'pol': 0.0}
    phase_confusion = torch.zeros(3, 3, dtype=torch.long)
    frag_confusion = torch.zeros(3, 3, dtype=torch.long)
    geom_confusion = torch.zeros(4, 4, dtype=torch.long)
    surf_confusion = torch.zeros(2, 2, dtype=torch.long)
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
    combined_attr_correct = 0
    combined_attr_count = 0

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
                    policy_loss_config=stage_config.get('policy_loss'),
                    temperature_clip=float(model_config['losses']['temperature_clip']),
                    temperature_inv=float(model_config['losses']['temperature_inv']),
                    phase_class_weights=phase_class_weights,
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

        repeated_frag = batch['fragility_label'][:, None].expand(-1, window_mask.shape[1]).reshape(-1)[flat_valid].detach().cpu()
        repeated_geom = batch['geometry_label'][:, None].expand(-1, window_mask.shape[1]).reshape(-1)[flat_valid].detach().cpu()
        repeated_surf = batch['surface_label'][:, None].expand(-1, window_mask.shape[1]).reshape(-1)[flat_valid].detach().cpu()
        frag_preds = outputs['fragility_logits'][flat_valid].argmax(dim=-1).detach().cpu()
        geom_preds = outputs['geometry_logits'][flat_valid].argmax(dim=-1).detach().cpu()
        surf_preds = outputs['surface_logits'][flat_valid].argmax(dim=-1).detach().cpu()
        for pred, target in zip(frag_preds, repeated_frag):
            frag_confusion[int(target), int(pred)] += 1
        for pred, target in zip(geom_preds, repeated_geom):
            geom_confusion[int(target), int(pred)] += 1
        for pred, target in zip(surf_preds, repeated_surf):
            surf_confusion[int(target), int(pred)] += 1
        combined_attr_correct += int(((frag_preds == repeated_frag) & (geom_preds == repeated_geom) & (surf_preds == repeated_surf)).sum().item())
        combined_attr_count += int(frag_preds.numel())

        force_pred = outputs['force_pred'].reshape_as(batch['expert_forces'])
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

    denom = max(1, len(loader))
    frag_macro_f1 = macro_f1_from_confusion(frag_confusion)
    geom_macro_f1 = macro_f1_from_confusion(geom_confusion)
    surf_macro_f1 = macro_f1_from_confusion(surf_confusion)
    metrics: dict[str, Any] = {
        'loss_total': loss_sums['total'] / denom,
        'loss_clip': loss_sums['clip'] / denom,
        'loss_inv': loss_sums['inv'] / denom,
        'loss_med': loss_sums['med'] / denom,
        'loss_attr': loss_sums['attr'] / denom,
        'loss_pol': loss_sums['pol'] / denom,
        'contrastive_loss_sum': (loss_sums['clip'] + loss_sums['inv']) / denom,
        'medium_accuracy': accuracy_from_confusion(phase_confusion),
        'medium_macro_f1': macro_f1_from_confusion(phase_confusion),
        'medium_f1_water': class_f1_from_confusion(phase_confusion, 0),
        'medium_f1_interface': class_f1_from_confusion(phase_confusion, 1),
        'medium_f1_air': class_f1_from_confusion(phase_confusion, 2),
        'fragility_accuracy': accuracy_from_confusion(frag_confusion),
        'geometry_accuracy': accuracy_from_confusion(geom_confusion),
        'surface_accuracy': accuracy_from_confusion(surf_confusion),
        'fragility_macro_f1': frag_macro_f1,
        'geometry_macro_f1': geom_macro_f1,
        'surface_macro_f1': surf_macro_f1,
        'attr_macro_f1_avg': (frag_macro_f1 + geom_macro_f1 + surf_macro_f1) / 3.0,
        'combined_attr_accuracy': combined_attr_correct / max(1, combined_attr_count),
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
        'joint_score': 0.6 * macro_f1_from_confusion(phase_confusion) + 0.4 * ((frag_macro_f1 + geom_macro_f1 + surf_macro_f1) / 3.0),
        'medium_confusion': phase_confusion.tolist(),
        'fragility_confusion': frag_confusion.tolist(),
        'geometry_confusion': geom_confusion.tolist(),
        'surface_confusion': surf_confusion.tolist(),
    }
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
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.stage_config = stage_config
        self.model_config = model_config
        self.data_config = data_config or {}
        self.run_dir = Path(run_dir)
        self.device = torch.device(
            config.get('device', 'cuda') if torch.cuda.is_available() else 'cpu'
        )
        self.model.to(self.device)
        freeze_modules(self.model, stage_config.get('freeze_modules', []))
        self.optimizer = build_optimizer(self.model, config)
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
        self.best_metrics_path = self.checkpoint_dir / f'{self.checkpoint_prefix}_best_metrics.json'
        self.generic_best_metrics_path = self.checkpoint_dir / 'best_metrics.json'
        self.metrics_jsonl_path = self.run_dir / 'metrics.jsonl'
        self.metrics_csv_path = self.run_dir / 'metrics.csv'
        with (self.run_dir / 'run_config.yaml').open('w', encoding='utf-8') as handle:
            run_config_payload = {
                'train': config,
                'stage': stage_config,
                'model': model_config,
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
        if self.selection_metric == 'joint_score':
            return float(0.6 * metrics['medium_macro_f1'] + 0.4 * metrics['attr_macro_f1_avg'])
        return float(metrics[self.selection_metric])

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
        best_metrics = best_record.get('metrics', {})
        latest_val_metrics = best_record.get('metrics', {}).copy() if isinstance(best_record.get('metrics'), dict) else {}
        stale_epochs = 0
        try:
            for epoch in range(self.start_epoch, int(self.config['epochs']) + 1):
                train_metrics = self.run_epoch(self.train_loader, training=True)
                validation_ran = self._should_run_validation(epoch)
                val_metrics = None
                if validation_ran:
                    val_metrics = self.run_epoch(self.val_loader, training=False)
                    val_metrics['selection_value'] = self._selection_value(val_metrics)
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
                    if self._is_better(val_metrics, epoch, best_record):
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
        )











