from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from cmg.constants import DIRECTION_TO_INDEX


def _accuracy(confusion: torch.Tensor) -> float:
    total = int(confusion.sum().item())
    return float(torch.diag(confusion).sum().item() / total) if total else 0.0


def _class_f1(confusion: torch.Tensor, index: int) -> float:
    true_positive = int(confusion[index, index].item())
    false_positive = int(confusion[:, index].sum().item()) - true_positive
    false_negative = int(confusion[index, :].sum().item()) - true_positive
    denominator = 2 * true_positive + false_positive + false_negative
    return float(2 * true_positive / denominator) if denominator else 0.0


def _harmonic_mean(first: float, second: float) -> float:
    denominator = float(first) + float(second)
    return float(2.0 * float(first) * float(second) / denominator) if denominator > 0.0 else 0.0


@dataclass
class _DirectionBucket:
    sample_count: int = 0
    medium_confusion: torch.Tensor = field(default_factory=lambda: torch.zeros(3, 3, dtype=torch.long))
    finger_control_interface_abs: float = 0.0
    finger_control_interface_count: int = 0
    finger_delta_interface_abs: float = 0.0
    finger_delta_interface_count: int = 0
    finger_large_delta_wrong_sign: int = 0
    finger_large_delta_count: int = 0
    stable_leakage: list[torch.Tensor] = field(default_factory=list)
    gate_stable_sum: float = 0.0
    gate_stable_count: int = 0
    gate_interface_miss_sum: float = 0.0
    gate_interface_count: int = 0


class DirectionMetricAccumulator:
    """Accumulate core Medium/Policy metrics separately for W2A and A2W."""

    def __init__(self, *, finger_large_delta_threshold: float = 100.0) -> None:
        self.finger_large_delta_threshold = float(finger_large_delta_threshold)
        self.buckets = {name.lower(): _DirectionBucket() for name in DIRECTION_TO_INDEX}

    def update(self, outputs: dict[str, torch.Tensor], batch: dict[str, Any]) -> None:
        if 'direction_ids' not in batch:
            raise RuntimeError('Direction metrics require batch["direction_ids"].')
        direction_ids = batch['direction_ids'].detach().reshape(-1)
        batch_size = int(batch['window_mask'].shape[0])
        if direction_ids.shape[0] != batch_size:
            raise RuntimeError('Direction metrics require one direction id per sample.')
        invalid = (direction_ids < 0) | (direction_ids >= len(DIRECTION_TO_INDEX))
        if invalid.any():
            raise RuntimeError('Direction metrics received an id outside the canonical W2A/A2W range.')

        window_mask = batch['window_mask'].bool()
        phase_targets = batch['phase_labels']
        phase_preds = outputs['medium_logits'].argmax(dim=-1)

        for direction_name, direction_index in DIRECTION_TO_INDEX.items():
            key = direction_name.lower()
            bucket = self.buckets[key]
            sample_mask = direction_ids == direction_index
            bucket.sample_count += int(sample_mask.sum().item())
            if not sample_mask.any():
                continue
            direction_windows = window_mask & sample_mask[:, None]
            targets = phase_targets[direction_windows].detach().cpu()
            predictions = phase_preds[direction_windows].detach().cpu()
            for prediction, target in zip(predictions, targets):
                bucket.medium_confusion[int(target), int(prediction)] += 1

            scalar_delta = outputs.get('force_interface_delta')
            scalar_gate = outputs.get('interface_gate')
            if scalar_delta is not None and scalar_gate is not None:
                scalar_delta = scalar_delta.reshape_as(window_mask)
                scalar_gate = scalar_gate.reshape_as(window_mask)
                stable_mask = direction_windows & batch['stable_masks'].bool()
                if stable_mask.any():
                    leakage = (scalar_gate * scalar_delta).abs()[stable_mask].detach().cpu()
                    bucket.stable_leakage.append(leakage)
                    bucket.gate_stable_sum += float(scalar_gate[stable_mask].sum().item())
                    bucket.gate_stable_count += int(stable_mask.sum().item())
                interface_mask = direction_windows & (phase_targets == 1)
                if interface_mask.any():
                    bucket.gate_interface_miss_sum += float((1.0 - scalar_gate[interface_mask]).abs().sum().item())
                    bucket.gate_interface_count += int(interface_mask.sum().item())

            required_finger = {
                'finger_force_pred',
                'finger_force_interface_delta',
            }
            required_batch = {
                'finger_control_force_targets',
                'finger_delta_force_targets',
                'has_finger_control_target',
                'delta_supervision_masks',
            }
            if not required_finger.issubset(outputs) or not required_batch.issubset(batch):
                continue
            finger_target = batch['finger_control_force_targets']
            finger_pred = outputs['finger_force_pred'].reshape_as(finger_target)
            finger_delta_target = batch['finger_delta_force_targets']
            finger_delta_pred = outputs['finger_force_interface_delta'].reshape_as(finger_delta_target)
            finger_direction_windows = direction_windows.unsqueeze(-1).expand_as(finger_target)
            delta_mask = (
                batch['delta_supervision_masks'].bool().unsqueeze(-1).expand_as(finger_target)
                & batch['has_finger_control_target'].bool()
                & finger_direction_windows
            )
            if not delta_mask.any():
                continue
            control_error = finger_pred[delta_mask] - finger_target[delta_mask]
            delta_error = finger_delta_pred[delta_mask] - finger_delta_target[delta_mask]
            bucket.finger_control_interface_abs += float(control_error.abs().sum().item())
            bucket.finger_control_interface_count += int(control_error.numel())
            bucket.finger_delta_interface_abs += float(delta_error.abs().sum().item())
            bucket.finger_delta_interface_count += int(delta_error.numel())
            selected_pred = finger_delta_pred[delta_mask]
            selected_target = finger_delta_target[delta_mask]
            large_mask = selected_target.abs() >= self.finger_large_delta_threshold
            if large_mask.any():
                large_pred = selected_pred[large_mask]
                large_target = selected_target[large_mask]
                bucket.finger_large_delta_wrong_sign += int((large_pred * large_target < 0.0).sum().item())
                bucket.finger_large_delta_count += int(large_target.numel())

    @staticmethod
    def _safe_mean(total: float, count: int) -> float:
        return float(total / count) if count else 0.0

    def _finalize_bucket(self, bucket: _DirectionBucket) -> dict[str, Any]:
        confusion = bucket.medium_confusion
        stable_leakage = torch.cat(bucket.stable_leakage) if bucket.stable_leakage else torch.empty(0)
        medium_f1_water = _class_f1(confusion, 0)
        medium_f1_interface = _class_f1(confusion, 1)
        medium_f1_air = _class_f1(confusion, 2)
        return {
            'sample_count': bucket.sample_count,
            'medium_window_count': int(confusion.sum().item()),
            'medium_accuracy': _accuracy(confusion),
            'medium_macro_f1': (medium_f1_water + medium_f1_interface + medium_f1_air) / 3.0,
            'medium_f1_water': medium_f1_water,
            'medium_f1_interface': medium_f1_interface,
            'medium_f1_air': medium_f1_air,
            'medium_water_interface_hmean': _harmonic_mean(medium_f1_water, medium_f1_interface),
            'medium_confusion': confusion.tolist(),
            'finger_control_interface_count': bucket.finger_control_interface_count,
            'finger_control_interface_mae': self._safe_mean(
                bucket.finger_control_interface_abs,
                bucket.finger_control_interface_count,
            ),
            'finger_delta_interface_count': bucket.finger_delta_interface_count,
            'finger_delta_interface_mae': self._safe_mean(
                bucket.finger_delta_interface_abs,
                bucket.finger_delta_interface_count,
            ),
            'finger_large_delta_count': bucket.finger_large_delta_count,
            'finger_large_delta_wrong_sign_rate': self._safe_mean(
                float(bucket.finger_large_delta_wrong_sign),
                bucket.finger_large_delta_count,
            ),
            'stable_leakage_count': int(stable_leakage.numel()),
            'stable_leakage_mean': float(stable_leakage.mean().item()) if stable_leakage.numel() else 0.0,
            'stable_leakage_p95': float(torch.quantile(stable_leakage, 0.95).item()) if stable_leakage.numel() else 0.0,
            'gate_false_positive_stable': self._safe_mean(bucket.gate_stable_sum, bucket.gate_stable_count),
            'gate_false_negative_interface': self._safe_mean(
                bucket.gate_interface_miss_sum,
                bucket.gate_interface_count,
            ),
        }

    def finalize(self) -> dict[str, Any]:
        by_direction = {key: self._finalize_bucket(bucket) for key, bucket in self.buckets.items()}
        flat: dict[str, Any] = {'direction_metrics': by_direction}
        for key, values in by_direction.items():
            for metric_name, value in values.items():
                flat[f'{metric_name}_{key}'] = value

        macro_specs = {
            'medium_accuracy': 'medium_window_count',
            'medium_macro_f1': 'medium_window_count',
            'medium_f1_water': 'medium_window_count',
            'medium_f1_interface': 'medium_window_count',
            'medium_f1_air': 'medium_window_count',
            'medium_water_interface_hmean': 'medium_window_count',
            'finger_control_interface_mae': 'finger_control_interface_count',
            'finger_delta_interface_mae': 'finger_delta_interface_count',
            'finger_large_delta_wrong_sign_rate': 'finger_large_delta_count',
            'stable_leakage_mean': 'stable_leakage_count',
            'stable_leakage_p95': 'stable_leakage_count',
            'gate_false_positive_stable': 'sample_count',
            'gate_false_negative_interface': 'sample_count',
        }
        macro_complete = True
        for metric_name, count_name in macro_specs.items():
            present = [
                float(values[metric_name])
                for values in by_direction.values()
                if int(values[count_name]) > 0
            ]
            complete = len(present) == len(DIRECTION_TO_INDEX)
            macro_complete &= complete
            flat[f'{metric_name}_macro_direction'] = float(sum(present) / len(present)) if present else 0.0
            flat[f'{metric_name}_macro_direction_complete'] = complete
        flat['direction_macro_complete'] = macro_complete
        return flat
