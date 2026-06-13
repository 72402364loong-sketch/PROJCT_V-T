from __future__ import annotations

from typing import Any

import math

import torch


PHYSICAL_OUTPUT_KEYS = (
    'sample_dry_mass_g_normalized_pred',
    'sample_capacity_ratio_normalized_pred',
    'sample_is_open_container_logit',
)


def _normalization_params(stats: dict[str, Any] | None, attribute: str) -> tuple[float, float]:
    continuous = stats.get('continuous', {}) if isinstance(stats, dict) else {}
    payload = continuous.get(attribute, {}) if isinstance(continuous, dict) else {}
    mean = float(payload.get('mean', 0.0)) if isinstance(payload, dict) else 0.0
    std = float(payload.get('std', 1.0)) if isinstance(payload, dict) else 1.0
    if not math.isfinite(std) or std < 1e-6:
        std = 1.0
    return mean, std


def _rankdata(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    cursor = 0
    while cursor < len(order):
        end = cursor + 1
        while end < len(order) and values[order[end]] == values[order[cursor]]:
            end += 1
        average_rank = 0.5 * (cursor + end - 1) + 1.0
        for offset in range(cursor, end):
            ranks[order[offset]] = average_rank
        cursor = end
    return ranks


def spearman(predictions: list[float], targets: list[float]) -> float:
    if len(predictions) < 2 or len(targets) < 2:
        return 0.0
    pred_ranks = _rankdata(predictions)
    target_ranks = _rankdata(targets)
    pred_mean = sum(pred_ranks) / len(pred_ranks)
    target_mean = sum(target_ranks) / len(target_ranks)
    numerator = sum((p - pred_mean) * (t - target_mean) for p, t in zip(pred_ranks, target_ranks))
    pred_var = sum((p - pred_mean) ** 2 for p in pred_ranks)
    target_var = sum((t - target_mean) ** 2 for t in target_ranks)
    denom = math.sqrt(pred_var * target_var)
    if denom <= 1e-12:
        return 0.0
    return float(numerator / denom)


def update_binary_confusion(confusion: torch.Tensor, logits: torch.Tensor, targets: torch.Tensor) -> None:
    preds = (torch.sigmoid(logits.detach().reshape(-1).cpu()) >= 0.5).to(dtype=torch.long)
    labels = targets.detach().reshape(-1).cpu().to(dtype=torch.long)
    valid = (labels >= 0) & (labels <= 1)
    if not valid.any():
        return
    flat = labels[valid] * 2 + preds[valid]
    confusion += torch.bincount(flat, minlength=4).reshape(2, 2)


def accuracy_from_confusion(confusion: torch.Tensor) -> float:
    total = int(confusion.sum().item())
    if total == 0:
        return 0.0
    return float(torch.diag(confusion).sum().item() / total)


def class_f1(confusion: torch.Tensor, index: int) -> float:
    tp = float(confusion[index, index].item())
    fp = float(confusion[:, index].sum().item() - tp)
    fn = float(confusion[index, :].sum().item() - tp)
    denom = 2.0 * tp + fp + fn
    return 0.0 if denom <= 0.0 else float((2.0 * tp) / denom)


def macro_f1(confusion: torch.Tensor) -> float:
    return float((class_f1(confusion, 0) + class_f1(confusion, 1)) / 2.0)


class ContinuousMetricStore:
    def __init__(self) -> None:
        self.predictions: list[float] = []
        self.targets: list[float] = []

    def extend(self, predictions: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor | None = None) -> None:
        pred = predictions.detach().reshape(-1).cpu()
        target = targets.detach().reshape(-1).cpu()
        if mask is not None:
            selected = mask.detach().reshape(-1).bool().cpu()
            pred = pred[selected]
            target = target[selected]
        if pred.numel() == 0:
            return
        self.predictions.extend(float(value) for value in pred.tolist())
        self.targets.extend(float(value) for value in target.tolist())

    def finalize(self, prefix: str) -> dict[str, Any]:
        count = len(self.predictions)
        if count == 0:
            return {
                f'{prefix}_count': 0,
                f'{prefix}_mae': 0.0,
                f'{prefix}_rmse': 0.0,
                f'{prefix}_spearman': 0.0,
            }
        errors = [pred - target for pred, target in zip(self.predictions, self.targets)]
        mae = sum(abs(error) for error in errors) / count
        rmse = math.sqrt(sum(error * error for error in errors) / count)
        return {
            f'{prefix}_count': int(count),
            f'{prefix}_mae': float(mae),
            f'{prefix}_rmse': float(rmse),
            f'{prefix}_spearman': spearman(self.predictions, self.targets),
        }


class PhysicalAttributeMetricAccumulator:
    def __init__(self, physical_stats: dict[str, Any] | None = None) -> None:
        self.dry_mean, self.dry_std = _normalization_params(physical_stats, 'dry_mass_g')
        self.capacity_mean, self.capacity_std = _normalization_params(physical_stats, 'capacity_ratio')

        self.sample_dry = ContinuousMetricStore()
        self.sample_capacity = ContinuousMetricStore()
        self.sample_open_confusion = torch.zeros(2, 2, dtype=torch.long)

        self.object_dry_predictions: dict[int, list[float]] = {}
        self.object_dry_targets: dict[int, float] = {}
        self.object_capacity_predictions: dict[int, list[float]] = {}
        self.object_capacity_targets: dict[int, float] = {}
        self.object_open_logits: dict[int, list[float]] = {}
        self.object_open_targets: dict[int, int] = {}

    def _unnormalize_dry(self, normalized: torch.Tensor) -> torch.Tensor:
        return normalized * float(self.dry_std) + float(self.dry_mean)

    def _unnormalize_capacity(self, normalized: torch.Tensor) -> torch.Tensor:
        return normalized * float(self.capacity_std) + float(self.capacity_mean)

    def update(self, outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> None:
        if not all(key in outputs for key in PHYSICAL_OUTPUT_KEYS):
            return

        dry_pred = self._unnormalize_dry(outputs['sample_dry_mass_g_normalized_pred'].reshape(-1)).detach().cpu()
        capacity_pred = self._unnormalize_capacity(outputs['sample_capacity_ratio_normalized_pred'].reshape(-1)).detach().cpu()
        open_logits = outputs['sample_is_open_container_logit'].reshape(-1).detach().cpu()

        dry_target = batch['dry_mass_g'].reshape(-1).detach().cpu()
        capacity_target = batch['capacity_ratio'].reshape(-1).detach().cpu()
        capacity_valid = batch['capacity_valid_mask'].reshape(-1).detach().cpu().bool()
        open_target = batch['is_open_container'].reshape(-1).detach().cpu().to(dtype=torch.long)
        object_indices = batch['object_index'].reshape(-1).detach().cpu().to(dtype=torch.long)

        self.sample_dry.extend(dry_pred, dry_target)
        self.sample_capacity.extend(capacity_pred, capacity_target, capacity_valid)
        update_binary_confusion(self.sample_open_confusion, open_logits, open_target)

        for row in range(int(object_indices.shape[0])):
            object_index = int(object_indices[row].item())
            self.object_dry_predictions.setdefault(object_index, []).append(float(dry_pred[row].item()))
            self.object_dry_targets.setdefault(object_index, float(dry_target[row].item()))
            self.object_open_logits.setdefault(object_index, []).append(float(open_logits[row].item()))
            self.object_open_targets.setdefault(object_index, int(open_target[row].item()))
            if bool(capacity_valid[row].item()):
                self.object_capacity_predictions.setdefault(object_index, []).append(float(capacity_pred[row].item()))
                self.object_capacity_targets.setdefault(object_index, float(capacity_target[row].item()))

    def _finalize_object_continuous(
        self,
        predictions_by_object: dict[int, list[float]],
        targets_by_object: dict[int, float],
        prefix: str,
    ) -> dict[str, Any]:
        store = ContinuousMetricStore()
        predictions = []
        targets = []
        for object_index, values in predictions_by_object.items():
            if not values or object_index not in targets_by_object:
                continue
            predictions.append(sum(values) / len(values))
            targets.append(float(targets_by_object[object_index]))
        if predictions:
            store.extend(torch.tensor(predictions, dtype=torch.float32), torch.tensor(targets, dtype=torch.float32))
        return store.finalize(prefix)

    def _finalize_object_open(self) -> dict[str, Any]:
        confusion = torch.zeros(2, 2, dtype=torch.long)
        logits = []
        targets = []
        for object_index, values in self.object_open_logits.items():
            if not values or object_index not in self.object_open_targets:
                continue
            logits.append(sum(values) / len(values))
            targets.append(int(self.object_open_targets[object_index]))
        if logits:
            update_binary_confusion(
                confusion,
                torch.tensor(logits, dtype=torch.float32),
                torch.tensor(targets, dtype=torch.long),
            )
        return {
            'is_open_count': int(confusion.sum().item()),
            'is_open_accuracy': accuracy_from_confusion(confusion),
            'is_open_macro_f1': macro_f1(confusion),
            'is_open_confusion': confusion.tolist(),
        }

    def finalize(self) -> dict[str, dict[str, Any]]:
        sample_metrics = {}
        sample_metrics.update(self.sample_dry.finalize('dry_mass_g'))
        sample_metrics.update(self.sample_capacity.finalize('capacity_ratio'))
        sample_metrics.update(
            {
                'is_open_count': int(self.sample_open_confusion.sum().item()),
                'is_open_accuracy': accuracy_from_confusion(self.sample_open_confusion),
                'is_open_macro_f1': macro_f1(self.sample_open_confusion),
                'is_open_confusion': self.sample_open_confusion.tolist(),
            }
        )

        object_metrics = {}
        object_metrics.update(
            self._finalize_object_continuous(
                self.object_dry_predictions,
                self.object_dry_targets,
                'dry_mass_g',
            )
        )
        object_metrics.update(
            self._finalize_object_continuous(
                self.object_capacity_predictions,
                self.object_capacity_targets,
                'capacity_ratio',
            )
        )
        object_metrics.update(self._finalize_object_open())
        return {
            'sample': sample_metrics,
            'object': object_metrics,
        }


def flatten_physical_metrics(metrics: dict[str, dict[str, Any]]) -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for level in ('sample', 'object'):
        for key, value in metrics.get(level, {}).items():
            if isinstance(value, (int, float)):
                flat[f'physical_{level}_{key}'] = value
    return flat
