from __future__ import annotations

from typing import Any

import torch


ATTRIBUTE_TASKS = ('fragility', 'geometry', 'surface')
ATTRIBUTE_LOGIT_KEYS = {
    'fragility': 'fragility_logits',
    'geometry': 'geometry_logits',
    'surface': 'surface_logits',
}
ATTRIBUTE_SAMPLE_LOGIT_KEYS = {
    'fragility': 'sample_fragility_logits',
    'geometry': 'sample_geometry_logits',
    'surface': 'sample_surface_logits',
}
ATTRIBUTE_LABEL_KEYS = {
    'fragility': 'fragility_label',
    'geometry': 'geometry_label',
    'surface': 'surface_label',
}
ATTRIBUTE_WINDOW_SLICES = (
    'all_windows',
    'water_only',
    'interface_only',
    'air_only',
    'stable_windows',
    'stable_water_only',
    'stable_air_only',
    'stable_water_air',
)
ATTRIBUTE_AGGREGATION_SLICES = (
    'all_windows',
    'stable_water_air',
    'water_only',
    'air_only',
)
ATTRIBUTE_AGGREGATION_METHODS = ('mean_logits', 'mean_probs')


def attribute_class_counts(model_config: dict[str, Any]) -> dict[str, int]:
    attributes_config = model_config.get('attributes', {}) if isinstance(model_config.get('attributes', {}), dict) else {}
    return {
        'fragility': int(attributes_config.get('fragility_classes', 3)),
        'geometry': int(attributes_config.get('geometry_classes', 4)),
        'surface': int(attributes_config.get('surface_classes', 2)),
    }


def new_attribute_confusions(class_counts: dict[str, int]) -> dict[str, torch.Tensor]:
    return {
        task: torch.zeros(int(class_counts[task]), int(class_counts[task]), dtype=torch.long)
        for task in ATTRIBUTE_TASKS
    }


def accuracy_from_confusion(confusion: torch.Tensor) -> float:
    total = confusion.sum().item()
    if total == 0:
        return 0.0
    return float(torch.diag(confusion).sum().item() / total)


def class_f1_from_confusion(confusion: torch.Tensor, index: int) -> float:
    tp = confusion[index, index].item()
    fp = confusion[:, index].sum().item() - tp
    fn = confusion[index, :].sum().item() - tp
    denom = 2 * tp + fp + fn
    return 0.0 if denom == 0 else float((2 * tp) / denom)


def macro_f1_from_confusion(confusion: torch.Tensor) -> float:
    scores = [class_f1_from_confusion(confusion, index) for index in range(confusion.shape[0])]
    return float(sum(scores) / max(1, len(scores)))


def average_selected(values: dict[str, float], selected_tasks: list[str]) -> float:
    selected = [float(values[task]) for task in selected_tasks if task in values]
    return float(sum(selected) / max(1, len(selected)))


def update_confusion(confusion: torch.Tensor, preds: torch.Tensor, targets: torch.Tensor) -> None:
    preds = preds.detach().reshape(-1).to(dtype=torch.long, device='cpu')
    targets = targets.detach().reshape(-1).to(dtype=torch.long, device='cpu')
    if preds.numel() == 0:
        return
    classes = int(confusion.shape[0])
    valid = (targets >= 0) & (targets < classes) & (preds >= 0) & (preds < classes)
    if not valid.any():
        return
    flat = targets[valid] * classes + preds[valid]
    confusion += torch.bincount(flat, minlength=classes * classes).reshape(classes, classes)


def scores_from_confusions(
    confusions: dict[str, torch.Tensor],
    selected_tasks: list[str],
    *,
    combined_correct: int | None = None,
    combined_count: int | None = None,
) -> dict[str, Any]:
    task_accuracies = {task: accuracy_from_confusion(confusions[task]) for task in ATTRIBUTE_TASKS}
    task_f1_scores = {task: macro_f1_from_confusion(confusions[task]) for task in ATTRIBUTE_TASKS}
    metrics: dict[str, Any] = {
        'count': int(confusions[ATTRIBUTE_TASKS[0]].sum().item()),
        'attr_accuracy_avg': average_selected(task_accuracies, selected_tasks),
        'attr_all_accuracy_avg': float(sum(task_accuracies.values()) / max(1, len(task_accuracies))),
        'attr_macro_f1_avg': average_selected(task_f1_scores, selected_tasks),
        'attr_all_macro_f1_avg': float(sum(task_f1_scores.values()) / max(1, len(task_f1_scores))),
    }
    for task in ATTRIBUTE_TASKS:
        metrics[f'{task}_accuracy'] = task_accuracies[task]
        metrics[f'{task}_macro_f1'] = task_f1_scores[task]
        metrics[f'{task}_confusion'] = confusions[task].tolist()
    if combined_correct is not None and combined_count is not None:
        metrics['combined_attr_accuracy'] = float(combined_correct / max(1, combined_count))
        metrics['combined_attr_count'] = int(combined_count)
    return metrics


def build_attribute_window_masks(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    valid = batch['window_mask'].bool()
    phase = batch['phase_labels']
    stable = batch['stable_masks'].bool() & valid
    stable_phase = batch['stable_phases']
    return {
        'all_windows': valid,
        'water_only': valid & (phase == 0),
        'interface_only': valid & (phase == 1),
        'air_only': valid & (phase == 2),
        'stable_windows': stable,
        'stable_water_only': stable & (stable_phase == 0),
        'stable_air_only': stable & (stable_phase == 2),
        'stable_water_air': stable & ((stable_phase == 0) | (stable_phase == 2)),
    }


def build_attribute_aggregation_masks(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    masks = build_attribute_window_masks(batch)
    return {
        'all_windows': masks['all_windows'],
        'stable_water_air': masks['stable_water_air'],
        'water_only': masks['water_only'],
        'air_only': masks['air_only'],
    }


def update_combined_counts(
    preds_by_task: dict[str, torch.Tensor],
    targets_by_task: dict[str, torch.Tensor],
    mask: torch.Tensor,
) -> tuple[int, int]:
    flat_mask = mask.detach().reshape(-1).bool().cpu()
    if not flat_mask.any():
        return 0, 0
    correct = torch.ones(int(flat_mask.sum().item()), dtype=torch.bool)
    for task in ATTRIBUTE_TASKS:
        preds = preds_by_task[task].detach().reshape(-1).cpu()[flat_mask]
        targets = targets_by_task[task].detach().reshape(-1).cpu()[flat_mask]
        correct &= preds == targets
    return int(correct.sum().item()), int(correct.numel())


class AttributeWindowMetricAccumulator:
    def __init__(self, class_counts: dict[str, int], selected_tasks: list[str]) -> None:
        self.class_counts = class_counts
        self.selected_tasks = selected_tasks
        self.confusions = {
            name: new_attribute_confusions(class_counts)
            for name in ATTRIBUTE_WINDOW_SLICES
        }
        self.combined_correct = {name: 0 for name in ATTRIBUTE_WINDOW_SLICES}
        self.combined_count = {name: 0 for name in ATTRIBUTE_WINDOW_SLICES}

    def update(self, outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> None:
        batch_size, max_windows = batch['window_mask'].shape
        masks = build_attribute_window_masks(batch)
        targets_by_task = {
            task: batch[ATTRIBUTE_LABEL_KEYS[task]][:, None].expand(-1, max_windows).reshape(-1)
            for task in ATTRIBUTE_TASKS
        }
        preds_by_task = {}
        for task in ATTRIBUTE_TASKS:
            logits = outputs[ATTRIBUTE_LOGIT_KEYS[task]].reshape(batch_size * max_windows, -1)
            preds_by_task[task] = logits.argmax(dim=-1)

        for name, mask in masks.items():
            flat_mask = mask.reshape(-1)
            for task in ATTRIBUTE_TASKS:
                update_confusion(
                    self.confusions[name][task],
                    preds_by_task[task][flat_mask],
                    targets_by_task[task][flat_mask],
                )
            correct, count = update_combined_counts(preds_by_task, targets_by_task, flat_mask)
            self.combined_correct[name] += correct
            self.combined_count[name] += count

    def finalize(self) -> dict[str, dict[str, Any]]:
        return {
            name: scores_from_confusions(
                self.confusions[name],
                self.selected_tasks,
                combined_correct=self.combined_correct[name],
                combined_count=self.combined_count[name],
            )
            for name in ATTRIBUTE_WINDOW_SLICES
        }


class AttributeAggregationAccumulator:
    def __init__(self, class_counts: dict[str, int], selected_tasks: list[str]) -> None:
        self.class_counts = class_counts
        self.selected_tasks = selected_tasks
        self.sample_confusions = {
            name: {
                method: new_attribute_confusions(class_counts)
                for method in ATTRIBUTE_AGGREGATION_METHODS
            }
            for name in ATTRIBUTE_AGGREGATION_SLICES
        }
        self.sample_combined_correct = {
            name: {method: 0 for method in ATTRIBUTE_AGGREGATION_METHODS}
            for name in ATTRIBUTE_AGGREGATION_SLICES
        }
        self.sample_combined_count = {
            name: {method: 0 for method in ATTRIBUTE_AGGREGATION_METHODS}
            for name in ATTRIBUTE_AGGREGATION_SLICES
        }
        self.object_sums: dict[str, dict[str, dict[int, dict[str, torch.Tensor]]]] = {
            name: {method: {} for method in ATTRIBUTE_AGGREGATION_METHODS}
            for name in ATTRIBUTE_AGGREGATION_SLICES
        }
        self.object_counts: dict[str, dict[str, dict[int, int]]] = {
            name: {method: {} for method in ATTRIBUTE_AGGREGATION_METHODS}
            for name in ATTRIBUTE_AGGREGATION_SLICES
        }
        self.object_targets: dict[int, dict[str, int]] = {}

        self.sample_model_confusions = new_attribute_confusions(class_counts)
        self.sample_model_combined_correct = 0
        self.sample_model_combined_count = 0
        self.object_model_sums: dict[str, dict[int, dict[str, torch.Tensor]]] = {
            method: {}
            for method in ATTRIBUTE_AGGREGATION_METHODS
        }
        self.object_model_counts: dict[str, dict[int, int]] = {
            method: {}
            for method in ATTRIBUTE_AGGREGATION_METHODS
        }

    def _remember_object_targets(self, object_indices: torch.Tensor, labels_by_task: dict[str, torch.Tensor]) -> None:
        for row in range(int(object_indices.shape[0])):
            object_index = int(object_indices[row].item())
            self.object_targets.setdefault(
                object_index,
                {task: int(labels_by_task[task][row].item()) for task in ATTRIBUTE_TASKS},
            )

    def _accumulate_object_values(
        self,
        sums: dict[int, dict[str, torch.Tensor]],
        counts: dict[int, int],
        object_indices: torch.Tensor,
        valid_rows: torch.Tensor,
        values_by_task: dict[str, torch.Tensor],
    ) -> None:
        valid_indices = valid_rows.nonzero(as_tuple=False).flatten().cpu()
        for row_tensor in valid_indices:
            row = int(row_tensor.item())
            object_index = int(object_indices[row].item())
            counts[object_index] = int(counts.get(object_index, 0)) + 1
            task_sums = sums.setdefault(
                object_index,
                {
                    task: torch.zeros(int(self.class_counts[task]), dtype=torch.float32)
                    for task in ATTRIBUTE_TASKS
                },
            )
            for task in ATTRIBUTE_TASKS:
                task_sums[task] += values_by_task[task][row].detach().cpu().to(dtype=torch.float32)

    def _update_sample_predictions(
        self,
        confusions: dict[str, torch.Tensor],
        labels_by_task: dict[str, torch.Tensor],
        values_by_task: dict[str, torch.Tensor],
        valid_rows: torch.Tensor,
    ) -> tuple[int, int]:
        valid_rows_cpu = valid_rows.detach().reshape(-1).bool().cpu()
        preds_by_task = {}
        targets_by_task = {}
        for task in ATTRIBUTE_TASKS:
            preds = values_by_task[task].argmax(dim=-1).detach().cpu()
            targets = labels_by_task[task].detach().cpu()
            preds_by_task[task] = preds
            targets_by_task[task] = targets
            update_confusion(confusions[task], preds[valid_rows_cpu], targets[valid_rows_cpu])
        return update_combined_counts(preds_by_task, targets_by_task, valid_rows_cpu)

    @staticmethod
    def _mean_over_windows(values: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        safe_mask = mask.detach().bool()
        counts = safe_mask.sum(dim=1, keepdim=True)
        pooled = (values * safe_mask.unsqueeze(-1).to(values.dtype)).sum(dim=1)
        pooled = pooled / counts.clamp_min(1).to(values.dtype)
        return pooled.detach().cpu(), (counts.squeeze(1) > 0).detach().cpu()

    def update(self, outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> None:
        batch_size, max_windows = batch['window_mask'].shape
        labels_by_task = {
            task: batch[ATTRIBUTE_LABEL_KEYS[task]].detach().cpu()
            for task in ATTRIBUTE_TASKS
        }
        object_indices = batch['object_index'].detach().cpu()
        self._remember_object_targets(object_indices, labels_by_task)

        masks = build_attribute_aggregation_masks(batch)
        window_logits_by_task = {
            task: outputs[ATTRIBUTE_LOGIT_KEYS[task]].reshape(batch_size, max_windows, -1).detach()
            for task in ATTRIBUTE_TASKS
        }
        window_probs_by_task = {
            task: torch.softmax(window_logits_by_task[task], dim=-1)
            for task in ATTRIBUTE_TASKS
        }
        for name, mask in masks.items():
            for method in ATTRIBUTE_AGGREGATION_METHODS:
                source = window_logits_by_task if method == 'mean_logits' else window_probs_by_task
                values_by_task: dict[str, torch.Tensor] = {}
                valid_rows = None
                for task in ATTRIBUTE_TASKS:
                    pooled, rows = self._mean_over_windows(source[task], mask)
                    values_by_task[task] = pooled
                    valid_rows = rows if valid_rows is None else (valid_rows & rows)
                assert valid_rows is not None
                correct, count = self._update_sample_predictions(
                    self.sample_confusions[name][method],
                    labels_by_task,
                    values_by_task,
                    valid_rows,
                )
                self.sample_combined_correct[name][method] += correct
                self.sample_combined_count[name][method] += count
                self._accumulate_object_values(
                    self.object_sums[name][method],
                    self.object_counts[name][method],
                    object_indices,
                    valid_rows,
                    values_by_task,
                )

        if all(key in outputs for key in ATTRIBUTE_SAMPLE_LOGIT_KEYS.values()):
            sample_logits_by_task = {
                task: outputs[ATTRIBUTE_SAMPLE_LOGIT_KEYS[task]].detach().cpu()
                for task in ATTRIBUTE_TASKS
            }
            sample_probs_by_task = {
                task: torch.softmax(sample_logits_by_task[task], dim=-1)
                for task in ATTRIBUTE_TASKS
            }
            valid_rows = torch.ones(batch_size, dtype=torch.bool)
            correct, count = self._update_sample_predictions(
                self.sample_model_confusions,
                labels_by_task,
                sample_logits_by_task,
                valid_rows,
            )
            self.sample_model_combined_correct += correct
            self.sample_model_combined_count += count
            for method in ATTRIBUTE_AGGREGATION_METHODS:
                source = sample_logits_by_task if method == 'mean_logits' else sample_probs_by_task
                self._accumulate_object_values(
                    self.object_model_sums[method],
                    self.object_model_counts[method],
                    object_indices,
                    valid_rows,
                    source,
                )

    def _finalize_object_metrics(
        self,
        sums: dict[int, dict[str, torch.Tensor]],
        counts: dict[int, int],
    ) -> dict[str, Any]:
        confusions = new_attribute_confusions(self.class_counts)
        combined_correct = 0
        combined_count = 0
        for object_index, task_sums in sums.items():
            denom = max(1, int(counts.get(object_index, 0)))
            all_correct = True
            for task in ATTRIBUTE_TASKS:
                prediction = int((task_sums[task] / denom).argmax().item())
                target = int(self.object_targets.get(object_index, {}).get(task, -1))
                update_confusion(
                    confusions[task],
                    torch.tensor([prediction], dtype=torch.long),
                    torch.tensor([target], dtype=torch.long),
                )
                all_correct = all_correct and prediction == target
            combined_correct += int(all_correct)
            combined_count += 1
        metrics = scores_from_confusions(
            confusions,
            self.selected_tasks,
            combined_correct=combined_correct,
            combined_count=combined_count,
        )
        metrics['object_count'] = int(combined_count)
        return metrics

    def finalize(self) -> dict[str, Any]:
        sample_aggregation = {}
        object_aggregation = {}
        for name in ATTRIBUTE_AGGREGATION_SLICES:
            sample_aggregation[name] = {}
            object_aggregation[name] = {}
            for method in ATTRIBUTE_AGGREGATION_METHODS:
                sample_aggregation[name][method] = scores_from_confusions(
                    self.sample_confusions[name][method],
                    self.selected_tasks,
                    combined_correct=self.sample_combined_correct[name][method],
                    combined_count=self.sample_combined_count[name][method],
                )
                object_aggregation[name][method] = self._finalize_object_metrics(
                    self.object_sums[name][method],
                    self.object_counts[name][method],
                )

        sample_model = scores_from_confusions(
            self.sample_model_confusions,
            self.selected_tasks,
            combined_correct=self.sample_model_combined_correct,
            combined_count=self.sample_model_combined_count,
        )
        object_model = {
            method: self._finalize_object_metrics(self.object_model_sums[method], self.object_model_counts[method])
            for method in ATTRIBUTE_AGGREGATION_METHODS
        }
        return {
            'sample_aggregation': sample_aggregation,
            'object_aggregation': object_aggregation,
            'sample_model_pool': sample_model,
            'object_model_pool': object_model,
        }


def flatten_window_metrics(window_metrics: dict[str, dict[str, Any]]) -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for name, metrics in window_metrics.items():
        prefix = f"attr_window_{name}"
        for key in (
            'count',
            'attr_accuracy_avg',
            'attr_macro_f1_avg',
            'attr_all_macro_f1_avg',
            'combined_attr_accuracy',
        ):
            if key in metrics:
                flat[f'{prefix}_{key}'] = metrics[key]
    return flat


def flatten_aggregation_metrics(aggregation_metrics: dict[str, Any]) -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for level in ('sample_aggregation', 'object_aggregation'):
        label = 'sample' if level == 'sample_aggregation' else 'object'
        for slice_name, by_method in aggregation_metrics[level].items():
            for method, metrics in by_method.items():
                prefix = f'{label}_attr_{slice_name}_{method}'
                for key in (
                    'count',
                    'object_count',
                    'attr_accuracy_avg',
                    'attr_macro_f1_avg',
                    'attr_all_macro_f1_avg',
                    'combined_attr_accuracy',
                ):
                    if key in metrics:
                        flat[f'{prefix}_{key}'] = metrics[key]

    sample_model = aggregation_metrics.get('sample_model_pool', {})
    for key in (
        'count',
        'attr_accuracy_avg',
        'attr_macro_f1_avg',
        'attr_all_macro_f1_avg',
        'combined_attr_accuracy',
    ):
        if key in sample_model:
            flat[f'sample_attr_model_pool_{key}'] = sample_model[key]

    for method, metrics in aggregation_metrics.get('object_model_pool', {}).items():
        prefix = f'object_attr_model_pool_{method}'
        for key in (
            'count',
            'object_count',
            'attr_accuracy_avg',
            'attr_macro_f1_avg',
            'attr_all_macro_f1_avg',
            'combined_attr_accuracy',
        ):
            if key in metrics:
                flat[f'{prefix}_{key}'] = metrics[key]
    return flat
