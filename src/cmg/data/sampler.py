from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Iterable, Iterator, Sequence, TypeVar

from torch.utils.data import Sampler

T = TypeVar('T')


def _safe_weight(weight: float) -> float:
    return max(float(weight), 1e-6)


# Weighted sampling without replacement using randomized priorities.
def _weighted_order(rng: random.Random, items: Sequence[T], weights: Sequence[float]) -> list[T]:
    keys: list[tuple[float, T]] = []
    for item, weight in zip(items, weights):
        u = max(rng.random(), 1e-12)
        key = u ** (1.0 / _safe_weight(weight))
        keys.append((key, item))
    keys.sort(key=lambda pair: pair[0], reverse=True)
    return [item for _, item in keys]


def _weighted_sample_without_replacement(
    rng: random.Random,
    items: Sequence[T],
    weights: Sequence[float],
    count: int,
) -> list[T]:
    if count <= 0 or not items:
        return []
    ordered = _weighted_order(rng, items, weights)
    return ordered[: min(len(ordered), count)]


def _interface_priority(sample: dict, *, alpha: float, min_interface_expert_windows: int) -> float:
    interface_expert_count = int(sample.get('interface_expert_count', sample.get('interface_window_count', 0)))
    interface_window_ratio = float(sample.get('interface_window_ratio', 0.0))
    if interface_expert_count < int(min_interface_expert_windows):
        return 1.0
    return 1.0 + float(alpha) * (float(interface_expert_count) + interface_window_ratio)


class InterfaceAwareSequenceSampler(Sampler[int]):
    def __init__(
        self,
        dataset,
        *,
        alpha: float = 1.0,
        min_interface_expert_windows: int = 1,
        seed: int = 42,
    ) -> None:
        self.dataset = dataset
        self.alpha = float(alpha)
        self.min_interface_expert_windows = int(min_interface_expert_windows)
        self.seed = int(seed)
        self._iteration = 0
        self.indices = list(range(len(dataset)))
        self.weights = [
            _interface_priority(
                dataset.sample_records[index],
                alpha=self.alpha,
                min_interface_expert_windows=self.min_interface_expert_windows,
            )
            for index in self.indices
        ]

    def __len__(self) -> int:
        return len(self.indices)

    def __iter__(self) -> Iterator[int]:
        rng = random.Random(self.seed + self._iteration)
        self._iteration += 1
        ordered = _weighted_order(rng, self.indices, self.weights)
        return iter(ordered)


class ObjectAwareBatchSampler(Sampler[list[int]]):
    def __init__(
        self,
        dataset,
        batch_size: int,
        *,
        samples_per_object: int = 2,
        drop_last: bool = False,
        seed: int = 42,
    ) -> None:
        if batch_size <= 0:
            raise ValueError(f'batch_size must be positive, got {batch_size}')
        if samples_per_object <= 0:
            raise ValueError(f'samples_per_object must be positive, got {samples_per_object}')
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.samples_per_object = int(samples_per_object)
        self.drop_last = bool(drop_last)
        self.seed = int(seed)
        self._iteration = 0

        indices_by_object: dict[str, list[int]] = defaultdict(list)
        for index, sample in enumerate(dataset.sample_records):
            indices_by_object[str(sample['object_id'])].append(index)
        self.indices_by_object = {
            object_id: sorted(indices)
            for object_id, indices in indices_by_object.items()
        }
        self.object_ids = sorted(self.indices_by_object)

    def __len__(self) -> int:
        total = len(self.dataset)
        if self.drop_last:
            return total // self.batch_size
        return math.ceil(total / self.batch_size)

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self.seed + self._iteration)
        self._iteration += 1

        queues = {
            object_id: indices.copy()
            for object_id, indices in self.indices_by_object.items()
        }
        for indices in queues.values():
            rng.shuffle(indices)

        objects_per_batch = max(1, self.batch_size // self.samples_per_object)
        active_objects = [object_id for object_id, indices in queues.items() if indices]

        while active_objects:
            rng.shuffle(active_objects)
            batch: list[int] = []
            primary_objects = active_objects[: min(len(active_objects), objects_per_batch)]

            for object_id in primary_objects:
                take = min(self.samples_per_object, len(queues[object_id]), self.batch_size - len(batch))
                if take <= 0:
                    continue
                batch.extend(queues[object_id][:take])
                del queues[object_id][:take]
                if len(batch) >= self.batch_size:
                    break

            if len(batch) < self.batch_size:
                refill_objects = [object_id for object_id in active_objects if queues[object_id]]
                rng.shuffle(refill_objects)
                for object_id in refill_objects:
                    while queues[object_id] and len(batch) < self.batch_size:
                        batch.append(queues[object_id].pop())
                    if len(batch) >= self.batch_size:
                        break

            active_objects = [object_id for object_id in active_objects if queues[object_id]]
            if len(batch) == self.batch_size:
                yield batch
            elif batch and not self.drop_last:
                yield batch


class InterfaceAwareObjectBatchSampler(Sampler[list[int]]):
    def __init__(
        self,
        dataset,
        batch_size: int,
        *,
        samples_per_object: int = 2,
        alpha: float = 1.0,
        min_interface_expert_windows: int = 1,
        drop_last: bool = False,
        seed: int = 42,
    ) -> None:
        if batch_size <= 0:
            raise ValueError(f'batch_size must be positive, got {batch_size}')
        if samples_per_object <= 0:
            raise ValueError(f'samples_per_object must be positive, got {samples_per_object}')
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.samples_per_object = int(samples_per_object)
        self.alpha = float(alpha)
        self.min_interface_expert_windows = int(min_interface_expert_windows)
        self.drop_last = bool(drop_last)
        self.seed = int(seed)
        self._iteration = 0

        indices_by_object: dict[str, list[int]] = defaultdict(list)
        sample_weights: dict[int, float] = {}
        for index, sample in enumerate(dataset.sample_records):
            indices_by_object[str(sample['object_id'])].append(index)
            sample_weights[index] = _interface_priority(
                sample,
                alpha=self.alpha,
                min_interface_expert_windows=self.min_interface_expert_windows,
            )
        self.indices_by_object = {
            object_id: sorted(indices)
            for object_id, indices in indices_by_object.items()
        }
        self.sample_weights = sample_weights
        self.object_weights = {
            object_id: max(self.sample_weights[index] for index in indices)
            for object_id, indices in self.indices_by_object.items()
        }
        self.object_ids = sorted(self.indices_by_object)

    def __len__(self) -> int:
        total = len(self.dataset)
        if self.drop_last:
            return total // self.batch_size
        return math.ceil(total / self.batch_size)

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self.seed + self._iteration)
        self._iteration += 1

        queues = {
            object_id: _weighted_order(
                rng,
                indices,
                [self.sample_weights[index] for index in indices],
            )
            for object_id, indices in self.indices_by_object.items()
        }
        objects_per_batch = max(1, self.batch_size // self.samples_per_object)
        active_objects = [object_id for object_id, indices in queues.items() if indices]

        while active_objects:
            object_weights = [self.object_weights[object_id] for object_id in active_objects]
            primary_objects = _weighted_sample_without_replacement(
                rng,
                active_objects,
                object_weights,
                min(len(active_objects), objects_per_batch),
            )
            batch: list[int] = []

            for object_id in primary_objects:
                take = min(self.samples_per_object, len(queues[object_id]), self.batch_size - len(batch))
                if take <= 0:
                    continue
                batch.extend(queues[object_id][:take])
                del queues[object_id][:take]
                if len(batch) >= self.batch_size:
                    break

            if len(batch) < self.batch_size:
                refill_objects = [object_id for object_id in active_objects if queues[object_id]]
                refill_weights = [self.object_weights[object_id] for object_id in refill_objects]
                refill_order = _weighted_order(rng, refill_objects, refill_weights)
                for object_id in refill_order:
                    while queues[object_id] and len(batch) < self.batch_size:
                        batch.append(queues[object_id].pop(0))
                    if len(batch) >= self.batch_size:
                        break

            active_objects = [object_id for object_id in active_objects if queues[object_id]]
            if len(batch) == self.batch_size:
                yield batch
            elif batch and not self.drop_last:
                yield batch
