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


class DirectionAwareObjectBatchSampler(Sampler[list[int]]):
    """Pair W2A/A2W samples within each object and cycle only the shorter direction queue."""

    def __init__(
        self,
        dataset,
        batch_size: int,
        *,
        direction_values: Sequence[str] = ('W2A', 'A2W'),
        balance_mode: str = 'paired_cycle',
        interface_alpha: float = 0.0,
        min_interface_expert_windows: int = 1,
        drop_last: bool = False,
        seed: int = 42,
    ) -> None:
        if batch_size <= 0 or batch_size % 2 != 0:
            raise ValueError(f'direction-aware batch_size must be a positive even number, got {batch_size}.')
        normalized_directions = tuple(str(value).strip().upper() for value in direction_values)
        if normalized_directions != ('W2A', 'A2W'):
            raise ValueError(f'direction_values must be (W2A, A2W), got {normalized_directions}.')
        if str(balance_mode).strip().lower() != 'paired_cycle':
            raise ValueError(f'Unsupported direction balance mode: {balance_mode!r}.')

        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.direction_values = normalized_directions
        self.balance_mode = 'paired_cycle'
        self.interface_alpha = float(interface_alpha)
        self.min_interface_expert_windows = int(min_interface_expert_windows)
        self.drop_last = bool(drop_last)
        self.seed = int(seed)
        self._iteration = 0
        self.last_epoch_summary: dict[str, object] = {}

        indices_by_object_direction: dict[str, dict[str, list[int]]] = defaultdict(
            lambda: {direction: [] for direction in self.direction_values}
        )
        self.sample_weights: dict[int, float] = {}
        for index, sample in enumerate(dataset.sample_records):
            object_id = str(sample['object_id'])
            direction = str(sample.get('direction', '')).strip().upper()
            if direction not in self.direction_values:
                raise ValueError(f'Sample {sample.get("sample_id", index)!r} has unsupported direction={direction!r}.')
            indices_by_object_direction[object_id][direction].append(index)
            self.sample_weights[index] = _interface_priority(
                sample,
                alpha=self.interface_alpha,
                min_interface_expert_windows=self.min_interface_expert_windows,
            )

        incomplete = {
            object_id: [direction for direction in self.direction_values if not queues[direction]]
            for object_id, queues in indices_by_object_direction.items()
            if any(not queues[direction] for direction in self.direction_values)
        }
        if incomplete:
            raise ValueError(f'Every sampled object must contain both directions: {incomplete}.')
        self.indices_by_object_direction = {
            object_id: {
                direction: sorted(indices)
                for direction, indices in queues.items()
            }
            for object_id, queues in indices_by_object_direction.items()
        }
        self.object_ids = sorted(self.indices_by_object_direction)
        self.pair_count_by_object = {
            object_id: max(len(queues[direction]) for direction in self.direction_values)
            for object_id, queues in self.indices_by_object_direction.items()
        }
        self.total_pair_count = sum(self.pair_count_by_object.values())

    def __len__(self) -> int:
        pairs_per_batch = self.batch_size // 2
        if self.drop_last:
            return self.total_pair_count // pairs_per_batch
        return math.ceil(self.total_pair_count / pairs_per_batch)

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self.seed + self._iteration)
        self._iteration += 1

        pair_queues: dict[str, list[tuple[int, int]]] = {}
        for object_id, direction_queues in self.indices_by_object_direction.items():
            ordered: dict[str, list[int]] = {}
            for direction in self.direction_values:
                indices = direction_queues[direction]
                ordered[direction] = _weighted_order(
                    rng,
                    indices,
                    [self.sample_weights[index] for index in indices],
                )
            pair_count = self.pair_count_by_object[object_id]
            pair_queues[object_id] = [
                (
                    ordered['W2A'][pair_index % len(ordered['W2A'])],
                    ordered['A2W'][pair_index % len(ordered['A2W'])],
                )
                for pair_index in range(pair_count)
            ]
            rng.shuffle(pair_queues[object_id])

        pairs_per_batch = self.batch_size // 2
        drawn_indices: list[int] = []
        direction_draw_counts = {direction: 0 for direction in self.direction_values}
        object_draw_counts = {object_id: 0 for object_id in self.object_ids}
        batch_count = 0
        active_objects = [object_id for object_id in self.object_ids if pair_queues[object_id]]
        while active_objects:
            rng.shuffle(active_objects)
            selected_objects = active_objects[: min(pairs_per_batch, len(active_objects))]
            batch: list[int] = []
            for object_id in selected_objects:
                w2a_index, a2w_index = pair_queues[object_id].pop()
                batch.extend([w2a_index, a2w_index])
                drawn_indices.extend([w2a_index, a2w_index])
                direction_draw_counts['W2A'] += 1
                direction_draw_counts['A2W'] += 1
                object_draw_counts[object_id] += 2
            while len(batch) < self.batch_size:
                refill_objects = [object_id for object_id in self.object_ids if pair_queues[object_id]]
                if not refill_objects:
                    break
                object_id = rng.choice(refill_objects)
                w2a_index, a2w_index = pair_queues[object_id].pop()
                batch.extend([w2a_index, a2w_index])
                drawn_indices.extend([w2a_index, a2w_index])
                direction_draw_counts['W2A'] += 1
                direction_draw_counts['A2W'] += 1
                object_draw_counts[object_id] += 2
            active_objects = [object_id for object_id in active_objects if pair_queues[object_id]]
            if len(batch) == self.batch_size:
                batch_count += 1
                yield batch
            elif batch and not self.drop_last:
                batch_count += 1
                yield batch

        unique_draws = len(set(drawn_indices))
        repeated_draws = len(drawn_indices) - unique_draws
        self.last_epoch_summary = {
            'balance_mode': self.balance_mode,
            'batch_count': batch_count,
            'total_draws': len(drawn_indices),
            'unique_draws': unique_draws,
            'repeated_draws': repeated_draws,
            'repeat_rate': float(repeated_draws / max(1, len(drawn_indices))),
            'direction_draw_counts': direction_draw_counts,
            'object_draw_counts': object_draw_counts,
        }
