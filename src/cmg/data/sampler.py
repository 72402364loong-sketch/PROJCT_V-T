from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Iterator

from torch.utils.data import Sampler


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
