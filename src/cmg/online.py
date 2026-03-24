from __future__ import annotations

import json
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch


@dataclass
class OnlineWindowReference:
    window_start: float
    window_end: float
    video_indices: list[int]
    tactile_indices: list[int]


class SlidingWindowCache:
    def __init__(self, *, window_size_sec: float, stride_sec: float) -> None:
        self.window_size_sec = float(window_size_sec)
        self.stride_sec = float(stride_sec)
        self.video_items: deque[tuple[float, int]] = deque()
        self.tactile_items: deque[tuple[float, int]] = deque()
        self.last_emit_end: float | None = None

    def reset(self) -> None:
        self.video_items.clear()
        self.tactile_items.clear()
        self.last_emit_end = None

    def _prune(self, current_time: float) -> None:
        min_time = current_time - self.window_size_sec
        while self.video_items and self.video_items[0][0] < min_time:
            self.video_items.popleft()
        while self.tactile_items and self.tactile_items[0][0] < min_time:
            self.tactile_items.popleft()

    def append_video_frame(self, *, timestamp: float, frame_index: int) -> None:
        self.video_items.append((float(timestamp), int(frame_index)))
        self._prune(float(timestamp))

    def append_tactile_sample(self, *, timestamp: float, sample_index: int) -> None:
        self.tactile_items.append((float(timestamp), int(sample_index)))
        self._prune(float(timestamp))

    def maybe_emit(self, current_time: float) -> OnlineWindowReference | None:
        current_time = float(current_time)
        if current_time < self.window_size_sec:
            return None
        if self.last_emit_end is not None and current_time < self.last_emit_end + self.stride_sec - 1e-8:
            return None
        window_start = current_time - self.window_size_sec
        video_indices = [index for timestamp, index in self.video_items if window_start <= timestamp < current_time]
        tactile_indices = [index for timestamp, index in self.tactile_items if window_start <= timestamp < current_time]
        self.last_emit_end = current_time
        return OnlineWindowReference(
            window_start=window_start,
            window_end=current_time,
            video_indices=video_indices,
            tactile_indices=tactile_indices,
        )


class OnlineInferenceStub:
    def __init__(self, model: torch.nn.Module, *, device: torch.device) -> None:
        self.model = model.to(device)
        self.device = device
        self.medium_hidden: torch.Tensor | None = None

    def reset(self) -> None:
        self.medium_hidden = None

    def step(
        self,
        *,
        video: torch.Tensor,
        frame_mask: torch.Tensor,
        tactile_high: torch.Tensor,
        tactile_low: torch.Tensor,
        tactile_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        batch = {
            'video': video.to(self.device),
            'frame_mask': frame_mask.to(self.device),
            'tactile_high': tactile_high.to(self.device),
            'tactile_low': tactile_low.to(self.device),
            'tactile_mask': tactile_mask.to(self.device),
        }
        with torch.no_grad():
            outputs = self.model.forward_online_step(batch, medium_hidden=self.medium_hidden)
        self.medium_hidden = outputs['medium_hidden']
        return {key: value.detach().cpu() if torch.is_tensor(value) else value for key, value in outputs.items()}


class OnlineJSONLLogger:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, record: dict[str, Any]) -> None:
        with self.path.open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + '\n')

    def write_window(self, window: OnlineWindowReference, **record: Any) -> None:
        payload = asdict(window)
        payload.update(record)
        self.write(payload)
