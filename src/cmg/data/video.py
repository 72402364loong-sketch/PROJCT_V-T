from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None


@lru_cache(maxsize=512)
def get_video_metadata(path: str | Path) -> dict[str, float]:
    if cv2 is None:
        raise RuntimeError('opencv-python is required for video decoding.')
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise FileNotFoundError(f'Unable to open video: {path}')
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    capture.release()
    fps = fps if fps > 0 else 30.0
    duration = frame_count / fps if fps > 0 else 0.0
    return {'frame_count': frame_count, 'fps': fps, 'duration': duration}


def build_window_frame_indices(
    frame_count: int,
    fps: float,
    start_time: float,
    end_time: float,
) -> list[int]:
    return [
        frame_index
        for frame_index in range(frame_count)
        if start_time <= frame_index / fps < end_time
    ]


def build_frame_time_array(indices: list[int], fps: float) -> np.ndarray:
    if not indices:
        return np.zeros(0, dtype=np.float32)
    return (np.asarray(indices, dtype=np.float32) / float(fps)).astype(np.float32)


def sample_frame_indices_with_mapping(indices: list[int], num_frames: int) -> tuple[list[int], list[bool], list[float]]:
    if num_frames <= 0:
        raise ValueError(f'num_frames must be positive, got {num_frames}')
    if not indices:
        return [], [], []
    if len(indices) <= num_frames:
        padded = indices + [indices[-1]] * (num_frames - len(indices))
        mask = [True] * len(indices) + [False] * (num_frames - len(indices))
        positions = list(np.linspace(0.0, max(0, len(indices) - 1), num=num_frames, dtype=np.float32)) if len(indices) > 1 else [0.0] * num_frames
        return padded, mask, positions
    positions = np.linspace(0, len(indices) - 1, num=num_frames, dtype=np.float32)
    sampled = [indices[int(round(float(position)))] for position in positions]
    return sampled, [True] * num_frames, [float(position) for position in positions]


def sample_frame_indices(indices: list[int], num_frames: int) -> tuple[list[int], list[bool]]:
    sampled, mask, _ = sample_frame_indices_with_mapping(indices, num_frames)
    return sampled, mask


def preprocess_frame(
    frame: np.ndarray,
    *,
    image_size: int,
    roi: dict[str, int] | None,
) -> np.ndarray:
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    if roi:
        x, y = roi['x'], roi['y']
        w, h = roi['width'], roi['height']
        frame = frame[y : y + h, x : x + w]
    frame = cv2.resize(frame, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
    return frame.astype(np.float32) / 255.0


def load_frame_at_index(
    video_path: str | Path,
    frame_index: int,
    image_size: int,
    roi: dict[str, int] | None,
) -> np.ndarray:
    if cv2 is None:
        raise RuntimeError('opencv-python is required for video decoding.')
    capture = cv2.VideoCapture(str(video_path))
    capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
    success, frame = capture.read()
    capture.release()
    if not success:
        raise RuntimeError(f'Unable to decode frame {frame_index} from {video_path}.')
    return preprocess_frame(frame, image_size=image_size, roi=roi)


def load_frames_by_indices(
    video_path: str | Path,
    frame_indices: list[int],
    image_size: int,
    roi: dict[str, int] | None,
) -> dict[int, np.ndarray]:
    if cv2 is None:
        raise RuntimeError('opencv-python is required for video decoding.')
    if not frame_indices:
        raise RuntimeError(f'No frame indices requested for {video_path}.')

    unique_indices = sorted({int(frame_index) for frame_index in frame_indices})
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f'Unable to open video: {video_path}')

    frames: dict[int, np.ndarray] = {}
    try:
        for frame_index in unique_indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            success, frame = capture.read()
            if not success:
                capture.release()
                capture = cv2.VideoCapture(str(video_path))
                if not capture.isOpened():
                    raise FileNotFoundError(f'Unable to reopen video: {video_path}')
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                success, frame = capture.read()
            if not success:
                raise RuntimeError(f'Unable to decode frame {frame_index} from {video_path}.')
            frames[frame_index] = preprocess_frame(frame, image_size=image_size, roi=roi)
    finally:
        capture.release()
    return frames


def load_window_frames(
    video_path: str | Path,
    frame_indices: list[int],
    image_size: int,
    roi: dict[str, int] | None,
) -> np.ndarray:
    frame_map = load_frames_by_indices(
        video_path=video_path,
        frame_indices=frame_indices,
        image_size=image_size,
        roi=roi,
    )
    frames = [frame_map[int(frame_index)] for frame_index in frame_indices]
    if not frames:
        raise RuntimeError(f'No frames decoded from {video_path} at {json.dumps(frame_indices)}')
    return np.stack(frames, axis=0)
