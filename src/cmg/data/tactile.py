from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np


@lru_cache(maxsize=512)
def load_tactile_array(path: str | Path) -> np.ndarray:
    rows: list[list[float]] = []
    with Path(path).open('r', encoding='utf-8') as handle:
        for raw_line in handle:
            normalized = raw_line.replace('，', ',')
            parts = [part.strip() for part in normalized.split(',') if part.strip()]
            if not parts:
                continue
            values = [float(part) for part in parts]
            if len(values) < 36:
                values.extend([0.0] * (36 - len(values)))
            elif len(values) > 36:
                values = values[:36]
            rows.append(values)
    array = np.asarray(rows, dtype=np.float32)
    array = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)
    if array.ndim != 2 or array.shape[1] != 36:
        raise ValueError(f'Unexpected tactile shape for {path}: {array.shape}')
    return array



def ema_filter(sequence: np.ndarray, alpha: float) -> np.ndarray:
    if sequence.size == 0:
        return sequence.copy()
    output = np.zeros_like(sequence, dtype=np.float32)
    output[0] = sequence[0]
    for index in range(1, len(sequence)):
        output[index] = alpha * sequence[index] + (1.0 - alpha) * output[index - 1]
    return output



def zero_phase_ema_filter(sequence: np.ndarray, alpha: float) -> np.ndarray:
    if sequence.size == 0:
        return sequence.copy()
    forward = ema_filter(sequence, alpha=alpha)
    backward = ema_filter(forward[::-1].copy(), alpha=alpha)
    return backward[::-1].astype(np.float32)



def build_tactile_time_axis(length: int, dt: float, *, offset_sec: float = 0.0) -> np.ndarray:
    return (float(offset_sec) + np.arange(length, dtype=np.float32) * float(dt)).astype(np.float32)



def split_ac_dc(sequence: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    low = ema_filter(sequence, alpha=alpha)
    high = sequence - low
    return high.astype(np.float32), low.astype(np.float32)



def compute_resample_mapping(source_length: int, target_length: int) -> dict[str, np.ndarray]:
    if target_length <= 0:
        raise ValueError(f'target_length must be positive, got {target_length}')
    target_positions = np.linspace(0.0, 1.0, num=target_length, dtype=np.float32)
    if source_length <= 0:
        return {
            'source_positions': np.zeros(0, dtype=np.float32),
            'target_positions': target_positions,
            'left_indices': np.full(target_length, -1, dtype=np.int32),
            'right_indices': np.full(target_length, -1, dtype=np.int32),
            'left_weights': np.zeros(target_length, dtype=np.float32),
            'right_weights': np.zeros(target_length, dtype=np.float32),
            'valid_mask': np.zeros(target_length, dtype=bool),
        }
    source_positions = np.linspace(0.0, 1.0, num=source_length, dtype=np.float32)
    if source_length == 1:
        return {
            'source_positions': source_positions,
            'target_positions': target_positions,
            'left_indices': np.zeros(target_length, dtype=np.int32),
            'right_indices': np.zeros(target_length, dtype=np.int32),
            'left_weights': np.ones(target_length, dtype=np.float32),
            'right_weights': np.zeros(target_length, dtype=np.float32),
            'valid_mask': np.ones(target_length, dtype=bool),
        }

    right_indices = np.searchsorted(source_positions, target_positions, side='left').astype(np.int32)
    right_indices = np.clip(right_indices, 0, source_length - 1)
    left_indices = np.clip(right_indices - 1, 0, source_length - 1).astype(np.int32)
    left_positions = source_positions[left_indices]
    right_positions = source_positions[right_indices]
    denom = right_positions - left_positions
    safe_same = denom <= 1e-8
    right_weights = np.zeros(target_length, dtype=np.float32)
    right_weights[~safe_same] = (target_positions[~safe_same] - left_positions[~safe_same]) / denom[~safe_same]
    left_weights = 1.0 - right_weights
    left_weights[safe_same] = 1.0
    right_weights[safe_same] = 0.0
    return {
        'source_positions': source_positions,
        'target_positions': target_positions,
        'left_indices': left_indices,
        'right_indices': right_indices,
        'left_weights': left_weights.astype(np.float32),
        'right_weights': right_weights.astype(np.float32),
        'valid_mask': np.ones(target_length, dtype=bool),
    }



def resample_tactile_window(sequence: np.ndarray, target_length: int) -> tuple[np.ndarray, np.ndarray]:
    mapping = compute_resample_mapping(sequence.shape[0], target_length)
    if sequence.size == 0:
        return np.zeros((target_length, 36), dtype=np.float32), mapping['valid_mask']
    if sequence.shape[0] == 1:
        repeated = np.repeat(sequence.astype(np.float32), target_length, axis=0)
        return repeated, mapping['valid_mask']
    left = sequence[mapping['left_indices']]
    right = sequence[mapping['right_indices']]
    left_weight = mapping['left_weights'][:, None]
    right_weight = mapping['right_weights'][:, None]
    resampled = (left * left_weight + right * right_weight).astype(np.float32)
    return resampled, mapping['valid_mask']



def standardize_tactile_window(sequence: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    safe_std = np.clip(std, a_min=1e-6, a_max=None)
    return ((sequence - mean) / safe_std).astype(np.float32)



def _reshape_finger_sensor_axis(sequence: np.ndarray) -> np.ndarray:
    if sequence.ndim != 2 or sequence.shape[1] != 36:
        raise ValueError(f'Expected tactile array with shape [T, 36], got {sequence.shape}')
    return sequence.reshape(sequence.shape[0], 3, 4, 3)



def normalize_normal_sign_table(normal_sign_table: Any) -> np.ndarray:
    if isinstance(normal_sign_table, dict):
        ordered = [normal_sign_table[key] for key in ['finger0', 'finger1', 'finger2']]
        table = np.asarray(ordered, dtype=np.float32)
    else:
        table = np.asarray(normal_sign_table, dtype=np.float32)
    if table.shape != (3, 4):
        raise ValueError(f'normal_sign_table must have shape [3, 4], got {table.shape}')
    if np.any(table == 0):
        raise ValueError('normal_sign_table cannot contain zeros.')
    return np.where(table > 0, 1.0, -1.0).astype(np.float32)



def extract_normal_axis(sequence: np.ndarray) -> np.ndarray:
    return _reshape_finger_sensor_axis(sequence)[..., 2]



def compute_signed_normal_force_curve(
    tactile_array: np.ndarray,
    normal_sign_table: Any,
    *,
    alpha: float | None = None,
) -> np.ndarray:
    z_axis = extract_normal_axis(tactile_array)
    normal_sign = normalize_normal_sign_table(normal_sign_table)
    calibrated = z_axis * normal_sign[None, :, :]
    force_sum = calibrated.sum(axis=(1, 2)).astype(np.float32)
    if alpha is not None:
        force_sum = ema_filter(force_sum, alpha=alpha)
    return force_sum.astype(np.float32)



def compute_measured_force_curve(
    tactile_array: np.ndarray,
    *,
    normal_sign_table: Any,
    alpha: float | None = None,
) -> np.ndarray:
    return compute_signed_normal_force_curve(
        tactile_array,
        normal_sign_table=normal_sign_table,
        alpha=alpha,
    )



def estimate_force_baseline(
    force_curve: np.ndarray,
    time_axis: np.ndarray,
    *,
    contact_time: float | None,
    mode: str,
    window_sec: float,
) -> float:
    baseline_mode = str(mode or 'none').strip().lower()
    if baseline_mode == 'none' or contact_time is None or np.isnan(contact_time):
        return 0.0

    mask = (time_axis >= float(contact_time) - float(window_sec)) & (time_axis < float(contact_time))
    if not mask.any():
        if len(force_curve) == 0:
            return 0.0
        fallback_count = max(1, min(len(force_curve), int(round(float(window_sec) / max(1e-6, float(time_axis[1] - time_axis[0])))) if len(time_axis) > 1 else 8))
        mask = np.zeros(len(force_curve), dtype=bool)
        mask[:fallback_count] = True

    selected = force_curve[mask]
    if selected.size == 0:
        return 0.0
    if baseline_mode == 'pre_contact_median':
        return float(np.median(selected))
    if baseline_mode == 'pre_contact_mean':
        return float(np.mean(selected))
    raise ValueError(f'Unsupported force baseline mode: {mode!r}')



def smooth_force_curve(force_curve: np.ndarray, *, alpha: float, mode: str) -> np.ndarray:
    smoothing_mode = str(mode or 'ema').strip().lower()
    if smoothing_mode == 'none':
        return force_curve.astype(np.float32)
    if smoothing_mode == 'ema':
        return ema_filter(force_curve, alpha=alpha).astype(np.float32)
    if smoothing_mode == 'zero_phase_ema':
        return zero_phase_ema_filter(force_curve, alpha=alpha).astype(np.float32)
    raise ValueError(f'Unsupported force smoothing mode: {mode!r}')



def compute_clean_force_curve(
    tactile_array: np.ndarray,
    *,
    dt: float,
    sync_offset_sec: float,
    contact_time: float | None,
    alpha: float,
    normal_sign_table: Any,
    smoothing_mode: str,
    baseline_mode: str,
    baseline_window_sec: float,
) -> np.ndarray:
    raw_force = compute_measured_force_curve(
        tactile_array,
        normal_sign_table=normal_sign_table,
        alpha=None,
    )
    time_axis = build_tactile_time_axis(raw_force.shape[0], dt=dt, offset_sec=sync_offset_sec)
    baseline = estimate_force_baseline(
        raw_force,
        time_axis,
        contact_time=contact_time,
        mode=baseline_mode,
        window_sec=baseline_window_sec,
    )
    centered = raw_force - baseline
    return smooth_force_curve(centered, alpha=alpha, mode=smoothing_mode)



def compute_expert_force_curve(
    tactile_array: np.ndarray,
    dt: float,
    trial_result: str,
    contact_time: float | None,
    alpha: float,
    normal_sign_table: Any,
) -> np.ndarray | None:
    _ = dt
    _ = contact_time
    if str(trial_result).strip().lower() != 'stable':
        return None
    return compute_measured_force_curve(
        tactile_array,
        normal_sign_table=normal_sign_table,
        alpha=alpha,
    )
