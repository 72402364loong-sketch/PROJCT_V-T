from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class WindowLabel:
    phase_label: str
    is_stable_mask: bool
    stable_phase: str | None


def compute_overlap(start_a: float, end_a: float, start_b: float, end_b: float) -> float:
    return max(0.0, min(end_a, end_b) - max(start_a, start_b))


def compute_phase_label(
    window_start: float,
    window_end: float,
    t_if_enter: float,
    t_if_exit: float,
    overlap_threshold: float,
) -> str:
    overlap = compute_overlap(window_start, window_end, t_if_enter, t_if_exit)
    if overlap >= overlap_threshold:
        return 'Interface'
    center = 0.5 * (window_start + window_end)
    if center < t_if_enter:
        return 'Water'
    if center > t_if_exit:
        return 'Air'
    return 'Interface'


def compute_stable_mask(
    window_start: float,
    window_end: float,
    t_if_enter: float,
    t_if_exit: float,
    stable_margin: float,
) -> tuple[bool, str | None]:
    if window_end <= t_if_enter - stable_margin:
        return True, 'Water'
    if window_start >= t_if_exit + stable_margin:
        return True, 'Air'
    return False, None


def label_window(
    window_start: float,
    window_end: float,
    t_if_enter: float,
    t_if_exit: float,
    overlap_threshold: float,
    stable_margin: float,
) -> WindowLabel:
    phase_label = compute_phase_label(
        window_start=window_start,
        window_end=window_end,
        t_if_enter=t_if_enter,
        t_if_exit=t_if_exit,
        overlap_threshold=overlap_threshold,
    )
    stable, stable_phase = compute_stable_mask(
        window_start=window_start,
        window_end=window_end,
        t_if_enter=t_if_enter,
        t_if_exit=t_if_exit,
        stable_margin=stable_margin,
    )
    return WindowLabel(phase_label=phase_label, is_stable_mask=stable, stable_phase=stable_phase)


def determine_tail_type(
    t_effective_end: float,
    t_if_exit: float,
    full_tail_sec: float,
    short_tail_min_sec: float,
) -> str:
    tail_length = max(0.0, t_effective_end - t_if_exit)
    if tail_length >= full_tail_sec:
        return 'full_tail'
    if tail_length >= short_tail_min_sec:
        return 'short_tail'
    return 'discard_tail'
