from __future__ import annotations

from typing import Any

import torch

from .constants import PHASE_TO_INDEX


def _evenly_sample_indices(indices: torch.Tensor, count: int) -> torch.Tensor:
    if count <= 0 or indices.numel() == 0:
        return indices[:0]
    if indices.numel() <= count:
        return indices
    positions = torch.linspace(0, indices.numel() - 1, steps=count, device=indices.device)
    sampled = indices[positions.round().long()]
    return torch.unique_consecutive(sampled)


def build_window_selection_masks(
    batch: dict[str, torch.Tensor],
    stage_config: dict[str, Any] | None,
) -> dict[str, torch.Tensor]:
    window_mask = batch['window_mask'].bool()
    stable_mask = batch['stable_masks'].bool() & window_mask
    stable_phases = batch['stable_phases']
    phase_labels = batch['phase_labels']
    stage_config = stage_config or {}

    clip_inv_config = stage_config.get('clip_inv_sampling', {})
    require_stable_mask = bool(clip_inv_config.get('require_stable_mask', False))
    eligible_only = bool(clip_inv_config.get('eligible_only', False))

    eligible_wa_mask = window_mask.clone()
    if require_stable_mask:
        eligible_wa_mask &= stable_mask
    else:
        eligible_wa_mask &= stable_mask
    if eligible_only:
        eligible_wa_mask &= stable_phases >= 0
    else:
        eligible_wa_mask &= stable_phases >= 0

    attr_sampling_config = stage_config.get('attr_window_sampling', {})
    if bool(attr_sampling_config.get('enabled', False)):
        attr_mask = torch.zeros_like(window_mask)
        phase_to_budget = {
            PHASE_TO_INDEX['Water']: int(attr_sampling_config.get('water_k', 0)),
            PHASE_TO_INDEX['Air']: int(attr_sampling_config.get('air_k', 0)),
            PHASE_TO_INDEX['Interface']: int(attr_sampling_config.get('interface_k', 0)),
        }
        batch_size = int(window_mask.shape[0])
        for batch_index in range(batch_size):
            valid_row = window_mask[batch_index]
            if not valid_row.any():
                continue
            for phase_index, budget in phase_to_budget.items():
                if budget <= 0:
                    continue
                candidates = ((phase_labels[batch_index] == phase_index) & valid_row).nonzero(as_tuple=False).flatten()
                selected = _evenly_sample_indices(candidates, budget)
                if selected.numel() > 0:
                    attr_mask[batch_index, selected] = True
    else:
        attr_mask = window_mask.clone()

    visual_sampling_enabled = bool(attr_sampling_config.get('enabled', False)) or require_stable_mask or eligible_only
    visual_train_mask = (attr_mask | eligible_wa_mask) & window_mask if visual_sampling_enabled else window_mask.clone()

    return {
        'window_mask': window_mask,
        'eligible_wa_mask': eligible_wa_mask,
        'attr_mask': attr_mask,
        'visual_train_mask': visual_train_mask,
        'visual_sampling_enabled': torch.tensor(int(visual_sampling_enabled), device=window_mask.device, dtype=torch.int64),
    }
