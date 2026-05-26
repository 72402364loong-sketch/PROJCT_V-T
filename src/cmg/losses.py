from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from cmg.attribute_metrics import (
    ATTRIBUTE_LABEL_KEYS,
    ATTRIBUTE_LOGIT_KEYS,
    ATTRIBUTE_SAMPLE_LOGIT_KEYS,
    ATTRIBUTE_TASKS,
    build_attribute_window_masks,
)


def info_nce_loss(view_a: torch.Tensor, view_b: torch.Tensor, temperature: float) -> torch.Tensor:
    if view_a.shape[0] <= 1:
        return view_a.new_tensor(0.0)
    logits = view_a @ view_b.t() / temperature
    targets = torch.arange(view_a.shape[0], device=view_a.device)
    loss_a = F.cross_entropy(logits, targets)
    loss_b = F.cross_entropy(logits.t(), targets)
    return 0.5 * (loss_a + loss_b)



def supcon_cross_medium_loss(
    features: torch.Tensor,
    object_ids: torch.Tensor,
    sample_ids: torch.Tensor,
    phase_ids: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    if features.shape[0] <= 1:
        return features.new_tensor(0.0)
    normalized = F.normalize(features, dim=-1)
    logits = normalized @ normalized.t() / temperature
    identity = torch.eye(features.shape[0], device=features.device, dtype=torch.bool)
    logits = logits.masked_fill(identity, float('-inf'))
    positive_mask = (object_ids[:, None] == object_ids[None, :]) & (sample_ids[:, None] != sample_ids[None, :])
    positive_mask &= (phase_ids[:, None] != phase_ids[None, :])
    positive_mask &= ~identity
    log_denominator = torch.logsumexp(logits, dim=-1)
    losses = []
    for index in range(features.shape[0]):
        positives = positive_mask[index]
        if positives.any():
            log_prob = logits[index, positives] - log_denominator[index]
            losses.append(-log_prob.mean())
    if not losses:
        return features.new_tensor(0.0)
    return torch.stack(losses).mean()



def masked_smooth_l1(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    beta: float,
    zero: torch.Tensor,
) -> torch.Tensor:
    if not mask.any():
        return zero
    return F.smooth_l1_loss(prediction[mask], target[mask], beta=beta)


def masked_scaled_smooth_l1(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    beta: float,
    scale: float,
    zero: torch.Tensor,
) -> torch.Tensor:
    if not mask.any():
        return zero
    if scale <= 0.0:
        raise ValueError(f'delta normalization scale must be positive, got {scale}.')
    if scale == 1.0:
        return F.smooth_l1_loss(prediction[mask], target[mask], beta=beta)
    return F.smooth_l1_loss(prediction[mask] / scale, target[mask] / scale, beta=beta)



def masked_mean(values: torch.Tensor, mask: torch.Tensor, *, zero: torch.Tensor) -> torch.Tensor:
    if not mask.any():
        return zero
    return values[mask].mean()



def compute_policy_loss(
    outputs: dict[str, torch.Tensor],
    *,
    force_targets: torch.Tensor,
    reference_targets: torch.Tensor | None,
    delta_targets: torch.Tensor | None,
    has_expert: torch.Tensor,
    has_reference: torch.Tensor,
    control_mask: torch.Tensor,
    reference_supervision_mask: torch.Tensor,
    delta_supervision_mask: torch.Tensor,
    quiet_supervision_mask: torch.Tensor,
    phase_targets: torch.Tensor,
    stable_mask: torch.Tensor,
    window_mask: torch.Tensor,
    zero: torch.Tensor,
    policy_loss_config: dict[str, Any] | None,
) -> torch.Tensor:
    force_pred = outputs['force_pred'].reshape(-1)
    policy_config = policy_loss_config or {}
    policy_type = str(policy_config.get('type', 'mse'))

    if policy_type == 'explicit_reference_delta_weighted_smooth_l1':
        all_weight = float(policy_config.get('all_weight', 0.25))
        interface_weight = float(policy_config.get('interface_weight', 4.0))
        stable_weight = float(policy_config.get('stable_weight', 0.25))
        base_reference_weight = float(policy_config.get('base_reference_weight', 1.0))
        base_stable_weight = float(policy_config.get('base_stable_weight', 1.0))
        base_normalization_scale = float(policy_config.get('base_normalization_scale', 1.0))
        residual_interface_weight = float(policy_config.get('residual_interface_weight', 6.0))
        residual_stable_zero_weight = float(policy_config.get('residual_stable_zero_weight', 1.0))
        residual_non_interface_weight = float(policy_config.get('residual_non_interface_weight', 0.25))
        gated_residual_interface_weight = float(policy_config.get('gated_residual_interface_weight', 0.0))
        gated_residual_min_gate = float(policy_config.get('gated_residual_min_gate', 0.0))
        quiet_weight = float(policy_config.get('quiet_weight', 0.0))
        interface_positive_bias_weight = float(policy_config.get('interface_positive_bias_weight', 0.0))
        interface_bias_margin = float(policy_config.get('interface_bias_margin', 0.0))
        beta = float(policy_config.get('beta', 1.0))
        delta_normalization_scale = float(policy_config.get('delta_normalization_scale', 1.0))

        if reference_targets is None or delta_targets is None:
            raise RuntimeError('Explicit reference-delta policy loss requires reference_targets and delta_targets.')
        if 'force_base' not in outputs or 'force_interface_delta' not in outputs:
            raise RuntimeError('Explicit reference-delta policy loss requires force_base and force_interface_delta outputs.')

        base_force = outputs['force_base'].reshape(-1)
        interface_delta = outputs['force_interface_delta'].reshape(-1)
        interface_gate = outputs['interface_gate'].reshape(-1) if 'interface_gate' in outputs else None
        interface_expert = delta_supervision_mask
        stable_expert = control_mask & stable_mask
        reference_mask = reference_supervision_mask
        stable_reference = reference_mask & stable_mask
        quiet_mask = quiet_supervision_mask
        zero_target = torch.zeros_like(interface_delta)

        total = zero
        if all_weight > 0.0:
            total = total + all_weight * masked_smooth_l1(force_pred, force_targets, control_mask, beta=beta, zero=zero)
        if interface_weight > 0.0:
            total = total + interface_weight * masked_smooth_l1(force_pred, force_targets, interface_expert, beta=beta, zero=zero)
        if stable_weight > 0.0:
            total = total + stable_weight * masked_smooth_l1(force_pred, force_targets, stable_expert, beta=beta, zero=zero)
        if base_reference_weight > 0.0:
            total = total + base_reference_weight * masked_scaled_smooth_l1(
                base_force,
                reference_targets,
                reference_mask,
                beta=beta,
                scale=base_normalization_scale,
                zero=zero,
            )
        if base_stable_weight > 0.0:
            total = total + base_stable_weight * masked_scaled_smooth_l1(
                base_force,
                reference_targets,
                stable_reference,
                beta=beta,
                scale=base_normalization_scale,
                zero=zero,
            )
        if residual_interface_weight > 0.0:
            total = total + residual_interface_weight * masked_scaled_smooth_l1(
                interface_delta,
                delta_targets,
                interface_expert,
                beta=beta,
                scale=delta_normalization_scale,
                zero=zero,
            )
        if residual_stable_zero_weight > 0.0:
            total = total + residual_stable_zero_weight * masked_scaled_smooth_l1(
                interface_delta,
                zero_target,
                stable_reference,
                beta=beta,
                scale=delta_normalization_scale,
                zero=zero,
            )
        if residual_non_interface_weight > 0.0:
            total = total + residual_non_interface_weight * masked_scaled_smooth_l1(
                interface_delta,
                zero_target,
                quiet_mask,
                beta=beta,
                scale=delta_normalization_scale,
                zero=zero,
            )
        if gated_residual_interface_weight > 0.0 and interface_gate is not None:
            gated_interface_mask = interface_expert & (interface_gate >= gated_residual_min_gate)
            total = total + gated_residual_interface_weight * masked_scaled_smooth_l1(
                interface_gate * interface_delta,
                delta_targets,
                gated_interface_mask,
                beta=beta,
                scale=delta_normalization_scale,
                zero=zero,
            )
        if quiet_weight > 0.0 and quiet_mask.any():
            quiet_delta = interface_delta[quiet_mask] / delta_normalization_scale
            total = total + quiet_weight * torch.mean(quiet_delta ** 2)
        if interface_positive_bias_weight > 0.0 and interface_expert.any():
            interface_error = force_pred - force_targets
            positive_bias = torch.clamp(masked_mean(interface_error, interface_expert, zero=zero) - interface_bias_margin, min=0.0)
            total = total + interface_positive_bias_weight * (positive_bias ** 2)
        return total

    if policy_type == 'decomposed_interface_residual_weighted_smooth_l1':
        all_weight = float(policy_config.get('all_weight', 0.25))
        interface_weight = float(policy_config.get('interface_weight', 4.0))
        stable_weight = float(policy_config.get('stable_weight', 0.25))
        base_weight = float(policy_config.get('base_weight', 0.5))
        base_stable_weight = float(policy_config.get('base_stable_weight', 1.0))
        base_normalization_scale = float(policy_config.get('base_normalization_scale', 1.0))
        residual_interface_weight = float(policy_config.get('residual_interface_weight', 6.0))
        residual_stable_zero_weight = float(policy_config.get('residual_stable_zero_weight', 1.0))
        residual_non_interface_weight = float(policy_config.get('residual_non_interface_weight', 0.25))
        gated_residual_interface_weight = float(policy_config.get('gated_residual_interface_weight', 0.0))
        gated_residual_min_gate = float(policy_config.get('gated_residual_min_gate', 0.0))
        quiet_weight = float(policy_config.get('quiet_weight', 0.0))
        interface_positive_bias_weight = float(policy_config.get('interface_positive_bias_weight', 0.0))
        interface_bias_margin = float(policy_config.get('interface_bias_margin', 0.0))
        beta = float(policy_config.get('beta', 1.0))
        detach_base_target = bool(policy_config.get('detach_base_target', True))
        delta_normalization_scale = float(policy_config.get('delta_normalization_scale', 1.0))

        if 'force_base' not in outputs or 'force_interface_delta' not in outputs:
            raise RuntimeError('Decomposed residual policy loss requires force_base and force_interface_delta outputs.')

        base_force = outputs['force_base'].reshape(-1)
        interface_delta = outputs['force_interface_delta'].reshape(-1)
        interface_gate = outputs['interface_gate'].reshape(-1) if 'interface_gate' in outputs else None
        interface_expert = control_mask & (phase_targets == 1)
        stable_expert = control_mask & stable_mask
        quiet_mask = quiet_supervision_mask

        total = zero
        if all_weight > 0.0:
            total = total + all_weight * masked_smooth_l1(force_pred, force_targets, control_mask, beta=beta, zero=zero)
        if interface_weight > 0.0:
            total = total + interface_weight * masked_smooth_l1(force_pred, force_targets, interface_expert, beta=beta, zero=zero)
        if stable_weight > 0.0:
            total = total + stable_weight * masked_smooth_l1(force_pred, force_targets, stable_expert, beta=beta, zero=zero)
        if base_weight > 0.0:
            total = total + base_weight * masked_scaled_smooth_l1(
                base_force,
                force_targets,
                control_mask,
                beta=beta,
                scale=base_normalization_scale,
                zero=zero,
            )
        if base_stable_weight > 0.0:
            total = total + base_stable_weight * masked_scaled_smooth_l1(
                base_force,
                force_targets,
                stable_expert,
                beta=beta,
                scale=base_normalization_scale,
                zero=zero,
            )

        base_reference = base_force.detach() if detach_base_target else base_force
        delta_target = force_targets - base_reference
        zero_target = torch.zeros_like(interface_delta)
        if residual_interface_weight > 0.0:
            total = total + residual_interface_weight * masked_scaled_smooth_l1(
                interface_delta,
                delta_target,
                interface_expert,
                beta=beta,
                scale=delta_normalization_scale,
                zero=zero,
            )
        if residual_stable_zero_weight > 0.0:
            total = total + residual_stable_zero_weight * masked_scaled_smooth_l1(
                interface_delta,
                zero_target,
                stable_expert,
                beta=beta,
                scale=delta_normalization_scale,
                zero=zero,
            )
        if residual_non_interface_weight > 0.0:
            total = total + residual_non_interface_weight * masked_scaled_smooth_l1(
                interface_delta,
                zero_target,
                quiet_mask,
                beta=beta,
                scale=delta_normalization_scale,
                zero=zero,
            )
        if gated_residual_interface_weight > 0.0 and interface_gate is not None:
            gated_interface_mask = interface_expert & (interface_gate >= gated_residual_min_gate)
            total = total + gated_residual_interface_weight * masked_scaled_smooth_l1(
                interface_gate * interface_delta,
                delta_target,
                gated_interface_mask,
                beta=beta,
                scale=delta_normalization_scale,
                zero=zero,
            )
        if quiet_weight > 0.0 and quiet_mask.any():
            quiet_delta = interface_delta[quiet_mask] / delta_normalization_scale
            total = total + quiet_weight * torch.mean(quiet_delta ** 2)
        if interface_positive_bias_weight > 0.0 and interface_expert.any():
            interface_error = force_pred - force_targets
            positive_bias = torch.clamp(masked_mean(interface_error, interface_expert, zero=zero) - interface_bias_margin, min=0.0)
            total = total + interface_positive_bias_weight * (positive_bias ** 2)
        return total

    if policy_type in {'interface_weighted_smooth_l1', 'interface_residual_weighted_smooth_l1'}:
        all_weight = float(policy_config.get('all_weight', 1.0))
        interface_weight = float(policy_config.get('interface_weight', 4.0))
        stable_weight = float(policy_config.get('stable_weight', 0.5))
        quiet_weight = float(policy_config.get('quiet_weight', 0.0))
        interface_positive_bias_weight = float(policy_config.get('interface_positive_bias_weight', 0.0))
        interface_bias_margin = float(policy_config.get('interface_bias_margin', 0.0))
        beta = float(policy_config.get('beta', 1.0))

        total = zero
        if all_weight > 0.0:
            total = total + masked_smooth_l1(force_pred, force_targets, control_mask, beta=beta, zero=zero)

        interface_expert = control_mask & (phase_targets == 1)
        if interface_weight > 0.0:
            total = total + interface_weight * masked_smooth_l1(force_pred, force_targets, interface_expert, beta=beta, zero=zero)

        stable_expert = control_mask & stable_mask
        if stable_weight > 0.0:
            total = total + stable_weight * masked_smooth_l1(force_pred, force_targets, stable_expert, beta=beta, zero=zero)

        if quiet_weight > 0.0 and 'force_interface_delta' in outputs and quiet_supervision_mask.any():
            interface_delta = outputs['force_interface_delta'].reshape(-1)[quiet_supervision_mask]
            total = total + quiet_weight * torch.mean(interface_delta ** 2)
        if interface_positive_bias_weight > 0.0 and interface_expert.any():
            interface_error = force_pred - force_targets
            positive_bias = torch.clamp(masked_mean(interface_error, interface_expert, zero=zero) - interface_bias_margin, min=0.0)
            total = total + interface_positive_bias_weight * (positive_bias ** 2)
        return total

    return F.mse_loss(force_pred[control_mask], force_targets[control_mask])



def compute_losses(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    *,
    loss_weights: dict[str, float],
    attribute_loss_config: dict[str, Any] | None = None,
    policy_loss_config: dict[str, Any] | None,
    temperature_clip: float,
    temperature_inv: float,
    phase_class_weights: torch.Tensor,
) -> dict[str, torch.Tensor]:
    window_mask = batch['window_mask'].reshape(-1)
    stable_mask = batch['stable_masks'].reshape(-1) & window_mask
    phase_targets = batch['phase_labels'].reshape(-1)
    medium_logits = outputs['medium_logits'].reshape(-1, outputs['medium_logits'].shape[-1])
    attr_logits = {task: outputs[ATTRIBUTE_LOGIT_KEYS[task]] for task in ATTRIBUTE_TASKS}
    sample_attr_logits = {
        task: outputs[ATTRIBUTE_SAMPLE_LOGIT_KEYS[task]]
        for task in ATTRIBUTE_TASKS
        if ATTRIBUTE_SAMPLE_LOGIT_KEYS[task] in outputs
    }

    repeated_attr_targets = {
        task: batch[ATTRIBUTE_LABEL_KEYS[task]][:, None].expand(-1, batch['window_mask'].shape[1]).reshape(-1)
        for task in ATTRIBUTE_TASKS
    }
    repeated_object = batch['object_index'][:, None].expand(-1, batch['window_mask'].shape[1]).reshape(-1)
    repeated_sample = batch['sample_index'][:, None].expand(-1, batch['window_mask'].shape[1]).reshape(-1)
    stable_phases = batch['stable_phases'].reshape(-1)

    losses: dict[str, torch.Tensor] = {}
    zero = medium_logits.new_tensor(0.0)
    med_weight = float(loss_weights.get('med', 0.0))
    attr_weight = float(loss_weights.get('attr', 0.0))
    clip_weight = float(loss_weights.get('clip', 0.0))
    inv_weight = float(loss_weights.get('inv', 0.0))
    pol_weight = float(loss_weights.get('pol', 0.0))
    attribute_loss_config = attribute_loss_config if isinstance(attribute_loss_config, dict) else {}
    attr_class_weights = attribute_loss_config.get('class_weights', {})
    attr_task_weights = attribute_loss_config.get('task_weights', {})
    attr_mode = str(
        attribute_loss_config.get(
            'mode',
            'sample_primary' if len(sample_attr_logits) == len(ATTRIBUTE_TASKS) else 'window',
        )
    ).strip().lower()
    if attr_mode in {'legacy', 'legacy_window', 'window', 'window_all'}:
        default_sample_weight = 0.0
        default_window_weight = 1.0
        default_window_selection = 'all_windows'
    else:
        default_sample_weight = 1.0
        default_window_weight = 0.2
        default_window_selection = 'stable_water_air'
    attr_sample_weight = float(attribute_loss_config.get('sample_weight', default_sample_weight))
    attr_window_weight = float(attribute_loss_config.get('window_weight', default_window_weight))
    attr_window_selection = str(attribute_loss_config.get('window_selection', default_window_selection)).strip().lower()

    def _optional_class_weight(task_name: str, logits: torch.Tensor) -> torch.Tensor | None:
        raw_weights = attr_class_weights.get(task_name) if isinstance(attr_class_weights, dict) else None
        if raw_weights is None:
            return None
        weights = torch.tensor(raw_weights, dtype=logits.dtype, device=logits.device)
        if weights.numel() != logits.shape[-1]:
            raise RuntimeError(
                f'attribute_loss.class_weights.{task_name} has {weights.numel()} values, '
                f'but logits have {logits.shape[-1]} classes.'
            )
        return weights

    def _task_weight(task_name: str) -> float:
        if not isinstance(attr_task_weights, dict):
            return 1.0
        return float(attr_task_weights.get(task_name, 1.0))

    def _attribute_window_loss_mask() -> torch.Tensor:
        masks = build_attribute_window_masks(batch)
        aliases = {
            'all': 'all_windows',
            'all_valid': 'all_windows',
            'stable': 'stable_windows',
            'stable_water': 'stable_water_only',
            'stable_air': 'stable_air_only',
            'stable_water_and_air': 'stable_water_air',
            'water': 'water_only',
            'air': 'air_only',
            'interface': 'interface_only',
        }
        mask_name = aliases.get(attr_window_selection, attr_window_selection)
        if mask_name not in masks:
            raise RuntimeError(
                f"Unsupported attribute_loss.window_selection={attr_window_selection!r}. "
                f"Expected one of {sorted(set(masks) | set(aliases))}."
            )
        return masks[mask_name].reshape(-1)

    losses['med'] = F.cross_entropy(medium_logits[window_mask], phase_targets[window_mask], weight=phase_class_weights) if med_weight > 0.0 else zero
    if attr_weight > 0.0:
        window_loss_mask = _attribute_window_loss_mask()
        sample_task_losses: dict[str, torch.Tensor] = {}
        window_task_losses: dict[str, torch.Tensor] = {}
        for task in ATTRIBUTE_TASKS:
            logits = attr_logits[task]
            if attr_window_weight > 0.0 and window_loss_mask.any():
                window_task_losses[task] = F.cross_entropy(
                    logits[window_loss_mask],
                    repeated_attr_targets[task][window_loss_mask],
                    weight=_optional_class_weight(task, logits),
                )
            else:
                window_task_losses[task] = zero

            sample_logits = sample_attr_logits.get(task)
            if attr_sample_weight > 0.0 and sample_logits is not None:
                sample_task_losses[task] = F.cross_entropy(
                    sample_logits,
                    batch[ATTRIBUTE_LABEL_KEYS[task]],
                    weight=_optional_class_weight(task, sample_logits),
                )
            else:
                sample_task_losses[task] = zero

        losses['frag'] = attr_sample_weight * sample_task_losses['fragility'] + attr_window_weight * window_task_losses['fragility']
        losses['geom'] = attr_sample_weight * sample_task_losses['geometry'] + attr_window_weight * window_task_losses['geometry']
        losses['surf'] = attr_sample_weight * sample_task_losses['surface'] + attr_window_weight * window_task_losses['surface']
        losses['attr_sample'] = (
            _task_weight('fragility') * sample_task_losses['fragility']
            + _task_weight('geometry') * sample_task_losses['geometry']
            + _task_weight('surface') * sample_task_losses['surface']
        )
        losses['attr_window'] = (
            _task_weight('fragility') * window_task_losses['fragility']
            + _task_weight('geometry') * window_task_losses['geometry']
            + _task_weight('surface') * window_task_losses['surface']
        )
        losses['attr'] = attr_sample_weight * losses['attr_sample'] + attr_window_weight * losses['attr_window']
    else:
        losses['frag'] = zero
        losses['geom'] = zero
        losses['surf'] = zero
        losses['attr_sample'] = zero
        losses['attr_window'] = zero
        losses['attr'] = zero

    if clip_weight > 0.0:
        stable_valid = stable_mask.nonzero(as_tuple=False).flatten()
        losses['clip'] = info_nce_loss(outputs['z_v'][stable_valid], outputs['z_t'][stable_valid], temperature=temperature_clip)
    else:
        losses['clip'] = zero

    if inv_weight > 0.0:
        stable_water_air = stable_mask & (stable_phases >= 0)
        stable_indices = stable_water_air.nonzero(as_tuple=False).flatten()
        losses['inv'] = supcon_cross_medium_loss(
            outputs['z_content'][stable_indices],
            repeated_object[stable_indices],
            repeated_sample[stable_indices],
            stable_phases[stable_indices],
            temperature=temperature_inv,
        )
    else:
        losses['inv'] = zero

    force_targets = batch['control_force_targets'].reshape(-1) if 'control_force_targets' in batch else batch['expert_forces'].reshape(-1)
    reference_targets = batch['reference_forces'].reshape(-1) if 'reference_forces' in batch else None
    delta_targets = batch['delta_force_targets'].reshape(-1) if 'delta_force_targets' in batch else None
    has_expert = batch['has_expert'].reshape(-1) & window_mask
    has_reference = batch['has_reference'].reshape(-1) & window_mask if 'has_reference' in batch else has_expert.new_zeros(has_expert.shape)
    control_mask = batch['has_control_target'].reshape(-1) & window_mask if 'has_control_target' in batch else has_expert
    reference_supervision_mask = batch['reference_supervision_masks'].reshape(-1) & window_mask if 'reference_supervision_masks' in batch else has_reference
    delta_supervision_mask = batch['delta_supervision_masks'].reshape(-1) & window_mask if 'delta_supervision_masks' in batch else (has_reference & (phase_targets == 1))
    quiet_supervision_mask = batch['quiet_supervision_masks'].reshape(-1) & window_mask if 'quiet_supervision_masks' in batch else (control_mask & (phase_targets != 1))
    if pol_weight > 0.0 and control_mask.any():
        losses['pol'] = compute_policy_loss(
            outputs,
            force_targets=force_targets,
            reference_targets=reference_targets,
            delta_targets=delta_targets,
            has_expert=has_expert,
            has_reference=has_reference,
            control_mask=control_mask,
            reference_supervision_mask=reference_supervision_mask,
            delta_supervision_mask=delta_supervision_mask,
            quiet_supervision_mask=quiet_supervision_mask,
            phase_targets=phase_targets,
            stable_mask=stable_mask,
            window_mask=window_mask,
            zero=zero,
            policy_loss_config=policy_loss_config,
        )
    else:
        losses['pol'] = zero

    total = (
        loss_weights.get('clip', 0.0) * losses['clip']
        + loss_weights.get('inv', 0.0) * losses['inv']
        + loss_weights.get('med', 0.0) * losses['med']
        + loss_weights.get('attr', 0.0) * losses['attr']
        + loss_weights.get('pol', 0.0) * losses['pol']
    )
    losses['total'] = total
    return losses
