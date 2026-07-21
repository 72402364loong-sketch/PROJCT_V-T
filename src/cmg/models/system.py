from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn

from .modules import (
    DirectionEmbedding,
    MediumBeliefHead,
    MultiAttributeHead,
    PhaseAwareGateStabilizer,
    PhysicalAttributeHead,
    PolicyHead,
    ResidualFiLMAdapter,
    TactileContentEncoder,
    TactileEvidenceEncoder,
    VisualEncoder,
)


class CrossMediumSystem(nn.Module):
    def __init__(self, model_config: dict) -> None:
        super().__init__()
        visual_config = model_config['visual']
        tactile_config = model_config['tactile']
        attribute_config = model_config['attributes']
        physical_attribute_config = model_config.get('physical_attributes', {})
        if not isinstance(physical_attribute_config, dict):
            physical_attribute_config = {}
        medium_config = model_config['medium']
        policy_config = model_config['policy']
        direction_config = model_config.get('direction_conditioning', {})
        if not isinstance(direction_config, dict):
            raise RuntimeError('model.direction_conditioning must be a mapping.')
        proj_dim = int(visual_config['proj_dim'])
        self.visual_proj_dim = proj_dim
        visual_hidden_dim = int(visual_config.get('hidden_dim', visual_config.get('token_dim', proj_dim)))
        self.stop_gradient = bool(attribute_config.get('stop_gradient', True))
        self.physical_attributes_enabled = bool(physical_attribute_config.get('enabled', False))
        self.policy_base_source = str(policy_config.get('base_source', 'learned')).strip().lower()
        self.policy_gate_source = str(policy_config.get('gate_source', 'medium_prob')).strip().lower()
        self.policy_head_type = str(policy_config.get('head_type', 'legacy')).strip()
        self.policy_per_finger = self.policy_head_type in {
            'state_residual_per_finger',
            'state_residual_per_finger_sign_specific',
        }
        self.policy_finger_count = int(policy_config.get('finger_count', 3))
        self.use_interface_context = bool(policy_config.get('use_interface_context', False))
        self.use_reference_force_context = bool(policy_config.get('use_reference_force_context', False))
        self.reference_force_context_scale = float(policy_config.get('reference_force_context_scale', 100.0))
        self.direction_conditioning_enabled = bool(direction_config.get('enabled', False))
        self.require_explicit_direction = bool(direction_config.get('require_explicit_direction', True))
        self.direction_num_directions = int(direction_config.get('num_directions', 2))
        self.direction_embedding_dim = int(direction_config.get('embedding_dim', 16))
        self.direction_diagnostics_enabled = bool(direction_config.get('return_diagnostics', False))
        medium_direction_config = direction_config.get('medium', {})
        policy_direction_config = direction_config.get('policy', {})
        if not isinstance(medium_direction_config, dict) or not isinstance(policy_direction_config, dict):
            raise RuntimeError('direction_conditioning.medium and .policy must be mappings.')
        self.medium_direction_conditioning_enabled = (
            self.direction_conditioning_enabled and bool(medium_direction_config.get('enabled', True))
        )
        self.policy_direction_conditioning_enabled = (
            self.direction_conditioning_enabled and bool(policy_direction_config.get('enabled', True))
        )
        for branch_name, branch_config, enabled in (
            ('medium', medium_direction_config, self.medium_direction_conditioning_enabled),
            ('policy', policy_direction_config, self.policy_direction_conditioning_enabled),
        ):
            if enabled and str(branch_config.get('mode', 'residual_film')).strip().lower() != 'residual_film':
                raise RuntimeError(
                    f'Unsupported direction_conditioning.{branch_name}.mode={branch_config.get("mode")!r}; '
                    "expected 'residual_film'."
                )
        if self.policy_direction_conditioning_enabled and self.policy_head_type not in {
            'state_residual',
            'state_residual_sign_specific',
            'state_residual_per_finger',
            'state_residual_per_finger_sign_specific',
        }:
            raise RuntimeError(
                'Policy direction conditioning requires a state_residual policy head, '
                f'got {self.policy_head_type!r}.'
            )
        if self.reference_force_context_scale <= 0.0:
            raise RuntimeError(
                f"policy.reference_force_context_scale must be positive, got {self.reference_force_context_scale}."
            )
        self.tactile = SimpleNamespace(
            input_dim=int(tactile_config['input_dim']),
            num_taxels=int(tactile_config.get('num_taxels', 12)),
            axis_dim=int(tactile_config.get('axis_dim', max(1, int(tactile_config['input_dim']) // 12))),
        )
        gate_stabilizer_config = policy_config.get('gate_stabilizer', {}) if isinstance(policy_config.get('gate_stabilizer', {}), dict) else {}
        self.gate_stabilizer_enabled = bool(gate_stabilizer_config.get('enabled', False))
        if self.policy_base_source not in {'learned', 'reference_force'}:
            raise RuntimeError(
                f"Unsupported policy.base_source={self.policy_base_source!r}. Expected 'learned' or 'reference_force'."
            )
        if self.policy_gate_source not in {'medium_prob', 'soft_target', 'hard_phase'}:
            raise RuntimeError(
                f"Unsupported policy.gate_source={self.policy_gate_source!r}. "
                "Expected 'medium_prob', 'soft_target', or 'hard_phase'."
            )
        self.gate_stabilizer = (
            PhaseAwareGateStabilizer(
                on_threshold=float(gate_stabilizer_config.get('on_threshold', 0.25)),
                off_threshold=float(gate_stabilizer_config.get('off_threshold', 0.10)),
                min_on_windows=int(gate_stabilizer_config.get('min_on_windows', 1)),
                min_off_windows=int(gate_stabilizer_config.get('min_off_windows', 1)),
                output_mode=str(gate_stabilizer_config.get('output_mode', 'masked_raw')),
            )
            if self.gate_stabilizer_enabled
            else None
        )

        if self.direction_conditioning_enabled:
            if not self.medium_direction_conditioning_enabled and not self.policy_direction_conditioning_enabled:
                raise RuntimeError('Direction conditioning is enabled but both Medium and Policy adapters are disabled.')
            self.direction_embedding = DirectionEmbedding(
                num_directions=self.direction_num_directions,
                embedding_dim=self.direction_embedding_dim,
            )
        else:
            self.direction_embedding = None
        self.medium_direction_adapter = (
            ResidualFiLMAdapter(
                condition_dim=self.direction_embedding_dim,
                feature_dim=int(tactile_config['evidence_hidden_dim']),
                hidden_dim=int(medium_direction_config.get('hidden_dim', 32)),
                zero_init=bool(medium_direction_config.get('zero_init', direction_config.get('zero_init', True))),
            )
            if self.medium_direction_conditioning_enabled
            else None
        )
        self.policy_direction_adapter = (
            ResidualFiLMAdapter(
                condition_dim=self.direction_embedding_dim,
                feature_dim=int(policy_config['hidden_dim']),
                hidden_dim=int(policy_direction_config.get('hidden_dim', 64)),
                zero_init=bool(policy_direction_config.get('zero_init', direction_config.get('zero_init', True))),
            )
            if self.policy_direction_conditioning_enabled
            else None
        )

        self.visual_encoder = VisualEncoder(visual_config)
        self.content_encoder = TactileContentEncoder(tactile_config, proj_dim=proj_dim)
        self.evidence_encoder = TactileEvidenceEncoder(tactile_config)
        self.medium_head = MediumBeliefHead(
            input_dim=int(tactile_config['evidence_hidden_dim']),
            hidden_dim=int(medium_config['hidden_dim']),
        )
        attribute_input_dim = visual_hidden_dim + int(tactile_config['content_hidden_dim'])
        self.attribute_head = MultiAttributeHead(
            input_dim=attribute_input_dim,
            hidden_dim=int(attribute_config['hidden_dim']),
            object_feature_dim=int(attribute_config['object_feature_dim']),
            fragility_classes=int(attribute_config.get('fragility_classes', 3)),
            geometry_classes=int(attribute_config.get('geometry_classes', 4)),
            surface_classes=int(attribute_config.get('surface_classes', 2)),
        )
        self.physical_attribute_head = (
            PhysicalAttributeHead(
                input_dim=attribute_input_dim,
                hidden_dim=int(physical_attribute_config.get('hidden_dim', attribute_config['hidden_dim'])),
                dropout=float(physical_attribute_config.get('dropout', 0.0)),
            )
            if self.physical_attributes_enabled
            else None
        )
        policy_input_dim = (
            visual_hidden_dim
            + int(tactile_config['content_hidden_dim'])
            + int(attribute_config['object_feature_dim'])
        )
        policy_context_dim = 0
        if self.use_interface_context:
            policy_context_dim += (
                visual_hidden_dim
                + int(tactile_config['content_hidden_dim'])
                + int(tactile_config['evidence_hidden_dim'])
            )
        if self.use_reference_force_context:
            policy_context_dim += 1
        policy_state_dim = int(tactile_config['evidence_hidden_dim']) + int(medium_config['hidden_dim']) + 3
        self.policy_head = PolicyHead(
            input_dim=policy_input_dim,
            hidden_dim=int(policy_config['hidden_dim']),
            film_hidden_dim=int(policy_config['film_hidden_dim']),
            head_type=self.policy_head_type,
            state_input_dim=policy_state_dim,
            context_input_dim=policy_context_dim,
            context_scale_init=float(policy_config.get('context_scale_init', 0.0)),
            residual_output_scale=float(policy_config.get('residual_output_scale', 1.0)),
            finger_count=self.policy_finger_count,
            finger_embedding_dim=int(policy_config.get('finger_embedding_dim', 16)),
        )

    def _direction_embedding_from_batch(
        self,
        batch: dict[str, torch.Tensor],
        *,
        batch_size: int,
        reference: torch.Tensor,
    ) -> torch.Tensor | None:
        if not self.direction_conditioning_enabled:
            return None
        direction_ids = batch.get('direction_ids')
        if direction_ids is None:
            if self.require_explicit_direction:
                raise RuntimeError('Direction-conditioned model requires batch["direction_ids"].')
            direction_ids = torch.zeros(batch_size, dtype=torch.long, device=reference.device)
        else:
            direction_ids = direction_ids.to(device=reference.device)
        if direction_ids.ndim != 1 or direction_ids.shape[0] != batch_size:
            raise RuntimeError(
                f'direction_ids must have shape [{batch_size}], got {tuple(direction_ids.shape)}.'
            )
        if self.direction_embedding is None:
            raise RuntimeError('Direction embedding module is missing while direction conditioning is enabled.')
        return self.direction_embedding(direction_ids).to(dtype=reference.dtype)

    @staticmethod
    def _apply_reference_force_base_override(
        policy_outputs: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        if 'finger_force_base' in policy_outputs and 'finger_reference_forces' in batch:
            learned_base = policy_outputs['finger_force_base']
            reference_force = batch['finger_reference_forces'].reshape_as(learned_base).to(
                device=learned_base.device,
                dtype=learned_base.dtype,
            )
            finite_reference = torch.isfinite(reference_force)
            if not finite_reference.any():
                return policy_outputs

            oracle_base = torch.where(finite_reference, reference_force, learned_base)
            interface_gate = policy_outputs['interface_gate'].reshape(-1, 1).to(
                device=learned_base.device,
                dtype=learned_base.dtype,
            )
            finger_delta = policy_outputs['finger_force_interface_delta']
            finger_force_pred = oracle_base + interface_gate * finger_delta
            updated = dict(policy_outputs)
            updated['finger_force_base_learned'] = learned_base
            updated['finger_force_base'] = oracle_base
            updated['finger_force_pred'] = finger_force_pred
            updated['force_base_learned'] = policy_outputs['force_base']
            updated['force_base'] = oracle_base.mean(dim=-1)
            updated['force_pred'] = finger_force_pred.mean(dim=-1)
            updated['force_interface_delta'] = finger_delta.mean(dim=-1)
            return updated

        if 'reference_forces' not in batch:
            return policy_outputs

        learned_base = policy_outputs['force_base']
        reference_force = batch['reference_forces'].reshape(-1).to(device=learned_base.device, dtype=learned_base.dtype)
        finite_reference = torch.isfinite(reference_force)
        if not finite_reference.any():
            return policy_outputs

        oracle_base = torch.where(finite_reference, reference_force, learned_base)
        updated = dict(policy_outputs)
        updated['force_base_learned'] = learned_base
        updated['force_base'] = oracle_base
        updated['force_pred'] = oracle_base + updated['interface_gate'] * updated['force_interface_delta']
        return updated

    @staticmethod
    def _build_sample_attribute_pool_mask(batch: dict[str, torch.Tensor]) -> torch.Tensor:
        valid = batch['window_mask'].bool()
        stable = batch['stable_masks'].bool() & valid
        stable_phases = batch['stable_phases']
        stable_water_air = stable & ((stable_phases == 0) | (stable_phases == 2))

        has_stable_water_air = stable_water_air.any(dim=1, keepdim=True)
        has_stable = stable.any(dim=1, keepdim=True)
        return torch.where(
            has_stable_water_air,
            stable_water_air,
            torch.where(has_stable, stable, valid),
        )

    @staticmethod
    def _masked_mean_windows(features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        counts = mask.sum(dim=1, keepdim=True)
        pooled = (features * mask.unsqueeze(-1).to(features.dtype)).sum(dim=1)
        pooled = pooled / counts.clamp_min(1).to(features.dtype)
        return pooled.masked_fill(counts == 0, 0.0)

    def _build_task_context(
        self,
        h_v: torch.Tensor,
        z_content: torch.Tensor,
        batch: dict[str, torch.Tensor] | None = None,
    ) -> tuple[
        dict[str, torch.Tensor],
        dict[str, torch.Tensor] | None,
        dict[str, torch.Tensor] | None,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
    ]:
        attribute_inputs = torch.cat([h_v, z_content], dim=-1)
        attribute_outputs = self.attribute_head(attribute_inputs)
        sample_attribute_outputs = None
        sample_physical_attribute_outputs = None
        sample_attribute_pool_mask = None
        if batch is not None:
            batch_size, max_windows = batch['window_mask'].shape
            sample_attribute_pool_mask = self._build_sample_attribute_pool_mask(batch)
            pooled_attribute_inputs = self._masked_mean_windows(
                attribute_inputs.reshape(batch_size, max_windows, -1),
                sample_attribute_pool_mask,
            )
            sample_attribute_outputs = self.attribute_head(pooled_attribute_inputs)
            if self.physical_attribute_head is not None:
                sample_physical_attribute_outputs = self.physical_attribute_head(pooled_attribute_inputs)
        g_obj_context = attribute_outputs['g_obj_context'].detach() if self.stop_gradient else attribute_outputs['g_obj_context']
        g_obj = self.attribute_head.object_projection(g_obj_context)
        task_context = torch.cat([h_v, z_content, g_obj], dim=-1)
        return (
            attribute_outputs,
            sample_attribute_outputs,
            sample_physical_attribute_outputs,
            g_obj,
            task_context,
            sample_attribute_pool_mask,
        )

    @staticmethod
    def _pool_interface_context(
        context_features: torch.Tensor,
        context_mask: torch.Tensor,
    ) -> torch.Tensor:
        if context_features.ndim != 3:
            raise RuntimeError(f'Expected context_features to be [B, W, D], got {tuple(context_features.shape)}.')
        if context_mask.ndim != 2:
            raise RuntimeError(f'Expected context_mask to be [B, W], got {tuple(context_mask.shape)}.')

        safe_mask = context_mask.bool()
        counts = safe_mask.sum(dim=1, keepdim=True)
        pooled = (context_features * safe_mask.unsqueeze(-1).to(context_features.dtype)).sum(dim=1)
        pooled = pooled / counts.clamp_min(1).to(context_features.dtype)
        pooled = pooled.masked_fill(counts == 0, 0.0)
        return pooled

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        batch_size, max_windows = batch['window_mask'].shape
        flat_tactile_high = batch['tactile_high'].reshape(batch_size * max_windows, *batch['tactile_high'].shape[2:])
        flat_tactile_low = batch['tactile_low'].reshape(batch_size * max_windows, *batch['tactile_low'].shape[2:])
        flat_tactile_mask = batch['tactile_mask'].reshape(batch_size * max_windows, batch['tactile_mask'].shape[-1])

        if batch.get('visual_features') is not None:
            h_v = batch['visual_features'].reshape(batch_size * max_windows, batch['visual_features'].shape[-1])
            z_v = h_v.new_zeros((h_v.shape[0], self.visual_proj_dim))
        else:
            flat_video = batch['video'].reshape(batch_size * max_windows, *batch['video'].shape[2:])
            flat_frame_mask = batch['frame_mask'].reshape(batch_size * max_windows, batch['frame_mask'].shape[-1])
            h_v, z_v = self.visual_encoder(flat_video, flat_frame_mask)
        z_content, z_t = self.content_encoder(flat_tactile_high, flat_tactile_mask)
        z_med_window = self.evidence_encoder(flat_tactile_low, flat_tactile_mask)
        z_med_sequence = z_med_window.reshape(batch_size, max_windows, -1)
        direction_embedding = self._direction_embedding_from_batch(
            batch,
            batch_size=batch_size,
            reference=z_med_window,
        )
        medium_direction_modulation = None
        conditioned_z_med_sequence = z_med_sequence
        if self.medium_direction_adapter is not None:
            if direction_embedding is None:
                raise RuntimeError('Medium direction adapter requires a direction embedding.')
            medium_direction_modulation = self.medium_direction_adapter.modulation(direction_embedding)
            conditioned_z_med_sequence = ResidualFiLMAdapter.apply_modulation(
                z_med_sequence,
                medium_direction_modulation[0],
                medium_direction_modulation[1],
            )
        medium_logits, p_medium, medium_hidden, medium_sequence_features = self.medium_head(
            conditioned_z_med_sequence,
            batch['window_lengths'],
        )
        flat_medium = p_medium.reshape(batch_size * max_windows, -1)
        flat_medium_features = medium_sequence_features.reshape(batch_size * max_windows, -1)
        state_context = torch.cat([z_med_window, flat_medium_features, flat_medium], dim=-1)
        residual_context = None
        residual_context_parts: list[torch.Tensor] = []
        pooled_context = None
        if self.use_interface_context:
            context_mask = batch.get('context_window_mask')
            if context_mask is None:
                context_mask = torch.zeros_like(batch['window_mask'])
            context_features = torch.cat(
                [
                    h_v.reshape(batch_size, max_windows, -1),
                    z_content.reshape(batch_size, max_windows, -1),
                    z_med_window.reshape(batch_size, max_windows, -1),
                ],
                dim=-1,
            )
            effective_context_mask = context_mask.bool() & batch['window_mask'].bool()
            pooled_context = self._pool_interface_context(context_features, effective_context_mask)
            expanded_context = pooled_context[:, None, :].expand(-1, max_windows, -1).reshape(batch_size * max_windows, -1)
            if self.policy_per_finger:
                expanded_context = expanded_context[:, None, :].expand(-1, self.policy_finger_count, -1)
            residual_context_parts.append(expanded_context)
        if self.use_reference_force_context:
            if self.policy_per_finger:
                finger_reference_force = batch.get('finger_reference_forces')
                if finger_reference_force is None:
                    reference_context = torch.zeros(
                        batch_size,
                        max_windows,
                        self.policy_finger_count,
                        1,
                        device=flat_medium.device,
                        dtype=flat_medium.dtype,
                    )
                else:
                    reference_context = finger_reference_force.to(device=flat_medium.device, dtype=flat_medium.dtype).unsqueeze(-1)
                    reference_context = torch.nan_to_num(reference_context, nan=0.0)
                    reference_context = reference_context / self.reference_force_context_scale
                residual_context_parts.append(reference_context.reshape(batch_size * max_windows, self.policy_finger_count, 1))
            else:
                reference_force = batch.get('reference_forces')
                if reference_force is None:
                    reference_context = torch.zeros(
                        batch_size,
                        max_windows,
                        1,
                        device=flat_medium.device,
                        dtype=flat_medium.dtype,
                    )
                else:
                    reference_context = reference_force.to(device=flat_medium.device, dtype=flat_medium.dtype).unsqueeze(-1)
                    reference_context = torch.nan_to_num(reference_context, nan=0.0)
                    reference_context = reference_context / self.reference_force_context_scale
                residual_context_parts.append(reference_context.reshape(batch_size * max_windows, 1))
        if residual_context_parts:
            residual_context = torch.cat(residual_context_parts, dim=-1)

        (
            attribute_outputs,
            sample_attribute_outputs,
            sample_physical_attribute_outputs,
            g_obj,
            task_context,
            sample_attribute_pool_mask,
        ) = self._build_task_context(
            h_v,
            z_content,
            batch,
        )
        interface_gate_override = None
        if self.policy_gate_source == 'soft_target':
            if 'soft_gate_targets' not in batch:
                raise RuntimeError("policy.gate_source='soft_target' requires soft_gate_targets in the batch.")
            interface_gate_override = batch['soft_gate_targets'].reshape(batch_size * max_windows)
        elif self.policy_gate_source == 'hard_phase':
            interface_gate_override = (batch['phase_labels'].reshape(batch_size * max_windows) == 1).to(flat_medium.dtype)
        elif self.gate_stabilizer is not None:
            raw_gate = p_medium[..., 1] if p_medium.shape[-1] > 1 else torch.zeros_like(batch['window_mask'], dtype=flat_medium.dtype)
            stabilized_gate = self.gate_stabilizer(raw_gate, batch['window_mask'])
            interface_gate_override = stabilized_gate.reshape(batch_size * max_windows)
        policy_direction_modulation = None
        if self.policy_direction_adapter is not None:
            if direction_embedding is None:
                raise RuntimeError('Policy direction adapter requires a direction embedding.')
            flat_direction_embedding = (
                direction_embedding[:, None, :]
                .expand(-1, max_windows, -1)
                .reshape(batch_size * max_windows, -1)
            )
            policy_direction_modulation = self.policy_direction_adapter.modulation(flat_direction_embedding)
        policy_outputs = self.policy_head(
            task_context,
            flat_medium,
            state_context=state_context,
            residual_context=residual_context,
            interface_gate_override=interface_gate_override,
            residual_modulation=policy_direction_modulation,
        )
        if self.policy_base_source == 'reference_force':
            policy_outputs = self._apply_reference_force_base_override(policy_outputs, batch)

        outputs = {
            'h_v': h_v,
            'z_v': z_v,
            'z_t': z_t,
            'z_content': z_content,
            'z_med_window': z_med_window,
            'medium_logits': medium_logits,
            'medium_probs': p_medium,
            'medium_hidden': medium_hidden,
            'medium_sequence_features': medium_sequence_features,
            'fragility_logits': attribute_outputs['fragility_logits'],
            'geometry_logits': attribute_outputs['geometry_logits'],
            'surface_logits': attribute_outputs['surface_logits'],
            'fragility_entropy': attribute_outputs['fragility_entropy'],
            'geometry_entropy': attribute_outputs['geometry_entropy'],
            'surface_entropy': attribute_outputs['surface_entropy'],
            'g_obj': g_obj,
            'force_pred': policy_outputs['force_pred'],
            'force_base': policy_outputs['force_base'],
            'force_interface_delta': policy_outputs['force_interface_delta'],
            'interface_gate': policy_outputs['interface_gate'],
        }
        for key in (
            'finger_force_pred',
            'finger_force_base',
            'finger_force_base_learned',
            'finger_force_interface_delta',
            'finger_force_interface_delta_raw',
            'finger_force_interface_delta_pos_raw',
            'finger_force_interface_delta_neg_raw',
            'finger_force_interface_delta_pos_magnitude',
            'finger_force_interface_delta_neg_magnitude',
            'finger_residual_direction_logit_neg',
            'finger_residual_direction_logit_pos',
            'finger_residual_direction_prob_neg',
            'finger_residual_direction_prob_pos',
            'force_interface_delta_raw',
            'force_interface_delta_pos_raw',
            'force_interface_delta_neg_raw',
            'force_interface_delta_pos_magnitude',
            'force_interface_delta_neg_magnitude',
            'residual_direction_logit_neg',
            'residual_direction_logit_pos',
            'residual_direction_prob_neg',
            'residual_direction_prob_pos',
        ):
            if key in policy_outputs:
                outputs[key] = policy_outputs[key]
        if sample_attribute_outputs is not None:
            outputs.update(
                {
                    'sample_fragility_logits': sample_attribute_outputs['fragility_logits'],
                    'sample_geometry_logits': sample_attribute_outputs['geometry_logits'],
                    'sample_surface_logits': sample_attribute_outputs['surface_logits'],
                    'sample_fragility_entropy': sample_attribute_outputs['fragility_entropy'],
                    'sample_geometry_entropy': sample_attribute_outputs['geometry_entropy'],
                    'sample_surface_entropy': sample_attribute_outputs['surface_entropy'],
                }
            )
        if sample_physical_attribute_outputs is not None:
            outputs.update(
                {
                    'sample_dry_mass_g_normalized_pred': sample_physical_attribute_outputs['dry_mass_g_normalized_pred'],
                    'sample_capacity_ratio_normalized_pred': sample_physical_attribute_outputs['capacity_ratio_normalized_pred'],
                    'sample_is_open_container_logit': sample_physical_attribute_outputs['is_open_container_logit'],
                }
            )
        if sample_attribute_pool_mask is not None:
            outputs['sample_attribute_pool_mask'] = sample_attribute_pool_mask
        if self.gate_stabilizer is not None and self.policy_gate_source == 'medium_prob':
            outputs['raw_interface_gate'] = flat_medium[..., 1] if flat_medium.shape[-1] > 1 else torch.zeros_like(policy_outputs['interface_gate'])
        if pooled_context is not None:
            outputs['interface_context_embedding'] = pooled_context
        if 'force_base_learned' in policy_outputs:
            outputs['force_base_learned'] = policy_outputs['force_base_learned']
        if direction_embedding is not None:
            outputs['transition_direction_embedding'] = direction_embedding
        if self.direction_diagnostics_enabled and medium_direction_modulation is not None:
            outputs['medium_direction_gamma'] = medium_direction_modulation[0]
            outputs['medium_direction_beta'] = medium_direction_modulation[1]
        if self.direction_diagnostics_enabled and policy_direction_modulation is not None:
            outputs['policy_direction_gamma'] = policy_direction_modulation[0]
            outputs['policy_direction_beta'] = policy_direction_modulation[1]
        return outputs

    def forward_online_step(
        self,
        batch: dict[str, torch.Tensor],
        *,
        medium_hidden: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        h_v, z_v = self.visual_encoder(batch['video'], batch['frame_mask'])
        z_content, z_t = self.content_encoder(batch['tactile_high'], batch['tactile_mask'])
        z_med_window = self.evidence_encoder(batch['tactile_low'], batch['tactile_mask'])
        direction_embedding = self._direction_embedding_from_batch(
            batch,
            batch_size=z_med_window.shape[0],
            reference=z_med_window,
        )
        medium_direction_modulation = None
        conditioned_z_med_window = z_med_window
        if self.medium_direction_adapter is not None:
            if direction_embedding is None:
                raise RuntimeError('Medium direction adapter requires a direction embedding.')
            medium_direction_modulation = self.medium_direction_adapter.modulation(direction_embedding)
            conditioned_z_med_window = ResidualFiLMAdapter.apply_modulation(
                z_med_window,
                medium_direction_modulation[0],
                medium_direction_modulation[1],
            )
        medium_logits, p_medium, next_hidden, medium_step_features = self.medium_head.step(
            conditioned_z_med_window,
            hidden_state=medium_hidden,
        )
        state_context = torch.cat([z_med_window, medium_step_features, p_medium], dim=-1)
        attribute_outputs, _, _, g_obj, task_context, _ = self._build_task_context(h_v, z_content)
        policy_direction_modulation = None
        if self.policy_direction_adapter is not None:
            if direction_embedding is None:
                raise RuntimeError('Policy direction adapter requires a direction embedding.')
            policy_direction_modulation = self.policy_direction_adapter.modulation(direction_embedding)
        policy_outputs = self.policy_head(
            task_context,
            p_medium,
            state_context=state_context,
            residual_context=None,
            residual_modulation=policy_direction_modulation,
        )
        if self.policy_base_source == 'reference_force':
            policy_outputs = self._apply_reference_force_base_override(policy_outputs, batch)
        outputs = {
            'h_v': h_v,
            'z_v': z_v,
            'z_t': z_t,
            'z_content': z_content,
            'z_med_window': z_med_window,
            'medium_logits': medium_logits,
            'medium_probs': p_medium,
            'medium_hidden': next_hidden,
            'medium_sequence_features': medium_step_features,
            'fragility_logits': attribute_outputs['fragility_logits'],
            'geometry_logits': attribute_outputs['geometry_logits'],
            'surface_logits': attribute_outputs['surface_logits'],
            'fragility_entropy': attribute_outputs['fragility_entropy'],
            'geometry_entropy': attribute_outputs['geometry_entropy'],
            'surface_entropy': attribute_outputs['surface_entropy'],
            'g_obj': g_obj,
            'force_pred': policy_outputs['force_pred'],
            'force_base': policy_outputs['force_base'],
            'force_interface_delta': policy_outputs['force_interface_delta'],
            'interface_gate': policy_outputs['interface_gate'],
        }
        for key in (
            'finger_force_pred',
            'finger_force_base',
            'finger_force_base_learned',
            'finger_force_interface_delta',
            'finger_force_interface_delta_raw',
            'finger_force_interface_delta_pos_raw',
            'finger_force_interface_delta_neg_raw',
            'finger_force_interface_delta_pos_magnitude',
            'finger_force_interface_delta_neg_magnitude',
            'finger_residual_direction_logit_neg',
            'finger_residual_direction_logit_pos',
            'finger_residual_direction_prob_neg',
            'finger_residual_direction_prob_pos',
            'force_interface_delta_raw',
            'force_interface_delta_pos_raw',
            'force_interface_delta_neg_raw',
            'force_interface_delta_pos_magnitude',
            'force_interface_delta_neg_magnitude',
            'residual_direction_logit_neg',
            'residual_direction_logit_pos',
            'residual_direction_prob_neg',
            'residual_direction_prob_pos',
        ):
            if key in policy_outputs:
                outputs[key] = policy_outputs[key]
        if 'force_base_learned' in policy_outputs:
            outputs['force_base_learned'] = policy_outputs['force_base_learned']
        if direction_embedding is not None:
            outputs['transition_direction_embedding'] = direction_embedding
        if self.direction_diagnostics_enabled and medium_direction_modulation is not None:
            outputs['medium_direction_gamma'] = medium_direction_modulation[0]
            outputs['medium_direction_beta'] = medium_direction_modulation[1]
        if self.direction_diagnostics_enabled and policy_direction_modulation is not None:
            outputs['policy_direction_gamma'] = policy_direction_modulation[0]
            outputs['policy_direction_beta'] = policy_direction_modulation[1]
        return outputs
