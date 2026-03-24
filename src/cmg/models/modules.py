from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

try:
    import open_clip
except ImportError:  # pragma: no cover
    open_clip = None

try:
    from peft import LoraConfig, get_peft_model
except ImportError:  # pragma: no cover
    LoraConfig = None
    get_peft_model = None

try:
    from torchvision.models import resnet18, vit_b_16
    from torchvision.models import ResNet18_Weights, ViT_B_16_Weights
except ImportError:  # pragma: no cover
    resnet18 = None
    vit_b_16 = None
    ResNet18_Weights = None
    ViT_B_16_Weights = None


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs)


class TemporalAttentionPooling(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.randn(input_dim))

    def forward(self, sequence: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        scores = torch.einsum('bld,d->bl', sequence, self.query) / math.sqrt(sequence.shape[-1])
        if mask is not None:
            scores = scores.masked_fill(~mask, float('-inf'))
        weights = torch.softmax(scores, dim=-1)
        return torch.einsum('bl,bld->bd', weights, sequence)


class VisualTemporalAttentionPooling(nn.Module):
    def __init__(self, config: dict) -> None:
        super().__init__()
        token_dim = int(config['token_dim'])
        pooling_config = config['pooling']
        score_hidden_dim = int(pooling_config['score_hidden_dim'])
        self.pre_norm_enabled = bool(pooling_config.get('pre_norm', True))
        self.output_post_norm_enabled = bool(pooling_config.get('output_post_norm', True))
        self.use_mean_residual = bool(pooling_config.get('use_mean_residual', True))
        self.dropout = nn.Dropout(float(pooling_config.get('dropout', 0.0)))
        self.pre_norm = nn.LayerNorm(token_dim)
        self.score_mlp = nn.Sequential(
            nn.Linear(token_dim, score_hidden_dim),
            nn.GELU(),
        )
        self.global_query = nn.Parameter(torch.randn(score_hidden_dim))
        self.output_norm = nn.LayerNorm(token_dim)

    def forward(self, tokens: torch.Tensor, frame_mask: torch.Tensor) -> torch.Tensor:
        mask = frame_mask.bool()
        safe_mask = mask.clone()
        invalid_rows = ~safe_mask.any(dim=-1)
        if invalid_rows.any():
            safe_mask[invalid_rows, 0] = True
        normalized = self.pre_norm(tokens) if self.pre_norm_enabled else tokens
        hidden = self.score_mlp(normalized)
        scores = torch.einsum('btd,d->bt', hidden, self.global_query) / math.sqrt(hidden.shape[-1])
        scores = scores.masked_fill(~safe_mask, float('-inf'))
        attention = torch.softmax(scores, dim=-1)
        attention = self.dropout(attention)
        context = torch.einsum('bt,btd->bd', attention, tokens)
        if self.use_mean_residual:
            valid_counts = safe_mask.sum(dim=-1, keepdim=True).clamp_min(1)
            mean_feature = (tokens * safe_mask.unsqueeze(-1)).sum(dim=1) / valid_counts
            context = context + mean_feature
        if self.output_post_norm_enabled:
            context = self.output_norm(context)
        return context


class LowRankAdapter(nn.Module):
    def __init__(self, feature_dim: int, rank: int) -> None:
        super().__init__()
        self.down = nn.Linear(feature_dim, rank, bias=False)
        self.up = nn.Linear(rank, feature_dim, bias=False)
        nn.init.zeros_(self.up.weight)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs + self.up(self.down(inputs))


class SmallCNNBackbone(nn.Module):
    def __init__(self, output_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, output_dim),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs)


class OpenClipQVLoraAttention(nn.Module):
    def __init__(self, base_attn: nn.MultiheadAttention, rank: int, alpha: int, dropout: float) -> None:
        super().__init__()
        if not getattr(base_attn, '_qkv_same_embed_dim', True):
            raise NotImplementedError('Only same-embed-dim MultiheadAttention is supported for q/v LoRA mapping.')
        self.base_attn = base_attn
        self.rank = int(rank)
        self.scaling = float(alpha) / float(max(1, rank))
        self.lora_dropout = nn.Dropout(float(dropout))
        embed_dim = int(base_attn.embed_dim)
        self.lora_q_A = nn.Linear(embed_dim, rank, bias=False)
        self.lora_q_B = nn.Linear(rank, embed_dim, bias=False)
        self.lora_v_A = nn.Linear(embed_dim, rank, bias=False)
        self.lora_v_B = nn.Linear(rank, embed_dim, bias=False)
        nn.init.kaiming_uniform_(self.lora_q_A.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.lora_v_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_q_B.weight)
        nn.init.zeros_(self.lora_v_B.weight)

    def _merged_in_proj_weight(self) -> torch.Tensor:
        in_proj_weight = self.base_attn.in_proj_weight
        embed_dim = self.base_attn.embed_dim
        delta = torch.zeros_like(in_proj_weight)
        q_delta = self.lora_q_B.weight @ self.lora_q_A.weight
        v_delta = self.lora_v_B.weight @ self.lora_v_A.weight
        delta[:embed_dim] = q_delta * self.scaling
        delta[2 * embed_dim : 3 * embed_dim] = v_delta * self.scaling
        return in_proj_weight + delta

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
        need_weights: bool = True,
        attn_mask: torch.Tensor | None = None,
        average_attn_weights: bool = True,
        is_causal: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        attn = self.base_attn
        merged_weight = self._merged_in_proj_weight()
        return F.multi_head_attention_forward(
            query=query,
            key=key,
            value=value,
            embed_dim_to_check=attn.embed_dim,
            num_heads=attn.num_heads,
            in_proj_weight=merged_weight,
            in_proj_bias=attn.in_proj_bias,
            bias_k=attn.bias_k,
            bias_v=attn.bias_v,
            add_zero_attn=attn.add_zero_attn,
            dropout_p=attn.dropout,
            out_proj_weight=attn.out_proj.weight,
            out_proj_bias=attn.out_proj.bias,
            training=self.training,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            attn_mask=attn_mask,
            average_attn_weights=average_attn_weights,
            use_separate_proj_weight=False,
            q_proj_weight=None,
            k_proj_weight=None,
            v_proj_weight=None,
            static_k=None,
            static_v=None,
            is_causal=is_causal,
        )



def apply_open_clip_qv_lora(
    module: nn.Module,
    *,
    rank: int,
    alpha: int,
    dropout: float,
    attention_module_name: str,
) -> nn.Module:
    replaced = 0

    def _replace(parent: nn.Module) -> None:
        nonlocal replaced
        for child_name, child in list(parent.named_children()):
            if child_name == attention_module_name and isinstance(child, nn.MultiheadAttention):
                setattr(parent, child_name, OpenClipQVLoraAttention(child, rank=rank, alpha=alpha, dropout=dropout))
                replaced += 1
            else:
                _replace(child)

    _replace(module)
    if replaced == 0:
        raise RuntimeError(
            f'No attention modules named {attention_module_name!r} were found for OpenCLIP q/v LoRA mapping.'
        )
    return module



def freeze_non_lora_parameters(module: nn.Module) -> None:
    for name, parameter in module.named_parameters():
        parameter.requires_grad = 'lora_' in name



def resolve_open_clip_lora_strategy(config: dict[str, Any], backbone_name: str) -> dict[str, Any]:
    semantics = [str(value) for value in config.get('lora_target_semantics', [])]
    if semantics:
        if semantics != ['q_proj', 'v_proj']:
            raise RuntimeError(
                f'Unsupported LoRA semantic target specification for the official OpenCLIP path: {semantics}'
            )
        backend_mapping = config.get('lora_semantic_backend_mapping', {})
        mapping = backend_mapping.get(backbone_name)
        if mapping is None or 'attention_module' not in mapping:
            raise RuntimeError(
                f'No versioned backend mapping is configured for semantic targets {semantics} on {backbone_name}. '
                'Please either provide an accepted mapping or mark this backend as an implementation deviation.'
            )
        return {
            'kind': 'semantic_qv',
            'attention_module': str(mapping['attention_module']),
            'mapping_version': str(config.get('lora_mapping_version', 'unspecified')),
            'semantics': semantics,
        }

    explicit = config.get('lora_target_modules')
    if explicit:
        return {
            'kind': 'peft',
            'target_modules': [str(value) for value in explicit],
            'mapping_version': str(config.get('lora_mapping_version', 'legacy_explicit_modules')),
            'semantics': [],
        }

    raise RuntimeError('No LoRA target specification was provided for the visual encoder.')


class VisualEncoder(nn.Module):
    def __init__(self, config: dict) -> None:
        super().__init__()
        backbone_name = str(config.get('backbone', 'small_cnn')).lower()
        token_dim = int(config['token_dim'])
        proj_dim = int(config['proj_dim'])
        self.temporal_pool = VisualTemporalAttentionPooling(config)
        self.align_head = MLP(token_dim, token_dim, proj_dim)
        self.backbone_kind = 'fallback'
        self.base_visual: nn.Module | None = None
        self.adapter: nn.Module | None = None
        self.token_projection: nn.Module = nn.Identity()
        self.lora_mapping_version = None

        if backbone_name in {'open_clip_vit_b_16', 'clip_vit_b_16', 'vit_b_16'} and open_clip is not None:
            self.backbone_kind = 'open_clip'
            model_name = str(config.get('model_name', 'ViT-B-16'))
            pretrained_tag = config.get('pretrained_tag')
            if pretrained_tag is None and bool(config.get('pretrained', False)):
                pretrained_tag = 'openai'
            clip_model = open_clip.create_model(
                model_name,
                pretrained=pretrained_tag,
                precision='fp32',
                device='cpu',
                force_quick_gelu=bool(config.get('force_quick_gelu', False)),
            )
            visual: nn.Module = clip_model.visual
            if bool(config.get('use_lora', True)):
                lora_strategy = resolve_open_clip_lora_strategy(config, backbone_name)
                if lora_strategy['kind'] == 'semantic_qv':
                    visual = apply_open_clip_qv_lora(
                        visual,
                        rank=int(config.get('lora_rank', 8)),
                        alpha=int(config.get('lora_alpha', 16)),
                        dropout=float(config.get('lora_dropout', 0.0)),
                        attention_module_name=lora_strategy['attention_module'],
                    )
                else:
                    if get_peft_model is None or LoraConfig is None:
                        raise RuntimeError('peft is required when explicit LoRA target modules are configured.')
                    lora_config = LoraConfig(
                        r=int(config.get('lora_rank', 8)),
                        lora_alpha=int(config.get('lora_alpha', 16)),
                        target_modules=list(lora_strategy['target_modules']),
                        lora_dropout=float(config.get('lora_dropout', 0.0)),
                        bias='none',
                    )
                    visual = get_peft_model(visual, lora_config)
                self.lora_mapping_version = str(lora_strategy['mapping_version'])
            self.backbone = visual
            self.base_visual = visual.base_model.model if hasattr(visual, 'base_model') and hasattr(visual.base_model, 'model') else visual
            feature_dim = int(self.base_visual.output_dim)
            self.token_projection = nn.Linear(feature_dim, token_dim) if feature_dim != token_dim else nn.Identity()
            if bool(config.get('freeze_backbone', False)):
                if bool(config.get('use_lora', True)):
                    freeze_non_lora_parameters(self.backbone)
                else:
                    for parameter in self.backbone.parameters():
                        parameter.requires_grad = False
        else:
            pretrained = bool(config.get('pretrained', False))
            if backbone_name == 'vit_b_16' and vit_b_16 is not None:
                weights = ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
                backbone = vit_b_16(weights=weights)
                feature_dim = backbone.heads.head.in_features
                backbone.heads = nn.Identity()
                self.backbone = backbone
            elif backbone_name == 'resnet18' and resnet18 is not None:
                weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
                backbone = resnet18(weights=weights)
                feature_dim = backbone.fc.in_features
                backbone.fc = nn.Identity()
                self.backbone = backbone
            else:
                feature_dim = token_dim
                self.backbone = SmallCNNBackbone(output_dim=token_dim)
            self.adapter = LowRankAdapter(feature_dim, rank=int(config.get('adapter_rank', 16)))
            self.token_projection = nn.Linear(feature_dim, token_dim) if feature_dim != token_dim else nn.Identity()
            if bool(config.get('freeze_backbone', False)):
                for parameter in self.backbone.parameters():
                    parameter.requires_grad = False

    def _forward_open_clip_frames(self, frames: torch.Tensor) -> torch.Tensor:
        if self.base_visual is None:
            raise RuntimeError('OpenCLIP visual backbone is not initialized.')
        x = self.base_visual._embeds(frames)
        x = self.base_visual.transformer(x)
        pooled, _ = self.base_visual._pool(x)
        if self.base_visual.proj is not None:
            pooled = pooled @ self.base_visual.proj
        return pooled

    def forward(self, video: torch.Tensor, frame_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, time_steps = video.shape[:2]
        frames = video.reshape(batch_size * time_steps, *video.shape[2:])
        if self.backbone_kind == 'open_clip':
            features = self._forward_open_clip_frames(frames)
        else:
            features = self.backbone(frames)
            if self.adapter is not None:
                features = self.adapter(features)
        tokens = self.token_projection(features).reshape(batch_size, time_steps, -1)
        window_features = self.temporal_pool(tokens, frame_mask)
        aligned = F.normalize(self.align_head(window_features), dim=-1)
        return window_features, aligned


class TactileContentEncoder(nn.Module):
    def __init__(self, config: dict, proj_dim: int) -> None:
        super().__init__()
        legacy_transformer = 'content_encoder_type' not in config and 'sensor_dim' in config and 'content_heads' in config
        encoder_type = str(config.get('content_encoder_type', 'legacy_transformer' if legacy_transformer else 'cnn1d')).lower()
        self.encoder_type = encoder_type
        input_dim = int(config['input_dim'])
        hidden_dim = int(config['content_hidden_dim'])

        if encoder_type == 'cnn1d':
            kernel_size = int(config.get('content_kernel_size', 3))
            num_layers = int(config.get('content_layers', 2))
            padding = kernel_size // 2
            layers: list[nn.Module] = []
            in_channels = input_dim
            for _ in range(num_layers):
                layers.append(nn.Conv1d(in_channels, hidden_dim, kernel_size=kernel_size, padding=padding))
                layers.append(nn.GELU())
                in_channels = hidden_dim
            self.conv = nn.Sequential(*layers)
            self.sensor_mlp = None
            self.spatial_projection = None
            self.temporal_encoder = None
        elif encoder_type in {'legacy_transformer', 'transformer'}:
            num_taxels = int(config['num_taxels'])
            axis_dim = int(config['axis_dim'])
            if input_dim != num_taxels * axis_dim:
                raise RuntimeError(
                    f'Legacy tactile transformer expects input_dim == num_taxels * axis_dim, got {input_dim} vs {num_taxels} * {axis_dim}.'
                )
            sensor_dim = int(config.get('sensor_dim', 16))
            num_layers = int(config.get('content_layers', 2))
            num_heads = int(config.get('content_heads', 4))
            self.num_taxels = num_taxels
            self.axis_dim = axis_dim
            self.conv = None
            self.sensor_mlp = nn.Sequential(
                nn.Linear(axis_dim, sensor_dim),
                nn.GELU(),
                nn.Linear(sensor_dim, sensor_dim),
                nn.GELU(),
            )
            self.spatial_projection = nn.Linear(num_taxels * sensor_dim, hidden_dim)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=int(config.get('content_ffn_dim', hidden_dim * 2)),
                activation='gelu',
                batch_first=True,
            )
            self.temporal_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        else:
            raise RuntimeError(f'Unsupported tactile content encoder type: {encoder_type!r}.')

        self.temporal_pool = TemporalAttentionPooling(hidden_dim)
        self.align_head = MLP(hidden_dim, hidden_dim, proj_dim)

    def forward(self, tactile: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.encoder_type == 'cnn1d':
            features = self.conv(tactile.transpose(1, 2)).transpose(1, 2)
        else:
            batch_size, time_steps, _ = tactile.shape
            features = tactile.reshape(batch_size, time_steps, self.num_taxels, self.axis_dim)
            features = self.sensor_mlp(features)
            features = features.reshape(batch_size, time_steps, -1)
            features = self.spatial_projection(features)
            features = self.temporal_encoder(
                features,
                src_key_padding_mask=~mask if mask is not None else None,
            )
        content = self.temporal_pool(features, mask=mask)
        aligned = F.normalize(self.align_head(content), dim=-1)
        return content, aligned

class TactileEvidenceEncoder(nn.Module):
    def __init__(self, config: dict) -> None:
        super().__init__()
        hidden_dim = int(config['evidence_hidden_dim'])
        kernel_size = int(config['evidence_kernel_size'])
        padding = kernel_size // 2
        self.conv = nn.Sequential(
            nn.Conv1d(int(config['input_dim']), hidden_dim, kernel_size=kernel_size, padding=padding),
            nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=kernel_size, padding=padding),
            nn.GELU(),
        )
        self.temporal_pool = TemporalAttentionPooling(hidden_dim)

    def forward(self, tactile: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        features = self.conv(tactile.transpose(1, 2)).transpose(1, 2)
        return self.temporal_pool(features, mask=mask)


class MediumBeliefHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.classifier = nn.Linear(hidden_dim, 3)

    def forward(self, sequence: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        packed = pack_padded_sequence(
            sequence,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_outputs, hidden = self.gru(packed)
        outputs, _ = pad_packed_sequence(
            packed_outputs,
            batch_first=True,
            total_length=sequence.shape[1],
        )
        logits = self.classifier(outputs)
        probabilities = torch.softmax(logits, dim=-1)
        return logits, probabilities, hidden[-1]

    def step(
        self,
        window_features: torch.Tensor,
        *,
        hidden_state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        outputs, next_hidden = self.gru(window_features.unsqueeze(1), hidden_state)
        logits = self.classifier(outputs[:, 0])
        probabilities = torch.softmax(logits, dim=-1)
        return logits, probabilities, next_hidden


class MultiAttributeHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, object_feature_dim: int) -> None:
        super().__init__()
        self.shared = MLP(input_dim, hidden_dim, hidden_dim)
        self.fragility = nn.Linear(hidden_dim, 3)
        self.geometry = nn.Linear(hidden_dim, 4)
        self.surface = nn.Linear(hidden_dim, 2)
        self.object_projection = nn.Linear(3 + 4 + 2 + 3, object_feature_dim)

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        shared = self.shared(features)
        fragility_logits = self.fragility(shared)
        geometry_logits = self.geometry(shared)
        surface_logits = self.surface(shared)
        fragility_probs = torch.softmax(fragility_logits, dim=-1)
        geometry_probs = torch.softmax(geometry_logits, dim=-1)
        surface_probs = torch.softmax(surface_logits, dim=-1)
        fragility_entropy = -(fragility_probs * torch.log(fragility_probs.clamp_min(1e-8))).sum(dim=-1, keepdim=True)
        geometry_entropy = -(geometry_probs * torch.log(geometry_probs.clamp_min(1e-8))).sum(dim=-1, keepdim=True)
        surface_entropy = -(surface_probs * torch.log(surface_probs.clamp_min(1e-8))).sum(dim=-1, keepdim=True)
        object_context = torch.cat(
            [
                fragility_probs,
                geometry_probs,
                surface_probs,
                fragility_entropy,
                geometry_entropy,
                surface_entropy,
            ],
            dim=-1,
        )
        return {
            'fragility_logits': fragility_logits,
            'geometry_logits': geometry_logits,
            'surface_logits': surface_logits,
            'fragility_entropy': fragility_entropy,
            'geometry_entropy': geometry_entropy,
            'surface_entropy': surface_entropy,
            'g_obj_context': object_context,
        }


class PolicyHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, film_hidden_dim: int) -> None:
        super().__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.hidden_layer = nn.Linear(hidden_dim, hidden_dim)
        self.film = nn.Sequential(
            nn.Linear(3, film_hidden_dim),
            nn.GELU(),
            nn.Linear(film_hidden_dim, hidden_dim * 2),
        )
        self.output_layer = nn.Linear(hidden_dim, 1)

    def forward(self, task_context: torch.Tensor, p_medium: torch.Tensor) -> torch.Tensor:
        hidden = F.gelu(self.input_layer(task_context))
        gamma, beta = self.film(p_medium).chunk(2, dim=-1)
        hidden = (1.0 + gamma) * hidden + beta
        hidden = F.gelu(self.hidden_layer(hidden))
        return self.output_layer(hidden).squeeze(-1)



