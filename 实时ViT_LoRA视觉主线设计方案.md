# 实时 ViT + LoRA 视觉主线设计方案

## 1. 方案定位

实时 ViT + LoRA 是本项目视觉侧的端到端主线方案。

它的核心目标是：在训练和推理阶段都实时运行视觉编码器，让视觉表征能够通过 LoRA 对跨介质抓取任务进行轻量适配，而不是完全依赖离线缓存的 frozen visual feature。

整体定位如下：

```text
window-level visual cache
    -> 最低成本 baseline / 快速 sanity check

frame-level visual cache + trainable temporal pooling
    -> 中等成本调参与消融方案

realtime ViT + LoRA
    -> 最终主线 / 端到端视觉适配方案
```

因此，实时 ViT + LoRA 不只是一个更重的实现版本，而是更接近最终部署形态的视觉路线。

## 2. 核心思想

当前项目使用 OpenCLIP ViT-B/16 作为视觉 backbone。

直接全量微调 ViT 的风险较高：

- 参数量大；
- 数据规模有限；
- 容易过拟合；
- 显存和训练时间成本高；
- 与多阶段训练衔接复杂。

因此，本方案采用：

```text
Frozen OpenCLIP ViT backbone + LoRA on attention q/v
```

也就是说，保留 OpenCLIP 的通用视觉表征能力，只在 attention 的 query/value 方向增加低秩可训练增量，用较小的参数量完成任务适配。

默认配置：

```yaml
visual:
  backbone: open_clip_vit_b_16
  model_name: ViT-B-16
  pretrained: true
  pretrained_tag: openai
  freeze_backbone: true
  use_lora: true
  lora_rank: 8
  lora_alpha: 16
  lora_dropout: 0.0
  lora_target_semantics: [q_proj, v_proj]
  lora_mapping_version: open_clip_mha_fused_qv_v1
```

## 3. 前向链路

实时 ViT + LoRA 的前向过程如下：

```text
视频窗口
    -> 每个窗口采样 N_v 帧
    -> ROI crop + resize + CLIP normalize
    -> OpenCLIP ViT-B/16 + q/v LoRA
    -> 每帧 pooled visual feature
    -> temporal attention pooling
    -> h_v
    -> attribute / medium / policy modules
```

具体到当前项目：

```text
video: [B, W, F, 3, 224, 224]
    -> flatten: [B * W * F, 3, 224, 224]
    -> OpenCLIP visual encoder
    -> frame features: [B * W, F, 512]
    -> VisualTemporalAttentionPooling
    -> h_v: [B * W, 512]
```

其中：

- `B`：batch size；
- `W`：每个 sample 的窗口数；
- `F`：每个窗口采样帧数，默认 8；
- `h_v`：窗口级视觉表征，输入后续对象属性、physical head 或 policy head。

## 4. 与缓存方案的区别

实时 ViT + LoRA 和视觉缓存方案的关键区别如下：

| 方案 | 训练时运行 ViT | 推理时运行 ViT | 可训练 LoRA | 可训练 temporal pooling | 成本 | 表达能力 |
|---|---:|---:|---:|---:|---|---|
| window-level cache | 否 | 否 | 否 | 否 | 最低 | 最低 |
| frame-level cache | 否 | 否 | 否 | 是 | 中低 | 中等 |
| realtime ViT + LoRA | 是 | 是 | 是 | 是 | 最高 | 最高 |

需要注意的是：

- LoRA 本身的推理开销较小；
- 主要计算成本来自 ViT backbone；
- 实时 ViT + LoRA 的复杂度主要体现在训练阶段，但推理阶段也需要实时运行 ViT，因此延迟高于缓存方案。

## 5. 训练复杂度分析

实时 ViT + LoRA 的主要复杂点在训练阶段。

### 5.1 显存压力

训练时每个 batch 都需要保留 ViT 前向的中间激活，用于 LoRA 参数反向传播。

因此，相比视觉缓存方案，它会显著增加：

```text
GPU 显存占用
反向传播计算量
训练时间
```

当前项目中通常需要降低：

```yaml
train:
  batch_size: 1 或 2

model:
  visual:
    max_windows_per_encode: 2 或 4
```

### 5.2 DataLoader 稳定性

Windows 环境下，多进程 DataLoader 加载 PyTorch/CUDA 相关 DLL 时，曾出现页面文件或系统提交内存不足问题。

因此，实时 ViT + LoRA 阶段建议优先使用：

```yaml
train:
  num_workers: 0
  pin_memory: false
  persistent_workers: false
```

如果环境稳定，再逐步提高 `num_workers`。

### 5.3 学习率分组

LoRA 参数应使用独立学习率，通常低于或接近主学习率。

当前默认：

```yaml
train:
  learning_rate: 0.0001
  base_learning_rate: 0.0001
  lora_learning_rate: 0.00005
```

推荐保留参数组区分：

```text
base / downstream modules
LoRA parameters
frozen visual backbone parameters
```

其中 frozen backbone 不参与 optimizer。

### 5.4 训练阶段衔接

推荐采用多阶段衔接，而不是从头端到端训练：

```text
Stage1:
  冻结视觉 backbone，不使用 LoRA或使用视觉缓存，先训练 medium / attribute。

Stage2:
  打开 realtime ViT + LoRA，引入 L_clip / L_inv，做视觉-触觉表征适配。

Stage3:
  冻结 visual_encoder 和感知模块，聚焦 policy head。

Stage4:
  打开 LoRA 和下游模块，做轻量联合微调。
```

这样可以避免策略损失过早干扰视觉和触觉表征。

## 6. 推理复杂度分析

实时 ViT + LoRA 在推理阶段的模型接口与缓存方案一致，最终都输出：

```text
h_v
medium_probs
attribute logits
force_pred
```

但计算路径不同：

```text
缓存方案:
  读取 h_v 或 frame_features
  不运行 ViT

实时 ViT + LoRA:
  实时读取视频帧
  运行 OpenCLIP ViT backbone
  应用 LoRA 增量
  temporal pooling
```

因此，推理阶段：

- 输出接口一致；
- 下游模块一致；
- LoRA 额外开销较小；
- 主要延迟来自 ViT backbone；
- 如果部署设备算力足够，可以直接使用实时 ViT + LoRA；
- 如果部署设备算力不足，可以退化为 frame-level cache 或预提取特征模式。

## 7. 推荐配置

### 7.1 Stage2 对齐配置

Stage2 适合用于打开 LoRA，并引入跨模态对齐损失。

#### 7.1.1 高成本 8 帧完整版

8 帧版本保留更多视觉时序信息，但训练成本非常高。基于 `stage2_alignment_physical_balanced` 在 RTX 5090 32GB 上的实测：

```text
num_frames_per_window: 8
batch_size: 1
max_windows_per_encode: 1
train samples: 146
val samples: 58
```

实际表现：

```text
20 epoch Stage2 约 5.5 - 6.5 天
GPU memory 约 31.8 / 32.6 GB
GPU utilization 接近 100%
```

因此 8 帧版本可以作为最终高成本确认实验，但不建议作为常规调参配置。

需要特别注意：

```text
8 frames + max_windows_per_encode 2
```

会把单次 ViT chunk 从：

```text
1 window * 8 frames = 8 frames
```

提升到：

```text
2 windows * 8 frames = 16 frames
```

在当前 32GB 显存环境下 OOM 风险很高。

#### 7.1.2 推荐可跑主线版：4 帧 LoRA fast

推荐把常规 Stage2 主线配置改为 4 帧版本：

```yaml
name: stage2_alignment_physical_balanced_lora4f_fast
loss_weights:
  clip: 1.0
  inv: 0.5
  med: 1.0
  attr: 1.0
  pol: 0.0
data:
  num_frames_per_window: 4
model:
  visual:
    use_lora: true
    max_windows_per_encode: 2
train:
  batch_size: 2
  num_workers: 0
  pin_memory: false
  persistent_workers: false
  learning_rate: 0.0001
  lora_learning_rate: 0.00005
```

该配置的核心折中是：

```text
4 frames + max_windows_per_encode 2
```

单次 ViT chunk 仍是：

```text
2 windows * 4 frames = 8 frames
```

这与已跑通的 8 帧慢速版：

```text
1 window * 8 frames = 8 frames
```

在 ViT chunk 帧数上接近，因此显存峰值预计相近，但训练吞吐会更好。

预期耗时：

```text
batch_size 2 稳定时：Stage2 20 epoch 约 2 天左右
若降到 batch_size 1：Stage2 20 epoch 约 3 天左右
```

OOM 回退顺序：

```yaml
# 首选
data:
  num_frames_per_window: 4
model:
  visual:
    max_windows_per_encode: 2
train:
  batch_size: 2

# 若 OOM，优先降 batch size
train:
  batch_size: 1

# 若仍 OOM，再降 chunk
model:
  visual:
    max_windows_per_encode: 1
```

该版本建议作为后续 realtime ViT + LoRA 的默认主线实验配置。8 帧版本只作为最终确认或高成本 ablation。

### 7.2 Stage4 联合微调配置

Stage4 适合用于最终端到端联合微调。

```yaml
name: stage4_joint_fold1
loss_weights:
  clip: 1.0
  inv: 0.5
  med: 1.0
  attr: 1.0
  pol: 2.0
model:
  visual:
    use_lora: true
    max_windows_per_encode: 2
train:
  batch_size: 1
  num_workers: 0
  pin_memory: false
  persistent_workers: false
  learning_rate: 0.00005
  lora_learning_rate: 0.00002
  grad_clip_norm: 1.0
```

### 7.3 策略阶段配置

如果策略阶段只希望训练 policy head，不希望继续更新视觉侧：

```yaml
freeze_modules:
  - visual_encoder
  - content_encoder
  - evidence_encoder
  - medium_head
  - attribute_head.shared
  - attribute_head.fragility
  - attribute_head.geometry
  - attribute_head.surface
model:
  visual:
    use_lora: true
```

这种情况下，训练时仍然实时运行 visual encoder 前向，但 visual encoder 不更新参数。

## 8. 与损失函数的关系

实时 ViT + LoRA 最适合配合以下损失：

```text
L_clip:
  视觉-触觉窗口表征对齐。

L_inv:
  跨介质内容不变性约束。

L_med:
  Water / Interface / Air 介质识别。

L_attr:
  对象属性识别。

L_pol:
  最终抓取力策略监督。
```

其中：

- Stage2 重点使用 `L_clip + L_inv + L_med + L_attr`；
- Stage3 重点使用 `L_pol`；
- Stage4 使用完整联合损失。

这样可以让视觉 LoRA 先学习与触觉和对象属性相关的表征，再进入策略控制联合优化。

## 9. 推荐实验矩阵

为了证明实时 ViT + LoRA 作为主线的必要性，建议做以下对比：

| 实验 | 视觉路径 | 目的 |
|---|---|---|
| A | window-level cache | 最低成本 baseline |
| B | frame-level cache + trainable pooling | 验证帧级时序选择收益 |
| C | realtime ViT frozen, no LoRA | 验证实时视觉但不适配的效果 |
| D | realtime ViT + LoRA rank 8 | 主线方案 |
| E | realtime ViT + LoRA rank 16 | 检查 LoRA 容量上限 |

重点比较指标：

```text
medium_f1_interface
medium_macro_f1
attribute accuracy / macro_f1
control_interface_mae
control_interface_hit_rate_200
training time per epoch
GPU memory usage
```

## 10. 风险与缓解策略

### 10.1 过拟合风险

风险：

```text
数据规模有限，LoRA 仍可能过拟合训练对象。
```

缓解：

```text
使用 object-level unseen split
降低 LoRA rank
增加 weight decay
early stopping
只在 Stage2/Stage4 打开 LoRA
```

### 10.2 训练不稳定

风险：

```text
batch size 过小，梯度噪声较大。
```

缓解：

```text
使用 gradient clipping
使用 warmup + cosine schedule
必要时使用梯度累积
先感知预训练再策略训练
```

### 10.3 资源成本高

风险：

```text
训练时间长，调参成本高。
```

缓解：

```text
先用 frame-level cache 做快速 ablation
再用 realtime ViT + LoRA 跑少量关键配置
降低 max_windows_per_encode
减少 num_frames_per_window 做低成本试验
```

### 10.4 推理延迟高

风险：

```text
部署时实时 ViT 推理可能不满足控制频率。
```

缓解：

```text
降低视频帧数
降低推理频率
异步视觉编码
使用最近窗口视觉特征复用
部署受限时退化到 frame-level cache / frozen feature 模式
```

## 11. 面试表达口径

可以这样说明：

> 我们把实时 ViT + LoRA 作为最终视觉主线。原因是跨介质抓取中的视觉信息不仅包含物体外观，也包含水面、出水状态、遮挡和动态场景上下文，完全使用离线 frozen feature 会限制视觉适配能力。直接全量微调 ViT 成本和过拟合风险都太高，因此我们冻结 OpenCLIP ViT-B/16 主体，只在 attention 的 q/v 投影上加入 LoRA，用很小的可训练参数完成任务适配。训练阶段它比缓存方案更复杂，主要体现在显存、速度和稳定性；但推理阶段接口是一样的，只是需要实时运行 ViT，LoRA 本身带来的额外开销很小。

## 12. 最终建议

建议项目视觉路线最终采用：

```text
主线:
  realtime OpenCLIP ViT-B/16 + q/v LoRA

快速实验:
  frame-level frozen OpenCLIP cache + trainable temporal pooling

低成本 baseline:
  window-level h_v cache
```

这样既保留最终端到端模型的表达能力，也保留快速调参和消融实验的工程效率。
