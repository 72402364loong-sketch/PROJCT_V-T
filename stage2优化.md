
---

# Stage 2 显存优化实施单

## 一、目标

当前 Stage 2 的主要压力点不只是 `batch_size`，而是：

- Stage 2 是当前正式链路里首次打开视觉 LoRA 的阶段。
- 当前实现会把一个 batch 内所有有效窗口先展平，再送入视觉编码器。
- `L_clip / L_inv` 实际只使用 stable Water / stable Air 窗口，但当前视觉前向仍会为更多窗口建图。

这份文档的目标不是改写论文训练语义，而是把 Stage 2 的优化方案整理成：

- 与现有代码实现一致
- 与整套 Stage 1 -> 4 设计一致
- 可以直接落地，不引入“文档有、代码没有”的配置项

---

## 二、必须保持不变的设计前提

下面这些属于当前工程和论文整体思路的一部分，本轮不应改动：

1. 阶段语义保持不变

- Stage 1：冻结视觉 backbone，不启用视觉 LoRA，主做介质和属性感知。
- Stage 2：打开视觉 LoRA，引入视触对齐损失。
- Stage 3：保留 LoRA 结构，但跟随 `visual_encoder` 一起冻结，聚焦策略回归。
- Stage 4：重新参与联合微调。

2. 跨阶段衔接语义保持不变

- `--init-from` 只加载 `model.state_dict()`。
- `--resume` 只用于同一 `run_dir`、同一 stage 的断点续训。
- Stage 1 -> Stage 2 / 4 首次注入 LoRA 时，允许 `lora_*` 参数缺失，并做 attention key remap。

3. Stage 2 的目标函数保持不变

- 仍然训练 `L_med + L_attr + L_clip + L_inv`。
- `L_med` 仍对全部有效窗口计算。
- `L_clip / L_inv` 仍然只对 stable Water / stable Air 窗口计算。

说明：
这点在当前代码里已经实现，不需要再额外新增“eligible_only”类配置。

---

## 三、与当前工程冲突的原提法，以及修正方式

下面这些提法和当前代码能力不完全一致，需要修正。

### 1. `physical_batch_size`

冲突原因：
当前训练器只有 `train.batch_size`，它就是实际物理 batch size，没有单独的 `physical_batch_size` 字段。

适配写法：

- 直接使用 `train.batch_size`
- 如果想表达“物理 batch 很小”，就在文档里明确 `train.batch_size=1` 或 `2`

### 2. `grad_accum_steps`

冲突原因：
当前 [training.py](D:/project-v&t/src/cmg/training.py) 没有梯度累积实现。

适配写法：

- 不能把它写成“当前可配置项”
- 只能列为“后续代码改造项”

### 3. `precision_mode: bf16`

冲突原因：
当前代码只有 `train.amp_enabled: true/false`，并且 `autocast` 没有暴露 `dtype` 配置；现实现不是一个通用的 `precision_mode` 框架。

适配写法：

- 当前可用配置只有 `amp_enabled`
- 如果 Stage 2 上 fp16 AMP 不稳定，当前最稳妥的做法是先在 Stage 2 显式设为 `amp_enabled: false`
- `bf16` 应列为“后续训练器能力扩展项”，不能写成现成可用开关

### 4. `visual_chunk_size`

冲突原因：
当前代码已经有同类能力，但配置名不是 `visual_chunk_size`，而是：

- `model.visual.max_windows_per_encode`

适配写法：

- 不新增新名字
- 直接复用现有 `max_windows_per_encode`

### 5. `tbptt_window_len`

冲突原因：
当前 `MediumBeliefHead` 仍然对整段序列做一次 GRU 前向，没有 TBPTT 切段逻辑。

适配写法：

- 不应写成当前可配能力
- 只能作为后续结构改造项

### 6. `visual_grad_checkpointing`

冲突原因：
当前 `VisualEncoder` 没有实现 OpenCLIP transformer block 的 gradient checkpointing 开关。

适配写法：

- 不能写成现有参数
- 只能列为后续优化项

### 7. `feature_queue / memory bank`

冲突原因：
当前 `losses.py` 和训练循环没有 queue 机制。

适配写法：

- 不能写成当前配置
- 只能作为后续增强项

### 8. `training:` / `stage2:` 配置块

冲突原因：
当前阶段配置文件的结构是：

- `model:`
- `train:`
- `loss_weights:`
- 其他 stage 元数据

并没有 `training:` 或 `stage2:` 这样的额外命名空间。

适配写法：

- 用现有 `train:` 和 `model.visual:` 结构表达

---

## 四、当前工程下可直接落地的 Stage 2 优化方案

### P0：立即可做，且与现有实现完全兼容

#### P0-1. 显式覆盖 Stage 2 的物理 batch

建议：

- 在 `configs/stages/stage2_alignment_fold1.yaml` 里显式写 `train.batch_size`
- 默认建议先用 `1`

原因：

- 基础训练配置里的 `batch_size=8` 对 Stage 2 来说过大
- Stage 2 是第一次打开视觉 LoRA 的阶段，显存压力与 Stage 1 不同

建议值：

```yaml
train:
  epochs: 20
  batch_size: 1
```

#### P0-2. 显式缩小视觉编码 chunk

建议：

- 使用当前已存在的 `model.visual.max_windows_per_encode`
- 默认建议从 `4` 开始
- 如果仍紧张，再降到 `2`

原因：

- 当前 [modules.py](D:/project-v&t/src/cmg/models/modules.py) 已经支持把窗口分块后再送入视觉 backbone
- 这是当前代码里唯一已经实现的视觉显存削峰开关

建议值：

```yaml
model:
  visual:
    use_lora: true
    max_windows_per_encode: 4
```

#### P0-3. Stage 2 显式关闭 AMP，先保证可稳定训练

建议：

- 如果已经观察到 `batch_size=1 + AMP` 仍不稳定，那么当前实现下应先关闭 AMP
- 在真正接入 bf16 精度控制前，不应把“默认 bf16”写成当前方案

建议值：

```yaml
train:
  amp_enabled: false
```

说明：

- 当前训练器的混合精度是布尔开关，不是可选 `fp16/bf16` 的精度模式
- 如果后续要切到 bf16，应该先改训练器接口，再改文档和配置

#### P0-4. 让 object-aware 采样语义与小 batch 一致

建议：

- 如果 `batch_size=1`，建议同时显式写 `samples_per_object: 1`

原因：

- 当前 `ObjectAwareBatchSampler` 在 `batch_size=1` 时仍可工作
- 但把 `samples_per_object` 写成 `1` 更符合这个阶段的小 batch 现实

建议值：

```yaml
train:
  batch_size: 1
  samples_per_object: 1
```

---

## 五、当前代码里已经成立，不需要再额外“规划实现”的点

下面这些内容在现有代码中已经成立，文档里不需要再写成待实现事项。

### 1. `--init-from` 与 `--resume` 的边界

当前 [train.py](D:/project-v&t/scripts/train.py) 和 [training.py](D:/project-v&t/src/cmg/training.py) 已经保证：

- `--resume` 只能恢复同一 stage / 同一 run
- `--init-from` 用于跨阶段初始化
- LoRA 首次注入时允许预期范围内的参数差异

### 2. `L_clip / L_inv` 只用 stable Water / stable Air

当前 [losses.py](D:/project-v&t/src/cmg/losses.py) 已经通过：

- `stable_masks`
- `stable_phases`

把 `L_clip / L_inv` 限制在 stable Water / stable Air 窗口上。

### 3. 视觉前向已经具备 chunk 化基础

当前 [modules.py](D:/project-v&t/src/cmg/models/modules.py) 中的 `VisualEncoder.forward()` 已经通过 `max_windows_per_encode` 分块编码窗口，只是它仍对“所有有效窗口”做视觉前向，而不是只对某个窗口子集做视觉前向。

---

## 六、确实值得做，但需要代码改造后才能写进正式配置的项

下面这些方向和论文思路并不冲突，但它们目前还不是现有工程能力，应标注为“后续增强项”。

### 1. 只让部分窗口进入 trainable visual path

方向说明：

- 保留 `L_med` 的全窗口语义
- 只让 `L_attr`、`L_clip`、`L_inv` 需要的窗口进入视觉建图路径

与当前代码的差距：

- 当前 `CrossMediumSystem.forward()` 默认给所有窗口都算 `h_v / z_v`
- 如果要选窗，需要同步改：
  - 前向逻辑
  - 属性头输入对齐方式
  - loss 对应索引

结论：

- 这是合理的后续优化方向
- 但不应写成“当前配置即可开启”

### 2. 属性损失只采样少量代表窗口

方向说明：

- 因为属性标签是对象级标签，理论上没必要对样本内每个窗口都强制算属性损失

与当前代码的差距：

- 当前 `L_attr` 对所有有效窗口计算
- 若改成代表窗口采样，需要重写：
  - 窗口采样策略
  - loss 索引逻辑
  - 日志统计

结论：

- 方向合理
- 但属于代码改造项，不是现成参数项

### 3. bf16 优先

方向说明：

- 这和当前论文目标并不冲突，也可能更稳

与当前代码的差距：

- 训练器还没有 `precision_mode`
- `autocast` 也未按 `dtype` 配置化

结论：

- 应列为训练器增强项
- 在代码支持前，不应写成现成配置

### 4. Gradient checkpointing

方向说明：

- 对 Stage 2 / Stage 4 都可能有帮助

与当前代码的差距：

- OpenCLIP 视觉路径没有该开关

结论：

- 可以做
- 但当前不能当作已支持能力

### 5. TBPTT

方向说明：

- 能进一步削减长序列训练的反向图

与当前代码的差距：

- 当前 GRU 仍是整段 pack/pad 序列训练

结论：

- 可以做
- 但会影响训练器与前向接口，不是简单加一个配置就能启用

### 6. Feature queue / memory bank

方向说明：

- 在超小 batch 下补足对比学习的负样本密度

与当前代码的差距：

- 现有损失函数和训练循环都没有 queue

结论：

- 可以作为后续增强
- 当前不能写成正式配置项

---

## 七、适配当前工程后的推荐配置草案

下面这份写法与当前配置结构一致，可以直接作为 Stage 2 调参基线理解：

```yaml
name: stage2_alignment_fold1
stage_index: 2
checkpoint_prefix: stage2
best_checkpoint_name: stage2_best_joint_f1.pt
split: data/splits/split_unseen_fold1_v1.yaml
tail_mode: all_valid
freeze_modules: []

loss_weights:
  clip: 1.0
  inv: 0.5
  med: 1.0
  attr: 1.0
  pol: 0.0

selection_metric: joint_score
maximize_metric: true
tie_breakers:
  - metric: contrastive_loss_sum
    mode: min
  - metric: medium_f1_interface
    mode: max
  - metric: epoch
    mode: min

model:
  visual:
    use_lora: true
    max_windows_per_encode: 4

train:
  epochs: 20
  batch_size: 1
  amp_enabled: false
  samples_per_object: 1
```

如果 `max_windows_per_encode=4` 仍然吃紧，可进一步降到：

```yaml
model:
  visual:
    max_windows_per_encode: 2
```

---

## 八、建议的实施顺序

按“先稳住，再增强”的顺序建议如下：

1. 先做当前代码已支持的优化

- Stage 2 显式设 `train.batch_size=1`
- Stage 2 显式设 `model.visual.max_windows_per_encode=4` 或 `2`
- Stage 2 显式设 `train.amp_enabled=false`

2. 确认可稳定训练后，再决定是否做结构升级

- 选窗进入视觉路径
- 属性窗口抽样
- bf16 精度模式
- gradient checkpointing
- TBPTT
- feature queue

---

## 九、结论

原文档里的总体方向大体合理，但有一部分内容把“未来可做的优化”写成了“当前工程已有的可配能力”，这会和现有代码产生冲突。修正后的原则是：

- 当前代码已经支持的能力，直接用现有配置名表达
- 当前代码还不支持的能力，明确降级为后续增强项
- 不改变 Stage 1 -> 4 的整体设计语义

按这个版本执行，Stage 2 优化方案会与当前工程、当前训练器实现，以及论文整体设计保持一致。
