# Direction-conditioned 模型与双向训练设计说明

## 1. 文档状态

- 版本：v0.1（设计评审稿）
- 日期：2026-07-19
- 依赖数据版本：`bidirectional_v1`
- 依赖 schema：`bidirectional-causal-v4`
- 依赖 split：`split_unseen_fixed_test_obj004_obj007_v1`
- 当前状态：Model Phase M1–M4 已完成；下一步进入 M5 E0、正式训练与消融

本设计承接已经完成并冻结的双向数据管线 Phase A–E。数据层已经提供可靠的
`direction_ids`、源/目标介质、方向正确的 phase、Reference 和 Policy supervision mask；
本阶段不再修改数据定义，也不重新解释 residual 的符号。

---

## 2. 设计结论

正式模型采用一套共享的视觉、触觉、属性、Medium 和 Policy 主干，不拆成 W2A/A2W
两套独立模型。在共享主干上增加：

```text
shared direction embedding
+ Medium direction adapter
+ Policy residual direction adapter
```

方向条件只进入以下两个位置：

1. `evidence_encoder -> Medium GRU` 之间；
2. Policy residual hidden feature 中、最终 per-finger residual 输出之前。

方向条件第一版不进入：

- visual encoder；
- tactile content encoder；
- attribute head；
- Policy reference/base 路径；
- 数据归一化统计；
- Reference 构造。

该边界用于保持对象属性和视觉/内容表征的方向不变性，同时让真正具有方向差异的
介质时序和跨界面力残差能够条件化。

---

## 3. 目标与非目标

### 3.1 目标

1. 同一模型共同训练 W2A 和 A2W。
2. direction 是显式条件，不要求模型从画面或力值中猜测方向。
3. 保持现有 W2A checkpoint 的绝大部分参数和张量形状可直接继承。
4. 第 0 步保持旧模型功能：新增 adapter 为恒等映射。
5. Medium、Policy 可以分别冻结、适配和评估。
6. 按 W2A、A2W 和 Direction Macro 分别选模与报告结果。
7. 保持训练、离线推理和在线单步推理的 direction 语义一致。

### 3.2 非目标

第一版不实施：

- W2A/A2W 双模型或双 checkpoint 路由；
- direction-specific visual backbone；
- direction-specific attribute taxonomy；
- direction-specific Reference 定义；
- 默认启用 direction-conditioned normalization；
- 根据 residual 正负号反推 W2A/A2W；
- 把 Policy 中已有的 residual sign classifier 当成跨介质方向分类器。

特别需要区分：

```text
transition direction: W2A / A2W，来自 batch.direction_ids
residual direction:    per-finger residual 正/负号，由 Policy 预测
```

两者不能共用命名或监督目标。

---

## 4. 输入契约

### 4.1 枚举

```text
W2A -> direction_id = 0
A2W -> direction_id = 1
```

映射继续以 `src/cmg/constants.py` 中的 `DIRECTION_TO_INDEX` 为唯一代码准据。

### 4.2 张量形状

离线 sequence forward：

```text
direction_ids: LongTensor[B]
window_mask:   BoolTensor[B, W]
```

方向是 sample/episode 级属性，在一个 sequence 内保持不变。模型内部将 embedding 扩展为：

```text
direction_embedding: [B, D]
medium direction:    [B, W, D]
policy direction:    [B*W, D]
```

在线 `forward_online_step` 必须显式接收 `[B]` 的 `direction_ids`。正式双向模式下缺失
direction 应直接报错，不能静默回退成 W2A。仅 legacy compatibility 模式可显式配置
`default_direction: W2A`。

### 4.3 mask

- padding window 不参与 Medium、Policy loss；
- `policy_supervision_eligible=false` 的 sample 不参与 Policy/Reference 训练；
- failed sample 继续完全排除 Policy/Reference 训练；
- direction 不改变 supervision mask。

---

## 5. Direction conditioner

### 5.1 模块划分

建议在 `CrossMediumSystem` 中增加三个可独立冻结的顶层模块：

```text
direction_embedding
medium_direction_adapter
policy_direction_adapter
```

推荐初始配置：

```yaml
direction_conditioning:
  enabled: true
  num_directions: 2
  embedding_dim: 16
  zero_init: true
  require_explicit_direction: true
  medium:
    enabled: true
    mode: residual_film
    hidden_dim: 32
  policy:
    enabled: true
    mode: residual_film
    hidden_dim: 64
```

embedding 为两行可学习参数；Medium 和 Policy 使用各自的投影器，避免某一分支的冻结
策略强迫另一分支同步变化。

### 5.2 恒等初始化

adapter 输出 `delta_gamma` 和 `delta_beta`：

\[
h'=(1+\Delta\gamma(d))\odot h+\Delta\beta(d)
\]

两个 adapter 的最终线性层权重和 bias 均初始化为 0，因此：

\[
h'=h
\]

这样新模型加载 W2A checkpoint 后，在固定输入上切换 W2A/A2W direction id，初始输出
应完全一致；新模型的 W2A 输出也应与旧模型在浮点容差内一致。

不使用 concat 扩大旧层输入维度，避免破坏 `medium_head.gru.weight_ih_*` 和
`policy_head.*_input_layer` 的 checkpoint 形状。

---

## 6. Direction-conditioned Medium

### 6.1 注入位置

现有路径：

```text
tactile_low
-> evidence_encoder
-> z_med_sequence [B, W, 128]
-> Medium GRU
-> medium logits/probabilities
```

改造后：

```text
z_med_sequence
-> medium_direction_adapter(z_med_sequence, direction_embedding)
-> conditioned z_med_sequence
-> 原 Medium GRU
```

公式：

\[
z'_{med}(t)=(1+\Delta\gamma_m(d))\odot z_{med}(t)+\Delta\beta_m(d)
\]

同一 episode 的所有窗口使用相同方向条件；padding 位置仍由原 sequence length/mask
控制。GRU 输入维度和 hidden state 维度不变。

### 6.2 Medium 输出语义

Medium 仍输出物理介质：

```text
Water / Interface / Air
```

方向只帮助时序模型理解状态转移顺序：

```text
W2A: Water -> Interface -> Air
A2W: Air   -> Interface -> Water
```

禁止把类别索引改成方向相关索引，也不为 A2W 反转 Water/Air label。

### 6.3 gate

Policy interface gate 继续来自 Medium 的 Interface probability 或当前配置指定的 gate
source。第一版不增加一个独立的 direction-to-gate shortcut；方向通过 conditioned Medium
间接影响 gate。

---

## 7. Direction-conditioned Policy

### 7.1 注入边界

现有 per-finger Policy 包含：

```text
reference/base branch
residual task/state/context branch
medium-probability FiLM
per-finger sign-specific residual outputs
```

方向只注入 residual hidden，不注入 reference/base branch：

```text
task/state/context aggregation
-> existing medium FiLM
-> policy_direction_adapter
-> residual hidden layer
-> pos/neg magnitude + residual sign classifier
```

公式：

\[
h_{pol,d}=(1+\Delta\gamma_p(d))\odot h_{pol}+\Delta\beta_p(d)
\]

随后继续使用现有 per-finger sign-specific 输出：

\[
\Delta F_f
=s\left[p_f^{+}\operatorname{softplus}(a_f^{+})
-p_f^{-}\operatorname{softplus}(a_f^{-})\right]
\]

最终控制力仍为：

\[
F_{pred,f}=F_{ref,f}+g_{if}\Delta F_f
\]

### 7.2 为什么不调整 Policy 输入维度

直接把 direction embedding concat 到 task/state/context 会改变已有线性层 shape，导致：

- W2A checkpoint 不能严格加载；
- 旧层必须随机重建；
- adapter-only warm-up 无法只训练新增参数；
- W2A 第 0 步不再保持原功能。

因此第一版必须使用保持维度的 residual FiLM。

### 7.3 Reference 与 residual

- W2A Reference 来自水中界面前稳定区间；
- A2W Reference 来自空气中界面前固定区间；
- A2W 不使用 `t_contact_all/t_grasp_stable`；
- residual 始终定义为目标控制力相对当前方向 Reference 的差值；
- 不假设 W2A/A2W residual 必然互为相反数。

---

## 8. Checkpoint 初始化契约

### 8.1 正式起点

正式 warm-start checkpoint：

```text
runs/stage38j_f20_v3_causal/checkpoints/best.pt
```

只读权重核验表明，该 checkpoint 的 visual/content/evidence/Medium/attribute 参数与
`stage38j_f20_v3_medium_recalibration/checkpoints/best.pt` 完全一致，而 Policy 参数已经
进一步更新，因此它是当前最完整的单一 W2A 起点。

### 8.2 加载规则

使用 `--init-from`，不得使用 `--resume`。允许的 missing key 只能来自：

```text
direction_embedding.*
medium_direction_adapter.*
policy_direction_adapter.*
```

必须满足：

- unexpected key = 0；
- shape mismatch = 0；
- 非 direction missing key = 0；
- 保存完整 initialization report；
- 报告每个顶层模块加载参数数和未加载参数数。

禁止用无约束 `strict=False` 吞掉结构错误。

### 8.3 新 split 下的 W2A 零点

旧 checkpoint 的历史指标来自旧 split，不能直接作为本轮负迁移基线。训练前必须在当前
18 对象 split 上执行：

```text
W2A-only evaluation
A2W zero-shot evaluation
W2A/A2W pooled evaluation without direction effect
```

结果保存为 `E0_w2a_warmstart_on_bidirectional_v1`，后续所有 W2A 保持率均相对这个零点计算。

---

## 9. 冻结与 warm-up 方案

当前训练器在 optimizer 创建前一次性冻结参数，不支持同一 run 中途解冻。因此正式训练
拆成四个 stage，每个 stage 从上一个 best checkpoint 使用 `--init-from`。

### 9.1 DC-W0：adapter warm-up

训练：

```text
direction_embedding
medium_direction_adapter
policy_direction_adapter
```

冻结所有原有模块。

建议：

```yaml
epochs: 3
base_learning_rate: 0.0001
warmup_ratio: 0.10
early_stopping_patience: 0
loss_weights:
  clip: 0.0
  inv: 0.0
  med: 1.0
  attr: 0.0
  pol: 1.0
```

当前 sampler 每 epoch 38 batch，约 114 optimizer steps，warmup 约 11 steps。

本阶段目标不是取得最终指标，而是让两个零初始化 adapter 离开恒等点，同时确认两方向
梯度、mask 和输出均正常。

### 9.2 DC-M：Medium 双向适配

训练：

```text
direction_embedding
medium_direction_adapter
evidence_encoder
medium_head
```

冻结：

```text
visual_encoder
content_encoder
attribute_head
physical_attribute_head（若启用）
policy_head
policy_direction_adapter
```

建议：

```yaml
epochs: 8-12
base_learning_rate: 0.00001
warmup_ratio: 0.05
early_stopping_patience: 4
module_learning_rate_multipliers:
  medium_direction_adapter: 5.0
  direction_embedding: 2.0
```

### 9.3 DC-P：Policy 双向适配

训练：

```text
policy_direction_adapter
Policy residual task/state/context/hidden layers
per-finger pos/neg magnitude layers
per-finger residual sign layer
```

冻结：

```text
visual_encoder
content_encoder
evidence_encoder
medium_head
medium_direction_adapter
direction_embedding
attribute_head
Policy reference/base branch
```

固定 shared direction embedding，避免只训练 Policy 时改变已经校准的 Medium 条件。

建议：

```yaml
epochs: 12
base_learning_rate: 0.00002
warmup_ratio: 0.05
early_stopping_patience: 4
module_learning_rate_multipliers:
  policy_direction_adapter: 5.0
```

### 9.4 DC-J：联合低学习率微调

训练：

```text
direction_embedding
medium_direction_adapter
policy_direction_adapter
evidence_encoder
medium_head
Policy residual branch
```

继续冻结：

```text
visual backbone
visual LoRA（首轮）
content_encoder（首轮）
attribute_head
Policy reference/base branch
```

建议：

```yaml
epochs: 6-8
base_learning_rate: 0.000005
lora_learning_rate: 0.000001
warmup_ratio: 0.05
early_stopping_patience: 3
```

只有 DC-J 明显欠拟合且 W2A/A2W 均受益时，才追加 DC-J-Lora：开放 LoRA，不开放完整
ViT backbone。content encoder 也只作为独立消融解冻，不能与 LoRA 同时首次开放。

---

## 10. 双向 loss reduction

### 10.1 方向宏平均

虽然 sampler 保证每个 batch 为 4 个 W2A + 4 个 A2W，但两方向 sequence 长度和有效窗口
数不同。现有把所有窗口直接 flatten 后求平均的方式可能使窗口更多的一侧主导梯度。

正式双向损失采用：

1. 在每个 sample 内对有效 window/finger 求平均；
2. 分别求 W2A sample mean 和 A2W sample mean；
3. 两方向等权宏平均。

\[
L_{bidir}=\frac{L_{W2A}+L_{A2W}}{2}
\]

object balance 由现有 direction-object-aware sampler 负责，不在 loss 中重复加权。

### 10.2 Medium loss

```text
L_medium = macro_direction_cross_entropy(Water, Interface, Air)
```

Interface class weight继续由 stage 配置控制。首轮不设置方向专属 class weight，先使用统一
权重，避免把方向数据量差异和真实难度差异混在一起。

### 10.3 Policy loss

继续沿用现有 per-finger sign-specific residual loss，包括：

- residual interface Smooth L1；
- stable residual zero；
- non-interface residual；
- large-delta sign/magnitude；
- stable leakage；
- per-finger control target。

所有项改为 direction-macro reduction。两方向初始 loss 权重均为 1.0；只有验证结果显示某方向
系统性欠拟合时，才允许引入方向 loss weight，并必须作为消融报告。

### 10.4 不增加 direction classification loss

direction 是已知控制条件，不是待预测标签。主模型不增加“预测 W2A/A2W”的辅助 loss，避免
强化画面背景、Reference 数值范围或 episode 长度等捷径。

### 10.5 W2A retention

第一版依靠冻结阶段和选模门槛保持 W2A，不默认使用 teacher distillation。若 DC-J 后 W2A
主指标恶化超过 5%，可追加小权重的 W2A retention loss 作为单独实验，不直接写入主线。

---

## 11. 采样方案

主线继续使用已经冻结验证的：

```yaml
sampling_mode: direction_object_aware
direction_balance_mode: paired_cycle
batch_size: 8
```

每个 batch：

```text
4 W2A + 4 A2W
同一 object 的两个方向成对进入 batch
```

当前 Train 每 epoch：

```text
38 batches
304 draws
152 W2A
152 A2W
11 repeated draws
repeat rate = 3.62%
```

主线不再叠加 interface oversampling。若后续需要 interface focus，应在 paired direction 内部
等规则采样，不能破坏每 batch 的方向平衡。

---

## 12. 指标与选模

### 12.1 所有核心指标必须三套输出

```text
metric_w2a
metric_a2w
metric_macro_direction
```

其中：

\[
M_{macro-dir}=\frac{M_{W2A}+M_{A2W}}{2}
\]

不能只报告 pooled 指标。

### 12.2 Medium 指标

- accuracy / macro F1；
- Water、Interface、Air F1；
- Interface onset/offset timing error；
- gate false positive stable；
- gate false negative interface；
- phase confusion matrix；
- 每个 unseen object 的分方向结果。

DC-M primary metric：

```text
medium_f1_interface_macro_direction（maximize）
```

### 12.3 Policy 指标

- finger control/interface MAE；
- finger delta/interface MAE；
- large-delta balanced MAE；
- large-delta wrong-sign rate；
- stable leakage mean/p95；
- Reference-only improvement；
- 每指和每对象指标。

DC-P/DC-J primary metric：

```text
finger_control_interface_mae_macro_direction（minimize）
```

建议 tie-breaker：

1. `finger_delta_interface_mae_macro_direction`；
2. `finger_large_delta_wrong_sign_rate_macro_direction`；
3. `stable_leakage_mean_macro_direction`；
4. W2A 主指标；
5. 较早 epoch。

### 12.4 W2A 非退化门槛

正式 best checkpoint 必须同时满足：

```text
W2A primary metric <= E0 W2A baseline * 1.05
```

即 W2A 主指标相对当前 split 的重新评测零点恶化不超过 5%。若没有 checkpoint 通过门槛，
该 stage 判定为没有正式 best，只保留 diagnostic checkpoint。

---

## 13. 必做消融

| 编号 | 模型 | 目的 |
|---|---|---|
| E0 | 原 W2A checkpoint，无 direction effect | 新 split 零点和 A2W zero-shot |
| E1 | 双向共同训练，不输入 direction | 判断数据本身能否支持共享模型 |
| E2 | 仅 Medium direction-conditioned | 测量时序方向条件收益 |
| E3 | 仅 Policy direction-conditioned | 测量力控制方向条件收益 |
| E4 | Medium + Policy direction-conditioned | 正式主模型 |
| E5 | E4 + direction-conditioned normalization | 检查统一归一化是否限制性能 |

补充诊断：

- direction-shuffled evaluation；
- direction embedding 置零；
- Reference-only baseline；
- 按 object、方向和 residual sign 分层；
- 检查 Reference 数值能否单独预测方向。

E5 不是默认主线。当前方向 Reference 均值差约 0.56 个 pooled standard deviation，方向专属
归一化可能提升性能，也可能增强 shortcut，必须通过消融决定。

---

## 14. 冻结和 checkpoint 基础设施要求

正式训练前必须完成：

1. checkpoint loader 允许且只允许 direction 模块 missing key；
2. 对 shape mismatch 给出完整错误；
3. `freeze_modules` 遇到不存在的路径时 fail-fast；
4. 支持 frozen module 的 eval-mode lock；
5. 记录 trainable/frozen 参数名、数量和比例；
6. 每个 optimizer group 记录实际 LR；
7. stage checkpoint 中记录 `initialized_from` 及其 SHA-256；
8. 保存 direction conditioner 配置和映射；
9. 禁止跨 stage 使用 `--resume`。

冻结模块中的 dropout 不应因 `model.train()` 被重新启用。对于冻结模块，应在每个 epoch/step
切换训练模式后重新锁定为 eval；解冻 stage 再解除该锁定。

---

## 15. 配置规划

建议新增：

```text
configs/model/direction_conditioned_v1.yaml
configs/stages/stage39_bidirectional_base.yaml
configs/stages/stage39a_direction_adapter_warmup.yaml
configs/stages/stage39b_bidirectional_medium.yaml
configs/stages/stage39c_bidirectional_policy.yaml
configs/stages/stage39d_bidirectional_joint.yaml
```

`stage39_bidirectional_base.yaml` 作为四个训练 stage 的共同父配置，并显式声明：

```yaml
data_config: configs/data/policy_20hz_bidirectional_v4.yaml
model_config: configs/model/direction_conditioned_v1.yaml
train_config: configs/train/bidirectional_v1.yaml
split: data/splits/split_unseen_fixed_test_obj004_obj007_v1.yaml
```

当前 `scripts/train.py` 固定加载 `configs/data/default.yaml`、`configs/model/default.yaml` 和
`configs/train/base.yaml`，不会自动消费上述独立配置。Model Phase M1 必须先支持这三个
repo-relative config path：指定文件作为 base，随后再用 stage 内的 `data/model/train` block
做局部 override。若路径缺失或与 stage 内 split 不一致，应直接报错。

不得复制或私自修改正式 split。实验配置如需改变 loss/normalization，只新增 stage/model
版本，不覆盖 `bidirectional_v1` 数据 release。

---

## 16. 代码改造范围

### 16.1 模型

`src/cmg/models/modules.py`

- 新增 `DirectionEmbedding` 或直接使用受控 `nn.Embedding`；
- 新增可复用的 `ResidualFiLMAdapter`；
- 保证最终投影零初始化。

`src/cmg/models/system.py`

- 构建三个顶层 direction 模块；
- sequence forward 消费 `batch['direction_ids']`；
- Medium 注入；
- Policy residual 注入；
- online step 显式消费 direction；
- 输出 direction embedding/adapter diagnostics（可配置）。

### 16.2 loss 和指标

`src/cmg/losses.py`

- 支持 sample-first、direction-macro reduction；
- 保持现有 supervision mask；
- 增加按方向 loss diagnostics。

`src/cmg/training.py`

- 增加按方向指标 accumulator；
- 增加 W2A best guard；
- 强化冻结和 checkpoint 审计；
- 保存 initialization provenance。

### 16.3 训练入口

`scripts/train.py`

- 支持 stage 选择 `data_config/model_config/train_config` base path；
- 将最终解析后的配置和三个源文件 SHA-256 写入 run config；
- 保持 `--init-from`/`--resume` 互斥；
- 输出 checkpoint compatibility report；
- 校验双向 stage 必须启用 direction-aware sampler；
- 校验正式模式必须显式 direction。

---

## 17. 测试与验收

### 17.1 单元测试

必须新增：

1. direction id 映射和非法 id；
2. direction adapter shape；
3. 零初始化恒等输出；
4. W2A checkpoint 只缺 direction keys；
5. shape mismatch 必须失败；
6. 无效 freeze path 必须失败；
7. frozen module 参数、梯度和 train/eval 状态；
8. sequence/online direction 一致性；
9. direction-macro loss 不受窗口长度比例影响；
10. failed sample 对 Policy loss 贡献为 0；
11. W2A/A2W 指标分桶正确；
12. W2A best guard 生效。

### 17.2 模型 smoke test

在一个 4 W2A + 4 A2W batch 上检查：

- forward/backward 无 NaN/Inf；
- direction adapter 有梯度；
- 冻结模块无梯度；
- 两方向均有有效 Medium 和 Policy supervision；
- zero-init 时新旧 W2A 输出一致；
- 保存/加载 checkpoint 后输出一致。

### 17.3 Stage 进入条件

```text
DC-W0 -> DC-M:
  adapter 梯度、checkpoint、freeze audit 全通过

DC-M -> DC-P:
  A2W Interface F1 相对 E0 改善
  W2A Medium 指标通过 5% guard

DC-P -> DC-J:
  A2W Policy 优于 Reference-only
  W2A Policy 指标通过 5% guard

DC-J -> 正式消融:
  Direction Macro 改善
  两方向均无明显 stable leakage/错误符号退化
```

---

## 18. 实施顺序

### Model Phase M1：基础设施

- checkpoint direction-key 白名单；
- freeze fail-fast 和 eval lock；
- initialization provenance；
- 新 split W2A/A2W E0 评测入口。

实施状态（2026-07-19）：已完成，详见 `Model Phase M1实施与校验报告.md`。

### Model Phase M2：方向模块

- direction embedding；
- residual FiLM adapter；
- Medium/Policy 注入；
- sequence/online parity。

实施状态（2026-07-19）：已完成，详见 `Model Phase M2实施与校验报告.md`。正式 W2A
checkpoint 已通过零扰动 warm-start 校验；本阶段未引入 M3 loss、选模指标或正式训练。

### Model Phase M3：双向 loss 与指标

- direction-macro reduction；
- W2A/A2W/Macro 指标；
- W2A checkpoint selection guard。

实施状态（2026-07-19）：已完成，详见 `Model Phase M3实施与校验报告.md`。M3 已提供
sample-first direction-macro loss、核心分方向指标和可配置 W2A retention guard；正式 stage
仍需在取得对应 E0 基线后填写 `baseline_value` 并启用 guard。

### Model Phase M4：配置与 smoke

- stage39a–39d 配置；
- checkpoint 初始化 smoke；
- 4+4 batch 梯度/freeze smoke；
- 全测试回归。

实施状态（2026-07-19）：已完成，详见 `Model Phase M4实施与校验报告.md`。stage39a–39d
配置、canonical checkpoint 初始化、冻结/optimizer 审计和真实 4+4 串行 stage smoke 均已通过。
由于当前 split 的 E0 尚未测量，正式训练入口保持阻塞，不能在 W2A guard 缺少基线时误启动。

### Model Phase M5：正式训练与消融

- E0；
- DC-W0；
- DC-M；
- DC-P；
- DC-J；
- E1–E5 消融。

---

## 19. 最终原则

> direction 是已知任务条件，不是介质标签、residual 符号或需要模型猜测的隐变量。正式模型
> 应在共享主干上用恒等初始化的轻量条件适配器表达 W2A/A2W 差异，以分阶段冻结和低学习率
> 联合微调保护原 W2A 能力，并用 W2A、A2W、Direction Macro 三套指标证明双向收益。
