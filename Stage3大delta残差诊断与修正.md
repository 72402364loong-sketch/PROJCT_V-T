# Stage3 大 delta 残差诊断与修正

## 背景

当前 Stage3 使用 reference-anchor policy：

```text
force_pred = reference_force + interface_gate * force_interface_delta
```

设计目标不是追随整条 raw measured force，而是在 Interface 窗口对 reference force 做 residual correction，同时在 Water/Air 稳定段保持低扰动。

## 原始 medium-gate 版本观察

配置：

```text
configs/stages/stage36d_b_physical_balanced_reference_anchor_medium_gate_stabilized_relaxed_delta.yaml
```

最佳 checkpoint：

```text
runs/stage36d_b_physical_balanced_reference_anchor_medium_gate_stabilized_relaxed_delta/checkpoints/best.pt
```

主要现象：

```text
control_interface_mae = 70.59
stable_leakage_mean = 0.0549
stable_leakage_p95 = 0.497
```

优点：

```text
medium gate 基本能定位 Interface；
reference anchor 有效压低 stable leakage；
Water/Air 稳定段基本不被 residual 干扰。
```

问题：

```text
policy force_pred 红线基本贴近 reference；
force_interface_delta 在 Interface 段多数接近 0；
S0093 / S0215 / S0254 这类大幅 target delta 没有被拟合。
```

初步判断：

```text
失败点不一定是 gate，也可能是 residual delta head 自身学不到大幅正/负修正。
```

## residual-only oracle gate 诊断

为排除 gate 影响，建立 oracle/hard-phase gate 诊断版：

```text
configs/stages/stage36d_b_physical_balanced_residual_oracle_gate_delta_diagnostic.yaml
```

关键设置：

```yaml
model:
  policy:
    gate_source: hard_phase
    gate_stabilizer:
      enabled: false

policy_loss:
  residual_interface_weight: 10.0
  residual_stable_zero_weight: 0.05
  residual_non_interface_weight: 0.0
  delta_normalization_scale: 100.0
  beta: 0.5
```

诊断结果：

```text
best epoch = 5
control_interface_mae = 69.99
stable_leakage_mean = 0.0
gate_false_negative_interface = 0.0
```

相对原始 medium-gate 版本：

```text
control_interface_mae 仅从 70.59 改善到 69.99；
gate 完全正确后，大 delta 样本仍然没有被拟合。
```

样本级观察：

```text
S0081 / S0084 / S0092:
target delta 本来较小，oracle-gate 版能给出十几到二十几的 residual。

S0093:
target delta 约 +200 到 +280，pred delta 只有约 +20。

S0215:
target delta 约 -400，pred delta 接近 0 到 +20，方向错误。

S0254:
target delta 有大幅负值，pred delta 仍是小正值。
```

结论：

```text
gate 不是主瓶颈；
residual delta head 当前只学到“小正 residual”；
模型没有学会大幅正/负 Interface correction。
```

## 修正方向：大 delta 强监督

新增代码支持三类机制：

```text
1. residual_large_delta_weight
   对 |delta_target| 超过阈值的 Interface 窗口增加额外 Smooth L1 loss。

2. large_delta_weight_alpha / large_delta_weight_cap
   在大 delta loss 内按 |delta_target| / scale 继续加权，避免大幅样本被小幅样本稀释。

3. residual_large_delta_sign_weight / large_delta_sign_margin
   加 sign/magnitude 辅助约束，要求预测 delta 至少朝正确方向跨过一个归一化 margin。
```

新增训练指标：

```text
delta_interface_mae
delta_interface_mse
delta_interface_bias
large_delta_threshold
large_delta_interface_count
large_delta_interface_mae
large_delta_interface_mse
large_delta_interface_bias
large_delta_interface_hit_rate_100 / 200 / 300
```

这些指标用于区分：

```text
整体 Interface MAE 变好
vs.
真正学会大幅 residual correction
```

## 新建大 delta 强监督诊断版

配置：

```text
configs/stages/stage36d_b_physical_balanced_residual_oracle_gate_large_delta_diagnostic.yaml
```

关键设置：

```yaml
model:
  policy:
    gate_source: hard_phase
    gate_stabilizer:
      enabled: false

policy_loss:
  residual_interface_weight: 4.0
  residual_large_delta_weight: 24.0
  residual_large_delta_sign_weight: 6.0
  large_delta_threshold: 100.0
  large_delta_weight_alpha: 2.0
  large_delta_weight_cap: 4.0
  large_delta_sign_margin: 0.35
  residual_stable_zero_weight: 0.0
  residual_non_interface_weight: 0.0
  delta_normalization_scale: 100.0
  beta: 0.5

selection_metric: large_delta_interface_mae
```

这版仍然是诊断版，不是最终部署版：

```text
使用 hard_phase oracle gate；
不惩罚 stable/non-interface residual；
优先验证 residual head 能不能学会大幅正/负 delta。
```

## 启动命令

```powershell
$env:PYTHONDONTWRITEBYTECODE="1"
.\.venv310\Scripts\python.exe -B scripts\train.py --project-root . --stage configs\stages\stage36d_b_physical_balanced_residual_oracle_gate_large_delta_diagnostic.yaml --init-from runs\stage2_alignment_physical_balanced\checkpoints\best.pt
```

## 结果判断标准

跑完后生成曲线：

```powershell
$env:PYTHONDONTWRITEBYTECODE="1"
.\.venv310\Scripts\python.exe -B scripts\visualize_policy_force.py --project-root . --stage configs\stages\stage36d_b_physical_balanced_residual_oracle_gate_large_delta_diagnostic.yaml --checkpoint runs\stage36d_b_physical_balanced_residual_oracle_gate_large_delta_diagnostic\checkpoints\best.pt --subset val --sample-id S0093 S0215 S0254 S0081 S0084 S0092 --limit 6 --batch-size 1 --num-workers 0
```

重点观察：

```text
S0093:
pred delta 是否能接近 +200 到 +280。

S0215:
pred delta 是否能变成明显负值，并接近 -400。

S0254:
pred delta 是否能学到大幅负修正。
```

指标目标：

```text
large_delta_interface_mae 明显下降；
large_delta_interface_bias 不再单侧偏正；
delta_interface_mae 下降；
stable_leakage 在 oracle 版可暂时不作为主约束。
```

## 后续演化

如果大 delta oracle 版成功：

```text
1. 切回 medium_prob gate；
2. 保留 large_delta loss；
3. 恢复少量 residual_stable_zero_weight；
4. 增加 gated_residual_interface_weight；
5. selection metric 使用 large_delta_interface_mae + stable_leakage tie-break。
```

如果大 delta oracle 版仍失败：

```text
1. residual head 容量不足，增加 residual hidden dim 或层数；
2. 当前冻结特征不足，考虑 unfreeze content_encoder；
3. delta target 本身可能由 reference/clean force 构造造成符号或幅值不稳定，需要单独诊断 target 曲线；
4. 引入 sample-level pooled residual context，让 residual head 能看到完整 Water/Interface/Air 形态。
```

## sign-balanced 诊断进展

新增配置：

```text
configs/stages/stage36d_b_physical_balanced_residual_oracle_gate_large_delta_sign_balanced_diagnostic.yaml
```

目的：

```text
只改变 large-delta loss 的正负分桶聚合；
继续使用 hard_phase oracle gate；
selection_metric 改为 large_delta_neg_mae；
验证“负向大 delta 失败是否只是正负样本在 loss 里被稀释”。
```

结果：

```text
best epoch = 1
large_delta_neg_mae = 189.95
large_delta_neg_wrong_sign_rate = 0.904
large_delta_neg_pred_mean = +1.10
large_delta_neg_target_mean = -188.85

epoch 5:
large_delta_neg_mae = 211.25
large_delta_neg_wrong_sign_rate = 0.723
large_delta_neg_pred_mean = +22.41
large_delta_neg_target_mean = -188.85
```

结论：

```text
sign-balanced loss 没有把负向 delta 拉回来；
问题不只是正负样本 loss 稀释；
更可能是 residual head 当前输入表征或容量不足。
```

## target/data sanity 与 context 诊断版

新增 target sanity 工具：

```text
scripts/diagnose_delta_targets.py
```

新增 physical-balanced context manifest：

```text
data/processed/interface_context/physical_balanced_fold1_manifest.csv
```

manifest 统计：

```text
rows = 275
has_context_rows = 274
stable_context_rows = 269
fallback_rows = 5
```

target sanity 输出：

```text
evals/delta_targets/stage36d_b_physical_balanced_residual_oracle_gate_large_delta_sign_balanced_context_diagnostic/all/
```

关键分布：

```text
all:
interface_target_count = 3354
large_delta_count = 1162
large_pos_count = 775
large_neg_count = 387
large_neg_sample_count = 57
large_neg_object_count = 14

train:
large_neg_count = 220

val:
large_neg_count = 94

test:
large_neg_count = 73
```

判断：

```text
负向 large-delta 不是 val-only 伪分布；
train 中也存在足够多负向 large-delta 监督；
但负向样本在 object/sample 上有明显集中，例如 OBJ017 / OBJ016 / OBJ015 / OBJ014 等。
```

新增 sample-level pooled residual context 诊断版：

```text
configs/stages/stage36d_b_physical_balanced_residual_oracle_gate_large_delta_sign_balanced_context_diagnostic.yaml
```

关键设置：

```yaml
data:
  interface_context_manifest: data/processed/interface_context/physical_balanced_fold1_manifest.csv

model:
  policy:
    use_interface_context: true
    context_scale_init: 0.05

train:
  module_learning_rate_multipliers:
    policy_head.residual_context_input_layer: 10.0
    policy_head.residual_context_scale: 10.0
```

下一步判断标准：

```text
large_delta_neg_pred_mean 是否从正数翻到负数；
large_delta_neg_wrong_sign_rate 是否明显低于 0.72 / 0.90；
S0215 / S0254 的 pred delta 是否出现稳定负向修正；
S0093 的正向大 delta 不应被明显牺牲。
```

## reference-aware residual 诊断版 A

结论更新：

```text
sample-level pooled residual context 版没有把负向 large-delta 拉回负值；
large_delta_neg_pred_mean 仍为正；
large_delta_neg_wrong_sign_rate 仍偏高；
说明问题更可能不是 gate 或 target/data sanity，而是 residual head 缺少 reference anchor 条件。
```

新增 A 版配置：

```text
configs/stages/stage36d_b_physical_balanced_residual_oracle_gate_large_delta_sign_balanced_reference_aware_diagnostic.yaml
```

诊断目的：

```text
只新增 reference_force scalar/context 注入 residual path；
继续保持 residual hidden_dim = 256；
不同时做 hidden 512，避免 reference-aware 与容量变化混在一起。
```

关键设置：

```yaml
model:
  policy:
    head_type: state_residual
    hidden_dim: 256
    base_source: reference_force
    gate_source: hard_phase
    use_interface_context: true
    use_reference_force_context: true
    reference_force_context_scale: 100.0
    context_scale_init: 0.05
    gate_stabilizer:
      enabled: false
```

启动命令：

```powershell
$env:PYTHONDONTWRITEBYTECODE="1"
.\.venv310\Scripts\python.exe -B scripts\train.py --project-root . --stage configs\stages\stage36d_b_physical_balanced_residual_oracle_gate_large_delta_sign_balanced_reference_aware_diagnostic.yaml --init-from runs\stage2_alignment_physical_balanced\checkpoints\best.pt
```

跑完后生成曲线：

```powershell
$env:PYTHONDONTWRITEBYTECODE="1"
.\.venv310\Scripts\python.exe -B scripts\visualize_policy_force.py --project-root . --stage configs\stages\stage36d_b_physical_balanced_residual_oracle_gate_large_delta_sign_balanced_reference_aware_diagnostic.yaml --checkpoint runs\stage36d_b_physical_balanced_residual_oracle_gate_large_delta_sign_balanced_reference_aware_diagnostic\checkpoints\best.pt --subset val --sample-id S0093 S0215 S0254 S0072 S0096 S0081 --limit 6 --batch-size 1 --num-workers 0
```

判定标准：

```text
large_delta_neg_pred_mean 是否翻到负数；
large_delta_neg_wrong_sign_rate 是否明显低于 0.7；
S0215 / S0254 是否出现稳定负向 pred delta；
S0093 的正向 large delta 不应被明显牺牲。
```

## reference-aware residual 诊断版 A 结果

run:
```text
runs/stage36d_b_physical_balanced_residual_oracle_gate_large_delta_sign_balanced_reference_aware_diagnostic
```

best checkpoint:
```text
epoch = 1
selection_metric = large_delta_neg_mae
selection_value = 190.98
```

关键 val 指标：
```text
delta_interface_mae = 70.37
large_delta_interface_count = 148
large_delta_interface_mae = 175.88
large_delta_wrong_sign_rate = 0.574
large_delta_pred_mean = +2.39
large_delta_target_mean = -64.33

large_delta_pos_count = 54
large_delta_pos_mae = 149.59
large_delta_pos_wrong_sign_rate = 0.000
large_delta_pos_pred_mean = +2.84
large_delta_pos_target_mean = +152.43

large_delta_neg_count = 94
large_delta_neg_mae = 190.98
large_delta_neg_wrong_sign_rate = 0.904
large_delta_neg_pred_mean = +2.13
large_delta_neg_target_mean = -188.85
```

epoch 趋势：
```text
epoch 1: val_neg_mae = 190.98, val_neg_wrong = 0.904, val_neg_pred_mean = +2.13
epoch 2: val_neg_mae = 196.38, val_neg_wrong = 0.904, val_neg_pred_mean = +7.53
epoch 3: val_neg_mae = 209.49, val_neg_wrong = 0.947, val_neg_pred_mean = +20.64
epoch 4: val_neg_mae = 218.39, val_neg_wrong = 0.904, val_neg_pred_mean = +29.54
epoch 5: val_neg_mae = 217.33, val_neg_wrong = 0.851, val_neg_pred_mean = +28.48
```

样本级 best checkpoint interface 聚合：
```text
S0093: target_delta_mean = +247.94, pred_delta_mean = +2.39, wrong_sign_rate = 0.000
S0215: target_delta_mean = -289.44, pred_delta_mean = +2.43, wrong_sign_rate = 1.000
S0254: target_delta_mean = -128.52, pred_delta_mean = +3.40, wrong_sign_rate = 0.667
S0072: target_delta_mean = -195.53, pred_delta_mean = -0.10, wrong_sign_rate = 0.615
S0096: target_delta_mean = -147.84, pred_delta_mean = +1.35, wrong_sign_rate = 0.692
S0081: target_delta_mean = +9.44, pred_delta_mean = +3.34, wrong_sign_rate = 0.273
```

与上一版 context diagnostic 对比：
```text
context best:           large_delta_neg_mae = 191.03, wrong_sign_rate = 0.904, pred_mean = +2.19
reference-aware A best: large_delta_neg_mae = 190.98, wrong_sign_rate = 0.904, pred_mean = +2.13
```

诊断结论：
```text
A 版没有通过判定标准。
reference_force scalar 已经进入 residual context，但没有让 residual 学出负向大修正。
best epoch 仍停在 epoch 1，后续训练主要把 residual 均值往正方向推：
正向 large-delta MAE 逐步下降，但负向 large-delta MAE 变差。
这说明当前问题不是单纯 hidden_dim 不够，也不是单纯缺 reference_force 输入；
更像是 residual path 在初始化、loss 尺度或目标参数化上被压在接近 0 的小 residual 区间。
```

checkpoint 参数补充：
```text
residual_context_scale = 0.0787
residual_context_input_layer total_dim = 897
reference_force column norm = 0.313
interface context columns norm = 11.53
```

下一步建议：
```text
不要直接把 A 扩成 hidden 512 作为主线。
hidden 512 可以作为容量对照，但当前证据显示主要瓶颈是 residual 学不出 100-300N 量级的 signed delta。

优先做 B0: residual delta 输出尺度/参数化诊断版。
核心改动：
1. 给 residual head 增加显式 output_scale，例如 force_interface_delta = output_scale * raw_delta，output_scale 先试 100.0；
2. 将 sign loss margin 从 0.35 提到 1.0 左右，让错误符号的大 delta 有更强梯度；
3. 暂时移除或降低普通 residual_interface_weight，避免小 delta/正 delta 主导方向；
4. 继续保留 hard_phase oracle gate + reference_force context；
5. selection_metric 仍用 large_delta_neg_mae，同时监控 large_delta_neg_pred_mean 和 S0215/S0254。

如果 B0 能让 large_delta_neg_pred_mean 翻负，再做 hidden 512；
如果 B0 仍然不翻负，就需要改为 sign-specific residual head 或对象/属性条件化 residual。
```

## B0 residual delta 输出尺度/参数化诊断版

配置：
```text
configs/stages/stage36d_b_physical_balanced_residual_b0_delta_scaled_reference_aware_diagnostic.yaml
```

相对 reference-aware A 的差异：
```yaml
policy_loss:
  residual_interface_weight: 1.0
  residual_large_delta_weight: 24.0
  residual_large_delta_sign_weight: 6.0
  large_delta_sign_margin: 1.0
  delta_normalization_scale: 100.0

model:
  policy:
    hidden_dim: 256
    use_interface_context: true
    use_reference_force_context: true
    residual_output_scale: 100.0
```

诊断目的：
```text
验证 residual head 是否只是被未缩放输出参数化压在接近 0 的区间。
如果 output_scale=100 后 large_delta_neg_pred_mean 能翻负，说明主要瓶颈是输出尺度/梯度传递；
如果仍不翻负，再考虑 sign-specific residual head 或对象/属性条件化 residual。
```

启动命令：
```powershell
$env:PYTHONDONTWRITEBYTECODE="1"
.\.venv310\Scripts\python.exe -B scripts\train.py --project-root . --stage configs\stages\stage36d_b_physical_balanced_residual_b0_delta_scaled_reference_aware_diagnostic.yaml --init-from runs\stage2_alignment_physical_balanced\checkpoints\best.pt
```

跑完后生成同一组样本曲线：
```powershell
$env:PYTHONDONTWRITEBYTECODE="1"
.\.venv310\Scripts\python.exe -B scripts\visualize_policy_force.py --project-root . --stage configs\stages\stage36d_b_physical_balanced_residual_b0_delta_scaled_reference_aware_diagnostic.yaml --checkpoint runs\stage36d_b_physical_balanced_residual_b0_delta_scaled_reference_aware_diagnostic\checkpoints\best.pt --subset val --sample-id S0093 S0215 S0254 S0072 S0096 S0081 --limit 6 --batch-size 1 --num-workers 0
```

判定标准：
```text
large_delta_neg_pred_mean < 0
large_delta_neg_wrong_sign_rate 明显低于 0.7
S0215 / S0254 出现稳定负向 pred delta
S0093 的正向 large delta 不应被明显牺牲
```

## B0 residual delta 输出尺度/参数化诊断版结果

run:
```text
runs/stage36d_b_physical_balanced_residual_b0_delta_scaled_reference_aware_diagnostic
```

best checkpoint:
```text
epoch = 2
selection_metric = large_delta_neg_mae
selection_value = 163.08
```

关键 val 指标：
```text
delta_interface_mae = 99.80
large_delta_interface_mae = 141.04
large_delta_wrong_sign_rate = 0.277
large_delta_pred_mean = -42.11
large_delta_target_mean = -64.33

large_delta_pos_count = 54
large_delta_pos_mae = 102.69
large_delta_pos_wrong_sign_rate = 0.278
large_delta_pos_pred_mean = +57.29
large_delta_pos_target_mean = +152.43

large_delta_neg_count = 94
large_delta_neg_mae = 163.08
large_delta_neg_wrong_sign_rate = 0.277
large_delta_neg_pred_mean = -99.21
large_delta_neg_target_mean = -188.85
```

epoch 趋势：
```text
epoch 1: val_neg_mae = 196.60, val_neg_wrong = 0.468, val_neg_pred_mean = -18.63
epoch 2: val_neg_mae = 163.08, val_neg_wrong = 0.277, val_neg_pred_mean = -99.21
epoch 3: val_neg_mae = 202.12, val_neg_wrong = 0.468, val_neg_pred_mean = -18.47
epoch 4: val_neg_mae = 181.88, val_neg_wrong = 0.426, val_neg_pred_mean = -36.47
epoch 5: val_neg_mae = 189.09, val_neg_wrong = 0.415, val_neg_pred_mean = -8.48
epoch 6: val_neg_mae = 205.36, val_neg_wrong = 0.532, val_neg_pred_mean = +10.97
```

样本级 best checkpoint interface 聚合：
```text
S0093: target_delta_mean = +247.94, pred_delta_mean = -13.94, raw_delta_mean = -0.139, wrong_sign_rate = 0.556
S0215: target_delta_mean = -289.44, pred_delta_mean = -76.39, raw_delta_mean = -0.764, wrong_sign_rate = 0.167
S0254: target_delta_mean = -128.52, pred_delta_mean = -38.37, raw_delta_mean = -0.384, wrong_sign_rate = 0.600
S0072: target_delta_mean = -195.53, pred_delta_mean = -250.48, raw_delta_mean = -2.505, wrong_sign_rate = 0.154
S0096: target_delta_mean = -147.84, pred_delta_mean = -135.35, raw_delta_mean = -1.353, wrong_sign_rate = 0.231
S0081: target_delta_mean = +9.44, pred_delta_mean = +51.68, raw_delta_mean = +0.517, wrong_sign_rate = 0.545
```

同一组样本 large-delta only：
```text
S0072: target = -243.51, pred = -261.82, raw = -2.618, wrong = 0.000
S0093: target = +247.94, pred = -13.94, raw = -0.139, wrong = 0.556
S0096: target = -185.57, pred = -178.09, raw = -1.781, wrong = 0.000
S0215: target = -372.24, pred = -64.61, raw = -0.646, wrong = 0.222
S0254: target = -186.39, pred = -34.69, raw = -0.347, wrong = 0.545
```

与 A 版对比：
```text
reference-aware A best: large_delta_neg_mae = 190.98, wrong_sign_rate = 0.904, pred_mean = +2.13
B0 best:                large_delta_neg_mae = 163.08, wrong_sign_rate = 0.277, pred_mean = -99.21
```

诊断结论：
```text
B0 证明 residual 输出尺度/参数化确实是关键瓶颈之一。
raw residual 现在在 -1 到 -2 量级，乘以 output_scale=100 后能形成 100N 量级修正。

但 B0 不是最终解：
1. S0215 已出现稳定负向修正，但幅度仍不足；
2. S0254 只翻成弱负向，wrong_sign_rate 仍高；
3. S0093 正向 large-delta 被明显牺牲，说明模型出现了负向偏置；
4. epoch 2 对 large_delta_neg 最优，但不是正负两侧均衡最优。
```

下一步建议：
```text
做 B1，而不是直接 hidden 512。
B1 目标是保留 output_scale=100 的有效尺度，同时抑制负向偏置、恢复正向 large-delta。

建议改动：
1. 保留 residual_output_scale = 100.0；
2. 保留 reference_force context + interface pooled context；
3. 提高正负 sign-balanced 的对称性，增加 large_delta_pos 约束或引入 pos/neg 分支均衡监控；
4. selection_metric 不再只看 large_delta_neg_mae，可以改为 large_delta_interface_mae，tie breaker 再看 neg_wrong/pos_wrong；
5. 重点判定：neg_pred_mean 保持 < 0，同时 S0093 pred_delta_mean 必须回到正数。

如果 B1 仍然在 S0093 和 S0215/S0254 之间摇摆，再考虑 sign-specific residual head。
```

## B1 positive-protected delta-scaled reference-aware 诊断版

配置：
```text
configs/stages/stage36d_b_physical_balanced_residual_b1_pos_protected_delta_scaled_reference_aware_diagnostic.yaml
```

相对 B0 的差异：
```yaml
policy_loss:
  residual_large_delta_weight: 24.0
  residual_large_delta_sign_weight: 6.0
  sign_balanced_large_delta: true
  large_delta_pos_loss_weight: 2.0
  large_delta_neg_loss_weight: 1.0
  large_delta_sign_margin: 1.0

selection_metric: large_delta_balanced_score
tie_breakers:
  - large_delta_pos_wrong_sign_rate
  - large_delta_neg_wrong_sign_rate
  - large_delta_pos_mae
  - large_delta_neg_mae

model:
  policy:
    residual_output_scale: 100.0
```

诊断目的：
```text
保留 B0 证明有效的 output_scale=100；
同时用更高的 positive large-delta 权重保护 S0093 这类正向大 delta；
best checkpoint 不再只按 large_delta_neg_mae 选，而按 pos/neg balanced score 选。
```

启动命令：
```powershell
$env:PYTHONDONTWRITEBYTECODE="1"
.\.venv310\Scripts\python.exe -B scripts\train.py --project-root . --stage configs\stages\stage36d_b_physical_balanced_residual_b1_pos_protected_delta_scaled_reference_aware_diagnostic.yaml --init-from runs\stage2_alignment_physical_balanced\checkpoints\best.pt
```

跑完后生成同一组样本曲线：
```powershell
$env:PYTHONDONTWRITEBYTECODE="1"
.\.venv310\Scripts\python.exe -B scripts\visualize_policy_force.py --project-root . --stage configs\stages\stage36d_b_physical_balanced_residual_b1_pos_protected_delta_scaled_reference_aware_diagnostic.yaml --checkpoint runs\stage36d_b_physical_balanced_residual_b1_pos_protected_delta_scaled_reference_aware_diagnostic\checkpoints\best.pt --subset val --sample-id S0093 S0215 S0254 S0072 S0096 S0081 --limit 6 --batch-size 1 --num-workers 0
```

判定标准：
```text
large_delta_neg_pred_mean < 0；
large_delta_neg_wrong_sign_rate 仍明显低于 0.7；
large_delta_pos_pred_mean > 0；
large_delta_pos_wrong_sign_rate 明显低于 B0；
S0093 pred_delta_mean 回到正数；
S0215 / S0254 仍保持负向修正。
```

## B1 positive-protected 诊断版结果

run:
```text
runs/stage36d_b_physical_balanced_residual_b1_pos_protected_delta_scaled_reference_aware_diagnostic
```

best checkpoint:
```text
epoch = 2
selection_metric = large_delta_balanced_score
selection_value = 160.27
```

关键 val 指标：
```text
large_delta_balanced_mae = 132.42
large_delta_balanced_wrong_sign_rate = 0.279
large_delta_balanced_score = 160.27

large_delta_interface_mae = 142.46
large_delta_wrong_sign_rate = 0.284
large_delta_pred_mean = -19.70
large_delta_target_mean = -64.33

large_delta_pos_count = 54
large_delta_pos_mae = 95.27
large_delta_pos_wrong_sign_rate = 0.259
large_delta_pos_pred_mean = +70.39
large_delta_pos_target_mean = +152.43

large_delta_neg_count = 94
large_delta_neg_mae = 169.56
large_delta_neg_wrong_sign_rate = 0.298
large_delta_neg_pred_mean = -71.46
large_delta_neg_target_mean = -188.85
```

epoch 趋势：
```text
epoch 1: balanced_score = 170.03, pos_pred = +88.22, neg_pred = -9.63
epoch 2: balanced_score = 160.27, pos_pred = +70.39, neg_pred = -71.46
epoch 3: balanced_score = 174.84, pos_pred = +96.32, neg_pred = -33.78
epoch 4: balanced_score = 175.73, pos_pred = +112.86, neg_pred = -12.12
epoch 5: balanced_score = 164.05, pos_pred = +83.44, neg_pred = -11.07
epoch 6: balanced_score = 168.50, pos_pred = +109.43, neg_pred = +5.97
```

样本级 best checkpoint interface 聚合：
```text
S0093: target_delta_mean = +247.94, pred_delta_mean = +2.54, raw_delta_mean = +0.025, wrong_sign_rate = 0.556
S0215: target_delta_mean = -289.44, pred_delta_mean = -49.30, raw_delta_mean = -0.493, wrong_sign_rate = 0.167
S0254: target_delta_mean = -128.52, pred_delta_mean = -14.34, raw_delta_mean = -0.143, wrong_sign_rate = 0.600
S0072: target_delta_mean = -195.53, pred_delta_mean = -205.01, raw_delta_mean = -2.050, wrong_sign_rate = 0.154
S0096: target_delta_mean = -147.84, pred_delta_mean = -107.49, raw_delta_mean = -1.075, wrong_sign_rate = 0.308
S0081: target_delta_mean = +9.44, pred_delta_mean = +58.82, raw_delta_mean = +0.588, wrong_sign_rate = 0.545
```

同一组样本 large-delta only：
```text
S0072: target = -243.51, pred = -215.07, raw = -2.151, wrong = 0.000
S0093: target = +247.94, pred = +2.54, raw = +0.025, wrong = 0.556
S0096: target = -185.57, pred = -145.16, raw = -1.452, wrong = 0.000
S0215: target = -372.24, pred = -36.92, raw = -0.369, wrong = 0.222
S0254: target = -186.39, pred = -8.97, raw = -0.090, wrong = 0.636
```

与 B0 对比：
```text
B0 aggregate:
pos_mae = 102.69, pos_wrong = 0.278, pos_pred = +57.29
neg_mae = 163.08, neg_wrong = 0.277, neg_pred = -99.21

B1 aggregate:
pos_mae = 95.27, pos_wrong = 0.259, pos_pred = +70.39
neg_mae = 169.56, neg_wrong = 0.298, neg_pred = -71.46
```

诊断结论：
```text
B1 起到了轻微正向保护作用，但没有真正解决 S0093。
S0093 从 B0 的 -13.94 拉回到 +2.54，只是贴近 0 的弱正向，不是稳定正向大修正。
与此同时，S0215 / S0254 的负向修正幅度比 B0 变弱。

这说明单纯调 pos/neg loss 权重和 selection metric 只能移动偏置，不能可靠区分正负 large-delta 样本。
下一步应转向 sign-specific / direction-conditioned residual，而不是继续调 hidden 512 或只调权重。
```

## C sign-specific residual heads + direction gate 诊断版

配置：
```text
configs/stages/stage36d_b_physical_balanced_residual_c_sign_specific_delta_scaled_reference_aware_diagnostic.yaml
```

核心结构：
```text
shared residual trunk -> pos magnitude head
                      -> neg magnitude head
                      -> direction gate

raw_delta = p_pos * softplus(pos_raw) - p_neg * softplus(neg_raw)
force_interface_delta = residual_output_scale * raw_delta
```

相对 B1 的差异：
```yaml
model:
  policy:
    head_type: state_residual_sign_specific
    residual_output_scale: 100.0

policy_loss:
  large_delta_pos_loss_weight: 1.0
  large_delta_neg_loss_weight: 1.0

selection_metric: large_delta_balanced_score
```

诊断目的：
```text
不再靠同一个 signed scalar residual 在正负方向之间折中；
显式给模型两个方向的 magnitude capacity，再由 direction gate 根据 context/reference/state 选择。
判定重点是 S0093 是否能形成明确正向大修正，同时 S0215/S0254 仍保持负向修正。
```

启动命令：
```powershell
$env:PYTHONDONTWRITEBYTECODE="1"
.\.venv310\Scripts\python.exe -B scripts\train.py --project-root . --stage configs\stages\stage36d_b_physical_balanced_residual_c_sign_specific_delta_scaled_reference_aware_diagnostic.yaml --init-from runs\stage2_alignment_physical_balanced\checkpoints\best.pt
```

跑完后生成同一组样本曲线：
```powershell
$env:PYTHONDONTWRITEBYTECODE="1"
.\.venv310\Scripts\python.exe -B scripts\visualize_policy_force.py --project-root . --stage configs\stages\stage36d_b_physical_balanced_residual_c_sign_specific_delta_scaled_reference_aware_diagnostic.yaml --checkpoint runs\stage36d_b_physical_balanced_residual_c_sign_specific_delta_scaled_reference_aware_diagnostic\checkpoints\best.pt --subset val --sample-id S0093 S0215 S0254 S0072 S0096 S0081 --limit 6 --batch-size 1 --num-workers 0
```

判定标准：
```text
large_delta_pos_pred_mean > 0，且明显高于 B1；
large_delta_neg_pred_mean < 0；
large_delta_pos_wrong_sign_rate 和 large_delta_neg_wrong_sign_rate 都低于 0.4，理想低于 0.3；
S0093 pred_delta_mean 明确为正，至少 > +50；
S0215 / S0254 pred_delta_mean 保持负向。
```

## C sign-specific 诊断版结果

run:
```text
runs/stage36d_b_physical_balanced_residual_c_sign_specific_delta_scaled_reference_aware_diagnostic
```

best checkpoint:
```text
epoch = 2
selection_metric = large_delta_balanced_score
selection_value = 164.04
```

关键 val 指标：
```text
large_delta_balanced_mae = 137.24
large_delta_balanced_wrong_sign_rate = 0.268
large_delta_balanced_score = 164.04

large_delta_interface_mae = 144.41
large_delta_wrong_sign_rate = 0.270
large_delta_pred_mean = -47.76
large_delta_target_mean = -64.33

large_delta_pos_count = 54
large_delta_pos_mae = 110.74
large_delta_pos_wrong_sign_rate = 0.259
large_delta_pos_pred_mean = +56.36
large_delta_pos_target_mean = +152.43

large_delta_neg_count = 94
large_delta_neg_mae = 163.75
large_delta_neg_wrong_sign_rate = 0.277
large_delta_neg_pred_mean = -107.57
large_delta_neg_target_mean = -188.85
```

epoch 趋势：
```text
epoch 1: balanced_score = 196.06, pos_pred = +104.95, neg_pred = +52.28
epoch 2: balanced_score = 164.04, pos_pred = +56.36,  neg_pred = -107.57
epoch 3: balanced_score = 167.78, pos_pred = +86.43,  neg_pred = -31.85
epoch 4: balanced_score = 179.51, pos_pred = +114.92, neg_pred = -15.41
epoch 5: balanced_score = 168.52, pos_pred = +82.27,  neg_pred = -9.97
epoch 6: balanced_score = 177.97, pos_pred = +109.98, neg_pred = +8.25
```

同一组样本 large-delta only：
```text
S0072: target = -243.51, pred = -297.81, raw = -2.978, p_neg = 0.876, wrong = 0.000
S0093: target = +247.94, pred = -33.35,  raw = -0.333, p_neg = 0.571, wrong = 0.556
S0096: target = -185.57, pred = -197.22, raw = -1.972, p_neg = 0.913, wrong = 0.000
S0215: target = -372.24, pred = -68.55,  raw = -0.685, p_neg = 0.703, wrong = 0.222
S0254: target = -186.39, pred = -13.38,  raw = -0.134, p_neg = 0.533, wrong = 0.545
```

direction gate 聚合：
```text
positive large-delta windows:
n = 11, target_mean = +221.60, pred_mean = -28.78
p_pos = 0.437, p_neg = 0.563, pos_mag = 0.771, neg_mag = 1.046

negative large-delta windows:
n = 37, target_mean = -262.48, pred_mean = -148.67
p_pos = 0.240, p_neg = 0.760, pos_mag = 0.529, neg_mag = 1.943
```

与 B0 / B1 对比：
```text
B0: large_delta_interface_mae = 141.04, wrong = 0.277, pos_pred = +57.29, neg_pred = -99.21
B1: large_delta_interface_mae = 142.46, wrong = 0.284, pos_pred = +70.39, neg_pred = -71.46
C:  large_delta_interface_mae = 144.41, wrong = 0.270, pos_pred = +56.36, neg_pred = -107.57
```

诊断结论：
```text
C 没有超过 B0/B1 的总体 large-delta 表现，只是把部分负样本压得更负。
direction gate 在负向 large-delta 上有分离度，但正向 large-delta 被错误路由到负方向：
S0093 的 p_neg = 0.571，pred = -33.35，仍没有形成稳定正向大修正。

S0215 的方向是对的，但幅度严重不够；S0254 则仍接近 0.5 gate，方向不稳定。
因此问题不再是“有没有正负 head 容量”，而是 direction gate 缺少显式方向监督，只靠 residual loss 间接学习会把不确定样本推向训练集中更常见/更便宜的负向路由。

下一版建议做 C1：保留 sign-specific heads，但给 residual_direction_logits 加 large-delta sign CE/BCE 监督。
监督目标用 sign(delta_force_target)，只在 interface 且 |delta_force_target| >= 100 的窗口上强监督；
可同时监控 direction_gate_acc_pos / direction_gate_acc_neg / direction_gate_margin。
```

## C1 direction-supervised sign-specific 诊断版

配置：
```text
configs/stages/stage36d_b_physical_balanced_residual_c1_direction_supervised_delta_scaled_reference_aware_diagnostic.yaml
```

相对 C 的差异：
```yaml
policy_loss:
  residual_direction_large_delta_sign_weight: 8.0
  direction_gate_pos_loss_weight: 1.0
  direction_gate_neg_loss_weight: 1.0

train:
  module_learning_rate_multipliers:
    policy_head.residual_direction_output_layer: 10.0
```

新增诊断指标：
```text
direction_gate_large_delta_acc
direction_gate_large_delta_margin
direction_gate_pos_acc
direction_gate_pos_margin
direction_gate_pos_prob_mean
direction_gate_neg_acc
direction_gate_neg_margin
direction_gate_neg_prob_mean
```

启动命令：
```powershell
$env:PYTHONDONTWRITEBYTECODE="1"
.\.venv310\Scripts\python.exe -B scripts\train.py --project-root . --stage configs\stages\stage36d_b_physical_balanced_residual_c1_direction_supervised_delta_scaled_reference_aware_diagnostic.yaml --init-from runs\stage2_alignment_physical_balanced\checkpoints\best.pt
```

跑完后生成同一组样本曲线：
```powershell
$env:PYTHONDONTWRITEBYTECODE="1"
.\.venv310\Scripts\python.exe -B scripts\visualize_policy_force.py --project-root . --stage configs\stages\stage36d_b_physical_balanced_residual_c1_direction_supervised_delta_scaled_reference_aware_diagnostic.yaml --checkpoint runs\stage36d_b_physical_balanced_residual_c1_direction_supervised_delta_scaled_reference_aware_diagnostic\checkpoints\best.pt --subset val --sample-id S0093 S0215 S0254 S0072 S0096 S0081 --limit 6 --batch-size 1 --num-workers 0
```

判定重点：
```text
direction_gate_pos_acc 明显高于 C，目标至少 > 0.7；
direction_gate_neg_acc 保持高位，目标至少 > 0.7；
direction_gate_pos_margin > 0，direction_gate_neg_margin > 0；
S0093 的 p_pos > p_neg，且 pred_delta_mean 稳定为正；
S0215 / S0254 仍保持 p_neg > p_pos，且 pred_delta_mean 为负。
```

## C1 direction-supervised 诊断版结果

run:
```text
runs/stage36d_b_physical_balanced_residual_c1_direction_supervised_delta_scaled_reference_aware_diagnostic
```

best checkpoint:
```text
epoch = 2
selection_metric = large_delta_balanced_score
selection_value = 165.66
```

关键 val 指标：
```text
large_delta_balanced_mae = 137.55
large_delta_balanced_wrong_sign_rate = 0.281
large_delta_balanced_score = 165.66

large_delta_interface_mae = 143.43
large_delta_wrong_sign_rate = 0.277
large_delta_pred_mean = -59.99

large_delta_pos_mae = 115.78
large_delta_pos_wrong_sign_rate = 0.296
large_delta_pos_pred_mean = +47.68

large_delta_neg_mae = 159.31
large_delta_neg_wrong_sign_rate = 0.266
large_delta_neg_pred_mean = -121.85

direction_gate_large_delta_acc = 0.730
direction_gate_pos_acc = 0.722
direction_gate_neg_acc = 0.734
direction_gate_pos_margin = +0.370
direction_gate_neg_margin = +0.472
```

epoch 趋势：
```text
epoch 1: score = 189.33, gate_pos_acc = 1.000, gate_neg_acc = 0.223, pos_pred = +102.17, neg_pred = +44.31
epoch 2: score = 165.66, gate_pos_acc = 0.722, gate_neg_acc = 0.734, pos_pred = +47.68,  neg_pred = -121.85
epoch 3: score = 169.89, gate_pos_acc = 0.815, gate_neg_acc = 0.564, pos_pred = +95.38,  neg_pred = -25.44
epoch 4: score = 176.53, gate_pos_acc = 0.815, gate_neg_acc = 0.553, pos_pred = +108.47, neg_pred = -17.49
epoch 5: score = 168.62, gate_pos_acc = 0.796, gate_neg_acc = 0.564, pos_pred = +82.42,  neg_pred = -15.64
epoch 6: score = 174.79, gate_pos_acc = 0.870, gate_neg_acc = 0.521, pos_pred = +108.00, neg_pred = -2.12
```

同一组样本 large-delta only：
```text
S0072: target = -243.51, pred = -321.71, raw = -3.217, p_neg = 0.926, direction_acc = 1.000, wrong = 0.000
S0093: target = +247.94, pred = -44.93,  raw = -0.449, p_pos = 0.408, direction_acc = 0.444, wrong = 0.556
S0096: target = -185.57, pred = -214.40, raw = -2.144, p_neg = 0.953, direction_acc = 1.000, wrong = 0.000
S0215: target = -372.24, pred = -87.13,  raw = -0.871, p_neg = 0.777, direction_acc = 0.778, wrong = 0.222
S0254: target = -186.39, pred = -24.52,  raw = -0.245, p_neg = 0.577, direction_acc = 0.455, wrong = 0.545
```

与 B0/B1/C 对比：
```text
B0: large_delta_mae = 141.04, wrong = 0.277, pos_pred = +57.29,  neg_pred = -99.21
B1: large_delta_mae = 142.46, wrong = 0.284, pos_pred = +70.39,  neg_pred = -71.46
C:  large_delta_mae = 144.41, wrong = 0.270, pos_pred = +56.36,  neg_pred = -107.57
C1: large_delta_mae = 143.43, wrong = 0.277, pos_pred = +47.68,  neg_pred = -121.85
```

诊断结论：
```text
C1 证明 direction gate 的显式监督是有效的：全局 large-delta gate acc 从不可观测/间接学习变成 0.73，
正负两侧 margin 都为正。但它没有改善总体 large-delta residual，selection score 比 C 更差，
主要代价是正向 large-delta 被压弱，S0093 仍然失败。

这说明当前瓶颈已经从“方向 gate 是否可学”转到“方向 head 的 magnitude 是否按目标幅度学习”。
S0215 的方向基本对，但 -372 的目标只输出 -87，是典型幅度不足；
S0254 的方向仍接近 0.5，且幅度更弱；
S0093 则同时存在方向失败和 magnitude 竞争：pos_mag 低于 neg_mag，p_pos 也低于 p_neg，最终仍为负。

下一步建议做 C2：保留 C1 的 direction CE，同时对 sign-specific magnitude heads 加显式幅度监督。
对正向 large-delta，监督 pos_magnitude * residual_output_scale -> abs(delta_target)，并压低 neg_magnitude；
对负向 large-delta，监督 neg_magnitude * residual_output_scale -> abs(delta_target)，并压低 pos_magnitude。
这样可以判断问题是否来自 magnitude branch 未被充分解耦，而不是继续只调 gate 权重。
```

## C2 direction + magnitude supervised 诊断版

配置：
```text
configs/stages/stage36d_b_physical_balanced_residual_c2_direction_magnitude_supervised_delta_scaled_reference_aware_diagnostic.yaml
```

相对 C1 的差异：
```yaml
policy_loss:
  residual_direction_large_delta_sign_weight: 8.0
  residual_sign_magnitude_weight: 12.0
  sign_magnitude_opposite_weight: 0.5
  sign_magnitude_scale: 100.0
  sign_magnitude_beta: 0.5

train:
  module_learning_rate_multipliers:
    policy_head.residual_pos_output_layer: 10.0
    policy_head.residual_neg_output_layer: 10.0
    policy_head.residual_direction_output_layer: 10.0
```

新增/重点诊断指标：
```text
direction_magnitude_pos_mae
direction_magnitude_pos_active_mean
direction_magnitude_pos_target_mean
direction_magnitude_pos_opposite_mean
direction_magnitude_neg_mae
direction_magnitude_neg_active_mean
direction_magnitude_neg_target_mean
direction_magnitude_neg_opposite_mean
```

启动命令：
```powershell
$env:PYTHONDONTWRITEBYTECODE="1"
.\.venv310\Scripts\python.exe -B scripts\train.py --project-root . --stage configs\stages\stage36d_b_physical_balanced_residual_c2_direction_magnitude_supervised_delta_scaled_reference_aware_diagnostic.yaml --init-from runs\stage2_alignment_physical_balanced\checkpoints\best.pt
```

跑完后生成同一组样本曲线：
```powershell
$env:PYTHONDONTWRITEBYTECODE="1"
.\.venv310\Scripts\python.exe -B scripts\visualize_policy_force.py --project-root . --stage configs\stages\stage36d_b_physical_balanced_residual_c2_direction_magnitude_supervised_delta_scaled_reference_aware_diagnostic.yaml --checkpoint runs\stage36d_b_physical_balanced_residual_c2_direction_magnitude_supervised_delta_scaled_reference_aware_diagnostic\checkpoints\best.pt --subset val --sample-id S0093 S0215 S0254 S0072 S0096 S0081 --limit 6 --batch-size 1 --num-workers 0
```

判定重点：
```text
direction_gate_pos_acc / direction_gate_neg_acc 不低于 C1；
direction_magnitude_pos_mae 和 direction_magnitude_neg_mae 明显下降；
active_mean 靠近 target_mean，opposite_mean 低于 active_mean；
S0093 同时满足 p_pos > p_neg、pos_mag > neg_mag、pred_delta_mean > 0；
S0215 / S0254 满足 p_neg > p_pos、neg_mag > pos_mag、pred_delta_mean < 0，且幅度高于 C1。
```

## C2 direction + magnitude supervised 诊断版结果

run:
```text
runs/stage36d_b_physical_balanced_residual_c2_direction_magnitude_supervised_delta_scaled_reference_aware_diagnostic
```

训练状态：
```text
resume 后已跑到 epoch 10。
best checkpoint 更新为 epoch 6。
selection_metric = large_delta_balanced_score
selection_value = 154.18
```

关键 val 指标：
```text
large_delta_balanced_mae = 127.51
large_delta_balanced_wrong_sign_rate = 0.267
large_delta_balanced_score = 154.18

delta_interface_mae = 90.80
large_delta_interface_mae = 134.74
large_delta_wrong_sign_rate = 0.284
large_delta_pred_mean = -12.26

large_delta_pos_mae = 100.73
large_delta_pos_wrong_sign_rate = 0.204
large_delta_pos_pred_mean = +71.99

large_delta_neg_mae = 154.29
large_delta_neg_wrong_sign_rate = 0.330
large_delta_neg_pred_mean = -60.66

direction_gate_large_delta_acc = 0.723
direction_gate_pos_acc = 0.796
direction_gate_neg_acc = 0.681
direction_gate_pos_margin = +0.503
direction_gate_neg_margin = +0.356

direction_magnitude_pos_mae = 0.670
direction_magnitude_pos_active_mean = 1.089
direction_magnitude_pos_target_mean = 1.524
direction_magnitude_pos_opposite_mean = 0.411

direction_magnitude_neg_mae = 1.084
direction_magnitude_neg_active_mean = 1.073
direction_magnitude_neg_target_mean = 1.888
direction_magnitude_neg_opposite_mean = 0.581
```

epoch 趋势摘要：
```text
epoch 2: score = 157.76, pos_pred = +54.48, neg_pred = -95.41, gate_acc = 0.723
epoch 6: score = 154.18, pos_pred = +71.99, neg_pred = -60.66, gate_acc = 0.723
epoch 7: score = 157.74, pos_pred = +76.40, neg_pred = -50.80, gate_acc = 0.682
epoch 10: score = 163.85, pos_pred = +90.46, neg_pred = -38.25, gate_acc = 0.676
```

同一组样本 large-delta only：
```text
S0072: target = -243.51, pred = -111.16, raw = -1.112, p_neg = 0.715, neg_mag = 1.543, wrong = 0.200
S0093: target = +247.94, pred = -30.26,  raw = -0.303, p_pos = 0.398, pos_mag = 0.611, neg_mag = 0.824, wrong = 0.556
S0096: target = -185.57, pred = -163.28, raw = -1.633, p_neg = 0.965, neg_mag = 1.685, wrong = 0.000
S0215: target = -372.24, pred = -133.27, raw = -1.333, p_neg = 0.951, neg_mag = 1.407, wrong = 0.000
S0254: target = -186.39, pred = -51.56,  raw = -0.516, p_neg = 0.715, neg_mag = 0.946, wrong = 0.545
```

与 B0/B1/C/C1 对比：
```text
B0: large_delta_mae = 141.04, score = 163.08, wrong = 0.277, pos_pred = +57.29, neg_pred = -99.21
B1: large_delta_mae = 142.46, score = 160.27, wrong = 0.284, pos_pred = +70.39, neg_pred = -71.46
C:  large_delta_mae = 144.41, score = 164.04, wrong = 0.270, pos_pred = +56.36, neg_pred = -107.57
C1: large_delta_mae = 143.43, score = 165.66, wrong = 0.277, pos_pred = +47.68, neg_pred = -121.85
C2: large_delta_mae = 134.74, score = 154.18, wrong = 0.284, pos_pred = +71.99, neg_pred = -60.66
```

诊断结论：
```text
C2 是目前总体 large-delta score 最好的版本，说明 magnitude supervision 有正向作用：
delta_interface_mae、large_delta_interface_mae、balanced_score 都明显优于 B0/B1/C/C1。

但 C2 的改善来自整体幅度校准和平衡，而不是彻底解决符号路由：
S0096 和 S0215 明显改善，S0215 从 C1 的 -87.13 增强到 -133.27，且 wrong = 0；
S0093 仍然没有翻正，p_pos = 0.398、pos_mag = 0.611 都弱于负向分支；
S0254 虽然 pred 从 -24.52 增强到 -51.56，但 wrong 仍为 0.545。

因此下一步不宜只继续提高 magnitude 权重。C2 已证明幅度监督有效，但 S0093/S0254 更像是样本/对象条件没有把方向与幅度分开。
下一版应考虑对象/属性条件化 direction/magnitude，或对正向稀缺样本做更强的 sign-balanced sampler/episode replay。
```
