# Model Phase M3 实施与校验报告

## 1. 结论

Model Phase M3 已完成。Medium/Policy loss 已支持 sample-first direction-macro reduction，训练与
评估会输出 W2A、A2W 和 Direction Macro 核心指标，checkpoint 选模已具备 W2A 非退化门槛与
diagnostic best 分流，可以进入 Model Phase M4。

本阶段没有启动正式训练，也没有生成 E0 基线。因此 W2A guard 基础设施已经完成，但公共 M3
训练配置保持 `enabled: false`；每个正式 stage 必须在取得相匹配的 E0 W2A 基线后填写
`baseline_value` 并启用。

正式校验报告：

```text
data/processed/stats/model_phase_m3_validation.json
```

最终结果：

```text
check_count:        5
passed_check_count: 5
error_count:        0
M3 tests:           8 passed / 0 failed
full function tests: 50 passed / 0 failed
```

---

## 2. 双向 loss reduction

正式配置：

```yaml
loss_reduction:
  mode: sample_direction_macro
  require_all_directions_per_batch: false
```

Medium 和 Policy 均执行：

1. 每个 sample 内对有效监督元素求平均；
2. 分别对有效 W2A sample 和 A2W sample 求平均；
3. 对两个方向等权宏平均。

```text
L = (L_W2A + L_A2W) / 2
```

因此某个方向拥有更多窗口或更长序列时，不会获得更大的方向权重。object balance 仍由
`direction_object_aware` sampler 负责，loss 不重复按 object 加权。

### 2.1 Medium

Medium cross entropy 先在每个 sample 的有效窗口内按既有 class weight 求均值，再做方向宏平均。
同时输出：

```text
loss_med_w2a
loss_med_a2w
loss_med
```

### 2.2 Policy

现有 per-finger sign-specific residual loss 的全部组成项继续复用原监督 mask，但现在按 sample
独立计算后再做方向宏平均。同时输出：

```text
loss_pol_w2a
loss_pol_a2w
loss_pol
```

没有有效 Policy supervision 的 sample 不进入方向 sample mean，因此 failed/无监督样本不会通过
零值稀释有效 Policy loss。

### 2.3 历史兼容

未配置 `loss_reduction` 时继续使用 `legacy_pooled`，历史单向 stage 的 loss 数值语义不变。
direction-macro 模式要求显式 `direction_ids`，缺失或非法枚举会直接报错。

---

## 3. 分方向指标

新增 `DirectionMetricAccumulator`，核心指标统一输出：

```text
<metric>_w2a
<metric>_a2w
<metric>_macro_direction
```

当前覆盖：

- Medium accuracy、macro F1、Water/Interface/Air F1 和 confusion；
- finger control interface MAE；
- finger delta interface MAE；
- finger large-delta wrong-sign rate；
- stable leakage mean/p95；
- gate false positive stable；
- gate false negative interface。

每个指标同时带有效 count；Macro 只有两方向都具有对应有效观测时才标记
`<metric>_macro_direction_complete=true`。正式 Macro 选模若缺少任一方向会 fail-fast，不能用单方向
数值冒充双向结果。

DC-M 正式 primary metric：

```text
medium_f1_interface_macro_direction  # maximize
```

DC-P/DC-J 正式 primary metric：

```text
finger_control_interface_mae_macro_direction  # minimize
```

---

## 4. W2A retention guard

新增可配置 guard：

```yaml
w2a_retention_guard:
  enabled: true
  metric: finger_control_interface_mae_w2a
  baseline_value: <E0_W2A_value>
  mode: min
  max_relative_degradation: 0.05
```

最小化指标门槛：

```text
current <= baseline * 1.05
```

最大化指标门槛：

```text
current >= baseline * 0.95
```

guard 会校验：

- metric 必须显式为 `*_w2a`；
- E0 baseline 必须提供且为有限值；
- W2A metric 必须有有效观测 count；
- mode 和 tolerance 必须合法。

每次验证写入当前值、基线、阈值、相对退化和通过状态。未通过 guard 的候选仍可写入：

```text
checkpoints/diagnostic_best.pt
checkpoints/<stage>_diagnostic_best.pt
```

但不会写入正式 `best.pt`。如果整个 stage 都没有通过门槛的 checkpoint，则该 stage 没有正式 best。

---

## 5. 配置边界

M3 新增：

```text
configs/train/direction_conditioned_v1.yaml
```

它继承冻结的 `configs/train/bidirectional_v1.yaml`，只增加模型训练层的 reduction 和 guard 配置。
Phase E 已冻结配置未被修改。

```text
source SHA-256:
1a3e37523e689558a1e176f13b95d2cb8cdcafccbfb3feb3f8d3572e983f0621

effective SHA-256:
915f0e5b63a3133f6372d955a8f4fa0ac91d9dccd8d1e29299b85ce70de1dc77
```

---

## 6. 真实双向 batch smoke

使用冻结 Train 数据和 direction-object-aware sampler 抽取真实 batch：

```text
batch size: 8
W2A:       4
A2W:       4
windows:   5 per sample
```

校验结果：

```text
forward/backward finite:                 passed
Medium adapter nonzero gradient values:  160
Policy adapter nonzero gradient values:  192
Medium direction macro complete:         true
Policy direction macro complete:         true
epoch metric key count:                  393
```

Policy 梯度 smoke 模拟正式 warm-start：保留新 direction adapter 的零初始化，同时使用非零的已训练
residual 输出层。原因是正式 M3 从 canonical W2A checkpoint 初始化，而不是从全零 Policy 输出层
冷启动。

smoke 中的随机模型指标仅用于验证链路，不代表训练效果，也不能作为 E0 baseline。

---

## 7. Frozen data release

Phase E release manifest 保持不变：

```text
d2eb241ac29c0eb1e11c2525fd8955fca9a5d1ce8afccd026f38f26a247eb280
```

```text
protected files checked: 25
mismatch:                 0
source drift:             train_entrypoint（既有预期项）
```

数据、标注、split、统计、sidecar 和 Phase A–E 报告均未改变。

---

## 8. 主要文件

```text
src/cmg/losses.py
src/cmg/direction_metrics.py
src/cmg/training.py
src/cmg/evaluation.py
configs/train/direction_conditioned_v1.yaml
tests/test_model_phase_m3.py
scripts/validate_model_phase_m3.py
data/processed/stats/model_phase_m3_validation.json
```

---

## 9. 下一阶段

Model Phase M4：配置与 smoke。

范围：

```text
stage39 common base
DC-W0 / DC-M / DC-P / DC-J stage 配置
freeze 与 optimizer group 审计
canonical checkpoint 初始化 smoke
4+4 batch save/load 与梯度/freeze smoke
正式选模 metric/tie-breaker/guard 配置契约
```

M4 可以建立 guard 配置模板，但实际 `baseline_value` 必须来自与当前 split、模型口径和评估实现
一致的 E0 结果，不能使用历史 pooled 指标代填。
