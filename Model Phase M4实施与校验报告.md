# Model Phase M4 实施与校验报告

## 1. 结论

Model Phase M4 已完成。stage39 公共配置和 DC-W0、DC-M、DC-P、DC-J 四阶段配置已建立，
canonical W2A checkpoint 初始化、正式模型冻结/optimizer 审计、真实 4+4 双向 batch 串行梯度
smoke，以及 checkpoint 保存/重载一致性均通过。

本阶段没有启动正式多 epoch 训练。由于当前 18 对象 split 下的 E0 W2A 基线尚未测量，四个
stage 明确标记为：

```text
blocked_pending_e0_w2a_baseline
```

训练入口会拒绝正式启动。完成 E0、填入对应 guard `baseline_value` 并把 readiness 改为 `ready`
之后，才允许进入 DC-W0。

正式校验报告：

```text
data/processed/stats/model_phase_m4_validation.json
```

最终结果：

```text
check_count:        5
passed_check_count: 5
error_count:        0
M4 tests:           8 passed / 0 failed
full function tests: 58 passed / 0 failed
```

---

## 2. stage39 配置体系

### 2.1 公共基线

```text
configs/stages/stage39_bidirectional_base.yaml
```

统一声明：

```text
data_config:  policy_20hz_bidirectional_v4.yaml
model_config: direction_conditioned_v1.yaml
train_config: direction_conditioned_v1.yaml
split:        split_unseen_fixed_test_obj004_obj007_v1.yaml
batch:        8（4 W2A + 4 A2W）
loss:         sample_direction_macro
```

Policy loss 继承 canonical stage38j 的 per-finger sign-specific residual 口径，包括 large-delta、
sign gate、magnitude、stable zero 和 non-interface 约束。

### 2.2 四阶段

| Stage | 训练范围 | Epoch | Base LR | Primary metric |
|---|---|---:|---:|---|
| DC-W0 / stage39a | 三个 direction 模块 | 3 | 1e-4 | finger control interface MAE macro |
| DC-M / stage39b | direction embedding、Medium adapter、evidence、Medium head | 10 | 1e-5 | Interface F1 macro |
| DC-P / stage39c | Policy adapter 和 Policy residual branch | 12 | 2e-5 | finger control interface MAE macro |
| DC-J / stage39d | Medium + Policy direction 联合低 LR | 8 | 5e-6 | finger control interface MAE macro |

四个配置文件：

```text
configs/stages/stage39a_direction_adapter_warmup.yaml
configs/stages/stage39b_bidirectional_medium.yaml
configs/stages/stage39c_bidirectional_policy.yaml
configs/stages/stage39d_bidirectional_joint.yaml
```

---

## 3. 初始化链

正式链路：

```text
stage38j_f20_v3_causal/best.pt
  -> stage39a DC-W0 best
  -> stage39b DC-M best
  -> stage39c DC-P best
  -> stage39d DC-J best
```

规则：

- stage39a 只允许 9 个 direction 新键缺失；
- stage39b–39d 必须 100% 严格加载，不允许 missing/unexpected key；
- 每个后续 stage 校验 checkpoint 内的 `stage_name`；
- 跨 stage 必须使用 `--init-from`，不能使用 `--resume`。

canonical checkpoint 校验：

```text
path: runs/stage38j_f20_v3_causal/checkpoints/best.pt
SHA-256:
6360dd44fd1e080c60a0a8c87f497b599a7880ea908e852d1e2f87fbd84dcb84

loaded tensors:       471
direction model:      480
allowed missing keys: 9
unexpected keys:      0
```

---

## 4. 正式模型冻结与 optimizer 审计

| Stage | Trainable tensors | Trainable values | Trainable fraction |
|---|---:|---:|---:|
| stage39a | 9 | 43,392 | 0.0489% |
| stage39b | 16 | 162,627 | 0.1833% |
| stage39c | 23 | 472,645 | 0.5327% |
| stage39d | 39 | 635,272 | 0.7159% |

所有 stage：

```text
unexpected trainable parameters: 0
missing expected prefixes:       0
invalid freeze paths:            0
```

关键 LR 分组：

```text
DC-M:
  base                         1e-5
  direction_embedding         2e-5
  medium_direction_adapter    5e-5

DC-P:
  base                         2e-5
  policy_direction_adapter    1e-4
  pos/neg magnitude outputs    1.6e-4
  residual sign output         2e-4

DC-J:
  base                         5e-6
  direction_embedding         1e-5
  Medium/Policy adapters       2.5e-5
```

Visual backbone、visual LoRA、content encoder 和 attribute head 在四阶段均保持冻结。DC-P/DC-J
还冻结 finger base/reference 分支，只开放 residual 分支。

---

## 5. 训练入口 readiness gate

新增 direction-conditioned stage preflight，正式启动前校验：

- direction 必须显式；
- sampler 必须为 `direction_object_aware`；
- direction 枚举必须为 `[W2A, A2W]`；
- batch size 必须为正偶数；
- loss 必须为 `sample_direction_macro`；
- selection metric 必须为 `*_macro_direction`；
- 必须提供 `--init-from`；
- stage readiness 必须为 `ready`；
- readiness 为 ready 时，W2A guard 必须启用并提供 E0 baseline。

M4 校验器可以显式使用 `allow_pending_e0=True` 做无训练 smoke；正常训练入口没有该绕过路径。

---

## 6. 真实 4+4 串行 smoke

使用冻结 Train 数据与真实 sampler：

```text
batch size: 8
W2A:       4
A2W:       4
windows:   5 per sample
```

串行模拟结果：

| Stage | Init missing | Frozen grad tensors | Reload max diff |
|---|---:|---:|---:|
| stage39a | 9（获准 direction keys） | 0 | 0.0 |
| stage39b | 0 | 0 | 0.0 |
| stage39c | 0 | 0 | 0.0 |
| stage39d | 0 | 0 | 0.0 |

非零梯度检查：

```text
stage39a: Medium adapter 180，Policy adapter 216
stage39b: direction embedding 12，Medium adapter 236，evidence 690，Medium head 693
stage39c: Policy adapter 272，Policy residual 1285
stage39d: direction embedding 12，两个 adapter 均非零，Medium/Policy 主干均非零
```

stage39a 第一 micro-step 的 direction embedding 梯度为 0 是零初始化 residual adapter 的预期行为：
第一步先更新 adapter 输出层；adapter 离开恒等点后，stage39b smoke 已验证 direction embedding 获得非零
梯度。

最终 stage39d checkpoint 重新构建模型并严格加载后：

```text
missing keys:             0
unexpected keys:          0
max absolute output diff: 0.0
```

smoke 使用真实数据和正式 stage loss/freeze/LR 口径，但使用轻量同构模型执行单步验证；其 loss/metric
数值不代表正式训练效果。

---

## 7. Frozen data release

Phase E release manifest 保持不变：

```text
d2eb241ac29c0eb1e11c2525fd8955fca9a5d1ce8afccd026f38f26a247eb280
```

```text
protected files checked: 25
mismatch:                 0
```

数据、标注、split、统计、sidecar 和 Phase A–E 报告均未改变。

---

## 8. 主要文件

```text
configs/stages/stage39_bidirectional_base.yaml
configs/stages/stage39a_direction_adapter_warmup.yaml
configs/stages/stage39b_bidirectional_medium.yaml
configs/stages/stage39c_bidirectional_policy.yaml
configs/stages/stage39d_bidirectional_joint.yaml
src/cmg/config.py
scripts/train.py
tests/test_model_phase_m4.py
scripts/validate_model_phase_m4.py
data/processed/stats/model_phase_m4_validation.json
```

---

## 9. 下一阶段

Model Phase M5 的第一步不是直接训练 DC-W0，而是建立 E0：

```text
1. 在当前 18 对象 split 上评估 canonical W2A checkpoint；
2. 分别保存 W2A-only、A2W zero-shot 和 pooled/direction-macro 结果；
3. 将对应 W2A baseline 写入四个 stage guard；
4. 把 training_readiness 改为 ready；
5. 再按 DC-W0 -> DC-M -> DC-P -> DC-J 顺序训练。
```

历史旧 split 或 pooled 指标不能作为当前 guard baseline。
