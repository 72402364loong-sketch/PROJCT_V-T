# Model Phase M2 实施与校验报告

## 1. 结论

Model Phase M2 已完成。direction embedding、Medium/Policy residual FiLM 注入、零初始化
warm-start，以及 sequence/online 一致性已经落实，可以进入 Model Phase M3。

本阶段只改造模型结构和推理接口，没有引入 direction-macro loss、方向分项选模逻辑，也没有启动
stage39 正式训练。

正式机器校验报告：

```text
data/processed/stats/model_phase_m2_validation.json
```

最终结果：

```text
check_count:        5
passed_check_count: 5
error_count:        0
function tests:     42 passed / 0 failed
M2 tests:           8 passed / 0 failed
```

---

## 2. 已实施结构

### 2.1 Direction embedding

新增共享 `DirectionEmbedding`，正式映射为：

```text
W2A -> direction_id = 0
A2W -> direction_id = 1
direction_ids: LongTensor[B]
direction_embedding: Tensor[B, 16]
```

正式 direction-conditioned 配置要求 batch 显式携带 `direction_ids`。缺失、形状错误、非整数或
越界值均 fail-fast，不会静默回退为 W2A。

### 2.2 Residual FiLM adapter

Medium 和 Policy 各自使用一个轻量 `ResidualFiLMAdapter`：

```text
y = (1 + gamma(direction)) * x + beta(direction)
```

两个 adapter 的输出层均以零初始化，因此第 0 步严格满足 `y == x`。新增方向参数能够接收梯度，
不会因为恒等初始化而永久失活。

### 2.3 Medium 注入边界

方向条件只注入：

```text
tactile evidence encoder -> Medium direction adapter -> existing Medium GRU
```

不修改 evidence 特征维数、GRU hidden 维数或 Medium 输出契约。

### 2.4 Policy 注入边界

方向条件只作用于 Policy residual hidden feature，位于既有介质 FiLM 之后、per-finger residual
输出之前。

以下路径保持不变：

- Policy reference/base 分支；
- visual encoder；
- tactile content encoder；
- attribute head。

单测确认改变方向调制不会改变 `finger_force_base`，只会影响 residual 分支。

### 2.5 Sequence/online 同构

离线 sequence forward 与 `forward_online_step` 共用同一组 direction embedding 和 adapter。
单窗口输入下，两条路径的 Medium、Policy 和 direction embedding 输出均数值一致。

在线封装现在会强制模型进入 `eval()`，并显式转发 `direction_ids`，避免 Dropout 等训练态行为混入
在线推理。

---

## 3. 正式 checkpoint warm-start 校验

初始化 checkpoint：

```text
runs/stage38j_f20_v3_causal/checkpoints/best.pt
SHA-256:
6360dd44fd1e080c60a0a8c87f497b599a7880ea908e852d1e2f87fbd84dcb84
```

加载结果：

```text
loaded old tensors:       471
direction model tensors:  480
allowed new missing keys: 9
unexpected keys:          0
```

9 个缺失键全部属于 M2 新模块：

```text
direction_embedding.*
medium_direction_adapter.*
policy_direction_adapter.*
```

零扰动结果：

```text
max_abs(new model - legacy model):        0.0
max_abs(direction flip at zero init):     0.0
max_abs(initial direction modulation):    0.0
```

因此 canonical W2A checkpoint 可以安全作为方向模型 warm-start 起点；新增模块在训练开始前不会改变
旧模型函数。

---

## 4. 配置与接口

正式模型配置：

```text
configs/model/direction_conditioned_v1.yaml
source SHA-256:
a5ba921353c1c584669cb32db433b2171c549bfbb0890169f7b46e68f4b0e355
effective SHA-256:
52a007808c16201ed56f221b12c16fa6c733e00ec8c629d49fd46c83665150b8
```

配置解析已与以下双向配置联调：

```text
dataset_version: bidirectional_v1
sampling_mode:   direction_object_aware
split:           split_unseen_fixed_test_obj004_obj007_v1
```

在线示例会从样本读取规范 direction 枚举，映射到 `direction_id` 后交给模型，不再隐式假设单向。

---

## 5. 校验覆盖

| 检查 | 结果 |
|---|---:|
| M1 前置校验 | 通过 |
| canonical checkpoint warm-start | 471/480，9 个新增键获准 |
| 零初始化旧模型等价 | 最大差值 0.0 |
| 零初始化方向翻转等价 | 最大差值 0.0 |
| direction 输入 fail-fast | 通过 |
| Policy base/residual 隔离 | 通过 |
| sequence/online parity | 通过 |
| online eval lock | 通过 |
| adapter 梯度可达 | 通过 |
| 全部零参数回归测试 | 42/42 |
| 冻结数据与报告 | 25/25，无 mismatch |

Phase E release manifest 保持不变：

```text
d2eb241ac29c0eb1e11c2525fd8955fca9a5d1ce8afccd026f38f26a247eb280
```

仅保留既有的 `train_entrypoint` 预期源码漂移；数据、标注、split、统计、sidecar 和 Phase A–E
报告均未改变。

Python 源码通过 `compileall`；Git whitespace check 无错误。

---

## 6. 主要文件

```text
src/cmg/models/modules.py
src/cmg/models/system.py
src/cmg/models/__init__.py
src/cmg/online.py
scripts/online_stub.py
configs/model/direction_conditioned_v1.yaml
tests/test_model_phase_m2.py
scripts/validate_model_phase_m2.py
data/processed/stats/model_phase_m2_validation.json
```

---

## 7. 下一阶段

Model Phase M3：双向 loss 与指标。

范围：

```text
W2A / A2W 分方向 loss reduction
Direction Macro loss / metrics
W2A、A2W、Macro 独立报告
W2A checkpoint selection guard
历史单向配置兼容与回归
```

M3 应继续复用本阶段的显式 `direction_ids`，不重新从介质标签或 residual 正负号推断方向。
