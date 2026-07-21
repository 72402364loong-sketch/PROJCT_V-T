# Model Phase M1 实施与校验报告

## 1. 结论

Model Phase M1 已完成，checkpoint、冻结和配置加载基础设施可以进入 Model Phase M2。

本阶段没有修改 direction-conditioned 模型结构，也没有启动训练。交付内容用于保证后续
新增 direction 模块时能够安全 warm-start、严格冻结和完整追踪配置来源。

正式机器校验报告：

```text
data/processed/stats/model_phase_m1_validation.json
```

最终结果：

```text
check_count:        7
passed_check_count: 7
error_count:        0
existing tests:     34 passed / 0 failed
```

---

## 2. 已实施内容

### 2.1 Checkpoint 初始化审计

`load_model_weights` 现在会在写入模型前完成：

- checkpoint SHA-256；
- missing/unexpected key；
- tensor shape mismatch；
- 显式 allowed missing prefix；
- LoRA 历史兼容映射；
- 按顶层模块统计加载 tensor/value 数；
- disallowed mismatch fail-fast。

未来 direction 模型只允许以下新增前缀缺失：

```text
direction_embedding.*
medium_direction_adapter.*
policy_direction_adapter.*
```

shape mismatch 一律报错，不能通过 `strict=False` 静默吞掉。

### 2.2 正式 W2A checkpoint 核验

核验对象：

```text
runs/stage38j_f20_v3_causal/checkpoints/best.pt
```

结果：

```text
SHA-256:
6360dd44fd1e080c60a0a8c87f497b599a7880ea908e852d1e2f87fbd84dcb84

loaded tensors:     471
target tensors:     471
missing keys:       0
unexpected keys:    0
shape mismatches:   0
```

因此该 checkpoint 可以继续作为 direction-conditioned 模型的正式 warm-start 起点。

### 2.3 冻结 fail-fast

`freeze_modules` 现在：

- 先解析全部模块路径，再执行冻结；
- 任一路径不存在时整体失败，不发生部分冻结；
- 去重冻结模块名；
- 输出被冻结参数名、tensor 数和 value 数；
- Trainer 强制使用 strict freeze。

历史上模块名拼写错误后静默继续训练的风险已经消除。

### 2.4 Frozen eval lock

冻结模块除 `requires_grad=False` 外，还会锁定为 eval mode。每次训练循环调用
`model.train()` 后，训练器都会重新执行 frozen eval lock，防止被冻结模块中的 dropout
或运行时 buffer 意外变化。

联合微调 stage 不需要解锁旧 run；它会新建模型并使用新的 `freeze_modules` 清单。

### 2.5 参数和 optimizer 审计

每个 run 新增：

```text
runs/<stage>/parameter_audit.json
```

其中记录：

- trainable/frozen 参数完整名称；
- trainable/frozen tensor/value 数；
- trainable fraction；
- freeze module 解析结果；
- optimizer group 名称；
- 每组实际学习率；
- 每组参数完整名称和数量。

`run_config.yaml` 保存审计文件路径、SHA-256 和摘要；checkpoint 也保存相同审计记录。

### 2.6 外部基础配置加载

stage 现在可以显式选择：

```yaml
data_config: configs/data/policy_20hz_bidirectional_v4.yaml
model_config: configs/model/direction_conditioned_v1.yaml
train_config: configs/train/bidirectional_v1.yaml
```

解析顺序：

1. 加载指定 data/model/train base；
2. 递归解析各自 `extends`；
3. 应用 stage 的 `data/model/train` 局部 override；
4. 写入 `run_name`；
5. 校验 stage split 与 data `split_path` 一致。

训练和评估入口现在共享该解析逻辑。

### 2.7 配置 provenance

`run_config.yaml` 和 checkpoint 记录：

- stage/data/model/train 源文件绝对路径和项目相对路径；
- 每个源文件 SHA-256；
- `extends` 依赖链及各文件 SHA-256；
- 依赖链 effective SHA-256；
- initialization checkpoint 路径与 SHA-256；
- checkpoint 加载审计结果。

### 2.8 E0 评估准备

评估接口新增 `use_archived_run_config`：

- `true`：保持历史同 stage 复现语义；
- `false`：使用当前 stage/data/split 评估旧 checkpoint。

这为下一阶段在当前18对象双向 split 上建立 W2A warm-start E0 零点提供入口。

---

## 3. 校验结果

| 检查 | 结果 |
|---|---:|
| direction missing-prefix 白名单 | 通过 |
| 未批准 missing key 拒绝 | 通过 |
| shape mismatch 拒绝 | 通过 |
| canonical W2A checkpoint 完整加载 | 471/471 |
| freeze 非法路径拒绝 | 通过 |
| frozen eval lock | 通过 |
| 历史 stage 配置解析 | 91/91 |
| 双向外部 data/train 配置解析 | 通过 |
| split 不一致拒绝 | 通过 |
| run/checkpoint provenance 持久化 | 通过 |
| 原有回归测试 | 34/34 |
| 冻结数据和报告文件 | 25/25 |

Python 源码已通过 `compileall`，Git whitespace check 无错误。

---

## 4. Frozen data release 边界

Phase E release manifest 保持原样，没有重新冻结或修改 checksum：

```text
d2eb241ac29c0eb1e11c2525fd8955fca9a5d1ce8afccd026f38f26a247eb280
```

校验结果：

```text
artifacts + validation reports: 25 checked, 0 mismatch
source_code drift:              train_entrypoint
```

`train_entrypoint` 漂移来自本次 M1 对 `scripts/train.py` 的预期升级。数据、标注、split、统计、
sidecar 索引和 Phase A–E 报告均未改变。旧 manifest 不应为后续模型代码开发反复改写。

---

## 5. 主要文件

```text
src/cmg/config.py
src/cmg/training.py
src/cmg/evaluation.py
scripts/train.py
scripts/validate_model_phase_m1.py
tests/test_model_phase_m1.py
data/processed/stats/model_phase_m1_validation.json
```

---

## 6. 下一阶段

Model Phase M2：方向模块实现。

范围：

```text
direction_embedding
ResidualFiLMAdapter
Medium direction injection
Policy residual direction injection
zero-init identity
sequence / online direction parity
checkpoint direction-prefix warm-start smoke
```

M2 不应提前实现 direction-macro loss 或正式 stage39 训练；它们分别属于 M3 和 M4。
