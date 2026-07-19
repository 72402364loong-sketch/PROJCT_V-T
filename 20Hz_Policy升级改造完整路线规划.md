# 20Hz Policy 升级改造完整路线规划

## 0. 目标与边界

本路线规划用于将当前 stage38j 三指 per-finger residual policy 从 4Hz 窗口级输出升级为 20Hz 离线训练与评估基线。第一阶段只落地离线数据、训练、诊断和频率消融闭环，不把在线 ROS2 runtime、异步视觉线程、stale input 回退、rate limiter 和真实执行器闭环纳入本轮实现。

第一阶段的最终定义为：

```text
20Hz
window-end anchored
strictly causal
1.0s tactile/visual context
80ms causal median target
0.75s raw-tactile reference
20Hz recalibrated medium gate
per-finger reference-anchored residual policy baseline
```

核心原则：

- 不插值旧 4Hz 或 10Hz 窗口，必须从原始视频、触觉、事件时间戳重新生成 policy 时刻与窗口。
- 不覆盖现有 `data/processed/` 主产物，所有 F4/F10/F20 causal-v2 数据使用独立目录。
- 训练、标签、评测、绘图统一以 `policy_timestamp = window_end` 为主时间轴。
- `window_center` 只保留为兼容字段，不能再作为控制语义上的预测时刻。
- F4/F10/F20 必须共享相同因果语义、reference 定义、target 构造、object split、模型容量和评测代码。

## 1. 已确认的关键决策

| 主题 | 决策 |
|---|---|
| 第一阶段范围 | 只做离线 20Hz 数据、训练、评估和 F4/F10/F20 配置准备 |
| Policy 时间锚点 | `policy_timestamp = window_end` |
| 输入因果窗口 | `[t_k - T, t_k]`，所有输入满足 `t <= t_k` |
| 上下文长度 | 第一版保持 1.0s tactile context 和 1.0s visual context |
| Target 构造 | 80ms 短因果窗口 median，少样本时回退 |
| Reference 定义 | 0.75s raw tactile median，按秒定义，不再依赖窗口数量 |
| Medium 策略 | 旧 4Hz medium 不能直接冻结使用，先做 20Hz recalibration |
| 数据目录 | 新增 `policy_4hz_causal_v2`、`policy_10hz_causal_v2`、`policy_20hz_causal_v2` |
| 视觉策略 | 第一版保持当前视觉输入语义，不引入 5Hz/10Hz low-rate cache |
| 频率消融 | 先运行 F20，同时准备 F4/F10/F20 三套可复现配置 |

## 2. 数据产物规划

### 2.1 目录结构

```text
data/processed/
  policy_4hz_causal_v2/
    samples.csv
    windows.csv
    stats/
    cache/
    manifest.yaml
  policy_10hz_causal_v2/
    samples.csv
    windows.csv
    stats/
    cache/
    manifest.yaml
  policy_20hz_causal_v2/
    samples.csv
    windows.csv
    stats/
    cache/
    manifest.yaml
```

`manifest.yaml` 至少记录：

- `policy_rate_hz` 和 `policy_stride_sec`
- `policy_timestamp_anchor: window_end`
- `causal_only: true`
- `tactile_context_sec`、`visual_context_sec`
- target 聚合方式和时长
- reference 来源、统计量和时长
- 原始传感器频率
- split 文件版本
- 数据生成脚本路径、git commit 或本地变更摘要
- 生成时间、生成命令、样本数、窗口数、phase 分布

### 2.2 Stage 配置读取路径

当前 dataset 默认读取：

```text
data/processed/samples.csv
data/processed/windows.csv
```

本轮需要让 stage data config 显式指定：

```yaml
data:
  samples_path: data/processed/policy_20hz_causal_v2/samples.csv
  windows_path: data/processed/policy_20hz_causal_v2/windows.csv
```

这能避免污染旧实验，也能让 F4/F10/F20 在同一代码路径下复现。

## 3. 时间轴与窗口定义

统一定义 policy 时刻：

```text
t_k = t_0 + k * policy_stride_sec
policy_timestamp = t_k = window_end
window_start = t_k - context_sec
window_end = t_k
window_center = window_start + 0.5 * context_sec
```

F20 第一版配置：

```yaml
data:
  policy_rate_hz: 20
  policy_stride_sec: 0.05
  policy_timestamp_anchor: window_end
  causal_only: true

temporal:
  tactile_context_sec: 1.0
  visual_context_sec: 1.0
```

窗口内采样：

- 触觉输入使用 `[t_k - 1.0, t_k]` 内的原始触觉采样，再重采样到模型需要的 `tactile_points_per_window`。
- 视觉输入保持当前语义，使用同一 1.0s 因果窗口内的视频帧采样。
- 任何 sidecar/cache 需要保留 `policy_timestamp`、原始触觉索引、原始触觉时间戳、视频帧索引和视频帧时间戳。

验收标准：

- 每个窗口所有 tactile/video 时间戳均 `<= policy_timestamp`。
- F20 同一 episode 相邻 `policy_timestamp` 差值稳定为 0.05s。
- 绘图 CSV 同时包含 `window_start`、`window_end`、`window_center`、`policy_timestamp`。
- 默认绘图横轴改用 `policy_timestamp`。

## 4. Target 与 Reference 重建

### 4.1 Per-finger target

第一版采用 80ms 因果 median：

```text
F_i*(t_k) = median { F_i(t) : t in [t_k - 0.08, t_k] }
```

回退规则：

- 窗口内至少 2 个原始触觉样本：使用 median。
- 只有 1 个样本：使用最近因果样本。
- 没有样本：标记 invalid，不生成监督。

配置建议：

```yaml
target:
  aggregation: median
  aggregation_sec: 0.08
  min_samples: 2
  fallback: latest_causal
```

需要额外做一次低成本标签诊断：

- latest causal sample
- 60ms median
- 80ms median
- 100ms median

诊断重点：

- 峰值是否被 median 明显压低
- 零交叉时刻是否偏移
- 正/负/零 residual 比例是否异常变化
- 每指 large-delta 分布是否被稀释

### 4.2 Reference force

第一版彻底改为按秒定义，不再使用 `reference_force_window_count`：

```text
F_ref,i = median { F_i(t) : t in [t_stable_end - 0.75, t_stable_end] }
```

建议配置：

```yaml
reference:
  duration_sec: 0.75
  statistic: median
  source: raw_tactile
```

要求：

- reference 基于原始约 26Hz 触觉曲线计算。
- reference 位于 interface 之前的稳定抓取阶段。
- reference 与 policy rate 无关，F4/F10/F20 共享完全相同定义。
- 每指 reference 计算一次后映射到该 episode 的所有 policy 时刻。

### 4.3 Residual target

每指残差：

```text
Delta F_i*(t_k) = F_i*(t_k) - F_ref,i
```

F20 数据生成后必须统计：

- 正残差比例、负残差比例、近零比例
- 每指 large-delta 数量和幅值分布
- 每对象、每 episode 的 residual 分布
- stable residual 噪声和死区阈值建议

## 5. Medium Recalibration 路线

旧 stage38j 的 `medium_head` 来自 4Hz 节奏，不能直接冻结用于 20Hz。推荐两步：

### 5.1 20Hz Medium Recalibration

目标：

- 在 F20 causal-v2 数据上重新适配 medium GRU 的时间步语义。
- 尽量保留旧 medium 表征能力，降低迁移风险。

训练策略：

- 加载旧 medium 权重。
- 解冻 `MediumBeliefHead/GRU`。
- 可选低学习率解冻 `TactileEvidenceEncoder`。
- 视觉编码器保持冻结或使用现有缓存。
- 学习率取旧训练学习率的 0.1 到 0.3 倍。

验收指标：

- `medium_f1_interface`
- interface 激活延迟
- gate 峰值时间误差
- gate 退出延迟
- 与 residual target 的时序相关性

### 5.2 Stage38j F20 Policy Training

目标：

- 加载已经 20Hz 适配的 medium 模块。
- 冻结 medium。
- 训练 per-finger policy residual head。
- 同时评测 predicted gate 和 oracle gate。

训练初始建议：

- 先保持 stage38j 原 policy loss 权重，建立 F20 baseline。
- 记录各 loss 分量和各 mask 有效样本数。
- 再根据 F20 residual 分布调整 interface/stable/direction/magnitude 权重。

## 6. 代码改造清单

### 6.1 数据管线

需要新增或改造：

- 支持 `policy_stride_sec`、`policy_rate_hz`、`policy_timestamp_anchor`。
- 支持 `samples_path` 和 `windows_path` 配置。
- 支持独立 processed 子目录输出。
- `windows.csv` 增加 `policy_timestamp`。
- sidecar cache 增加 policy 时间戳、因果输入时间戳、target 聚合时间段和 reference 时间段。
- 数据生成时保留 `window_center` 兼容字段，但控制语义不再使用它。

建议涉及文件：

- `configs/data/default.yaml`
- `src/cmg/data/preprocess.py`
- `src/cmg/data/dataset.py`
- `src/cmg/evaluation.py`
- `scripts/preprocess.py` 或新增 `scripts/preprocess_policy_rate.py`

### 6.2 标签与 loss

需要新增或改造：

- raw tactile per-finger target 的 80ms 因果 median 聚合。
- raw tactile reference 的 0.75s median 计算。
- residual dead-zone 统计脚本。
- F20 下正/负/零、large-delta、per-finger 分布诊断。
- 时序正则项如后续启用，必须按 `dt=0.05` 归一化。

建议涉及文件：

- `src/cmg/data/tactile.py`
- `src/cmg/data/dataset.py`
- `src/cmg/losses.py`
- `src/cmg/training.py`
- `scripts/diagnose_per_finger_sign_oracles.py`

### 6.3 训练与评估

需要新增或改造：

- F20 medium recalibration stage config。
- F20 stage38j policy config。
- F4/F10/F20 causal-v2 config。
- 评估结果按 `policy_timestamp` 对齐。
- 可视化脚本默认横轴使用 `policy_timestamp`。
- 时移诊断支持 F20 的 `k=-4..4`，对应 `-200ms..200ms`。

建议涉及文件：

- `configs/stages/stage38j_f20_medium_recalibration.yaml`
- `configs/stages/stage38j_f20_causal.yaml`
- `configs/stages/stage38j_f10_causal.yaml`
- `configs/stages/stage38j_f4_causal.yaml`
- `scripts/visualize_finger_policy_predictions.py`
- 新增 `scripts/diagnose_policy_timeshift.py`

## 7. 实验矩阵

### 7.1 第一优先级

| 实验 | 数据 | Medium | Policy | 目的 |
|---|---|---|---|---|
| M20 | F20 causal-v2 | 解冻适配 | 不训练 policy | 20Hz medium gate 适配 |
| P20 | F20 causal-v2 | 冻结 M20 | 训练 stage38j policy | 建立 20Hz per-finger baseline |
| P20 oracle gate | F20 causal-v2 | oracle | 训练或评测 policy | 分离 gate 误差与 residual 误差 |

### 7.2 第二优先级

| 实验 | 数据 | 目的 |
|---|---|---|
| F10 causal-v2 | 10Hz, same causal semantics | 频率消融中间点 |
| F4 causal-v2 | 4Hz, same causal semantics | 与旧 4Hz 语义隔离，公平对照 |
| Target aggregation ablation | F20 | latest / 60ms / 80ms / 100ms 标签差异 |

### 7.3 后续扩展

| 实验 | 内容 |
|---|---|
| V20 | 每个 20Hz policy 时刻更新视觉输入 |
| V10-cache | 视觉 10Hz cache，policy 20Hz |
| V5-cache | 视觉 5Hz cache，policy 20Hz |
| Local per-finger tactile | 增加每指局部 tactile branch |
| Runtime closed-loop | stale input、timeout、clamp、rate limiter、执行器闭环 |

## 8. 诊断指标

### 8.1 控制误差

- `finger_control_interface_mae`
- `finger_delta_interface_mae`
- large-delta MAE
- 相对 reference-only 的净改善

### 8.2 方向指标

- wrong-sign rate
- sign macro-F1
- positive recall
- negative recall
- 每指方向准确率

### 8.3 幅值指标

```text
r_mag = mean(abs(pred_delta)) / mean(abs(target_delta))
```

需要分整体、interface、large-delta、finger0/1/2 报告。

### 8.4 时序指标

- interface 激活延迟
- gate 峰值时间误差
- residual 峰值时间误差
- 零交叉时间误差
- prediction-target 最优时移
- 时移前后 wrong-sign 变化

F20 时移诊断：

```text
k in {-4,-3,-2,-1,0,1,2,3,4}
time shift = k * 0.05s
range = -200ms..200ms
```

### 8.5 稳定性指标

- stable leakage
- 最大单步变化
- 输出变化率
- 命令抖动代理指标
- 各 mask 有效样本数和 loss 占比

## 9. 推荐执行顺序

### 阶段 A：实现数据路径与 causal-v2 schema

产物：

- 支持 `samples_path` / `windows_path` 的 dataset/evaluation。
- `windows.csv` 新增 `policy_timestamp`。
- 独立输出目录和 `manifest.yaml`。

验收：

- 不影响旧 `data/processed/windows.csv` 读取。
- 旧 stage 仍可正常加载。
- 新 stage 能显式读取 `policy_20hz_causal_v2`。

### 阶段 B：生成并检查 F20 数据

产物：

- `data/processed/policy_20hz_causal_v2/`
- F20 数据统计报告。
- target aggregation 对比报告。

验收：

- 相邻 policy timestamp 间隔为 0.05s。
- 所有输入无未来帧。
- 每个 interface 区间包含足够 policy 点。
- target/reference/residual 分布合理。

### 阶段 C：20Hz Medium Recalibration

产物：

- `stage38j_f20_medium_recalibration.yaml`
- recalibrated checkpoint
- gate 时序诊断图和指标

验收：

- medium interface F1 不明显劣化。
- gate 与 interface/residual 的时序关系合理。
- hidden state 在 20Hz 下无明显不稳定。

### 阶段 D：Stage38j F20 Policy Baseline

产物：

- `stage38j_f20_causal.yaml`
- F20 policy checkpoint
- F20 三指曲线图和 summary CSV

验收：

- predicted gate 与 oracle gate 两套评估完成。
- `finger_control_interface_mae`、wrong-sign、stable leakage 均有记录。
- 三指输出曲线按 `policy_timestamp` 绘制。

### 阶段 E：补齐 F10/F4 causal-v2 消融

产物：

- `policy_10hz_causal_v2/`
- `policy_4hz_causal_v2/`
- `stage38j_f10_causal.yaml`
- `stage38j_f4_causal.yaml`
- F4/F10/F20 统一消融表

验收：

- 三套实验共享同一 raw-tactile reference 定义。
- 三套实验共享同一 80ms target 聚合规则，低频版本按 policy timestamp 取同样因果规则。
- 不再直接用旧 F4 stage38j 与新 F20 比较。

### 阶段 F：后续 runtime 与闭环

不纳入第一阶段代码交付，但路线保留：

- 20Hz online policy loop。
- 5Hz/10Hz 异步视觉 cache。
- stale input 和 timeout 回退。
- output clamp、residual clamp、rate limiter。
- 执行器映射与真实闭环对照。

## 10. 第一版配置草案

```yaml
name: stage38j_f20_causal

data:
  samples_path: data/processed/policy_20hz_causal_v2/samples.csv
  windows_path: data/processed/policy_20hz_causal_v2/windows.csv
  policy_rate_hz: 20
  policy_stride_sec: 0.05
  policy_timestamp_anchor: window_end
  causal_only: true
  tactile_input_axes: [z]
  attribute_taxonomy: coarse_v2

temporal:
  tactile_context_sec: 1.0
  visual_context_sec: 1.0

target:
  aggregation: median
  aggregation_sec: 0.08
  min_samples: 2
  fallback: latest_causal

reference:
  duration_sec: 0.75
  statistic: median
  source: raw_tactile

medium:
  recalibrate_at_20hz: true
  freeze_during_policy_training: true

visual:
  preserve_current_input_semantics: true
  low_rate_cache_in_baseline: false

policy:
  head_type: state_residual_per_finger_sign_specific
  base_source: reference_force
  use_reference_force_context: true
  reference_force_context_scale: 100.0
  residual_output_scale: 20.0
  finger_count: 3
```

## 11. 风险与控制

| 风险 | 表现 | 控制手段 |
|---|---|---|
| 20Hz 窗口高度重叠 | 指标虚高、训练冗余 | episode/object 聚合评估，固定 step 对比 |
| target median 压峰 | residual 幅值更小 | 做 latest/60/80/100ms 标签诊断 |
| 旧 medium 时间尺度失配 | gate 滞后或提前 | 先做 20Hz recalibration |
| 近零 residual 增多 | zero-residual 偏置增强 | 重新估计 dead-zone 和 sign 分布 |
| 数据污染旧实验 | 旧 stage 不可复现 | 独立 processed 子目录，manifest 记录 |
| 绘图时间轴误导 | 看似提前或滞后 | 统一 `policy_timestamp` 绘图 |
| 训练成本增加 | step/time 约 5 倍 | 固定采样量或比较相同优化 step |

## 12. 第一阶段完成标准

第一阶段视为完成，需要同时满足：

1. F20 causal-v2 数据目录和 manifest 完整生成。
2. 所有窗口通过严格因果检查。
3. F20 medium recalibration 有 checkpoint 和 gate 诊断。
4. Stage38j F20 policy baseline 完成训练或至少完成可复现实验启动配置。
5. F20 三指预测可视化和 summary CSV 使用 `policy_timestamp`。
6. F4/F10/F20 causal-v2 配置已准备，且能复用同一评估脚本。
7. 关键指标包含控制误差、方向、幅值、时序、稳定性五类。
8. 文档中明确第一阶段不声明在线闭环性能，只声明离线 20Hz causal policy baseline。

## 13. 推荐论文表述边界

当前第一阶段可以使用：

```text
20-Hz causal per-finger residual policy baseline
window-end anchored policy prediction
timestamp-aligned tactile observations
raw-tactile reference anchored force proxy
offline frequency ablation among 4/10/20 Hz causal policies
```

在线 runtime 和真实执行器闭环完成前，避免使用：

```text
high-bandwidth force control
instantaneous slip suppression
continuous-time force servo
validated real-time closed-loop control
```

闭环阶段完成后，再升级为：

```text
real-time tactile-in-the-loop per-finger force control
20-Hz multimodal force adaptation
sampled-data tactile feedback control
actuator-aligned force-reference regulation
```
