# 跨介质视触抓取系统 20 Hz Policy 升级实施说明

## 1. 改动目标

本次改动将力预测 Policy 的输出频率统一提升至 **20 Hz**，即：

```text
policy_stride_sec: 0.05
policy_update_rate: 20 Hz
```

相较当前或历史设置：

| 设置 | Stride | Policy 频率 |
|---|---:|---:|
| 原始版本 | 0.25 s | 4 Hz |
| 过渡方案 | 0.10 s | 10 Hz |
| 本次目标 | 0.05 s | 20 Hz |

系统现有原始传感器频率为：

```text
RGB video: 30 Hz
Tactile:   approximately 26 Hz
```

因此，20 Hz Policy 每 50 ms 更新一次，每个周期平均可获得：

```text
new video frames per policy step:   30 / 20 = 1.5
new tactile frames per policy step: 26 / 20 = 1.3
```

20 Hz 低于触觉真实采样率，绝大多数 Policy 周期能够获得新的触觉观测，同时避免 30 Hz Policy 中部分周期无新触觉数据的问题。

本次升级的核心目的不是单纯增加窗口数量，而是：

1. 提高水—气界面阶段的时间分辨率；
2. 缩短每指残差控制的响应周期；
3. 改善 residual 正负方向切换、峰值时间和零交叉的表达；
4. 支撑“实时触觉在环的三指力控制”论文叙事；
5. 构建“低频视觉全局上下文 + 20 Hz 触觉驱动 Policy + 更高频执行器跟踪”的多速率系统。

---

## 2. 改动后的系统定位

建议将最终系统定义为三级多速率架构。

### 2.1 原始传感器采集层

```text
RGB stream:       30 Hz
Tactile stream:   approximately 26 Hz
```

所有原始数据必须保留各自的真实时间戳，不应假设视频第 n 帧与触觉第 n 帧天然同步。

### 2.2 学习策略层

```text
Visual context encoder: 5–10 Hz, cached
Tactile/medium update:   20 Hz
Per-finger policy:       20 Hz
```

视觉被定位为全局对象—场景辅助上下文，因此不需要在每个 20 Hz Policy 周期重新运行 OpenCLIP ViT-B/16。视觉特征可按 5–10 Hz 更新，并在两个视觉更新时刻之间缓存。

每指 Policy 在 20 Hz 下读取：

- 最新可用的视觉全局特征；
- 截止当前 Policy 时刻的因果触觉窗口；
- 当前介质信念；
- 当前 episode 的每指 reference force；
- 每指局部或全局触觉上下文；
- finger embedding。

### 2.3 执行器控制层

Policy 输出：

\[
\mathbf F_{\mathrm{des}}(t_k)
=
\mathbf F_{\mathrm{ref}}
+
g(t_k)\hat{\Delta\mathbf F}(t_k)
\]

其中：

- \(\mathbf F_{\mathrm{des}}\) 为三指目标力或触觉力代理目标；
- \(\mathbf F_{\mathrm{ref}}\) 为当前 episode 稳定抓取阶段的三指参考值；
- \(g(t_k)\) 为介质阶段门控；
- \(\hat{\Delta\mathbf F}(t_k)\) 为模型预测的每指残差。

执行器层应以不低于 20 Hz 的频率持续跟踪最新目标。若 ESP32、驱动器或底层控制器支持更高频率，应使用其实际稳定频率执行：

- 目标保持；
- 变化率限制；
- 饱和约束；
- PI/PID 或其他闭环跟踪；
- 异常回退与安全保护。

论文中应区分：

```text
Policy update rate
Actuator command update rate
Low-level servo rate
Sensor acquisition rate
```

不要将四者混写成同一个“控制频率”。

---

## 3. 时间轴与因果同步

## 3.1 以 Policy 时刻为统一锚点

定义 Policy 输出时刻：

\[
t_k = t_0 + k \times 0.05\ \mathrm{s}.
\]

每次生成输入时，只允许使用：

\[
t \le t_k
\]

的传感器数据。严禁使用 \(t > t_k\) 的未来帧，否则离线评测会产生因果泄漏，在线部署无法复现。

建议同步规则如下：

```text
policy timestamp: t_k

visual frame:
    use the latest frame with timestamp <= t_k
    or use a causal visual window ending at t_k

tactile window:
    include all tactile samples in [t_k - T_tactile, t_k]

medium state:
    recurrent state updated only with observations available by t_k

reference force:
    computed only from the pre-interface stable interval
```

## 3.2 不采用帧号硬对齐

视频约 30 Hz，触觉约 26 Hz，二者存在：

- 不同采样周期；
- 采集抖动；
- ROS2/串口传输延迟；
- 丢帧或重复帧；
- 启动时刻偏差。

因此不应采用：

```text
video_frame[n] <-> tactile_frame[n]
```

而应采用真实时间戳进行异步融合。

推荐保存以下字段：

```text
policy_timestamp
latest_visual_timestamp
latest_tactile_timestamp
visual_age_ms
tactile_age_ms
num_visual_samples_in_window
num_tactile_samples_in_window
```

其中：

\[
\text{visual\_age}=t_k-t_{\mathrm{visual,last}}
\]

\[
\text{tactile\_age}=t_k-t_{\mathrm{tactile,last}}
\]

可用于检测输入是否过旧。

## 3.3 输入过期处理

建议设置最大允许数据年龄，例如：

```text
max_visual_age_ms: 100–200
max_tactile_age_ms: 60–80
```

若超过阈值：

- 保持上一目标；
- 降低残差幅值；
- 回退到 reference-only；
- 或触发安全停止。

具体阈值应根据实际采集抖动统计确定，不宜仅凭经验固定。

---

## 4. 数据窗口重建

## 4.1 必须从原始时间戳重新生成 20 Hz 样本

不建议将已有 4 Hz 或 10 Hz 窗口直接插值到 20 Hz。正确方式是从：

- 原始 30 Hz 视频；
- 原始约 26 Hz 触觉；
- 原始事件标注；
- 原始时间戳；

重新生成 Policy 时刻和窗口。

需要重新构建：

- `windows.csv`；
- medium phase label；
- stable mask；
- interface mask；
- reference window；
- per-finger force target；
- per-finger residual target；
- gate target；
- 相关 sidecar cache。

20 Hz 只提高时间采样密度，不会创造新的独立 episode 或对象。

## 4.2 保持真实时间感受野不变

Stride 改变后，不能继续固定使用原来的窗口数量而忽略时间长度。

若原设置为：

```text
sequence_steps_old = N
stride_old = 0.25 s
```

则原真实时间跨度近似为：

\[
T_{\mathrm{context}} = N \times 0.25.
\]

在 20 Hz 下，为保持相同时间跨度：

\[
N_{\mathrm{new}}
\approx
\frac{T_{\mathrm{context}}}{0.05}
=
5N.
\]

例如：

| 原窗口步数 | 原时间跨度 | 20 Hz 对应步数 |
|---:|---:|---:|
| 4 | 1.0 s | 20 |
| 8 | 2.0 s | 40 |
| 12 | 3.0 s | 60 |

但不建议机械地将所有序列长度扩大 5 倍，否则计算量和序列冗余可能显著增加。

更合理的做法是将配置从“固定步数”改为“固定秒数”，例如：

```yaml
temporal:
  tactile_context_sec: 1.0
  medium_context_sec: 2.0
  visual_context_sec: 0.5
  policy_stride_sec: 0.05
```

由数据管线根据真实采样率动态选取窗口内原始样本。

## 4.3 视觉与触觉窗口无需相同步数

视觉和触觉可分别定义时间窗口：

```text
visual global context:  0.3–1.0 s
tactile local context:  0.3–1.0 s
medium GRU context:     1.0–3.0 s or recurrent hidden state
```

视觉主要提供慢变化的对象和场景信息，触觉负责局部接触动态，两者不必强制使用相同采样点数。

## 4.4 参考力区间按秒定义

Reference force 不能继续使用固定“若干窗口”定义。

建议配置为：

```yaml
reference:
  duration_sec: 0.5–1.0
  statistic: median
  min_valid_tactile_samples: <根据26 Hz设置>
```

例如 1 秒 reference 区间应包含约 26 个原始触觉采样，而不是 20 个 Policy 窗口简单平均。

每指 reference 建议直接基于原始触觉时序计算，然后映射到所有 20 Hz Policy 时刻。

---

## 5. 标签重建细节

## 5.1 Per-finger target

当前每指标签为：

\[
F_i(t)
=
\operatorname{mean}_{j}
\left|
z_{i,j}(t)-b_{i,j}
\right|.
\]

在 20 Hz Policy 时刻 \(t_k\)，目标值建议通过原始触觉数据构造，而不是先生成 26 Hz 标签再简单复制。

可选方式：

### 方式 A：最近因果样本

\[
F_i^*(t_k)=F_i(t_{\mathrm{latest}}\le t_k).
\]

优点：

- 严格因果；
- 适合在线复现；
- 不引入未来信息。

缺点：

- 可能形成阶梯状 target。

### 方式 B：短因果窗口稳健聚合

\[
F_i^*(t_k)
=
\operatorname{median/mean}
\{F_i(t):t\in[t_k-\tau,t_k]\}.
\]

建议 \(\tau\) 取 40–100 ms，并通过实验确定。

优点：

- 降低单帧噪声；
- 保持因果性；
- 更适合稳定执行目标。

### 不建议：双向线性插值

若插值使用 \(t_k\) 后的触觉值，会产生未来信息泄漏。离线生成控制标签时，应确保部署可用性。

## 5.2 Residual target

每指残差：

\[
\Delta F_i^*(t_k)
=
F_i^*(t_k)-F_{\mathrm{ref},i}.
\]

需要重新统计 20 Hz 下：

- 正残差比例；
- 负残差比例；
- 近零残差比例；
- large-delta 分布；
- 每根手指分布；
- 每个对象和 episode 分布。

由于时间分辨率增加，零交叉附近会出现更多小残差样本，可能进一步强化模型的 zero-residual 偏置。

因此 sign-first 训练应采用死区：

\[
c_i=
\begin{cases}
-1,&\Delta F_i^*<-\tau_i,\\
0,&|\Delta F_i^*|\le\tau_i,\\
+1,&\Delta F_i^*>\tau_i.
\end{cases}
\]

阈值 \(\tau_i\) 应根据 stable 阶段噪声确定，例如：

\[
\tau_i=k\sigma_{\mathrm{stable},i}
\]

或使用 stable residual 绝对值的高分位数。

## 5.3 Medium phase 与 Interface mask

阶段标签仍由：

- `t_if_enter`；
- `t_if_exit`；

按真实时间映射到 20 Hz Policy 时刻。

应同时区分：

```text
semantic interface mask
control-effective transition mask
```

原因是语义上的水—气界面区间不一定与力变化最显著的区间完全重合。

建议额外保留：

```text
pre_interface_margin_sec
post_interface_margin_sec
```

用于分析 gate 是否应提前激活或延后退出。

## 5.4 Gate target

若采用 \(p_{\mathrm{interface}}\) 作为 gate，需要确认 20 Hz 下 gate 的：

- 激活延迟；
- 峰值时刻；
- 退出延迟；
- 与 residual target 的时间相关性。

不要只比较 medium classification F1，还应比较控制意义上的时序指标。

---

## 6. 高频重叠窗口与样本独立性

20 Hz 相较 4 Hz 会生成约 5 倍的 Policy 时刻。若原窗口数约为 24,611，则理论上可能增加至约：

\[
24,611\times5\approx123,055
\]

个窗口，实际数量取决于 episode 长度和边界处理。

但这些窗口高度重叠，不能将其解释为“有效数据量增加 5 倍”。

需要注意：

1. 相邻样本可能共享 90% 以上输入；
2. 训练梯度高度冗余；
3. 长 episode 会产生更多窗口并主导训练；
4. window-level 统计显著性会被高估；
5. 同一 episode 内相邻窗口不能跨 split；
6. 评测应以 episode 和 object 为统计单位。

推荐采样方式：

```text
object-balanced
episode-balanced
phase-balanced
direction-balanced
large-delta-aware
```

而不是直接对所有窗口均匀随机采样。

一个 batch 内应尽量包含来自不同 episode 和不同对象的样本，避免连续窗口集中出现。

---

## 7. 模型结构注意事项

## 7.1 视觉特征缓存

建议：

```yaml
runtime:
  policy_rate_hz: 20
  visual_encode_rate_hz: 5 or 10
  cache_visual_feature: true
```

在 20 Hz Policy 周期中复用最近的视觉全局表示：

\[
h_v(t_k)=h_v(t_{v,\mathrm{latest}}).
\]

应向模型提供或记录视觉特征年龄，以分析旧视觉特征是否影响性能。

## 7.2 触觉编码

若仍采用 global tactile context，20 Hz 可先作为频率升级基线。

后续推荐加入 local per-finger tactile branch：

\[
h_{t,i}^{\mathrm{local}}
=
E_{\mathrm{finger}}(X_i).
\]

每指 Policy 输入为：

\[
[
h_{t,i}^{\mathrm{local}},
h_v^{\mathrm{global}},
p_{\mathrm{medium}},
F_{\mathrm{ref},i},
e_i
].
\]

20 Hz 本身不会解决 wrong-sign；它只是提高时序分辨率。若方向问题持续，应继续推进：

- sign-first curriculum；
- direction/magnitude decoupled head；
- local per-finger tactile branch。

## 7.3 Recurrent state

如果 GRU 以 Policy 时刻更新，其更新率将从 4/10 Hz 变为 20 Hz。

需要重新检查：

- hidden state 的时间尺度；
- 序列长度；
- truncated BPTT 长度；
- 梯度稳定性；
- dropout 和 recurrent dropout；
- 是否重复输入同一触觉观测。

推荐将 \(\Delta t\) 或 observation age 作为显式输入，避免默认固定等间隔。

---

## 8. 损失函数与权重调整

## 8.1 原权重不可直接照搬

20 Hz 下窗口密度、零残差比例、相邻样本相关性均发生变化，因此以下权重需要重新验证：

```text
interface_weight
large_delta_weight
stable_zero_weight
direction_weight
magnitude_weight
smoothness_weight
```

建议先在保持原权重的情况下建立 20 Hz baseline，然后根据 loss 分解进行调整。

应记录每个 epoch 中：

- Interface loss 占比；
- Stable loss 占比；
- Direction loss 占比；
- Magnitude loss 占比；
- 每种 mask 的有效样本数；
- 正/负/零方向样本数；
- 每个 episode 的平均贡献。

## 8.2 Interface 与 Stable 分开约束

推荐：

\[
\mathcal L
=
\lambda_{\mathrm{int}}\mathcal L_{\mathrm{interface}}
+
\lambda_{\mathrm{stable}}\mathcal L_{\mathrm{stable}}
+
\lambda_{\mathrm{dir}}\mathcal L_{\mathrm{direction}}
+
\lambda_{\mathrm{mag}}\mathcal L_{\mathrm{magnitude}}
+
\lambda_{\mathrm{rate}}\mathcal L_{\mathrm{rate}}.
\]

其中：

\[
\mathcal L_{\mathrm{stable}}
=
\frac{1}{N_s}
\sum_{t\in\mathcal S}
|\hat{\Delta F}(t)|
\]

用于抑制 stable leakage。

## 8.3 所有时序正则项应按 \(\Delta t\) 归一化

若使用平滑约束：

\[
|\hat{\Delta F}_{k+1}-\hat{\Delta F}_k|,
\]

其物理意义会随频率变化。

建议改为变化率：

\[
\mathcal L_{\mathrm{rate}}
=
\frac{1}{T-1}
\sum_k
\left|
\frac{
\hat{\Delta F}_{k+1}-\hat{\Delta F}_k
}{0.05}
\right|.
\]

若使用最大允许变化率：

\[
\mathcal L_{\mathrm{rate}}
=
\sum_k
\max
\left(
0,
\frac{
|\hat{\Delta F}_{k+1}-\hat{\Delta F}_k|
}{0.05}
-r_{\max}
\right).
\]

同理，所有一阶差分、二阶差分、速度和加速度特征都应显式除以对应时间间隔。

---

## 9. 训练成本与资源

## 9.1 训练时间

若从 4 Hz 提升到 20 Hz，在 episode 总时长不变的情况下，窗口数量理论上约增加 5 倍。

若：

- batch size 不变；
- epoch 数不变；
- 每个窗口均完整训练；

训练总 step 和时间可能接近增加 5 倍。

但有效独立信息不会增加 5 倍，因此建议：

- 固定每 epoch 的采样窗口数；
- 使用 episode-balanced sampler；
- 对高重叠窗口进行随机子采样；
- 比较相同优化 step，而非只比较相同 epoch；
- Stage1/Stage2 的重型视觉训练不必对所有 20 Hz 窗口重复计算。

## 9.2 在线推理

若完整模型从 10 Hz 提升至 20 Hz，每秒 Policy 前向次数约增加 2 倍。

但采用视觉缓存后：

\[
C_{\mathrm{total}}
=
f_v C_v + f_p C_p
\]

其中：

- \(C_v\) 为视觉编码成本；
- \(C_p\) 为轻量触觉和 Policy 成本；
- \(f_v\) 可固定为 5–10 Hz；
- \(f_p=20\) Hz。

因此总算力不一定严格翻倍。

## 9.3 显存

若：

- 在线 batch size = 1；
- 模型结构不变；
- 单次输入上下文长度不显著增加；

峰值显存通常不会随 Policy 频率成倍增长。

显著增加的主要是：

- GPU/CPU 利用率；
- 每秒前向吞吐；
- 数据预处理负载；
- I/O；
- 训练样本数量；
- 缓存和索引元数据。

---

## 10. 实时部署要求

## 10.1 Deadline

20 Hz 周期预算为：

\[
T_{\mathrm{deadline}}=50\ \mathrm{ms}.
\]

端到端延迟包括：

\[
T_{\mathrm{e2e}}
=
T_{\mathrm{sensor}}
+
T_{\mathrm{sync}}
+
T_{\mathrm{preprocess}}
+
T_{\mathrm{inference}}
+
T_{\mathrm{ROS2}}
+
T_{\mathrm{ESP32}}
+
T_{\mathrm{actuation}}.
\]

建议目标：

```text
mean latency: clearly below 50 ms
p95 latency:  below 50 ms
deadline miss rate: as close to 0 as possible
```

更稳妥的工程目标是让模型和数据处理部分控制在 25–35 ms 内，为通信和调度抖动保留余量。

论文应报告：

- mean；
- standard deviation；
- p95；
- p99 或 maximum；
- deadline miss rate；
- 实际平均 Policy 频率。

## 10.2 异步线程

推荐至少拆分：

```text
Thread/Node A: RGB acquisition
Thread/Node B: tactile acquisition
Thread/Node C: visual encoding and cache update
Thread/Node D: 20 Hz policy inference
Thread/Node E: actuator command / low-level tracking
Thread/Node F: logging and synchronization
```

避免视觉推理阻塞触觉采集和执行器命令。

## 10.3 过期数据与超时回退

Policy 超时时：

1. 不发送未经完成的新命令；
2. 保持上一目标；
3. 对残差逐渐衰减；
4. 或回退至 reference-only；
5. 连续多次超时后触发安全停止。

## 10.4 输出安全约束

每指输出应执行：

```text
absolute force/proxy bounds
residual bounds
rate limit
optional low-pass filtering
NaN/Inf check
sensor-validity check
```

例如：

\[
F_{\min,i}
\le
F_{\mathrm{des},i}
\le
F_{\max,i}
\]

\[
|F_{\mathrm{des},i}(k)-F_{\mathrm{des},i}(k-1)|
\le
r_{\max,i}\Delta t.
\]

具体上下限必须来自执行器和传感器实验，而不是任意设置。

---

## 11. 执行器映射与闭环控制

模型输出不应在未验证的情况下直接映射为原始 PWM、电机位置或电流。

推荐链路：

\[
F_{\mathrm{des},i}
\rightarrow
e_i=F_{\mathrm{des},i}-F_{\mathrm{meas},i}
\rightarrow
u_i
\rightarrow
\text{actuator}
\rightarrow
F_{\mathrm{meas},i}^{\mathrm{next}}.
\]

需要验证：

1. 执行器命令与触觉力代理之间的单调性；
2. 不同对象和姿态下的重复性；
3. 迟滞；
4. 死区；
5. 响应时间；
6. 稳态误差；
7. 三指之间的差异。

若当前标签尚未标定为牛顿，论文应使用：

```text
tactile-derived per-finger force reference
contact-force proxy
sensor-domain force target
```

而不是未经说明地直接写 physical force in N。

---

## 12. 必须补充的实验

## 12.1 频率消融

至少比较：

| 设置 | Policy rate |
|---|---:|
| F4 | 4 Hz |
| F10 | 10 Hz |
| F20 | 20 Hz |

控制变量：

- 相同 object split；
- 相同原始 episode；
- 相同真实时间感受野；
- 相同 reference 定义；
- 相同模型容量；
- 相同标签构造逻辑；
- 尽可能比较相同优化 step 或充分收敛结果。

重点指标：

### 控制误差

- `finger_control_interface_mae`；
- `finger_delta_interface_mae`；
- large-delta MAE；
- 相对 reference-only 的净改善。

### 方向

- wrong-sign rate；
- sign macro-F1；
- positive recall；
- negative recall；
- 每指方向准确率。

### 幅值

\[
r_{\mathrm{mag}}
=
\frac{\mathbb E|\hat{\Delta F}|}
{\mathbb E|\Delta F^*|}.
\]

### 时序

- Interface 激活延迟；
- 峰值时间误差；
- 零交叉时间误差；
- prediction-target 最优时移；
- 时移前后 wrong-sign 变化。

### 稳定性

- stable leakage；
- 最大单步变化；
- 输出变化率；
- 命令抖动；
- deadline miss。

## 12.2 时移诊断

20 Hz 下一个 step 为 50 ms。建议评估：

\[
k\in\{-4,-3,-2,-1,0,1,2,3,4\}
\]

对应 \(-200\) ms 到 \(+200\) ms 的时移。

若平移 1–2 个 step 后 wrong-sign 显著下降，说明问题包含控制延迟或标签错位，而不只是方向分类失败。

## 12.3 闭环执行对照

至少比较：

1. Fixed conservative force；
2. Reference-only；
3. Learned absolute force；
4. Reference-anchored residual at 10 Hz；
5. Reference-anchored residual at 20 Hz。

报告：

- 成功率；
- 掉落率；
- 界面滑移；
- 峰值接触力代理；
- 三指受力不均；
- 过度夹持；
- 输出平滑度；
- 对未见对象的表现；
- 推理和控制延迟。

---

## 13. 视觉增益消融与 20 Hz 的关系

视觉当前定位为全局辅助上下文，因此最关键比较为：

```text
Reference + local tactile at 20 Hz
vs.
Reference + local tactile at 20 Hz + cached global vision
```

建议设置：

| 配置 | 视觉 | 触觉 Policy |
|---|---:|---:|
| V0 | 无 | 20 Hz |
| V1 | 5 Hz cached | 20 Hz |
| V2 | 10 Hz cached | 20 Hz |
| V3 | shuffled vision | 20 Hz |

判断视觉是否有效时，应关注：

- 未见对象结果；
- wrong-sign；
- negative recall；
- large-delta；
- 特定对象组；
- 随机种子方差。

若视觉无稳定增益，不应为追求 20 Hz 而重复运行重型 ViT。

---

## 14. 推荐配置示例

```yaml
data:
  policy_stride_sec: 0.05
  video_source_rate_hz: 30.0
  tactile_source_rate_hz: 26.0
  use_timestamp_alignment: true
  causal_only: true

temporal:
  tactile_context_sec: 0.5
  visual_context_sec: 0.5
  medium_context_sec: 2.0
  reference_duration_sec: 1.0

runtime:
  policy_rate_hz: 20
  visual_encode_rate_hz: 5
  cache_visual_feature: true
  max_visual_age_ms: 200
  max_tactile_age_ms: 80
  deadline_ms: 50

policy:
  head_type: state_residual_per_finger
  finger_count: 3
  use_local_finger_tactile: true
  direction_magnitude_decoupling: optional

safety:
  enable_output_clamp: true
  enable_rate_limit: true
  fallback_to_reference_on_timeout: true
  hold_last_command_on_single_miss: true
```

上述数值为建议起点，最终应根据真实数据分布、延迟测试和执行器响应实验确定。

---

## 15. 代码与数据迁移检查清单

### 数据层

- [ ] 将 Policy stride 设为 0.05 s；
- [ ] 从原始时间戳重新生成 Policy 时刻；
- [ ] 禁止直接插值旧 4/10 Hz 窗口；
- [ ] 使用异步时间戳对齐视频与触觉；
- [ ] 所有输入严格满足 \(t\le t_k\)；
- [ ] 按秒重新定义 reference 区间；
- [ ] 重新生成每指 target 和 residual；
- [ ] 重新统计正/负/零与 large-delta 分布；
- [ ] 重新生成 medium、stable、interface mask；
- [ ] 检查每个 Interface 区间包含的 Policy 点数；
- [ ] 确保 object/episode split 不变。

### 模型层

- [ ] 检查 GRU 更新率变化；
- [ ] 保持或重新定义真实时间感受野；
- [ ] 视觉特征支持缓存；
- [ ] Policy 可读取输入 age 或时间差；
- [ ] 检查相同触觉帧重复使用时的行为；
- [ ] 继续保留 reference-only 路径；
- [ ] 为 sign/magnitude 诊断保留独立指标；
- [ ] 后续评估 local per-finger tactile branch。

### 损失层

- [ ] 原权重先建立 20 Hz baseline；
- [ ] 重新调 interface/stable 权重；
- [ ] 时序损失按 \(\Delta t=0.05\) 归一化；
- [ ] 死区阈值按 stable 噪声重新估计；
- [ ] 避免零交叉附近小残差过度主导；
- [ ] 记录各 mask 的有效样本数和 loss 占比。

### 训练层

- [ ] 使用 episode-balanced sampler；
- [ ] batch 内避免大量相邻窗口；
- [ ] 固定每 epoch 采样量或比较相同 step；
- [ ] 评测按 episode/object 聚合；
- [ ] 至少使用多个随机种子；
- [ ] 保留 4/10/20 Hz 频率消融；
- [ ] 检查训练时间、I/O 和缓存规模。

### 部署层

- [ ] 20 Hz Policy deadline 为 50 ms；
- [ ] 测量 mean/p95/p99/max 延迟；
- [ ] 记录 deadline miss rate；
- [ ] 视觉编码异步运行；
- [ ] 触觉采集不得被推理阻塞；
- [ ] 输出增加 clamp 和 rate limiter；
- [ ] 实现 stale input 和 timeout 回退；
- [ ] 明确 Policy 输出到执行器命令的映射；
- [ ] 测试每指阶跃、斜坡和动态跟踪；
- [ ] 完成真实闭环任务对照。

---

## 16. 论文中的推荐表述

### 方法描述

> The RGB and tactile streams are acquired at 30 Hz and approximately 26 Hz, respectively. A causal per-finger residual policy operates at 20 Hz using timestamp-aligned tactile observations and asynchronously cached global visual context. At each policy step, the model predicts actuator-aligned per-finger force references relative to the episode-level stable grasp anchor.

### 控制描述

> The predicted force references are transmitted online to the independently controlled fingers. Subsequent tactile measurements are fed back into the policy, forming a tactile-in-the-loop sampled-data control process.

### 多速率描述

> To avoid constraining the control rate by the visual backbone, global visual features are updated asynchronously at a lower rate and cached, whereas tactile state estimation and per-finger force adaptation are executed at 20 Hz.

### 需要避免的表述

在未完成相应验证前，不建议使用：

```text
high-bandwidth force control
high-frequency tactile reflex
instantaneous slip suppression
30 Hz tactile feedback
continuous-time force servo
```

推荐使用：

```text
real-time tactile-in-the-loop per-finger force control
20-Hz multimodal force adaptation
sampled-data tactile feedback control
actuator-aligned force-reference regulation
```

---

## 17. 预期收益与不能解决的问题

### 预期可改善

1. Interface 阶段时间分辨率；
2. residual 正负切换的时间定位；
3. 峰值和零交叉表达；
4. Policy 响应延迟；
5. 实时控制论文叙事；
6. 10 Hz 与 20 Hz 的闭环频率消融；
7. 每指目标曲线的连续性。

### 不会自动解决

1. residual 幅值欠预测；
2. wrong-sign 的表征不足；
3. global context 无法解释每指局部状态；
4. gate 与 residual 的耦合；
5. 正负方向样本不平衡；
6. 触觉标签噪声；
7. 模型输出到执行器命令的映射问题。

因此，20 Hz 升级应与以下路线并行：

```text
sign-first training
direction/magnitude decoupling
local per-finger tactile branch
reference-only baseline
gate/residual diagnostic
closed-loop actuator tracking
```

---

## 18. 最终实施建议

本次升级的推荐主线为：

```text
原始视频 30 Hz
        \
         -> 异步时间戳对齐 -> 低频视觉缓存
        /
原始触觉约 26 Hz
         -> 20 Hz causal tactile window
         -> medium belief
         -> per-finger residual policy
         -> predicted per-finger force references
         -> low-level actuator tracking
         -> new tactile feedback
```

最终系统应明确体现：

> 视觉负责慢变化的全局对象—场景条件，触觉负责快速的局部接触反馈；20 Hz Policy 在两者基础上生成三指执行器对齐的目标力参考，并通过后续触觉反馈闭合控制环。

20 Hz 是当前 30 Hz 视频与约 26 Hz 触觉条件下，在新观测密度、实时响应、计算开销和论文叙事之间较合理的折中。其价值必须通过严格的因果数据构造、端到端延迟测试、4/10/20 Hz 频率消融以及真实闭环执行实验共同证明。
