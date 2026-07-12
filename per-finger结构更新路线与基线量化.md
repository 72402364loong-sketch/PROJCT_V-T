# per-finger residual 结构更新路线与基线量化

更新时间：2026-07-10

## 总体判断

当前瓶颈不应继续简单理解为 `per-finger` 标签是否成立，而应理解为：

1. 模型相对 reference 只学到了较小 residual；
2. residual 方向仍不稳定，尤其 large-delta 子集需要单独看；
3. gate 与 residual 的乘积监督可能削弱 residual head 的有效梯度；
4. 仅使用 global context + finger embedding，可能不足以判断每根手指的局部增力/减力方向。

因此路线应按“基线量化 -> sign 诊断 -> oracle 诊断 -> 38g/38h curriculum -> direction/magnitude 解耦 -> local per-finger tactile branch”推进。

## 路线总览

| 顺序 | 阶段 | 目标 | 成功/失败信号 |
|---:|---|---|---|
| 1 | 基线量化 | 量化 full model 相对 reference-only 的净增益 | 若净增益很小，说明 residual 有效控制贡献有限 |
| 2 | dead-zone sign 诊断 | 避免把近零噪声强行二分类 | sign 指标只在 valid-interface / large-delta 子集上解释 |
| 3 | oracle 诊断 | 分离方向、幅值、gate 三类瓶颈 | 判断下一步应改 loss、head 还是输入表征 |
| 4 | stage38g sign-first | 优先压 per-finger large-delta wrong-sign | wrong-sign 明显下降，stable leakage 不明显恶化 |
| 5 | stage38h magnitude follow-up | 在方向不崩的前提下恢复幅值 | MAE 下降，pred_delta_abs 接近 target，wrong-sign 不反弹 |
| 6 | per-finger direction/magnitude head | 结构上解耦离散方向与连续幅值 | 单头 signed regression 不再承担两个难题 |
| 7 | local per-finger tactile branch | 给每根手指局部触觉状态 | 若方向仍不稳，优先补局部信息而不是继续堆权重 |

## 1. 基线量化

### 计算口径

这里的 reference-only baseline 不是训练日志中的 `finger_control_reference_mae`。训练代码里的该字段表示“模型预测在有 reference 窗口上的误差”，不是“直接输出 reference 的误差”。

本节重新按验证集计算：

```text
reference_only_pred_i = finger_reference_force_i
target_i = finger_control_force_target_i
reference_only_interface_mae = mean(abs(reference_only_pred_i - target_i))
```

聚合 mask：

```text
window_mask
AND phase_label == Interface
AND has_finger_control_target
AND has_finger_reference
```

使用配置：`configs/stages/stage38f_policy_fold1_per_finger_reference_delta_large_delta_guard.yaml`

验证集规模：30 个 sample。

### reference-only baseline

| 指标 | 数值 |
|---|---:|
| reference-only overall MAE | 7.341831 |
| reference-only interface MAE | 8.930208 |
| interface target delta abs mean | 8.930208 |
| interface target delta signed mean | 4.165614 |
| interface count | 1248 |
| overall count | 1518 |
| large-delta threshold | 8.0 |
| large target delta abs mean | 17.387189 |
| large target delta signed mean | 9.030155 |
| large target delta count | 501 |
| large positive count | 362 |
| large negative count | 139 |

### per-finger reference-only interface MAE

| Finger | MAE | Count |
|---|---:|---:|
| finger0 | 7.730863 | 416 |
| finger1 | 8.582875 | 416 |
| finger2 | 10.476884 | 416 |

结论：reference-only 在 interface 上的 MAE 是 `8.930208`，其中 finger2 最难。large-delta 子集正向样本明显多于负向样本，因此后续 sign 评估必须使用 balanced wrong-sign，而不能只看 overall sign accuracy。

### 已有 stage38 best metrics 对比

| Run | Epoch | Selection metric | Selection value | Interface MAE | Gain vs ref-only | Relative gain | Delta MAE | Delta bias | Stable leakage |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| stage38 | 1 | finger_control_interface_mae | 8.938590 | 8.938590 | -0.008383 | -0.09% | 8.941296 | -4.205755 | 0.001574 |
| stage38b | 12 | finger_control_interface_mae | 8.775734 | 8.775734 | 0.154474 | 1.73% | 8.681931 | -3.549571 | 0.028769 |
| stage38c | 24 | finger_control_interface_mae | 8.627478 | 8.627478 | 0.302730 | 3.39% | 8.501940 | -2.939562 | 0.065122 |
| stage38d | 12 | finger_control_interface_mae | 8.741311 | 8.741311 | 0.188896 | 2.12% | 8.645075 | -3.289862 | 0.037400 |
| stage38e | 4 | finger_control_interface_mae | 8.554015 | 8.554015 | 0.376193 | 4.21% | 8.454316 | -2.268469 | 0.088177 |
| stage38f | 6 | finger_control_interface_mae | 8.515643 | 8.515643 | 0.414565 | 4.64% | 8.437849 | -2.262963 | 0.101930 |
| stage38g | 8 | finger_large_delta_balanced_wrong_sign_rate | 0.337494 | 8.503219 | 0.426989 | 4.78% | 8.403708 | -2.499043 | 0.084979 |

补充：`stage38g` 的 per-finger large-delta 指标中，`finger_large_delta_pred_abs_mean = 2.155388`，`finger_large_delta_target_abs_mean = 17.387189`，说明 sign-first 后幅值仍严重不足。

### 基线量化结论

1. `stage38f` 相对 reference-only 的 interface MAE 只降低 `0.414565`，相对增益约 `4.64%`。
2. `stage38g` 略好于 `stage38f`，interface MAE 降到 `8.503219`，相对 reference-only 增益约 `4.78%`，但仍属于小幅改善。
3. `stage38g` 的 stable leakage 比 `stage38f` 低，从 `0.101930` 降到 `0.084979`，说明 sign-first 并没有直接恶化 stable 区域。
4. `stage38g` 的 selection metric 是 `finger_large_delta_balanced_wrong_sign_rate = 0.337494`，已经开始按正确问题选 checkpoint；但由于历史 `stage38f` best metrics 没有同口径 per-finger large-delta wrong-sign 字段，建议后续用当前代码重新 evaluate `stage38f` best checkpoint，得到严格可比的 wrong-sign baseline。
5. large-delta 目标幅值均值是 `17.387189`，而 `stage38g` 预测 large-delta 幅值均值只有 `2.155388`。这说明当前即使方向指标有所改善，幅值恢复仍远远不够。

阶段判断：第 1 步支持原指导意见。模型确实相对 reference 有净增益，但增益很小；当前问题不能继续靠简单提高 residual 权重解决。

## 2. dead-zone sign 诊断

目标：避免把近零 residual 噪声当成正负方向错误。

建议定义：

```text
delta_i > tau_i      -> positive
delta_i < -tau_i     -> negative
abs(delta_i) <= tau_i -> near-zero
```

`tau_i` 不建议拍脑袋，优先从 stable 区域每指 residual 噪声估计：

```text
tau_i = max(P90(abs(stable_delta_i)), 2 * std(stable_delta_i))
```

报告指标：

1. valid-interface sign accuracy；
2. large-delta wrong-sign rate；
3. positive/negative balanced wrong-sign rate；
4. per-finger wrong-sign rate；
5. near-zero 命中率。

### 已落成脚本

已新增：

```text
scripts/diagnose_per_finger_sign_oracles.py
```

该脚本一次性完成：

1. 从 stable expert-reference residual 估计 per-finger dead-zone `tau_i`；
2. 使用 dead-zone 将 target/pred delta 分成 positive / negative / near-zero；
3. 分开报告 opposite-sign 与 near-zero directional miss；
4. 同时产出四个 oracle 变体的 per-finger 指标。

运行命令示例：

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
.\.venv310\Scripts\python.exe scripts\diagnose_per_finger_sign_oracles.py `
  --project-root . `
  --stage configs\stages\stage38g_policy_fold1_per_finger_reference_delta_sign_first.yaml `
  --checkpoint runs\stage38g_policy_fold1_per_finger_reference_delta_sign_first\checkpoints\best.pt `
  --subset val `
  --num-workers 0
```

输出位置：

```text
evals/per_finger_oracles/<stage_name>/best__val/per_finger_sign_oracle_summary.json
evals/per_finger_oracles/<stage_name>/best__val/per_finger_sign_oracle_metrics.csv
```

### dead-zone 阈值结果

使用 `stage38g` 验证集、stable expert-reference residual 估计：

| Finger | stable count | P90 abs stable delta | 2 * std(abs stable delta) | tau |
|---|---:|---:|---:|---:|
| finger0 | 90 | 4.588029 | 4.701658 | 4.701658 |
| finger1 | 90 | 6.560332 | 5.737427 | 6.560332 |
| finger2 | 90 | 1.355197 | 2.862557 | 2.862557 |

注意：最初若直接用 stable control target residual 估计，三个 finger 的 `tau` 都是 0，因为 stable control residual 在标签构造中被严格置零。这个口径无法代表真实噪声。因此当前脚本改用 stable `finger_expert_forces - finger_reference_forces` 估计 dead-zone。

### stage38f / stage38g dead-zone sign 对比

| Stage | Variant | Interface MAE | Pred abs mean | Stable leakage | Directional count | Opposite rate | Near-zero pred rate | Directional miss | Large balanced opposite | Large balanced miss |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| stage38f | model | 8.515642 | 2.529075 | 0.102700 | 711 | 0.014065 | 0.817159 | 0.831224 | 0.005525 | 0.900234 |
| stage38g | model | 8.503219 | 2.224483 | 0.088648 | 711 | 0.011252 | 0.857947 | 0.869198 | 0.005525 | 0.914881 |

阶段结论：

1. 在 dead-zone 口径下，真正 opposite-sign 并不是最大问题：overall opposite rate 只有约 `1.1% - 1.4%`，large balanced opposite 约 `0.55%`。
2. 更主要的问题是 directional target 上预测落在 near-zero：`stage38f = 81.7%`，`stage38g = 85.8%`。
3. `stage38g` 相比 `stage38f` 略降 MAE 和 stable leakage，但 dead-zone sign 激活反而更保守。
4. 因此后续 sign-first 不应只压“反号”，还应显式关注“该动但没动”的 activation / near-zero miss。

## 3. oracle 诊断

优先做四个低成本实验：

| 实验 | 目的 | 解释 |
|---|---|---|
| oracle sign + learned magnitude | 判断方向是否主瓶颈 | 若明显改善，方向是主要卡点 |
| learned sign + oracle magnitude | 判断 sign 错误对最终控制的伤害 | 若仍差，sign 分类非常严重 |
| oracle gate + learned residual | 排除 gate-residual 耦合 | 若 residual 幅值上升，说明 gate 在压梯度 |
| reference-only baseline | 量化净增益 | 已完成第一版，后续可扩展到更多 split |

### 已落成的四个 oracle 变体

脚本内固定产出以下五行，其中 `model` 是原模型，其余四行是 oracle 实验：

| Variant | 定义 |
|---|---|
| model | 原模型输出 |
| reference_only | `pred = reference` |
| oracle_sign_learned_magnitude | `delta = sign(target_delta, tau) * abs(pred_delta)`，使用模型幅值、oracle 方向 |
| learned_sign_oracle_magnitude | `delta = sign(pred_delta, tau) * abs(target_delta)`，使用模型方向、oracle 幅值 |
| oracle_gate_learned_residual | `pred = reference + hard_interface_gate * pred_delta`，使用 oracle gate、模型 residual |

### stage38f oracle 结果

| Variant | Interface MAE | Delta MAE | Pred abs mean | Stable leakage | Directional miss | Large balanced miss |
|---|---:|---:|---:|---:|---:|---:|
| model | 8.515642 | 8.437848 | 2.529075 | 0.102700 | 0.831224 | 0.900234 |
| reference_only | 8.930207 | 8.930207 | 0.000000 | 0.000000 | 1.000000 | 1.000000 |
| oracle_sign_learned_magnitude | 8.045020 | 7.594269 | 1.348364 | 0.000000 | 0.817159 | 0.894710 |
| learned_sign_oracle_magnitude | 7.834815 | 7.647990 | 1.523300 | 0.000000 | 0.831224 | 0.900234 |
| oracle_gate_learned_residual | 8.437848 | 8.437848 | 2.529075 | 0.000000 | 0.831224 | 0.900234 |

### stage38g oracle 结果

| Variant | Interface MAE | Delta MAE | Pred abs mean | Stable leakage | Directional miss | Large balanced miss |
|---|---:|---:|---:|---:|---:|---:|
| model | 8.503219 | 8.403708 | 2.224483 | 0.088648 | 0.869198 | 0.914881 |
| reference_only | 8.930207 | 8.930207 | 0.000000 | 0.000000 | 1.000000 | 1.000000 |
| oracle_sign_learned_magnitude | 8.123202 | 7.731464 | 1.203570 | 0.000000 | 0.857947 | 0.909357 |
| learned_sign_oracle_magnitude | 7.935971 | 7.829272 | 1.315005 | 0.000000 | 0.869198 | 0.914881 |
| oracle_gate_learned_residual | 8.403707 | 8.403708 | 2.224483 | 0.000000 | 0.869198 | 0.914881 |

### oracle 诊断结论

1. `oracle_gate_learned_residual` 只带来约 `0.08 - 0.10` 的 interface MAE 改善，说明 gate 耦合存在，但不是最大瓶颈。
2. `oracle_sign_learned_magnitude` 能把 stage38f 从 `8.515642` 降到 `8.045020`，说明方向修正有价值；但因为模型幅值仍很小，改善有限。
3. `learned_sign_oracle_magnitude` 能把 stage38f 降到 `7.834815`，说明如果幅值正确，当前少量非零方向预测仍能产生更大收益；幅值/激活不足是强瓶颈。
4. dead-zone 后的关键不是“大量反号”，而是“该输出明显 residual 时仍输出 near-zero”。后续 loss/metric 应增加 activation-aware sign 目标，而不仅是 `pred * target < 0` 的 wrong-sign。
5. `stage38g` 在旧 wrong-sign 选择指标下略有收益，但在 dead-zone activation 口径下更保守，因此 `stage38h` 若只是 magnitude follow-up，必须观察它是否能降低 near-zero miss。

## 4. stage38g sign-first

已有配置：

```text
configs/stages/stage38g_policy_fold1_per_finger_reference_delta_sign_first.yaml
```

目标不是优先压 MAE，而是优先压：

```text
finger_large_delta_balanced_wrong_sign_rate
```

成功标准：

1. balanced wrong-sign 明显下降；
2. `finger_large_delta_balanced_mae` 不明显恶化；
3. `stable_leakage_mean` 不明显高于 stage38f；
4. `finger_control_interface_mae` 不需要立刻大降，但不能明显退化。

## 5. stage38h magnitude follow-up

已有配置：

```text
configs/stages/stage38h_policy_fold1_per_finger_reference_delta_sign_then_magnitude.yaml
```

关键要求：必须从 `stage38g` best checkpoint warm-start，否则不能称为真正的 sign-first -> magnitude follow-up。

成功标准：

1. `finger_control_interface_mae < 8.503219`，至少应优于当前 `stage38g`；
2. wrong-sign 不反弹；
3. `pred_delta_abs_mean / target_delta_abs_mean` 明显上升；
4. `stable_leakage_mean` 不明显高于 `stage38f` 的 `0.101930`。

## 6. per-finger direction/magnitude 解耦 head

如果 `stage38g/38h` 只有小幅改善，新增：

```text
state_residual_per_finger_sign_specific
```

每根手指输出：

```text
direction_logits_i
pos_magnitude_i = softplus(pos_raw_i)
neg_magnitude_i = softplus(neg_raw_i)
delta_i = P(pos_i) * pos_magnitude_i - P(neg_i) * neg_magnitude_i
```

建议阶段：

```text
stage39a_per_finger_sign_specific_sign_first
stage39b_per_finger_sign_specific_magnitude_followup
```

实现优先复用已有 `state_residual_sign_specific` 的非 per-finger 结构。

## 7. local per-finger tactile branch

如果方向仍不稳，说明 shared global context + finger embedding 不足以判断每指局部状态，应新增：

```text
h_i_local = FingerEncoder(X_i)
```

其中 `X_i` 是第 `i` 根手指的局部触觉时序。

最终每指 residual 输入建议为：

```text
[h_global, h_i_local, reference_force_i, finger_embedding_i]
```

全局分支回答“当前整体处于什么介质和任务状态”，局部分支回答“这一根手指当前应该增力还是减力”。

## 下一步建议

1. 将后续 sign 指标从单纯 opposite-sign 扩展为 activation-aware sign：同时报告 opposite-sign、near-zero pred、directional miss。
2. 若继续做 sign-first，应把 near-zero miss 纳入 selection/tie-breaker，否则模型可能通过更保守的 near-zero 输出获得较低 leakage 但不解决 residual。
3. `stage38h` 仍值得跑，但必须从 `stage38g` best checkpoint warm-start，并重点观察 `pred_delta_abs_interface_mean` 和 near-zero miss 是否改善。
4. 如果 `stage38h` 仍无法把 pred abs mean 从 `2.x` 明显拉近 target `8.93`，优先进入 direction/magnitude 解耦 head，而不是继续提高 residual 权重。
5. 若 direction/magnitude 解耦后 near-zero miss 仍高，说明全局表示不足，应推进 local per-finger tactile branch。

## 2026-07-12 进展同步：38h / 38i / 38j

### stage38h 结果

`stage38h_policy_fold1_per_finger_reference_delta_sign_then_magnitude` 已完成。

| 指标 | stage38g best | stage38h best |
|---|---:|---:|
| best epoch | 8 | 10 |
| finger_control_interface_mae | 8.503219 | 8.425064 |
| finger_large_delta_balanced_mae | 15.830053 | 15.307127 |
| finger_large_delta_balanced_wrong_sign_rate | 0.337494 | 0.338845 |
| finger_large_delta_balanced_score | 49.579408 | 49.191622 |
| finger_large_delta_pred_abs_mean | 2.155388 | 2.758446 |
| stable_leakage_mean | 0.084979 | 0.105321 |

结论：38h 相比 38g 明显改善 MAE 和大 delta 幅值，但 wrong-sign 没有继续下降，stable leakage 略升。方向上证明“magnitude follow-up”有效，但仍没有解决大 delta 欠激活。

### stage38i 结果

`stage38i_policy_fold1_per_finger_reference_delta_balanced_score_magnitude_guard` 已完成。该阶段使用 `finger_large_delta_balanced_score` 作为 selection metric。

| 指标 | stage38h best | stage38i best | stage38i latest |
|---|---:|---:|---:|
| epoch | 10 | 4 | 8 |
| finger_control_interface_mae | 8.425064 | 8.430894 | 8.379626 |
| finger_delta_interface_mae | 8.363445 | 8.419597 | 8.368854 |
| finger_large_delta_balanced_score | 49.191622 | 45.828175 | 46.367234 |
| finger_large_delta_balanced_mae | 15.307127 | 14.965462 | 14.920237 |
| finger_large_delta_balanced_wrong_sign_rate | 0.338845 | 0.308627 | 0.314470 |
| finger_large_delta_pred_abs_mean | 2.758446 | 2.915915 | 3.082917 |
| stable_leakage_mean | 0.105321 | 0.104920 | 0.112871 |

结论：

1. 38i 的 loss/selection 调整有效，large balanced score、large balanced MAE、large pred abs mean 均优于 38h。
2. 38i 的 `best.pt` 由 balanced score 选择在 epoch 4，但 `latest.pt` 在 MAE、幅值和 oracle 上限上更好。
3. 后续 warm-start 应优先使用 `stage38i ... latest.pt`，而不是 `best.pt`。

### 38i oracle 诊断

| Checkpoint | model interface MAE | oracle sign MAE | oracle magnitude MAE | large hit | large near-zero | large balanced miss |
|---|---:|---:|---:|---:|---:|---:|
| stage38h best | 8.425064 | 7.895657 | 7.156311 | 0.227545 | 0.758483 | 0.811519 |
| stage38i best | 8.430893 | 7.843459 | 7.189045 | 0.235529 | 0.746507 | 0.768323 |
| stage38i latest | 8.379624 | 7.781322 | 7.017883 | 0.261477 | 0.722555 | 0.761447 |

per-finger large-delta hit rate：

| Checkpoint | finger0 | finger1 | finger2 |
|---|---:|---:|---:|
| stage38h best | 0.162602 | 0.040698 | 0.422330 |
| stage38i best | 0.292683 | 0.046512 | 0.359223 |
| stage38i latest | 0.252033 | 0.046512 | 0.446602 |

结论：38i 改善了 finger0/finger2 的激活，但 finger1 仍几乎不动；大 delta near-zero 仍超过 72%。这说明仅靠 signed residual 单头继续调 loss 的收益已经有限，应进入结构改造。

### stage38j 结构改造

已新增结构：

```text
state_residual_per_finger_sign_specific
```

每个 finger 独立输出：

```text
finger_residual_direction_logit_neg_i
finger_residual_direction_logit_pos_i
finger_force_interface_delta_pos_magnitude_i = softplus(pos_raw_i)
finger_force_interface_delta_neg_magnitude_i = softplus(neg_raw_i)
delta_i = residual_output_scale * (P(pos_i) * pos_magnitude_i - P(neg_i) * neg_magnitude_i)
```

已新增配置：

```text
configs/stages/stage38j_policy_fold1_per_finger_sign_specific_direction_magnitude.yaml
```

38j 设计要点：

1. 从 `stage38i latest.pt` warm-start，保留已学到的 base/context/hidden 表征。
2. 新增 direction / pos magnitude / neg magnitude 输出层；旧单 residual 输出层不再使用。
3. `residual_output_scale = 20.0`，与 `delta_normalization_scale = 20.0` 和 `sign_magnitude_scale = 20.0` 对齐。
4. primary selection 回到 `finger_control_interface_mae`，把 `finger_large_delta_balanced_score` 放入 tie-breaker，避免再次出现 38i best 选早的问题。
5. 对新输出层设置更高学习率 multiplier：pos/neg magnitude 输出层 `8x`，direction 输出层 `10x`。

38j 成功信号：

| 指标 | 期望 |
|---|---|
| finger_control_interface_mae | 优于 38i latest 的 8.379626 |
| finger_large_delta_pred_abs_mean | 高于 3.08，越接近 target 17.39 越好 |
| large near-zero pred rate | 低于 72.3% |
| finger1 large hit rate | 明显高于 4.65% |
| stable_leakage_mean | 尽量不超过 0.12 |

如果 38j 后 finger1 仍无明显激活，则应推进下一阶段 local per-finger tactile branch，而不是继续堆全局 loss 权重。

### stage38j 结果与 stage38k 计划

`stage38j_policy_fold1_per_finger_sign_specific_direction_magnitude` 已完成。训练在 epoch 9 早停，best checkpoint 为 epoch 5。

| 指标 | stage38i latest | stage38j best | stage38j latest |
|---|---:|---:|---:|
| finger_control_interface_mae | 8.379626 | 8.100162 | 8.130840 |
| finger_delta_interface_mae | 8.368854 | 8.299366 | 8.574956 |
| finger_large_delta_balanced_score | 46.367234 | 36.434607 | 34.850908 |
| finger_large_delta_balanced_wrong_sign_rate | 0.314470 | 0.228397 | 0.217517 |
| finger_large_delta_pred_abs_mean | 3.082917 | 4.060528 | 4.577742 |
| stable_leakage_mean | 0.112871 | 0.142007 | 0.150748 |

38j oracle 诊断：

| Checkpoint | model interface MAE | oracle sign MAE | oracle magnitude MAE | large hit | large near-zero |
|---|---:|---:|---:|---:|---:|
| stage38i latest | 8.379624 | 7.781322 | 7.017883 | 0.261477 | 0.722555 |
| stage38j best | 8.100163 | 7.508820 | 6.521337 | 0.383234 | 0.586826 |
| stage38j latest | 8.130839 | 7.415576 | 6.569492 | 0.393214 | 0.544910 |

per-finger large-delta hit rate：

| Checkpoint | finger0 | finger1 | finger2 |
|---|---:|---:|---:|
| stage38i latest | 0.252033 | 0.046512 | 0.446602 |
| stage38j best | 0.414634 | 0.075581 | 0.621359 |
| stage38j latest | 0.487805 | 0.127907 | 0.558252 |

结论：

1. 38j 的 sign-specific 结构有效，显著降低 near-zero，并打开 oracle 上限。
2. `best.pt` 的主 MAE 最好，适合作为下一步 warm-start；`latest.pt` 更激进，但 stable leakage 更高。
3. finger1 仍是主要瓶颈，38k 不再继续堆全局激活，而是加入 per-finger loss reweight。

已新增配置：

```text
configs/stages/stage38k_policy_fold1_per_finger_sign_specific_finger1_guard.yaml
```

38k 设计：

1. 从 `stage38j ... best.pt` warm-start。
2. 保留 `state_residual_per_finger_sign_specific` 结构。
3. 新增 `policy_loss.finger_loss_weights: [1.0, 2.5, 1.0]`，只对 large-delta 方向、sign、magnitude 相关 loss 按 finger 加权，重点推 finger1。
4. 将 `residual_stable_zero_weight` 提到 `2.5`、`residual_non_interface_weight` 提到 `0.5`，抑制 38j latest 出现的 stable leakage 上升。
5. selection 继续使用 `finger_control_interface_mae`，tie-breaker 中优先参考 `stable_leakage_mean`，避免选到过激 checkpoint。

38k 成功标准：

| 指标 | 目标 |
|---|---|
| finger_control_interface_mae | 尽量保持在 8.10 附近，至少不明显差于 38j latest 的 8.13 |
| large near-zero pred rate | 低于 38j best 的 58.7%，理想接近或低于 50% |
| finger1 large hit rate | 明显高于 38j latest 的 12.8%，第一目标 20%+ |
| stable_leakage_mean | 回落到 0.14 附近，至少不高于 0.16 |
| oracle magnitude MAE | 不劣于 38j best 的 6.52 |
