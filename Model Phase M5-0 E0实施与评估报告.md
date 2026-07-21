# Model Phase M5-0 E0 实施与评估报告

## 1. 结论

M5-0 E0 已实现并完成正式运行，结果为 **PASS**。

- 使用 canonical W2A checkpoint：`runs/stage38j_f20_v3_causal/checkpoints/best.pt`
- checkpoint SHA256：`6360dd44fd1e080c60a0a8c87f497b599a7880ea908e852d1e2f87fbd84dcb84`
- 评估范围严格限定为 stable-only Val；未评估 Test，OBJ004/OBJ007 保持冻结
- direction-conditioned 模型加载 471/480 个张量；仅缺失预期的 9 个 direction 新参数
- direction adapter 零初始化生效：W2A/A2W 方向翻转对七类核心输出的最大差值为 `0.0`
- 三组评估及 provenance、缓存、哈希与跨报告一致性检查全部通过

E0 结果经确认后，Stage39a–39d 已分别写入 evidence、baseline 与 guard，并通过正式训练解锁校验。Stage39 base 模板继续保持 `blocked_pending_e0_w2a_baseline`，防止未配置完整的派生阶段被误启动。

## 2. E0 口径

| 项目 | 口径 |
|---|---|
| 数据版本 | `bidirectional_v1` / `bidirectional-causal-v4` |
| split | `split_unseen_fixed_test_obj004_obj007_v1` |
| subset | Val |
| trial_result | stable only |
| Val 对象 | OBJ005、OBJ010、OBJ014 |
| 样本数 | 83（W2A 44；A2W 39） |
| 窗口数 | 40,889（W2A 19,128；A2W 21,761） |
| checkpoint 配置 | 使用当前双向数据/模型配置，不回退到 archived W2A run config |
| 方向参数 | 新增 direction 参数保持零初始化，不进行训练 |
| Test | 禁止访问和评估 |

## 3. 核心结果

| Cohort | Interface Medium F1 ↑ | Finger Control Interface MAE ↓ | Finger Delta Interface MAE ↓ |
|---|---:|---:|---:|
| W2A | 0.655166 | 7.069269 | 7.010270 |
| A2W zero-shot | 0.019512 | 5.253923 | 6.354967 |
| Direction macro | 0.337339 | 6.161596 | 6.682619 |

双向 pooled 指标为：Interface Medium F1 `0.501701`、Finger Control Interface MAE `6.522813`。pooled 指标受两个方向的观察数影响，不用于 W2A retention guard。

## 4. Reference-only 对照

Reference-only 定义为：在 delta-supervised Interface 观察上，直接把 `finger_reference_forces` 作为控制力预测。

| Cohort | Reference-only MAE ↓ | Canonical model E0 MAE ↓ | 观察 |
|---|---:|---:|---|
| W2A | 8.136029 | 7.069269 | canonical model 优于 reference-only |
| A2W | 5.237994 | 5.253923 | zero-shot 与 reference-only 基本持平，略差 0.015929 |
| Direction macro | 6.687012 | 6.161596 | canonical model macro 更优 |

## 5. 结果解读

1. W2A 保持良好。canonical checkpoint 在新 split、新 Val 对象和新处理口径下仍有可用的 Medium 与 Policy 基线。
2. A2W Policy zero-shot 没有崩溃，Control MAE 与 reference-only 接近；这主要说明现有 reference-force 基座能提供合理起点，并不等价于模型已学会 A2W。
3. A2W Interface Medium F1 仅为 `0.019512`，是 E0 最明确的短板。后续 direction-conditioned 训练必须优先学习反向介质转移的时序与 Interface 判别。
4. Direction macro 明显低于 W2A，证明不能再使用 pooled 指标替代方向显式选择与 guard。

## 6. 建议写回的 W2A retention guard

### Policy 阶段（Stage39a/39c/39d）

- metric：`finger_control_interface_mae_w2a`
- mode：`min`
- baseline：`7.069268751150515`
- 最大相对退化：`5%`
- guard threshold：`7.422732188708041`

### Medium 阶段（Stage39b）

- metric：`medium_f1_interface_w2a`
- mode：`max`
- baseline：`0.6551664654131447`
- 最大相对退化：`5%`
- guard threshold：`0.6224081421424874`

## 7. 实施内容

- 新增 E0 专用 stage：`configs/stages/stage39e0_warmstart_baseline.yaml`
- 新增一体化缓存、评估与报告入口：`scripts/run_m5_e0.py`
- 扩展评估加载接口，支持显式 architecture-delta 白名单及跳过冗余 pretrained 初始化：`src/cmg/evaluation.py`
- 优化密集滑窗视频帧的顺序解码：`src/cmg/data/video.py`
- 生成 canonical Val 视觉缓存：`data/processed/cache/visual_stage39e0_val_canonical`

缓存审计：83/83 样本文件完整；S0068 前两个窗口的缓存相对原始 `VisualEncoder` 直算结果最大绝对差为 `3.6656856536865234e-06`，低于 `2e-4` 阈值。

## 8. 校验结果

| 校验 | 结果 |
|---|---|
| checkpoint 加载边界（471/480；仅 9 个 direction key 缺失） | PASS |
| direction 零作用 | PASS，最大差值 0.0 |
| stable-only Val cohort | PASS，83 samples / 40,889 windows |
| Test 未评估 | PASS |
| 缓存完整性与直算等价性 | PASS |
| W2A 单向报告与双向报告 W2A 指标一致 | PASS |
| A2W 单向报告与双向报告 A2W 指标一致 | PASS |
| direction macro complete | PASS |
| manifest/report SHA256 provenance | PASS |
| M4 回归 | PASS，5/5 checks；43 function tests，0 failures |

## 9. 产物

- 汇总：`evals/stage39e0_warmstart_baseline/e0_summary.json`
- W2A：`evals/stage39e0_warmstart_baseline/e0_w2a_val.json`
- A2W：`evals/stage39e0_warmstart_baseline/e0_a2w_val.json`
- 双向：`evals/stage39e0_warmstart_baseline/e0_bidirectional_val.json`
- 校验快照：`data/processed/stats/model_phase_m5_e0_validation.json`
- 缓存 manifest：`data/processed/cache/visual_stage39e0_val_canonical/manifest.json`

## 10. 解锁状态与下一步

已完成：

1. Policy/Medium 两类 baseline 已写回 Stage39a–39d；
2. 四个正式阶段均已更新为 `ready` 并启用 guard；
3. 四阶段 strict preflight 全部通过；
4. M4 回归为 5/5 checks、43/43 function tests。

解锁校验产物：`data/processed/stats/model_phase_m5_training_unlock_validation.json`。

下一步为启动 Stage39a direction-adapter warm-up；正式训练尚未启动。

### Stage39a 首次启动修复

首次正式启动暴露了 cuDNN 限制：冻结且 eval-mode 的 `medium_head` GRU 无法把 Medium loss 反传给上游 `medium_direction_adapter`。现已将 Stage39a 配置为 `medium_head` 参数保持冻结、训练批次中保留 training-mode 前向；其他冻结模块继续锁定 eval，验证批次仍统一使用 eval。CUDA 专项 backward 回归及完整 M4/M5 解锁回归均已通过。
