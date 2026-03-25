# 跨介质视触抓取项目

这个仓库按照论文设计说明与实施规格，搭了一个可扩展的 Python/PyTorch 工程骨架，覆盖了以下主链路：

- 对象级/样本级标注规范化
- `samples.csv` / `windows.csv` 生成
- object-level unseen 与 mechanism split 配置
- 视频窗口 + 触觉窗口的序列级 Dataset
- `open_clip` 视觉骨干 + q/v 语义 LoRA 映射
- 高频 / 低频触觉解耦、1D CNN 高频编码、固定长度重采样、train-split 标准化与 object-aware batch 训练
- 递归介质信念、多属性推断、FiLM 条件化策略头
- `L_clip` / `L_inv` / `L_med` / `L_attr` / `L_pol` 联合训练框架
- 标注校验、时间线审计、在线 replay stub 与 sidecar cache

## 目录

```text
configs/
  data/
  model/
  train/
  stages/
data/
  annotations/
  raw/
  splits/
scripts/
src/cmg/
```

## 快速开始

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e .
python scripts/validate_annotations.py --project-root .
python scripts/preprocess.py --project-root .
python scripts/train.py --project-root . --stage configs/stages/stage1_perception_fold1.yaml
python scripts/train.py --project-root . --stage configs/stages/stage2_alignment_fold1.yaml --init-from runs/stage1_perception_fold1/checkpoints/stage1_best_joint_f1.pt
python scripts/train.py --project-root . --stage configs/stages/stage3_policy_fold1.yaml --init-from runs/stage2_alignment_fold1/checkpoints/stage2_best_joint_f1.pt
python scripts/train.py --project-root . --stage configs/stages/stage4_joint_fold1.yaml --init-from runs/stage3_policy_fold1/checkpoints/stage3_best_interface_mae.pt
python scripts/evaluate.py --project-root . --stage configs/stages/stage4_joint_fold1.yaml --checkpoint runs/stage4_joint_fold1/checkpoints/stage4_best_interface_mae.pt --subset test
python scripts/visualize_timeline.py --project-root . --sample-id S0001
python scripts/online_stub.py --project-root . --stage configs/stages/stage4_joint_fold1.yaml --checkpoint runs/stage4_joint_fold1/checkpoints/stage4_best_interface_mae.pt --sample-id S0001
powershell -ExecutionPolicy Bypass -File scripts/start_stage1_formal.ps1
powershell -ExecutionPolicy Bypass -File scripts/watch_stage1_formal.ps1 -Follow
```

`--resume` 现在只用于同一 `run/stage` 的断点续训；跨阶段衔接必须使用 `--init-from`，它只恢复 `model.state_dict()`。当前推荐的正式链路是：Stage1 冻结视觉 backbone 且不注入 LoRA，Stage2 / Stage4 再打开视觉 LoRA；因此 `--init-from` 在 Stage1 -> Stage2 / Stage4 时会自动重映射 OpenCLIP attention 权重，并允许目标模型新增 `lora_*` 参数缺失。

## 默认数据处理

- 视频：固定 ROI 裁剪 `(460, 28, 1024, 1024)`，再 resize 到 `224x224`
- 视频帧：每窗口均匀采样 `8` 帧，附带 `frame_mask`
- 触觉：按 `EMA` 做 AC/DC 分解，expert 路径固定使用 `normal_sign_table` + z 轴 signed-sum
- 触觉窗口：固定重采样到 `26` 点
- 触觉标准化：使用同一 split 的 train subset 统计量做 channel-wise 标准化，缓存到 `data/processed/cache/`
- 低覆盖率窗口：默认过滤 `valid_ratio_video < 0.8` 或 `valid_ratio_tactile < 0.8`
- 细粒度 sidecar cache：每个窗口额外写入 `data/processed/cache/<sample_id>/<window_id>.npz`

## 预处理输出

预处理脚本会生成：

- `data/processed/samples.csv`
- `data/processed/windows.csv`
- `data/processed/stats/dataset_summary.json`
- `data/processed/cache/<sample_id>/<window_id>.npz`

其中 `windows.csv` 已按规格包含：

- `phase_label`
- `is_stable_mask`
- `tail_type`
- `valid_ratio_video`
- `valid_ratio_tactile`
- `video_frame_indices_json`
- `tactile_start_idx` / `tactile_end_idx`
- `sidecar_cache_path`

## Split 说明

- `data/splits/split_seen_debug_v1.yaml`
  - seen-object 快速 sanity check
- `data/splits/split_unseen_fold{1,2,3}_v1.yaml`
  - 主任务对象的 3-fold object-level unseen split
- `data/splits/split_unseen_online_benchmark_v1.yaml`
  - rollout / online benchmark 默认 held-out fold
- `data/splits/probe_mechanism_v1.yaml`
  - 规则体/机制验证对象池

## 训练阶段

目前提供了以下阶段配置：

- `stage0_mechanism.yaml`
  - 用 mechanism object pool 做表征和状态预训练
- `stage1_perception_fold1.yaml`
  - 训练介质识别与对象属性分支，视觉 backbone 冻结且不注入 LoRA，canonical 指标为 `joint_score`
- `stage2_alignment_fold1.yaml`
  - 在 Stage 1 基础上重新打开视觉 LoRA，并加入 `L_clip` 与 `L_inv`
- `stage3_policy_fold1.yaml`
  - 保留视觉 LoRA 结构但跟随 `visual_encoder` 一起冻结，聚焦策略回归
- `stage4_joint_fold1.yaml`
  - 解冻全模型并重新参与视觉 LoRA 联合微调，canonical 指标为 `interface_mae`

训练器支持：

- stage 化 checkpoint 命名与 canonical best
- `--resume` 同阶段断点续训
- `--init-from` 跨阶段权重初始化
- object-aware batch sampler
- LoRA / 新模块分组学习率
- cosine warmup 调度
- AMP
- 梯度裁剪
- early stopping
- `metrics.csv` + TensorBoard 日志

## Windows 长跑启动

如果你希望在自己的 PowerShell 里启动正式 Stage1，并且关闭 Codex 或当前终端后训练仍继续运行，使用：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/start_stage1_formal.ps1
```

这个脚本会启动一个独立的 PowerShell host 进程来承载 Stage1。默认会优先从 `runs/stage1_perception_fold1/checkpoints/stage1_latest.pt` 续训；如果你想强制从头开始，可加 `-StartFresh`。关闭 Codex 或启动它的那个 PowerShell，不会影响这个独立训练宿主。日志会写到：

- `runs/stage1_perception_fold1/hosted_stage1.stdout.log`
- `runs/stage1_perception_fold1/hosted_stage1.stderr.log`
- `runs/stage1_perception_fold1/hosted_stage1.launch.json`

查看进度：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/watch_stage1_formal.ps1 -Follow
```

停止并移除后台任务：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/stop_stage1_formal.ps1
```

## 审计与调试脚本

- `scripts/validate_annotations.py`
  - 校验枚举字段、时间先后关系、`t_start=0` 与 fail 语义
- `scripts/visualize_timeline.py`
  - 生成单样本时间线图，服务 `sync_offset_sec` 审计闭环
- `scripts/online_stub.py`
  - 离线 replay 样本的在线推理 stub，输出 `F_des / F_meas / p_medium` JSONL 日志





