# Implement — MILD symw 五折 M/W 消融

## 前置

- [ ] S0.1 任务已 `start`（status=in_progress），experiment.md 跑前登记 S1/S2 Run 块
- [ ] S0.2 GPU 空闲确认（nvidia-smi）；基线五折 config/history/ckpt 可读核验

## S1 冒烟（~10 min）

- [ ] S1.1 `run_ablation_mw.py` 驱动脚本就绪（含 CLI 对拍守卫）
- [ ] S1.2 冒烟跑：fold0 × A3 臂 × `--epochs 2 --limit-subjects 2` → 跑通 + 守卫通过（W 零/log_sigma 冻结/A_eff≡0 验证）→ 结果登记 experiment.md（标注 smoke 不产正式数）

## S2 正式训练批（15 跑串行，~3 h）

- [ ] S2.1 后台启动 15 跑批（nohup；fail-fast；stdout 逐跑落 `run_{arm}_fold{k}_stage2.log`）
- [ ] S2.2 批完成核验：15 × 四件套齐（ckpt/history/config/log）；CLI 对拍 15/15
- [ ] S2.3 `ablation_guards.py` ×15 → guards JSON 15/15 通过（AC2）
- [ ] S2.4 experiment.md 回填 15 跑结果（best val loss/datafit/best epoch）

## S3 M2 幅度抽查（15 × ~29 min ≈ 7 h）

- [ ] S3.1 `quality_check_mild.py` ×15（同基线口径：默认 3 test 被试 × 30 trial × K=32、`--flow-exit-gnn`）→ quality_meta.json ×15
- [ ] S3.2 experiment.md 回填（各折 std_ratio full/trimmed）

## S4 M3 FC + M4 A_eff↔GC（各 ~2–5 h）

- [ ] S4.1 `fc_metric_mild_ablation.py` 就绪 + 自冒烟（fold0 × A1 单折对拍镜像口径字段面）
- [ ] S4.2 M3 ×15 → fc JSON ×15；experiment.md 回填（组 r/被试 r±sd/t/p）
- [ ] S4.3 `aeff_gc_corr.py` 就绪（含 A3 NaN(by-design) 守卫 + refs SHA 全验）
- [ ] S4.4 M4 ×15 + A0 ×5（基线五折 ckpt 现提取，与 result 注册 SHA 对拍）→ JSON ×20
- [ ] S4.5 experiment.md 回填（spearman/pearson × raw/abs × aligned/flipped × 主/副口径）

## S5 汇总（~30 min）

- [ ] S5.1 `ablation_summary.py`：M1–M4 四臂五折 mean±sd + 逐折 Δ vs A0 + 损失曲线并置（loss_curves.npz/png，R4/要求 7）
- [ ] S5.2 `summary/summary_5fold.json` + `README.md`（表 1 主表/表 2 逐折/结论段：Q-A/Q-B/Q-C 回答）
- [ ] S5.3 experiment.md 汇总 Run 块回填结论

## S6 收尾

- [ ] S6.1 trellis-check（Agent）全量：AC1–AC5 逐条 + experiment.md 无空块 + 产物路径隔离
- [ ] S6.2 trellis-update-spec：消融口径（γ 置零协议 + M4 A_eff↔GC 对照协议）沉淀 evaluation-calibers / experiment-log
- [ ] S6.3 progress.md 当前状态同步 + 里程碑
- [ ] S6.4 批量 commit（工作提交 → 归档/日志提交）

## 验证命令备忘

```bash
# S1 冒烟
python3 models/latent-flow-v1/run_ablation_mw.py --smoke
# S2 正式批（后台）
nohup python3 models/latent-flow-v1/run_ablation_mw.py --stage2-batch > outputs/10-30-symw-ablation-mw/batch.log 2>&1 &
# S2 守卫
python3 models/latent-flow-v1/ablation_guards.py --all
# S3 单折示例（窗口旗标必传：默认 1500/1125/375 会静默跑错口径；subjects 逐折钉死 = 基线同折 triad）
python3 models/latent-flow-v1/quality_check_mild.py --fold 0 --ckpt outputs/10-30-symw-ablation-mw/run_noM_fold0/checkpoints/fold0_mild_flow_best_graph_mexhat.pt --flow-exit-gnn --window-len 500 --obs-len 375 --forecast-len 125 --subjects 1,7,21 --out-dir outputs/10-30-symw-ablation-mw/run_noM_fold0/quality
# 逐折 --subjects：fold0=1,7,21  fold1=2,12,22  fold2=3,6,23  fold3=9,13,29  fold4=4,15,24（与基线 quality_meta 一致，已核）
# S4
python3 models/latent-flow-v1/fc_metric_mild_ablation.py --fold 0 --arm noM --ckpt <...>
python3 models/latent-flow-v1/aeff_gc_corr.py --fold 0 --arm noM --ckpt <...>
# S5
python3 models/latent-flow-v1/ablation_summary.py
```

## 回滚点

- S2 前删 `outputs/10-30-symw-ablation-mw/` 即回滚；脚本为新增文件不触既有代码。
- 任一守卫失败 → 停批修设计（fail-fast），已跑合法单跑可保留复用。
