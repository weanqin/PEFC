# implement — NI-MSE-50-q0.90 执行（θ\* = v5-fmt-01）

> 权威参数源：ni-mse-design.md v1.5（经本任务 design.md §1 实例化落定）。三期断点续跑；每期产物按 (stage, subject[, c]) 命名。

## S1 基线登记

- [ ] S1.1 git 快照基线（HEAD/脏路径清单）+ θ\* 与 refs 只读指纹登记（ckpt SHA256、refs 目录 SHA256SUMS 复核）
- 验证：experiment.md Run 000 块登记（跑前契约）

## S2 结构落地（`models/ni-mse-v1/`）

- [ ] S2.1 `_blas_guard.py` + `thostar.py`（装载契约：**先预占 sys.modules[config/model/graph] 再 import sample** + 七项 `__file__` 断言 + config_hash pin + EMA shadow 回灌 + `gen_lowpass_hz=None`）
- [ ] S2.2 `windows.py`（loader + SeedSequence 窗子集（spawn_key 整数映射，design §1b-2）+ ⊂ 断言）+ `noise.py`（P 算子 + ξ/η 双种子树）+ `stages.py`（null/perturb/R2 同批断言/Ŷ_ref 落盘/失败标记文件）+ `report.py` + `run.py`（CLI 三期，manifest 原子写）
- [ ] S2.3 单测 `tests/test_ni_mse.py` 七项（P≡X 恒等/窗子集确定性/τ_b 池化单点/refs 映射/R2 GPU 前向强形式/装载遮蔽契约/**sampler.gen_lowpass_hz is None**）
- 验证：`cd models/ni-mse-v1 && python3 -m pytest tests/ -q` 全绿；回归 v5-fmt tests 11 passed 不变（只读零改动复核）

## S3 冒烟（`ni-mse-v5fmt-smoke`，材料目录非主产物）

- [ ] S3.1 1 例 × W_null=8 × W_pert=4 × c∈{0.5} × K=4 × K_null=4 全链路小规模跑通（R2 断言在真 θ\* 上通过；NaN/Inf 无；τ_b 形状 (62,)）
- [ ] S3.2 耗时速率实测登记（外推主 run 预算 vs 设计 1.5 h 对照）
- 验证：experiment.md Run 001 块（跑前登记 + 跑后补记）；冒烟产物落 `outputs/10-20-ni-mse-v5fmt-f0/ni-mse-v5fmt-smoke/`

## S4 主 run `ni-mse-v5fmt-01`（启动确认沿 v5-main 先例——执行期用户确认项）

- [ ] S4.1 null 期（10 例 × 200 窗 × B_ref 64 + K_null 64 → τ_b + split-half + Q95 旗标）
- [ ] S4.2 perturb 期（10 例 × 50 窗 × 62 源 × c∈{0.05,0.1,0.3,0.5} × K 32 → R̄；R2 每例首窗内嵌；R4 退化窗处置）
- [ ] S4.3 report 期（S/G 矩阵逐被试+组级 + refs 相关四件套 + 距离偏 + flip + 阴性对照 + R5/R6 旗标 + manifest 完整化）
- 验证：`outputs/10-20-ni-mse-v5fmt-f0/ni-mse-v5fmt-01/{null,perturb,report,manifest.json,logs/}` 齐全；experiment.md Run 002 块全程契约；R1–R6 处置结论逐条登记
- 门：R2 必须通过（失败即中止排查，不得带病续跑）；R3 >10% 按 design §5 加密重跑并双登记

## S5 判读汇总

- [ ] S5.1 判读四件套 + 结构预测（design §4 六条）写入 experiment.md Run 003 块（数值表 + 如实判读，正负结果同权登记）
- [ ] S5.2 与 v4a-fmt 旧路径（10-17 任务 AC6 未获结构收敛）对照讨论一段（θ\* 不同、路径不同，只作背景非等价比较）

## S6 质检与收尾

- [ ] trellis-check（全范围：AC1–AC7 逐条 + 只读契约 git status 复核 + refs/θ\* 校验和复验）
- [ ] spec 回填（3.3）：`context.md` Study II 行登记 NI-MSE 路径与代码位置；若判读有流程教训 → perturb-ec-pipeline.md 或新 spec 增补
- [ ] 提交（3.4）+ 归档；journal 记录

## 回滚点

- S2 前无产物；S3/S4 产物均在 `outputs/10-20-ni-mse-v5fmt-f0/` 内自隔离，删目录即回滚；代码目录 `models/ni-mse-v1/` 为纯新增，`git checkout` 可整体撤销。θ\*/refs 全程只读，无回滚需求。
