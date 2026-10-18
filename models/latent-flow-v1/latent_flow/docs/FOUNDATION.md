# FOUNDATION.md — latent_flow 地基文档

> **声明：此后所有工作必须基于此文档推进。** 本文档固化 2026-08-01 迁移完成时
> 的项目状态（阶段成果、实验清单、关键决策、已知问题、下一步）。后续实验、
> 代码改动、spec 更新都应以本文档为基准，重大方向变更须更新本文档。

---

## 1. 当前阶段成果

**项目定位**：MODMA / EDRA 静息态 EEG 的潜空间生成式预测模型研究。核心模型
**OC-LatentFlow**（obs-conditioned latent flow）——两阶段纯 1D-conv 潜空间流匹配，
带可选的 band-pass graph-mixer（Mexican-hat / DoG kernel）。

**已达成的研究里程碑**：

| 维度 | 成果 | 数值 |
|---|---|---|
| 跨被试预测 | OC-LatentFlow 首次达到 1e-1 量级跨被试 Pearson（JET 为 1e-4） | Pearson 0.172 (proj) / 0.157 (flow) |
| Graph-mixer 改进 | Mexican-hat band-pass mixer 将 Pearson 从 0.449 提升到 0.747 | corr2 loss 0.551→0.253 |
| 脑连接推断 | 有向 DE 连接性与 GC/TE baseline 相关性恢复 | r_abs(GC)=0.585 (MODMA fold2), 0.413 (EDRA) |
| 连接性结构 | offdiag CV 128%（>50% 目标），距离衰减 Pearson r=0.842 | 强于 GC 0.698, TE 0.504 |
| 分类（阴性） | DCGNN EC-graph HC/MDD 分类为阴性（≈chance） | acc ~0.52, SHAM≈real |

**工程状态**：
- 训练链完整可运行（stage1 VAE → stage2 flow，`--smoke` 冒烟不污染正式 ckpt）
- 95 个单元测试通过，shape_check 通过
- 代码库已清理（Phase 1-4：弃用 GRU/VAR/SC/一次性脚本，净减 27%）
- Trellis spec 重构为 6 个 ML 科研层 + 实验记录 sub-agent

---

## 2. 已完成实验清单

按时间线（提炼自 `.trellis/spec/domain.md` Key Findings，22 个节点）：

**架构探索期（2026-06）**：
- 06-12：Fold 0 诊断（z_obs PCA 崩塌、跨被试泛化 gap）
- 06-13：MAE 误导性发现（V2/V3 伪低 MAE，输出崩塌）
- 06-15：JET v2 bundle 失败（RevIN 是崩塌元凶）
- 06-23：**OC-LatentFlow 首次 1e-1 跨被试 Pearson**（~430x over JET）
- 06-28：Stage-2 flow 范式修正（flow target 从 proj(z_obs) 改为真实 fc latent）

**连接性推断期（2026-06 ~ 07）**：
- 06-25：OC 连接性推断（bisection σ 搜索）+ 归因性能（解码器是瓶颈）
- 06-30：**Mexican-hat 突破**（channel-selectivity 根因 + band-pass mixer）
- 07-02：SC 先验集成（lead-field 投影）—— *SC 后已弃用*
- 07-03：有向 Mexican-hat mixer + SC-init 消融
- 07-04：DE vs GC/TE 零相关根因（扰动-推断范式）
- 07-05：DE 连接方向（5 个方向指标全失败）
- 07-23：**DE 方向符号恢复**（biased-noise 扰动）
- 07-30：DoG kernel 消融 + EDRA A_eff 结构教训

**分类期（2026-07）**：
- 07-31：**DE-EC graph 分类阴性结果**（DCGNN，N=10 不可信，SHAM 对照必需）

详细发现见 `.trellis/spec/domain.md` "Key Findings" 各节。

---

## 3. 关键决策（摘录自父仓库 ADR）

> ADR 源文件在父仓库 `ude-vae-npi/docs/adr/`，未复制进本项目。此处摘录关键决策。

**ADR-0003：转向 flow matching（superseded by 0004）**
- NCDE 的低 MAE 反映均值预测，非真实生成；潜空间流匹配消除分布失配。
- 信号空间流（signal flow）遇 MAE 天花板（~1.25 单次 / ~2.1 fold0），无论 backbone。
- 潜空间流优于信号空间流：Test MAE 0.66 vs 1.25（单次）。

**ADR-0004：归档 JET 与 legacy 路径（current，2026-06-29）**
- JET 跨被试方向耗尽（Pearson ≈ 0）；OC-LatentFlow 是唯一活跃模型。
- 删除 14 个模型 / 26 个脚本 / ~15000 行；活跃面 = `oc_latent_flow.py` + 其簇 + `baseline/`。
- JET/latent_flow(v1-3)/NCDE/UDE/Mamba/signal-flow 全部可经 git 历史恢复。

**CONTEXT.md 关键发现（摘录）**：
- 潜空间流优于信号空间流（消除分布失配）。
- `flow(z_obs) → z_fc` 范式（强制编码器编码未来信息到 z_obs）是跨被试突破的关键。
- U-Net skip 是强制的（非可选）——无 skip 时 Pearson 0.086 / std_ratio 0.30。
- Mexican-hat band-pass 优于 Gaussian（Gaussian 低通会 isotropize，抑制高频通道结构）。
- corr2 = `1 − Pearson r`（loss），低 corr2 = 高相关，勿混淆。

**本会话的决策（2026-08-01）**：
- SC 先验整条线弃用（从模型核心摘除 sc_base/gamma_sc，旧 ckpt 兼容加载）。
- GRU/VAR 基线弃用（归档）。
- batch_size 默认 64→32（RTX 5070 Ti 12GB，batch=64 触发 cuDNN workspace 错误）。
- 结果目录统一为 `results/`（`results_repo/` 归档）。

---

## 4. 已知问题

**硬件/平台**：
- `torch.compile` 在 Blackwell sm_120 坏（triton 3.2.0 缺 nvvm selector）——禁用。
- Windows `conda run` 触发 gbk 编码错误——用直接 python 路径。
- **GPU 不稳定**：DPC_WATCHDOG_VIOLATION (0x133) + PCIe AER 错误，GPU 重负载下反复蓝屏
  （7 月以来 4 次）——驱动/散热/PCIe 电源管理问题，非代码 bug。

**研究**：
- EDRA 分类阴性（N=10 单 fold 不可信；LOO 记忆；需全样本 5-fold CV + SHAM 对照）。
- Stage-1 History 不落盘（只有 stage-2 持久化）。
- 无环境锁定文件的历史（现已补 `requirements.txt`）。
- 无 model_registry 的历史（现已补 `docs/model_registry.yaml`）。

---

## 5. 下一步计划

基于当前成果（特别是 EDRA 分类阴性）的后续方向：

1. **EDRA 分类后续**：若要继续 HC/MDD 分类，必须用全样本 5-fold CV + window-level
   训练 + SHAM 对照（`train_dcgnn_full_cv.py` 已支持）；或转向其他下游任务。
2. **连接性推断深化**：DE 方向符号已恢复（biased-noise），可探索更强的方向指标；
   DoG kernel 的 A_eff 结构教训（抑制位置 > 抑制数量）值得在 EDRA 验证。
3. **可复现性补强**：实现 HP 快照随 ckpt 落盘（`experiments/reproducibility.md` ★待实现）。
4. **训练改进**：NaN 守卫（`training/numerics.md` ★待实现）。

---

## 6. 待确认项

- `archive/results_repo/` 含 49 个独有实验目录（历史策划库），是否有需提取回
  `results/` 的产物——目前全部归档保留，未逐一甄别。
- DCGNN 分类链（`dcgnn_classifier` + `de_features` + `subject_classification`）保留，
  但其上游 OC 连接性结果（DE-EC）是否仍可复现（依赖 `archive/results_repo/` 的 baseline JSON）。

---

## 引用

- `.trellis/spec/domain.md` — 完整领域术语表 + 22 个 Key Findings 详述
- `.trellis/spec/decisions.md` — ADR 索引
- `docs/model_registry.yaml` — checkpoint 版本注册表
- `docs/inventory.csv` — 全量资源清单（826 条）
- `docs/path_mapping.csv` — 旧→新路径映射
