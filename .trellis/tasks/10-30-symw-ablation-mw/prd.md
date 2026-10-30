# MILD symw 五折 stage-2 GNN M/W 消融

## Goal

在 MILD 五折 symw 基线（`10-25-latent-flow-exit-gnn-symw` + f12 + f34 五折版图）上，量化 stage-2 GNN 图混合 `A_eff = γ_M·M + γ_W·W` 中 M（几何先验对称距离核）与 W（可学习对称残差）各自及联合对模型性能的贡献：

- **Q-A** W 的边际贡献 = A0(full) − A2(no-W)
- **Q-B** M 的边际贡献 = A0(full) − A1(no-M)
- **Q-C** GNN 图混合总贡献 = A0(full) − A3(no-M∧no-W)

## Requirements

### R1 消融臂（γ 置零硬移除，用户 2026-10-30 拍板认可）

| 臂 | CLI 差异（其余与基线逐字相同） | A_eff |
|---|---|---|
| A0 full（基线，**复用不重训**） | —（γ_M=1.0, γ_W=0.1） | M + W |
| A1 no-M | `--mexhat-gamma-m 0` | γ_W·W |
| A2 no-W | `--mexhat-gamma 0` | γ_M·M |
| A3 no-M∧no-W | 两旗标同置 0 | ≡ 0（消息传递退化为 GELU(bias) 常数残差，跨通道混合严格为零） |

- 实现方式 = scale-to-zero：γ 为固定 buffer，置 0 后分量全程零贡献；被消分量参数自动失活（∂A/∂W ∝ γ_W = 0 ⇒ W 逐位保持零初始化；∂A/∂σ ∝ γ_M = 0 ⇒ log_sigma 冻结于初值 log(median_d)）。不引入随机初始化/恒等映射/训练代码改动。
- 训练入口零改动：`train_mild_depress.py` 既有 `--mexhat-gamma` / `--mexhat-gamma-m` 旗标。
- A3 的 anti-uniform 正则退化为常数 ≈0.2（梯度为零）——登记口径 caveat，主指标用 datafit 口径（见 R3）。

### R2 公平性（保持训练超参数和其他模块设置不变）

- 同折同 stage-1：每臂复用该折既有 VAE best ckpt（fold0 = w500 的 `fold0_mild_vae_best.pt`；fold1–4 = symw 各折自训）；stage-2 除两个 γ 键外全部超参与基线逐字相同：epochs 150 / lr 1e-3 / batch 64 / patience 30 / seed 42 / `--anti-uniform-lambda 0.2` / `--unfreeze-decoder` / `--flow-exit-gnn` / `--sym-w` / obs 375 / forecast 125 / window 500 / manifest `mild_5fold_seed42.json`。
- 消融只动 A_eff 组成，不动出口结构（exit-gnn/sym-w 四臂全开）；四臂参数量恒等。
- 单 seed/折（五折为主），与该线既有惯例一致。
- CLI 对拍守卫：每跑的 `stage2_config.json.cli` 与基线同折对拍，diff 键 ⊆ {run_dir, mexhat_gamma, mexhat_gamma_m}，否则拒收。

### R3 评估指标（四层；用户 2026-10-30 拍板：M3 纳入 + 新增 M4）

- **M1 预测质量（主指标，每折必报）**：stage-2 **val datafit loss = recon2 + corr2 + cons2**（trainer 按 val loss 选 best ckpt；datafit 从 history 剔除 λ·uni_reg 项重算，四臂可比；基线同口径重算）。同时报原始 val loss 与三分量 recon2/corr2/cons2（用户要求 7：汇报 stage-2 各项损失变化）。
- **M2 幅度校准**：`quality_check_mild.py` std_ratio full_mean（同基线口径：每折 3 test 被试（逐折 triad：fold0=1,7,21 / fold1=2,12,22 / fold2=3,6,23 / fold3=9,13,29 / fold4=4,15,24，2MDD+1HC）× 30 trial × K=32 确定性采样；窗口旗标 500/375/125 必传）。
- **M3 FC 一致性（指标 1 v2.1 镜像）**：gen-fc vs real-fc 组平均 Pearson r + 逐被试 r_s（8 test 被试/折；确定性采样 K=1）；需新增可装载任意 ckpt 的变体脚本，口径逐字镜像 `fc_metric_mild_symw.py`，仅 ckpt 来源/守卫改为显式路径 + SHA 随档登记。
- **M4 A_eff↔组平均 GC-EC 结构一致性（本任务新增）**：每折从 best ckpt 提取 A_eff（128×128），与组平均 GC 参考矩阵（`refs/granger_bivariate_spectral/mild40-trial500-zscore-fs250-fixed8-f1-45-favg/`，40 例 128×128，ec[a,b]=a→b）做 offdiag Spearman/Pearson：主口径 = 折 test 8 例组平均；副口径 = 全 40 例组平均。协议沿用 EC×GC 先例命名（aligned=vs ec、flipped=vs ec.T）× {raw A_eff, |A_eff|}；另加**距离偏相关 Spearman 副口径**（通道对 3D 距离作偏出变量——A2 臂 A_eff 为纯距离核，raw 相关受共同距离梯度混杂，partial 口径为该臂主判读；评审 Mandate A 发现 1 采纳）。A3 臂 A_eff≡0 ⇒ 相关系数无定义，登记 NaN(by-design)。
- 汇总：每臂每指标五折 mean±sd（ddof=1）+ 逐折明细 + 每折 Δ vs A0。
- **不做** EC 扰动-推断对拍（用户拍板）。

### R4 训练日志全量保留（用户要求 6）

每跑保留：完整 stage-2 stdout log（含逐 epoch 损失变化）、`stage2_history.json`（train/val 全 epoch 全字段）、`stage2_config.json`、best ckpt。汇总时另存四臂并置的损失曲线数据（含基线 history 复用）。

### R5 产物与登记纪律

- 全部产物落 `outputs/10-30-symw-ablation-mw/`（变体目录隔离）；**不写 `result/`**（result-dir-policy：无用户明确指令）。
- 跑前登记：每个 Run 在 `experiment.md` 登记目的+配置；跑完回填结果+结论（四要素齐全）。
- 训后守卫（断言即拒跑惯例）：A2/A3 训后 W 逐位为零；A1/A3 log_sigma == log(median_d)（<1e-12）；A1 W `torch.equal` 对称；best ckpt val 与 history 一致。

## Acceptance Criteria

- [ ] **AC1** 训练完备：A1/A2/A3 × fold0–4 共 15 跑全部完成（best ckpt + history + config + stdout log 四件套齐），CLI 对拍守卫 15/15 通过。
- [ ] **AC2** 训后守卫 15/15 通过（W 零 / log_sigma 冻结 / W 对称按臂适用）。
- [ ] **AC3** M1 四臂五折表齐（datafit 主口径 + 原始 loss + recon2/corr2/cons2 分量），含逐折 Δ vs A0。
- [ ] **AC4** M2（15 折质量抽查）+ M3（15 折 FC 核算）+ M4（15 折 A_eff↔GC）数值齐并置基线，五折 mean±sd 汇总表落 `outputs/10-30-symw-ablation-mw/summary/`。
- [ ] **AC5** `experiment.md` 所有 Run 块四要素齐全无空块；`.trellis/research/progress.md` 当前状态同步；消融结论（M/W 各自贡献方向与量级）写入 summary README。

## Notes

- 基线 A0 数值来源（复用，不重训）：五折 histories/quality/fc summary 已核（2026-10-30 会话提取：val loss 0.8054/0.8264/0.8433/0.8358/0.8769，datafit 0.8345±0.0262；幅度 0.895±0.143；FC 组 r 0.849±0.116、被试 r 0.813±0.110）。该线 CLI 对拍/逐位确定性已有先例验证（10-28 归档任务）。
- 预算：stage-2 ≈11.4 min/跑 ⇒ S2 ≈3 h；M2 ≈29 min/折 ⇒ ≈7 h；M3/M4 各 ≈2–5 h；合计 ≈13–15 h GPU 串行。GPU 已确认空闲（用户 2026-10-30）。

## 子研究路由（1.0b / 1.0c）

<!-- trellis-research-routing -->

- **分类结果**：`study1` —— EEG 代理脑（框架 Study I：模型训练与代理脑有效性指标）
- **依据**：任务主体是 mild 代理脑模型 stage-2 训练变体与代理脑质量指标（幅度/FC 一致性），M4 为 A_eff 结构描述性并置，非 EC 推断算法研究。
- **新路径/方法检测（1.0c）**：`已有路径：latent-flow mild w500 → exit-gnn → symw 消融扩展` —— 方法为既有 symw 线的消融臂（γ 置零消融口径在 `oc_latent_flow.py effective_adjacency` docstring 登记，task 07-15 先例），无新路径；M4 对照协议镜像既有 EC×GC 先例（gccorr，10-24/10-28 任务）。
- **research 产物**：`无`（无新路径；复用先例口径）
