# 实验日志 — latent-flow fold2 nosc global-tau 单被试 GC 相关性

## Run 001 — 同被试现算谱 GC × global-tau signed connectivity 相关（2026-10-21）

### 目的与假设

- 对 global-tau 首跑被试（fold2 test[0] `02010010EOG.set_61_.mat`）按 GC 计算主路径（原库谱 GC）现算 GC，按旧库判决工具 recipe（r_abs/r_signed）求与 global-tau connectivity 的相关，单被试存证。
- 参照：FOUNDATION r_abs(GC)=0.585 为 fold2 nosc 10 例均值（AIC 阶选择 GC 口径，历史 baseline JSON method_params 实证 `order_select: aic`）；本任务 n=1、fixed8 口径，仅描述性并置。

### 配置

- 代码：`latent_flow/baseline/granger.py`（主路径，零改动）+ `latent_flow/baseline/data_utils.load_subject_timeseries`（mat 分支加载，零改动）；相关性脚本会话内联（recipe 复刻 `_compare_signed_de_ablation.py`）
- 参数：`granger_connectivity(data, fs=250, max_order=8, n_freq=65, fmin=1, fmax=125, fixed_order=8, n_jobs=8)`；data = 全长连续 (128, 42501) 逐通道 z-score
- 环境：base conda @ 本机（8 joblib workers，各 ~266% CPU）
- 数据：`/root/autodl-tmp/modma/MDD/02010010EOG.set_61_.mat`；对照矩阵 = `outputs/10-21-latent-flow-globaltau-f2-nosc/run_01/subject_0.json`
- 对齐：GC [a,b]=a→b 行=源 = connectivity 行=源，offdiag（全双三角）直接对齐，无转置
- 命令：`python3 /tmp/gc_corr_f2nosc.py`（脚本全文随本块语义，recipe 见 prd.md；GC 矩阵 npy + 相关 json 落 outputs）

### 结果

- GC：474.5 s 完成；128×128、min 0 / max 781.57、对角 0、非负 ✓；directedness = 0.021（近对称，静息谱 GC 常态）
- **r_abs = 0.1655**（|signed_conn| offdiag z vs |GC| offdiag z）；**r_signed = 0.1150**；r_unsigned_conn_vs_gc = 0.1635；Spearman(|·| offdiag) = 0.2292
- 产物：`outputs/10-21-latent-flow-gc-corr-f2nosc/run_01/`（`gc_subject_0_fixed8.npy` + `gc_corr_subject_0.json`）
- **口径 caveat（描述性登记）**：①n=1 无统计效力，fold2 的 0.585 是 10 例均值，单例波动大；②历史 baseline GC 为 AIC 阶选择（max_order=8），本任务按 npi 主路径拍板用 fixed8（与 refs 归档同口径）——两口径不混用；③对照目标为 32 窗 global-tau 单例矩阵（与历史 per-subject 口径一致）。

### 结论

- 支持：单被试 GC 相关性存证完成（AC1–AC3 兑现）；r_abs=0.166 / r_signed=0.115（fixed8 口径），显著低于 FOUNDATION fold2 10 例均值 0.585，但**因 n=1 + 口径差异（AIC vs fixed8）不构成复现失败判定**——10 例对标（~1.1 h global-tau + ~10×8 min GC，或历史 AIC 口径复核）属后续拍板
- 方向性说明：signed（0.115）与 unsigned（0.164）相当，GC 近对称而 DE 符号半正半负（57.6%/42.4%），r_signed 无额外增益为预期行为
