# latent-flow × mild-w500 无偏噪声扰动归因 EC（global-τ 两阶段）

## Goal

以 10-24 任务 mild-w500 fold0 代理脑（obs 375 / fc 125 前缀窗 500）为对象，用**无偏噪声**
（μ=0，扰动 = σ·ξ）信号空间扰动 + 两阶段 global-τ 口径（**q=0.5 钉死**）计算扰动归因
EC：主跑 fold0 test **8 例 × 30 trial**（用户确认直接使用），产 σ*/conn 矩阵（128×128
有向）与诊断；同时将 autodl-tmp 下载的 **mild GC 参考注册入 refs**（VAR 阶 8 固定、
1–45 Hz），并做 EC×GC 描述性对照。

## 背景与动机

- 用户指令（2026-10-24 两轮）：先方案后执行；方案提交后用户**明确同意并三项拍板**：
  1. **注册 GC 参考**：autodl-tmp 的 mild GC 参考数据（mne-connectivity 状态空间谱
     GC、**gc_n_lags=8 VAR 阶数固定**、**fmin=1/fmax=45 Hz + faverage**、每 trial 前
     500 点、逐 trial 逐通道 z-score、30 trials 池化、(40,128,128) mat[i,j]=GC(i→j)
     对角 0）注册到 refs；
  2. **q 保持默认值 0.5，无需测试或计算 q 的其他取值**（原方案 q=0.25 + 三档 ladder
     校准门作废）；
  3. **分析时窗 8×30（8 例 × 30 trial）直接使用**（经验证合适）。
- **无偏口径房内先例**（10-21 晚未归档会话，`outputs/10-21-ec-globaltau-v5main/
  unbiased_mu0_*` + `models/ec-globaltau-v1/run_ec_globaltau_unbiased.py` 等未提交
  脚本）：μ=0 σ·ξ、z-score 后入模前源通道逐点、σ_ref=0.3、log[0.01,0.48]×24、
  conn=−(log σ*+ln 2πe)+1。本任务 = 同口径换代理脑迁移（gpfm-v5-main → latent-flow
  mild-w500），q 例外（0.25→0.5，用户拍板）。
- **接口事实（可行性依据）**：`OCLatentFlow.sample(x_obs, n_steps, use_proj)` 原生即
  `bisect_sigma.SignalProbeSite` 设计契约——channel-last (B, T_obs, C)、`model.obs_len`
  切 fc 尾、确定性采样 ⇒ **零适配复用包体原生 probe**（`use_biased_noise=False`）。
- **代理脑资格 caveat（登记不阻塞）**：w500 stage2 best 0.8356@E150 顶格未收敛、波形
  判读待拍板；用户明确指定该模型为代理脑。EC 产物**不构成代理脑资格判定**。
- **GC subject 映射已逐位验证**：GC zip 按 EEG_ALL.mat 顺序编号 s0–s39（前 20 MDD 后
  20 HC）；EEG_ALL 第 i 片与 manifest 文件（m_1/m_20/h_1/h_20 四锚点）**bitwise 相等**
  ⇒ sid = index + 1 赋名安全。

## Requirements

- **R1 代理脑与数据口径（只读复用，零重训）**：checkpoint =
  `outputs/10-24-latent-flow-mild-w500/run_w500_fold0/checkpoints/
  fold0_mild_flow_best_graph_mexhat.pt`；模型经 `build_model_mild(stage=2, obs_len=375,
  forecast_len=125)` + `viz_mild.load_flow_ckpt` strict 加载；数据 = manifest
  `splits/mild_5fold_seed42.json`（SHA256 断言重放）fold0 **test 8 例**（sid
  1,5,7,10,21,25,27,30）× 每例 30 trial **全量**（用户拍板 8×30 直接使用）；归一化 =
  `compute_mild_train_norm(train_f, window_len=500)`（train 折统计严格 held-out）；
  采样确定性沿用。
- **R2 扰动范式（无偏噪声，房内口径逐字对齐）**：扰动 = σ·ξ（ξ~N(0,1)），逐点加在
  z-score 后、入模前 x_obs（375 点 obs 段）**源通道**；ξ 以 seed+source 播种（seed=42），
  **同一 ξ 跨该源全部 σ 与 σ_ref 复用**；其余通道逐位不动；μ_bias 关闭（mean_diff/sign
  随档保存不解读）；128 源全扫。
- **R3 两阶段 global-τ EC 口径**：stage1 = 每源一次 σ_ref=0.3 前向 → (128,128) 响应
  矩阵 → τ_global = q 分位（`compute_global_tau`，逐被试）；stage2 = 每源
  log[0.01,0.48]×24 网格走查 → `sigma_star_with_global_tau` 单调走查（线性插值 + 边界
  钳制）→ σ* 矩阵；conn = −(ln σ* + ln 2πe) + 1。**q = 0.5（用户拍板钉死，不做任何
  其他取值的测试或计算）**。
- **R4 执行形态（校准门取消）**：工程 mini 干跑（代码正确性校验：形状/断言/无 NaN/
  conn 值域，8 源 × 4 网格 ~1 min——非科学口径测试）→ Run 001 main 直接全量（8 例 ×
  30 trial）。主跑模式**顺带**落盘每例完整 σ–MSE 曲线 npz（128×24×128，后置健康诊断
  用——dyn_ratio/单调性/钳边从落盘曲线重算登记，**非 gate、非 q 测试**）。
- **R5 产物与登记**：`subject_{0..7}.json`（07-04/unbiased schema：σ*/conn/sign/
  raw_response/mean_diff/全超参 + diagnostics + 凭据）+ 逐例曲线 npz；Run 002 离线汇总
  （组级分布、行/列/残差方差分解、跨被试 offdiag Spearman 稳定性、1/(σ*+1e-4) 重映射
  对照、曲线后置诊断）；Run 003 GC 相关对照（协议同 unbiased 先例：|EC| 与 raw 两口径
  × aligned/flipped × Spearman/Pearson，offdiag 128×127=16256 对，**描述性登记不构成
  资格判定**）；输出 README；experiment.md 每 Run 跑前登记+跑后回填。
- **R6 GC 参考注册**：`refs/granger_bivariate_spectral/mild40-trial500-zscore-fs250-
  fixed8-f1-45-favg/`：40 × `refs_s{sid}.npz`（`ec` float32 (128,128)，ec[a,b]=a→b
  行=源/列=靶、对角 0 + `subject_id` + `group` + `params` 串，sid=index+1）+ `meta.json`
  （来源 zip 路径与 SHA256、方法/参数、映射验证记录）+ `SHA256SUMS` + 源 README 归档
  （README-source.md）+ 注册脚本与日志（register_refs.py / register.log）；refs/MANIFEST.md
  追加条目（mne-connectivity 实现单独成节，与 latent_flow 原库 GC 档口径隔离）。
- **R7 工程纪律**：包体 `latent_flow/` 零改动；变体层仅新增 `models/latent-flow-v1/
  ec_mild_unbiased.py` 单文件；10-24 任务未提交脏路径（train_mild_depress.py /
  viz_mild.py / quality_check_mild.py）零触碰；产物落 `outputs/10-24-mild-w500-ec-unbiased/`；
  refs 注册只新增子目录不改旧档；BLAS 线程上限在 numpy 导入前设置。

## Acceptance Criteria

- [ ] **AC0 GC 参考注册完整**：40 npz + meta.json + SHA256SUMS + 源 README + 注册脚本/
      日志落 refs 子目录；MANIFEST.md 追加条目；映射验证（四锚点 bitwise）记录在 meta。
- [ ] **AC1 mini 干跑通过**：8 源 × 4 网格干跑产物形状/断言/无 NaN 核对通过。
- [ ] **AC2 主跑完整落盘**：Run 001 fold0 test 8 例 subject_{0..7}.json + 曲线 npz 齐全
      （q=0.5、σ_ref=0.3、log[0.01,0.48]×24），diagnostics（τ/interior/@min/@max）与
      凭据字段完整；源失败数 = 0。
- [ ] **AC3 汇总与对照完整**：Run 002 组级汇总（分布/方差分解/稳定性/重映射/曲线后置
      诊断）+ Run 003 GC 相关（两口径 × aligned/flipped × Spearman/Pearson）落盘并回填
      experiment.md。
- [ ] **AC4 路径隔离**：`latent_flow/` 包体零 diff；变体层仅新增 1 文件；旧任务输出树
      与 10-24 未提交脏路径零触碰；refs 旧档零改动。
- [ ] **AC5 复现凭据**：payload 记录 ckpt/manifest SHA256/norm hash/全超参（q=0.5）/
      seed/git head；同 seed 复跑可追溯。
- [ ] **AC6 数字回填**：experiment.md Run 001/002/003 三块目的-配置-结果-结论齐备；
      输出 README.md 落盘。

## 边界（本轮不做）

- **q 敏感性分析不做**（用户拍板：q=0.5 钉死，不测试不计算其他取值）。
- GC 对照为**描述性并置**（协议同 unbiased 先例），不构成代理脑资格判定；mild GC 档
  与 edra 两档口径隔离，不做跨数据集对比交付。
- 不做代理脑资格判定/三项检验；不做 5-fold；不重训、不改采样确定性、不动架构与包体。
- 不解读 sign/mean_diff（μ=0 无方向性设计依据，仅随档保存）。
- r\*→连接强度最终映射仍为全局待拍板项（本任务沿用 unbiased 登记式，σ* 原矩阵随档）。
- 不做多 ξ 重复方差估计（房内设计 = 每源单条 ξ 跨 σ 复用）。

## 子研究路由（1.0b / 1.0c）

- **分类结果**：`study2` —— EC 推断（Study II）任务：既有代理脑上虚拟扰动计算 EC +
  GC 观测参考对照；代理脑只读引用。
- **依据**：任务核心交付 = 扰动归因 EC 矩阵 + 诊断 + GC 参考注册与对照，属 Study II
  四要素范畴。
- **新路径/方法检测（1.0c）**：`已有路径：Study II global-τ 扰动 EC（无偏口径变体）`
  —— 技术路线表已登记 global-τ 线；本轮 = 同路径换代理脑对象 + 观测参考接入，
  无新方法引入，无需适配研究。
- **research 产物**：`无`（已有路径；unbiased 先例产物与 07-04 原语文档即直接输入，
  见 design.md 引用清单）。
