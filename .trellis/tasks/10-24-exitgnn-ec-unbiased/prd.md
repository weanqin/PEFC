# exit-gnn 代理脑无偏噪声扰动归因 EC（global-τ 两阶段，q=0.5）+ 双 GC 参考对照，conn v4 公式交付

## 子研究路由（1.0b / 1.0c）

<!-- trellis-research-routing -->

- **分类结果**：`study2`（EC 推断算法，框架 Study II——固定 θ 候选代理脑做虚拟扰动推断连接）
- **依据**：任务主体是扰动归因 EC 计算与 GC 参考对照（推断算法侧）；代理脑（Study I 产物 exit-gnn 变体）仅作为固定输入使用，本任务不训练、不做资格判定。
- **新路径/方法检测（1.0c）**：`已有路径：global-tau 无偏 EC 口径（10-24-mild-w500-ec-unbiased 已归档先例，mild 线迁移换代理脑；GC 参考注册路径同 Run 000 先例）`——无新方法，research adaptation 不触发。
- **research 产物**：`无`
- **路由备注**：EC 结论继承 exit-gnn 代理脑 provisional 状态（幅度欠校准 + 二因素 caveat，见「代理脑 caveat」节），本任务产物不构成代理脑资格判定。

## Goal

以 10-24-latent-flow-exit-gnn stage2 best ckpt（best val 0.8241@E116，真收敛）为代理脑，复用
`models/latent-flow-v1/ec_mild_unbiased.py` 无偏 μ=0 口径（**全部默认参数**：q=0.5、σ_ref=0.3、
σ 网格 log[0.01,0.48]×24、fold0 test 8 例 × 30 trial 全量）主跑扰动归因 EC；交付公式
**v4：conn = −(ln σ* + 0.5·ln 2πe) + 5/7**；离线组级 summary；EC×GC 描述性对照**双参考**
（v1 已注册 mild40 1–45 Hz 固定阶 8 + v2 新注册 autodl-tmp/EC_GC_1_125 1–125 Hz AIC 逐被试
选阶 4–25）。

## 背景与动机

- 用户指令（2026-10-24，两项拍板）：
  1. **实验配置**：代理脑 = 10-24-latent-flow-exit-gnn 模型；EC 公式 v4 =
     −(ln σ* + 1/2·ln 2πe) + 5/7；**不使用偏置噪声**（无偏 μ=0）；σ 取值范围 [0.01, 0.48]、
     σ_ref = 0.3、q = 0.5；其余参数默认。
  2. **交付范围**（AskUserQuestion 拍板）：全套默认流程（主跑 + 组级 summary + v4 重映射），
     **双 GC 对照——新加的 GC 参考在 autodl-tmp/EC_GC_1_125**。
- 口径事实：上述 σ/q/σ_ref/无偏噪声恰为 `ec_mild_unbiased.py` 现行默认值（上任务用户钉死
  q=0.5 的延续）；**唯一硬件差异 = 代理脑架构**——exit-gnn 变体 flow（+33,024 参数，出口
  K 轮共享 A_eff 消息传递 + 零初始化出口投影），当前脚本硬走 w500 基座，需加
  `--flow-exit-gnn` 旗标（`build_model_mild`/`viz_mild._model_args` 已支持同路径，viz 已验证
  strict load 正常）。
- v4 公式数学事实（Run 007 先例在档）：v4 = 房内原式 v1（−(ln σ*+ln2πe)+1）严格常数平移
  （Δ=+0.5·ln2πe−2/7≈+1.1332）⇒ **raw 口径 Spearman/Pearson 逐位不变**；零点
  σ*=e^(5/7−0.5·ln2πe)=0.4943 **落在网格上界 0.48 之外 ⇒ 全部 conn 严格为正 ⇒ abs≡raw**。
- v2 GC 参考（EC_GC_1_125.zip，README 在档）：1–125 Hz faverage、**AIC 逐被试选阶（4–25，
  池化二元 VAR 完全模型 OLS-AIC，统一 500−25 截断）**、mne-connectivity 状态空间谱 GC、
  (40,128,128) float64 mat[i,j]=GC(i→j)、对角 0 非负无 NaN（README QC：mean=0.0475
  max=0.568）；与 v1 逐被试矩阵 Pearson median=0.148（两参考差异大，双对照正当）。
  README 注明数据有效带宽 ≈0.5–50 Hz，51–125 Hz 已衰减 −24 dB 以下 ⇒ 1–125 积分主要由
  50 Hz 以下贡献（caliber note 随档登记，不阻塞）。
- 先例：10-24-mild-w500-ec-unbiased（已归档）= 同口径 w500 代理脑版（Run 001–008）；本任务
  = 换代理脑迁移 + v4 交付 + 双 GC 对照。

## Requirements

- **R1（v2 GC 参考注册）**：解压 `/root/autodl-tmp/EC_GC_1_125.zip` → staging → 注册 40 ×
  `refs_s{sid}.npz` 到 `refs/granger_bivariate_spectral/mild40-trial500-zscore-fs250-aic4-25-f1-125-favg/`
  （镜像 v1 register_refs.py 先例：ec float32 + subject_id + group + params + SHA256SUMS +
  meta.json + README-source.md + 注册脚本随档 + **refs/MANIFEST.md 追加 v2 行**）；subject
  映射 sid=index+1 **四锚点 bitwise 复验**（EEG_ALL 第 i 片 vs manifest m_1/m_20/h_1/h_20）；
  sanity 断言（shape/dtype/对角 0/非负/无 NaN）+ 与 README QC 量级复核。
- **R2（代理脑适配）**：`ec_mild_unbiased.py` 加 `--flow-exit-gnn` 旗标（默认 False，旧调用
  面 w500 复现零影响）→ `_model_args(flow_exit_gnn=...)` + build_model_mild stage2 变体子类
  strict load；surrogate 元数据随旗标条件化（task/run/best-epoch/变体描述），凭据链完整
  （ckpt 路径 + config flow_exit_gnn:true + git head + norm/manifest SHA 断言照旧）。
- **R3（主跑，用户口径钉死）**：GPU 两阶段 global-τ EC：无偏 μ=0 扰动 σ·ξ（use_biased_noise
  =False，断言 μ_bias None）、q=0.5、σ_ref=0.3、log[0.01,0.48]×24、8 例（MDD 4：s1,5,7,10 /
  HC 4：s21,25,27,30）× 30 trial 全量、seed 42、bf16 autocast、断点续跑；输出
  `outputs/10-24-exitgnn-ec-unbiased/ec_main/`（逐例 json + 曲线 npz）。τ 分位含对角口径
  照旧（房内先例）。
- **R4（v4 公式交付）**：离线重映射脚本（正式入库，非临时）：σ* → conn v4 =
  −(ln σ*+0.5·ln2πe)+5/7；断言：①常数平移 ⇒ raw Spearman vs v1 **逐位一致**（8/8）②全正
  （min>0，8/8）③abs≡raw（逐位一致，8/8）；逐例 json 重映射档（sign/元数据随档拷贝）。
- **R5（组级 summary）**：离线（interior%/dyn_ratio/方差分解/跨被试稳定性/重映射不变性
  断言），与 w500 Run 002 同口径可对照。
- **R6（双 GC 对照）**：EC v4 × {v1 refs, v2 refs} 两套，offdiag 16256 对、aligned/flipped ×
  Spearman/Pearson，协议对齐 gen_unbiased_gc_corr.py 先例；描述性登记不构成代理脑资格判定。
- **R7（caveat 随档）**：exit-gnn 幅度欠校准回归（medoid/real std 0.645/0.581 vs w500
  0.957/0.996）+ lr 1e-3 与出口结构**二因素同变** + θ 候选判断不如 w500 基座（待用户拍板）
  ——逐例 payload caveat 字段 + README + experiment.md 三处登记，EC 结果标 provisional。
- **R8（路径隔离）**：改动仅 `models/latent-flow-v1/`（两脚本）+ 新 refs 目录 + 新 outputs
  任务树 + 本任务目录；latent_flow 包体零改动；旧任务输出树零触碰。

## Acceptance Criteria

- [x] v2 refs 注册完成：40 npz + SHA256SUMS + meta.json + README-source.md + 脚本随档 +
      MANIFEST.md 追加 v2 行；四锚点
      bitwise 复验 maxabsdiff=0.0 在档；sanity 断言全过（对角 0/非负/无 NaN/shape dtype）。
- [x] ec_mild_unbiased.py：`--flow-exit-gnn` 默认 False 时旧行为等价——**运行时证据**：
      旗标关同参 mini 跑与本任务 scratch，σ*/τ/conn 与 w500 在档 dryrun_mini 逐位对拍
      max|Δ|=0.0（或实测差登记+根因查清）；旗标开
      时 strict load exit-gnn ckpt 成功（dry-run 验证）。
- [x] Run 001 主跑 8/8 例成功；逐例 json 含完整凭据（ckpt/config 旗标/git head/norm SHA
      断言通过/manifest SHA/τ 含对角口径注记/caveat 字段）；P(resp>τ)=0.500 逐例精确。
- [x] v4 重映射：三断言（raw Spearman 逐位不变 / 全正 / abs≡raw）8/8 PASS；recompute_v4/
      逐例 json + 汇总落档。
- [x] 组级 summary json 落档（跨被试稳定性矩阵 + 方差分解 + 诊断组均）。
- [x] 双 GC 对照：gccorr 汇总含 v1/v2 两块 per-subject + group + group_mean；flipped 对照
      照登；v1 结果与 w500 先例可并置描述（跨代理脑并置注记）。
- [x] experiment.md Run 000/000b/001/002/003 块齐备（目的-配置-结果-结论）；研究记忆
      （progress.md/context.md 如适用）当场回写；README.md 于 outputs 任务树。
- [ ] 收尾三件套（2.2 check / 3.3 spec-update 判定 / 3.4 commit 仅本任务路径）完成。

## 代理脑 caveat（登记不阻塞）

- exit-gnn Run 002 质量抽查：medoid/real std 全程 0.645 / 去末 0.581（w500：0.957/0.996）
  ——幅度欠校准回落至接近 10-21 旧线区间；机制假设 = 出口邻居混合「再平均」平滑 v。
- lr 1e-3 与出口结构两因素同变，单变量归因待消融（拍板项）；θ 候选判断现倾向 w500 基座。
- 用户明确指定该模型为代理脑 ⇒ 照用；EC 产物不构成代理脑资格判定，标 provisional。

## 约束

- 环境：npi 本机（RTX 4080 SUPER 32 GB，当前空闲）；GPU 主跑预算 ≈50 min（w500 实测
  46.9 min 参照）；BLAS 线程守卫在任何 numpy 导入前（脚本已内置）。
- 数据：Mild Depression（契约 spec/data/mild-depression.md）；w500 口径硬三元组
  500/375/125 断言 + norm SHA d184e647… 对拍照旧（同一训练口径复用，无需改）。
