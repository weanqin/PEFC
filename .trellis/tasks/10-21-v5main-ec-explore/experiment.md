# 实验日志 — v5-main 双方法 EC 推断探索

> 每次「跑批/训练/评估/统计」追加一个 Run 块：跑前先写「目的+配置」，跑后同会话补「结果+结论」，不留空块。
> 格式契约见 `.trellis/spec/research/experiment-log.md`；产物只落本任务声明的变体目录（`.trellis/spec/research/path-isolation.md`）。

## Run 001 — 双方法 EC 探索全量（噪声→熵失稳 σ* / 常量偏移→均值响应）× v4 GC refs（2026-10-21）

### 目的与假设

- 用户直给规格的两条扰动→EC 读出在 v5-main-01（v4 数据 fold0 模型）test 集 10 例上可行性与读出质量：方法1（源通道噪声 σ∈[0.05,0.48] log×24 网格，靶通道高斯拟合微分熵涨幅>20% 判失稳记 σ*，EC1=-(ln σ*+ln 2πe)+1）；方法2（源通道 +0.5std 常量，靶通道预测均值变化带符号=EC2）。两法与 edrav4-46 GC refs 逐被试相关。
- 假设（简报预期）：方法2 响应非退化；方法1 在 K=1 + 40 Hz 产物口径下敏感性未知——若靶通道熵对源噪声不敏感则大面积删失（EC1 退化为常数、相关不可定义），属合法结构性阴性。

### 配置

- 代码：`outputs/10-21-v5main-ec-explore/gen_ec_explore.py`（v1 sample.py 经 v5 config/model 双遮蔽复用；采样 = EMA shadow + Euler 16 步 + GPR 条件先验 + 终点 40 Hz 低通；显式 eps/eta CRN；扁平批 cap=1000 行分块）
- 参数：K=1；m1：σ 网格 logspace(log0.05, log0.48, 24)、50 窗/例等间隔、失稳阈值 = mean_w ln Var 增幅 > ln(1.44)（高斯熵比 1.2 解析等价）、删失=σ*0.48、seed=2026+sid；m2：c=0.5×全窗实际 std、全量窗 1321/例、seed=2026+100000+sid
- 环境：本地 base env（CUDA RTX 4080 SUPER，mamba_ssm fast path），BLAS 五变量 setdefault "8"，cudnn deterministic
- 数据：edra_process_v4 fold0 test 10 例（12/17/18/24/30/38/44/55/58/61，固化 manifest `models/gpfm-mamba-v1/splits/fold0_status_ok_seed42.json`；z-score = run 目录训练折统计）；refs = `refs/granger_bivariate_spectral/edrav4-46-resample250-cont-zscore-fs250-fixed8-f1-125-nf65/`
- 命令：`python3 outputs/10-21-v5main-ec-explore/gen_ec_explore.py > outputs/10-21-v5main-ec-explore/run01.log 2>&1`（逐被试 npz 断点续跑）

### 结果

- **方法1：失败（用户 2026-10-21 宣告取消）**。已算 4 例（s12/s17/s18/s24）offdiag 删失率均 **1.000**：σ 网格扫满（上限 0.48）无一边触发 20% 熵涨幅，EC1 全边恒定 −1.104 → 与 refs 相关不可定义（Spearman NaN / Pearson ~0）。诊断量（logvar_infl）定位结构原因：σ=0.48 时靶通道 ln-方差膨胀 mean 0.019 / 中位 0.014 / p95 0.079 / max 0.240，**远低于阈值 ln(1.44)=0.365**——条件流 + GPR 先验 + 40 Hz 产物口径下，单输入通道噪声几乎不改变其余通道预测分布的离散度。与既有两条 EC_pert 阴性路径判读一致（代理脑跨通道传导弱）。4 例 npz 含 62源×24σ×62靶 膨胀矩阵，作为失败证据随档；剩余 6 例按用户指令停算。
- **方法2：10/10 例完成**（~110 s/例）。逐被试 aligned（行=源，与 ec 同向）：Spearman ρ = **−0.098 ~ −0.151（10/10 全负）**、mean±sd = **−0.126±0.019**；Pearson r = −0.144 ~ −0.230、mean±sd = −0.188±0.027；flipped ≈ aligned（逐例 ρ 差 ≤0.016、组级差 0.002，方向信息≈零——均值偏移响应近对称）。组均值（10 例平均 EC2 vs 平均 GC）：**Spearman −0.188 / Pearson −0.317**。响应量级：|Δmean| 中位 ~0.07 z（s12）/ 0.056 z（s24），非退化。
- 产物：`outputs/10-21-v5main-ec-explore/`（`ec_explore_stats.json` 逐被试+组级+方法1 失败宣告段；`m1/` 4 例证据 npz；`m2/` 10 例 npz；`README.md` 口径登记；`run01.log`/`run01_m2.log`）

### 结论

- **方法1 判失败（用户宣告）**：在「K=1 + 终点 40 Hz 产物口径 + 20% 熵涨幅阈值 + σ≤0.48」规格下，v5-main-01 靶通道熵对源噪声注入结构性不敏感；如未来重启需重设计（如 σ 上限外扩/阈值改绝对 nats/直接用响应方差比做连续读出），属新口径另立。
- **方法2 判读（探索性）**：+0.5std 源偏移的靶均值响应与 GC refs 呈**一致但微弱的负相关**（10/10 例同号，组均值 ρ −0.188 / r −0.317）——强 GC 边上源偏移反而压低靶预测均值（抑制型响应模式），方向与「兴奋性传导」直觉相反且幅值弱；方向性检查≈零说明读出无方向分辨。**该负号一致性本身是本探索的最强信号**，可供 Study I 迭代（损失侧跨通道监督）参考，但不构成可用 EC 推断。
- 派生登记：两条用户自设计快读出（熵失稳/均值偏移）与此前增益算子、NI-MSE 同向汇聚——v5-main θ* 的跨通道响应（无论噪声/常量/扰动形态）均未达观测 GC 结构；下一步重点在代理脑侧（Study I 损失侧跨通道监督迭代）。
