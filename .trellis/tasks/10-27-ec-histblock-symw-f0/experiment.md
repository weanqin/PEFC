# 实验日志 — <任务名>

> 每次「跑批/训练/评估/统计」追加一个 Run 块：跑前先写「目的+配置」，跑后同会话补「结果+结论」，不留空块。
> 格式契约见 `.trellis/spec/research/experiment-log.md`；产物只落本任务声明的变体目录（`.trellis/spec/research/path-isolation.md`）。

## Run 001 — <一句话名称>（YYYY-MM-DD）

### 目的与假设

- 要回答什么问题 / 验证什么假设？

### 配置

- 代码：`<变体目录/文件 + 版本快照>`
- 参数：<与默认不同的超参 / 阈值 / seed>
- 环境：<env 名 @ 机器（本地/远程），对照 .trellis/research/environment.md>
- 数据：<输入路径 + 筛选口径（qc_pass、分组等）>
- 命令：<可复现启动命令>

### 结果

- 指标：<数值 / 小表>
- 产物：`<本变体 outputs/... 路径>`
- 图表：<路径 + 一句话说明>

### 结论

- 支持 / 证伪 / 不确定 + 一句话依据
- 下一步 / 派生决策项：<同步到 .trellis/research/progress.md>

## Run 001 — hist-block 全量主跑（2026-10-27）

### 目的与假设
- 首次 hist-block 噪声范式全量 EC：8 例 σ*/τ/conn + P(resp>τ) 口径健康度。

### 配置
- 代码: models/latent-flow-v1/ec_mild_unbiased.py @ fe17225（hist-block 已交付版，方法零改动）
- 代理脑: run_symw_fold0（用户标识 10-25-md-lf-anti0.2-gnnexit；anti0.2+gnnexit+sym_w 对拍唯一匹配）
- 参数: --symw --fold 0 --noise-mode hist-block（全共享默认）--subjects 8 --task-label 10-27-ec-histblock-symw-f0；其余默认 q=0.5/σ_ref=0.3/log[0.01,0.48]×24/seed42/30trial
- 命令: OMP_NUM_THREADS=8 /root/miniconda3/bin/python models/latent-flow-v1/ec_mild_unbiased.py --mode main --symw --fold 0 --noise-mode hist-block --subjects 8 --task-label 10-27-ec-histblock-symw-f0 --out-dir /root/npi/outputs/10-27-ec-histblock-symw-f0/ec_histblock
- 环境: 本机 RTX 4080 SUPER（GPU 空闲已核）

### 结果

- 主跑：8/8 例完成、52.5 min（6.55–6.57 min/例），fold0 test s{1,5,7,10,21,25,27,30}
  （MDD4/HC4），逐例 json+curves.npz 落 `outputs/10-27-ec-histblock-symw-f0/ec_histblock/`
- 全例 P(resp>tau)=0.500 精确；τ 逐例 5.13e-05–1.214e-04；interior 51.7–71.7%
  （组均 62.9%）@min 8.1% @max 28.9%
- summary（Run 002）：dyn_ratio_med 2.17×、sigma_median_offdiag 0.3017、
  nonmono 0.9922、conn_v4 mean_offdiag +0.822（0.592–1.070）、方差分解
  靶份额 0.819 / 源份额 0.018 / 残差 0.164、跨被试 stab rho 0.381

### 结论

- hist-block 范式全量主跑健康：量化口径（P=0.500、interior、sigma 分布）与 iid 三跑
  同型；与 iid-symw 同代理脑并置（同 v4 公式）：interior 62.9 vs 64.2%、@max 28.9 vs
  29.5%、sigma_median 0.3017 vs 0.3023、nonmono 0.9922 vs 0.9922、conn_v4 0.822 vs
  0.756（差在被试间散布内）、τ 略低（~5–15%）、dyn 略平（2.17 vs 2.55×）、靶主效应
  更主导（0.819 vs 0.753）、stab 略降（0.381 vs 0.425）
- 判读口径：hist-block 与 iid 的数值差异含噪声谱形因素（15 段阶跃低频集中 vs 白噪
  声），描述性并置、不作跨范式定量结论；EC 有效性判断 provisional 随 symw θ 拍板联动
- 下一步（待拍板）：hist-block gccorr（GC 对照）是否跑；三代理脑 hist-block 复扫是否扩

## Run 002 — 组级 summary（离线，2026-10-27）

### 目的与假设
- AC3：组级分布/方差分解/跨被试稳定性落档。

### 配置
- 命令：`--mode summary --symw --task-label 10-27-ec-histblock-symw-f0 --main-dir
  .../ec_histblock --out-dir .../ec_histblock`（离线无 GPU）

### 结果
- group_summary.json + group_mean.npz 落 `outputs/10-27-ec-histblock-symw-f0/ec_histblock/`
- 指标见 Run 001 结果段

### 结论
- AC3 过；重映射 Spearman 不变断言 8/8 PASS（summary 内建）。
