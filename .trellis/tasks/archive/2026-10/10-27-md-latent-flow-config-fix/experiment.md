# 实验日志 — 10-27-md-latent-flow-config-fix

> 每次「跑批/训练/评估/统计」追加一个 Run 块：跑前先写「目的+配置」，跑后同会话补「结果+结论」，不留空块。
> 格式契约见 `.trellis/spec/research/experiment-log.md`；产物只落本任务声明的变体目录（`.trellis/spec/research/path-isolation.md`）。
> 本任务为配置固定（无 GPU 跑批）；Run 001 = AC 只读验证（性质复核 + 语法编译 + 引用扫描）。

## Run 001 — AC 只读验证：v4 性质复核 + 语法/引用扫描（2026-10-27）

### 目的与假设

- 验证默认公式切 v4 后数学性质不变坏（全正/abs≡raw/重映射 Spearman 不变性/常数平移关系），且 refs v2 对比项活跃引用零残留、4 个改动 .py 语法可编译。

### 配置

- 代码：`models/latent-flow-v1/{ec_mild_unbiased,recompute_conn_v4,crossfold_summary_f12,viz_ec_pert}.py` + `refs/MANIFEST.md`（本任务工作区快照，改动见 git diff）
- 参数：σ 网格 log[0.01,0.48]×24（与主跑钉死口径同）；HALF_LOG_2PIE = 0.5·ln2πe；V4_OFFSET = 5/7
- 环境：base miniconda @ 本机（/root/npi），只读——不载入 ckpt、不写 outputs
- 数据：无外部数据（性质复核在合成网格上）
- 命令：`/root/miniconda3/bin/python -m py_compile <4 文件>` + 内联 numpy/scipy 性质脚本 + grep 扫描

### 结果

- 指标：
  - v4 网格全正：min = **+0.0293**（σ*=0.48 处），`abs==raw` 逐位 True；
  - v4 − v1 常数平移：min=max=**+1.133224**（= 0.5·ln2πe − 2/7，性质保持）；
  - spearman(conn_v4, 1/(σ*+1e-4)) = **1.000000000000**（重映射不变性保持）；
  - py_compile **PASS ×4**（末轮复编译 recompute/crossfold PASS ×2）；
  - 活跃 v2 引用（REFS 字典/REFS_KEYS/REFS_DIRS/加载路径/MANIFEST 表行）**零残留**；命中仅 3 处注销记注（注释/清单注记，供溯源）；
  - ec_mild_unbiased v1 计算路径零残留（计算行/payload 串/caliber/gccorr protocol/docstring 均 v4）。
- 产物：无文件产物（控制台验证，随本块登记）。
- 图表：无。

### 结论

- 支持：两项配置固定（refs 无 v2 对比项 + 线默认 v4）在代码层成立且性质无回归；v4 切换不改变任何秩相关读数（常数平移 + 严格单调等价），只改绝对值与全正性。
- 下一步 / 派生决策项：无新增待拍板；后续 md-latent-flow EC 任务直接以 v4 默认 + v1 单参考跑（recompute 断言计数新档将为 32/折）。
