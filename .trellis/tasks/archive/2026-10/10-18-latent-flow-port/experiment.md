# 实验日志 — 10-18-latent-flow-port（latent_flow 移植）

> 每次「跑批/训练/评估/统计」追加一个 Run 块：跑前先写「目的+配置」，跑后同会话补「结果+结论」，不留空块。
> 格式契约见 `.trellis/spec/research/experiment-log.md`；产物只落本任务声明的变体目录（`.trellis/spec/research/path-isolation.md`）。

## Run 001 — latent_flow 移植落位 + 裁剪 A + G1/G2 门禁（2026-10-18）

### 目的与假设

- 把 `/root/autodl-tmp/latent_flow`（源锚点 git HEAD `9efe1e3`，分支 08-15-substrate-cross-channel-probe，内容=工作树 2026-08-24 状态）移植为 npi 变体目录 `models/latent-flow-v1/`，执行用户拍板的裁剪 A（黑名单式删除 de/dysib/dcgnn/fm_gate 已终止线代码，判决文档全保留），并验证：包可导入、单测通过、脚本入口 import 闭包完整、shape_check GPU 冒烟可跑。
- 假设：OC 活跃面与 de/dysib/dcgnn/fm_gate 终止线物理可分（`__init__.py` 不 import 被删模块），裁剪后 pytest 与 import 冒烟双判据全绿。

### 配置

- 代码：`/root/npi/models/latent-flow-v1/`（新增变体目录；包本体 `latent_flow/` 嵌套，源码零逻辑改动，仅行尾统一 LF）
- 源库：`/root/autodl-tmp/latent_flow` 全程只读；恢复文件经 `git show HEAD:data/<f>` 物化
- 环境：base @ 本机（Python 3.12.3 / torch 2.8.0+cu128 / pytest 9.1.1，对照 `.trellis/research/environment.md`）
- 数据：`MODMA_DATA_ROOT=/root/autodl-tmp/modma`（S0.2 新解压，HC 29 / MDD 24 共 53 .mat；geometry.pt 不存在→2 项维持 skip）
- 命令（关键序列）：
  - `rsync -a /root/autodl-tmp/latent_flow/ /root/npi/models/latent-flow-v1/latent_flow/ --exclude={.git,.trellis,.zcode,.claude,.agents,.codex,archive,checkpoints,results,outputs,logs,experiments,.obsidian,__pycache__,.pytest_cache} --exclude={AGENTS.md,.gitignore,.gitattributes} --exclude='*.pdf'`
  - `git -C /root/autodl-tmp/latent_flow show HEAD:data/<f> > latent_flow/data/<f>`（9 文件）
  - `find … \( -name '*.py' -o -name '*.md' -o -name '*.csv' -o -name '*.yaml' \) -exec sed -i 's/\r$//' {} +`
  - G1：`cd latent_flow && OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 MODMA_DATA_ROOT=/root/autodl-tmp/modma python3 -m pytest tests/ -q`
  - G1：`for m in latent_flow/scripts/*.py; do PYTHONPATH=/root/npi/models/latent-flow-v1 python3 -c "import latent_flow.scripts.$(basename $m .py)" || echo "FAIL $m"; done`（从包父目录跑）
  - G2：`python3 /root/npi/models/latent-flow-v1/run.py shape_check_oc_latent_flow`

#### S0 前置结果

- S0.1 bundle 备份：`/root/autodl-tmp/latent_flow.bundle`（1.1 GB，`--all`）`git bundle verify` 通过（"The bundle records a complete history"，含 `9efe1e3` HEAD 与全部分支）。
- S0.2 MODMA zip 复测：**可读**。实测文件名为 `modma_resting_process.zip`（正确拼写 modma；研究期 ENOENT 实为引用了误拼名 `modda_resting_process.zip`——该名字的路径自然 ENOENT，目录项亦不存在）。PK 头有效，`unzip -l` 正常（53 .mat：HC 29 + MDD 24）。已解压 .mat 至 `/root/autodl-tmp/modma/{HC,MDD}/`（1017 MB，仅 .mat，无其他文件散落）。抽查 `HC/02020012EOG.set_76_.mat`：键 `data`、(128, 42501) float32，与旧库 data 契约一致。
- S0.3 geometry.pt：zip 内无、源库工作树与 git 索引均无（仅 `scripts/build_edra_geometry.py` 为 EDRA 侧生成器，无 MODMA 侧生成脚本）→ `test_graph_utils.py` 2 项 skip 维持，如实登记。

#### S1 落位与恢复

- S1.1 rsync：140 py + docs/norms/configs 等，2.4 MB；`configs/` 源树即为空目录（保留原样）。排除项复核零混入（无 .git/archive/pdf/AGENTS.md 等）。
- S1.2 恢复：`data/` 9 加载器（`__init__.py` 空 blob 属正常），合计 60,832 字节。
- S1.3 笔记：`experiments/*.md` 20 份 → `latent_flow/docs/experiments-notes/`（含中文名 `实验二.md`）。
- S1.4 行尾：转换前 CRLF 文件 107 个，`sed 's/\r$//'` 后全树 `grep -rIl $'\r'` = 0；`.gitattributes`（py/md/csv/yaml/txt eol=lf）与变体级 `.gitignore`（results/、.pytest_cache/、__pycache__/、*.pyc）落 `models/latent-flow-v1/`。
- S1.4b 逐字节核验：9 恢复文件 `git show HEAD:data/<f> | sed 's/\r$//' | diff - <dest>` 全部零 diff。

#### S1.5 裁剪冻结清单（删除执行前冻结；逐文件清单副本亦存本任务目录 `frozen_delete.txt` / `frozen_keep.txt`）

口径：黑名单式——删 de/dysib/dcgnn/fm_gate 已终止线代码；判决文档（docs/ + experiments-notes/）全保留。归属判定以 import 图为准。总 py 149 = 删 75 文件 / 29,864 行 + 留 74 文件 / 17,232 行。

**Delete（75 文件 / 29,864 行）**：

- `models/`：`de_vae.py dysib.py dcgnn_classifier.py fm_model.py`（fm_model.py 仅被 de/fm_gate 线 import）
- `analysis/de/`（7）+ `analysis/fm_gate/`（12）：整目录
- `training/`：`train_dysib.py train_dysib_v22.py`
- `scripts/` de 线（17）：`audit_de_vae_latent coord_audit sensitivity_uncertainty p0_diagnostics run_tlat_ablation extract_fm_gate_latents eval_de_vae eval_de_flow plot_de_vae plot_de_flow_aeff probe_de_flow_condition train_de_vae train_de_flow shape_check_de_flow run_de_response run_de_vae_d_ablation run_layerwise_audit run_mixer_probe`
- `scripts/` fm_gate 线（9）：`run_band_cond_gain run_cross_channel_probe run_fm_gate_controller run_fm_skill run_learnability_probe run_probe_matrix run_probe_raw run_task_snr_scan run_transition_audit`
- `scripts/` dysib 线（5）：`eval_dysib_probe_v21 eval_dysib_probe_v22 eval_dysib_v22_addendum plot_dysib_v22_figs precheck_dysib_task`
- `scripts/` dcgnn 线（4）：`train_dcgnn_classifier_fold0 train_dcgnn_full_cv train_dcgnn_window_fold0 run_ec_graph_comparison_full`（后者聚合 dcgnn 分类结果 json）
- `tests/`（14）：`test_band_cond_gain test_coord_audit test_dcgnn_classifier test_de_flow test_de_prep test_de_vae test_dysib test_dysib_v22_addendum test_fm_gate test_layerwise_audit test_mixer_probe test_probe_matrix test_probe_raw test_task_snr`

**Keep（74 文件 / 17,232 行）**，design 保护项全部落实：

- `models/graph_utils.py`（OC 共享）、`viz/probe.py`（probe_oc_latent_flow*.py import）、`scripts/run_oc_connectivity_global_tau.py` + `run_oc_connectivity_relative.py`（真实 OC 归因入口）、`scripts/build_edra_geometry.py`（EDRA shim 依赖）、`norms/`、`data/` 全部 9 文件（含 `de_features.py`——data 层组件，design「data/ 全留」）
- OC 归因 readout 家族（名字含 "de" 但属 σ 二分 DE readout，非 DE-VAE 线，经 docstring+产物路径 `results/oc_connectivity_*` 判定保留）：`_compare_signed_de_ablation _run_signed_de_ablation zscore_signed_de plot_de_signed_zscore plot_de_inference_zscore _plot_de_signed_heatmaps diag_de_vs_baseline`
- OC 图模块消融家族（07-15-graph-module-ablation，mexhat/SC/W 消融属 OC mixer）：`run_ablation_pipeline _summarize_ablation _plot_ablation_connectivity plot_component_ablation probe_component_ablation compare_params_vs_baseline`
- OC 主体与基线：`train/eval/diag/plot/probe_oc_latent_flow* shape_check_oc_latent_flow _connectivity_driver eval_fc_correlation plot_fc_correlation extract_graph_connectivity measure_obs_fc_latent_distance verify_dog_aeff plot_edra_training` + `baseline/` 全部 + `attribution/` 全部
- 保留测试（8 + __init__）：`test_baseline_connectivity test_bisect_sigma test_component_probe test_de_features test_gpdc test_graph_utils test_oc_latent_flow test_subject_classification`

**对 design 保护项的两处归属修正（import 图判据优先，冻结于此）**：

1. `scripts/coord_audit.py`：design 保护条件为「其测试保留时」；实测其模块级 import `latent_flow.models.de_vae.DEVAE` + `scripts.p0_diagnostics`（后者又 import `scripts.train_de_vae`）——按 import 图归 de 线，与 `models/de_vae.py` 删除互斥，故 coord_audit.py 与 `tests/test_coord_audit.py`（3 项既有 TODO skip）一并删除。保留它将使 G1 import 冒烟判据必然失败。
2. `tests/test_mixer_probe.py`：import `analysis.de.mixer_probe`（删除项）+ `models.oc_latent_flow`（保留项）；主被测对象为 de 线 mixer_probe，按测试归属删除。OC 侧 mexhat mixer（`LearnableMexHatMixer`）在 `models/oc_latent_flow.py` 内保留，`verify_dog_aeff.py` 承担其核验。

**静态闭包核验（删除前）**：保留 74 py 对被删模块（`models.{de_vae,dysib,dcgnn_classifier,fm_model}`、`analysis.{de,fm_gate}`、`training.train_dysib*`、被删 script 模块）的 import 引用 = 0 违例。

### 结果

- **裁剪执行与核对**：删除后实际剩余 py 集 vs 冻结 keep 列 `comm`/`diff` 零差（剩余 74 py / 17,232 行；删除 75 文件 / 29,864 行）；空目录 `analysis/de`、`analysis/fm_gate`、`analysis/` 已移除（analysis 子包整体无保留价值：其仅含被删两子目录）。
- **S1.5b per-file 冻结计数**（MODMA_DATA_ROOT=/root/autodl-tmp/modma、BLAS 8 线程）：
  | 测试文件 | 计数 |
  |---|---|
  | test_baseline_connectivity | 7 passed |
  | test_bisect_sigma | 15 passed |
  | test_component_probe | 7 passed |
  | test_de_features | 13 passed |
  | test_gpdc | 7 passed |
  | test_graph_utils | 8 passed, 2 skipped（geometry.pt 缺，既有口径） |
  | test_oc_latent_flow | 10 passed |
  | test_subject_classification | 15 skipped（见下偏差 2） |
  | **冻结 G1 预期** | **67 passed / 0 failed / 17 skipped** |
- **G1 门禁（三判据全过）**：
  1. 全量 pytest：`67 passed, 17 skipped in 13.16s` —— 与冻结口径逐位一致，0 failed；
  2. 全部 31 个保留脚本 import 冒烟（`PYTHONPATH=/root/npi/models/latent-flow-v1`，从包父目录跑）：**0 失败**；
  3. 行尾硬检查 `grep -rIl $'\r' /root/npi/models/latent-flow-v1 | wc -l` = **0**。
- **G2 门禁（通过）**：`python3 /root/npi/models/latent-flow-v1/run.py shape_check_oc_latent_flow` → 退出码 **0**；自检打印 `OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8`（≤8）；`Device: cuda`（RTX 4080 SUPER）；OCLatentFlow 5.36M 参数，stage1 / stage2 / inference sample(use_proj=False/True) 全前向形状自洽（sample out (2,500,128)、fc slice (2,125,128)），结尾 "Done."。GPU 冒烟（S3.1=AC3）由此承担。
- **S3.2 干净性三查**：`git status --short` 排除 `latent-flow-v1` 与 `.trellis` 后为空（完整变更仅 `?? models/latent-flow-v1/`、`?? .trellis/tasks/10-18-latent-flow-port/`、既有前会话遗留 ` M .trellis/research/environment.md` 不属本任务）；变体内 `.pt` 文件数 = 0；`.pytest_cache`/`__pycache__` 已清理。
- **产物**：无落盘产物（shape_check 为 torch.randn 合成前向自检，stdout-only；`outputs/latent-flow-port/` 未创建——AC3 按「登记无需产物」口径满足）。变体目录最终 107 文件 / 75 py（74 包内 + run.py）/ 1.9 MB。
- **包装层**：`models/latent-flow-v1/run.py`（BLAS 守卫 + PYTHONPATH 注入子进程 + 脚本名转译 + `--output-root` 注入 + baseline `--n-jobs 8` 默认）+ `README-port.md`（两种入口/env var 表/baseline 警告/checkpoints 原位/裁剪口径与锚点/边界声明/FOUNDATION 数字/MODMA 状态/train 阻塞说明）。

### 偏差与处置（对照 implement.md 预期）

1. **研究期 MODMA ENOENT 定性更正（S0.2）**：zip 实际可读，真实文件名为 `modma_resting_process.zip`；研究期（及任务提示）引用的 `modda_resting_process.zip` 为误拼（该路径确不存在，`find -ls` 亦无目录项）。已解压 53 .mat 至 `/root/autodl-tmp/modma/` 并挂 `MODMA_DATA_ROOT`。此项应在 3.3 spec update 时同步更正 environment.md 登记。
2. **test_subject_classification 预期口径偏差（S1.5b）**：implement.md 预期「4 失败转通过」；实测该文件有模块级 `pytestmark = skipif(not _DE_EC_DIR.exists())` 守卫（`tests/test_subject_classification.py:32`，`_DE_EC_DIR = results/oc_connectivity_global_tau_signed_signal_q0.5`）——研究结果中的 4 个失败仅在 results/ 产物在位（沙箱符号链接）时才触发执行。本移植按设计排除 results/，故 15 项全部 skip（0 failed）。MODMA 原始数据依赖已解除（zip 已修），但该模块还需 fold-0 DE-EC 归因产物（需旧 ckpt 重跑 `run_oc_connectivity_global_tau` 可再生）——属数据产物依赖而非环境问题，如实登记，转通过条件与 EDRA shim 一并留给后续任务。
3. **run.py 首版缺陷（S2.3 首跑）**：子进程 `python -m` 无法解析 latent_flow（PYTHONPATH 未传递），修正为向子进程 env 注入 `PYTHONPATH=<变体目录>` 后 G2 通过（保持调用方 cwd 不变）。
4. **裁剪归属修正 2 处（S1.5 冻结清单已详述）**：coord_audit.py + test_coord_audit.py（import 图归 de 线，design 保护条件「其测试保留时」不成立）；test_mixer_probe.py（主被测 `analysis/de/mixer_probe.py`）。

### 结论

- **支持**：裁剪 A 后 OC 活跃面自洽成立——pytest 67P/0F/17S 与冻结口径逐位一致、31 脚本 import 冒烟零失败、GPU 合成前向冒烟退出码 0，G1/G2 双门禁通过；latent-flow-v1 已具备作为 npi 第三条实现路径（Study I 候选）的可运行基础。
- MODMA 数据线已恢复可用（zip 可读性为文件名误拼所致），geometry.pt 与 fold-0 DE-EC 产物两缺口如实维持 skip，不阻塞本任务验收。
- 下一步 / 派生决策项（转 S4 与后续任务）：S4.1 trellis-check；S4.2 spec update（context.md 技术路线表回填 latent-flow 条 + environment.md 登记移植事实、MODMA 更正、dos2unix 未装 sed 替代惯例）；EDRA npz shim 与 MODMA 线训练/归因补跑另立任务。
