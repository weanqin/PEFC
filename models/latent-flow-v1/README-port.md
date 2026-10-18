# latent-flow-v1 — latent_flow 移植说明（npi 侧）

> 本目录是 `/root/autodl-tmp/latent_flow`（旧研究库，Windows 机，2026-06~08）向 npi
> 的**目标实现路径移植**。源锚点：git HEAD `9efe1e3`（分支 `08-15-substrate-cross-channel-probe`，
> 领先 main +204 文件/+44k 行），内容基线 = 工作树 2026-08-24 状态 + `data/` 9 个加载器经
> `git show HEAD:data/<f>` 物化恢复（7z 打包删失）。源库全程只读，完整 git 历史与被裁线代码
> 已备份至 `/root/autodl-tmp/latent_flow.bundle`（1.1 GB，`git bundle verify` 通过）。
>
> 包本体 `latent_flow/` **零逻辑改动**：仅行尾统一 LF（源工作树 107 个文件为 CRLF）；
> 一切 npi 适配经本目录包装层（`run.py` / 本 README / `.gitattributes` / `.gitignore`）。

## 运行方式（两种）

### 推荐：run.py 包装入口

```bash
python3 /root/npi/models/latent-flow-v1/run.py <脚本名> [原样透传参数]
# 例（GPU 合成前向冒烟，秒级）：
python3 /root/npi/models/latent-flow-v1/run.py shape_check_oc_latent_flow
```

- 首个位置参数 = 脚本名（`latent_flow/scripts/` 优先，其次 `baseline/` 等子包），其余参数
  **不经 `--` 分隔原样透传**给 `python -m latent_flow.<子包>.<脚本名>`。
- 内置 BLAS 线程守卫（numpy/torch 启动前 setdefault OMP/OPENBLAS/MKL 等 ≤8；本机 cgroup
  有效 16 核而可见 128 核，防线程超订）；自检行打印解析后的线程数。
- 输出落点：对已核实支持 `--output-dir` 类 flag 的脚本（`run_oc_connectivity_{global_tau,relative}`、
  `extract_graph_connectivity`、`probe_oc_latent_flow_latent`、`probe_component_ablation`、
  `run_baseline_connectivity`），默认注入 `--output-root`（默认
  `/root/npi/outputs/latent-flow-port/`，可用 `--output-root <dir>` 覆盖）下的路径；
  用户显式传对应 flag 时不注入。
- baseline 并行：`run_baseline_connectivity` 上游默认 `--n-jobs -1`（全部可见核），run.py
  默认注入 `--n-jobs 8`（16 核 cgroup 口径）；显式传 `--n-jobs` 时不注入。

### 直跑：`-m` + env 前置（绕过守卫属使用者责任）

```bash
cd /root/npi/models/latent-flow-v1
OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 MODMA_DATA_ROOT=/root/autodl-tmp/modma \
  python3 -m latent_flow.scripts.<脚本名> [参数]
```

包内无守卫，直跑**必须**自行前置 BLAS env（上例写法照抄即可）。包自 README
（`latent_flow/README.md`）为旧库原文，运行方式以其为准但 env 前置以本文件为准。

## 环境变量表（`latent_flow/config.py:24-44`，全部先查 env）

| 变量 | 用途 | 本机指向 |
|---|---|---|
| `MODMA_DATA_ROOT` | MODMA 数据集根（`HC/`、`MDD/` 子目录 + 可选 `geometry.pt`） | `/root/autodl-tmp/modma`（2026-10-18 从 `modma_resting_process.zip` 解压：HC 29 / MDD 24 共 53 个 .mat，键 `data`、(128, ~42501) float32） |
| `EDRA_DATA_ROOT` | 旧库 EDRA 数据根（**BrainVision .vhdr 契约**，本机 0 个 .vhdr） | 暂无可用指向；需 npz→Descriptor shim（后续任务，见下边界声明） |
| `SC_DATA_ROOT` | OSF 对照数据根（次要） | 本机未在位 |
| `MODMA_DEVICE` | 设备覆盖（缺省 `cuda if available`） | 缺省即可（本机 4080 SUPER / cu128） |

注意：`geometry.pt`（GSN-HydroCel-129）在 MODMA 数据根缺失 → `test_graph_utils.py`
2 项维持 skip；`MODMA_DATA_ROOT` 下亦无生成脚本（`scripts/build_edra_geometry.py`
仅生成 EDRA 侧 `geometry_edra.pt`）。

## baseline 相对路径产物警告

`run_baseline_connectivity.py` 等脚本的默认产物路径是 **cwd 相对** `results/...`、
`outputs/...`（如 `DEFAULT_RESULTS_DIR = "results/baseline_connectivity"`）。npi 根
`.gitignore` **未排除** `results/`。因此：

- 勿在 npi 仓库根裸跑 `-m` baseline 脚本（会在仓库根落 `results/` 目录）；
- 经 `run.py` 跑则默认注入 outputs 区路径（见上），无此问题；
- 变体级 `.gitignore` 已加 `results/`、`.pytest_cache/` 兜底（仅覆盖本变体目录内）。
- `run_ablation_pipeline.py` / `_summarize_ablation.py` 等无 CLI 输出参数的脚本硬编码
  相对 `results/`——只应在包目录 `latent_flow/` 内运行（被变体 .gitignore 兜底）。

## 旧 θ\* checkpoints（558 MB，原位不随移）

- 原位：`/root/autodl-tmp/latent_flow/checkpoints/`（80 个 .pt：fold0 OC-LatentFlow
  全系权重 `fold0_oc_latent_flow_flow_best_graph_mexhat*.pt` 等、gru/var forecaster、
  `de_flow/ de_vae/ dysib/ edra/` 子目录）。**复现 FOUNDATION 数字与后续对照的唯一凭据**。
- 新训练产物落包内 `latent_flow/checkpoints/`（`CHECKPOINT_DIR` 由 `_PROJECT_ROOT` 派生），
  npi `.gitignore` 的 `*.pt` 规则排除入库。
- 旧 θ\* **不自动继承 npi 代理脑资格**：固定 θ\* 前须按
  `.trellis/spec/research/surrogate-brain-qualification.md` 三项检验重新验收。

## 裁剪口径与源锚点

- 裁剪 = 用户拍板的 **A 案（黑名单式）**：删除 de-VAE / de-flow / dysib / DCGNN /
  fm_gate 已终止线代码（75 文件 / 29,864 行），**判决文档全量保留**（`latent_flow/docs/`
  全部 + `docs/experiments-notes/` 20 份判决笔记）。裁剪后 74 py / 17,232 行。
- 源锚点：`9efe1e3`。冻结两列清单（delete/keep）存任务目录
  `.trellis/tasks/10-18-latent-flow-port/frozen_{delete,keep}.txt` 与该任务 experiment.md。
- **未来读者注意**：`docs/` 与判决笔记会引用 de/dysib/dcgnn/fm_gate 代码与
  `results/`/`outputs/` 产物——这些**不在本移植内**，考古走源库原位
  `/root/autodl-tmp/latent_flow`（只读）或 bundle（`git clone /root/autodl-tmp/latent_flow.bundle`）。
- 两处归属修正（import 图判据，详见任务 experiment.md Run 001）：
  `scripts/coord_audit.py` + `tests/test_coord_audit.py`（模块级 import DEVAE，归 de 线删）；
  `tests/test_mixer_probe.py`（主被测对象为 `analysis/de/mixer_probe.py`，随删）。
  名字含 "de" 但属 σ 二分 DE readout 的脚本（`_compare_signed_de_ablation`、
  `zscore_signed_de` 等 7 个）与 OC 图模块消融家族（07-15 任务）**保留**。

## Study I / II 边界声明（design §5 摘要，移植≠融合）

- **σ 二分 ≠ perturb-ec-v1 四要素**：算子（源通道加性高斯 vs 增益乘性）、度量（逐靶 MSE
  vs CRN 配对 RMS）、确定性（路径确定性隐式 vs 显式 CRN）、τ（共享中位数阈 vs 相对自响应
  τ̃）均不同；方向语义一致（行=源、列=靶，σ\* readout 自带翻转使大=强，与 npi
  `larger_is_stronger` 同向）。对照表见任务 research/latent-flow-adaptation.md §4.2。
- **EDRA npz shim 属后续任务**：旧库吃 BrainVision .vhdr（本机 0 个），npi 现行契约为
  npz（edra_process_v4 @500Hz / edra_processed_fmt @250Hz）；窗长 375/125 vs 384/128。
  shim = npz Descriptor + reader + `geometry_edra.pt` 生成（依赖已随移的
  `scripts/build_edra_geometry.py`）。
- **train `--smoke` 数据阻塞**：`--smoke` 仅设 patience=0 禁存 ckpt，仍走真实 cohort 加载
  （`train_oc_latent_flow.py:490-497`；`--dataset` 仅 modma/edra）。MODMA 已可加载（见上
  env 表）但经 npz 管线的 EDRA 不可；本任务 GPU 冒烟由 `shape_check_oc_latent_flow`
  （`torch.randn` 合成自足）承担，训练冒烟待 shim 后补跑（MODMA 线现已具备数据条件，
  可在后续任务中直接以 `--dataset modma` 启动）。
- 移植后 npi 出现第三条实现路径：Study I 侧 gpfm-mamba 系与 latent-flow 并存（候选关系
  留后续实验任务裁决），互不写对方目录。

## FOUNDATION 定量基线（`latent_flow/docs/FOUNDATION.md`，移植后可比对）

- 跨被试 Pearson：0.172（proj）/ 0.157（flow）
- Mexican-hat mixer 引入后：0.449 → 0.747
- DE vs GC r_abs：0.585（MODMA fold2）/ 0.413（EDRA）
- offdiag CV 128%；距离衰减 r = 0.842

## 测试面（本机实测，2026-10-18）

裁剪后全量 pytest（`cd latent_flow && OMP_NUM_THREADS=8 ... MODMA_DATA_ROOT=/root/autodl-tmp/modma
python3 -m pytest tests/ -q`）：**67 passed / 0 failed / 17 skipped**（13.2s）。
17 skip = `test_subject_classification` 15 项（模块级守卫要求 fold-0 DE-EC 产物
`results/oc_connectivity_global_tau_signed_signal_q0.5`，未随移——需旧 ckpt 重跑归因可再生）
+ `test_graph_utils` 2 项（`geometry.pt` 缺）。全部 31 个保留脚本 import 冒烟零失败。
