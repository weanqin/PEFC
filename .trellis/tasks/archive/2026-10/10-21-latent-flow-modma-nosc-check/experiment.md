# 实验日志 — latent-flow MODMA no-sc ckpt 本机检查与首跑

> 每次「跑批/训练/评估/统计」追加一个 Run 块：跑前先写「目的+配置」，跑后同会话补「结果+结论」，不留空块。
> 格式契约见 `.trellis/spec/research/experiment-log.md`；产物只落本任务声明的变体目录（`.trellis/spec/research/path-isolation.md`）。

## Run 001 — geometry.pt 重建（MNE GSN-129 模板 + ckpt pairwise_d 验证）（2026-10-21）

### 目的与假设

- 旧库 MODMA 线的 `geometry.pt`（GSN-HydroCel-128 电极坐标，mexhat 图 mixer 构建依赖）本机缺失且原物不可恢复（旧库 `.gitignore` 排除 `data/modma/`、7z 不含、git 全历史无）。假设：旧 geometry 即 MNE 标准 GSN-HydroCel-129 模板（E1–E128 序），可用 ckpt 内嵌 `flow.graph_mixer.pairwise_d` 逐位验证。

### 配置

- 代码：`models/latent-flow-v1/latent_flow/models/graph_utils.py`（load_geometry 契约，零改动）；重建逻辑为会话内联 python（命令全文见下）
- 参数：模板 `mne.channels.make_standard_montage('GSN-HydroCel-129')`，取前 128 导（E1–E128，第 129 导 Cz=参考位弃用），坐标米制 float32
- 环境：base conda（Python 3.12.3 / torch 2.8.0+cu128 / MNE 1.13.2）@ 本机，对照 environment.md
- 数据：基准 = `fold2_..._nosc.pt` 内嵌 `flow.graph_mixer.pairwise_d`（128×128 float32）；写入 `/root/autodl-tmp/modma/geometry.pt`（用户拍板「MNE 模板 + 验证」口径，2026-10-21）
- 命令（会话内联，关键行）：
  ```python
  mont = mne.channels.make_standard_montage('GSN-HydroCel-129')
  coords = np.stack([pos[n] for n in mont.ch_names[:128]]).astype(np.float32)
  torch.save({'coords': torch.from_numpy(coords), 'channel_names': mont.ch_names[:128]},
             '/root/autodl-tmp/modma/geometry.pt')
  ```

### 结果

- 模板原始序（E1–E128）对 ckpt `pairwise_d` 上三角 offdiag：**Pearson corr = 1.0000000000，max_abs_diff = 3.353e-08 m**（float32 量化级）→ 模板即原始几何来源，通道序与 MODMA 数据序一致，**未触发**匈牙利重排 / MDS 回退
- 尺度核对：模板坐标范数 0.079–0.110 m，ckpt 距离中位 0.1308 m / 最大 0.2079 m，一致
- 复读校验：带 `MODMA_DATA_ROOT=/root/autodl-tmp/modma` 时包内 `load_geometry('modma')` 返回 (128,3) float32 + E1..E128 ✓（注意：`config.GEOMETRY_PATH = DATA_ROOT/'geometry.pt'`，`DATA_ROOT` 优先取 env；无 env 时解析到包内 `data/modma/`——本机必须带 env 运行）
- 产物：`/root/autodl-tmp/modma/geometry.pt`（唯一写入数据根的工件）

### 结论

- 支持：geometry.pt 以「MNE GSN-129 模板」口径重建成立，验证逐位命中（corr=1.0、diff=3.4e-08 m），R1/R2 通过
- 备注：README-port.md 环境变量表中「MODMA 数据根 + 可选 geometry.pt」描述与 config 实现一致，但补全该工件前 MODMA 图线在本机不可跑——现已补全

## Run 002 — fold2 nosc ckpt 加载断言（2026-10-21）

### 目的与假设

- 验证旧库 MODMA SOTA no-sc ckpt 在本机经包内标准路径完整可用：自包含性（无需 stage1 VAE）、变体自动检测、buffer 逐位一致。

### 配置

- 代码：`latent_flow/scripts/_connectivity_driver.load_model`（零改动）
- 对象：`/root/autodl-tmp/latent_flow/checkpoints/fold2_oc_latent_flow_flow_best_graph_mexhat_screg_l0.01_w0.1_nosc.pt`
- 环境：同 Run 001（GPU cuda）
- 命令（会话内联）：`load_model(CKPT, cuda)` 后对 `flow.graph_mixer.W / pairwise_d` 做 `torch.equal` 逐位断言

### 结果

- state_dict 前缀 `{encoder, decoder, flow, proj}` **自包含**（stage1+stage2 一体；fold0–4 nosc 5 折同构；fold0 无同名 vae_best 不影响 flow 线）
- 加载横幅：`Loaded checkpoint: fold2_..._nosc.pt (1.24M params, stage-2 mexhat-graph flow, dataset=modma)`
- `W`、`pairwise_d`：`bitwise_equal=True`（128×128）
- 标量逐位一致：`log_sigma=-1.9590264558792114`、`gamma_m=1.0`、`gamma_buf=0.1`；ckpt 内 `gamma_sc=0.0` 为弃用 SC 先验残留键，loader 白名单静默丢弃（`allow_unexpected`，模型无此属性）——与 nosc 语义一致
- strict 断言零真实 missing/unexpected（loader 内建 assert 通过）

### 结论

- 支持：fold2 nosc ckpt 本机经标准路径完整加载，AC2 通过；nosc 变体语义（SC 先验归零弃用）与加载行为互证

## Run 003 — probe_oc_latent_flow fold2 nosc 首跑（2026-10-21）

### 目的与假设

- MODMA 原生线本机首跑（10-18 移植时「可跑未跑」项）：经包内标准 σ 扰动 probe 对 fold2 nosc 跑 1 名折内 test 被试，产出 per-target MSE 响应矩阵与 channel-sensitivity CV。

### 配置

- 代码：`latent_flow/scripts/probe_oc_latent_flow.py`（**零包体改动**；经 wrapper 注入 `N_CHANNELS=128` 后调用原 `main()`，见下方缺陷记录）
- 参数：`--fold 2 --subjects 1`（128 源 × σ∈{0.1,0.2,0.3}，n_steps=10，splitter seed=42）；BLAS 守卫 OMP/OPENBLAS/MKL=8；`MPLBACKEND=Agg`
- 环境：base conda @ 本机 GPU（4080 SUPER / cu128）
- 数据：fold2 test 第 1 名被试（MODMADataset 契约：逐被试逐通道 z-score，obs=375）
- 命令：
  ```bash
  cd /root/npi/models/latent-flow-v1 && OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 \
  MODMA_DATA_ROOT=/root/autodl-tmp/modma MPLBACKEND=Agg python3 -c "
  import sys; sys.path.insert(0, '/root/npi/models/latent-flow-v1')
  import latent_flow.scripts.probe_oc_latent_flow as P
  from latent_flow.scripts._connectivity_driver import N_CHANNELS
  P.N_CHANNELS = int(N_CHANNELS)
  sys.argv = ['probe_oc_latent_flow', '--fold', '2', '--checkpoint', '<fold2 nosc ckpt>', '--subjects', '1',
              '--output-dir', '/root/npi/outputs/10-21-latent-flow-modma-nosc-check/run_01/results',
              '--outputs-dir', '/root/npi/outputs/10-21-latent-flow-modma-nosc-check/run_01/outputs']
  P.main()"
  ```

### 结果

- 被试：fold2 test 第 1 名 = `02010010EOG.set_61_.mat`（169 窗），128 源 × σ∈{0.1,0.2,0.3}，每源前向 0.89 s，单 σ 114.2 s，总 ~5.7 min/σ × 3 ≈ 17 min（GPU 4080 SUPER）
- channel-sensitivity CV：**σ=0.1 → 318.18%，σ=0.2 → 587.12%，σ=0.3 → 712.18%**，脚本原样判读「ABOVE ceiling（JET v2 范围 9.5%–17.5%）」
- 响应幅值随 σ 近线性：mean_resp 1.298e-4 → 2.896e-4 → 5.628e-4；max_resp 5.42e-3 → 2.23e-2 → 5.25e-2
- forecast sanity：mean 0.0231 / std 0.8152 / abs_max 11.48
- 产物：`outputs/10-21-latent-flow-modma-nosc-check/run_01/results/probe_subject_0.json` + `outputs/.../outputs/subject_0_{comparison,out_degree_hist,sigma_*_heatmap}.png`（5 张）

### 结论

- 支持：MODMA 原生线（数据契约 → load_model → probe → 产物落盘）本机端到端跑通，AC3 通过；CV 量级与旧库「per-channel latent 打破 JET 通道塌缩天花板」的既有结论方向一致（首跑单被试存证，不做新判读、不外推到合格性）
- 边界重申：本任务不构成代理脑资格验收（三项检验另立任务）

## 缺陷记录（本任务发现，未修）

- **`probe_oc_latent_flow.py` 潜在 NameError（旧库 HEAD 同款，非移植引入）**：`parse_args` 中 `--sources` 的 `default=N_CHANNELS`，但该脚本 import 块漏了 `_connectivity_driver.N_CHANNELS`（定义于 driver 第 57 行）。首次 `run.py` 直跑即崩（argparse 构建期，无法用 CLI 绕过）。本任务以 wrapper 注入模块属性绕过，包体零改动；AST 静态排查确认同文件无其他未定义大写名。后续是否正式补 1 行 import（包体改动）留用户拍板。
