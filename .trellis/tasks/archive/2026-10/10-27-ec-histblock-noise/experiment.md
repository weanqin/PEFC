# 实验日志 — 10-27-ec-histblock-noise

> 每次「跑批/训练/评估/统计」追加一个 Run 块：跑前先写「目的+配置」，跑后同会话补「结果+结论」，不留空块。
> 格式契约见 `.trellis/spec/research/experiment-log.md`；产物只落本任务声明的变体目录（`.trellis/spec/research/path-isolation.md`）。

环境统一：本机云服务器（environment.md 登记：torch 2.8.0+cu128 @ RTX 4080 SUPER、
miniconda3 base Python 3.12.3）；代理脑 = w500 fold0 stage2 best（在档默认 ckpt）；
数据 = Mild Depression fold0 test（manifest SHA 钉死断言通过）。

## Run 001 — 改动前 iid 默认基线干跑（2026-10-27）

### 目的与假设
- 为 AC1（默认调用面逐位等价）固定改动前基线锚点。

### 配置
- 代码：`models/latent-flow-v1/ec_mild_unbiased.py` @ git 01c731e（改动前，工作区净）
- 参数：默认全参（iid 噪声、q=0.5、σ_ref=0.3、log[0.01,0.48]×24、seed 42、30 trial）+
  `--subjects 1 --sources 8`
- 命令：`OMP_NUM_THREADS=8 /root/miniconda3/bin/python models/latent-flow-v1/ec_mild_unbiased.py --mode main --subjects 1 --sources 8 --out-dir /root/npi/outputs/10-27-ec-histblock-noise/baseline_iid_pre`

### 结果
- 指标：0.39 min；tau=1.692e-04；interior 51.8% / @min 12.6% / @max 35.6%；P(resp>τ)=0.500
- 产物：`outputs/10-27-ec-histblock-noise/baseline_iid_pre/subject_0.{json,_curves.npz}`

### 结论
- 基线锚点在档；改动后同配置对拍见 Run 002。

## Run 002 — 改动后 iid 默认对拍 + hist-block 构造检查（2026-10-27）

### 目的与假设
- AC1（默认等价逐位）：hist-block 代码并入后，默认旗标路径输出与 Run 001 逐位一致；
- AC2/AC3/AC4/AC6 构造面：五项检查脚本全过。

### 配置
- 代码：`ec_mild_unbiased.py` 本任务改动版（+141/−6；py_compile 过）
- 命令①（对拍）：同 Run 001，`--out-dir .../post_iid_default`
- 命令②（构造检查）：`/root/miniconda3/bin/python .trellis/tasks/10-27-ec-histblock-noise/research/test_hist_block_noise.py`（CPU）

### 结果
- 对拍：`sigma_matrix` / `raw_response_matrix` / `mean_diff_matrix` 逐位 IDENTICAL；
  `tau_global` 相等；`curves.npz` 五数组逐位 IDENTICAL；iid payload hparams 无 hist 键
  （键面不变）
- 构造检查：15/15 PASS——V0 构造性（与独立 randn(30,15)+repeat_interleave 逐位一致）、
  V1 同被试跨通道（src=0/1/127 同引用 torch.equal）、V2 跨被试（独立重建 torch.equal）
  + 键控退化生效、V3 组内 25 点恒等（15/15 组）、V4 15 组 + 组代表=基抽取 + N=200k
  相关 sanity |offdiag|max=0.0078<0.01、V5 CRN 结构幂等 + 口径断言负例（hist_len≠375
  拒跑 / 非整除拒跑）+ 默认面
- 产物：`outputs/10-27-ec-histblock-noise/post_iid_default/`；检查日志在会话记录

### 结论
- **AC1/AC2/AC3/AC4/AC6 全过**：默认行为逐位不变；hist-block 共享与分组契约成立。

## Run 003 — hist-block GPU 全链路冒烟（2026-10-27）

### 目的与假设
- hist-block 模式端到端可跑（AC5 集成面 + payload 契约 AC7）；量化性质 P(resp>τ)=0.5
  在新噪声结构下保持。

### 配置
- 命令：`--mode main --subjects 2 --sources 8 --noise-mode hist-block --out-dir .../smoke_histblock`（其余默认）

### 结果
- subject_0 (s1)：0.39 min，tau=1.582e-04（≠iid 基线 1.692e-04 ⇒ 噪声结构实变），
  interior 50.9%，P(resp>τ)=0.500
- subject_1 (s5)：0.38 min，tau=8.470e-05，interior 70.9%，P(resp>τ)=0.500
- payload：hparams 含 7 个 hist 键（noise_mode/hist_len/hist_group_size/hist_n_groups/
  hist_share_channels/hist_share_subjects/hist_xi_seed_offset=9375）；
  noise_model = unbiased_gaussian_mu0_hist_block（含重建配方串）
- 产物：`outputs/10-27-ec-histblock-noise/smoke_histblock/subject_{0,1}.{json,_curves.npz}`

### 结论
- **AC5/AC7 集成面过**：hist-block 全链路健康，逐点量化口径不被破坏；变体生效且
  可与 iid 区分。方法开关交付完成；**hist-block 全量 8 例主跑未跑**（范围外，待用户
  另行决策跑哪个代理脑/fold）。
