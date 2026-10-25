# 实验日志 — 10-25-latent-flow-exit-gnn-symw

> 每次「跑批/训练/评估/统计」追加一个 Run 块：跑前先写「目的+配置」，跑后同会话补「结果+结论」，不留空块。
> 格式契约见 `.trellis/spec/research/experiment-log.md`；产物只落本任务声明目录（path-isolation）。

## Run 000 — 实施自验 + 旗标关 bit 兼容对拍 + --sym-w 冒烟（2026-10-25）

### 目的与假设

- 变体接线（SymWAdam 逐步投影 + `--sym-w` 旗标）在真实 wrapper 全链路成立：每步投影后
  W 严格对称、uni_reg 以 λ=0.2 生效、config 凭据落盘、守卫拒跑；旗标关闭时默认路径
  行为不变。

### 配置

- 代码：`flow_exit_gnn_symw.py`（新，SymWAdam = Adam 子类，step() 尾部
  `W ← (W+Wᵀ)/2` 原位投影，目标 `flow.graph_mixer.W`）+ `train_mild_depress.py`
  接线（旗标/四守卫/口径守卫/config 凭据；`train_args` Namespace 透传 `sym_w`——
  实施代理发现的静默失效点，已补）；包体 `latent_flow/` 零 diff。
- bit 兼容对拍：改动前（HEAD）/改动后（旗标关）/改动后重跑 三方，同命令 2ep/2例
  （exit-gnn Run 000 同款：λ=0.1 + --flow-exit-gnn，无 --sym-w）。
- 冒烟：上命令基础上 λ→0.2 + `--sym-w`，`--run-dir .../smoke/r000`。
- 环境：npi @ 4080S（本机，environment.md 在档）；VAE 只读复用 w500
  `run_w500_fold0/checkpoints/fold0_mild_vae_best.pt`（best 2.0992@E57）。

### 结果

- **CPU 自验**（`python3 flow_exit_gnn_symw.py`）：randn 非对称 W + 随机梯度一步
  SymWAdam 后 `torch.equal(W, Wᵀ)` 严格对称；A_eff 对称；参数量 base 488,897 →
  exit-gnn 521,921（+33,024 复验）；SymWAdam 零参数增量。
- **bit 兼容（方法学发现）**：三方 val E2 loss = 2.391752 / 2.391675 / 2.391650，
  ckpt 全张量 max|Δ| = 1.43e-3（pre vs post）**vs 1.42e-3（post vs post2 同代码重跑）**
  ——差异量级与同代码重跑噪声完全一致 ⇒ **exit-gnn stage2 GPU 路径（bf16 AMP/cuDNN）
  本就非逐位确定**，改动前后差异全部落在重跑噪声内；bit 兼容凭据 = ①新增逻辑全在
  `args.sym_w` 分支内（代码面检查）+ ②数值一致至重跑噪声内。**修正预期：本路径此后
  不做逐位对拍，只做噪声内一致性检查**（w500 任务在档的逐位等价口径不适用于
  unfreeze-decoder bf16 路径）。
- **实施期静态核验**（implement agent）：守卫五条拒跑路径实测（stage≠2 / dense 互斥 /
  gaussian 互斥 / 须 --unfreeze-decoder / 新口径须显式 --run-dir），`--help` 旗标在列，
  py_compile 过，`git diff latent_flow/` 空。
- **冒烟 Run 000**：wall 26.6 s；打印 `lambda=0.2` + `forced W symmetrization: ON`；
  训后 ckpt `flow.graph_mixer.W` **torch.equal 级严格对称（max|W−Wᵀ| = 0.0）**且已从
  零初始化打开（max|W| = 3.1e-3）；config 凭据 `sym_w: true` / `flow_variant:
  graph-mexhat-exit-gnn-symw` / `cli.anti_uniform_lambda: 0.2` / `stage2.sym_w: true`；
  loss 算术核验 loss−(recon2+corr2+cons2) = 0.2·uni_reg（残差 2.58e-3 逐位吻合）；
  2ep val 2.3924。

### 结论

- 接线假设全成立：逐步投影生效且不扼杀学习（W 已开）、λ=0.2 精确入 loss、凭据齐、
  默认路径不受影响（噪声内一致）。放行 Run 001 正式训练。
- 顺带发现登记：`torch.cdist`（CPU，P=128>25）自身 ~1 ulp 不对称 ⇒ 真实 geometry 下
  A_eff 对称度受 M 侧 ~2e-7 地板约束（W 本体不受影响，仍严格对称）。

## Run 001 — stage2 正式 · W 强制对称化 + λ0.2（exit-gnn 基座衍生）（2026-10-25）

### 目的与假设

- 用户变体假设：exit-gnn 基座上提高 anti-uniform λ 至 0.2 并每步强制 W 对称
  （A_eff 退化为无向混合），检验 fc 生成质量（幅度校准/收敛性）与 val 的作用——
  相对 exit-gnn Run 001 的同口径对照。**两因素（λ 0.1→0.2 与 sym-W）同变，单变量
  归因须谨慎（登记）。**

### 配置

- 命令：`--fold 0 --stage 2 --window-len 500 --obs-len 375 --forecast-len 125
  --batch-size 64 --lr 1e-3 --epochs 150 --patience 30 --seed 42 --unfreeze-decoder
  --anti-uniform-lambda 0.2 --flow-exit-gnn --sym-w --vae-ckpt <w500 Run 001 VAE ckpt>
  --run-dir outputs/10-25-latent-flow-exit-gnn-symw/run_symw_fold0`（其余 = exit-gnn
  Run 001 逐项同参）
- 模型：flow 521,921 / 全模型 1,276,258 / stage2 trainable 936,482（与 exit-gnn 逐位
  同构——sym-W 仅训练期投影，零参数增量）；A_eff 入口/出口共享、W 对称约束下训练
- 产物：`outputs/10-25-latent-flow-exit-gnn-symw/run_symw_fold0/{configs,checkpoints,logs}`
  + `run_symw_fold0_train.log`（nohup）

### 结果

- **best val 0.8054@E144，150ep 顶格未早停**（末 10 ep val 0.81–0.92 波动，E144 仍在
  改进 ⇒ patience 未触发）；wall 736.2 s（exit-gnn 706.8 s，+4%）。
- best-epoch 分量：recon2 0.412 / corr2 0.195 / cons2 0.196 / uni_reg 0.0154
  （贡献 0.2×0.0154≈0.003）。
- **训后 best ckpt W 对称断言：torch.equal 级（max|W−Wᵀ| = 0.0）**；W 已学到实质
  对称结构（max|W| 1.59，对照冒烟 2ep 时 3.1e-3）。
- 对照：exit-gnn Run 001 best 0.8241@E116（早停 E146）/ w500 Run 002 best 0.8356@E150
  顶格 ⇒ **本变体 val 三者中最优**。
- smoke：MAE 0.8974 / std_ratio 0.8170（exit-gnn 0.8874/0.7757，w500 0.9581/0.8947
  ——std_ratio 较 exit-gnn 回升 +0.041，幅度预警缓解迹象，移交 Run 002 判读）。

### 结论

- 训练动力学假设成立：对称约束 + λ0.2 下 val 优于两个前序基座，150ep 预算内 best 出现在
  E144（贴顶格，收敛性弱于 exit-gnn 的早停@E146 语境下同为预算尾部，两跑均未给出
  离开噪声平台的证据）。
- 幅度校准判读移交 Run 002；两因素同变 caveat 在档（如需归因：exit-gnn + λ0.2 不对称
  或 symw + λ0.1 补跑，属拍板项）。

## Run 002 — 质量数值抽查 + viz（symw best ckpt）（2026-10-25）

### 目的与假设

- 与 w500 Run 003 / exit-gnn Run 002 同口径对照：sym-W + λ0.2 下 medoid/real std
  （全程 125 / 去末 25 双口径）与 coverage（结构性照登）。

### 配置

- ckpt：Run 001 `fold0_mild_flow_best_graph_mexhat.pt`（best 0.8054@E144；config 凭据
  `sym_w: true` / `flow_variant: graph-mexhat-exit-gnn-symw`）
- 命令：`quality_check_mild.py --fold 0 --ckpt <best> --window-len 500 --obs-len 375
  --forecast-len 125 --k 32 --seed 42 --subjects 1,7,21 --flow-exit-gnn --out-dir
  run_symw_fold0/quality`；`viz_mild.py` 同参（--flow-exit-gnn 透传）→ `viz/`

### 结果

| sid | 组 | 全程 mean (min/max) | 去末 25 mean (min/max) | exit-gnn 对照（全程/去末） | w500 对照（全程/去末） |
|---|---|---|---|---|---|
| 1 | MDD | 0.802 (0.70/0.89) | 0.771 (0.64/0.89) | 0.601 / 0.496 | 1.016 / 1.053 |
| 7 | MDD | 0.898 (0.59/1.13) | 0.891 (0.55/1.23) | 0.696 / 0.662 | 0.884 / 0.904 |
| 21 | HC | 0.875 (0.61/1.07) | 0.880 (0.63/1.12) | 0.637 / 0.584 | 0.973 / 1.032 |
| ALL | | **0.858 (0.59/1.13)** | **0.848 (0.55/1.23)** | **0.645 / 0.581** | **0.957 / 0.996** |

- coverage 双口径 0.0000（确定性采样结构性，照登，不携带质量信息）。
- viz 三 PNG + meta 落 `run_symw_fold0/viz/`（--flow-exit-gnn strict load 正常）。

### 结论

- **幅度校准大幅回升：medoid/real std 全程 0.858 / 去末 0.848**——较 exit-gnn
  （0.645/0.581）回升 +0.21/+0.27，收复其"出口再平均"回归的绝大部分；与 w500
  （0.957/0.996）仍差 ~0.10–0.15。方向与「对称 A_eff 抑制出口再平均的各向异性收缩」
  一致，但 λ 与 sym-W 两因素同变，机制归因不唯一（登记）。
- **θ 候选判断留用户**：val 最优 + 幅度回升至 0.85 档（仍未达 w500 的 ~1.0 档）+
  波形人工判读待拍板；对称约束 ⇒ EC 方向性按构造受限（prd 登记），若进入 EC 线
  须按无向口径重设预期。单次运行（fold0 单 seed，bf16 AMP）。

