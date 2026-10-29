# symw 变体 fold3/fold4 训练 + 扰动归因 EC × GC 对拍（任务 10-28-latent-flow-symw-f34，2026-10-28）

用户工作令：以「10-25-latent-flow-symw」系代理脑补齐 fold3/fold4（训练 + EC +
相关性 + 图件，按 fold 组织）。代理脑 = symw 变体（FlowVelocityNetExitGNN +
anti-uniform λ0.2 + SymWAdam 每步 W←(W+Wᵀ)/2 投影；**λ/sym-W 两因素同变未消融**，
θ 候选判断待拍板——**EC 结果 provisional，不构成代理脑资格判定**）。

## 目录

```
outputs/10-28-latent-flow-symw-f34/     # 训练侧
  run_symw_fold{3,4}/                    # configs/logs/checkpoints/quality
  run_symw_fold{3,4}_stage{1,2}.log      # 训练日志（含逐步 loss）
  quality_checks.log
outputs/10-28-symw-ec-f34/               # EC 侧（本 README）
  fold{3,4}/ec_main/                     # 8 例 subject_*.json（conn v4 直出）+ curves npz
  fold{3,4}/summary/                     # group_summary.json + group_mean.npz（组平均 EC）
  fold{3,4}/gccorr/                      # gccorr_summary.json（逐被试+组+组均值矩阵）
  fold{3,4}/viz_ec/                      # 三图 ×2 折（PNG300+PDF）
  fold{3,4}_ec_main.log                  # EC 运行日志
  make_ec_viz_f34.py                     # 任务本地出图脚本（prd 图件路径修订在档）
```

## 口径（与 fold0-2 先例逐字同参，唯一差异 = 折）

- 训练：stage1 = w500 fold0 口径镜像（window500/375/125、150ep/pat30/lr5e-4/64/
  seed42）；stage2 = symw Run 001 verbatim（lr1e-3/unfreeze-decoder/anti 0.2/
  flow-exit-gnn/sym-w）；CLI 对拍 fold1 逐键相等。
- EC：q=0.5 / σ_ref=0.3 / log[0.01,0.48]×24 / seed 42 / iid 噪声 / 确定性采样 /
  global-τ 含对角；conn = v4 = −(ln σ*+0.5·ln2πe)+5/7（10-27 线默认）。
- GC 对照：v1 单参考（mne 谱 GC 固定阶 8 / 1–45 Hz favg / trial 池化；
  v2 已注销）。offdiag 16256 对，aligned = 行(源) vs GC，flipped = vs GCᵀ。
- 代码改动两处守卫扩展（f12 同款形态，prd 在档）：symw ckpt 出处白名单 +
  EXPECTED_NORM_SHA_BY_FOLD 注册 fold3/4（训练侧两 stage 一致值，负例拦截实测）。

## 训练结果（Run 001）

| 项 | fold3 | fold4 | fold0/1/2 先例 |
|---|---|---|---|
| stage1 best val | 2.0326@E61（91ep 早停） | 2.0580@E62（92ep 早停） | 2.0992/2.0358/2.1210 |
| stage2 best val | 0.8358@E113（143ep 真早停） | 0.8769@E144（150ep 顶格） | 0.8054 顶格/0.8264/0.8433 |
| W 对称断言 | torch.equal 级，max\|W\| 1.239 | torch.equal 级，max\|W\| 1.539 | 三折 1.25–1.59 |
| 幅度抽查 std_full/trim | **1.038/0.968**（≈完美） | **0.680/0.679**（回落） | 0.858/0.848 · 1.011/1.043 · 0.886/0.896 |

五折 stage2 best val：0.8054/0.8264/0.8433/**0.8358**/**0.8769**（均值 0.8376）；
五折幅度均值 ≈0.895（fold4 单折回落拖低，描述性登记）。

## EC 运行（Run 003/004）

- fold3：8/8 例 51.3 min；P(resp>τ)=0.500 逐例精确；v4 公式串 8/8 PASS。
  interior 组均 72.0%、dyn_ratio_med 3.10×、跨被试稳定 ρ 0.451。
- fold4：8/8 例 51.2 min；P(resp>τ)=0.500 逐例精确；v4 公式串 8/8 PASS。
  interior 组均 58.4%、dyn_ratio_med 2.00×、跨被试稳定 ρ 0.591。

## 逐被试 EC×GC 相关性（v1 单参考，aligned；abs≡raw 全正）

**fold3（test = s9,13,18,20 MDD / s29,33,38,40 HC）——8/8 全正：**

| 被试 | 组 | Spearman | Pearson | Spearman(flipped) |
|---|---|---|---|---|
| s9 | MDD | +0.3180 | +0.2344 | +0.2157 |
| s13 | MDD | +0.3526 | +0.2741 | −0.0008 |
| s18 | MDD | +0.0923 | +0.2056 | +0.1426 |
| s20 | MDD | +0.1016 | +0.0783 | +0.1770 |
| s29 | HC | +0.1510 | +0.1287 | +0.0883 |
| s33 | HC | +0.2272 | +0.2338 | −0.1820 |
| s38 | HC | +0.1319 | +0.1448 | +0.1057 |
| s40 | HC | +0.3244 | +0.2050 | +0.0135 |

**组平均（fold3）**：跨被试 Spearman **+0.2124 ± 0.1073** / Pearson
**+0.1881 ± 0.0651**；**组均值矩阵一次相关 Spearman +0.3503 / Pearson +0.3418**；
flipped 组均 +0.0700（aligned−flipped 方向增量 ≈ +0.14，五折最大）。

**fold4（test = s4,15,16,19 MDD / s24,35,36,39 HC）——8/8 全正：**

| 被试 | 组 | Spearman | Pearson | Spearman(flipped) |
|---|---|---|---|---|
| s4 | MDD | +0.2071 | +0.2376 | −0.0573 |
| s15 | MDD | +0.2593 | +0.1587 | −0.0191 |
| s16 | MDD | +0.2019 | +0.1666 | +0.0485 |
| s19 | MDD | +0.2168 | +0.1147 | +0.0816 |
| s24 | HC | +0.1528 | +0.2411 | +0.1262 |
| s35 | HC | +0.0121 | −0.0095 | +0.0464 |
| s36 | HC | +0.3072 | +0.2174 | +0.0500 |
| s39 | HC | +0.1004 | +0.1674 | −0.0575 |

**组平均（fold4）**：跨被试 Spearman **+0.1822 ± 0.0928** / Pearson
**+0.1618 ± 0.0817**；**组均值矩阵一次相关 Spearman +0.2541 / Pearson +0.2134**；
flipped 组均 +0.0273。

## 五折并置（symw 全版图；fold0-2 引先例在档值，同口径同参考）

| fold | 跨被试 Spearman | 跨被试 Pearson | 组均值矩阵 ρ / r |
|---|---|---|---|
| 0（10-25-symw-ec-unbiased） | +0.1344±0.1587 | +0.1723±0.1787 | +0.1806 / +0.2338 |
| 1（10-25-symw-ec-f12） | +0.1590±0.1164 | +0.2102±0.1015 | +0.3805 / +0.3759 |
| 2（10-25-symw-ec-f12） | +0.1757±0.1269 | +0.1854±0.1148 | +0.2675 / +0.2494 |
| **3（本任务）** | **+0.2124±0.1073** | +0.1881±0.0651 | **+0.3503 / +0.3418** |
| **4（本任务）** | +0.1822±0.0928 | +0.1618±0.0817 | +0.2541 / +0.2134 |
| 五折均值 | +0.1727 | +0.1836 | +0.2866（ρ 均值） |

判读（描述性，provisional 随 θ 拍板⑩联动）：

- 五折跨被试 Spearman 全部落在 +0.13–0.21 微弱-中等正相关带，**单调抬升无回撤**
  （fold0→4：0.134/0.159/0.176/0.212/0.182），fold3/4 首次出现 8/8 全正被试集
  （fold0-2 均有负例）——「GC 相关形态由 GNN 架构面承载」的跨折稳定结论在
  全五折成立且后两折更强。
- 组均值矩阵相关（+0.18–0.38）系统性高于跨被试均值——主要为对称强度图案贡献
  （flipped 亦高），方向特异增量以逐被试 aligned−flipped 为准（fold3 ≈ +0.14、
  fold4 ≈ +0.15，高于 fold0 的 ≈+0.03）。
- fold4 幅度回落（0.68）与 EC 相关强度无可见联动（组 ρ 五折第二高）——幅度
  校准与 GC 对齐是两个独立面，描述性并置。

## 图件（fold{3,4}/viz_ec/，PNG300+PDF）

每折三图：`f{k}_ec_subjects_v4`（逐被试 2×4 热图，题注逐被试 ρ）、
`f{k}_ec_group_vs_gc`（组均值 EC×GC 二联，题注 ρ/r）、`f{k}_ec_mdd_hc`
（MDD/HC 均值 + 差异）。图面 ρ 与 gccorr json 漂移对拍断言 1e-9 全过。

## 复现命令

```bash
# 训练（每折两条，k∈{3,4}）
OMP_NUM_THREADS=8 /root/miniconda3/bin/python models/latent-flow-v1/train_mild_depress.py \
  --fold k --stage 1 --window-len 500 --obs-len 375 --forecast-len 125 \
  --epochs 150 --patience 30 --batch-size 64 --lr 5e-4 --seed 42 \
  --run-dir outputs/10-28-latent-flow-symw-f34/run_symw_foldk
OMP_NUM_THREADS=8 /root/miniconda3/bin/python models/latent-flow-v1/train_mild_depress.py \
  --fold k --stage 2 --window-len 500 --obs-len 375 --forecast-len 125 \
  --batch-size 64 --lr 1e-3 --epochs 150 --patience 30 --seed 42 \
  --unfreeze-decoder --anti-uniform-lambda 0.2 --flow-exit-gnn --sym-w \
  --vae-ckpt outputs/10-28-latent-flow-symw-f34/run_symw_foldk/checkpoints/foldk_mild_vae_best.pt \
  --run-dir outputs/10-28-latent-flow-symw-f34/run_symw_foldk
# EC（main/summary/gccorr）+ 图件
OMP_NUM_THREADS=8 /root/miniconda3/bin/python models/latent-flow-v1/ec_mild_unbiased.py \
  --mode main --fold k --symw --task-label 10-28-symw-ec-f34-foldk \
  --ckpt outputs/10-28-latent-flow-symw-f34/run_symw_foldk/checkpoints/foldk_mild_flow_best_graph_mexhat.pt \
  --out-dir outputs/10-28-symw-ec-f34/foldk/ec_main
#   summary/gccorr 同款 --mode 换 --main-dir/--out-dir；图件：
OMP_NUM_THREADS=4 /root/miniconda3/bin/python outputs/10-28-symw-ec-f34/make_ec_viz_f34.py \
  --fold k --ec-root /root/npi/outputs/10-28-symw-ec-f34
```

## 已知边界

- 单 seed 单次运行（bf16 AMP 确定性档）；描述性并置登记，不构成代理脑资格判定；
  λ/sym-W 两因素同变 caveat 继承；五折 40 被试全域覆盖（test 折互斥并集 = 40）。
- `fold3/viz_ec/` 内另有一组早期弃用产物（`ec_subjects_v4.png/pdf` +
  alignment 附件，包内 viz_symw_ec_pairs 首次尝试因 fold0 被试表硬编码失败后
  残留的部分输出）与空目录 `fold3/viz_pairs/`——弃用不覆盖销毁，正式交付以
  `f3_*`/`f4_*` 契约名三图为准。
- ec_main payload 顶层 `surrogate.task` 标签沿用脚本内 fold0 世代常量串（纯元数据
  外观），真实出处以 `provenance.ckpt_path/checkpoint` 为准（本任务 fold3/4 ckpt
  路径逐例正确；修正需第三处代码改动，越界仅登记）。
