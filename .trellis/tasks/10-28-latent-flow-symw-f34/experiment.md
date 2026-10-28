# 实验日志 — 10-28-latent-flow-symw-f34

> 每次「跑批/训练/评估/统计」追加一个 Run 块：跑前先写「目的+配置」，跑后同会话补「结果+结论」，不留空块。
> 格式契约见 `.trellis/spec/research/experiment-log.md`；产物只落本任务声明目录（path-isolation）。

## Run 001 — fold3/fold4 双折训练（stage1 + stage2 symw，2026-10-28）

### 目的与假设

- symw 变体五折版图收尾（fold0-2 先例在档）：假设 fold3/4 复现 val 0.81–0.84 档、
  真早停/顶格形态、W torch.equal 级对称、幅度校准 0.85–1.04 档。

### 配置

- 代码：训练零改动（f12 交付版）；命令逐字镜像 f12 Run 001/002，仅 --fold/run_dir/
  vae_ckpt 换折（CLI 对拍见结果）。
- stage1（w500 fold0 口径镜像）：`--fold k --stage 1 --window-len 500 --obs-len 375
  --forecast-len 125 --epochs 150 --patience 30 --batch-size 64 --lr 5e-4 --seed 42`。
- stage2（symw Run 001 verbatim）：`--fold k --stage 2 --window-len 500 --obs-len 375
  --forecast-len 125 --batch-size 64 --lr 1e-3 --epochs 150 --patience 30 --seed 42
  --unfreeze-decoder --anti-uniform-lambda 0.2 --flow-exit-gnn --sym-w --vae-ckpt
  <本折 stage1 best>`。
- 环境：conda npi @ 本机 RTX 4080S（GPU 空闲预检 1 MiB/32 GB）；数据 manifest
  `splits/mild_5fold_seed42.json`（fold3 test [9,13,18,20,29,33,38,40] / fold4 test
  [4,15,16,19,24,35,36,39]，v1 GC refs 逐例存在性预检通过）。
- 日志：`run_symw_fold{3,4}_stage{1,2}.log` + `logs/*_history.json` 全保留。

### 结果

| 项 | fold3 | fold4 | 先例 fold0/1/2 |
|---|---|---|---|
| stage1 best val | 2.0326@E61（91ep 早停） | 2.0580@E62（92ep 早停） | 2.0992@E57 / 2.0358@E57 / 2.1210@E49 |
| stage1 wall | 435.9 s | 440.9 s | — |
| stage2 best val | **0.8358@E113**（143ep 真早停） | **0.8769@E144**（150ep 顶格） | 0.8054@E144 顶格 / 0.8264@E116 / 0.8433@E113 |
| stage2 wall | 683.3 s | 716.6 s | 706.5 / 692.0 s（f12） |
| smoke MAE/std_ratio | 1.0280 / 0.9032 | 0.7925 / 0.6471 | — |
| best 分量 recon2/corr2/cons2/uni_reg | 0.413/0.224/0.196/0.015 | 0.435/0.230/0.209/0.015 | fold1 0.418/0.202/0.204/0.0149 |

- **CLI 对拍**：fold3/fold4 stage2 cli 块剔除 fold/run_dir/vae_ckpt 后与 f12 fold1
  **逐键相等**（Python 字典相等断言）。
- **W 对称断言**：双折 best ckpt `flow.graph_mixer.W` (128,128) `torch.equal(W,Wᵀ)`
  = True；max|W| fold3 1.2390 / fold4 1.5385（先例三折 1.25–1.59 同域）。
- **质量抽查**（k=32/seed42/双口径，coverage 结构性 0.0000 照登）：
  - fold3（sid 9,13,29 = MDD,MDD,HC）：ALL std_full **1.038 (0.81/1.46)** /
    std_trim 0.968 (0.72/1.35)——≈完美校准（fold1 1.011/1.043 档）。
  - fold4（sid 4,15,24）：ALL std_full **0.680 (0.50/0.90)** / std_trim 0.679
    (0.50/1.02)——幅度回落（低于 fold2 0.886，接近 exit-gnn 0.645 档）。
- 产物：`outputs/10-28-latent-flow-symw-f34/run_symw_fold{3,4}/`（configs/logs/
  checkpoints/quality + 四条 stage 日志）。

### 结论

- 五折版图补齐：stage2 best val 五折 = 0.8054/0.8264/0.8433/0.8358/0.8769（均值
  0.8376）；fold4 为五折最高但顶格未早停 + 幅度回落 0.68——描述性登记，跨折幅度
  均值五折 (0.858+1.011+0.886+1.038+0.680)/5 ≈ 0.895，fold4 单折拖低；θ 判断仍
  留用户（先例拍板项⑩）。

## Run 002 — symw ckpt 出处守卫扩展（Execute 发现的 prd 缺陷修订，2026-10-28）

### 目的与假设

- fold3 EC main 首启被 `_caliber_asserts` symw 出处守卫拦截（只认 fold0 原跑与
  f12 目录）；假设按 f12 扩展同款形态增补本任务 run_dir 白名单即可，且原守卫
  行为零回归。

### 配置

- 改动：`ec_mild_unbiased.py` symw 分支新增 `elif "10-28-latent-flow-symw-f34"
  in a.ckpt`（断言 fold ∈ {3,4} + `fold{k}_` 前缀）；prd/implement 同步修订
  （Boundaries 由「零代码改动」改为「仅此一处声明性扩展」）。

### 结果

- 语法过；负例双拦截实测：fold1 × f34 ckpt 拦（前缀错配提示正确）、fold3 × f12
  ckpt 拦（f12 分支行为不变）；正例 fold3 × f34 ckpt 过闸进入主跑。

### 结论

- 守卫扩展符合 f12 先例形态，防折-权重错配防护保留；prd 缺陷闭环。

## Run 003 — fold3 EC main + summary + gccorr（2026-10-28）

### 目的与假设

- fold3 test 8 例（s9,13,18,20,29,33,38,40）× 30 trial 全量扰动归因 EC；假设
  P(resp>τ)=0.500 逐例构造性自洽、interior 与 fold0-2 同域、GC 相关形态微弱正。

### 配置

- `OMP_NUM_THREADS=8 nohup .../ec_mild_unbiased.py --mode main --fold 3 --symw
  --task-label 10-28-symw-ec-f34-fold3 --ckpt .../run_symw_fold3/checkpoints/
  fold3_mild_flow_best_graph_mexhat.pt --out-dir outputs/10-28-symw-ec-f34/fold3/
  ec_main`；summary/gccorr 同款 --main-dir 离线（gccorr = v1 单参考现行版）。

### 结果

- **8/8 例成功 51.3 min**（单例 6.40–6.51 min）；P(resp>τ)=0.500 逐例精确；
  connectivity_formula = v4 串 8/8 核对 PASS（G2 门过）。
- summary：interior 组均 **72.0%**（fold0 64.2 / f12 74.1/71.8 同域偏上）、
  dyn_ratio_med **3.10×**、row_share 0.081、跨被试稳定 ρ 0.451。
- **gccorr（v1 单参考，offdiag 16256，raw≡abs 全正）**——五折最强：
  - 逐被试 Spearman aligned：s9 **+0.318** / s13 **+0.353** / s18 +0.092 /
    s20 +0.102 / s29 +0.151 / s33 +0.227 / s38 +0.132 / s40 **+0.324**
    （**8/8 全正**，先例四折均有负例）
  - 逐被试 Pearson aligned：+0.234 / +0.274 / +0.206 / +0.078 / +0.129 /
    +0.234 / +0.145 / +0.205
  - flipped Spearman：+0.216 / −0.001 / +0.143 / +0.177 / +0.088 / −0.182 /
    +0.106 / +0.014（flipped 组均 +0.0700 ⇒ aligned−flipped 均值 +0.1424，
    方向增量五折最大）
  - **组：Spearman +0.2124 ± 0.1073 / Pearson +0.1881 ± 0.0651**；
    **组均值矩阵 Spearman +0.3503**（图面重算一致断言过）
- 产物：`outputs/10-28-symw-ec-f34/fold3/{ec_main,summary,gccorr}/`。

### 结论

- 假设成立且强于预期：fold3 是五折 GC 相关最强折（组 ρ +0.212 vs fold0-2
  +0.134/+0.159/+0.176），且首次 8/8 全正 + 组均值矩阵 ρ +0.35；「GC 相关形态
  由 GNN 架构面承载」的跨折描述性稳定结论在 fold3 延续为更强形态（仍属微弱-中
  等正相关档，provisional 随 θ 拍板联动）。

## Run 004 — fold3 图件 + fold4 EC main（2026-10-28）

### 目的与假设

- 用户交付项「扰动归因EC图」：包内 viz 脚本 fold0 锚定不可复用（见 prd 修订），
  任务本地脚本同款三图契约出图；fold4 EC main 同口径推进。

### 配置

- 本地脚本 `outputs/10-28-symw-ec-f34/make_ec_viz_f34.py`（只读 ec_main + v1 refs
  + gccorr json；漂移对拍断言 ρ 1e-9）；fold4 main 命令同 Run 003 换折。

### 结果

- fold3 三图落 `fold3/viz_ec/`（PNG300+PDF）：逐被试 2×4 / 组均值 EC×GC 二联
  （题注 ρ=+0.3503）/ MDD-HC 均值+差异；漂移断言过。
- fold4 main：**8/8 例成功 51.2 min**；P(resp>τ)=0.500 逐例精确；v4 公式串 8/8
  PASS（G2 门过）。summary：interior 58.4%、dyn_ratio_med 2.00×、stab ρ 0.591。
- fold4 gccorr：逐被试 Spearman s4 +0.207 / s15 +0.259 / s16 +0.202 / s19 +0.217 /
  s24 +0.153 / s35 +0.012 / s36 +0.307 / s39 +0.100——**8/8 全正**（连续两折）；
  组 Spearman **+0.1822±0.0928** / Pearson +0.1618±0.0817；组均值矩阵
  **+0.2541/+0.2134**；flipped 组均 +0.0273（方向增量 ≈ +0.15）。
- fold4 三图落 `fold4/viz_ec/`（漂移断言过）。

### 结论

- fold4 假设成立：EC 相关强度五折第二高（组 ρ +0.182），幅度回落（质检 0.68）
  与 GC 对齐强度无可见联动（两独立面，描述性）。

## Run 005 — 五折全版图汇总（AC3/AC5，2026-10-28）

### 目的与假设

- symw 五折版图收尾判读：fold0-4 跨被试/组均值矩阵两口径并置（fold0-2 引先例
  在档值，同口径同参考）。

### 配置

- 纯离线汇总（README + 本表）；产物 `outputs/10-28-symw-ec-f34/README.md`。

### 结果

| fold | 跨被试 Spearman | 跨被试 Pearson | 组均值矩阵 ρ / r |
|---|---|---|---|
| 0 | +0.1344±0.1587 | +0.1723±0.1787 | +0.1806 / +0.2338 |
| 1 | +0.1590±0.1164 | +0.2102±0.1015 | +0.3805 / +0.3759 |
| 2 | +0.1757±0.1269 | +0.1854±0.1148 | +0.2675 / +0.2494 |
| 3（本任务） | **+0.2124±0.1073** | +0.1881±0.0651 | **+0.3503 / +0.3418** |
| 4（本任务） | +0.1822±0.0928 | +0.1618±0.0817 | +0.2541 / +0.2134 |

- 五折跨被试 Spearman 全部 +0.13–0.21、无回撤抬升；fold3/4 首次连续 8/8 全正
  被试集；五折 test 互斥并集 = 40 全被试覆盖。

### 结论

- 「GC 相关形态由 GNN 架构面承载、对 W 对称化不敏感」在全五折成立且后两折
  更强；五折均值组 ρ +0.173（Spearman）。全量描述性并置，provisional 随
  θ 拍板⑩联动；不作代理脑资格判定。


