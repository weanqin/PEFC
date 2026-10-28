# latent-flow sym-W 变体 fold3/fold4 双折训练 + 扰动归因 EC × GC 对拍

## Goal

symw 变体（exit-gnn + anti-uniform λ0.2 + SymWAdam 每步 W←(W+Wᵀ)/2 投影）在
fold3/fold4 补齐五折版图的最后两折：每折 stage1 VAE → stage2 symw（均逐字镜像
f12 任务口径）→ 扰动归因 EC（ec_mild_unbiased.py 现行版，v4 直出）→ 逐被试 +
组平均（含组均值矩阵）EC×GC(v1 单参考) 相关性 + EC 图，按 fold 组织产物并报告数值。

## Background（侦察已钉死）

- **先例链**：训练镜像 `10-25-latent-flow-symw-f12`（fold1/2，PRD-only 纯跑批）；
  EC 镜像 `10-25-symw-ec-f12`（fold1/2 EC + 跨折并置）。本任务 = 两者在 fold3/4
  的合并复刻 + 相关性/图件交付面（用户工作令 2026-10-28）。
- **脚本面（全部包内、零代码改动）**：
  - 训练 `train_mild_depress.py`（stage1/stage2、`--fold {0..4}`、`--sym-w`/
    `--flow-exit-gnn`/`--anti-uniform-lambda` 全在 CLI）
  - EC `ec_mild_unbiased.py`（`--mode main/summary/gccorr`、`--fold`、`--symw`；
    10-27 配置修复后 main 直出 v4 公式、gccorr = v1 单参考
    REFS_DIR=mild40 fixed8 f1-45 favg，产出 per_subject/group/group_mean 三块）
  - 图件 `viz_ec_pert.py`（EC_pert v4 三图，--main-dir/--out-dir 通用）；
    `viz_symw_ec_pairs.py`（逐被试 EC×GC 并排 + 组均值两版，--recompute-dir 指向
    ec_main 的 schema 兼容性需运行时验证，不兼容则降级只交付 viz_ec_pert）
  - 质检 `quality_check_mild.py --flow-exit-gnn`（symw ckpt 经 exit-gnn 架构装载：
    SymWAdam 仅 optimizer 投影不进 ckpt，架构零差异——symw fold0 README 在档结论）
- **折划分**（manifest `splits/mild_5fold_seed42.json`）：fold3 test=[9,13,18,20,
  29,33,38,40]（MDD 4/HC 4）、val=[16,19,36,39]；fold4 test=[4,15,16,19,24,35,
  36,39]、val=[1,7,21,27]；train 均 28。GC v1 refs 全 40 例在档（refs_s1..s40）。
- **预计成本**：每折 stage1 ~15 min + stage2 ~12 min（f12 实测 wall 706.5s）+
  EC main 8 例 ~52 min + 离线 summary/gccorr/viz 分钟级；两折串行 ≈ 3 h 单卡。

## Requirements

- **R1 训练 fold3/fold4**：每折 stage1（w500 fold0 stage1 实际口径镜像：window500/
  375/125、150ep/patience30/lr5e-4/batch64/seed42）→ stage2（symw Run 001 逐字同参：
  lr1e-3、unfreeze-decoder、anti-uniform 0.2、flow-exit-gnn、sym-w、vae-ckpt 指本折
  stage1 best）。训练日志（history json + run log）与 best ckpt 全保留。
- **R2 EC fold3/fold4**：每折 test 8 例全量 30 trial，同归因口径（q=0.5/σ_ref=0.3/
  log[0.01,0.48]×24/seed42/iid 噪声、确定性采样、global-τ 含对角），v4 公式直出。
- **R3 相关性**：每折 gccorr（v1 单参考）：逐被试 Spearman/Pearson × abs/raw ×
  aligned/flipped + 组统计（mean±std）+ 组均值矩阵一次相关；数值明确写入
  experiment.md 与 README（用户主读数：逐被试相关系数 + 组平均相关性）。
- **R4 组平均 EC 与图件**：每折 group_mean.npz（组平均扰动归因 EC）+ EC 图
  （viz_ec_pert 三图；viz_symw_ec_pairs 通过则附加）。
- **R5 组织与登记**：训练产物按 `run_symw_fold{3,4}`、EC 产物按 `fold{3,4}/`
  子目录组织；experiment.md 逐 Run 登记；README 汇总表（fold0-4 五折并置可读）；
  研究记忆回填（progress.md symw 条目补 fold3/4）。
- **R6 收尾**：trellis-check → spec 更新（若有沉淀点）→ commit。

## Acceptance Criteria

- [ ] **AC1**：fold3/fold4 stage1+stage2 完成；best val、早停形态、W 对称断言
  （best ckpt torch.equal 级，参照 f12 inline 只读断言）逐折登记。
- [ ] **AC2**：每折 EC main 8/8 例成功；P(resp>τ)=0.500 逐例；connectivity_formula
  = v4。
- [ ] **AC3**：每折 gccorr_summary.json 齐全，experiment.md/README 明确报告每折
  逐被试相关系数与组平均相关性数值（含组均值矩阵口径）。
- [ ] **AC4**：每折 group_mean.npz + EC 图 PNG/SVG 落档（viz_ec_pert 必交付；
  pairs 图视 schema 兼容性，不兼容时登记降级原因）。
- [ ] **AC5**：产物按 fold 组织、README + 研究记忆回填 + commit 完成。

## Boundaries

- 代码改动仅两处同缺陷类（Execute 阶段发现的 prd 缺陷，2026-10-28 修订）：
  `ec_mild_unbiased.py` ① symw ckpt 出处守卫扩展——另接受
  `10-28-latent-flow-symw-f34` run_dir + `fold{3,4}_` 前缀（f12 扩展同款形态，
  防折-权重错配防护保留）；② `EXPECTED_NORM_SHA_BY_FOLD` 注册 fold3/4 归一化
  SHA（训练侧 configs 两 stage 一致值登记，f12 同款方法）；除此零代码改动，
  W 对称断言等仍以 inline 只读 python 执行。
- 图件交付路径修订（2026-10-28 Execute 发现）：包内 `viz_ec_pert.py`/
  `viz_symw_ec_pairs.py` 深度绑定 fold0（组均值锚点断言 +0.1934、旧版 dual-summary
  结构、硬编码被试表），改造将扩大代码改动面 ⇒ 改为任务本地脚本
  `outputs/10-28-symw-ec-f34/make_ec_viz_f34.py`（同款三图契约：逐被试 2×4 热图/
  组均值 EC×GC 二联/MDD-HC 均值+差异；PNG300+PDF；只读本折 ec_main+gccorr+refs）。
- 路径隔离：只写 `outputs/10-28-latent-flow-symw-f34/`（训练）、
  `outputs/10-28-symw-ec-f34/`（EC）与本任务目录。
- GC 对照仅 v1 单参考（v2 已注销 10-27 拍板）；描述性并置不构成代理脑资格判定。
- 口径不调参：全部沿用先例（训练/EC/图件），fold3/4 不引入任何新配置。
- 单 seed 单次运行（bf16 AMP，先例确定性档）；GPU 串行，训练与 EC 不并行。

## 子研究路由（1.0b / 1.0c）

- **分类结果**：`study1`
- **依据**：latent-flow mild 线 symw 变体跨折（五折版图收尾）+ 同线 EC 归因对拍，
  属 Study I 候选族训练与评估档案。
- **新路径/方法检测（1.0c）**：`已有路径：latent-flow symw 变体（10-25 任务线），
  无新方法`。
- **research 产物**：`无`。
