# symw f12 双折无偏扰动归因 EC（fold1/fold2，同 fold0 口径逐字复用）+ 跨折汇总对比

## 子研究路由（1.0b / 1.0c）

<!-- trellis-research-routing -->

- **分类结果**：`study2`（EC 推断算法，框架 Study II——固定 θ 候选代理脑做虚拟扰动推断连接）
- **依据**：任务主体 = fold1/fold2 扰动归因 EC 计算与跨折汇总（推断算法侧）；代理脑（f12 双折
  symw，Study I 产物）仅作固定输入，不训练、不做资格判定。
- **新路径/方法检测（1.0c）**：`已有路径：global-tau 无偏 EC 口径（w500/exit-gnn/symw-fold0
  三先例已归档；本任务 = 同口径换折迁移 + 跨折汇总）`——无新方法，research adaptation 不触发。
- **research 产物**：`无`
- **路由备注**：EC 结论继承 symw 代理脑 provisional 状态（θ 候选判断待用户）；跨折并置为
  描述性口径（各折 θ 不同代理脑实例，不作复现性判读）。

## Goal

以 `10-25-latent-flow-symw-f12` 训练的 **fold1 / fold2 stage2 best ckpt** 为代理脑，用与
fold0（任务 10-25-symw-ec-unbiased）**逐字相同**的无偏扰动归因口径（q=0.5、σ_ref=0.3、
log[0.01,0.48]×24、无偏 μ=0、seed 42、bf16、各折 test 8 例 × 30 trial 全量、τ 分位含对角）
计算 EC_pert；v4 公式交付；各折组级 summary + 双 GC 逐被试相关；**跨折（fold0/1/2）汇总
对比表**。配置与模型版本随档注明（用户要求③：可与 fold0 直接汇总）。

## 背景与用户指令（2026-10-25，三项）

1. 扰动策略/步长/指标/流程与 fold0 完全一致，结果可直接对比；
2. 加载 fold1/fold2 训练好的模型权重计算；
3. 输出各 fold 归因结果（分数/曲线/表格）+ 配置注明，便于与 fold0 汇总比较。

## 关键事实（侦察在档）

- **ckpt**：fold1 = `outputs/10-25-latent-flow-symw-f12/run_symw_fold1/checkpoints/
  fold1_mild_flow_best_graph_mexhat.pt`（best val **0.8264@E116**，真早停 E146）；fold2 =
  `.../run_symw_fold2/checkpoints/fold2_mild_flow_best_graph_mexhat.pt`（**0.8433@E113**，
  真早停 E143）。训练口径 = symw Run 001 逐字同参（f12 任务登记），模型类同构
  FlowVelocityNetExitGNN（`--symw` 架构路径直接可用）。
- **test 被试**：fold1 = s{2,12,14,17,22,32,34,37}；fold2 = s{3,6,8,11,23,26,28,31}
  （manifest `splits/mild_5fold_seed42.json`，单一文件全折共用 ⇒ manifest SHA 断言不变）。
- **norm SHA（训练侧 configs 登记，fold1/2 stage1/2 一致）**：fold1 =
  `0f10b15350f69757d883beca19500342fc8279a71dc327252e495be08e1d3383`；fold2 =
  `a2c27a2d2577210c2696e0310b0bd1a792d8b2d530107b9bdf8d100446c3d102`；fold0 参照 =
  `d184e647…`（脚本现行钉死值）。
- **脚本适配面（算法零改动）**：①`EXPECTED_NORM_SHA` 常量 → fold 键控映射 {0,1,2}
  （值 = 训练侧登记）；②`--symw` ckpt 守卫扩展接受 f12 run_dir（`10-25-latent-flow-symw-f12`）
  并断言 ckpt 文件名 fold 前缀与 `--fold` 一致；③payload surrogate 元数据走 f12 分支
  （task/best-val/早停注记/caveat）。
- **GC refs**：v1+v2 双参考覆盖全部 40 被试 ⇒ fold1/2 的 16 个新被试可直接对照。
- GPU 空闲（f12 训练已归档结束）。

## Requirements

- **R1（脚本适配，算法零改动）**：`ec_mild_unbiased.py` 三处元数据级适配（上述①②③）；
  σ*/τ/conn/probe/bisection 计算路径逐行零改动（diff 审确认）；`--fold 1/2` + `--symw` +
  f12 ckpt 为新调用面，旧调用面（fold0）行为逐位不变。
- **R2（等价性证据）**：适配后旗标关 fold0 mini 跑（1 例 × 8 源 × 4 网格）与在档
  `outputs/10-24-mild-w500-ec-unbiased/dryrun_mini/subject_0.json` 逐位对拍 max|Δ|=0
  （守卫/SHA 改动不触计算路径的运行时证据）；`--fold 1 --symw` 干跑：strict load + fold1
  norm SHA 断言通过 + payload 元数据核对 + 守卫负例（fold0 ckpt 配 --fold 1 应被 fold
  前缀断言拦）。
- **R3（主跑 ×2，用户口径钉死）**：fold1、fold2 各 8 test 例 × 30 trial，全默认参数 +
  seed 42 + bf16 + 断点续跑；输出 `outputs/10-25-symw-ec-f12/fold{1,2}/ec_main/`
  （逐例 json+npz 原子写，凭据链完整：ckpt 路径/fold/symw:true/git head/各折 norm+manifest
  SHA 断言注记）。预算 ≈52 min/折。
- **R4（v4 + 双 GC，每折）**：`recompute_conn_v4.py --main-dir fold{k}/ec_main --out-dir
  fold{k}/recompute_v4 --task-label 10-25-symw-ec-f12-fold{k} --surrogate-name
  "sym-w f12 fold{k} surrogate (best 0.8xxx@Exxx)"`；三断言 8/8；v1/v2 双块 per-subject +
  group + group_mean（refs 全 40 被试覆盖）。
- **R5（组级 summary，每折）**：`--mode summary --fold {k} --symw --main-dir … --task-label …`；
  interior%/dyn_ratio/方差分解/跨被试稳定性。
- **R6（跨折汇总对比表 = 用户核心交付③）**：离线汇总 json + 表：fold0（在档
  10-25-symw-ec-unbiased）vs fold1 vs fold2——各折 best val/interior/dyn_ratio/τ 中位/
  conn 分布/v1·v2 GC ρ（group mean±std + group_mean）/逐被试 ρ 明细；**跨代理脑实例描述性
  并置注记**（各折 θ 不同，不作复现性判读）；落 `outputs/10-25-symw-ec-f12/
  crossfold/crossfold_summary.json` + README 表。
- **R7（caveat 与 provisional 标注）**：f12 代理脑 θ 候选判断待用户；跨折一致性为描述性；
  逐例 payload caveat + README + experiment.md 三处登记。
- **R8（路径隔离）**：改动仅 `models/latent-flow-v1/ec_mild_unbiased.py`（元数据级）+
  新 `outputs/10-25-symw-ec-f12/` 树 + 任务目录；latent_flow 包体零改动；旧任务输出树/
  refs 只读；预存在脏路径（ec-globaltau-v1 5 脚本、两个 10-24 任务目录）不擅动。

## Acceptance Criteria

- [ ] R1 diff 审：计算路径零改动；R2 两项证据在档（逐位对拍 max|Δ|=0；f12 干跑全绿）。
- [ ] R3 两折主跑各 8/8 成功；P(resp>τ)=0.500 逐例精确；凭据链完整（fold 正确、SHA 各折
      断言通过）。
- [ ] R4 每折三断言 8/8 + 双 GC per-subject/group/group_mean 落档。
- [ ] R5 每折 summary 落档。
- [ ] R6 跨折对比表落档（fold0/1/2 三列齐 + 并置注记）。
- [ ] experiment.md 各 Run 块齐 + outputs README（配置/模型版本/复现命令）+ 研究记忆回写。
- [ ] 收尾三件套（2.2 check / 3.3 spec-update 判定 / 3.4 commit 仅本任务路径）。

## 约束

- **轻量任务声明**：PRD + implement 执行序（镜像 10-25-symw-ec-unbiased 结构：R1 适配 →
  R2 干跑 → R3 主跑×2 → R4/R5 离线 → R6 汇总 → README/收尾）；design 不适用。
- 环境：npi 本机（RTX 4080 SUPER 32 GB，已确认空闲）；GPU 预算 ≈110 min（两折串行）；
  BLAS 守卫内置；跑前登记 experiment.md。
- 数据：Mild Depression（契约在档）；500/375/125 硬三元组断言照旧；manifest 单文件全折
  共用（SHA 不变）；norm SHA 按 fold 断言（训练侧登记值）。
