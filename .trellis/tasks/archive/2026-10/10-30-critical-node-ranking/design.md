# Design — 关键节点排序实验（critical node ranking）

> 依据 prd.md R1–R5。技术基座 = 双线 EC_pert 主管线的零拷贝复用（`models/latent-flow-v1/`），
> 新增单文件脚本 + 只读消费 `result/checkpoints/` 注册件；`latent_flow` 包体零改动、既有脚本零改动。

## 1. 边界与产物路径

- **新增**：`models/latent-flow-v1/critical_node_ranking.py`（唯一代码改动点）。
- **只读复用**：
  - `latent_flow/attribution/bisect_sigma.py::SignalProbeSite`（μ=0 iid 噪声 / 确定性采样 /
    CRN：`set_source(src)` 生成 ξ 后跨 σ 复用；`perturb_and_measure(σ, src)` → (mse_vec(C,),
    mean_diff_vec(C,))，MSE = 扰动生成 vs 站内缓存干净基线，跨 trial×fc 时间点平均）；
  - MODMA 装载链：`_connectivity_driver.{load_model, get_fold_test_subjects, load_subject_trials,
    subsample_trials, sanitize_for_json, json_default}`；
  - mild 装载链：`train_mild_depress.{MANIFEST_DEFAULT, TRIALS, build_model_mild,
    compute_mild_train_norm, load_manifest, MildTrialDataset}` + `viz_mild.{_model_args,
    load_flow_ckpt}` + `ec_mild_unbiased.{EXPECTED_NORM_SHA_BY_FOLD, EXPECTED_MANIFEST_SHA,
    W500_WINDOW/W500_OBS/W500_FC}`（module import 安全，有 main 守卫）；
  - `latent_flow/training/seeding.set_seed`。
- **产物**（全部 `outputs/10-30-critical-node-ranking/`，path-isolation：不写任何其他树）：
  - `modma/fold{k}/subject_{i}.json`、`mild/fold{k}/subject_{i}.json`（main 模式，原子写 +
    已存在即跳过 = 断点续跑）；
  - `robustness/{line}_robustness.json` + `robustness/{line}_top_nodes.md`（summary 模式）；
  - `figs/{line}_*.png`（figs 模式）；
  - `REPORT.md`（study 2 实验文档，手工撰写 + 数字从 json 抄录）。

## 2. 方法契约（双线统一，差异点列表后置）

### 2.1 影响度量与排序定义

对被试 s、噪声档 σ、源通道 c：

```
mse_vec = SignalProbeSite.perturb_and_measure(σ, c)        # (128,) 逐通道未来段 MSE
I_c(σ) = Σ_{b≠c} mse_vec[b]                                # 排除源自响应（127 项和）
ranking(σ) = argsort(I(σ), descending, stable-by-index)     # rank 1 = 影响最大
```

- 并列（浮点下概率可忽略）：按通道序号升序稳定排序，规则随档登记。
- 单调性预期：MSE_b(σ) 对 σ 单调非降（bisect_sigma 模块 docstring 论证）⇒ I_c(σ) 应单调非降
  （I(0.7)≥I(0.5)≥I(0.3)）；作诊断断言（bf16 容差 `rel_tol=1e-4`），违例如实计数登记不中断。

### 2.2 剂量与执行序（CRN 配对）

- σ 三档 `(0.3, 0.5, 0.7)` 钉死为默认；逐源执行序 = `set_source(c)` 一次 → 循环三档
  `perturb_and_measure(σ, c)`（同 ξ 跨 σ 缩放 ⇒ 三档配对，剂量-响应无蒙特卡洛抖动）。
- **autocast 契约（评审 B-1，σ=0 断言成立前提）**：每次 `perturb_and_measure` 调用（含 σ=0）
  必须包在 `torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=enabled)` 域内——站点基线
  在构造器内经 bf16 autocast 计算，而 `perturb_and_measure` 本体不管理 dtype（ProbeSite 协议）；
  裸调则扰动前向走 fp32 与 bf16 基线相减，σ=0 恒等断言必失败。镜像双线 EC 主跑循环
  （ec_mild_unbiased.py L424-439 / ec_modma_globaltau_unbiased.py L210-235 先例）。
- σ=0 结构性恒等断言：每被试首个源调用 `perturb_and_measure(0.0, c)`，断言 mse_vec 全 0
  （确定性采样 ⇒ 扰动=0 时逐位回基线；CRN 契约的免费检查点）。
- 固定超参：seed 42 / n_steps 10 / batch 32 / bf16 autocast（GPU）/ cudnn deterministic 双开关；
  SignalProbeSite 构造显式传 `sigma_min=0.01, sigma_max=0.7`（元数据可读性，engine 网格不走）。
- **剂量外推 caveat 按线分写**（评审 A-6/B-4）：modma 线 EC 口径网格 = log[0.01,0.3]×16 ⇒
  三档全部 ≥ 网格顶 0.3；mild 线网格 = log[0.01,0.48]×24 ⇒ 0.5/0.7 超上限。caveat 写入 json
  元数据与 REPORT（不预设结论）。

### 2.3 双线差异（各镜像其 EC 主跑口径）

| 项 | MODMA·0705 | mild·symw |
|---|---|---|
| ckpt（只读） | `result/checkpoints/modma/latent_flow_0705/fold{k}/stage2/fold{k}_modma_0705_graph_mexhat.pt` | `result/checkpoints/mild/latent_flow_symw/fold{k}/fold{k}_mild_flow_best_graph_mexhat.pt` |
| 模型构建 | `load_model(ckpt, device)`（_connectivity_driver） | `build_model_mild(device, _model_args(True), stage=2, obs_len=375, forecast_len=125)` + `load_flow_ckpt(model, ckpt)`（symw 架构侧 = exit_gnn，先例） |
| 被试集 | `get_fold_test_subjects(fold, 42, dataset="modma")` → 10/10/11/11/11 | `load_manifest(MANIFEST_DEFAULT, fold)` fold_dict["test"] → 8/折 |
| 数据 | `load_subject_trials(..., normalize=True)` + `subsample_trials(x, 32, 42, subj_idx)` | `MildTrialDataset([file], (mean,std), 375/125/500)` 30 trial 全量 |
| 归一化 | 逐被试逐通道 z-score（dataset 内） | `compute_mild_train_norm(train_f, window_len=500)` 训练集统计 |
| 运行 env | **必须 `MODMA_DATA_ROOT=/root/autodl-tmp/modma`**（latent_flow/config.py 缺省解析到包内不存在路径，environment.md 登记） | 无特殊 env |

### 2.4 守卫（跑前对拍，任一不符即拒跑）

- **MODMA**：ckpt 路径 = 注册件路径；SHA256 对拍脚本内钉死常量表（源自
  `result/checkpoints/modma/latent_flow_0705/sha256.json`，2026-10-30 抄录）：
  fold0 `5c6fbf31…` / fold1 `deb199e0…` / fold2 `fa4a0ba5…` / fold3 `6bb74a19…` /
  fold4 `6c51a4c2…`；文件名 `fold{k}_` 前缀与 `--fold` 锁定。
- **mild**：ckpt SHA256 对拍钉死常量表（源自 `result/checkpoints/MANIFEST.md` 表，stage2 五件）：
  fold0 `43564dd3…` / fold1 `06eb59fe…` / fold2 `4aaad72e…` / fold3 `421edf87…` /
  fold4 `3d8a4e97…`；norm SHA 按 fold 对拍 `EXPECTED_NORM_SHA_BY_FOLD`；manifest SHA 对拍
  `EXPECTED_MANIFEST_SHA`；窗三元组断言 500/375/125。
- 负例实测：冒烟阶段以 fold1 ckpt 冒充 fold0 跑一次（CLI 暴露 `--ckpt` 覆盖旗标，先例
  ec_modma_globaltau_unbiased.py L106），断言被守卫拦截（登记命令与报错）。
- mild 侧只 import `ec_mild_unbiased` 的常量（EXPECTED_*_SHA / W500_*），**不复用其
  `_caliber_asserts`**（run_dir 白名单会拒 result/ 路径，评审 B-6）；AST 核过模块级无副作用。

## 3. 稳健性分析（summary 模式，双线各自成档）

两级粒度 × 三对 σ（0.3↔0.5、0.3↔0.7、0.5↔0.7）：

1. **排序相关性**（128 维影响向量）：Spearman ρ + Kendall τ-b；逐被试级（每折 mean±sd + 全线
   合并分布）与组平均级（折内被试均值向量、全线池化均值向量）。
2. **Top-K 一致性**：K∈{10,20}；三对 σ 的 top-K 通道集合交叠率 |A∩B|/K；逐被试级 + 组平均级；
   读数附机会水平参照（随机 K/128 ≈ 7.8%/15.6%，图注注明）。
3. **Top-20 名单并置**：组平均三 σ 各自 top-20 通道号表（`*_top_nodes.md`），通道身份重合数直读。
4. **比例性诊断（评审 A-1，区分平凡稳定与非线性重排）**：低剂量线性近似下 I_c(σ) ≈ σ²·G_c(ξ)，
   σ² 因子在排序中约去 ⇒ 跨 σ 高一致部分是构造性必然；为此计算逐被试
   ratio_c = I_c(0.7)/I_c(0.3) 的分布（对照 (0.7/0.3)²≈5.44，偏离即非线性证据）与组平均
   log I(σ_hi) vs log I(σ_lo) 的通道级斜率分布（线性响应 ⇒ 斜率 ≈2）——把「σ² 缩放稳定」
   与「有意义的非线性重排」分开陈述。
5. **诊断**：σ=0 恒等断言结果、单调性违例计数、影响值分布摘要（min/median/max）、（描述性）
   同 σ 跨折组平均排序 Spearman——跨折稳定性作背景信息不参与稳健性判读主口径。

**判读纪律（评审 A-1/A-2 caveat，REPORT 必写）**：
- 跨 σ 一致性结论须区分三层——①σ² 缩放下的构造性稳定（比例性诊断≈2 时）；②偏离比例性的
  非线性重排（一致但非平凡）；③排序实质变化（低 ρ/低交叠）。以 ρ 高低单独陈述稳健性属过度主张。
- CRN 设计下检验的是**同一噪声实现下的剂量稳定性**，不是实现无关稳定性；若需后者须独立 ξ
  重复（σ=0.5 × 第二 seed × 被试子集，是否加 = 用户拍板项）。

**组级聚合口径（评审 A-3，跑前钉死）**：主口径 = 原始影响向量均值（默认）；副口径 = 逐被试
秩均值（aggregation-robustness 并列交付）；两口径同出 summary，REPORT 主文用主口径、副口径
作稳健性附注。

## 4. 数据流与 json 契约

```
per (line, fold, subject):
  load ckpt(守卫过) → load trials → SignalProbeSite(基线缓存)
  → σ=0 恒等断言 → for c in 0..127: set_source(c); [I_c(0.3), I_c(0.5), I_c(0.7)]
  → impact(128×3) / ranking(128×3, 通道号序) / 诊断 → subject_{i}.json（原子写）
```

`subject_{i}.json` 关键字段：`line/fold/subject_idx/subject_file/n_trials`、`sigmas=[0.3,0.5,0.7]`、
`impact_matrix`(128×3)、`ranking`(128×3，每 σ 一列按 rank 序的通道号)、`impact_rank_channelwise`
(每 σ 一列每通道名次)、`self_response`(128×3，源通道自身 MSE，诊断用不入和)、
`monotone_violations`、`sigma0_identity_pass`、`hparams`（seed/n_steps/batch/amp/CRN 说明/按线
剂量外推 caveat/nan_to_num 置零局限）、`checkpoint`+`ckpt_sha256`、`git_head`、`elapsed_min`、
`timestamp`。FAILED 被试写占位 json（诊断字段标失败）；重试须 `--overwrite`（「已存在即跳过」
语义下断点续跑不重试失败例），summary 侧排除并登记。

## 5. 风险与 caveat

- **剂量外推（按线）**：modma 三档全部 ≥ EC 网格顶 0.3；mild 0.5/0.7 超 0.48 ⇒ 响应可能饱和/
  排序扁平化；这正是稳健性检验对象，如实报告（不中断、不剪裁）。
- **NaN 处置改写（评审 B-3）**：`forecast_fc`/`perturb_and_measure` 内部 `torch.nan_to_num`
  在 MSE 计算前静默置零 ⇒ NaN 通道**不可事后检出**，真实风险形态 = 前向数值异常被零化后 MSE
  失真（无告警）；处置 = 逐源 RuntimeError/OOM 捕获 + json/REPORT 登记 nan_to_num 置零的读出
  局限；超 10% 源抛异常则该被试标 FAILED。
- **GPU 复现性**（10-29 教训）：共享 GPU 并发下前向核漂移 ⇒ 本任务跑批期间独占 GPU（串行、
  nohup 后台化 + 日志落 `$OUT/run/logs/`）；冒烟复跑逐位对拍，若漂移 → 该例降 CPU 单线程复算
  并在 REPORT 登记（正式数字口径随之声明）。
- **预算**：modma ≈42 min + mild ≈31 min GPU（评审 B-8 按 10-27/10-25 实测折算复核）+ 冒烟/
  分析 ⇒ 合计 <1.5 h。

## 6. 回滚

产物全部在 `outputs/10-30-critical-node-ranking/`（gitignore 内）+ 任务目录文档；代码为单新增
文件，回滚 = 删文件；无任何共享代码/注册件改动。
