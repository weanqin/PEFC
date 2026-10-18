# 移植 latent_flow 代码库为 npi 目标实现路径

## Goal

将 `/root/autodl-tmp/latent_flow`（由 `latent_flow.7z` 解压出的先前研究代码库）作为一条**目标实现路径**移植进 `/root/npi` 仓库：代码落入 npi 变体目录体系、在当前服务器环境可导入可测试可冒烟运行、遵循路径隔离与科研规范，并在技术路线 registry 登记——为后续在该路径上做数据适配与实验（独立后续任务）奠定基础。

## 背景事实（2026-10-18 勘察）

- **源库性质**：同一研究思路的先前原型（Windows 机，2026-06~08）——OC-LatentFlow 两阶段（stage1 VAE → stage2 潜空间流匹配）obs-conditioned 代理生成模型 + `attribution/bisect_sigma.py` 二分 σ 扰动搜索→有向连接矩阵（即「临界扰动强度」先前实现）+ GC/TE 基线 + DCGNN 分类（阴性）。含 de/dysib 后期探针线（2026-08，结论多为阴性/调试面）。
- **体量**：解压 4367 文件 / 7.5 GB；活跃代码面（除 `archive/`、`.git/`、产物目录）约 45.6k 行 Python，含 95 个单测。
- **git 状态**：检出分支 `08-15-substrate-cross-channel-probe`，HEAD `9efe1e3`（领先 `main` 204 文件 / +44k 行，含 de 线与 dysib）；工作树另有 2026-08-19~08-24 未提交改动（166 M + 48 untracked）与删失（645 D：AI 工具链目录 + **`data/` 加载器代码 9 个 .py**——可从 git HEAD 恢复，源库保持只读不动）。
- **数据与产物**：数据集不在包内（加载器经 env var 指向数据根）；`checkpoints/` 558 MB、`results/` 4.8 GB、`outputs/`/`logs/` 为可再生或历史产物；旧 `.trellis/` 已在工作树删除。
- **环境兼容**（当前 base 环境实测）：所需 torch/numpy/scipy/scikit-learn/mne/matplotlib/joblib/pytest **全部已在**（Python 3.12.3 / torch 2.8.0+cu128）；requirements 中 tqdm/numba/dateutil 活跃代码实际未用。旧机为 Python 3.13 / torch 2.12 nightly，存在 API 兼容性核验需求但无已知阻断项。

## Requirements

- **R1 移植载荷（代码面）**：活跃代码（含从 git HEAD 恢复的 `data/` 加载器）落位 npi 变体目录；排除旧 `.git`、旧 `.trellis`/AI 工具链目录、`archive/`、`checkpoints|outputs|results|logs`、PDF 论文与 `.obsidian`；具体基线（工作树 08-24 vs HEAD vs main）与目录形态由 design.md 定稿、用户评审确认。
- **R2 可运行可验证**：在当前 base 环境下包可导入、既有单测通过、至少一个入口脚本冒烟跑通（合成数据或 `--smoke`）；不安装任何新依赖（如遇缺失依赖，停下来问用户）。
- **R3 规范遵循**：路径隔离（一条路径一个目录，只读对待既有变体）；入口脚本 numpy 导入前 BLAS 线程守卫；任何运行产物落 `outputs/<变体>/`；experiment.md 登记 Run 块；源库 `/root/autodl-tmp/latent_flow` 全程只读。
- **R4 注册登记**：任务收尾（3.3）把 latent-flow 作为新路径回填 `.trellis/research/context.md` 技术路线表，并在 `environment.md` 登记移植事实。

### Non-goals（本任务不做，留后续任务）

- 不做 edra npz 新数据管道适配（spec/research/data-pipeline.md 契约接入另立任务）
- 不做任何训练/评估/扰动实验（含 fold0 前置验证——那是 study1 门禁流程）
- 不修改 latent_flow 模型算法本身（移植≠重构；仅做环境适配所需的最小改动并逐一登记）

## Acceptance Criteria

- [ ] AC1：npi 变体目录（名字经 design 定稿）内移植代码完整落位，`git status` 仅含本任务预期新增/修改，未触碰任何既有变体目录与共享区
- [ ] AC2：包导入冒烟通过 + 既有单元测试在当前环境通过（通过数/失败数如实登记；如有 torch 版本差异导致的失败，逐条登记原因与处置）
- [ ] AC3：至少一个入口脚本冒烟运行成功，产物落 `outputs/<变体>/`（或登记无需产物）
- [ ] AC4：experiment.md 含规划/执行/验证 Run 块；删失代码恢复清单与排除清单落 design.md
- [ ] AC5：3.3 完成技术路线表回填 + 环境契约登记；trellis-check 通过

## Notes

- 复杂任务：需要 `design.md`（落位形态/基线选择/目录映射/适配点/风险）与 `implement.md`（分步清单）后才能 start。
- 源库 7z 与解压副本留在 `/root/autodl-tmp`（数据盘），作为历史事实源；npi 侧只取代码面。

## 子研究路由（1.0b / 1.0c）

<!-- trellis-research-routing -->

> 任务创建时由 hook 自动播种。规划阶段必须填写以下四项；全部落定前不得进入 1.4 start 门禁（见 `.trellis/workflow.md` Phase 1.0b / 1.0c）。

- **分类结果**：`study1`（2026-10-18 用户拍板）——OC-LatentFlow 是 X_past→X_future 代理生成器主体，与 gpfm-mamba 系并列的 Study I 候选路径；σ 二分归因作为该路径自带能力登记，不改变路由。
- **依据**：用户在 1.4 评审问答中明确选择 study1（选项含 study2/保持 pending）。
- **新路径/方法检测（1.0c）**：`新路径：latent-flow → research 已触发` —— `.trellis/research/context.md` 技术路线表仅登记 gpfm-mamba 系（study1）与 perturb-ec/bivar-gc 系（study2），latent_flow（OC-LatentFlow 潜空间流代理 + bisection σ 归因）不在表中；适配研究已于 2026-10-18 完成。
- **research 产物**：`research/latent-flow-adaptation.md` —— 落位形态（嵌套包）/基线（HEAD `9efe1e3` + 工作树真实改动）/环境适配（零源码改动）/接口边界（EDRA 需 shim 属后续任务）/测试面（沙箱实测 293/302 通过）/裁剪与风险，末尾含推荐决策包。

### 用户决策记录（2026-10-18，1.4 评审问答）

1. **裁剪策略 = A + bundle 备份**：黑名单式删除 de/dysib/dcgnn/fm_gate 已终止线代码，判决文档全量保留；另做 `git bundle` 源库备份（单盘耐久性缓解）。
2. **路由 = study1**（见上）。
3. **checkpoints 558 MB 留 autodl-tmp 原位**（用户确认）。
4. 20 份实验笔记进 git、AC3 冒烟改道 shape_check：默认项未勾选亦未提出异议，按评审默认执行（wrap-up 时再提请复核）。
5. **用户提示**：latent_flow 所用 MODMA resting 数据在 autodl-tmp 下（`modda_resting_process.zip`）——实施时复测该 zip 可读性（研究期实测 ENOENT 异常）；可读则解压到数据盘并挂为 `MODMA_DATA_ROOT`，使 MODMA 依赖测试转通过；不可读则按边界声明登记。
