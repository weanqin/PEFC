# 可复现性规范

**目标**：任何实验结果都能被「三个月后的自己」或「审稿人」按记录复现。不可复现的结果不得写入结论。

## 跑前（写代码时）

- **随机性全部显式固定**：python / numpy / 框架 / random 的 seed 同源统一设置；dataloader 的 shuffle 与 worker seed 一并固定；框架级确定性开关（如 `torch.backends.cudnn.deterministic`、`use_deterministic_algorithms`）按所用框架惯例显式声明开或关
- **seed 是实验配置的一部分**，必须写入 `{TASK_DIR}/experiment.md` 的 Run 块配置区
- **环境快照**：依赖清单（`pip freeze` / `conda env export` / lock 文件）随变体目录保存；环境任何变更至少在 Run 块注明，重大变更（框架大版本）视为新口径
- **变体实验一律走 default-off config 钩子，不改旧路径默认行为**（惯例确立：v3 `prior_bandlimit_hz` → `split_manifest_override` → 早停 `early_stop_patience` → `lr_schedule` → v4a `bottleneck_rank` → fmt 任务 `norm_type`/`fs` + 函数参/CLI 钩子 `run_all(skip_gc)`/`--skip-gc-only`/`--outputs-root` → adaln 任务 `cond_adaln` → symw 任务 `--sym-w`/SymWAdam optimizer 注入钩子）：新变体能力以 config 字段 + `getattr(cfg, 字段, None)` 透传实现，**None/默认分支与原结构逐位一致**（state_dict 键不变、既有变体 config_hash 零漂移，须单测断言）；启用与否由变体目录的 resolved config 呈现，不以代码默认值区分变体。训练口径同样以钩子表达（早停 patience、恒定/退火 lr、norm 类型等），不复制训练脚本
- **训练期参数硬约束经 optimizer 子类挂接（symw 任务 2026-10-25 确立）**：每步参数投影/裁剪类约束（如 W←(W+Wᵀ)/2 对称化）以变体层 `Adam` 子类实现——`step()` 内 `super().step()` 后对受管参数 no_grad 原位投影；挂点依据 = `BaseTrainer` 的 optimizer 为外部注入且训练循环无逐步回调钩子（`latent_flow/training/trainer.py`）。纪律：受管参数列表单独传参（不进 param_groups/defaults/state，optimizer state 键面零污染）；旗标默认关时 else 半支 = 原 optimizer 构造逐字保留；约束语义（"每步"）随 super().step() 调用无条件生效，与 AMP/梯度裁剪共存无需额外处理
- **RNG 中性构建惯例（adaln 任务 2026-10-17 确立）**：钩子若引入**构建期消耗全局 RNG 的子模块**（如 nn.Linear 的 kaiming 初始化），必须在 `with torch.random.fork_rng(devices=[]):` 内构建——否则启用分支的构建会推移后续模块的随机数流（零初始化不归还随机数），同 seed 下共享权重与禁用分支不同，「同 seed 双构建逐位一致」断言必红、run 级 epoch-0 等价失效（trellis-review 双 Mandate 共同 Major 实证）。fork_rng 下两分支同 seed 共享权重逐位相同，epoch-0 step-1 损失可跨 Run 逐位复现（实测 v4afmt-main-01 vs v4afmt-adaln-main-01 = 1.4787474870681763 逐位相等）

## 跑中

- 训练/评估脚本必须支持从配置文件或命令行传入 seed 与全部关键参数；禁止散落硬编码的随机数或魔法常量
- **以 resolved config 为准**：把实际生效的完整配置 dump 到产物目录（如 `outputs/<变体>/run_<id>/config.yaml`），不以「代码里的默认值」作为口径依据

## 跑后

- **复现三要素齐全才算 Run 完结**：精确命令 + 环境快照 + resolved config（含 seed）。缺一即在 Run 块标注复现性缺口
- **已知非确定性来源**（多卡同步顺序、原子浮点归约、cudnn benchmark、dataloader 多进程）——要么消除，要么在结论中声明其量级（如同配置重跑两次的指标差）
- **非逐位确定 GPU 路径上的「旗标关兼容」判据（symw 任务 2026-10-25 实测确立）**：latent-flow mild 线 stage2 unfreeze-decoder 路径（bf16 AMP/cuDNN @4080S）同 seed 同代码重跑即非逐位——ckpt 全张量 max|Δ|≈1.4e-3、val loss 末位漂移 ~1e-4；「逐位等价」断言在该类路径上**不可证**（w500 在档的逐位等价先例不适用于此路径）。合规判据 = ①代码面：新增逻辑全部位于旗标为真的分支内（else 半支与原代码逐字一致）；②数值面：**三方对拍**（改动前 / 改动后旗标关 / 改动后旗标关重跑），接受条件 = |改动前−改动后| 与 |重跑−重跑| 同量级（改动前后差异不超出同代码重跑噪声）。对拍 run 须在 experiment.md 登记三方数值与 max|Δ|
- **原始产物缺失时的复现存证模式（10-27-lf-globaltau-f0-unbiased-repro 确立）**：复现目标 Run 的逐被试产物已被清理时，"100% 复现"的可承诺口径 = 配置/代码路径/seed 100% 对齐 + **文档仅存锚点统计级并置**，四步走：①从旧库任务档（prd/implement）+ domain.md 钉死原始配置的非默认值（本次实测 n_grid=16 非脚本默认 24、无偏噪声早于 sign 字段存在——两处口径陷阱均靠读档规避）；②盘点锚点可复算性并如实登记缺口（本次 GC/TE 基线矩阵已删即标 NOT RECOMPUTABLE，不静默降级）；③recipe 侧证：多读出变体并置（raw/abs/flipped），与原始值吻合者即原始口径的证据（本次 raw aligned +0.301 vs 原始 +0.31 吻合、abs 变体 +0.004 不吻合 ⇒ 原始为 raw-vs-raw Spearman）；④同机同栈确定性自检（重跑逐位 np.array_equal）+ 跨栈 caveat 写入产物（同 seed 跨 torch/硬件统计级一致，逐位等同不承诺且原始产物已删不可验）。锚点并置表与 recipe 假设随 `anchors_summary.json` 落档

## 自检问题

- 换一台机器重跑，指标差异是否可预期、可解释？答不上来 → 存在复现性缺口，先补记录再继续
