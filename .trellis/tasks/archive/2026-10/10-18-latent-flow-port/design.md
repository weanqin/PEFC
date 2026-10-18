# Design：latent_flow 移植进 npi（10-18-latent-flow-port）

> 依据：`research/latent-flow-adaptation.md`（2026-10-18，含 /tmp 影子沙箱实测）+ 两轮 trellis-review（Mandate A/B，2026-10-18）修订。本设计直接引用研究推荐决策包，标注偏离处及理由；评审发现的修正（闭包语义/冒烟改道/行尾方向/工具在位性）已并入。

## 1. 目标落位与目录结构

**落位**：`/root/npi/models/latent-flow-v1/`（npi 变体命名惯例 `<方法名>-<变体名>`），内嵌完整包目录 `latent_flow/`（研究方案 B：源码 130 个文件的 `latent_flow.xxx` 绝对导入**零改动**成立；连字符目录不可导入，方案 A 排除；平铺拆散需重写 130 文件导入，方案 C 排除）。

```
models/latent-flow-v1/
├── README-port.md          # npi 侧移植说明（新增）：运行方式/env var/边界声明/裁剪口径与源锚点/旧基线数字
├── run.py                  # npi 风格包装入口（新增）：BLAS 守卫 + sys.path.insert + python -m 转译
├── .gitattributes          # *.py/md/csv/yaml text eol=lf（防行尾伪 diff，见 §4）
├── .gitignore              # 变体级补充排除：results/、.pytest_cache/（baseline 系脚本默认相对路径产物不入库，见 §3）
└── latent_flow/            # 移植包本体（源码零逻辑改动；仅行尾统一 LF）
    ├── __init__.py  config.py  requirements.txt  README.md（旧库 README，env var 文档源，保留）
    ├── models/  data/  training/  viz/  attribution/  baseline/
    ├── analysis/  scripts/  tests/  norms/  configs/
    └── docs/               # FOUNDATION.md / 架构与决策文档 / inventory / registry / experiments-notes/（20 份判决笔记）
```

**运行方式（两种，README-port 都写明，run.py 为推荐入口）**：
- 推荐：`python3 models/latent-flow-v1/run.py <scripts.name> [原样透传参数]`（run.py 契约：首个位置参数=脚本名，其余参数不经 `--` 分隔原样透传；内部置 BLAS 守卫后以 `python -m latent_flow.scripts.<name>` 执行）
- 直跑：`cd models/latent-flow-v1 && OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 python3 -m latent_flow.scripts.<name>`（env 前置写法必写进 README-port——包内无守卫，直跑绕过守卫属使用者责任）

此为对 npi「入口直跑」惯例的**声明式偏离**（path-isolation.md 只约束目录隔离/配置随目录/outputs 命名，不约束 import 风格；跨目录 sys.path 复用有 perturb-ec-v1/loader.py 先例）。

## 2. 移植基线与载荷

**基线**（研究 §2 实测：166 个"修改"中 160 个纯行尾差异，OC 核心工作树 ≡ HEAD 逐字相同；未提交实验全部完结带预注册判决，非半成品）：
- git 锚点 = 源库 HEAD `9efe1e3`（分支 08-15-substrate-cross-channel-probe，领先 main +44k 行含 de/dysib 线）
- 内容 = 工作树（2026-08-24 状态）+ 6 个真实改动文件随基线（dysib v2.2 线为完结代码、自带 441 行测试且通过）

**源→目标映射（rsync 源路径写死，防双层嵌套——源库是「仓库根=包根」布局）**：
`rsync -a /root/autodl-tmp/latent_flow/ /root/npi/models/latent-flow-v1/latent_flow/ --exclude=…`

**随移清单（In）**：包内代码与文档——`models/ data/ training/ viz/ attribution/ baseline/ analysis/ scripts/ tests/ norms/ configs/`、`__init__.py config.py requirements.txt README.md`、`docs/*.md|csv|yaml`（判决文档全量保留）；`experiments/*.md`（20 份判决笔记，含中文名）独立 cp 到 `latent_flow/docs/experiments-notes/`（PDF 与 `.obsidian/` 不随移）。

**排除清单（Out，rsync --exclude）**：`.git/ .trellis/ .zcode/ .claude/ .agents/ .codex/ archive/ checkpoints/ results/ outputs/ logs/ experiments/ .obsidian/ *.pdf __pycache__/ .pytest_cache/ AGENTS.md .gitignore .gitattributes`（旧 AGENTS.md 是旧库 Trellis 指令模板，注入子树会误导后续 AI 会话；旧 .gitignore/.gitattributes 由变体级新文件替代）。

**`data/` 9 个加载器恢复清单（内联，git HEAD 物化）**：`__init__.py, brainvision.py, datasets.py, de_features.py, electrode_positions.py, modma.py, norm_stats.py, splitter.py, subject_classification.py`——经 `git -C /root/autodl-tmp/latent_flow show HEAD:data/<f> > 目标` 落盘（源库全程只读；`__init__.py` 为空 blob 属正常）。恢复后与 rsync 落位文件逐字节核验（见 §7 步骤 0）。

**裁剪决策（待用户拍板，默认 A；语义 = 黑名单式「枚举删除」，非白名单）**：
- **A（推荐）**：**删除 de/dysib/dcgnn/fm_gate 已终止线代码 + 判决文档全量保留**。删除清单（执行前在 implement.md 冻结两列逐文件清单，以 import 图核验归属）：
  - `models/{de_vae.py, dysib.py, dcgnn_classifier.py, fm_model.py}`（fm_model.py 实测仅被 de/fm_gate 线 import）
  - `analysis/de/`、`analysis/fm_gate/`、`training/train_dysib*.py`
  - scripts 与 tests 中 de/dysib/fm_gate/dcgnn 归属文件（逐文件判定，冻结清单后 diff 验证）
  - **保留**（黑名单式的关键保护项，评审 Major-1 修正）：`models/graph_utils.py`（OC 共享）、`viz/probe.py`（active 探针脚本 import）、`scripts/run_oc_connectivity_global_tau.py` 与 `run_oc_connectivity_relative.py`（**真实存在的 OC 归因入口**——README Script Status 表中 `run_oc_connectivity_attribution`/`plot_stage2_compare` 名字与实际文件不符，以 git ls-tree 为准）、`scripts/build_edra_geometry.py`（§5 EDRA shim 的依赖）、`scripts/coord_audit.py`（其测试保留时）、`norms/`（DE 线归一化统计，小文件，随判决链保留）
  - 闭包正确性双判据：①裁剪后全量 pytest（预期口径预冻结，见 implement S1.5b）；②**剩余全部 `scripts/*.py` 逐个 import 冒烟**（`python -c "import latent_flow.scripts.<name>"` 循环）——pytest 只证「没删多」，import 冒烟把验证面扩到入口面、冻结清单 diff 证「没多留」
- **B**：全量随移（170 py / 45.6k 行），不裁剪。

**源库耐久性维度（评审 A-Minor-4 补入，随裁剪决策一并拍板）**：源库与 7z 的唯一副本在 `/root/autodl-tmp` 单数据盘；npi 仓库有 GitHub 远端。若选 A，被裁代码 + 完整 git 历史仅存单盘——建议低成本缓解：`git bundle` 存档源库（`git -C /root/autodl-tmp/latent_flow bundle create /root/autodl-tmp/latent_flow.bundle --all`，或推私有远端）；README-port 写明裁剪口径与源锚点 `9efe1e3`，防未来读者按判决文档找 de/dysib 代码时困惑。

## 3. 源码改动策略：零改动 + 包装层

**原则：latent_flow 包内源码零逻辑改动**（环境已实测兼容：Python 3.12 无 3.13-only API、torch 2.8 无 2.9+ 符号、依赖全满足、Agg 已统一、device 自动、数据根全有 env var）。允许的动作仅三类：行尾统一 LF（无语义变更）、`git show` 物化恢复文件、docs 内 README 追加 Linux 运行说明（文档性）。

一切适配经**包装层（npi 侧新增文件，不属于包本体）**：

| 适配点 | 落点 | 方案 |
|---|---|---|
| BLAS 16 核守卫 | `run.py` 顶部（numpy 导入前 setdefault OMP/MKL/OPENBLAS ≤8，`_blas_guard.py` 同款）；直跑入口的 env 前置写法入 README-port | 包装层 |
| outputs 落点 | `run.py` 默认 `--output-root /root/npi/outputs/latent-flow-port/` 透传给**有** `--output-dir` 类参数的脚本（OC 归因入口实证支持）；train 系无输出参数（产物只有 CHECKPOINT_DIR 下 .pt，已被 `.gitignore *.pt` 排除），不做透传 | 包装层 |
| **baseline 系相对路径产物**（评审 A-Minor-3：`run_baseline_connectivity.py:97` 等默认 `results/...` 相对路径，npi 根 .gitignore 未排除 `results/`） | 变体级 `.gitignore` 加 `results/` + README-port 显式警告（-m 直跑 baseline 会在 cwd 落 results/，建议经 run.py 且显式传 flag） | 包装层 |
| 数据根 | 环境变量 `MODMA_DATA_ROOT` / `EDRA_DATA_ROOT`（config.py:24,33 先查 env），运行说明给本机指向 | 零改动 |
| 旧 θ\* 权重（558 MB） | 不随移；README-port 登记原位路径与逐 ckpt 清单引用；`CHECKPOINT_DIR` 保持包内派生（新训练产物落包内，`.gitignore *.pt` 排除） | 零改动 |
| baseline 并行 | `--n_jobs` 经 run.py 默认 ≤8（16 核 cgroup 口径） | 包装层 |

## 4. 入库卫生：行尾与 git

- **行尾方向事实（评审修正，与 research §2.3 表述相反）**：实测工作树 `models/oc_latent_flow.py`、`attribution/bisect_sigma.py` 等为 **CRLF**，git HEAD blob 为 **LF**。移植统一目标 = LF：`find … \( -name '*.py' -o -name '*.md' -o -name '*.csv' -o -name '*.yaml' \) -exec sed -i 's/\r$//' {} +`（**dos2unix 本机未装，不安装**，用 GNU sed）；`git show` 恢复的 9 个文件天然 LF 无需转换。
- 落 `models/latent-flow-v1/.gitattributes`（`*.py text eol=lf`，md/csv/yaml 同）。
- 验收硬检查：变体目录内 `grep -rIl $'\r'` 为空；对 9 个恢复文件做 `git show HEAD:data/<f> | sed 's/\r$//' | diff - <dest>` 逐字节核（零语义 diff）。
- 落位 commit 注明源锚点 `9efe1e3` 与 7z 来源。**注意**：工作区既有未提交变更 ` M .trellis/research/environment.md`（前会话遗留）不并入移植 commit。

## 5. 与 Study I/II 的边界声明（本任务只移植不融合）

- **σ 二分 ≠ perturb-ec-v1 四要素**：算子（源通道加性高斯 vs 增益乘性）、度量（逐靶 MSE vs CRN 配对 RMS）、确定性（路径确定性隐式 vs 显式 CRN 同批共享噪声）、τ（共享中位数阈 vs 相对自响应 τ̃=0.25）均不同；方向语义一致（行=源、列=靶；σ\* readout 自带 `-(log σ* + 0.5·log(2πe))` 翻转使大=强，与 npi `larger_is_stronger` 契约同向）。对照表见研究 §4.2。
- **旧库 θ\* 不自动继承 npi 代理脑资格**：固定 θ\* 前须按 `surrogate-brain-qualification.md` 三项检验重新验收。
- **EDRA 数据接入 shim 属后续任务**：旧库吃 BrainVision .vhdr（本机 0 个），现行契约为 npz（edra_process_v4 @500Hz / edra_processed_fmt @250Hz）；窗长 375/125 vs 384/128。shim = npz Descriptor + reader + geometry_edra.pt 生成（依赖 `build_edra_geometry.py` 随移保留），另立任务。
- **MODMA 线当前不可验**：`/root/autodl-tmp/modda_resting_process.zip` 读取异常（目录项可见、open 恒 ENOENT，研究 §4.3 实测）；修复前 MODMA 相关运行/测试（4 个数据依赖失败项转通过）不在本任务验收内，如实登记。
- **train `--smoke` 数据阻塞（评审 Major 修正）**：`--smoke` 仅设 patience=0 禁存 ckpt，仍走真实 cohort 加载（`train_oc_latent_flow.py:490-497`）；本机两 cohort 均不可加载 ⇒ **训练冒烟登记为数据阻塞项**（待 MODMA zip 修复或 EDRA shim 后补跑），本任务 GPU 冒烟由 `shape_check_oc_latent_flow`（`torch.randn` 合成自足、秒级）承担。
- 移植后 npi 出现第三条实现路径：Study I 侧 gpfm-mamba 系与 latent-flow 并存（候选关系留后续实验任务裁决），互不写对方目录。

## 6. 读写路径清单（path-isolation 契约）

**写**：`/root/npi/models/latent-flow-v1/**`（新增）、`/root/npi/outputs/latent-flow-port/**`（运行产物，gitignore 排除）、任务目录工件（prd/design/implement/experiment.md/journal）；数据盘两处（§9 拍板项）：`/root/autodl-tmp/latent_flow.bundle`（源库备份）、`/root/autodl-tmp/modma/{HC,MDD}/`（zip 解压，S0.2 可读分支）。
**读**：`/root/autodl-tmp/latent_flow/**`（源库，全程只读，含其 `.git`）、`/root/npi/.trellis/spec/**`。
**不触碰**：`/root/npi/models/` 既有 17 个变体目录、`/root/npi/common/`、既有 `outputs/` 各任务目录、数据集目录（本任务不读数据）。

## 7. 验证方案

0. 恢复盘逐字节核（§4 的 9 文件 diff 核验）。
1. 移植后全量 `OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 python3 -m pytest tests/ -q`（cd 包目录）：全量口径 **293 passed / 4 failed（MODMA 数据缺）/ 5 skipped**（沙箱实测）；若选裁剪 A，按冻结清单的 per-file 预期口径判 PASS/FAIL。
2. 裁剪 A 追加：剩余全部 `scripts/*.py` 逐个 import 冒烟循环（零 ImportError）。
3. `run.py shape_check_oc_latent_flow`：GPU 合成前向冒烟（BLAS 守卫生效判据：run.py 自检打印 OMP_NUM_THREADS ≤8；shape_check 退出码 0）。此步承担 AC3。
4. `git status` 干净性：变更仅 `models/latent-flow-v1/**` + 任务工件；`find models/latent-flow-v1 -name '*.pt' | wc -l` = 0；既有变体目录零变更；包内 `.pytest_cache/` 已清理不入库。
5. trellis-check（spec 合规 + 交叉一致性）。

## 8. 风险与回滚

| 风险 | 缓解 | 回滚 |
|---|---|---|
| 裁剪闭包不完整（漏删/误删） | 冻结两列清单 + 双判据（pytest 预期口径 + 全量 import 冒烟 + 清单 diff） | 补齐/回退文件重跑判据 |
| 行尾统一引入语义 diff | 9 个恢复文件逐字节 diff 核；其余文件 worktree 本为 LF 目标一致 | revert 单 commit |
| torch 2.8 运行期差异（单测外路径） | 单测 + import 冒烟 + shape_check 多层兜底 | 变体目录自包含，`rm -rf models/latent-flow-v1` 即完全回滚（源库原位未动） |
| MODMA zip 异常 / train 冒烟阻塞 | 边界声明 + 数据阻塞登记（§5） | 不适用（登记） |
| 45.6k 行（或黑名单裁剪后 ~30k）一次入库 review 负担 | review 焦点收窄：config/包装层/裁剪边界三类，算法内部信任单测+预注册判决链 | — |
| 源库单盘耐久性 | 裁剪决策附带 git bundle 备份选项（§2） | 用户拍板 |

**回滚总案**：任一阶段失败且不可修：`rm -rf /root/npi/models/latent-flow-v1 /root/npi/outputs/latent-flow-port` + revert 相关提交；experiment.md 以「中止登记」Run 块收尾（append-only，不删除）；若已做 3.3 的 context.md/environment.md 编辑则手工还原（未提交状态）。

## 9. 开放决策（2026-10-18 用户已全部拍板，见 prd「用户决策记录」）

1. ~~裁剪 A/B~~ → **裁剪 A + git bundle 备份源库**（黑名单式删除 de/dysib/dcgnn/fm_gate 线，判决文档全保留；`git -C /root/autodl-tmp/latent_flow bundle create /root/autodl-tmp/latent_flow.bundle --all`）
2. ~~子研究路由~~ → **study1**
3. ~~checkpoints 558 MB~~ → **留 autodl-tmp 原位**（用户确认）
4. ~~experiments 笔记~~ → **进 git 落 `docs/experiments-notes/`**（默认未异议）
5. ~~AC3 冒烟~~ → **shape_check 承担**（评审 Major：train `--smoke` 本机物理不可跑；默认未异议）
6. **MODMA 提示（用户新增）**：MODMA resting 数据在 autodl-tmp（`modda_resting_process.zip`）——实施时复测 zip 可读性；可读则解压到数据盘挂 `MODMA_DATA_ROOT` 使 MODMA 依赖测试转通过，不可读则按 §5 边界登记
