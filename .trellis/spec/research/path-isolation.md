# 路径隔离规范（多方法 / 多路径并行）

**动机**：科研需要并行尝试不同技术路线与参数路径；文件混放会让结果无法归因、误覆盖、误复用。核心原则：**一条路径 = 一个独立目录**，代码、配置、输出三处同构隔离。

## 目录契约

- **每个方法/变体独占一个目录**，三处同构：
  - 代码：`<代码区>/<方法名>-<变体名>/`（如 `models/mymethod-v1/`、`preprocessing/datasetA-v2/`），或方法仓库内的 `variants/<name>/`
  - 配置：随代码目录存放，config 文件与该变体一一对应（一个变体一个 config，不做共享 config + 开关混装）
  - 输出：`outputs/<变体名>/...`，目录名必须含变体名（沿用项目既有输出目录命名惯例——初始化时在 `.trellis/research/context.md` 确认）
- **新变体一律新目录**。不再采用同目录后缀并排（如 `config_v2.py` / `config_v4.py` 共存于一个文件夹）的方式；既有的并排文件保留不动、视为已定版。
- **禁止**：
  - 不同变体写同一个文件或同一个输出目录
  - 跨变体读取其他变体的**中间产物**（`outputs/` 下非共享区的任何东西）
  - 就地修改已定版变体的代码或产物——新想法 = 新变体目录，旧版留给对照
- **只读共享区**（可直接引用，不得改写）：`dataset/`（含预处理产物契约，见 `spec/data/`）、论文与文献、跨变体公共 utils（依赖关系须在 `design.md` 声明）。

## 流程挂钩

- **Planning（1.1）**：`design.md` 必须列出本任务全部读写路径清单——新建变体目录、输出目录、只读依赖；与既有变体重名/交叠即规划缺陷。
- **1.4 审查**：`trellis-review`（Mandate B）核对路径清单与既有变体无冲突、无共享可变文件。
- **2.2 检查**：产物落点合规；发现跨变体写入 → 阻断并迁移。
- **归档**：变体目录与其 `outputs/` 对应关系写入 `experiment.md`，保证事后能从结论回溯到产物。

## 薄变体协议（2026-10-13 沉淀，gpfm-mamba v2/v3 实践）

消融变体的标准实现（「一条路径=一个目录」的细化，适用于**单变量消融**且基线已定版的场景）：

- 变体目录仅含：`config.py` **完整副本**（单变量差异 + `variant_name` + run 前缀语义）+ 各入口包装（`train.py` / `evaluate.py` / 凡按 `cfg.variant_name` 定位 checkpoint 的诊断脚本都要有对应包装）。
- 包装三要素：①sys.path 同时插入基线与变体目录、变体 config 遮蔽 `sys.modules['config']`（import 后校验 `__file__` 归属变体目录）；②run_id 前缀守卫（含 `--resume` 派生 run 名的校验）；③以独立模块名 importlib 加载基线入口并调其 `main()`。
- 基线代码只读复用零改动；跨变体共享**只读**固化划分 manifest（同参数时校验到基线的 manifest，不重写）。
- 共享库确需新能力：只加**默认关闭的可选参数**，经 `getattr(cfg, 字段, 默认关闭值)` 透传；必须验证基线 config_hash 逐位不变 + 既有单测回归通过。
- 已知坑（实测）：①按 `cfg.variant_name` 定位 checkpoint 的共享脚本若无变体包装，会在变体 run 目录找基线 checkpoint 名（FileNotFoundError）——v3 的 diag_hf 案例；②`F.pad(mode='reflect')` 长度 2 padding 仅支持 3-D 输入，跨维度共用须折叠 reshape（见 `gpr_prior.brickwall_lowpass`）。
