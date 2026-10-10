# 可复现性规范

**目标**：任何实验结果都能被「三个月后的自己」或「审稿人」按记录复现。不可复现的结果不得写入结论。

## 跑前（写代码时）

- **随机性全部显式固定**：python / numpy / 框架 / random 的 seed 同源统一设置；dataloader 的 shuffle 与 worker seed 一并固定；框架级确定性开关（如 `torch.backends.cudnn.deterministic`、`use_deterministic_algorithms`）按所用框架惯例显式声明开或关
- **seed 是实验配置的一部分**，必须写入 `{TASK_DIR}/experiment.md` 的 Run 块配置区
- **环境快照**：依赖清单（`pip freeze` / `conda env export` / lock 文件）随变体目录保存；环境任何变更至少在 Run 块注明，重大变更（框架大版本）视为新口径

## 跑中

- 训练/评估脚本必须支持从配置文件或命令行传入 seed 与全部关键参数；禁止散落硬编码的随机数或魔法常量
- **以 resolved config 为准**：把实际生效的完整配置 dump 到产物目录（如 `outputs/<变体>/run_<id>/config.yaml`），不以「代码里的默认值」作为口径依据

## 跑后

- **复现三要素齐全才算 Run 完结**：精确命令 + 环境快照 + resolved config（含 seed）。缺一即在 Run 块标注复现性缺口
- **已知非确定性来源**（多卡同步顺序、原子浮点归约、cudnn benchmark、dataloader 多进程）——要么消除，要么在结论中声明其量级（如同配置重跑两次的指标差）

## 自检问题

- 换一台机器重跑，指标差异是否可预期、可解释？答不上来 → 存在复现性缺口，先补记录再继续
