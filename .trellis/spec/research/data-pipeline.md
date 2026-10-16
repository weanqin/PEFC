# 数据管道规范（dataloader 统一接口与数据集划分）

**适用范围**：本研究全部 EEG 建模代码（Study I 代理脑训练与检验、Study II 扰动推断）的数据加载、预处理与划分。数据集事实以 [../data/](../data/index.md) 契约为准。

## Dataloader 统一接口

- 数据进入模型前**必须**经 dataloader 加载；对数据的**全部处理——包括但不限于 z-score 标准化、时间窗切分——都在 dataloader 内完成**。
- **模型侧不得再做任何额外数据变换**：模型 forward 收到的张量即 dataloader 输出；归一化、窗切分、通道选择等一律不得出现在模型代码内。
- dataloader 输出的形状与含义（如 `X_past ∈ R^{C×T_p}`、`X_future ∈ R^{C×T_f}`）在 design.md 声明，窗口长度与数据契约一致（edra：`T_p=384`、`T_f=128`）。
- 标准化等统计参数的估计口径（如仅用训练折估计、验证/测试折复用）须在 dataloader 内固定实现并在 experiment.md 登记，防止折间泄漏。

## 数据集划分（5-fold 分层）

- 含分组标签的数据集统一 **5-fold 划分**，按分组（MDD / HC）**分层**，保证每折内两类受试者数量平衡。
- 划分以**受试者**为单位：同一受试者的全部数据（含全部时间窗）必须落在同一折，禁止跨折（防泄漏）。
- 划分确定后固定 fold 标识（fold id + 划分 seed）并登记，全程沿用；变更按 [evaluation-calibers.md](./evaluation-calibers.md) 口径变更纪律执行。
- study1 前置验证固定使用 **fold0**（门禁见 [surrogate-brain-qualification.md](./surrogate-brain-qualification.md)）。
- **fold0 内层长期口径（用户 2026-10-16 拍板，v4 系起累计）**：fold0 的训练折再切 **7/1/2 = 32/4/10 受试者**（train/val/test，经 `split_manifest_override` 固定 manifest 文件逐字节复用）；变体间对比一律同 manifest 同口径。此前 v1/v2 的 7/2/1（34/2/10）属历史口径，与 v4 系对比时须在判读随附口径差异清单（见任务 10-16 experiment.md）。
