# 可视化规范（visualizer 统一与 nature-figure 风格）

**适用范围**：本研究全部产图场景——实验分析图、代理脑检验可视化（指标 2 人工视觉检验 / 指标 3 PSD 对比）、论文配图。

## 统一 visualizer

- 所有可视化**必须**经统一的 visualizer 类/模块产出；实验/训练代码内不得散落临时绘图代码。
- visualizer 以「数据 + 图类型/配置」为输入契约，图文件输出到本变体目录（路径纪律见 [artifacts-management.md](./artifacts-management.md)）；新增图类型扩展 visualizer，不另起私有脚本。
- 检验可视化（人工视觉检验、PSD 对比等）同样走 visualizer，图路径登记进 `{TASK_DIR}/experiment.md` 对应 Run 块。

## 美学风格统一（nature-figure）

- visualizer 的默认风格（字体/字号/配色/线型/版式）遵循 **nature-figure 技能**（`.zcode/skills/nature-figure/`，Python matplotlib/seaborn · R ggplot2 论文级绘图规范）的 figure contract。
- 创建或修订 visualizer 内的绘图代码、产出论文级图时，加载 nature-figure 技能按其流程执行；不得另起私有样式。

## trajectories 图类型口径 v2：medoid + 90% 预测区间（2026-10-13 登记，用户指定）

- 生成轨迹**不做全 K 叠加**；每面板 = past 上下文（灰）+ **medoid 代表轨迹**（红实线）+ **90% 预测区间**（红半透明带）+ 真实 future（蓝，独立参考曲线）。
- **medoid 定义**：k\* = argmin_k (1/(K−1))·Σ_{j≠k} d(x_k, x_j)，d = **完整多通道轨迹** RMSE（= Euclidean/√(C·F)，全局常数不改 argmin）；**代表曲线必须来自实际生成样本，禁止逐时间点统计量重构的虚拟波形**。
- **区间用分位数**（逐时间点 Q05–Q95），**禁 min-max envelope**（防少量极端样本主导可视化范围）。
- 数据契约：`gen_medoid/gen_q05/gen_q95 (W,C,F)` + `K`（图例标注）+ 可选 `note`（图例 medoid 项追加注记，如滤波口径）；medoid/分位计算在**正式产物口径滤波之后**进行（滤波口径见 [evaluation-calibers.md](./evaluation-calibers.md)）。
- 版本保全：口径变更重渲染时旧图语义后缀存档（见 [artifacts-management.md](./artifacts-management.md)），canonical 文件名承载当前口径。

## 跨方法连接矩阵对比热力图：量级悬殊时的色标契约（2026-10-25 登记，任务 10-25-symw-ec-viz）

- **坑**：两类连接度量直接并排对比时值域可差量级（实测 EC_pert(v4) [0.03,3.90] vs GC-EC
  [0,0.32]，~13×）——字面共用绝对色标会使弱度量面板仅占色标底部 ~7%，视觉不可读且易误读
  为"无结构"。
- **契约**：主图用**秩归一同尺度**——两面板同 cmap + 共享单一 [0,1] colorbar，颜色 = offdiag
  百分位秩（对角双侧掩蔽），原始数值范围注于面板标题；**另附绝对值补充图**（各测
  vmax=pooled offdiag P99.5 独立 colorbar，suptitle 明示 scales NOT shared）保量级诚实。
  逐被试相关 ρ 角注与矩阵重算值对拍断言（<1e-9）防注记-数据脱节。
- **对角掩蔽实现注记**：PDF collision 审计下 **hatch/patch 掩蔽不合规**（hatch stroke 未裁
  剪平铺几何进 PDF 且批量 fill 合并为整带 bbox，触发 FIX BEFORE DELIVERY）——用单 imshow
  栅格纹理叠加（RGBA fill-only）视觉等价且审计干净（先例 `viz_symw_ec_pairs.py`）。
