# baseline EC 观测参考管线（GC+gPDC）与扰动 EC 主轨比较器

## Goal

在真实 EDRA 数据上计算两种 **observational directed-connectivity references**——全模型条件 GC 与 gPDC——作为 Study II 扰动推断 EC（EC_pert）的对比参考，并交付「距离偏相关 Spearman」主轨比较器（EC_pert 接口就绪、本轮以合成矩阵联调）与 split-half 噪声天花板。

## 背景与定位

- 本任务由 2026-10-13 与用户的多轮方案讨论收敛而成（比较关系修正、术语修正、方法实现细节修正、EC_pert readout 指示、两个诊断实验设计）。
- **比较关系（用户指定）**：主问题 = 两条独立配对比较 EC_pert↔EC^(GC)、EC_pert↔EC^(gPDC)；参考法互比不是科学问题（GC 与 gPDC 共享 VAR 拟合，互一致部分构造使然）。
- **术语（用户指定）**：GC/gPDC 一律称 observational directed-connectivity references，目标是**结构收敛性验证**，绝不称 ground-truth / 恢复真实因果。
- **TE 全系取消（用户 2026-10-13 拍板）**：条件 KSG-TE 逐边 S=50/S=100 置换实测 35–70 天不可行（cgroup 有效算力 16 核 + 8 维单边 5.46 s），二元/Gaussian 档亦不做；比较器按参考清单设计，TE 可后补。
- 上游：Study I 代理脑（任务 10-12-gpfm-mamba-edra-f0）；下游：Study II 扰动 EC（EC_pert 计算不在本任务范围）。

## Requirements

- **R1 数据口径**：EDRA v4，46 例（manifest `status=='ok'`，与 Study I 相同过滤）；逐例核对 `filter`（实测 bp1-40+notch50）与 `ref_chain` 字段，**不重复滤波**；降采 500→125 Hz（`scipy.signal.resample_poly(x,1,4)` 自带抗混叠，fs 原值/目标值入产物元数据）；逐被试逐通道 z-score（全记录统计，数据侧描述量、无折耦合；为 gen 侧可比性留口径：GC ln 比对角缩放不变、gPDC 构造尺度不变）。
- **R2 共享 VAR（单一全模型）**：受试者级 Gram 聚合 OLS VAR（整段单窗输入，复用 `models/gpfm-mamba-v1/gc_psd.py` 叶子函数，只读导入）；BIC 选阶 p∈{1..24}（对齐行集 row_from=p_max）；阶数分布集中则固定全体共用 p*；残差白性 QC（Ljung-Box 型）逐被试落表。
- **R3 GC 参考口径**：主口径 = 成对条件 GC（删列 OLS 额外平方和 / 块 Schur 补，全模型条件化，**行=源/列=靶**，无逐边独立重拟合——满足用户「single-full-model」要求）；mne-connectivity state-space GC 为**二元交叉核对列**（实测发现其逐连接建模，非全通道条件化，故不作主口径；方向语义 seed→target 已实测验证）。
- **R4 gPDC 参考口径**：与 GC 共享同一 VAR 拟合；readout = **|gPDC|²**（新息方差归一、列=源归一，值域 [0,1]、列平方和=1）；0.5 Hz 频率网格；宽带 = 1–40 Hz 均值进比较器，分频段（δ/θ/α/β/低γ，上限 40 Hz）矩阵落盘存档；**方向转置显式处理**（A_ij=目标 i 行/源 j 列 → 存储转置为行=源）并以已知方向合成 VAR 单测锁定。
- **R5 比较器（仅主轨）**：流程 = 规范化（行=源/列=靶、带符号 readout→|·|+符号图存档、断言大=强）→ **两法统一原始边值**（用户指定，TE 已不存在此问题）→ 配对（3782 有向边展平，非三角）→ **主指标 = 逐被试距离偏相关 Spearman ρ**（通道对距离取 chan_coords_62.npy 三维欧氏；三向量秩变换后偏相关）→ 上下文行：未偏相关 ρ → 检验：单样本 t（n=46）× Holm（跨 2 条比较）+ 1000×受试者级 bootstrap 95% CI（seed 2026）+ **ρ/split-half 天花板比值**。空间先验控制为主指标是用户三轮明确要求（Run 008/009 教训：K_s~R=0.891）。
- **R6 EC_pert 接口与 readout 登记**：比较器接受 npz（矩阵 (62,62) + subject_ids 或 'global' + readout∈{slope,auc,rstar}）；readout 设计约束（登记待 Study II 落地）：**主 readout = 低剂量斜率**（剂量-响应线性段 OLS），AUC 次、r* 降级（右删失+τ 敏感性）；r 网格低密度段加密；斜率/AUC 天然大=强免方向翻转。
- **R7 split-half 噪声天花板**：每被试奇偶交错 2 s 段分两半，各半同口径（含共用 p*）算两种 EC → 逐被试两半 Spearman → 各法天花板 mean±sd；用于 R5 的比值解释（避免孤立解释 ρ=0.3 算高算低）。
- **R8 产物与纪律**：代码 `models/baseline-ec-v1/`（路径隔离）；产物 `/root/npi/outputs/baseline-ec-v1/<run-id>/`（data/eval/logs）；图经 `common/visualizer.py` 新增 `ec_matrix` 图型（实验代码内不落散图）；单测沿用独立脚本惯例（合成真值方向断言，rng 2026）；experiment.md 四要素 Run 块先写后跑；环境契约回填 mne-connectivity 0.9.0 / pandas 3.0.6（IDTxl 随 TE 取消不装）；旧口径不动（gc_psd.py/fc_metric.py 保持 numpy-only）。

## 明确不在范围

- TE 任何变体（用户拍板取消）；次级指标轨（top-k/节点强度/不对称剖面/BA 等，用户「仅保留主轨」）；三角诊断 EC_gen/EC_pert 三腿（用户拍板只留接口）；EC_pert 本体计算（Study II）。
