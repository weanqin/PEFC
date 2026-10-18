# 实验日志 — O×M-PSD 完整 EC 推断（v1-fmt θ\*·fold0 test；镜像 ec-nmtime 流程）

## Run 001 — O×M-PSD 剂量-响应 80 窗主 Run + 三 readout EC 矩阵（2026-10-18）

### 目的与假设

- 用 O×M-PSD（搜索任务 Gate A 最强 6.64 / Gate B 完美弥散 1.01 的对照极端）生产完整 EC_pert 矩阵，与 N×M-time 产物成对照；**只出矩阵不做对照判定**（镜像 10-18-ec-nmtime-f0 口径）。完成后按用户请求在最大强度 r=R_MAX=4.0 下可视化 past+future 波形。

### 配置

- 代码：`models/ec-ompsd-v1/`（镜像 ec-nmtime-v1：O 算子 + M-PSD 度量 + O 尺度网格）；θ\* 同 pin。
- 输入端：O（x_A ← x_A + r·σ_A，DC 偏置）；**R_GRID = (0, 0.0625, 0.125, 0.25, 0.5, 1.0, 2.0, 3.0, 4.0)**（9 点含原点，r0=1.0 居格点序 5，上端 4.0 = 搜索任务 O 网格顶点 = 波形图最大强度）；R_NEXT=5.0；slope 段前 5 点；批 497 行。
- 输出端：M-PSD（mean_k ‖log10 PSD^pert − log10 PSD^base‖₂，1–40 Hz，Welch nperseg=128/Hann/50%/ε=1e-12——perturb-search-v1 钉死值）。
- 窗集/CRN/聚合/readout/描述统计六项：同 ec-nmtime-f0（80 窗、win_idx 0..79、seed 2026、逐 subject 8 窗 median + global、slope/AUC/r\*(τ̃=0.25, flip)、33 契约 npz）。
- 命令：`cd models/ec-ompsd-v1 && nohup python3 run.py main > ../../outputs/ec-ompsd-f0/logs/main.log 2>&1 &`。

### 结果

- **Run#1 一次通过**：wall 8.1 min；80 窗 × 497 行；B0 七项单测全绿 + smoke 过（镜像骨架的适配面：O 算子 / M-PSD 度量 / O 尺度网格与常量）。
- 产物 33 契约 npz + dbar_curves.npz + summary.md（同 ec-nmtime 结构）。
- 描述统计六项（global）：**slope 自跨比 0.989**（subject 范围 0.917–1.076——全体 ≈1）；**删失率 0.0%**（3782/3782 全部达阈）；有效路径 3782；Spearman(slope, 距离) 见 summary；**split-half ρ 0.918**；dbar 单调占比见 summary。
- **最大强度波形图（用户请求）**：`figs/ec_ompsd_waveforms_r4.{pdf,png,…}`——r=R_MAX=4.0（=搜索任务 O 网格顶点，批内格点行 = 与主 Run 逐位一致）；四面板：past DC 偏置足迹（A=ch16）/ future 自响应（±SD）/ **M-PSD 读出面**（log10 PSD 1–40 Hz，gen-pert vs gen-base）/ future 跨响应（B=ch7）；QA 全过（碰撞 PASS、visual-judge pass）；快照 `figs/waveforms_win0_a16_b7_r4.npz`。

### 结论

- **对照极端完全复现**：O×M-PSD 的 EC 自跨比 0.989 ≈ 1（完美弥散，无归因能力）vs N×M-time 的 2.109——搜索任务 Gate B 的单点判定（1.01 vs 2.17）在 slope readout 与全量 80 窗上成立。
- 强度与覆盖极强：0% 删失、全路径有效（O@4σ 剂量下所有边达阈）——「幅度-特异性解耦」的矩阵级证据（强而弥散 vs N×M-time 的弱而特异）。
- 稳定性依旧高（split-half 0.918 ≈ N×M-time 0.904）：弥散本身在受试者间高度一致——θ\* 的 DC 通路是系统性的全局耦合。
- 本产物与 N×M-time 33 npz 构成同一契约接口下的成对对照资产，供后续对照/判定任务直接消费。
