# implement — 生成器 past 依赖审计（E1–E4）

> 代码只落 `models/pastdep-audit-v1/`，产物只落 `outputs/pastdep-audit-f0/`（path-isolation）。

## B0 骨架与契约单测

- [x] config.py（θ\* pin / 24 窗集 / K=16 / SEED=2026 / N 常量同源 / DOSES=(0.5,1,2) / E2 子窗 4×96 + r=1.0 / E3 臂 seed 偏置 / 批 435 行）
- [x] loader.py（复制自 ec-nmtime-v1，窗枚举换回 3 例 × 8 = 24 窗）
- [x] candidates.py（N 全段 + N 子窗掩码算子（boxcar + RMS 子窗重归一，σ_A 全 past std）/ crn_noise / E1b+E2 批构造）
- [x] run.py（E1a 解析 A 包络与子窗质量 → E1b/E2/E4 批采样存全 128 点 |δ̄| → E3 四臂独立 seed 采样 + 能量距离 → summary 判据表）
- [x] 单测：① N 子窗算子只动源通道/支集/RMS 重归一 ② r=0 恒等 ③ 批 435 行索引与公式 ④ CRN 逐位可复现（含臂 seed 偏置）⑤ E3 置换映射形状 ⑥ loader 窗枚举 0..23
- [x] 验证：`python3 tests/run_tests.py` 全绿

## B1 smoke（1 窗）

- [x] 批 435 行跑通；|δ̄| 曲线有限；E3 四臂采样跑通

## B2 主 Run + 判据表

- [x] experiment.md 跑前登记；主 Run（24 窗）
- [x] summary.md：T50 采样 vs 解析对拍（E1）、子窗位置响应（E2）、ρ 四臂（E3）、剂量×自跨 T50/φ（E4）、三假说判定
- [x] 诊断图（3 面板）+ nature-figure QA 链
- [x] Run 块回填

## 回滚点

单测不过不进 smoke；smoke 异常修 B0 不动口径；主 Run 异常即停呈报。
