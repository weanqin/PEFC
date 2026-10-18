# implement — N×M-time 完整 EC 推断（只出矩阵版）

> 执行序 B0→B2；异常即停呈报。代码只落 `models/ec-nmtime-v1/`，产物只落 `outputs/ec-nmtime-f0/`（path-isolation）。

## B0 骨架与契约单测

- [x] 建目录 `models/ec-nmtime-v1/`（config.py / loader.py / candidates.py / metrics.py / readout.py / run.py / tests/）与 `outputs/ec-nmtime-f0/{run_01,logs}/`
- [x] `config.py`：θ\* 路径 + hash pin、窗集（全 10 例 × 8 窗，win_idx 0..79）、K=16、SEED=2026、N 算子常量（与 perturb-search-v1 登记逐位一致）、R_GRID 9 点含原点 + R_NEXT 推导 + SLOPE_SEG/SENS、MTIME_EARLY_SAMPLES=32、TAU_TILDE=0.25（+0.50 敏感）、ORIENTATION、批 497 行、产物根
- [x] `candidates.py`：N 算子（自实现，读自身 config；r=0 短路 clone / 只动源通道 / 新张量）+ crn_noise（同 SEED 纪律）+ 批构造（行 1 + a·8 + j）
- [x] `metrics.py`：d_time（|δ̄| 前 32 均值 + full/peak 随附）
- [x] `readout.py`：slope/auc/dtil/rstar/effective_paths/flip_rstar/save_ec_pert_npz（沿 perturb-ec-v1 结构自实现，读自身 config）
- [x] `loader.py`：从 perturb-search-v1 复制适配（窗枚举扩全 10 例；config 引用换本任务唯一名 `ec_config`）
- [x] `run.py`：BLAS 护栏 → 装载 → 逐窗批采样 → dbar（0 槽构造）→ subject/global 聚合 → 三 readout → 33 契约 npz + dbar_curves.npz + summary.md（描述统计六项）
- [x] 单测（run_tests.py 直调）：① N：r=0 逐位恒等 + 只动源通道 + 公式重建 + 波形带限/RMS=1/全局共享 ② 批 497 行索引/逆映射 + 行 0 逐位 baseline + 变体行只动源 ③ CRN 两次 sample() 逐位相同（GPU）④ d_time 合成解析值（恒等 0 / 脉冲 1.0 / full/peak）⑤ readout 合成全套（线性 dbar→slope 解析、AUC 梯形、dtil 对角≡1 与 0 槽、rstar 命中档/删失、flip、有效路径）⑥ loader 口径（hash/gen_lowpass/窗枚举 0..79 × 10 例×8）⑦ 输出级 r=0 恒等（GPU）
- [x] 验证：`python3 tests/run_tests.py` 全绿

## B1 smoke（1 窗）

- [x] 1 窗 497 行跑通：形状 (497,16,62,128)、耗时登记、dbar 0 槽 = 0 且 r>0 槽正有限

## B2 主 Run（80 窗）+ readout 落盘

- [x] experiment.md 跑前登记（本文件齐后先登记再跑）
- [x] 主 Run：dbar_curves.npz（10 subject + global + sha256 + 窗登记）
- [x] 三 readout × (10 subject + global) = 33 契约 npz 落盘
- [x] summary.md 六项描述统计 + experiment.md 回填

## B3 收尾（待后续）

- [ ] trellis-check / progress-context 同步 / 提交

## 回滚点

B0 单测不过不进 B1；smoke 形状/dbar 异常修 B0 不动口径；主 Run 异常即停呈报。
