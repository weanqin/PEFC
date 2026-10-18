# implement — 虚拟扰动范式×响应度量低成本搜索（v1-fmt θ\*·fold0）

> 执行序 B0→B4；B2→B3 自动衔接，阴性/异常即停呈报。代码只落 `models/perturb-search-v1/`，产物只落 `outputs/perturb-search-f0/`（path-isolation）。

## B0 骨架与契约单测

- [x] 建目录 `models/perturb-search-v1/`（candidates.py / metrics.py / scorecard.py / run_search.py / config.py / tests/）与 `outputs/perturb-search-f0/{phase1-metric-search,phase2-paradigm-search}/run_01/`、`logs/`
- [x] `config.py`：候选注册表（6 算子定义式键 + r0 + 度量键）、窗集（fold0 test 前 3 例 × 前 8 窗，**win_idx = 受试者序 × 窗序固定枚举**）、K=16、seed=2026、gates 阈值（A/B/C，含 Gate C 预注册指派 例#1+#2 vs 例#3）、θ\* 路径——全部从本任务 design §2–§5 字面转录；**钉死常数单列**：N 波形（固定 seed/FFT 1–40 Hz 掩码/RMS 重归一/全局共享）、B 滤波（Butterworth ≤6 阶 + filtfilt）、Welch（nperseg=128/Hann/50% overlap/bin 中心归属/ε=1e-12）、M-FC（逐 k 相关阵后均摊）、null（逐靶 B 计算、逐边 24 值取 std）
- [x] `config.py` Gate D 增补（2026-10-18 用户拍板）：R_GRID_MULT 5 点倍率梯 + R_CLAMP 逐算子上限钳制（G 0.32/L 1.0）+ R_GRID 字面网格（钳制去重；r0 本点在格）+ GATE_D_STRICT=1.0（聚合曲线 100%）/GATE_D_RTOL=1e-9 + 批形状 559/1737
- [x] `candidates.py`：6 算子（G 沿用 perturb-ec-v1::apply_gain 只读复用——纯函数无 config 访问；N/O/P/L/B 新实现）；全部满足「只动源通道 / **r==0 短路返回 clone**（L 用代数形式 x_A − r·(x_A−μ_A)）/ 新张量」三契约；crn_noise 与批构造**本任务自实现**（读自身 config，不复用旧任务 build_variant_batch/crn_noise）；批构造 **r 网格化**（variant_specs 平铺格点表，idx = 1 + a·N_RT + j，r0 本点并入不重复行）
- [x] `metrics.py`：6 度量 + baseline 两半 null 距离同构实现（M-RMS 的 s_{w,B} 沿 response.py::baseline_scale 同构）
- [x] `scorecard.py`：Gate A/B/C + **gate_d（增补）**——|Δ| 聚合曲线（D_edge 24 窗 median → offdiag 边池 median）随 r 单调不减、100% 步不减才过、违反步明细返回、逐边单调占比登记项；阴性诊断含 D 缺口
- [x] θ\* 装载：loader 配方沿 perturb-ec-v1/loader.py，变体 config 换 v1-fmt（唯一名 `v1fmt_config`）+ config_hash 校验 + EMA
- [x] `run_search.py`：BLAS 护栏（numpy 前 setdefault 8 线程）→ 装载 → 逐窗批构造（**559/1737 行 r 网格化**）→ 单次 sample()（显式 eps/eta，SEED+win_idx；跨 r CRN 逐位配对）→ 度量/地板 → scorecard（**A/B/C @ r0 格点切片 + Gate D 全网格**）→ npz（逐格点 D 键 + gateD_curve + r_grid）+csv+summary.md（含 Gate D 违反明细段）落盘
- [x] 单测（run_tests.py 直调）：① r=0 六算子全部逐位 == baseline（短路路径）；② 扰动仅改源通道（其余 61 通道逐位断言）；③ 显式 eps/eta 同输入两次调用逐位相同；④ 度量合成数据 sanity（构造已知差 → D>0；恒等输入 → D=0）；⑤ 批构造行索引/逆映射（559/1737 批形）；⑥ `sampler.gen_lowpass_hz is None` 断言（fs=500 brickwall 坑防重开）+ 装载后 config_hash 校验；⑦ 输出级恒等：测试批（baseline 行 + 一行 r=0 变体）单次 sample() 后两行**逐位**相等（AC2 第一款的采样器级读法，同批形内成立）；⑧ Gate D：r 网格推导复算（倍率梯×r0→钳制→去重/r0 在格/批形 559·1737）+ 判定器合成数据（升势过/注入下降抓到该步/常数曲线过——「不减」含等号/R≤1 不判）
- [x] 验证：`cd models/perturb-search-v1 && python3 tests/run_tests.py` 全绿（本机 base 无 pytest，沿 run_tests.py 直调模式；2026-10-18 含 ⑧ 八项 PASS，log `outputs/perturb-search-f0/logs/b0-tests.log`）

## B1 smoke（1 窗）

- [x] 阶段一批 1 窗跑通：形状 (559,16,62,128)、耗时登记、null 地板非零有限（2026-10-18：wall 0.4 min / sample 3.2s；六度量 null 全正有限；log `logs/smoke.log`）
- [x] 验证：npz 可读（含逐格点 D 键 54/gateD_curve 12/r_grid 2）、scorecard.csv 行数 = 度量数 × 探针数 = 12；G 聚合曲线 0.0115→0.0585 单调升（机械正确性）

## B2 阶段一主 Run（24 窗 × 双探针 r 网格，定 M\*）

- [x] experiment.md 首个 Run 块**跑前**登记实现口径（受试者 ID/窗/K/seed/批形状/win_idx 固定枚举规则/通道顺序对齐 splits manifest/r 网格与 Gate D 口径）
- [x] 主 Run（nohup，logs/ 落盘）：scorecard.csv（6 度量 × 2 探针 × gates A–D 列）+ 逐格点 per_edge_D.npz + summary.md（含 Gate D 违反明细段）——4 次调用达成（前 3 次崩于 B0 起潜伏的三处 scorecard 阴性/合并池路径 bug，已修 + 单测 ⑨ 回归锁定；Run#1 崩溃前 npz 与 Run#3 逐位一致 = CRN 跨 Run 复现）
- [x] M\* 判定：**如实阴性**（全候选未过四门；最佳可得 M-PSD maximin 0.56 照登；Gate D 12/12 全过为首个阳性事实）
- [x] 验证：wall 7.4 min（AC6 30 min 预算内）；Run 块结果/结论已回填

## B3 阶段二主 Run（24 窗 × 6 范式 r 网格 × 三度量集）

- [x] **度量集 = {M-RMS, M-PSD, M-time}（用户 2026-10-18 拍板**：阶段一阴性无单 M\* 的显式偏离，design §6 登记；run_search.py 扩多度量——`--metric` nargs="+"、缺省 config.PHASE2_METRICS、逐度量 P\* + 3×6 矩阵呈报）
- [x] 批 1737 行（62 源 × 28 格点），同 win_idx 复用 eps/eta：**硬断言 = 同窗 eps/eta 张量跨阶段逐位相同**（24/24 sha256 过）；baseline 输出 allclose（rtol=2e-6）24/24 过且**逐位相等（max_rel=0）如实照登**
- [x] 主 Run + scorecard（6 范式 × 3 度量 × gates A–D 列）+ 逐度量 P\* 判定显式登记：**三度量均如实阴性**（最佳可得 L/L/O maximin 0.54–0.60；唯一 Gate B 过线 = N×M-time 2.166；结论限定「各自 r0 下」；联合 {P×M} 无过线候选留档）
- [x] 验证：wall 9.9 min（预算内）；experiment.md 第二 Run 块已回填（2026-10-18 Run#1 一次通过）

## B4 收尾

- [ ] trellis-check（对照 check.jsonl 清单；重点：口径与 design 一致、CRN 契约、路径隔离、记录简洁性）
- [ ] progress.md 当前状态更新（M\*/P\* 或阴性）；context.md Study II 行补一句（3.3 spec 候选：evaluation-calibers/perturb-ec-pipeline 的算子 P/度量 D 行更新与否）
- [ ] 3.4 提交（chore/feat 按仓库惯例）

## 回滚点

- B0 单测不过 → 不进 B1；B1 形状/地板异常 → 修 B0 不动口径；B2 阴性 → 停在 B2 呈报（B3 不启动，避免在无 M\* 下空转）。
