# design — baseline-ec-v1

> 口径总源：本文件与 prd.md；变更须走 evaluation-calibers 纪律。方向约定全仓库统一：**矩阵 行=源 source，列=靶 target**。

## 1. 模块布局

```
models/baseline-ec-v1/
├── config.py        # 全部口径常量（路径/采样率/阶数范围/频段/种子）
├── resample_io.py   # npz 读取+字段核对、500→125 抗混叠降采、逐通道 z、缓存、通道距离矩阵
├── ec_refs.py       # 共享 VAR（Gram OLS+BIC）、条件 GC（Schur）、gPDC(|gPDC|²)、mne-c 二元交叉核对
├── compare.py       # 规范化→3782 边配对→距离偏相关 Spearman 主指标→t×Holm+bootstrap+天花板比值；EC_pert 接口
├── split_half.py    # 奇偶 2 s 段两半 EC → 天花板
├── run.py           # CLI：--stage {resample,refs,compare,split-half,smoke} --subjects N
└── tests/           # 独立脚本单测（①②③断言惯例，rng 2026）
```

- 复用：`sys.path.insert` 导入 `models/gpfm-mamba-v1/gc_psd.py` 叶子函数（`gram_pieces/bic_from_gram/conditional_gc/rank_transform/spearman_rho`），**只读复用、零改动**（v2 变体同模式先例）；该文件保持 numpy-only 不动。
- 图：`common/visualizer.py` 新增 `_draw_ec_matrix`（data={"ec_a","ec_b","label_a","label_b","stats"}），300 dpi PNG。

## 2. 数据口径（config 锁定）

- 46 例（`status=='ok'`）；逐例断言 `filter=='bp1-40+notch50(保真冗余)'`（异常即中止并列出，不静默跳过）；`ref_chain` 记录入 meta。
- `scipy.signal.resample_poly(x, up=1, down=4, axis=-1)` → (62, 21250)；元数据记录 `fs_raw=500, fs=125`。
- z-score：逐通道 (x−μ)/σ，μ/σ 为该被试全记录统计（ddof=0），落盘 float32。
- 通道距离：`chan_coords_62.npy`（米制三维）→ 62×62 欧氏距离矩阵 D，去对角展平为偏相关控制向量。

## 3. 共享 VAR 与两种参考（ec_refs.py；2026-10-13 定标，证据见 experiment.md Run 001）

- **定标结论（覆盖初稿的 BIC/OLS 口径）**：OLS Gram 条件数 ~1e18、split-half 天花板 ≈0 → **ridge λ_rel=1e-2 固定**（相对 tr(G_lag)/k_lag，截距不罚；GCV 欠平滑弃用；λ 由三被试 split-half 校准 0.91–0.94）；**p=8 固定**（ASCT p/2 谱峰锚点；p=8 与 16 等价）。BIC 表与 Ljung-Box 白性阶梯逐被试存档（LB 对真实数据超严 lb_frac≈1 属常态，不作判据）。
- **条件 GC**：`ridge_conditional_gc` —— 对每个源 j 删其滞后列、同 λ 重解（62 次/被试，非逐边），gc[源,靶]=ln(SSR_red/SSR_full)（SSR 二次型口径，可为微负，登记）；行=源。mne-c 交叉核对列：`spectral_connectivity_epochs(method='gc', mode='multitaper', indices=([seeds],[targets]), gc_n_lags=8, fdecim=64)`，1–40 Hz 带平均，seed→target 方向（已实测验证）；全量 46 例逐例计算（~25 s/例）。
- **gPDC**：与 GC 共享同一 Gram；ridge B → A_l[i,j]（目标 i 行/源 j 列）；新息 Σ = SSR_quad/n；**gPDC²_ij(f) = |Ā_ij(f)|² / Σ_k |Ā_kj(f)|²**（列=源归一 → ∈[0,1]、行=源存储后行平方和=1）；FFT n_fft=250（0.5 Hz @125 Hz）；宽带 = mean over f∈[1,40]；分频段落盘存档；转置独立函数 + 单测锁定。

## 4. 比较器（compare.py）

- 输入：refs = {"gc": (S,62,62), "gpdc": (S,62,62)} + subject_ids；EC_pert 接口 `load_ec_pert(npz)`：键 `ec (62,62)`、`subject_ids`（list 或 'global'）、`readout∈{slope,auc,rstar}`。
- 规范化：断言方向约定（行=源）；readout 为 rstar → 断言已按「大=强」翻转（接口契约：调用方负责翻转，comparator 只断言非负）；带符号 readout → |ec| 入主轨 + 符号矩阵存档。
- 配对：`offdiag_flat` → 3782 维向量（有向，非三角）。
- **主指标**：v̂=rank(v)（v_pert, v_ref, v_dist 三向量）→ 偏相关 r_{ab·c} = (r_ab−r_ac·r_bc)/√((1−r_ac²)(1−r_bc²))；逐被试逐方法得 ρ_s^{(m)}。
- 检验：单样本 t（n=46，H0 均值=0）→ Holm 跨 2 法；1000×受试者 bootstrap CI（seed 2026）；天花板比值 ρ̄/ceiling_m。
- 上下文行：未偏相关 ρ（同 bootstrap）。输出 `eval/comparison.json` + `eval/comparison_matrices.npz`。

## 5. split-half（split_half.py）

- 奇偶交错 2 s 段（250 样本）：段 idx 偶→半 A、奇→半 B（各 ~85 s 连续性中断每 2 s，两半同构）；各半以同口径（p_shared 同阶）算 gc/gpdc；天花板^{(m)} = mean_s Spearman(offdiag(EC_A,s), offdiag(EC_B,s)) ± sd；结果 `eval/split_half.json`。

## 6. EC_pert readout 登记（Study II 设计约束，本轮无代码消费）

主 readout = 低剂量斜率（剂量-响应 D_j(r) 线性段 OLS 斜率，r 网格低密度段加密）；AUC 次；r* 降级（右删失标注 + τ 阈值敏感性登记）。斜率/AUC 天然大=强，免方向翻转。写入 `.trellis/spec/research/evaluation-calibers.md` 附注。

## 7. 单测矩阵（tests/）

- `test_resample.py`：①带内(10 Hz)幅值保持 ② Nyquist 外(>62.5 Hz)分量被抗混叠抑制 ③长度=⌈85000/4⌉。
- `test_ec_refs.py`：①3 节点已知方向 0→1：gc[0,1] 显著>gc[1,0]（Schur 主口径）②同系统 gPDC²[源0,靶1] 列内占比最大且列和=1、值域 [0,1] ③转置函数正确性 ④BIC 恢复真阶（p=2 附近）⑤mne-c 交叉核对方向一致（容差断言）。
- `test_compare.py`：①偏相关解析例（构造 v_c = v_a + v_b 线性混合，偏相关符号/幅值符合解析值）②offdiag_flat 维度=3782 ③规范化断言（方向/非负/rstar 契约）④bootstrap CI 复现（seed 固定）。
- `test_split_half.py`：合成平稳 AR 两半 EC 高可靠（ceiling>0.8）。

## 8. 执行序（run.py stages）

resample(46) → refs(46：BIC 表 → p_shared → GC/gPDC/mne-c 交叉核对) → split_half(46) → compare（EC_pert 合成矩阵联调）。smoke = 1 被试全链。产物布局 `outputs/baseline-ec-v1/<run-id>/{data,eval,logs}`；Run 块先登记后跑。
