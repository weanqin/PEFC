# design — 10-24-exitgnn-ec-unbiased

## 0. 总览

三处改动 + 四步运行，全部增量式，旧调用面零影响：

| # | 改动 | 位置 | 性质 |
|---|---|---|---|
| C1 | v2 GC 参考注册（register_refs.py 镜像 v1 先例） | `refs/granger_bivariate_spectral/mild40-trial500-zscore-fs250-aic4-25-f1-125-favg/` | 新增目录 + 随档脚本 |
| C2 | `--flow-exit-gnn` 旗标 + `--task-label` + surrogate 元数据条件化 | `models/latent-flow-v1/ec_mild_unbiased.py` | 既有脚本增量（默认关） |
| C3 | v4 重映射 + 双 GC 对照离线脚本 | `models/latent-flow-v1/recompute_conn_v4.py` | 新增单文件 |

数据流：

```
EC_GC_1_125.zip ──staging──> refs v2 (40×refs_s{sid}.npz) ──┐
                                                            ├─> recompute_conn_v4.py
ec_main (σ* + conn v1 逐例 json/npz) ───────────────────────┤     ├─ recompute_v4/subject_*.json
   ▲                                                        │     └─ recompute_v4/gccorr_dual_summary.json
   │ main 模式（--flow-exit-gnn）                            │
exit-gnn ckpt ─> build_model_mild(stage2, flow_exit_gnn) ─> SignalProbeSite(μ=0) ─> τ(q=0.5@σ_ref=.3) ─> σ* 网格走查
                                                            │
refs v1 mild40（已注册）───────────────────────────────────┘
```

## 1. C1 — v2 GC 参考注册

### 1.1 路径与命名

- 源：`/root/autodl-tmp/EC_GC_1_125.zip`（用户 2026-10-24 拍板提供）→ 解压 staging
  `/root/autodl-tmp/ecgc_1_125_staging/`（仅取 `EC_GC_subject.npy` + `README.md`；interim/qc
  不入库——refs 目录只放注册产物，源 zip SHA256 留 meta）。
- 目标：`refs/granger_bivariate_spectral/mild40-trial500-zscore-fs250-aic4-25-f1-125-favg/`
  （命名对齐 v1 `…fixed8-f1-45-favg`，差异段 = 阶数选择与频段）。
- 产物：40 × `refs_s{sid}.npz`（ec float32 (128,128) + subject_id + group + params 串）+
  SHA256SUMS + meta.json + README-source.md（zip 内原文）+ register_refs.py（随档）+ register.log
  + **`refs/MANIFEST.md` 在「granger_bivariate_spectral（mne-connectivity 谱 GC）」小节表
  追加 v2 行 + 内容/边界注记**（refs 建档约定「新增参考另建子目录并在本清单追加一行」；
  评审 F1）。

### 1.2 subject 映射与四锚点 bitwise 复验

- 规则（同 v1）：sid = EEG_ALL 顺序 index + 1（前 20 MDD / 后 20 HC）。
- 复验（v1 先例同款，重做不沿用）：EEG_ALL.mat `[1200,128,1500]` 第 idx 片
  `EEG[idx*30:(idx+1)*30]` vs manifest 数据文件
  `np.concatenate([x_p, x_f], axis=2)`（(30,128,1500)）四锚点 idx ∈ {0,19,20,39} ↔
  {m_1, m_20, h_1, h_20}（**h_20 即 sid 40**；评审 F2 勘误——manifest 无 h_40 文件）逐位
  相等（maxabsdiff=0.0 断言）。
- 组别交叉断言：manifest group vs index<20⇒MDD 逐 sid 一致。

### 1.3 sanity 断言 + meta

- shape/dtype：(40,128,128) float64；对角全 0；非负（≥0）；无 NaN；量级复核 vs README QC
  （mean≈0.0475，max≈0.568——容差对拍，防解压/版本错包）。
- meta.json：source（zip 路径+SHA256、staging、generator 描述）、method_params（1–125 Hz
  faverage、AIC per-subject 4–25、500−25 截断、fs 250、逐 trial z-score、30 trials 池化）、
  subject_mapping（规则+四锚点复验结果）、caliber_note（**数据有效带宽 ≈0.5–50 Hz，
  51–125 Hz 已衰减 −24 dB 以下 ⇒ 1–125 积分主要由 <50 Hz 贡献**；与 v1 逐被试矩阵
  Pearson median=0.148 登记为参考间差异事实）、files、sanity 统计。
- 幂等：可重放（覆盖本目录 npz/meta，不动旧档）。

## 2. C2 — ec_mild_unbiased.py 代理脑适配

### 2.1 CLI 增量

- `--flow-exit-gnn`（store_true，默认 False = w500 基座，旧调用面零影响——镜像 viz_mild
  先例「旗标默认关 = 原架构」纪律）。
- `--task-label`（默认空串 = 保持现行为硬编码标签）。落点规约（评审 F7/Mandate A 钉死）：
  - **main 模式**：非空时 payload **新增顶层 `task` 字段**（现行无此字段；空串则不写
    ——旧面零变化）；`surrogate.task` 由 §2.3 旗标条件化接管，与 task-label 单一来源不叠。
  - **summary/gccorr 模式**：非空时替换 task 串的**任务名段**（保留 `Run 00X (…)` 注记尾）。
  - **summary 的 `reading_notes` w500 caveat 句**（现行硬编码「继承 w500 顶格未收敛」）：
    旗标开时替换为 exit-gnn caveat 句（幅度欠校准 + 二因素 + provisional），旗标关不动
    （评审 F5——防陈旧文案随新任务档案）。

### 2.1b 逐例 json 原子写（评审 F9 采纳）

- json 落盘改 tmp 文件 + `os.replace`（原子）；npz 同理 tmp+rename。数值零变化，纯写
  入机制增强——防 50 min 长跑中断留半截 json 被断点续跑「已存在即跳过」误判。

### 2.2 变体接线（run_main）

- `build_model_mild(device, _model_args(a.flow_exit_gnn), stage=2, obs_len, forecast_len)`：
  train_mild_depress 现行路径即在此分支替换 `model.flow = FlowVelocityNetExitGNN(...)`
  （viz Run 002 已验证 strict load）。
- `load_flow_ckpt` strict=True 即**架构错配安全失败**：w500 ckpt 进 exit-gnn 模型因缺
  exit_* 键必 fail，exit-gnn ckpt 进基座模型因多键必 fail——无需额外守卫。
- `_caliber_asserts` 增补：旗标开时断言 ckpt 路径含 `10-24-latent-flow-exit-gnn`（防同
  名 ckpt 文件 `fold0_mild_flow_best_graph_mexhat.pt` 在两个 run_dir 间的手误交叉）。

### 2.3 元数据条件化（payload）

- 旗标开时覆盖：`method` 中 surrogate 名（mild-w500 → mild exit-gnn variant）、`surrogate`
  块（task=10-24-latent-flow-exit-gnn run_exitgnn_fold0；ckpt_stage2_best=0.8241@E116
  早停 E146 真收敛；variant 描述 + flow_exit_gnn:true + exit-GNN 结构一句话）、新增
  `surrogate_caveat` 字段（幅度欠校准 0.645/0.581 + lr/结构二因素 + θ 候选判断待拍板 +
  EC provisional）。
- 旗标关（默认）：**旧字段值全部不变 + 纯新增键**（评审 F3 口径澄清）：`hparams` 增
  `flow_exit_gnn` 布尔两态都写（新增凭据键，非行为变化）；除此之外 payload 与现行逐字段
  一致（w500 复现面不动——diff 审预期范围即此一键 + §2.1/§2.1b 的落点/写入机制差异）。
### 2.4 兼容性

- summary/gccorr 模式逻辑不变（gccorr 的 REFS_DIR 常量本任务不用于交付——双 GC 对照走
  C3 新脚本；保留只为旧命令重放可用）。
- latent_flow 包体零改动；probe/数据管线/断言链（norm SHA/manifest SHA/硬三元组）零改动。

## 3. C3 — recompute_conn_v4.py（v4 重映射 + 双 GC 对照）

### 3.1 输入/输出

- `--main-dir`（默认 `outputs/10-24-exitgnn-ec-unbiased/ec_main`）、`--out-dir`（默认
  `…/recompute_v4`）；无 GPU；BLAS 守卫前置。
- 逐例：读 σ*/conn v1/sign/元数据 → `conn_v4 = −(ln σ* + 0.5·ln2πe) + 5/7` → 断言组（下）
  → 重映射档 subject_{i}.json（元数据随档拷贝 + formula/note 字段替换，先例
  recompute_inv_conn_gc.py 同款）。
- 汇总：`gccorr_dual_summary.json`，双参考块 `{v1_1_45_fixed8, v2_1_125_aic4_25}` 各含
  per_subject / group（mean±std/min/max）/ group_mean（组均值矩阵 Pearson/Spearman）。

### 3.2 断言组（8/8 例全过才写汇总）

1. **常数平移不变性**：spearmanr(conn_v4, conn_v1)=1.0 且 Pearson(conn_v4, refs)=
   Pearson(conn_v1, refs)（同精度 |Δ|<1e-9；v4 = v1 + Δc, Δc=0.5·ln2πe−2/7）。
2. **全正**：conn_v4.min() > 0（数学保证：零点 σ*=0.4943 > σ_max=0.48 ⇒ 钳上界的对也只是
   conn(0.48)=+0.0293>0；防 σ* 出网格异常值的哨兵）。
3. **abs≡raw**：np.array_equal(|conn_v4|, conn_v4)（全正的逐位推论，登记用）。

### 3.3 GC 对照协议（对齐 gen_unbiased_gc_corr.py / Run 007 先例）

- offdiag 16256 对（~eye(128)）；aligned（行=源 vs refs ec）/ flipped（vs ec.T）×
  Spearman/Pearson；v4 下 abs≡raw ⇒ abs 块与 raw 块相同（断言后登记一次）。
- refs 加载：两目录 `refs_s{sid}.npz` 各自 glob；sid ↔ subject_file 逐例断言对齐（先例）。
- 描述性登记字段（descriptive_only）照搬先例：不构成代理脑资格判定。
- 并置纪律：与 w500 Run 007 先例数字并置时显式标注**跨代理脑（θ 不同）描述性并置**——
  流程口径相同但代理脑不同，数字差异 = 代理脑差异 + 噪声，不可作复现性判读（评审建议）。

## 4. 运行序列（与 implement.md 对应）

| Run | 模式 | 成本 | 门 |
|---|---|---|---|
| Run 000 | C1 注册 | CPU 分钟级 | 四锚点 + sanity 断言 + MANIFEST 追加 |
| Run 000b | dry-run：旗标开 1 例 × 8 源 × 4 网格 + **旗标关等价性对拍**（评审 Mandate A 建议采纳：旗标关同参 mini 跑入本任务树 scratch，σ*/τ 与 w500 在档 `dryrun_mini/subject_0.json` 对拍 max|Δ|=0 逐位断言——零成本运行时等价证据，不触旧树） | GPU ~3+3 min | strict load + 全链路 + 等价对拍 PASS（非正式口径存证） |
| Run 001 | main 旗标开 8 例 × 30 trial | GPU ≈50 min（w500 参照 46.9 min + 出口消息传递前向增量未实测——**参照非硬门**，首例 elapsed 出来登记迁移成本事实） | 8/8 + P(resp>τ)=0.500 |
| Run 002 | summary | CPU 秒级 | 断言组过 |
| Run 003 | C3 v4 + 双 GC | CPU 分钟级 | 断言组 8/8 |

- 断点续跑照旧（json 已存在跳过）；GPU 失败单例重跑不重算全量。

## 5. 风险与回滚

| 风险 | 缓解 |
|---|---|
| 同名 ckpt 文件跨 run_dir 手误 | §2.2 路径断言 + strict load 双保险 |
| v2 参考顺序错位 | 四锚点 bitwise 重验 + 组别交叉断言 |
| v2 与 v1 GC 结构差异大 | 双对照本意；参考间 Pearson median 0.148 登记为事实，不做优劣判断 |
| exit-gnn 幅度欠校准污染 EC 读数 | caveat 三处随档（payload/README/experiment.md），结果标 provisional；与 w500 EC 先例并置判读 |
| GPU 中断 | 逐例 json 落盘即跳过续跑 |
| 回滚 | 三改动全增量：旗标默认关 / 新 refs 目录 / 新脚本新输出树；git revert 单提交即净 |

## 6. 明确不做

- 不改 latent_flow 包体任何文件；不动 w500 EC 旧输出树与 v1 refs。
- 不做 q ladder、不做其他公式变体重算（v1 已随 main 落档，v2/v3 先例在 w500 任务）。
- 不做 θ 候选资格判定、不做 exit-gnn 消融（拍板项，另行任务）。
- 不做含对角口径 GC 对照（Run 008 先例已证协议算术结果，不携新信息）。
