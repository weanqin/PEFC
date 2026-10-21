# design — 10-21-latent-flow-mild-train

> 技术设计。需求与验收见 prd.md；执行清单见 implement.md。范式先例 = 10-19-latent-flow-edra-fmt-train（wrapper 模式零包体改动），本设计只写差异与新增点。

## 1. 总体形态

```
models/latent-flow-v1/                 ← 变体层（新增 4 处，包体 latent_flow/ 零改动）
├── make_mild_manifest.py              ← S1：5-fold manifest 生成（一次生成即钉死）
├── build_mild_geometry.py             ← S2：GSN-128 geometry 构建（D1 豁免案）
├── train_mild_depress.py              ← S3：两阶段训练入口 wrapper
├── splits/mild_5fold_seed42.json      ← S1 产物（只读复用）
└── latent_flow/                       ← 零 diff（import 复用 _run_stage1/_run_stage2 等）

outputs/10-21-latent-flow-mild-train/
├── run_fold0/ ... run_fold4/          ← 每折：configs/ checkpoints/ logs/ + nohup 日志
└── summary.md                         ← S5：逐折 best/耗时/smoke 汇总表

/root/autodl-tmp/Mild Depression/
└── geometry_mild.pt                   ← D1 豁免（待拍板；替代案 = 落 run 目录）

.trellis/spec/data/mild-depression.md  ← S1：数据契约 spec（新增）
```

## 2. 数据适配（R2 落实）

### 2.1 数据事实（2026-10-21 本机实测）

- 逐例 .mat：`x_p (30,128,1125)` + `x_f (30,128,375)`，float64；`label`（'HC'/'MDD'）、`label_code`（0/1）、`subject_id`（MDD 1–20 / HC 21–40）、`trial_ids/trial_index`、`fs=250`、`x_p_start_sample=0 / x_f_start_sample=1125`、`source_file=EEG_ALL.mat`。
- 40 例 = h_1..h_20（HC）+ m_1..m_20（MDD）；`EEG_ALL.mat (1200,128,1500)` = 40×30 合集（不消费）。
- trial 内 x_p/x_f 连续相接（start 0 → 1125），每 trial 即 1500 点连续窗。

### 2.2 5-fold manifest（生成一次即钉死）

`make_mild_manifest.py`（变体层，BLAS 守卫 + argparse）：

- 例集发现：`sorted(glob('h_*.mat'))` / `sorted(glob('m_*.mat'))`（**必须排序后洗牌**，glob 目录序不可作确定性来源）。
- 划分算法（RNG 调用序列钉死：每群恰一次 `permutation` 调用、零追加随机调用）：MDD、HC 文件各自 `np.random.RandomState(42).permutation(n)` 独立洗牌 → 连续 5 段（bounds [0,4,8,12,16,20]）→ fold i 的 test = 两群第 i 段各 4 例；**val 轮转取** = 两群第 `(i+1) % 5` 段的前 2 例（各折 val 随折轮转、避免 4/5 折共用同一 val 例做早停参照；折内 val ∩ train ∩ test = ∅ 成立，跨折重叠无害——各折独立训练评估）；train = 余 28 例。manifest 内 `file` 字段钉**绝对路径**。
- 生成时全量断言（40 例、每折 28/4/8、test 逐群 4+4、三集无交、并集=全例、各折 test 互斥且覆盖全 40 例）+ 逐例 shape/fs/label/trial 数断言（30 trial、1125/375、fs=250）。
- 落 `splits/mild_5fold_seed42.json`：`{"dataset": "mild_depress", "seed": 42, "n_folds": 5, "subjects": {<sid>: {"group","file","label_code"}}, "folds": [{"fold": i, "train": [...], "val": [...], "test": [...]}, ...]}`。**已存在即拒跑**（防重复生成改口径）。
- 训练入口 `--fold` 只读加载该折，重放全部断言（计数/无交/文件存在/shape/fs），失败即拒跑——与 edra_fmt 入口断言同强度。

### 2.3 trial 级 Dataset（wrapper 内嵌，不用包内 MODMADataset）

- `MildTrialDataset(torch.utils.data.Dataset)`：输入 = 逐例文件列表 + `(mean, std)`（(128,1) float32）；逐例 `sio.loadmat` → `x = concat([x_p, x_f], axis=2)` (30,128,1500) → 归一化 `(x-mean)/std` → 每 trial 转 (1500,128) float32；产出 dict 与包内窗口契约逐键同构：`x_obs=(1125,128)`、`x_target=(1500,128)`、`t_obs/t_target`（`arange*dt`, dt=1/250）。
- collate 直接复用包内 `collate_fn`（契约同构故无需新写）。
- 断言：每例 30 trial → 每折窗数 = 30×例数（train 840 / val 120 / test 240）。
- 显存/内存：40 例×30 trial×1500×128×float32 ≈ 0.9 GB，全量常驻内存可行；逐例加载即弃，实际峰值远小。

### 2.4 归一化（严格 held-out）

- wrapper 内实现：train 折 28 例全部 trial 的 1500 点全长并入逐通道 float64 累加（sum/sumsq → mean/std；**std==0 → 1.0 护栏，对齐包内 norm_stats 口径**）；记录 mean/std 的 SHA256 与 train 文件列表哈希进 config JSON。
- 不调用包内 `compute_train_norm_stats`（其 reader 面向连续 (C,N) 文件，trial 文件不适用）——口径等价、代码面在 wrapper。

## 3. geometry（GSN-128，D1 待拍板）

`build_mild_geometry.py`（变体层，复刻 build_edra_fmt_geometry.py 的保存契约）：

- 基座 = `mne.channels.make_standard_montage('GSN-HydroCel-128')`（128 点，E1..E128 规范序；MNE 本机已验证可用）。
- 保存 `{"coords": float32 Tensor(128,3), "channel_names": list[str]}` → `load_geometry(path=...)` 消费契约一致；默认落数据根 `geometry_mild.pt`（D1 豁免案），`--out` 可改落 run 目录（替代案）。
- **模板一致性校验（防代码 bug，非数据通道序证据）**：与 `/root/autodl-tmp/modma/geometry.pt`（GSN-129 模板前 128 点重建）逐位坐标比对——本机已预检两者恒等（GSN-128 ≡ GSN-129 前 128），故该校验只能防构建代码错误，**对「数据通道 i ↔ E(i+1)」零证明力**（数据源无通道名且未重排，见数据根处理报告）。假设链（源无通道名 → 未重排 → E 序假设 → mexhat 先验错排风险 → dense 流不受影响）完整登记入 spec/data/mild-depression.md，D4 签认时向用户如实呈现局限。
- **通道序假设**：数据无通道名，强制假设通道 i ↔ E(i+1)。风险见 §8。

## 4. 模型与两阶段（R3 落实）

- `build_model_mild()` 复刻 train_edra_fmt.build_model_fmt：`OCLatentFlow(n_channels=128, latent_dim=64, t_lat=16, obs_len=1125, forecast_len=375, beta_kl_final=1e-4, use_fc_decoder=False)`；stage1 dense、stage2 默认 mexhat-graph（a_hat/coords 来自 geometry_mild.pt，`--dense-fallback` 反切）。
- 训练循环 import 复用：`_run_stage1/_run_stage2/Stage1Step/Stage2Step/_KLAnnealCallback`；`train_args` 钉死 `dataset="mild"`（决定包内 history 落 `logs/mild/`，wrapper 复制进 run 目录）、`unfreeze_vae=False`、正则 0、scheduler none。
- 超参（D3 拍板项，默认=旧库 CLI 默认）：epochs 50 / lr 1e-3 / batch 32 / patience 10 / seed 42 / beta_kl 1e-4。
- 规模预估（评审 B-F2 修正口径）：窗数为 edra_fmt 的 1/38，但每窗计算量 ≈ (1500×128)/(512×62) ≈ 6×，净缩放 ≈ 0.16 ×（edra_fmt 单 stage 60–66 min）→ **每折每 stage ~10–15 min，10 stage 全量 ≈ 1.5–2.5 h**（含 IO 余量），单卡顺序跑；以 fold0 实测回填 experiment.md。
- 判读 caveat（同 edra_fmt 先例「跨线不可比」登记）：obs 1125 → t_lat 16 的池化步长 ≈ 70.3 样本/latent 步（旧线 375→16 ≈ 23.4；edra_fmt 384→16 ≈ 24），latent 时间分辨率粗化 ~3×；本任务 smoke/best-val 数字**不与 MODMA/EDRA/edra_fmt 线直接对比**。

## 5. 训练入口 `train_mild_depress.py`（包装层）

复刻 train_edra_fmt.py 结构，差异点：

1. 常量：`OBS_LEN=1125, FORECAST_LEN=375, WINDOW=1500, TRIALS=30, N_CHANNELS=128`；`--fold {0..4}` 必选。
2. manifest 加载 = §2.2 只读断言路径；数据集 = §2.3 内嵌 MildTrialDataset。
3. 归一化 = §2.4 wrapper 内实现。
4. run 目录 = `outputs/10-21-latent-flow-mild-train/run_fold{N}/`；ckpt 命名 `fold{N}_mild_vae_best.pt` / `fold{N}_mild_flow_best_graph_mexhat.pt`。
5. smoke 评估 = wrapper 自带 1125/375 版（复刻 `_smoke_eval_fmt`，常量不同——包内 `_smoke_eval` 硬编码 375，不得 import，同先例纪律）。
6. config JSON 增记 fold 号、trial 契约断言记录、geometry 路径与 SHA256。
7. `--stage {1,2}`、`--limit-subjects`（冒烟）、`--save-path`（冒烟防覆盖）同先例；`save_path.parent.mkdir(parents=True, exist_ok=True)` 防冒烟子目录 ENOENT。

## 6. 产物布局（每折）

```
run_fold{N}/
├── configs/stage{1,2}_config.json     ← CLI+钉死 args+manifest SHA+norm SHA+geometry SHA+git HEAD
├── checkpoints/fold{N}_mild_vae_best.pt
│                fold{N}_mild_flow_best_graph_mexhat.pt
├── logs/stage{1,2}_history.json       ← stage1 wrapper 自写 / stage2 从包内 logs/mild/ 复制
├── stage{1,2}.log                     ← nohup 前台重定向
└── (smoke 结果行在 stage2 日志尾)
```

## 7. 决策点（1.4 评审拍板）

- **D1 geometry 落点**：数据根单文件豁免 `geometry_mild.pt`（推荐，同 edra_fmt D2 先例、后续合格性/EC 任务免 env 特判）vs 落 run 目录。
- **D2 执行节奏 + fold0 前置验证门禁偏离签认**：surrogate-brain-qualification.md §2 要求「先 fold0 单 seed 三项指标核算落盘、经用户同意后才 5-fold」；用户 2026-10-21 指令为直接 5-fold 训练（本任务不含合格性判定）——该偏离须在 1.4 由用户显式签认落档（选项：a. 签认豁免直接 5 折全量；b. 先跑 fold0、检视后再续 1–4 折——不违背用户 5-fold 意向，仅插一个人工检视点）。
- **D3 超参**：默认沿用旧库 CLI 默认（epochs 50 / patience 10 / lr 1e-3 / batch 32 / beta_kl 1e-4 / seed 42）。
- **D4 通道序假设签认**：GSN-128 规范序 E1..E128 假设；模板比对为恒等校验（零数据侧证据，见 §3），`--dense-fallback` 为兜底；如用户知道真实通道序/montage 文献需此时给出。

## 8. 风险与对策

| 风险 | 影响 | 对策 |
|---|---|---|
| 通道序非 E1..E128 规范序 | mexhat 图先验错排（邻接失真） | D4 用户签认；§3 与 MODMA geometry 旁证；`--dense-fallback` 可整跑不受影响 |
| 数据实际为轻度抑郁病人单群混杂（如 HC 也含亚临床） | 组别语义错 | 仅影响下游分组用途；本任务 MIX 训练不分组消费，label 只做断言 |
| 训练发散/KL 异常（128ch×1125 obs 比旧线 obs 更长） | θ 候选质量差 | 沿用旧库 KL 退火与 AMP/grad-clip；如实登记 history，不做本任务内调参（迭代属后续） |
| 早停过快（数据量小、val 120 窗噪声大） | best epoch 偏早 | patience 10 为默认；如实登记逐折曲线，异常折在 summary.md 标注 |
| manifest 生成算法与未来任务不一致 | 划分口径漂移 | manifest 钉死（sorted glob + 钉死 RNG 调用序列）+ 入口只读断言；spec/data/mild-depression.md 登记算法与文件哈希 |
| t_lat=16 池化粗化（1125→16 ≈ 70 样本/步，旧线 ~23） | latent 时间分辨率下降，数字跨线不可比 | §4 已登记判读 caveat；属旧库设计继承，不在本任务调参 |

## 9. 回滚

- 新增文件均为变体层/任务 outputs 的**未跟踪新文件**，回滚 = 手动删除清单：`models/latent-flow-v1/{make_mild_manifest.py, build_mild_geometry.py, train_mild_depress.py, splits/}`（`splits/` 整目录）+ `outputs/10-21-latent-flow-mild-train/` + 数据根 `geometry_mild.pt`（若建）+ `.trellis/spec/data/mild-depression.md`。不触及任何既有产物。
- 单折失败不影响他折（目录互斥）；重跑=删该折 run 目录重执行。
