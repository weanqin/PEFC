# 数据集契约 — mild-depression（静息态样带 EEG · HC/MDD · 预切 trial）

> 实测日期：2026-10-21（数据根逐例字段核对 + 5-fold manifest 生成，任务 10-21-latent-flow-mild-train S1）。「待验证」条目逐项补测后去除。

## 1. 来源与版本

- 路径：`/root/autodl-tmp/Mild Depression/`；来源 `Mild Depression.zip` 解压（目录名含空格，脚本引用须加引号），上游切分脚本与处理报告在数据根内（`split_eeg.py` / `处理报告.md`，2026-10-21）
- 版本标识：逐例预切 release（无 v 号）；**同根合集源 `EEG_ALL.mat (1200,128,1500)` 只读不消费**——带标签逐例文件为唯一消费入口
- 规模：**HC 20 / MDD 20 共 40 例**（h_1..h_20 / m_1..m_20），每例 30 trial

## 2. 组织与字段

- 每例 .mat（float64）：`x_p (30,128,1125)` + `x_f (30,128,375)`（trial,channel,sample，trial 内 0–1124 / 1125–1499 连续相接，3:1）；元数据 `label`（'HC'/'MDD'）、`label_code`（MDD=1/HC=0）、`subject_id`、`subject_code`、`trial_ids/trial_index`、`fs=250`、`x_p/x_f_start_sample`（0/1125）、`source_file=EEG_ALL.mat`
- **分组/label 语义（登记为事实）**：上游切分按任务默认规则划分——前 20 被试（全局试次 0–599）= MDD = subject_id 1–20，后 20（600–1199）= HC = subject_id 21–40；**分组唯一依据 = 上游处理报告 §4 的堆叠位置规则**，文件名前缀 m_/h_ 与其一一对应但系派生表象
- fs=250 Hz，每 trial 1500 点 = 6.000 s；数值量级 μV（min −489.8 / max 494.4，NaN/Inf = 0）

## 3. 预处理契约（上游已做，本侧不重复）

- 上游：`EEG_ALL.mat` 按 40×30 试次连续块切分逐例 + 3:1 时间窗（x_p/x_f）切分；**无事件对齐、无试次剔除、无滤波记录**（上游报告未载滤波/参考/伪迹处理口径——判读时按未知处理）
- 本侧消费（latent-flow-v1 线）：trial 预切、不重新切窗，每 trial 即一个 1500 点窗（obs=x_p 1125 + forecast=x_f 375）；归一化 = 训练折 28 例全长逐通道 mean/std（wrapper 内实现，严格 held-out，同 edra_fmt 口径）
- 瞳孔数据 `pupil_ALL.mat` 上游全程未读取、本线同样不消费

## 4. 已知坑（实测沉淀）

- **通道序无数据侧证据（假设链，用户 D4 签认 2026-10-21）**：源文件无通道名字段 → 上游未重排（报告 §3「通道顺序与源数据完全一致」）→ 强制假设数据通道 i ↔ E(i+1)（GSN-HydroCel-128 规范序）→ 若实际非规范序，mexhat 图先验错排（**dense 流不受影响**，`--dense-fallback` 为兜底）。`build_mild_geometry.py` 与 MODMA geometry（GSN-129 模板前 128）的恒等比对**只能防构建代码 bug，对通道序假设零证明力**
- 目录名含空格（`Mild Depression`），shell/Python 路径引用一律加引号
- mat 元数据均为 (1,1) 数组，取值用 `.item()`（`int(arr)` 触发 numpy 弃用告警）
- t_lat=16 池化粗化：obs 1125 → 16 ≈ 70.3 样本/latent 步（旧线 ≈ 23.4），本线 smoke/best-val 数字**不与 MODMA/EDRA/edra_fmt 线直接对比**

## 5. 下游消费规则

- 划分：**只读复用** `models/latent-flow-v1/splits/mild_5fold_seed42.json`（生成器 `make_mild_manifest.py`，已存在即拒跑）。算法（RNG 调用序列钉死，每群恰一次 `RandomState(42).permutation(20)` 调用）：MDD/HC 各自 `sorted(glob)` 后独立洗牌 → 连续 5 段 bounds [0,4,8,12,16,20] → fold i test = 两群第 i 段各 4 例（4+4=8）；val = 两群第 (i+1)%5 段前 2 例（2+2=4，随折轮转）；train = 余 28。**manifest SHA256 = `47987909c04348e57d98e77e1668678a7885109336df90bd0a5a635f865d881b`**（2026-10-21 生成；重生成须整链重登记）
- 逐折计数契约：train 28 例/840 trial、val 4 例/120 trial、test 8 例/240 trial；训练入口全量断言重放（计数/无交/文件存在/逐例 shape/fs/label/trial 数），失败即拒跑
- **被试 id 解析坑（10-24 实测沉淀）**：文件名 `m_/h_` 前缀是**组内编号**（h_i ↔ subject_id 20+i），不得从文件名 stem 解析被试 id——正确来源 = `load_manifest` 返回的 fold 记录 `train/val/test`（真实 subject_id 列表）；stem 解析曾致 viz 自动挑选必然 StopIteration、quality 脚本 test 集合错位（10-24 已修，既往 viz 靠显式 --subjects 绕过）
- geometry：`/root/autodl-tmp/Mild Depression/geometry_mild.pt`（D1 数据根单文件豁免，MNE GSN-HydroCel-128 模板 E1..E128，`build_mild_geometry.py` 可再生，默认拒覆盖除非 --force）
- **前缀窗 500 消费口径（10-24-latent-flow-mild-w500，用户 2026-10-24 拍板）**：每 trial 取前 500 点单窗（完整落于数据文件 x_p 字段内，**x_f 字段不参与**），窗内切 obs 375 / forecast 125，无滑窗无多窗；归一化 = 训练折 trial 前 500 点逐通道 mean/std（`train_mild_depress.py --window-len 500`，window_len 联动归一化范围与断言 obs+fc==window_len；默认 1500 = 原全窗口径 bit 兼容）；新口径特征（window≠1500/unfreeze-decoder/anti-uniform≠0）不显式传 `--run-dir` 时守卫拒跑
- 变更本契约任何口径 → 逐项排查对下游的影响并告知用户后方可生效

## 6. 训练实测补记（10-21-latent-flow-mild-train，2026-10-21）

- 窗内重切分已参数化：`train_mild_depress.py --obs-len/--forecast-len`（默认 1125/375；5:1=1250/250 → `run_ratio5to1/`、9:1=1350/150 → `run_ratio9to1/` 已实测，obs+fc==1500 硬断言）——数据文件 1125/375 形状断言不变。
- 数据侧观察（Run 005 viz 判读备查）：真实 trial 末段 ~5.8 s 后幅度衰减；生成侧末端 Euler 发散尖峰与之叠加，判读末段时须用去末端口径（见 evaluation-calibers.md 生成质量抽查条目）。

## 7. 训练实测补记二（10-24-latent-flow-mild-w500，2026-10-24）

- 前缀窗 500 口径 fold0 重训（用户拍板：batch 64/lr 5e-4/stage2 decoder+flow 联合仅冻结
  encoder/anti-uniform 0.1/归一化前 500 点/确定性采样沿用）：stage1 best 2.0992@E57 早停
  E87 健康；stage2 best 0.8356@**E150 顶格未收敛**（decoder 联合训练收敛慢于上轮
  flow-only 的 E122/E125）——150ep 预算对该冻结面偏低，延长属口径变更须拍板。
- 生成质量抽查（新口径）：medoid/real std 全程 0.957 / 去末 25 点 0.996（逐例均值
  0.88–1.05）——**上轮 0.5–0.74 欠幅基本消除**；coverage 双口径 0.0000 属确定性采样
  构造性（判读规则见 evaluation-calibers.md 增记）。数值抽查已持久化为
  `quality_check_mild.py`（上轮内联脚本无落盘，本轮修正）。
- forecast 段 1.5–2.0 s 远离尾段 ~5.8 s 衰减区——上轮「末端 Euler 发散 × 尾段衰减叠加」
  伪影在本口径不适用（去末口径仍保留对 Euler 末步发散的敏感）。
