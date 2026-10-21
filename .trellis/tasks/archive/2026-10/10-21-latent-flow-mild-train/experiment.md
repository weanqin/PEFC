# 实验日志 — 10-21-latent-flow-mild-train

> 每次「跑批/训练/评估/统计」追加一个 Run 块：跑前先写「目的+配置」，跑后同会话补「结果+结论」，不留空块。
> 格式契约见 `.trellis/spec/research/experiment-log.md`；产物只落本任务声明的变体目录（`.trellis/spec/research/path-isolation.md`）。

## Run 001 — fold0 两阶段正式训练（stage1 VAE → stage2 mexhat CFM 流）（2026-10-21）

### 目的与假设

- 在 Mild Depression 数据集（128ch GSN-128、1500 点预切 trial、obs 1125/fc 375=3:1）上完成 latent-flow-v1 fold0 两阶段代理脑训练，产出 θ 候选 ckpt 并向用户呈报（D2=b 案检视点）；假设两阶段在该数据规模（train 28 例×30 trial）下稳定收敛（best val 有限值、无发散）。

### 配置

- 代码：`models/latent-flow-v1/{train_mild_depress.py, make_mild_manifest.py, build_mild_geometry.py}`（git HEAD 启动时记录于 config JSON；包体零改动）
- 参数：旧库 CLI 默认（D3 拍板）——epochs 50 / patience 10 / lr 1e-3 / batch 32 / beta_kl 1e-4 / seed 42；stage2 = mexhat-graph（geometry_mild.pt，D4 E 序假设签认）
- 环境：base conda @ 本机 4080S 32GB（`.trellis/research/environment.md`）
- 数据：`/root/autodl-tmp/Mild Depression/` 40 例；fold0（manifest `splits/mild_5fold_seed42.json` SHA256 4798790…d881b）train 28 / val 4 / test 8；归一化 = train 折全长逐通道统计（严格 held-out）
- 命令：`nohup python3 /root/npi/models/latent-flow-v1/train_mild_depress.py --fold 0 --stage 1 > outputs/10-21-latent-flow-mild-train/run_fold0/stage1.log 2>&1` → stage 2 同式

### 结果

- fold0 stage1（VAE）：50 epoch，best val **2.4368 @ E40**（E41–50 无改善触发早停，恰好跑满上限）；wall 312.5 s；smoke MAE=0.8099 / std_ratio=0.8297。
- fold0 stage2（mexhat-graph CFM）：50 epoch，best val **2.3315 @ E50（末 epoch，仍在改善、顶格未收敛）**；wall 248.7 s；smoke MAE=0.8591 / std_ratio=1.0135；证据链：adjacency (128,128) @ geometry_mild.pt（D4 签认）、非 flow 键零失配（flow 键 fresh 15/4 属图流预期）、trainable=flow only 488,897（总参 1,243,234）。
- 产物：`outputs/10-21-latent-flow-mild-train/run_fold0/{configs,checkpoints,logs}/`（双 ckpt+双 config+双 history+两 nohup 日志）。
- 波形检验可视化（visualizer `viz_mild.py`，trajectories-v2 口径：past 灰 + gen medoid 红实样本 + Q05–Q95 带 + real future 蓝；K=32、seed 42、test 3 例 sid 1/7/21 × trial0、8 通道 E{1,18,36,54,62,75,100,128}）：`run_fold0/viz/wave_subj{01,07,21}_trial00.png` + `viz_meta.json`；判读登记（AI 预判 + 待用户人工判）：生成段幅度量级与真实一致、无塌缩/无均值漂移、90% PI 带宽合理（与 std_ratio 1.0135 一致）；已知瑕疵 = medoid 高频偏平滑、末端 ~6.0s 部分面板分位带展宽（Euler 末步边缘效应，非全通道系统性）。
- 单折两阶段实测 ~9.3 min（评审预估 20–30 min/折为高估，原 20–40 min 全量预估量级反而接近）。

### 结论

- 两阶段稳定收敛（train/val 有限值、无发散、无 NaN）——θ 候选产出成功；**检视点：stage2 best 落在 epoch 上限且 val 仍在下降（E40 2.509→E50 2.332），50 epoch 预算对该数据量偏紧**，是否延长 epochs 属 D3 变更、待用户拍板。
- 下一步：D2=b 案检视点——向用户呈 fold0 结果，确认后续 fold1–4（口径不变或调 epochs）。
- **检视落定（2026-10-21 用户）**：波形判读暂判**不通过**；z-score 口径三问复核（训练折池化统计/输出对 z 域真实/viz 全 z 域，单域无失配）用户**确认无问题**；拍板 **epochs 150 全量重跑**（Run 002）+ **viz 口径 v2.1**（仅 real-future + gen-future，去 past 灰线）。

## Run 002 — 150-epoch 全量重跑 fold0–4（D3 变更）（2026-10-21）

### 目的与假设

- 修正 Run 001 检出的「50-epoch 预算偏紧（stage2 best 顶格未收敛）」：epochs 150 / patience 10 不变，fold0 重跑 + fold1–4 同口径，产出 5 折 θ 候选；假设 stage2 收敛到平台（best epoch 不再顶格）、val loss 低于 Run 001 同折水平。

### 配置

- 代码：同 Run 001（wrapper 零改动，仅 CLI `--epochs 150 --run-dir .../run_e150/fold{N}`）；viz 渲染口径改 v2.1（viz_mild.py，仅影响出图不影响训练）
- 参数：**epochs 150**（D3 变更，用户 2026-10-21 拍板）/ patience 10 / lr 1e-3 / batch 32 / beta_kl 1e-4 / seed 42；stage2 = mexhat-graph（geometry_mild.pt）
- 环境：base conda @ 本机 4080S 32GB
- 数据：同 Run 001（manifest 钉死不变；train 折归一化统计不变）
- 命令：`for k in 0..4: python3 models/latent-flow-v1/train_mild_depress.py --fold $k --stage {1,2} --epochs 150 --run-dir outputs/10-21-latent-flow-mild-train/run_e150/fold$k`（nohup 串行）
- 版本保全：Run 001 产物 `run_fold0/` 不覆盖，新跑落 `run_e150/fold{0..4}/`

### 结果

- **用户中止（2026-10-21 20:01，指令「结束训练进程」）**：fold0 stage1 进行至 ~E34（约 8.5 min）被 SIGTERM 终止（rc=143），串行脚本按设计 ABORT，fold1–4 未启动；残留产物 = fold0 部分日志 + config + 一个中途 best ckpt（**不完整、不构成 θ 候选**）；`run_e150/` 目录整体属未完成状态。GPU 现存 628 MiB 占用为**他任务进程**（ec-globaltau-v1 search_threshold.py，非本会话启动，未触碰）。

### 结论

- Run 002 未获结果即中止；50-epoch Run 001（run_fold0/）仍是本任务唯一完整训练产物；中止原因与后续口径（是否换超参/换执行方式/终止本任务）待用户指示。
- **后续落定（2026-10-21 用户）**：不做 5 折全量重跑，仅 fold0 stage2 @150 epoch（Run 003）。

## Run 003 — fold0 stage2 单段 150 epoch（Run 002 收窄版）（2026-10-21）

### 目的与假设

- 只针对 Run 001 检出的欠训练位（stage2 best 顶格 E50）延长训练：fold0 stage2 epochs 150 / patience 10 不变；假设 stage2 收敛到平台（best epoch 显著早于上限）、val loss 低于 Run 001 的 2.3315。

### 配置

- 代码：同 Run 001 wrapper 零改动；CLI `--fold 0 --stage 2 --epochs 150 --run-dir .../run_s2e150/fold0 --vae-ckpt .../run_fold0/checkpoints/fold0_mild_vae_best.pt`
- **stage1 VAE = 复用 Run 001 完整产物**（50-epoch best@E40，patience 触发早停=真收敛；不做 stage1 重训）
- 参数：epochs 150 / patience 10 / lr 1e-3 / batch 32 / beta_kl 1e-4 / seed 42；stage2 = mexhat-graph（geometry_mild.pt）
- 数据：同 Run 001（manifest 钉死、train 折归一化统计不变）
- 版本保全：run_fold0/（Run 001）与 run_e150/（Run 002 中止存档）均不覆盖，新产物落 `run_s2e150/fold0/`
- 训后动作：viz v2.1 口径（仅 real-future+gen-future）以新 flow ckpt 重渲染 fold0 test 3 例

### 结果

- **早停 E17（154.3 s）**：best val **2.5061 @ E7**（E8–17 连续 10 epoch 无改善，patience 10 触发）——**劣于 Run 001 的 2.3315 @ E50**；smoke MAE=0.8363 / std_ratio=0.9721；产物 `run_s2e150/fold0/`（ckpt+config+history+日志）。
- viz v2.1（仅 real-future+gen-future）已用**当前最优 ckpt（Run 001 50-epoch flow）**渲染 fold0 test 3 例 → `run_fold0/viz_v2p1/wave_subj{01,07,21}_trial00.png`（旧 v2 图版本保全于 `run_fold0/viz/`）。

### 结论

- **证伪「150-epoch 上限可解决 stage2 欠训练」**：瓶颈在 patience=10 × val 噪声平台（val 仅 120 窗），不在 epoch 上限——本轨迹 E7 后即入平台被早停截杀；且训练非逐位确定（bf16 AMP，Run 001/002/003 同 seed 同 epoch 的 val 已见偏差），stage2 val 下降缓慢高方差，单轨迹 best 高度依赖运气。
- 当前 fold0 最优 θ 候选仍是 **Run 001（2.3315）**；若续训需调的是 patience（如 30–50）而非仅 epochs 上限——属 D3 再变更，待用户拍板。

## Run 004 — Run 001 θ 候选生成质量数值抽查（coverage/medoid 幅度比）（2026-10-21）

### 目的与假设

- 对用户「波形判读不通过」给客观判据：在 fold0 test 3 例（sid 1/7/21）× 30 trial × K=32（seed 42，与 viz 同样本）上量化 ①逐时间点 Q05–Q95 覆盖率 ②medoid/real 逐通道 std 比。

### 配置

- 代码：viz_mild.py 同一采样管线（`sample_forecast` K=32 / medoid 同式）/ 内联脚本（无落盘代码，数字随本块登记）；ckpt = Run 001 flow best（2.3315@E50）。

### 结果

- **逐时间点 Q05–Q95 覆盖率 = 0.00%（90 trial × 128ch 全体，mean=max=0）**——真实未来曲线完全落在生成 90% 带之外。
- **medoid/real std 比 = 0.53–0.58**（3 例均值 0.58/0.53/0.56，min 0.38）——代表轨迹幅度仅真实的约一半。
- 与 smoke std_ratio 1.01 表观矛盾，机制假设（登记待验）：少量重尾样本（含末端 Euler 发散）抬高池化 std，而 90% 分位带仍窄——中心轨迹平滑欠幅 + 少数发散样本并存。

### 结论

- **支持用户「不通过」判读的硬证据：条件分布逐点覆盖塌缩 + 代表幅度减半**——该 θ 候选的生成质量缺陷非渲染口径问题（z-score 单域已确认、v2.1 图已核为 forecast-only 渲染）。
- 下一步方向（待用户拍板）：①stage2 续训调 patience 再试（Run 003 证据提示单轨迹高方差，收益存疑）；②按 Study I 迭代路径另立任务（损失侧/条件分布校准）；③仅登记负结果、任务收尾。

## Run 005 — 窗内划分 5:1（obs 1250 / forecast 250）fold0 两阶段重跑（2026-10-21）

### 目的与假设

- 检验 x^p:x^f 从 3:1 改 5:1（forecast 缩短至 250 点 = 1.0 s、obs 加长至 1250 点 = 5.0 s，用户拍板）是否改善条件生成质量——针对 Run 004 检出的 coverage 0% + medoid 半幅（0.53–0.58）；假设更短 forecast 段降低条件分布学习难度、更长 obs 段提供更强条件，coverage/medoid 幅度比改善。

### 配置

- 代码：`models/latent-flow-v1/train_mild_depress.py` 参数化窗内划分（新增 `--obs-len/--forecast-len`，默认 1125/375 = 数据原切分，向后兼容、Run 001 口径可原样复现；obs+forecast=1500 且均为正硬断言；数据文件 x_p 1125/x_f 375 形状断言不变——新划分是对拼接后 1500 点窗的重新切分，obs=前 1250 / forecast=后 250）；`viz_mild.py` 同步参数化（t_fc 轴/切片/xlim/标题）。包体 `latent_flow/` 零改动。
- 参数：**--obs-len 1250 --forecast-len 250** / epochs 150（两阶段）/ patience 10 / lr 1e-3 / batch 32 / beta_kl 1e-4 / seed 42；stage2 = mexhat-graph（geometry_mild.pt，E 序假设 D4 签认）
- 环境：base conda @ 本机 4080S 32GB
- 数据：同 Run 001（manifest 钉死不变；train 折归一化统计 = 全长 1500 点口径不变）
- 命令：`python3 models/latent-flow-v1/train_mild_depress.py --fold 0 --stage {1,2} --epochs 150 --obs-len 1250 --forecast-len 250 --run-dir outputs/10-21-latent-flow-mild-train/run_ratio5to1/fold0`（nohup 串行；stage2 自动加载同 run-dir 的 stage1 ckpt）
- 版本保全：run_fold0/（Run 001）、run_e150/（Run 002 中止存档）、run_s2e150/（Run 003）均不覆盖；新产物落 `run_ratio5to1/fold0/`
- 训后动作：与 Run 004 同法质量复查（test 3 例 sid 1/7/21 × 30 trial × K=32 seed 42：逐时间点 Q05–Q95 coverage + medoid/real std 比）+ viz v2.1 渲染（新口径 1250/250）

### 结果

- **stage1（VAE, 1250/250）**：早停 E44（patience 10），best val **2.5230 @ E34**；wall 286.0 s；smoke MAE=0.8823 / std_ratio=1.0795。
- **stage2（mexhat-graph CFM, 1250/250）**：早停 E132（patience 10），best val **2.0081 @ E122**（best 不再顶格、显著早于 150 上限，E122 后入平台）；wall 637.8 s；smoke MAE=0.8703 / std_ratio=1.0894；证据链：adjacency (128,128) @ geometry_mild.pt、非 flow 键零失配（flow 键 fresh 15/4 属图流预期）、trainable=flow only 488,897（总参 1,243,234）、stage1 ckpt = 同 run-dir 默认路径（Run 005 自产 VAE）。注：1250/250 与 1125/375 的 val loss 归一化维度不同，**跨口径 best val 数字不可直接比较**。
- 产物：`run_ratio5to1/fold0/{configs,checkpoints,logs,viz}/` + 双 nohup 日志；既有 run_fold0/ / run_e150/ / run_s2e150/ 零触碰。
- **质量复查（Run 004 同法：3 例 × 30 trial × K=32 seed 42，逐时间点 Q05–Q95 coverage + medoid/real 逐通道 std 比）**：

  | 例 | coverage mean/max（Run 004 → Run 005） | medoid/real std 比 mean（Run 004 → Run 005） | Run 005 std 比 min/max |
  |---|---|---|---|
  | sid 1 | 0.00%/0 → **0.00% / 0.00%** | 0.58 → **0.949** | 0.468 / 2.113 |
  | sid 7 | 0.00%/0 → **0.00% / 0.00%** | 0.53 → **1.034** | 0.358 / 2.231 |
  | sid 21 | 0.00%/0 → **0.00% / 0.00%** | 0.56 → **1.010** | 0.446 / 5.331 |

- viz v2.1（新 ckpt、1250/250，forecast 段 5.0–6.0 s）：`run_ratio5to1/fold0/viz/wave_subj{01,07,21}_trial00.png` + viz_meta.json，PNG 均为 2160×900；AI 判读（待用户人工判）：medoid 主体仍平滑低幅（约 ±0.3–0.5 z vs real ±1.5–2.5 z）、90% PI 带极窄且 real 多数时间在带外；**末端 ~5.85–5.9 s medoid 出现双极大尖峰（Euler 末步发散，E75/E128/E62 最重）**——std 比「≈1」主要由该末端尖峰抬高 medoid 整体 std 所致，属表观改善非真实幅度校准；另观察到 real 本身末端 ~5.8 s 后衰减（数据侧 trial 末段特征，登记备查）；无 DC 偏移。

### 结论

- **证伪「缩短 forecast/加长 obs 可修复生成质量」的核心缺陷**：训练侧健康（两阶段真收敛、stage2 best E122/150 不顶格）、medoid/real std 比表观从 0.53–0.58 升至 0.95–1.03，但 ①逐时间点 Q05–Q95 coverage 三例全部仍为 **0.00%**（90% PI 逐点覆盖塌缩未改善）；②std 比「≈1」经 viz 判读证实主要由末端 Euler 发散尖峰贡献（medoid 主体仍低幅）——与 Run 004 机制假设（重尾/发散样本抬高 std 而分位带仍窄）一致且在此口径下更显著。
- 窗口划分比例（3:1 → 5:1）不是瓶颈；证据链（Run 001/003/004/005）一致指向生成分布逐点方差不足/损失侧校准问题。下一步建议（待用户拍板）：按 Study I 迭代路径另立任务（损失侧/条件分布校准/采样策略），或登记负结果任务收尾。
- 用户拍板（2026-10-21）：继续试 obs=1350/forecast=150（9:1）→ Run 006。

## Run 006 — fold0 9:1 划分（obs 1350 / forecast 150）两阶段重跑（2026-10-21）

### 目的与假设

- forecast 缩至 150 点（0.6 s）、obs 加长至 1350（5.4 s）：检验极短 horizon 是否改善逐点覆盖（Run 004/005 的 coverage 0% 塌缩）；假设 horizon 缩短降低条件生成难度，coverage 显著回升。

### 配置

- 代码：同 Run 005 参数化 wrapper（--obs-len 1350 --forecast-len 150）；CLI：`--fold 0 --stage {1,2} --epochs 150 --obs-len 1350 --forecast-len 150 --run-dir outputs/10-21-latent-flow-mild-train/run_ratio9to1/fold0`
- 参数：epochs 150 / patience 10 / lr 1e-3 / batch 32 / beta_kl 1e-4 / seed 42；stage2 = mexhat-graph（geometry_mild.pt）
- 数据：同 Run 001（manifest 钉死；train 折归一化统计不变；数据文件 1125/375 原状，窗口重切前 1350/后 150）
- 版本保全：run_fold0/、run_e150/、run_s2e150/、run_ratio5to1/ 均不覆盖
- 训后动作：coverage/medoid 复查（**含去末端尖峰口径**：剔除最后 25 点后重算，防 Run 005 型末端发散抬高均值）+ viz v2.1

### 结果

- 训练（fold0，两阶段，obs 1350/fc 150）：stage1 best **2.5975@E28**（早停 E38，257.6 s）；stage2 best **2.0371@E125**（早停 E135，真收敛，692.3 s）；smoke stage2 MAE=0.8900 / std_ratio=0.9880。产物 `run_ratio9to1/fold0/`（双 ckpt+双 config+双 history+日志+viz）。
- 质量复查（3 例 × 30 trial × K=32 seed42；**双口径 = 全程 / 去末端 25 点**）：
  - coverage：subj01/07/21 全程 **0.00%/0.00%/0.00%**，去末端仍 **0.00%**——塌缩非末端伪影。
  - medoid/real std：全程 0.98/0.93/1.16（再次被末端尖峰抬高），去末端 **0.59/0.68/0.74**（min 0.42）——主体仍欠幅。
- viz v2.1：`run_ratio9to1/fold0/viz/wave_subj{01,07,21}_trial00.png`（forecast 段 5.4–6.0 s）。

### 结论

- **证伪「极短 horizon（0.6 s）可修复覆盖塌缩」**：三口径（3:1/5:1/9:1）coverage 一律 0.00%，medoid 主体幅度一律 ~0.5–0.7×，末端 Euler 发散尖峰在 5:1/9:1 两口径复现并系统抬高表观 std 比——**窗口划分维度已充分扫过，瓶颈确证为结构性（生成分布逐点方差不足 + 末端发散），指向损失侧/采样侧迭代**。
- 下一步（待用户拍板）：任务收尾登记负证据链（Run 001–006），Study I 迭代另立任务。






