# summary — 10-21-latent-flow-mild-train

> 任务级汇总；逐 Run 细节见 experiment.md（Run 001–006），决策链见 prd.md 决策记录。

## 交付物

- **代码（变体层新增，包体 latent_flow/ 零改动）**：`make_mild_manifest.py`（5-fold 钉死划分）、`build_mild_geometry.py`（GSN-128 geometry，D4 E 序假设签认）、`train_mild_depress.py`（两阶段训练入口，obs/fc 划分参数化）、`viz_mild.py`（v2.1 波形检验 visualizer）、`splits/mild_5fold_seed42.json`（SHA256 4798790…d881b）
- **数据契约**：`.trellis/spec/data/mild-depression.md`（新）；environment.md 数据落点回填
- **数据根豁免单文件（D1）**：`/root/autodl-tmp/Mild Depression/geometry_mild.pt`
- **训练产物（全部版本保全、互不覆盖）**：
  | 目录 | 口径 | 状态 |
  |---|---|---|
  | `run_fold0/` | 3:1（1125/375）50 epoch | 完整（Run 001，含 v2/v2.1 viz） |
  | `run_e150/` | 3:1 150 epoch（5 折全量） | **用户中止**（fold0 stage1 ~E34，无完整产物） |
  | `run_s2e150/fold0/` | 3:1 stage2@150 | 完整但早停 E17、劣于 Run 001 |
  | `run_ratio5to1/fold0/` | 5:1（1250/250）150 epoch | 完整（Run 005，含 viz） |
  | `run_ratio9to1/fold0/` | 9:1（1350/150）150 epoch | 完整（Run 006，含 viz） |

## 训练结果一览（fold0）

| Run | 口径 | stage1 best | stage2 best | smoke MAE/std_ratio |
|---|---|---|---|---|
| 001 | 3:1 @50ep | 2.4368@E40（早停E50） | 2.3315@E50（**顶格未收敛**） | 0.8591 / 1.0135 |
| 003 | 3:1 stage2@150 | （复用001） | 2.5061@E7（早停E17，**劣化**） | 0.8363 / 0.9721 |
| 005 | 5:1 @150ep | 2.5230@E34 | 2.0081@E122（真收敛） | 0.8703 / 1.0894 |
| 006 | 9:1 @150ep | 2.5975@E28 | 2.0371@E125（真收敛） | 0.8900 / 0.9880 |

（不同口径 val loss 不可直接互比——forecast 长度不同。）

## 核心结论（负证据链）

1. **窗口划分维度充分扫描（3:1/5:1/9:1）× coverage 全 0.00%**：逐时间点 Q05–Q95 覆盖塌缩在全部口径、全部 90 trial × 128 通道无一例外（9:1 已核去末端口径仍 0%）。
2. **medoid 主体欠幅**：medoid/real std 主体 ≈ 0.5–0.74（3:1 全程 0.53–0.58、9:1 去末端 0.59–0.74；5:1 未单独算去末端口径，viz 判读主体同型欠幅）；表观 std≈1 由末端 Euler 发散尖峰抬高（5:1/9:1 复现）。
3. **训练侧非瓶颈**：两阶段各口径均稳定收敛、无发散；stage2 在 5:1/9:1 真收敛（E122/E125）；patience=10 在 3:1 高方差轨迹上会截杀（Run 003），bf16 AMP 下训练非逐位确定。
4. **用户波形判读「不通过」获数值硬证据支持**（Run 004/005/006）；z-score 口径三问复核无失配；viz 口径已按用户要求改 v2.1（仅 real-future+gen-future）。
5. **定性**：Mild Depression 线 latent-flow-v1 代理脑条件生成质量存在**结构性缺陷**（生成分布逐点方差不足 + 末端发散），**未取得可用 θ 候选**；迭代方向 = Study I 损失侧/条件分布校准/采样侧，另立任务。

## 边界与移交

- 本任务不做代理脑合格性三项检验（未变）；产物不继承资格。
- 移交后续：①Study I 迭代任务（损失侧跨通道/方差校准——本任务负证据链为其直接判据）；②若复用本线代码：wrapper 已参数化 obs/fc、visualizer 可直接出 coverage/medoid 数字；③被中止的 run_e150/ 目录保留存档，如清理属后续拍板。
