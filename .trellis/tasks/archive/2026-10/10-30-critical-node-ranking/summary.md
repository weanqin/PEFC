# Summary — 10-30-critical-node-ranking

> Study II 新实验：关键节点排序（用户直给四步设计）。一句话结论：**关键节点排序对噪声方差
> 稳健（逐被试 ρ 0.867–0.980、组平均 ≥0.982、top-K 交叠 4–11× 机会水平），且比例性诊断
> 排除 σ² 缩放的平凡解释（比值 2.61/1.99 ≪ 5.44、log-log 斜率 1.46/1.44 ≪ 2）——属
> 「非线性但一致的剂量响应」下的稳健排序**。

## 交付物

| 类 | 内容 | 位置 |
|---|---|---|
| 代码 | `critical_node_ranking.py`（--line {modma,mild} × --mode {main,summary,figs}，863 行，包体零改动） | `models/latent-flow-v1/` |
| 数据 | 逐被试 json ×93（impact 128×3 + 排序 + 诊断 + 守卫元数据） | `outputs/10-30-critical-node-ranking/{modma,mild}/fold{0..4}/` |
| 分析 | robustness json ×2 + top-20 名单 ×2（组平均双口径） | `outputs/.../robustness/` |
| 图件 | 8 PNG（影响谱/rank-rank/逐被试 ρ 箱线/top-K 曲线） | `outputs/.../figs/` |
| 文档 | **REPORT.md 四段**（目的/方法/结果/稳健性，数字可溯源至 json） | `outputs/10-30-critical-node-ranking/` |
| 规范 | perturb-ec-pipeline.md §10（CRN 配对 + σ² 退化判读 + 双口径聚合 + 堆叠坑） | `.trellis/spec/research/` |

## 关键数字

- **执行契约**：σ=0 恒等 93/93（max|mse|=0）、源失败 0、单调性违例 modma 0 / mild 1 行
  （容差级）、ckpt SHA×10 对拍注册件全过 + 负例拒跑、同 seed 复跑逐位一致。
- **稳健性（pooled）**：modma ρ 0.928/0.867/0.961、mild 0.973/0.929/0.980（三对 σ）；组平均
  primary 0.982–0.997；top-10 交叠 0.58–0.85（机会 0.078）。
- **头部节点跨 σ 锁定**：modma rank1–3 = 107/56/38 三档一致；mild rank1 = 88 三档一致。

## 判读边界（随档 caveat）

CRN 单实现语义（剂量稳定非实现稳定，独立 ξ 重复档留拍板）；剂量外推按线（modma 三档全 ≥
网格顶 0.3，mild 0.5/0.7 超 0.48）；mild θ\* 资格待拍板前提；mild 侧 1 行容差级单调违例。

## 过程事实

- 双 mandate 评审零阻塞，修正全落档（autocast 包裹契约、σ² 退化判读、NaN→nan_to_num 局限
  改写、MODMA_DATA_ROOT、nohup 规约）。
- 执行期 2 次翻车如实登记：①主跑首启 run/logs 目录缺失致重定向失败（mkdir 后重启，零污染）；
  ②summary `impact_distribution` 堆叠轴向 bug（np.stack 后 [:, j] 取到被试矩阵）——主判读
  口径不受影响，修复后与独立复算逐位一致（experiment.md Run 005）。
- trellis-check 有条件过：AC1–AC3/AC5/AC6 过，AC4 两个 Kendall 单元格 + 折标 + 比值修字后
  过（独立复算确认科学结论不受影响）。

## 后续可拍板项

① 独立 ξ 重复档（σ=0.5 × 第二 seed，升级「实现间稳定」证据，~10-15 min GPU）；②关键节点
× 生理/解剖先验（SC/电极位置）或 GC 出度的外部对照（描述性）；③是否注册 result/（本轮按
拍板零写入）。
