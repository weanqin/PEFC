# implement — gpfm-mamba-v5-fmt（fmt 数据训练 v5-fmt-01）

> 2026-10-20 立项（用户拍板：暂停 10-19 任务、新立项本任务）。上游计划 = 10-19 任务 implement.md S7'；下游 = ni-mse-design.md v1.5 主实例化（θ\* 消费）。
> 本清单为立项时种子；动工前经 design/派发流程细化。

## S1 结构落地（fmt 口径变体）

- [x] S1.0 git 快照基线登记
- [x] S1.1 `models/gpfm-mamba-v5-fmt/`：config（v5 副本 + fmt 口径：data_root=edra_processed_fmt、fs=250、LayerNorm/p=25/无低通——逐字段对照 v1-fmt）+ model/graph 自 v5 **只读复制**（零结构改动）+ train/evaluate 双遮蔽包装（前缀 `v5-`）
- [x] S1.2 划分固化接 v1-fmt manifest（34/4/10 只读）
- [x] S1.3 单测：v5 十项 fmt 实例 + fmt 差异项 + 回归零漂移
- 验证：pytest 全绿；回归 v1/v1-fmt/v5 全绿 + hash pin 复验

## S2 训练链

- [x] S2.1 smoke（`v5-fmt-smoke-a`）→ 登记
- [x] S2.2 bench 1-epoch（`v5-fmt-bench-1ep`）→ 材料登记（显存/it/s/W 范数点值 ckpt 提取）
- [x] S2.3 主 Run **`v5-fmt-01`**（启动确认沿 v5-main 先例——执行期用户确认项；nohup + 显式重定向 + 等待器）
- [x] S2.4 A_eff/W 快读出（EMA shadow，对齐 v5-main Run 006 协议；与 v5-main-01 对照登记）

## S3 质检与收尾

- [x] trellis-check（全范围 + 零改动复核）；spec 回填；提交；归档
