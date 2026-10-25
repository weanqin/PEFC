# consent — 10-25-symw-ec-viz

## 1.4 同意记录（implementation consent gate）

- **时间**：2026-10-25（方案提交同日）
- **用户确认原话（同意链）**：
  1. 原始指令（本会话首条消息，完整明确工作令）：「请对研究项目 `10-25-symw-ec` 的有效连接性
     （EC）分析结果进行可视化」+ 三项规格（个体 8×2 并排同 cmap 同色标 / 组均值并排 /
     格式输出含标注、路径说明、图像文件与模式总结）。
  2. 规划前 AskUserQuestion 两问（任务创建 + 色标策略）**未应答**（系统返回"继续按最佳
     判断执行"）。
- **未应答处置（透明在档，镜像 10-24-exitgnn-ec-unbiased consent 先例）**：①原始消息为
  配置完整的明确工作令；②决策均有实测证据（色标：EC [0.03,3.90] vs GC [0,0.29] 量级差
  ~13×，字面同绝对色标 GC 面板仅占 ~7% 不可读 ⇒ 推荐项 = 秩归一同尺度 + 组级绝对值补充
  版，两方案都产出可对比/量级诚实）；③全部改动可逆（新脚本 + 新 outputs 树 + 任务目录，
  旧树只读）；④镜像 10-24-exitgnn-ec-viz 宅例（同类工作全为任务制）。⇒ 创建轻量任务并
  执行；如用户返回后对色标策略有异议，脚本单文件可改、图可整批重出。
- **确认对象**：prd.md（轻量 PRD-only；含色标决策记录与执行序）。
- **流程凭据**：implement.jsonl 4 条 / check.jsonl 3 条；backend = python（技能存档默认，
  宅例延续）；nature-figure 契约三门纳入 AC。
- **关键口径钉死清单（执行不得偏离）**：
  - 数据只读：EC_pert = recompute_v4/subject_*.json `connectivity_matrix`（v4 口径）；
    GC-EC = refs mild40 v1_1_45_fixed8 `refs_s{sid}.npz` 的 `ec`；方向行=源列=靶；
  - 8 被试 s1/5/7/10（MDD）s21/25/27/30（HC）；
  - 主图秩归一同尺度（offdiag 百分位秩、对角掩蔽、同 cmap 同 [0,1]）+ 组级 abs 补充版；
  - 通道标签 E1–E128 稀疏刻度（每 16）+ 128×128 维度注记；ρ 注记取 gccorr v1 块；
  - 三门 QA（validate/alignment/collision + visual-judge）不过不出档；
  - GPU 禁占（并行训练会话在跑）；路径隔离照 R7。
