# consent — 10-29-modma-ec-te-corr

- **时间**：2026-10-30（会话 62da5e76）
- **同意凭据**：用户下达完整规格指令（两级分析 + 输出内容逐项），用户供全规格执行令即同意（本会话四次同模式先例）。
- **用户指令原文**（摘）：
  > 计算MODMA数据集上EC_pert与TE-EC两种有效连接指标之间的相关性……1. 逐被试分析：对每个fold内的每一位被试，分别计算其EC_pert与TE-EC之间的Pearson相关系数和Spearman相关系数；2. 组水平分析：对每个fold计算组平均后的EC_pert与TE-EC，再计算两者之间的Pearson和Spearman相关系数；3. 输出内容：每个fold的逐被试相关系数、组水平相关系数，以及对应的p值，并以表格形式汇总展示。
- **我方补充并纳入口径**：EC_pert 5 折同口径需补训 fold0（10-29 任务只覆盖 fold1-4；fold0 现存产物为异口径）——补训参数 = 07-05 架构损失 + 50ep/b64/p10/lr1e-3/seed42 + v4 网格 EC（与 fold1-4 逐字段同轨）；TE-EC = refs/symbolic_transfer_entropy/modma5fold 档（10 例/折）；fold1-4 相关取 TE∩EC 10 例交集。GPU ~45min（训练 ~27min + EC ~15min）。
- **briefing**：改动 = train_modma_0705.py 的 --fold choices 放宽含 0（向后兼容）+ 新任务 outputs 树 + 离线相关脚本产物；风险 = 无（纯增量）；验证 = fold0 训练/EC 校验同先例（P(resp>τ)=0.500 等）+ 相关表 p 值齐。
