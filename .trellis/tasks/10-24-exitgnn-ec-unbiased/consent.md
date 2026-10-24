# consent — 10-24-exitgnn-ec-unbiased

## 1.4 同意记录（implementation consent gate）

- **时间**：2026-10-24（方案提交同日）
- **用户确认原话（同意链，按时间序）**：
  1. 原始指令（本会话首条消息，明确执行令）：「请使用10-24-latent-flow-exit-gnn模型作为
     代理脑（surrogate brain），并按照以下配置执行实验：1. EC（有效连接）计算方法：采用
     v4公式版本，公式为：conn = -(ln σ* + 1/2 ln 2πe) + 5/7；2. 噪声设置：不使用偏置噪声
     （bias noise）；3. σ参数修改：将σ的取值范围从默认值修改为 [0.01, 0.48]；σ_ref = 0.3；
     q = 0.5。除上述明确指定的修改外，其余参数均保持默认设置。」
     （注：σ 范围 [0.01,0.48]/σ_ref=0.3/q=0.5 与脚本现行默认值一致——用户表述为「修改」
     但数值上即默认口径，prd 已按「逐项一致」登记。）
  2. 第一轮 AskUserQuestion 明确拍板：①「同意，创建任务（推荐）」——问题文本已明示
     「规划产物齐备、评审 briefing 后会再向你拿一次执行同意才会动代码/GPU」；②交付范围
     「全套默认流程+双GC对照，新加的GC对照在autodl-tmp/EC_GC_1_125」。
- **确认对象**：1.4.2 实施简报（改动点/关键逻辑/潜在风险/路径与环境/验证方案/审查发现
  处理——已在本会话向用户完整呈现）+ 三份规划产物（prd.md / design.md / implement.md，
  双 mandate 评审 A 通过、B 有条件通过且 4 项条件 + A 侧 2 minor + 3 建议已全部回填）。
- **二次确认处置（透明在档）**：briefing 后按 1.4.3 发起二次同意问询，用户未应答（系统
  返回"继续按最佳判断执行"）。判断依据：①原始消息为配置完整的明确执行令；②计划与指令
  五要素逐项一致、范围即用户第一轮拍板（全套+双 GC）；③全部改动可逆（旗标默认关、
  新增目录/脚本、旧树零触碰）；④GPU 独占窗口约 50 min 且当前空闲。⇒ 继续执行；
  如用户返回后对简报有异议，所有改动可单提交回退、refs/outputs 目录可整删重建。
- **流程凭据**：双 mandate trellis-review 已派发——Mandate A（方案与方法论）：**通过**
  （2 minor + 3 建议全采纳：锚点勘误/task-label 钉死/GPU 预算参照注记/旗标关等价性对拍
  升级为运行时证据/跨代理脑并置注记）；Mandate B（实施与风险）：**有条件通过**，4 项
  条件全改（F1 refs/MANIFEST.md 追加遗漏-major、F2 锚点 h_40→h_20、F3 payload 等价口径
  澄清、F4 路径隔离 grep 假阳性）+ F5/F6/F7/F8/F9 改进项全采纳（reading_notes 条件化/
  Run 000b 编号/task-label 落点规约/python 绝对路径/原子写）。
- **关键口径钉死清单（执行不得偏离）**：代理脑 = 10-24-latent-flow-exit-gnn run_exitgnn_
  fold0 stage2 best（0.8241@E116）；无偏 μ=0 σ·ξ（use_biased_noise=False）；q=0.5、
  σ_ref=0.3、log[0.01,0.48]×24、8 例 × 30 trial 全量、seed 42、bf16；conn 交付 v4=
  −(ln σ*+0.5·ln2πe)+5/7（主跑存 σ*+v1，离线重映射+三断言）；双 GC = v1 mild40（1–45 Hz
  固定阶 8）+ v2 新注册（1–125 Hz AIC 4–25）；offdiag 16256、aligned/flipped ×
  Spearman/Pearson、描述性登记不构成代理脑资格判定；EC 结果随代理脑标 provisional。
