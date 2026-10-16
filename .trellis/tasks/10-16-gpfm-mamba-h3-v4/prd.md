# H3 跨通道交互假设：v4-A 低秩输入瓶颈变体探究（v4-B 方向已放弃存档）

## Goal

在 v2（d_model=62 共享骨干）基础上实现 **v4-A**：in_proj 分解为 `Linear(79→r=16) + Linear(r→62)` 纯线性低秩瓶颈（接口强制跨通道混合、容量不变），fold0 单 seed 正式口径与 v1/v2 对比，检验 H3：v1 的 GC 一致性优势（0.2755 vs v2 0.1227）是否由 d16 瓶颈强制混合承载。产出 5-fold 候选集拍板依据。

## 方向变更登记（用户 2026-10-16 拍板）

- **v4-B（CI 骨干 + cMLP head）方向放弃**：d_ch=1 与 d_ch=8 两档剂量对照（Run 006/007/009/010）结论「容量必要而不充分」——GC 幅值/峰频/逐通道 r 三恢复但 GC r 仅 0.133→0.163（未达部分支持线 0.18），指标 3 未达标（δ–β 0.519>0.5，生成过散 std_t +74%）、CRPS +8.2%。代码与 Run 记录**存档不删**（models/gpfm-mamba-v4/ 与 v4b-* 产物），不再加 seed / 不做 d_ch=16。
- **v4-A 复活**：即本任务最初设计（第一轮 trellis-review 已修正确认），按**当前累计口径**实施（划分 32/4/10、早停 patience=10、lr 恒定 1e-3——均为用户 2026-10-16 会话内拍板，v4-A 继承）。
- v4-A 与 v4-B 的科学关系：v4-A 检验「接口强制混合 × 容量保持」（2×2 缺失格），与 v4-B 的「trunk 零混合 + 显式组件读出」互为对照臂；两者共享新训练口径，与 v1/v2 的口径差异清单一致登记。

## Requirements

- **R1 变体定义（单变量原则，第一轮评审版）**：v4-A 唯一架构变化 = in_proj `Linear(79→62)` 分解为 `Linear(79→r=16) + Linear(16→62)`（**纯线性，两层间无非线性**；PyTorch 默认初始化）；d_model=62 与其余一切与 v2 一致。参数：in_proj 4960→2334，总 ≈129.9k（≈0.98× v2）。
- **R2 实现路径**：v1 `model.py::VelocityNetwork` 增 `bottleneck_rank=None` default-off 参数（None 分支结构与 state_dict 键逐位不变；r 分支键 `in_proj.0/1.*`）+ `train.py`/`sample.py::load_eval_model` getattr 透传（先例 `prior_bandlimit_hz`）；新目录 `models/gpfm-mamba-v4a/`（v2 式单遮蔽 wrapper：仅遮蔽 config；`RUN_PREFIX="v4a-"`；r 扫描经 wrapper `--rank` 覆写被遮蔽 config 实例，config.py 默认唯一）。
- **R3 训练口径（当前累计口径）**：划分 **32/4/10**（7/1/2，`n_val_subjects=4` + `split_manifest_override` 指向 v4a 自备 manifest，内容与 v4b 的 val4 manifest 逐字节一致由测试断言）；**早停 patience=10**；**lr 恒定 1e-3**（无 cosine）；batch 256、seed 1234、epochs 上限 100、EMA 0.999、损失 {1,1,0.1,0.1}、best=val4 总 loss（固定 t 网格 EMA 复算）。
- **R4 评估口径（Run 017 同口径）**：evaluate + fc_metric，七维度与 v1/v2/v4b-d1/v4b-d8 并列；三检验核算（指标 1 ≥0.95 / 指标 3 低频 ≤0.5 / 指标 2 用户目视）；R1 附分子分母；std_t 照登。无 cMLP head ⇒ 无 readout 副产品（v4-A 不适用）。
- **R5 H3 判定框架（判定权在用户）**：主判据 GC r 三态（支持 ≥0.2+ 且五维度不塌缩 / 部分 ≥0.18 或 r 单调 / 不支持 ≈0.12）；不对称守护（不支持 ≠ 混合被证伪，残留解释含损失侧）；0.15–0.22 邻域 → 加 seed 再判；预注册防挪阈。对照材料：v4b 剂量对照（容量必要不充分）随附——若 v4-A GC 回升而 v4-B 未回升，指向「接口混合」优于「顶层显式读出」。
- **R6 不可比注记（vs v1/v2）**：CI 无关但训练口径四项不同（32 例 vs 34 / val4 vs val2 / 早停+恒定 lr vs 100ep+cosine）+ 结构（低秩接口）；test 折相同 ⇒ test 指标可比。
- **R7 纪律**：v1 既有 7 测回归 + 三变体 config_hash 零漂移（v1 `c6ff80ac1c43` / v2 `fc2a9ce14948` / v3lp `0c2de30a4868`，默认 build_config 口径）；Run 跑前登记；零新增依赖；不启动 5-fold、不固定 θ*。
- **R8 v4-B 资产处置**：`models/gpfm-mamba-v4/` 与 `outputs/gpfm-mamba-f0/v4b-*` 原样保留（存档）；其 v1 侧钩子（早停/lr/manifest override）为 default-off，v4-A 继续复用。

## Acceptance Criteria

- [ ] v1 钩子单测：None 分支前向逐位一致 + state_dict 键集不变 + 三变体 hash 零漂移 + v1 七测回归 PASS
- [ ] v4a 单测：r=16 两层结构断言（rank≤16 构造）、前向形状 (B,62,128)、因果性、梯度有限、参数量 ≈129.9k、config↔model 一致、遮蔽与前缀守卫、`--rank` 覆写正反例、val4 manifest 一致性
- [ ] 1-epoch bench 登记（显存/墙钟/损失走势，预期 ≈v2 量级 ~9 s/ep）后主 Run（用户「尝试 v4a」已授权方向；bench 无异常即续跑）
- [ ] 主 Run：早停或 100ep 收，无 NaN，best/last 就位
- [ ] 正式评估：七维度五列对比表 + 三检验 + R1 + std_t 落 experiment.md
- [ ] H3 判读材料（含与 v4b 剂量对照的联合陈述）交用户判定
- [ ] 未启动 5-fold；v4-B 资产未被改动

## 待用户拍板

- 无新增拍板项（r=16 默认、纯线性、当前累计口径继承均已定）；r 扫描 {8,32} 仍为条件扩展（主 Run 判读后议）。

## 子研究路由（1.0b / 1.0c）

<!-- trellis-research-routing -->

- **分类结果**：`study1`
- **依据**：EEG 代理脑（Study I）模型结构变体 fold0 前置验证对比。
- **新路径/方法检测（1.0c）**：`已有路径：Study I gpfm-mamba（context.md 技术路线表第 1 行）`——v4-A 为其下薄变体（v2 基础单变量分解）；3.3 时与 v4-B 一并回填变体注记。
- **research 产物**：`.trellis/research/h3-cross-channel-seed.md`（A/B/C 设计空间；A 方案机制依据）
