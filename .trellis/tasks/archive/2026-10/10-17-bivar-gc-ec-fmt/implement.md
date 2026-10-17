# implement.md — bivar-gc-ec-fmt 执行清单

> 前置：prd.md（三轮拍板）+ design.md 定稿。方向约定：EC 矩阵行=源/列=靶/对角=0。
> 纪律：experiment.md Run 块**先写后跑**（四要素：口径/输入/产物/判读）；BIC 网格与主口径一经 refs-01 定版不得中途换；旧 Run 原口径不动。

## B0 装载与环境冒烟 `smoke-00`

- [ ] `models/bivar-gc-ec-v1/` 骨架：config.py（唯一名 `bgec_config` 装载配方）/ data_io.py（复刻 refs_fmt.prepare_subject 逐式 + 字段断言）
- [ ] visualizer 图型盘点（ec_matrix 可用性确认；缺口登记纯增量计划）
- [ ] 单例（s10）加载走通：断言全过、形状 (62,30001)、z-score 复核（均值≈0/σ≈1）
- 验证：`python3 models/bivar-gc-ec-v1/run.py --stage smoke --subjects 1`（控制台断言输出；无落图）

## B1 核心实现 + 单测全绿 `tests/`

- [ ] bivar_gc.py：全通道滞后 Gram（row_from=24、含截距）→ 逐无序对逐阶 slice 求解 → BIC（K=2）→ p\* → 双向 ln(SSR) 读出
- [ ] 单测 1–6（design §5）：方向/独立零基线/样本内非负/Gram-slice 等价（rtol=1e-10）/逐位复现/装载断言
- 验证：`python3 models/bivar-gc-ec-v1/tests/run_tests.py`（全绿；输出逐项断言行）
- 回滚点：单测不全绿不得进入 B2

## B2 主 Run `refs-01`（48 例全量）

- [ ] experiment.md Run 块先写（口径/输入/产物/判读四要素 + 不做 EC_pert 比较/genGC 口径隔离声明）
- [ ] `smoke-01`：2 例先行（条件数登记、统计量量级人工核对、耗时标定）
- [ ] `refs-01`：48 例 16 进程并行 + nohup；产物 `refs_all.npz`（bivar_gc (48,62,62) f32 + subject_ids）+ 逐例 `refs_s{sid}.npz`（矩阵+逐对阶数）+ `refs_summary.json`（caliber/refs_stats/阶数分布）+ `config_snapshot.json`
- [ ] 完整性断言：形状/非有限值/对角=0/offdiag≥0（§2.2 实现正确性断言）
- 验证：`tail outputs/bivar-gc-ec-fmt/refs-01/logs/run.log`；refs_summary.json 阶数分布与统计量就位
- 回滚点：数值异常 → 停、诊断、口径变更最多一次并登记后重跑新 run_id

## B3 split-half `sh-01`

- [ ] split_half.py：段式复刻（250 样本/段、偶A奇B、尾巴丢弃、段内设计不跨段）→ 每半同口径（含逐对 BIC）→ 逐例 offdiag Spearman
- [ ] `sh-01` 产物：逐例 sh_rho + 天花板 mean±sd（ddof=1）入 `refs_summary.json`（或独立 split_half.json，口径注明）
- 验证：天花板数值落 [0,1] 合理区间且与 baseline v4 侧（0.90±0.03）同量级对照登记（方向性对照只登记不判读）

## B4 图与登记回填

- [ ] figs：ec_matrix 逐例抽样 + 组均值；阶数分布图（visualizer 复用或纯增量，B0 盘点结论执行）
- [ ] experiment.md 各 Run 块完整回填（实测值/耗时/偏离）；refs_summary 口径完备性自查
- 验证：图文件落 figs/ 且经 visualizer 生成；experiment.md 无 TBD 残留

## B5 质量检查 + spec 更新 + 提交

- [ ] trellis-check：spec 合规 / lint / 路径扫描（唯一共享写入点=visualizer 纯增量，以 design §1 声明为准）/ 数据流一致性
- [ ] spec 更新（3.3）：evaluation-calibers.md 增「双变量 GC 参考口径」条目（ln SSR/逐对 BIC/125 Hz/与条件 GC 区分）；edra-fmt.md 下游消费行回填
- [ ] research 记忆同步：context.md Study II 参考侧行、progress.md、journal
- [ ] 提交（分块：B1+B2 / B3 / B4+B5 或按实际节奏）
- 验证：`git status` 干净；task.py archive 流程（含 description 非空检查）

## Review gates

- **G1（B1 后）**：单测全绿 + smoke-01 条件数与统计量量级 → 才允许 refs-01 全量
- **G2（B5）**：trellis-check 全过（术语纪律：无 ground-truth 表述）+ spec/记忆同步完成 → 归档
