# Implement — 关键节点排序实验

> 顺序执行；每步验证命令过才进下一步。产物根 = `outputs/10-30-critical-node-ranking/`（下称 $OUT）。
> 代码 = 单新增文件 `models/latent-flow-v1/critical_node_ranking.py`（--line {modma,mild} ×
> --mode {main,summary,figs}）。

## S0 规划收尾（1.3 / 1.4）

- [ ] `implement.jsonl` / `check.jsonl` 填 spec/research 清单。
- [ ] briefing（改动点/关键逻辑/风险）→ 用户明确同意 → 写 `consent.md` → `task.py start`。

## S1 脚本实现

- [ ] 读源核对四处契约后落码：`bisect_sigma.SignalProbeSite`（set_source ξ 生命周期 /
      perturb_and_measure 返回对 / **每次调用须包 bf16 autocast 域，含 σ=0**——评审 B-1，AC6
      前提）、`ec_mild_unbiased.run_main`（mild 装载链 / 站点构造 / autocast 先例 L424-439）、
      `ec_modma_globaltau_unbiased.run_main`（modma 装载链 / autocast 先例 L210-235）、
      `latent_flow/config.py`（MODMA_DATA_ROOT 依赖）。
- [ ] 实现：BLAS 守卫入口、双线装载 + §2.4 守卫常量表（`--ckpt` 覆盖旗标供负例）、main 循环
      （σ=0 断言 + 逐源三档，每源单 autocast 域）、summary（两级三对 Spearman/Kendall + top-K
      + 比例性诊断 ratio/log-log 斜率 + 双组级口径）、figs（≥3 张）。
- [ ] `python3 -m py_compile models/latent-flow-v1/critical_node_ranking.py`。
- 验证：编译过 + `--help` 双 line 可见；`git status` 无 tracked 文件改动、untracked 新增仅
  `critical_node_ranking.py`（variant 目录另有 10-30-symw-ablation-mw 遗留 5 个 untracked
  脚本，非本任务产出，不擅动）。

## S2 冒烟（GPU）

- [ ] 守卫负例：`--line modma --fold 0 --ckpt <fold1 注册件路径>` 预期拒跑（登记报错行）。
- [ ] modma 冒烟：`MODMA_DATA_ROOT=/root/autodl-tmp/modma` 前缀，fold0 × 1 被试全 128 源
      （`--subjects 1`）→ 检查 json 字段齐 / σ=0 断言过 / 单调性诊断合理。
- [ ] mild 冒烟：fold0 × 1 被试 → 同上。
- [ ] 复跑对拍：modma 冒烟例删 json 重跑一次，impact 矩阵逐位比对（GPU 独占口径；漂移则登记
      并对该例做 CPU 单线程复算对拍）。
- 验证：三冒烟 json 落 `$OUT/{modma,mild}/fold0/`；耗时段（subj elapsed_min）外推五折预算 ≤1.5 h。

## S3 modma 主跑（五折全量 53 例）

- [ ] Run 块先写 experiment.md（目的+配置）再启动；逐折
      `MODMA_DATA_ROOT=/root/autodl-tmp/modma nohup … --mode main --line modma --fold {k}` 后台化，
      日志 `$OUT/run/logs/modma_fold{k}.log`。
- [ ] 串行五折（GPU 独占）；每折完成即抽查 json 计数（10/10/11/11/11）。
- 验证：`ls $OUT/modma/fold{k}/subject_*.json | wc -l` 与折表一致；无 FAILED 被试（或有则登记，
      重试须 --overwrite）。

## S4 mild 主跑（五折全量 40 例）

- [ ] 同 S3，`--line mild`（无特殊 env），每折 8 例，nohup + 日志同规。
- 验证：同 S3；norm SHA 守卫五折全过（日志行）。

## S5 稳健性分析 + 图件

- [ ] `--mode summary` 双线 → `$OUT/robustness/{modma,mild}_robustness.json` +
      `{line}_top_nodes.md`（组平均 top-20 三 σ 并置，主口径原始均值 + 副口径秩均值）。
- [ ] 比例性诊断纳入 summary：逐被试 ratio I(0.7)/I(0.3) 分布（对照 5.44）+ 组平均 log-log
      通道级斜率分布（线性响应 ≈2）。
- [ ] `--mode figs` 双线 → `$OUT/figs/`：①组平均影响谱（128 通道三 σ 并置，按 σ=0.5 排序，
      图注注明绝对量纲随 σ 增长）②rank-rank 散点三对（组平均+逐被试边际）③逐被试 ρ 分布
      （折分组箱线）④top-K 交叠曲线（附机会水平 K/128 参照）。
- 验证：AC3 指标齐（三对 × 两级 × ρ/τ-b/top-K + 比例性诊断）；图件可打开、坐标轴/图例完整。

## S6 REPORT.md + 收尾

- [ ] `$OUT/REPORT.md` 四段：实验目的（关键节点概念与 Study II 定位）/ 方法描述（四步 + 契约表）/
      结果展示（双线五折影响分布 + top 节点名单）/ 稳健性分析（跨 σ 一致性判读 + 比例性诊断 +
      caveat：σ² 缩放退化、CRN 单实现语义、按线剂量外推、mild θ\* 资格待拍板前提、nan_to_num
      局限）；数字全部从 json 抄录可溯源。
- [ ] experiment.md 各 Run 块补结果+结论（无空块）；`progress.md` 当前状态同步；
      result/ 零写入登记。
- [ ] 2.2 check（trellis-check 全范围）→ 3.3 spec 更新评估（有沉淀则写）→ 3.4 commit。

## 回滚点

- S2 前：删脚本即净。
- S3–S5：删 `$OUT` 子目录即净（断点续跑使中断无害）。
- S6 commit 前发现口径缺陷：回 Plan 修 prd/design，产物按「已定版不动」原则新版本并排。
