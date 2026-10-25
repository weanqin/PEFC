# implement — 10-25-symw-ec-unbiased

执行序（前置：1.4.2 briefing + 用户明确同意 + task.py start）。轻量任务：design.md 不适用
（复用两先例设计，适配点见 prd R1）。

## S0 — Run 000：脚本适配 + 静态检查

- [ ] `ec_mild_unbiased.py` 加 `--symw` 旗标（prd R1 四点：架构侧等价 flow_exit_gnn=True /
      守卫改查 symw run_dir + 互斥断言 / payload symw 分支含 caveat / hparams `symw` 键）；
      默认 False 时旧分支逐行不动
- [ ] `recompute_conn_v4.py` 加 `--task-label`（默认 = 现常量串，旧调用面不变）
- [ ] `python -m py_compile` 两脚本 + `--help` 新旗标出现
- [ ] diff 自审：计算路径（probe/bisection/τ/conn/v4 公式）零改动

## S1 — Run 000b：dry-run 双证据（工程干跑存证，非正式口径 8 源 × 4 网格）

- [ ] 旗标关等价对拍：默认 w500 ckpt、seed 42、1 例 × 8 源 × 4 网格 →
      `outputs/10-25-symw-ec-unbiased/scratch_equiv/`；σ*_matrix / tau_global /
      connectivity_matrix 与在档 `outputs/10-24-mild-w500-ec-unbiased/dryrun_mini/
      subject_0.json` 对拍 **max|Δ|=0 逐位断言**（非零则登记实测差 + 停查根因，不触旧树）
- [ ] `--symw` 干跑：`--ckpt outputs/10-25-latent-flow-exit-gnn-symw/run_symw_fold0/
      checkpoints/fold0_mild_flow_best_graph_mexhat.pt` → `dryrun_symw/`；核对 strict load
      成功、守卫生效、payload surrogate 块 / hparams.symw / caveat 字段、norm/manifest SHA
      断言通过
- [ ] experiment.md Run 000 / 000b 登记

**Gate G1**：strict load、断言链或对拍失败 ⇒ 停，报告根因。

## S2 — Run 001：GPU 主跑（正式口径）

- [ ] 跑前登记 experiment.md Run 001（目的/配置/预算）；查 GPU 空闲
- [ ] `OMP_NUM_THREADS=8 /root/miniconda3/bin/python models/latent-flow-v1/ec_mild_unbiased.py
      --mode main --symw --task-label 10-25-symw-ec-unbiased
      --ckpt /root/npi/outputs/10-25-latent-flow-exit-gnn-symw/run_symw_fold0/checkpoints/
      fold0_mild_flow_best_graph_mexhat.pt
      --out-dir /root/npi/outputs/10-25-symw-ec-unbiased/ec_main`
      （后台 nohup + log；断点续跑语义：中断后原命令重发）
- [ ] 逐例核对 stdout：8/8、τ/interior/@min/@max、P(resp>τ)=0.500 精确、elapsed
- [ ] 跑后回填 experiment.md Run 001（对照 w500 46.9 min / exit-gnn 同量级；w500 interior
      60.3% / dyn 2.62× 三方并置）

**Gate G2**：单例失败 ⇒ 原命令续跑（json 跳过语义）；系统性失败 ⇒ 停 + 根因分析。

## S3 — Run 002：组级 summary（离线）

- [ ] `--mode summary --main-dir .../ec_main --out-dir .../summary --task-label …`
- [ ] 核对：重映射不变性断言 8/8、跨被试稳定性、方差分解、dyn_ratio（三方并置）
- [ ] experiment.md 登记

## S4 — Run 003：v4 重映射 + 双 GC 对照 + 对称性诊断

- [ ] 写对称性诊断小脚本（prd R7：逐被试 EC v4 vs EC.T，offdiag Pearson/Spearman +
      ‖M−Mᵀ‖_F/‖M‖_F；输入 ec_main json，输出 `outputs/10-25-symw-ec-unbiased/
      symmetry_diag/`）；py_compile
- [ ] `recompute_conn_v4.py --main-dir .../ec_main --out-dir .../recompute_v4
      --task-label 10-25-symw-ec-unbiased`
- [ ] 核对：三断言 8/8；v1/v2 双块 per-subject + group + group_mean（aligned/flipped）；
      与 w500（v1 −0.099/+0.118）、exit-gnn（v1 +0.149±0.168 / v2 −0.010±0.085）并置
      ——标注跨代理脑描述性并置
- [ ] experiment.md 登记（含对称性诊断结果与方向性判读段）

## S5 — outputs README + 收尾三件套

- [ ] `outputs/10-25-symw-ec-unbiased/README.md`（配置/运行/诊断摘要/复现命令/判读
      要点——镜像先例结构 + A_eff 无向 caveat + 模型版本注明）
- [ ] 2.2 check：dispatch trellis-check（jsonl 契约 + AC 全项 + 路径隔离核查）
- [ ] 3.3 spec-update 判定（有可沉淀契约则 trellis-update-spec，否则登记不适用理由）
- [ ] 3.4 commit（仅本任务路径：models/latent-flow-v1/ 三脚本 + 任务目录；预存在脏路径
      models/ec-globaltau-v1/* 与两个 in_progress 任务目录不擅动）
- [ ] 研究记忆回写：progress.md（Study II 状态 + symw EC 线结果）/ context.md 如适用；
      journal 登记会话

## 回滚点

- S1 后：代码改动均可 `git checkout -- models/latent-flow-v1/` 对应文件回退。
- S2 前：outputs 任务树独立，可整目录删除重跑（json 跳过语义天然幂等）。
- 任一 Gate 失败即停，产物保留存证，向用户报告。

## 验证命令速查

```bash
# 主跑断点续跑核对
ls /root/npi/outputs/10-25-symw-ec-unbiased/ec_main/subject_*.json | wc -l  # 8
# v4 断言组（按实现输出核对 PASS 计数）
ls /root/npi/outputs/10-25-symw-ec-unbiased/recompute_v4/ | head
# 路径隔离（refs 为既有目录，本任务零新增 refs 路径）
git status --short | grep -v -E "10-25-symw-ec-unbiased|latent-flow-v1|ec-globaltau-v1|tasks/10-2[45]-"  # 空
```
