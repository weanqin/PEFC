# implement — 10-24-exitgnn-ec-unbiased

执行序（前置：1.4 双 mandate 评审通过 + 用户 briefing 同意 + task.py start）。

## S0 — Run 000：v2 GC 参考注册（C1）

- [ ] staging 解压：`unzip EC_GC_1_125.zip -d /root/autodl-tmp/ecgc_1_125_staging/`（取
      EC_GC_subject.npy + README.md）
- [ ] 写 `refs/granger_bivariate_spectral/mild40-trial500-zscore-fs250-aic4-25-f1-125-favg/
      register_refs.py`（镜像 v1 先例 + design §1：四锚点 bitwise 复验 + sanity 断言 +
      meta.json 含 caliber_note + 幂等）
- [ ] 执行并重定向 register.log；核对 stdout 四锚点 maxabsdiff=0.0 + sanity + 量级复核
      （README QC mean≈0.0475 max≈0.568）
- [ ] **refs/MANIFEST.md 追加 v2 行**（「mne-connectivity 谱 GC」小节表 + 内容/边界注记；
      评审 F1）
- [ ] 验证：`ls refs/.../ | wc`（40 npz + 5 附件）；experiment.md Run 000 登记

**Gate G0**：四锚点断言失败 ⇒ 停，向用户报告映射歧义（不得猜序）。

## S1 — ec_mild_unbiased.py 适配（C2，design §2）

- [ ] `--flow-exit-gnn` + `--task-label` CLI（落点规约 design §2.1：main 顶层 task 字段/
      summary·gccorr 任务名段/reading_notes 旗标条件化）；`_model_args(a.flow_exit_gnn)`；
      `_caliber_asserts` 增 ckpt 路径断言；payload 元数据条件化（method/surrogate/
      surrogate_caveat/hparams.flow_exit_gnn 新增键）；逐例 json/npz 原子写（design §2.1b）
- [ ] 静态检查：`python -m py_compile`；`--help` 两旗标出现
- [ ] 验证默认面：**diff 审预期范围**（评审 F3）= hparams 新增 flow_exit_gnn 键 + 原子写
      机制 + label 空串不写字段——除此以外旗标关分支与原逻辑逐行等价；运行时等价证据在
      S2 对拍（不跑旧口径输出树）

## S2 — Run 000b：dry-run + 旗标关等价性对拍（工程干跑存证，非正式口径）

- [ ] 旗标开干跑：`OMP_NUM_THREADS=8 /root/miniconda3/bin/python
      models/latent-flow-v1/ec_mild_unbiased.py --mode main --flow-exit-gnn
      --task-label 10-24-exitgnn-ec-unbiased --ckpt <exit-gnn best> --subjects 1
      --sources 8 --n-grid 4 --out-dir outputs/10-24-exitgnn-ec-unbiased/dryrun_mini`
- [ ] 旗标关等价对拍（评审 Mandate A 建议采纳）：同参（默认 w500 ckpt、seed 42、1 例 ×
      8 源 × 4 网格）跑入 `outputs/10-24-exitgnn-ec-unbiased/scratch_equiv/`；σ*_matrix/
      tau_global/connectivity_matrix 与 w500 在档 `outputs/10-24-mild-w500-ec-unbiased/
      dryrun_mini/subject_0.json` 对拍 **max|Δ|=0 逐位断言**（cudnn.deterministic 已设；
      若非零逐位则登记实测差并停查根因——不触旧树）
- [ ] 核对：strict load 成功（exit-gnn 键匹配）、payload surrogate 块/hparams/caveat 字段、
      norm/manifest SHA 断言通过、json+npz 落盘
- [ ] experiment.md Run 000b 登记（标注非正式口径 8 源 × 4 网格 + 等价对拍结果）

**Gate G2**：strict load、断言链或等价对拍失败 ⇒ 停在此，报告根因。

## S3 — Run 001：GPU 主跑（正式口径）

- [ ] 跑前登记 experiment.md Run 001（目的/配置/预算）
- [ ] `OMP_NUM_THREADS=8 /root/miniconda3/bin/python models/latent-flow-v1/ec_mild_unbiased.py
      --mode main --flow-exit-gnn --task-label 10-24-exitgnn-ec-unbiased
      --ckpt /root/npi/outputs/10-24-latent-flow-exit-gnn/run_exitgnn_fold0/checkpoints/
      fold0_mild_flow_best_graph_mexhat.pt
      --out-dir /root/npi/outputs/10-24-exitgnn-ec-unbiased/ec_main`
      （后台 nohup + log；断点续跑语义：中断后原命令重发）
- [ ] 逐例核对 stdout：8/8、τ/interior/@min/@max、P(resp>τ)=0.500 精确、elapsed
- [ ] 跑后回填 experiment.md Run 001 表（对照 w500 先例 46.9 min / interior 60.3% / dyn 2.62×）

**Gate G3**：单例失败 ⇒ 原命令续跑（json 跳过语义）；系统性失败 ⇒ 停 + 根因分析。

## S4 — Run 002：组级 summary（离线）

- [ ] `--mode summary --main-dir .../ec_main --out-dir .../summary --task-label …`
- [ ] 核对：重映射不变性断言 8/8 PASS、跨被试稳定性、方差分解（行/列/残差 vs w500
      0.037/0.641/0.322）、dyn_ratio（vs w500 2.62×）
- [ ] experiment.md 登记 + 与 w500 并置判读段

## S5 — Run 003：v4 重映射 + 双 GC 对照（C3）

- [ ] 写 `models/latent-flow-v1/recompute_conn_v4.py`（design §3：断言组 1/2/3 + 双 refs
      块汇总）；py_compile
- [ ] 执行 → `outputs/10-24-exitgnn-ec-unbiased/recompute_v4/`（subject_*.json +
      gccorr_dual_summary.json）
- [ ] 核对：三断言 8/8；v1 块 vs w500 Run 007 先例（+0.167±0.166）并置——**标注跨代理脑
      描述性并置**（θ 不同，差异=代理脑差异+噪声，不作复现性判读）；v2 块独立登记
- [ ] experiment.md 登记（含 v1/v2 参考差异事实 median Pearson 0.148 引用）

## S6 — outputs README + 收尾三件套

- [ ] `outputs/10-24-exitgnn-ec-unbiased/README.md`（配置/运行/诊断摘要/复现命令/判读
      要点——镜像 w500 EC README 结构 + 双 GC + caveat）
- [ ] 2.2 check：dispatch trellis-check（jsonl 契约 + AC 全项 + 路径隔离核查：
      `git status` 仅本任务路径）
- [ ] 3.3 spec-update 判定（有可沉淀契约则 trellis-update-spec，否则登记不适用理由）
- [ ] 3.4 commit（仅本任务路径：models/latent-flow-v1/ 两脚本 + refs v2 目录 + 任务目录；
      预存在脏路径 ec-globaltau-v1/* 与两个 in_progress 任务目录不擅动）
- [ ] 研究记忆回写：progress.md（Study II 状态 + exit-gnn EC 线结果）/ context.md 若适用；
      journal 登记会话

## 回滚点

- S0 后：refs v2 目录独立，可整目录删除重注册（幂等）。
- S3 前：代码改动均可 `git checkout -- models/latent-flow-v1/ec_mild_unbiased.py` 单文件回退。
- 任一 Gate 失败即停，产物保留存证，向用户报告。

## 验证命令速查

```bash
# refs v2 sanity 复核
ls /root/npi/refs/granger_bivariate_spectral/mild40-trial500-zscore-fs250-aic4-25-f1-125-favg/ | wc -l  # 45
# 主跑断点续跑核对
ls /root/npi/outputs/10-24-exitgnn-ec-unbiased/ec_main/subject_*.json | wc -l  # 8
# v4 断言组
grep -c PASS /root/npi/outputs/10-24-exitgnn-ec-unbiased/recompute_v4/register.log  # 按实现
# 路径隔离（评审 F4：refs 为 git 跟踪目录，新 v2 档属本任务合法新增路径）
git status --short | grep -v -E "10-24-exitgnn-ec-unbiased|latent-flow-v1|refs/granger_bivariate_spectral|ec-globaltau-v1|tasks/10-24-latent-flow"  # 空
```
