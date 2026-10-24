# implement — latent-flow × mild-w500 无偏噪声扰动归因 EC（global-τ 两阶段）

前置：1.4 门（方案审阅 → 用户拍板修订（q=0.5 钉死 / 8×30 直接使用 / GC refs 注册 +
对照）→ briefing → consent.md → task.py start）通过后动工。
执行中每步跑前/跑后按 experiment.md Run 块登记惯例回填。

## S0 GC 参考注册（R6）

- [ ] staging 解包 `/root/autodl-tmp/mild-depression-gc.zip`（已完成勘察：README/
      AIC csv/finalize 日志齐，(40,128,128) float64）。
- [ ] 注册脚本 `register_refs.py`（落 refs 子目录内，可重放）：读 staging npy →
      sid=index+1 赋名 → 写 40 × `refs_s{sid}.npz`（ec float32 + subject_id + group +
      params 串）+ SHA256SUMS + meta.json（zip SHA256/方法参数/映射验证四锚点记录）+
      README-source.md 拷贝 + register.log。
- [ ] 注册校验：全 40 npz 载入断言 shape/dtype/对角 0/无负值/sid↔group 与 manifest
      一致；refs/MANIFEST.md 追加「mne-connectivity 谱 GC」小节条目。

## S1 脚本落地（唯一新代码文件）

- [ ] `models/latent-flow-v1/ec_mild_unbiased.py`：三模式 `--mode main|summary|gccorr`，
      design §1–§7、§10 契约全部落地（口径断言链 / SignalProbeSite 零适配复用 /
      两阶段流程 q=0.5 / payload schema + 凭据 / 曲线 npz 顺带落盘 / BLAS 守卫 /
      out-dir 必填）。
- [ ] 干跑验证：`--help`；缺 `--ckpt`/`--out-dir`、obs+fc≠window、subjects 越界
      fold test → 拒跑（断言生效）。

## S2 mini 干跑（工程校验，非 q 测试）

- [ ] `--mode main --sources 8 --n-grid 4 --subjects 1`（~1 min）→
      `outputs/10-24-mild-w500-ec-unbiased/dryrun_mini/`；核对 json/npz 形状
      （8×8 σ*/conn、8×4×128 曲线）、无 NaN、conn 值域、凭据字段。保留作开发存证。

## S3 Run 001 主跑

- [ ] 跑前登记 experiment.md（目的/全配置：q=0.5、σ_ref=0.3、log[0.01,0.48]×24、
      8 例 × 30 trial、seed 42、bf16）。
- [ ] 执行 `--mode main`（fold0 test 8 例全量）。
- [ ] 逐例核对：subject_{0..7}.json + _curves.npz 齐全、n_source_failures=0、
      diagnostics 完整；回填 experiment.md 逐例摘要表（τ/interior/@min/@max）。

## S4 Run 002 离线汇总

- [ ] `--mode summary`：组级 σ*/conn 分布、行/列/残差方差分解、跨被试 offdiag
      Spearman 稳定性、1/(σ*+1e-4) 重映射对照（Spearman 不变断言）、曲线后置健康
      诊断（dyn_ratio/nonmono/钳边 @q=0.5 照登）。
- [ ] 回填 experiment.md 结论块。

## S5 Run 003 GC 相关对照

- [ ] `--mode gccorr`：refs_s{sid} × subject_{i} 对齐断言；|EC| 与 raw（conn 与
      重映射两式）× aligned/flipped × Spearman/Pearson，offdiag 16256 对，逐例 +
      组汇总 + 组均值矩阵（协议逐字对齐 gen_unbiased_gc_corr.py）。
- [ ] 回填 experiment.md（描述性登记，不构成资格判定）。

## S6 收尾三件套

- [ ] 输出目录 README.md（配置/诊断表/复现命令，unbiased 先例同款）。
- [ ] 2.2 check：spec 合规（reproducibility/experiment-log/path-isolation）+ lint +
      干跑复验 + 路径隔离核查（git status 只圈本任务文件：refs 新子目录 + 1 个变体层
      新文件 + 任务目录）。
- [ ] 3.3 spec update 判断：无偏 EC 口径 / mild GC refs 档是否登记 spec（有沉淀则
      trellis-update-spec，无则登记「无」）。
- [ ] 3.4 commit：只圈本任务文件；**不圈** 10-24 任务脏路径与其未提交脚本
      （train_mild_depress.py / viz_mild.py / quality_check_mild.py /
      models/ec-globaltau-v1/* 未跟踪件）。**共享记忆文件处置（评审 B）**：
      .trellis/research/{context,progress}.md 与两份 spec 已含 10-24 训练任务未提交
      编辑，本任务的记忆回填写入后**不随本任务提交**（防 10-24 待提交内容被裹挟），
      随 10-24 任务收尾一并入库；此处置在 journal/提交信息中声明。

## 验证命令速查

```bash
# mini 干跑（工程校验）
OMP_NUM_THREADS=8 /root/miniconda3/bin/python models/latent-flow-v1/ec_mild_unbiased.py \
  --mode main --sources 8 --n-grid 4 --subjects 1 \
  --ckpt outputs/10-24-latent-flow-mild-w500/run_w500_fold0/checkpoints/fold0_mild_flow_best_graph_mexhat.pt \
  --out-dir outputs/10-24-mild-w500-ec-unbiased/dryrun_mini
# Run 001/002/003：--mode main|summary|gccorr（experiment.md 登记完整命令）
```

## 回滚点

- 每 Run 独立输出子目录互不覆盖；refs 注册只新增子目录（旧档零改动，可整目录删除回滚）。
- 唯一代码改动 = 单文件新增，整体可删回滚；包体与既有变体层文件零改动。
