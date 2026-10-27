# implement.md — 执行清单（有序）

前置：环境已核对（environment.md：本机 torch 2.8.0+cu128 / RTX 4080 SUPER / w500 ckpt
在档）；`nvidia-smi` 确认 GPU 空闲或占用方可；config-fix 并发改动监控（每步动
`ec_mild_unbiased.py` 前先 `git diff --stat` 复核外来增量）。

## S0 基线（改文件之前，必做）

- [ ] `git diff models/latent-flow-v1/ec_mild_unbiased.py > research/pre_change_config_fix.diff`
      + `sha256sum models/latent-flow-v1/ec_mild_unbiased.py`（外来改动复核锚）
- [ ] iid 默认基线干跑（默认旗标，改动前代码）：
      `OMP_NUM_THREADS=8 /root/miniconda3/bin/python models/latent-flow-v1/ec_mild_unbiased.py
       --mode main --subjects 1 --sources 8
       --out-dir /root/npi/outputs/10-27-ec-histblock-noise/baseline_iid_pre`
- [ ] 基线锚点提取：subject json 的 `sigma_matrix`/`raw_response_matrix`/`tau_global`
      + `curves.npz(mse_curve)` 留档 hash（AC1 对拍基准）

## S1 代码实施（ec_mild_unbiased.py，~+90 行）

- [ ] 模块常量：`_HIST_XI_SEED_OFFSET = 9375`；`build_hist_block_xi()`（design §3 逐字）
- [ ] `HistBlockProbeSite(SignalProbeSite)`（design §4 逐字）
- [ ] `parse_args` 5 旗标（design §2）；`_caliber_asserts` hist-block 两断言
- [ ] `run_main`：循环外 `XI_HIST_GLOBAL` 构造 + site 构造二选一 + 头部打印 noise-mode 段
- [ ] payload：hist-block 分支 hparams 7 键 + `noise_model` 分支串；iid 分支零触碰
- [ ] `python -m py_compile` 通过；全文件 diff 自查（原路径行未动）

## S2 验证（2.2，五项 = AC1–AC5）

- [ ] `research/test_hist_block_noise.py`（CPU 构造类检查，无模型依赖）：
      V1 provider 跨源同引用/逐位；V2 两次独立重建 `torch.equal` + 退化键控路径可用；
      V3 `ξ == ξ[:, ::25].repeat_interleave(25,1)`；V4 `G==15` + `ξ[:,::25]==base` +
      N=200k 相关 sanity（seeded，|offdiag|<0.01）；口径断言（hist_len 整除等）负例
- [ ] iid 等价对拍（AC1）：同 S0 命令 → `out-dir=.../post_iid_default`；脚本逐位比较
      `sigma_matrix`/`raw_response_matrix`/`mse_curve`/`tau_global`（conn 因 v4 平移不作锚）
- [ ] hist-block GPU 集成冒烟（AC2/3/5 集成面）：`--noise-mode hist-block --subjects 2
      --sources 8 --out-dir=.../smoke_histblock`；运行中断言双被试 `site._xi` 一致 +
      `set_source(0/1/127)` ξ 不变 + 同源跨 σ digest 不变（临时探针脚本或脚本内断言）
- [ ] 结果登记 `experiment.md`（Run 块四要素）

## S3 收尾（3.3 / 3.4）

- [ ] spec 沉淀评估：hist-block 噪声构造契约（种子偏移正交、四前提复现注意事项）是否入
      `.trellis/spec/research/perturb-ec-pipeline.md`（评估后定，无沉淀则记因）
- [ ] 提交计划起草并向用户一次征求确认（3.4 协议）：本任务 hunks（ec_mild_unbiased.py
      hist-block 段 + 任务档 + 验证脚本）与 config-fix 未提交 hunks（v4 公式段）分离
      策略——等其先提交 / 本提交捎带注明 / 部分暂存，用户拍板后执行
- [ ] `progress.md` 速览同步 + journal 记会话

## 回滚点

- S1 后任一步失败：`git checkout -- models/latent-flow-v1/ec_mild_unbiased.py`（单文件净
  可逆；注意会连带丢弃 config-fix 未提交 hunks——回滚前先征用户确认，或改用手工逆 diff）。
- outputs/10-27-ec-histblock-noise/ 为全部新产物落点（gitignore 内，无清理负担）。
