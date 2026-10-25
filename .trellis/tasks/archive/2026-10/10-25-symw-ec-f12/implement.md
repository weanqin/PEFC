# implement — 10-25-symw-ec-f12

执行序（前置：consent + task.py start）。

## S0 — Run 000：脚本适配 + 静态检查

- [ ] `ec_mild_unbiased.py`：①EXPECTED_NORM_SHA → {0: "d184e647…", 1: "0f10b153…",
      2: "a2c27a2d…"} fold 键控（值 = f12 configs/stage{1,2}_config.json norm_stats.
      mean_std_sha256 训练侧登记，两 stage 一致）；②`--symw` 守卫扩展：接受
      `10-25-latent-flow-symw`（fold0）或 `10-25-latent-flow-symw-f12` run_dir，且 f12
      ckpt 文件名 `fold{k}_` 前缀须与 `--fold` 一致（负例拦截）；③payload surrogate f12
      分支（task/ckpt_stage2_best=各折 best val@epoch + 真早停注记/variant 含 f12 双折
      训练来源/caveat 含 θ 待拍板 + 跨折描述性）；④hparams 增 `fold` 键显式化
- [ ] py_compile + `--help`；diff 自审：计算路径逐行零改动

## S1 — Run 000b：干跑双证据（Gate）

- [ ] 旗标关 fold0 等价对拍：默认参 mini（subject_idx 0、8 源、4 网格、seed 42）→
      `outputs/10-25-symw-ec-f12/scratch_equiv/`；与 `outputs/10-24-mild-w500-ec-unbiased/
      dryrun_mini/subject_0.json` 六矩阵+τ 逐位 max|Δ|=0
- [ ] `--fold 1 --symw` 干跑（fold1 ckpt）→ `dryrun_f1/`：strict load + fold1 norm SHA
      断言通过 + payload fold/symw 元数据 + P=0.500；守卫负例：fold0 ckpt 配 `--fold 1`
      应被前缀断言拦（登记）
- [ ] experiment.md Run 000/000b 登记

**Gate**：任一失败 ⇒ 停 + 根因报告。

## S2 — Run 001/002：两折主跑（串行，各 ≈52 min）

- [ ] 跑前登记；fold1：
      `OMP_NUM_THREADS=8 nohup /root/miniconda3/bin/python models/latent-flow-v1/ec_mild_unbiased.py
      --mode main --fold 1 --symw --task-label 10-25-symw-ec-f12-fold1
      --ckpt /root/npi/outputs/10-25-latent-flow-symw-f12/run_symw_fold1/checkpoints/fold1_mild_flow_best_graph_mexhat.pt
      --out-dir /root/npi/outputs/10-25-symw-ec-f12/fold1/ec_main > …/fold1_ec_main.log 2>&1 &`
- [ ] fold1 8/8 核对（test s{2,12,14,17,22,32,34,37}、P=0.500 精确）后同款 fold2
      （`--fold 2` + fold2 ckpt + fold2 输出树）
- [ ] 跑后回填两 Run 表

**Gate**：单例失败续跑原命令；系统性失败停 + 根因。

## S3 — Run 003/004：每折 summary + v4 双 GC（离线）

- [ ] 每折：`--mode summary --fold {k} --symw --main-dir fold{k}/ec_main --out-dir
      fold{k}/summary --task-label …`；`recompute_conn_v4.py --main-dir … --out-dir
      fold{k}/recompute_v4 --task-label 10-25-symw-ec-f12-fold{k} --surrogate-name
      "sym-w f12 fold{k} surrogate (SymWAdam W←(W+Wᵀ)/2, λ0.2; best 0.8xxx@Exxx)"`
- [ ] 核对：重映射不变性 8/8、三断言 8/8、双 GC per-subject（各折 8 例）/group/group_mean

## S4 — Run 005：跨折汇总（离线小脚本或 python heredoc 一次性 → 落档脚本）

- [ ] `outputs/10-25-symw-ec-f12/crossfold/crossfold_summary.json` + README 表：
      fold0（在档 symw-ec-unbiased summary/recompute_v4 读档）vs fold1 vs fold2——best
      val/interior/dyn_ratio/τ 中位/conn offdiag mean±std/v1·v2 GC ρ group mean±std +
      group_mean/逐被试 ρ 明细；跨代理脑实例描述性并置注记
- [ ] experiment.md 登记 + 跨折判读段

## S5 — README + 收尾三件套

- [ ] outputs README（配置/模型版本/ckpt 路径/复现命令/跨折表/caveat）
- [ ] 2.2 check dispatch；3.3 spec 判定；3.4 commit（仅本任务路径）+ 归档；研究记忆回写
      （progress.md Study II 状态 + 待拍板⑩联动注记）+ journal

## 回滚点

- S1 后代码可单文件回退；outputs 树独立可整删重跑；任一 Gate 失败停 + 存证报告。
