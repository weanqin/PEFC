# 执行清单 — 10-28-latent-flow-symw-f34

> 零代码改动纯跑批；所有命令在 `/root/npi` 下执行；k ∈ {3,4}。
> 环境：conda npi（`/root/miniconda3/bin/python`）@ 本机 RTX 4080S，
> 对照 `.trellis/research/environment.md`（在案场景，无新环境）。

## S0 预检（启动前）

- [ ] GPU 空闲确认（`nvidia-smi`）；磁盘余量确认。
- [ ] manifest fold3/4 划分回读（test 名单与 prd 一致）；v1 refs refs_s*.npz
  对折内 8 例存在性确认。

## S1 fold3 + fold4 训练（R1，镜像 f12 Run 001/002 逐字）

- [ ] 每折 stage1（w500 fold0 stage1 口径）：
  ```bash
  OMP_NUM_THREADS=8 /root/miniconda3/bin/python models/latent-flow-v1/train_mild_depress.py \
    --fold k --stage 1 --window-len 500 --obs-len 375 --forecast-len 125 \
    --epochs 150 --patience 30 --batch-size 64 --lr 5e-4 --seed 42 \
    --run-dir outputs/10-28-latent-flow-symw-f34/run_symw_foldk \
    2>&1 | tee outputs/10-28-latent-flow-symw-f34/run_symw_foldk_stage1.log
  ```
- [ ] 每折 stage2（symw Run 001 verbatim，--vae-ckpt 指本折 stage1 best）：
  ```bash
  OMP_NUM_THREADS=8 /root/miniconda3/bin/python models/latent-flow-v1/train_mild_depress.py \
    --fold k --stage 2 --window-len 500 --obs-len 375 --forecast-len 125 \
    --batch-size 64 --lr 1e-3 --epochs 150 --patience 30 --seed 42 \
    --unfreeze-decoder --anti-uniform-lambda 0.2 --flow-exit-gnn --sym-w \
    --vae-ckpt outputs/10-28-latent-flow-symw-f34/run_symw_foldk/checkpoints/foldk_mild_vae_best.pt \
    --run-dir outputs/10-28-latent-flow-symw-f34/run_symw_foldk \
    2>&1 | tee outputs/10-28-latent-flow-symw-f34/run_symw_foldk_stage2.log
  ```
- [ ] 验证门 G1：stage1/stage2 history json 落档；CLI 对拍（config json 的 cli 块
  剔除 fold/run_dir/vae_ckpt 后与 fold0-2 逐键相等，f12 先例）；W 对称断言
  （inline 只读 python：best ckpt state_dict 中 W 键 torch.equal(W, W.T)、max|W|）；
  质量抽查（quality_check_mild.py --fold k --flow-exit-gnn --ckpt <stage2 best>
  --subjects <折内 3 例：2 MDD + 1 HC> --k 32 --seed 42 --window-len 500
  --obs-len 375 --forecast-len 125 --out-dir run_symw_foldk/quality）——medoid/real
  std 与三折先例并置。
- 回滚点：训练失败仅删 `run_symw_foldk/` 重跑，不影响其他折。

## S2 fold3 + fold4 扰动归因 EC（R2，镜像 symw-ec-f12 Run 001-004）

- [ ] **S2.0 守卫扩展（2026-10-28 Execute 发现的 prd 缺陷修订）**：
  `ec_mild_unbiased.py` `_caliber_asserts` symw 分支另接受
  `10-28-latent-flow-symw-f34` run_dir（fold ∈ {3,4} + `fold{k}_` 前缀断言，
  f12 扩展同款）；改动后 fold0/1/2 原路径行为不变（分支互斥，无回归面）。

- [ ] 每折 main（nohup 后台，json 逐例原子写 = 断点续跑）：
  ```bash
  OMP_NUM_THREADS=8 nohup /root/miniconda3/bin/python models/latent-flow-v1/ec_mild_unbiased.py \
    --mode main --fold k --symw --task-label 10-28-symw-ec-f34-foldk \
    --ckpt outputs/10-28-latent-flow-symw-f34/run_symw_foldk/checkpoints/foldk_mild_flow_best_graph_mexhat.pt \
    --out-dir outputs/10-28-symw-ec-f34/foldk/ec_main \
    > outputs/10-28-symw-ec-f34/foldk_ec_main.log 2>&1 &
  ```
- [ ] 验证门 G2：8/8 例 subject_*.json；P(resp>τ)=0.500 逐例；connectivity_formula
  = v4 串逐例核对。
- [ ] 每折 summary（组平均 EC = group_mean.npz + group_summary.json）：
  ```bash
  OMP_NUM_THREADS=8 /root/miniconda3/bin/python models/latent-flow-v1/ec_mild_unbiased.py \
    --mode summary --fold k --symw \
    --main-dir outputs/10-28-symw-ec-f34/foldk/ec_main \
    --out-dir outputs/10-28-symw-ec-f34/foldk/summary \
    --task-label 10-28-symw-ec-f34-foldk
  ```
- [ ] 每折 gccorr（v1 单参考：逐被试 + 组统计 + 组均值矩阵）：
  ```bash
  OMP_NUM_THREADS=8 /root/miniconda3/bin/python models/latent-flow-v1/ec_mild_unbiased.py \
    --mode gccorr --fold k --symw \
    --main-dir outputs/10-28-symw-ec-f34/foldk/ec_main \
    --out-dir outputs/10-28-symw-ec-f34/foldk/gccorr \
    --task-label 10-28-symw-ec-f34-foldk
  ```
- 回滚点：EC 失败重跑单例（断点续跑天然支持）；不影响训练产物。

## S3 图件（R4）

- [ ] 每折 viz_ec_pert（必交付）：`viz_ec_pert.py --main-dir .../foldk/ec_main
  --out-dir .../foldk/viz_ec`。
- [ ] 每折 viz_symw_ec_pairs（schema 验证后）：`--recompute-dir .../foldk/ec_main
  --gc-dir /root/npi/refs/granger_bivariate_spectral/mild40-trial500-zscore-fs250-fixed8-f1-45-favg
  --out-dir .../foldk/viz_pairs`；不兼容则登记降级原因（AC4 不依赖）。

## S4 汇总与报告（R3/R5）

- [ ] README 落 `outputs/10-28-symw-ec-f34/`：每折逐被试相关系数表 + 组平均
  （跨被试 mean±std + 组均值矩阵）数值 + fold0-4 五折并置（fold0-2 引先例在档值）。
- [ ] experiment.md 逐 Run 块登记（S1/S2/S3 各一 Run）；数值与 README 一致。
- [ ] 研究记忆回填：progress.md symw 条目补 fold3/4（val/幅度/EC 相关形态）。

## S5 收尾（R6）

- [ ] trellis-check（AC1-AC5 逐项 + 路径隔离核查）。
- [ ] spec 更新判断：若 viz_symw_ec_pairs 的 main-直出 schema 兼容性结论有价值
  则记 evaluation-calibers/visualization；无沉淀点则登记"无"。
- [ ] commit（产物 gitignore 外的任务文件 + spec/记忆变更）。

## 风险与降级

- viz_symw_ec_pairs 期望 recompute 目录 schema（先例 recompute_v4 产物）——main
  直出后字段一致性未验证：运行时校验，失败降级 viz_ec_pert-only（AC4 不受影响）。
- EC 与训练同卡串行：训练先行完成两折再统一跑 EC，避免显存竞争。
- 单 seed 单次（先例确定性档）；如某例异常挂起（s9 型先例）按断点续跑处理并登记。
