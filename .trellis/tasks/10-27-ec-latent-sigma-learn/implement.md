# Implement — EC 潜空间可学 σ 扰动首实现

> 前置：prd.md（R1–R4 契约）+ design.md（§2 修复型 loss、§3 接缝、§5 数据流）+ research/ec-latent-sigma-adaptation.md（θ/数据接口与风险清单）。执行顺序 S1→S6，每步带验证命令；S3 前有 review 门。

## S1 接口核实（research 已完成，结论随档）

- [x] θ 装载链确认：`build_model_mild(_model_args(True), stage=2, 375/125)` + `load_flow_ckpt`；默认 ckpt `outputs/10-25-latent-flow-exit-gnn-symw/run_symw_fold0/checkpoints/fold0_mild_flow_best_graph_mexhat.pt`；use_fc_decoder=False 固定全窗 decoder；geometry_mild.pt 断言沿用（research §1.1）
- [x] 数据管线确认：load_manifest（SHA 4798790…）+ compute_mild_train_norm(500)（fold0 norm SHA d184e647…）+ MildTrialDataset 375/125 + W500 三元组断言（research §1.2）
- [x] 包体零改动判定成立；ε 独立 Generator；EC 落盘 orientation 行=源（research §1.3/§3.1）

## S2 脚本实现 `models/latent-flow-v1/ec_latent_sigma_learn.py`

- [ ] BLAS 守卫头部 + `sys.path.insert(VARIANT_DIR)`（ec_mild_unbiased.py:56-74 惯例）
- [ ] `NoiseLayer`（forward(z_obs, eps) 接受外部 ε——独立 Generator 供给；project 盒投影；entropy 排除靶）
- [ ] `stage2_single_step`（:1111-1121 复刻）+ `decode_fc`（全窗切 [:, obs_len:, :]，skips 干净侧）+ `pearson_fc`（逐 (样本,通道) 时间维，eps=1e-8）
- [ ] cudnn.deterministic=True + benchmark=False + set_seed（ec_mild_unbiased.py:310-311,339 同款；AC6 复现依赖——review B MAJOR）
- [ ] 被试级缓存（encode/t/fc_origin，z-std 诊断）+ 逐靶循环（128 靶或 --target-channel 单靶）+ **分母探针**（逐（被试,靶）σ_ref=0.01 全通道 no_grad 单前向 ⇒ MAE_ref 标量缓存 + 护栏 max(MAE_ref, 1e-6·mean|fc0|) + 双种子偏差检查，R3.5）+ 优化循环（SGD、loss=λ·L_f−H（L_f=MAE/MAE_ref，--fidelity mae 默认；pearson 备选 λ(1−P)−H）、project 每步、逐 epoch 诊断 jsonl 含 |∂L_f/∂σ| vs 1/(127σ) 对照行、P 描述性行、末 5 epoch σ 均值与逐靶 q0=0 掩码）
- [ ] skips 的 n_mc 维度感知 repeat（view(B,C,·,·).repeat_interleave(dim=0).flatten(0,1)；naive repeat 会静默打乱 (b,c) 配对——review B MAJOR）+ pearson_fc 返回 P 本身（≠ _pearson_corr_fc 的 1−P 口径）
- [ ] σ*→EC_v4（靶位 NaN）+ orientation 转置（行=源）+ payload（design §5 字段）+ NaN 守卫（nan_to_num + 有限性断言 + 拒跑路径）
- [ ] CLI：`--mode {construct,smoke,main}` + design §6 全参数（含 --fidelity 默认 mae/--sigma-ref 默认 0.01/--eps/--bf16/--lam 默认 0.1/--lr 默认 0.5）
- 验证：`python3 -c "import …"` 编译通过；`--help` 全参数可见

## S3 Review 门（实现前不得跑 GPU）

- [ ] design §2/§3 与代码逐条对读（loss 符号/熵排除靶/CRN t 固定/穿堂 no_grad 边界/投影时序）
- [ ] trellis-review 派发结论处理（见会话记录；blocking 项先改再继续）

## S4 CPU 构造检查（AC2 十项，`--mode construct`）

- [ ] ①σ 盒投影（含边界值）②H 解析梯度 vs autograd（fp64）③靶通道恒零噪声 + 靶位 σ 无梯度 ④CRN t 固定下 fc_origin 逐位复现 ⑤pearson（返回 P 本身）vs scipy.stats.pearsonr（fp64）+ 常数序列护栏（备选口径路径）⑥单步构造复刻 vs 手写公式（自参照对拍局限注记）⑦穿堂检查（含 MAE_ref detach 不进计算图）⑧合成驻点对拍：MAE 口径 σ*=σ_ref/λ 闭式精确（含 1/λ 标度）；pearson 备选口径 σ*∝1/√λ ⑨NaN 注入拒跑 ⑩decoder 净/扰动 z_obs 靶读出逐位等价 ⑪**锚点断言：NoiseLayer 增益置 0 ⇒ fc_p 与 fc_origin 逐位等价**（天然覆盖 CRN 配对与 skips repeat 正确性——review B）⑫MAE 逐点参考实现 vs 脚本（numpy，fp64）+ 分母护栏（合成 MAE_ref=0 ⇒ 护栏触发不 NaN）+ 常数序列护栏（pert≡origin ⇒ 分子 0）⑬分母探针契约（同 seed 逐位复现、双种子一致性 <5% 超标落噪声底 flag、探针输出 detach）
- 验证：`python3 models/latent-flow-v1/ec_latent_sigma_learn.py --mode construct` 全 PASS 退出码 0

## S5 GPU 冒烟（AC3/AC4/AC6，`--mode smoke`）

- [ ] 2 例 × 3 epoch × 靶 {0,64}（默认 fidelity=mae、λ=0.1）：σ 更新轨迹、逐 epoch 诊断（含 |∂L_f/∂σ| vs 1/(127σ) 行、P 描述性行、末 5 epoch σ 均值）、EC_v4 落盘、无 NaN、显存 <24 GiB、逐例 <5 min、计时入档
- [ ] **λ 梯子**：同 2 例 × λ∈{0.05,0.1,0.2,1.0} × lr∈{5e-3,0.5} interior 对照表（每格 <1 min）→ experiment.md（闭式先验 σ*≈σ_ref/λ 并置；λ=1 预期全钉下界零判别如实登记；首 epoch 实测 |∂L_f/∂σ| 量级核验解析先验）
- [ ] 跑前先写 experiment.md Run 块「目的+配置」（experiment-log 规范；跑后回填结果/结论）
- [ ] 同 seed 重跑锚点逐位等价（σ 终值/EC_v4）
- [ ] payload + `experiment.md` Run 块登记（目的/配置/结果/结论）
- 验证：run 目录产物齐（payload.json/diagnostics.jsonl/ec npz/construct.log）；experiment.md 无空块

## S6 质量与收尾

- [ ] trellis-check 派发（对照 prd AC1–AC7 + spec：perturb-ec-pipeline/reproducibility/path-isolation）
- [ ] spec 沉淀：perturb-ec-pipeline.md 新增潜空间可学 σ 变体契约节（修复型 loss 双口径语义（MAE 比值默认+pearson 备选）/分母探针契约/联合边际 caveat/潜空间 σ 标度注记/CRN=t 固定/EC 读数边界（跨 loss/跨 λ/跨 σ_ref 不可直比、DC 漂移语义差异））
- [ ] research 记忆回填：context.md 技术路线表增行（3.3 spec-update 时）+ progress.md 当前状态同步
- [ ] 批量 commit（工作提交 → 归档 → journal 三段式）

## 回滚点

- S2 后：删单文件即可（无包体改动前提）
- S5 后：删 outputs/10-27-ec-latent-sigma-learn/ + 单文件
- 若 research 判定包体须改动：回到 design §9 复议后再实现
