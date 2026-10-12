# Implement — GP先验流匹配 Mamba 代理脑（edra fold0）

> 前置条件（0. 完成前不得写任何代码）：① 用户「确认启动」（**唯一剩余门禁**，四项设计决策已全部落定：核=时空可分 K_s⊗K_t、K_s=Ricker 三维正定形式 σ=0.045、Mamba=官方 mamba_ssm 2.3.1（已装已验证）、GC/PSD=future-only、outputs=`/root/npi/outputs/gpfm-mamba-f0/`）；② `task.py start` 已执行。
> 每步完成即勾选；跑任何训练/评估前先在 `{TASK_DIR}/experiment.md` 登记 Run（目的/配置/seed/指标口径）。

## 0. 决策与登记（gate）

- [ ] 0.1 启动确认收到后：核对 design §8 全部已定项回填 config 默认值（K_s=Ricker 3D `(1−d²/(3σ²))e^{−d²/2σ²}` σ=0.080+jitter_s=1e-3、K_t=OU γ=1/p=50/iso=1e-2、权重 {1,1,0.1,0.1}）；PRD 待拍板 #6 默认口径（tv 纯平滑、corr 逐通道、x̂1 参数化、σ_s=0.080 默认值）留痕；identity 消融臂不跑（默认）
- [ ] 0.2 `task.py start`；在 experiment.md 登记 Run-0（目的：fold0 单 seed 前置验证训练；配置：seed=1234、窗口 384/128、损失权重 {1,1,0.1,0.1}、主指标 test-MAE 等，完整 resolved config 跑前补齐）
- 验证：`python3 ./.trellis/scripts/task.py current` 指向本任务且 status=in_progress

## 1. 骨架与数据模块

- [ ] 1.1 创建 `models/gpfm-mamba-v1/`（config.py：全部超参与默认值，一处定义）
- [ ] 1.2 `data.py`：受试者级 5-fold 分层划分（seed=42）、fold0、qc 过滤、z-score（仅训练折）、stride=64 窗切分（与 design §4 一致，config 可调）、内部验证受试者切分
- [ ] 1.3 `tests/test_data.py`：断言①受试者不跨折且分层平衡 ②z-score 统计仅来自训练折 ③形状 (B,62,384)/(B,62,128) ④同一 seed 划分可复现
- 验证：`python3 models/gpfm-mamba-v1/tests/test_data.py`（退出码 0，打印各折受试者数/窗数）
- 回滚点：commit "data module + tests"

## 2. GPR 条件先验

- [ ] 2.1 `gpr_prior.py`：**时空可分核实现（design §2.1）**——时间核注册表（OU 默认/Matérn-3/2 备选），时间网格 τ_i=i·π/p（p=50，禁止整窗 [0,1] 归一）；空间核 **K_s = Ricker 三维正定形式 `(1−d²/(3σ²))·e^{−d²/(2σ²)}`**（chan_coords_62.npy，σ=0.080，`+jitter_s·I`，jitter_s=1e-3；**禁止实现严格 1D Ricker——实测非 PSD**）；白噪 iso=1e-2 折叠进 K_t（保持 Kronecker 结构，不做联合恒等白噪）；float64 构造 → `A=K'_t,fp·cholesky_solve(K'_t,pp+1e-4I)`、`L_t=chol(K'_{t,f|p})`（128×128）、`L_s=chol(K_s)`（62×62）→ float32 buffer；`sample_prior(X_past) → x0^f = μ + L_s·ε·L_tᵀ`、`posterior_mean(X_past) → μ = A·y^p`
- [ ] 2.2 `tests/test_gpr.py`：①A 与 numpy 直接求解一致（rtol 1e-5）②**可分性检验：K_s=I 时与逐通道实现逐元素一致；K_s≠I 时 A 不变（K_s 消去）、采样协方差 ≈ K_s⊗K'_{t,f|p}**（大样本经验矩，容差登记）③每通道边缘方差 ≈ diag(K'_{t,f|p})、通道间经验相关 ≈ K_s ④**K_s 在真实电极坐标上 PSD**（min_eig > −1e-8，σ∈{0.045,0.080,0.137} 三档 + jitter 后 Cholesky 成功）、单位对角 ⑤μ 随 past 线性
- 验证：`python3 models/gpfm-mamba-v1/tests/test_gpr.py`
- [ ] 2.3 **训练折核诊断复跑**（~1 min）：在 fold0 训练折正式 z-score 口径下复跑核诊断（R²(μ)/post_std/条件数 + **σ_s 扫描 {0.045, 0.080, 0.137}（强中心-周边 / 平衡·默认 / 全局相干）：每档报告 min_eig、邻位相关、负瓣范围、有效秩与 post_std 匹配度**），确认 σ=0.080 默认档复现适配研究结论（若诊断推翻默认档，改档属口径变更须登记）；产物作 design §2.1 附件落任务 `research/`
- 回滚点：commit "gpr prior + kernel re-diagnosis + tests"

## 3. Mamba 骨干与速度网络

- [ ] 3.1 `mamba.py`：官方 mamba-ssm 适配层（`mamba_ssm.Mamba` API，d_model=16/d_state=16/d_conv=4/expand=2；环境已装 2.3.1 并验证）
- [ ] 3.2 `model.py`：token 化（62+段嵌入+t 正弦嵌入）→ 4×Mamba(d_model=16,d_state=16,d_conv=4,expand=2) → 头输出 v_θ(B,62,128)；EMA 封装
- [ ] 3.3 `tests/test_model.py`：①前向形状 ②因果性（改 future 输入不改 past 位置输出）③梯度有限 ④参数量打印（登记）
- 验证：`python3 models/gpfm-mamba-v1/tests/test_model.py`（含 GPU 冒烟：1 batch forward+backward）
- 回滚点：commit "mamba + model + tests"

## 4. 损失与训练循环

- [ ] 4.1 `losses.py`：mae/cons/tv/corr 四项 + 权重合成；eps 与 nan 守护
- [ ] 4.2 `tests/test_losses.py`：手工小张量对照独立参考实现；常数序列边界（corr eps）
- [ ] 4.3 `train.py`：路径采样 `x_t=(1−t)x0+t·x1+σ_min·η`，**t = sigmoid(N(−0.8, 0.8²))（logit-normal，clamp [1e-4, 1−1e-4]；不用 U[0,1]）** → v_θ → **一步生成 x̂1 = x_t+(1−t)v_θ**（训练全程不积分）→ 复合损失（四项，仅 future 列）；AdamW+cosine、EMA、按 t 分桶监控、**‖v_θ‖ 均值 / 生成幅度监控（x̂1 的 std_t、幅度直方图、多样本展宽 vs GT）**；**R9**：每 step 进度行（四项损失/it/s/ETA）→ stdout，每 epoch 追加 `logs/metrics.jsonl`，每 epoch 末**内部验证总 loss**（固定 t 网格 {0.1,0.3,0.5,0.7,0.9}+固定 ε 确定性复算）→ 创新优覆写 `best`、无条件覆写 `last`（均含 model+EMA+optimizer+scheduler+epoch/step+RNG 状态+best 指标+config hash），`--resume <last.pt>` 全量恢复续跑、dataloader epoch 种子化；**随机性同源固定**（python/numpy/torch/random seed、dataloader shuffle+worker seed）与 `cudnn.deterministic=True`（benchmark=False）显式声明入 config
- [ ] 4.4 冒烟：小子集（~4 受试者）1 epoch，确认四项损失均有限且下降、吞吐（it/s）记录到 experiment.md；**断点续传冒烟**：训练 2 epoch → kill 进程 → `--resume` 续跑 → 校验 metrics.jsonl 无缺口、loss 连续、epoch 计数/RNG/optimizer 状态恢复正确
- [ ] 4.5 **1-epoch 基准测试（用户要求）+ 用户确认门禁**：全量训练折 1 epoch——记录 peak 显存（`torch.cuda.max_memory_allocated()`）、epoch 墙钟时间、it/s、样本吞吐、100-epoch 外推时长、四项损失走势与 t 分桶 sanity；结果写 run 目录（bench 记录入 `logs/`）+ experiment.md；**→ 基准摘要汇报用户，等待确认「进入 fold0 完整训练」或「先优化模型」；决策与理由记入 experiment.md，未确认不得执行 §5**
- 验证：`python3 models/gpfm-mamba-v1/train.py --smoke`
- 回滚点：commit "prior+model+losses+train smoke"

## 5. 完整训练（fold0 单 seed · **前置门禁：4.5 基准经用户确认放行**）

- [ ] 5.0 前提核对：4.5 用户门禁已放行（experiment.md 有决策留痕）；若用户选择先优化，回到对应模块步骤并按口径变更纪律登记

- [ ] 5.1 跑前核对 experiment.md Run-0 配置齐全（seed/划分/权重/口径）；创建 `run_<id>/`，dump resolved `config.yaml`（含全部超参与 seed 与确定性开关）与 `env_snapshot.txt`（pip freeze）；**z-score 逐通道统计量一并落盘 run_<id>/**
- [ ] 5.2 nohup 后台完整训练（epochs=100，batch=256；日志 `outputs/gpfm-mamba-f0/run_<id>/logs/train.log`；checkpoint best/last，命名含变体名）
- [ ] 5.3 训练完检查：loss 曲线、按 t 分桶损失、无 NaN；best/last 保留策略执行（**best 按内部验证总 loss（固定 t 网格口径），每 epoch 评估**）；`metrics.jsonl` 全程无缺口；复现三要素（命令+快照+resolved config）齐备；若发生过续跑，experiment.md 已按「同一 Run 延续」补记断点位置与差异
- 验证：训练日志尾部无异常；`ls outputs/gpfm-mamba-f0/run_<id>/checkpoints/` 两份 ckpt

## 6. 采样与评估（前置验证）

- [ ] 6.1 `sample.py`：Euler ODE 16 步采样（EMA 权重），测试折每窗 K=16 样本；逐批打印进度与 ETA（R9）
- [ ] 6.2 `evaluate.py`：主指标 test-MAE（**ODE 终点样本 x̂(1) 口径**：多样本均值为点预测；勿实现成训练期 x̂1 公式）+ CRPS + 逐通道 r；持久性基线与 GPR 均值基线同口径对比；**采样终点误差监控**（R1 触发判断：高 t 桶 [0.8,1] mae / 中 t 桶 [0.4,0.6] mae ≥1.5，或终点 MAE ≥ 训练末期 x̂1-MAE 的 1.5× → 报告并提请预案）；**全部进度与结构化结果写 run_<id>/logs/eval.log + eval_metrics.jsonl（R9 留存）**
- [ ] 6.3 `gc_psd.py`：numpy-only；**默认 future-only 口径（design §6，若用户拍板整窗则附加对照同报）**；GC=全模型 VAR(p) OLS（BIC 真实组选 p∈{1..10} 后两组共用）+ 块 Schur 补成对条件 F 统计 + BH-FDR 附 raw p；生成侧每窗固定 seed 取 1 条样本；PSD=Welch（两组完全相同分段，nperseg 登记默认 512）；真实/生成两组 62×62 矩阵 → Pearson/Spearman + bootstrap 95% CI；GC/PSD 各阶段打印进度与 ETA（R9）
- [ ] 6.4 `common/visualizer.py`（共享区起点）：实现前加载 nature-figure 技能按其 figure contract 写绘图代码；图类型：①生成 vs 真实轨迹叠加（多窗多样本）②PSD 对比③GC 矩阵对比④训练曲线；输出到 `run_<id>/figs/`
- 验证：`python3 models/gpfm-mamba-v1/evaluate.py` 产出全部数值与图，路径写入 experiment.md

## 7. 收尾（Phase 3 前半）

- [ ] 7.1 experiment.md：Run-0 补结果/结论（Q1 预测质量与 Q2 三检验分开陈述；单 seed 标注）
- [ ] 7.2 同步 `.trellis/research/progress.md` 当前状态与里程碑
- [ ] 7.3 向用户汇报前置验证结果 → **等待用户对 5-fold 的决定**（未同意不得启动）
- [ ] 7.4 trellis-check 全量检查（spec 合规/产物落点/复现命令）→ 3.3 spec 更新 → 3.4 commit

## 关键验证命令汇总

```bash
python3 models/gpfm-mamba-v1/tests/test_data.py
python3 models/gpfm-mamba-v1/tests/test_gpr.py
python3 models/gpfm-mamba-v1/tests/test_model.py
python3 models/gpfm-mamba-v1/tests/test_losses.py
python3 models/gpfm-mamba-v1/train.py --smoke
nohup python3 models/gpfm-mamba-v1/train.py > <outputs>/logs/train.log 2>&1 &
python3 models/gpfm-mamba-v1/evaluate.py
```
