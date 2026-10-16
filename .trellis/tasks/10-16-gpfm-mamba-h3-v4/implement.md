# Implement — H3 v4-B 通道独立骨干 + cMLP 组件式读出（执行计划）

> 顺序执行；每步含验证命令与门禁。v4-B = 用户 2026-10-16 指定架构（design §2）；v1 代码零改动。Phase 2 开始前须 `task.py start`（1.4 用户明确同意后）。

## 0. 前置（Phase 1 收口）——已完成 2026-10-16

- [x] 待拍板项全部落定（用户 2026-10-16 拍板 + 批准启动）：CI 共享 / d_z=16 / SiLU / 不加 lasso / **d_ch=1（用户指定，design §9.10 特例风险登记）**
- [x] 1.0c 复检：方法清单变化（A→B）已回填路由区块（v4-B 仍为 gpfm-mamba 路线下结构变体，不触发新路径 research）
- [x] `implement.jsonl` / `check.jsonl` 已配置（v4-B 无需变更）

## 1. v4 变体目录（design §4）

- [ ] `config.py`：v2 config 全量副本 + `d_model=16`（语义=d_ch，注记）+ `d_z=16` + `head_act="silu"` + `gen_lowpass_hz=40.0` + `variant_name="gpfm-mamba-v4b"` + docstring（用户架构规格与别名登记）
- [ ] `model.py`：`VelocityNetwork`（v4-B 语义：CI 编码 + 4×CI-Mamba + 逐通道 Norm + cMLP head；**类名沿用 VelocityNetwork 并按名导出 EMA——运行时契约四件套见 design §4.3**）+ `time_embedding`/`EMA` 复制（docstring 注明「复制自 v1 model.py，勿单独演化」）；`from mamba import build_mamba_backbone` 复用 v1 mamba.py
- [ ] `train.py` / `evaluate.py`：v2 wrapper 扩展为 **config + model 双遮蔽**（占据 sys.modules 后 importlib 加载 v1 主流程）+ `RUN_PREFIX="v4b-"` 前缀守卫 + 遮蔽自检 + **加载前运行时断言 `model 类默认 d_z/head_act == cfg 字段`**（d_z/d_ch 双处一致性防线，design §4.3）
- [ ] 单测（`tests/`，design §4.4 七项）：
  - [ ] CI 隔离性（扰动通道 j 输入，k≠j 的 h 逐位不变）
  - [ ] 因果性（扰动 future 输入不改 past 位置 h）
  - [ ] head 结构（W¹ 形状/列组 reshape 视图/62 子 MLP 参数独立）
  - [ ] 前向形状 (B,62,128)、梯度有限、无 NaN
  - [ ] 参数量断言 **≈64,056（d_ch=1, d_z=16 口径，±1%，design §3.2）**
  - [ ] config↔model 一致性（cfg.d_z/head_act 与类默认同值；config_hash 稳定）
  - [ ] 双遮蔽命中 + 前缀守卫正反例
- [ ] **v1 零改动回归**：v1 既有 7 个单测全跑 PASS（本方案不动 v1 任何文件，此项为守护性回归）
- [ ] 验证命令：`python3 models/gpfm-mamba-v4/tests/test_v4b.py` + v1 既有 7 测试
- [ ] **提交点 A**（回滚 = 删 v4 目录，v1 无回滚项）

## 2. 冒烟 + 1-epoch 基准（硬门禁——计算形态改变，必须实测）

- [ ] 冒烟：`python3 models/gpfm-mamba-v4/train.py --smoke --run-id v4b-smoke-a`（limit 批次全链路 + 划分校验 + R9 监控落盘）
- [ ] 基准：`python3 models/gpfm-mamba-v4/train.py --epochs 1 --bench --run-id v4b-bench-1ep`——登记 peak 显存 / epoch 墙钟 / it/s / 四项损失走势 / t 分桶 sanity / 100-epoch 外推（experiment.md + run 目录）
- [ ] **eval 前向探针（评审补充，入同一门禁）**：装载 v4 model 按 sampler 路径跑一个评估 batch（默认 64×K=16）量峰值显存与耗时——评估路径不被训练 bench 覆盖（元素量 ~16× v1）
- [ ] **用户确认门禁**：摘要（显存/时长/外推 + 预案建议：训练侧 batch 256→128 / d_z 16→8 / d_ch 16→8、评估侧 --batch-size 降档）汇报 → 用户确认后启动 §3；决策留痕 experiment.md；**若确认变更 d_z/d_ch：config+model 双处同改 + 重跑 §1 全部单测（口径变更流程）**
- [ ] **提交点 B**

## 3. 主 Run：v4b-main-01 100-epoch

- [ ] 跑前登记 Run 块（目的/配置含 seed=1234 与指标口径/代码 commit/命令）
- [ ] `mkdir -p /root/npi/outputs/gpfm-mamba-f0/v4b-main-01/logs` 后 `nohup python3 models/gpfm-mamba-v4/train.py --run-id v4b-main-01 > …/logs/train.log 2>&1 &`
- [ ] 监控：metrics.jsonl 无缺口、无 NaN、train/val 总 loss 差（过拟合观察项，design §9.1）、best/last 每 epoch 就位；中断 `--resume` 续跑（同一 Run 延续口径）
- [ ] 断点续传冒烟（kill → resume → 曲线连续）留痕
- [ ] **提交点 C**

## 4. 正式口径评估（Run 017 同口径）

- [ ] `python3 models/gpfm-mamba-v4/evaluate.py --run-id v4b-main-01`（Pass A/B + GC + PSD + 空间归因 + 全部图；正式 40 Hz 低通默认）
- [ ] `python3 models/gpfm-mamba-v1/fc_metric.py --run-id v4b-main-01`（指标 1 v2.1，硬编码共享 outputs 根）
- [ ] 七维度并列表（design §6 对照）+ 三检验核算 + R1 分子分母 + std_t 幅度 → 落盘 experiment.md
- [ ] **诊断副产品**：内部 readout 矩阵 **M[j,i]=‖W^{(i),1}_{:,j}‖_F（行=源、列=靶，全仓库约定）** 热图 + 与外部 GC 矩阵 Spearman（仅诊断，经 visualizer）
- [ ] **提交点 D**

## 5. H3 判读与收尾（Phase 3）

- [ ] 三态判读建议（支持/部分支持/不支持 + 滞后语义对应表述 + 归因局限声明）写入 experiment.md；交用户判定，**判定留痕**
- [ ] 2.2 全量质量检查（trellis-check：spec 合规 / 单测 / 产物完整 / 口径一致）
- [ ] 3.3 spec 更新：v4-B 回填 context.md 技术路线表变体注记；沉淀（双遮蔽模式、CI+cMLP 判读口径）按 trellis-update-spec 落 spec
- [ ] `.trellis/research/progress.md` 同步；experiment.md 收尾块（Q1/Q2 分开、H3 结论、移交项：归因分解变体/group lasso/5-fold 候选拍板）
- [ ] 3.4 提交（提交点 A–D）；归档前 prd 验收标准逐项勾销

## 验证命令汇总

```bash
python3 models/gpfm-mamba-v4/tests/test_v4b.py                       # §1 全部单测
for t in test_model test_sample test_gpr test_losses test_data test_gc_psd test_bandlimit; do \
  python3 models/gpfm-mamba-v1/tests/$t.py || exit 1; done           # §1 v1 既有 7 测试逐一全跑
python3 models/gpfm-mamba-v4/train.py --smoke --run-id v4b-smoke-a
python3 models/gpfm-mamba-v4/train.py --epochs 1 --bench --run-id v4b-bench-1ep
python3 models/gpfm-mamba-v4/train.py --run-id v4b-main-01           # 100-epoch（nohup，见 §3）
python3 models/gpfm-mamba-v4/evaluate.py --run-id v4b-main-01
python3 models/gpfm-mamba-v1/fc_metric.py --run-id v4b-main-01
```
