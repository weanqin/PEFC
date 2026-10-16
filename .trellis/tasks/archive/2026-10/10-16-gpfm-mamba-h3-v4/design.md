# Design — H3：v4-A 低秩输入瓶颈变体（v4-B 已放弃存档；当前累计口径）

> 需求见 `prd.md`（含方向变更登记）。v4-A 主体 = 本任务最初设计（第一轮 trellis-review 修正版），本文件按当前口径（32/4/10、早停 10、恒定 lr）整合。

## 1. 假设与因子设计（2×2 缺失格，恢复原框架）

| | 容量 16 | 容量 62（trunk 宽度） |
|---|---|---|
| 接口强制混合（rank r<C） | v1（已测，GC 0.2755） | **v4-A（本方向）** |
| 无接口瓶颈 | —— | v2（已测，GC 0.1227） |

- v4-A vs v2 单变量：in_proj 79→62 满秩 vs 79→**r=16**→62 低秩；trunk 容量与 head 全同 v2。
- 与 v4-B 的联合对照（判读材料随附）：v4-B 证「容量必要而不充分」（GC 幅值/峰频/逐通道 r 恢复但 GC r 0.163）——v4-A 若在共享 trunk 上 GC 回升，指向「接口瓶颈混合」是 GC 载体；若同样不回升，缺口共同指向损失侧。
- 三态判读与不对称守护、0.15–0.22 邻域加 seed 规则：沿用任务内预注册（experiment.md 判读框架行）。

## 2. v4-A 架构（第一轮评审版原样）

```
v2:  in_proj = Linear(79 → 62)                 # 满秩，通道对齐可学
v4a: in_proj = Linear(79 → r=16) → Linear(16 → 62)   # 纯线性低秩（无激活）；rank≤16
     （4×MambaBlock d_model=62 / head 62→62 / 其余全部与 v2 逐字段一致）
```

- 纯线性：两层复合 = rank≤16 线性映射，全部 62 通道逐 token 信息经 16 维共享潜表示——「接口强制混合」的干净单变量（加激活 = 1 隐层 MLP 第二语义，评审背书排除）。
- 初始化 PyTorch 默认（两层复合谱窄于单层——bench 异常走势第一怀疑项，登记备查）。
- 瓶颈位置语义：v1 的 d16 贯穿 trunk 全深 vs v4-A 仅输入接口——若 GC 部分回升，残留归因「深度持续混合」（原 C 方案域，不追）。

## 3. 参数量

| 模块 | v2 | v4-A |
|---|---|---|
| in_proj | 4960 | **2334**（(79×16+16)+(16×62+62)） |
| trunk + head + 其余 | ≈127.5k | 不变（d_model=62） |
| **总** | ≈132.4k | **≈129.8k（0.98× v2）** |

计算/显存 ≈ v2 量级（v2 实测 9.45 s/epoch@34 例、1.94 GiB；v4a@32 例预期 ~9 s/ep、~2 GiB，100ep 上界 ~15 min，早停可能缩短）——bench 实测为准。

## 4. 实现路径

1. **v1 侧（default-off 钩子，原 §3.1 评审版）**：`model.py::VelocityNetwork` 增 `bottleneck_rank=None`（None 分支结构/state_dict 键逐位不变；r 分支 `nn.Sequential(Linear(79→r), Linear(r→62))`，键 `in_proj.0/1.*`）；`train.py`（模型构建处）与 `sample.py::load_eval_model` 增 `bottleneck_rank=getattr(cfg, "bottleneck_rank", None)`。**既有 default-off 钩子（早停/lr/manifest override）原样复用。**
2. **v4a 目录（v2 式薄变体）**：`models/gpfm-mamba-v4a/{config.py, train.py, evaluate.py, tests/}`——wrapper 仅遮蔽 config（模型经 v1 钩子）；`RUN_PREFIX="v4a-"`；r 扫描经 wrapper `--rank {8,16,32}` 改写被遮蔽 config 实例（config.py 默认唯一 16）+ run_id `rXX` 一致性校验；resume hash 校验防 r 串用。
3. **config（v2 副本 + 当前口径字段）**：`bottleneck_rank=16`、`n_val_subjects=4`、`split_manifest_override=<v4a>/splits/fold0_status_ok_seed42_val4.json`（fixate 首建；与 v4b manifest 逐字节一致由测试断言）、`early_stop_patience=10`、`lr_schedule="constant"`、`gen_lowpass_hz=40.0`、`variant_name="gpfm-mamba-v4a"`；**不含** d_z/head_act（v4b 专属字段，模型也不含 cMLP head）。
4. **单测**（v4a/tests，脚本式）：①None 分支等价（同 seed 前向逐位一致 + 键集不变）②r=16 两层结构/rank 构造断言 ③前向形状/因果/梯度有限 ④参数量 ≈129,832±1%（明细：in_proj 2334 + trunk 123,504 + norm 62 + head 3,906 + seg 2 + head_norm… 以实现实测精确数登记）⑤三变体 hash 零漂移（基线 v1 `c6ff80ac1c43`/v2 `fc2a9ce14948`/v3lp `0c2de30a4868`）⑥v1 七测回归 ⑦遮蔽/前缀守卫/`--rank` 覆写正反例 ⑧val4 manifest 一致性（32/4/10、test 逐 ID 同 v1、内容 == v4b 的 val4 manifest）。

## 5. 训练与评估口径

- 训练：当前累计口径（32/4/10、早停 10、恒定 lr 1e-3、batch 256、seed 1234、EMA 0.999、R9 全套监控/续传；nohup 脱离式启动——**会话事件不得再杀训练进程**，v4b-d8 中断教训登记）。
- 评估：`evaluate.py --run-id v4a-r16-main-01` + `fc_metric.py`（Run 017 同口径）；七维度**五列**对比（v1/v2/v4b-d1/v4b-d8/v4a）+ 三检验 + R1 + std_t；无 readout（无 cMLP head）。
- Run 编号接续 experiment.md（Run 011 起）。

## 6. 风险（精选）

1. MAE 轻微劣化（v1 容量代价参考 0.7083 vs 0.7010）——五维度并列登记无硬阈值。
2. 两层默认初始化谱窄——bench 异常走势第一怀疑项。
3. 单 seed 邻域敏感——预注册加 seed 规则。
4. 归因残留「接口 ≠ 全程混合」——部分支持时归 C 方案域。
5. 与 v1/v2 的口径差异四项（32 例/val4/早停/恒定 lr）——判读随附；与 v4b 同口径 ⇒ v4a vs v4b 干净可比。

## 7. 回滚

v4a = 删 `models/gpfm-mamba-v4a/` 与 `outputs/gpfm-mamba-f0/v4a-*`；v1 钩子 revert 三文件（default-off 零行为影响）。里程碑提交：钩子+单测 / v4a 目录+bench / 主 Run / 评估。
