# hist-block 无偏 EC 主跑 × symw fold0 代理脑（10-27-ec-histblock-symw-f0）

## 子研究路由（1.0b / 1.0c）

<!-- trellis-research-routing -->

- **分类结果**：`study2`（EC 推断算法——hist-block 噪声范式主跑执行）
- **依据**：扰动归因 EC 计算（ec_mild_unbiased.py main 模式）；代理脑为固定输入不训练。
- **新路径/方法检测（1.0c）**：`已有路径：global-tau 无偏 EC × hist-block 噪声变体
  （10-27-ec-histblock-noise 已交付开关并通过全部验证；本任务 = 该范式首次全量主跑）`
  ——无新方法，research adaptation 不触发。
- **research 产物**：`无`
- **路由备注**：执行型任务，方法侧零改动（--noise-mode hist-block 旗标直用）。

## Goal

用 hist-block 历史段分组共享噪声范式（`--noise-mode hist-block` 全共享默认：375=15×25
组内复制组间独立、通道/被试共享、种子 seed+9375）在 **fold0 test 全部 8 例**上运行
global-τ 无偏扰动归因 EC（两阶段：σ_ref 响应扫描 → τ_global → log 网格 σ* 走查 →
conn v4）。代理脑 = **run_symw_fold0**（用户标识 `10-25-md-lf-anti0.2-gnnexit`）。

## 用户指令（2026-10-27，五要素严格保留）

1. 噪声构造范式 = hist-block（严格保留）；2. fold 划分 = fold0；3. 样例数量 = 全部
8 例；4. 分析类型 = 扰动归因 EC；5. 代理脑标识 = `10-25-md-lf-anti0.2-gnnexit`。

## 代理脑身份对拍（侦察在档）

- 用户标识映射：`10-25-md-lf-anti0.2-gnnexit` → 全库唯一同时满足「10-25 任务 +
  anti-uniform λ0.2 + GNN exit + fold0」的 ckpt = `outputs/10-25-latent-flow-exit-
  gnn-symw/run_symw_fold0/checkpoints/fold0_mild_flow_best_graph_mexhat.pt`
  （stage2_config.json 实测：anti_uniform_lambda=0.2 ✓ flow_exit_gnn=true ✓
  sym_w=true（简记未含、模型固有属性，随档注明）best val 0.8054@E144）。
- 调用面：`--symw --fold 0`（run_dir 段守卫 + norm SHA fold0 键 d184e… 与训练侧一致，
  10-25-symw-ec-unbiased 先例同路径）。

## 口径（全部不动）

q=0.5 / σ_ref=0.3 / log[0.01,0.48]×24 / seed 42 / 30 trial 全量 / bf16 / τ 分位含对角
/ conn v4 = −(lnσ*+0.5·ln2πe)+5/7 / hist-block 全共享默认（组 25、段 375、双共享开）。

## 验收标准（AC）

- AC1 主跑 8/8 例完成（json+curves.npz 全落本任务 outputs），P(resp>τ) 口径与既往
  一致性抽查；
- AC2 payload 完整：noise_mode=hist-block + 7 hist 配方键 + symw 代理脑分支
  （task/ckpt_stage2_best/variant/caveat）；
- AC3 组级 summary（离线 mode=summary）落档：分布/方差分解/跨被试稳定性；
- AC4 判读登记：与 iid 三跑（w500/exit-gnn/symw）τ/interior 的描述性并置（hist-block
  谱形不同 ⇒ 数值不可比性注明），不下 EC 有效性结论（provisional 随 θ 拍板联动）；
- AC5 experiment.md Run 块四要素齐全 + progress.md 速览同步。

## Out of Scope

gccorr（GC 对照）不在本令（另令另跑）；方法侧代码零改动；5-fold 扩展另议。
