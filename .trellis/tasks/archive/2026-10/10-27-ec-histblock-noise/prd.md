# EC 无偏噪声历史段分组共享变体（hist-block）——方法开关与验证

## 子研究路由（1.0b / 1.0c）

<!-- trellis-research-routing -->

- **分类结果**：`study2`（EC 推断算法——扰动噪声构造的可配置变体）
- **依据**：全部改动落在 EC 计算侧噪声构造（`ec_mild_unbiased.py`）；代理脑为固定输入，
  不训练、不做资格判定。
- **新路径/方法检测（1.0c）**：`已有路径：global-tau 无偏 EC 口径（w500 / exit-gnn /
  symw-fold0 / symw-f12 四先例已归档）`——本任务 = 该口径噪声构造的可配置扩展（默认关），
  无新方法，research adaptation 不触发。
- **research 产物**：`无`
- **路由备注**：hist-block 仅在显式旗标开启时生效，任何既有口径复跑不受影响；EC 结论的
  provisional 状态继承所属代理脑，与本任务无关。

## Goal

在 `models/latent-flow-v1/ec_mild_unbiased.py` 的无偏噪声口径（μ=0，σ·ξ）上新增可配置的
**历史段分组共享噪声**变体 hist-block，满足用户四条原始要求：

1. **通道与被试共享**：历史段噪声构造在通道维（128 源）与被试维完全共享——被试循环外
   只生成一次，广播/切片复用；禁止按通道或被试独立采样。
2. **时间维分组共享**：历史段 375 点按连续 25 点一组分 15 组；每组独立采样一次、组内
   25 点复制同一值；组间独立。
3. **配置化**：通道共享 / 被试共享 / 组大小（默认 25）/ 历史段长度（默认 375）均可配置；
   默认配置（iid 模式）与原实现行为逐位一致。
4. **输出与验证**：修改文件/函数、关键代码、维度变换说明 + 五项验证（同被试跨通道逐元素
   相同 / 跨被试逐元素相同 / 组内相同 / 15 组组间独立 / 非历史段噪声逻辑未变）。

## 用户指令与拍板（2026-10-27）

- 原始要求 = 上述四条（会话 2026-10-27 首条消息）。
- **拍板①（落盘）**：不采纳 ξ 数组落盘增强（无 `--hist-xi-npz`、无 `xi_sha256` 凭据链）；
  跨任务复现 = `hparams` 落盘**种子 + 完整构造配方**、确定性重建，成立前提四条（同 seed /
  同形状 (N,15) / 同设备类型 / 同 torch 版本）作为跨任务对拍注意事项写入交付说明。
- **拍板②（trial 维）**：trial 维不共享——基抽取 `(N=30, G=15)` 行独立，保留 trial 级
  随机性（30 trial 响应均值仍是蒙特卡洛平均）。
- **拍板③（种子）**：ξ 种子偏移 = `+9375`（= 375×25；与既有 `seed+source∈[42,169]`、
  `seed*10000+sid` 流正交）。
- **拍板④（立项）**：AskUserQuestion 2026-10-27 明确同意创建任务并实施。

## 范围

- **In**：`ec_mild_unbiased.py` 单文件改动（约 +90 行，净新增为主）；任务档验证脚本
  `research/test_hist_block_noise.py`；mini 干跑验证（改动前后 iid 对拍 + hist-block
  GPU 冒烟）；`experiment.md` 登记。
- **Out**：不跑 hist-block 全量 8 例主跑（是否跑、跑哪个代理脑/fold 为后续独立决策）；
  不改 `latent_flow` 包体；不改 `summary` / `gccorr` 模式行为（仅容忍 payload 新键）。

## 约束（硬）

1. `latent_flow` 包体零改动（mild 线纪律）；改动仅 `ec_mild_unbiased.py` + 任务档。
2. 默认调用面（不传新旗标）输出与改动前**逐位一致**；对拍锚点 = `sigma_matrix` /
   `raw_response_matrix` / `curves.npz(mse_curve)` / `tau_global`（conn 字段受
   config-fix 未提交 v4 公式平移影响，不作锚点）。
3. 历史段长度 = `obs_len` = 375（`_caliber_asserts` 钉死）；`hist_len % group_size == 0`。
4. 非历史段噪声逻辑（`perturb_channel` / `forecast_fc` / `model.sample` /
   `per_target_mse` 路径）零改动。
5. CRN（同源跨 σ 复用同一 ξ）在 iid 与 hist-block 两模式下均保持。
6. 与并行任务 `10-27-md-latent-flow-config-fix` 共享文件 `ec_mild_unbiased.py`：基于当前
   工作区状态（含其未提交 v4 公式改动）实施；改动前生成 iid 基线干跑；提交归属在 3.4
   向用户征求拍板（本任务 hunks vs config-fix hunks 分批或合并）。

## 验收标准（AC）

- **AC1 默认等价**：改动前后默认旗标 mini 干跑（`--subjects 1 --sources 8`，w500 ckpt）
  的 `sigma_matrix` / `raw_response_matrix` / `mse_curve` 逐位一致、`tau_global` 相等。
- **AC2 通道共享**：hist-block 全共享下 `set_source(0/1/127)` 后 `site._xi` 逐元素相同
  （`torch.equal`，且等于被试循环外全局构造的切片）。
- **AC3 被试共享**：两被试（或两次独立同参数重建）的 ξ 逐元素相同（`torch.equal`）。
- **AC4 分组结构**：`ξ == ξ[:, ::25].repeat_interleave(25, dim=1)`（组内 25 点恒等、
  共 15 组）；组代表向量 15×15 Pearson |offdiag| < 0.01（N=200,000 seeded sanity）。
- **AC5 隔离**：非历史段噪声逻辑未被改变——diff 审查上述路径零改动 + `_mu_bias is None`
  断言保留 + CRN 断言（同源扫 24 σ 期间 ξ digest 不变）。
- **AC6 配置面**：5 旗标默认 = iid 原行为；hist-block 口径断言生效（`hist_len==obs_len`、
  整除）；`--no-hist-share-channels` / `--no-hist-share-subjects` 退化路径可构造（键控
  种子公式在档）。
- **AC7 交付**：验证脚本五项全 PASS 留任务档；`experiment.md` Run 块四要素齐全；
  payload hist-block 分支 `hparams`/`noise_model` 完整、iid 分支键面不变。

## 关键口径（现实现基线，侦察在档）

- 噪声生成：`SignalProbeSite.set_source`（`bisect_sigma.py:675-681`）按 `seed+source`
  逐源播种 ξ `(N=30, 375)`；同源跨 σ 复用；`perturb_channel`（`:186-212`）施加
  `x_pert[:,:,src] += σ·ξ`。
- 数据：`MildTrialDataset` → `x_obs=(30, 375, 128)`（trial/时间/通道）；`--obs-len`
  已被 `_caliber_asserts` 钉死 375。
- 代理脑 w500 fold0 ckpt 在档；本机环境已登记（torch 2.8.0+cu128、RTX 4080 SUPER）；
  代理脑为确定性采样 ⇒ ξ 是流水线唯一随机源。
