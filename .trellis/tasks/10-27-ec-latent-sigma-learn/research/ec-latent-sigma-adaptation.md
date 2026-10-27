# Research: 潜空间扰动 + 逐通道可学 σ 梯度优化 EC —— 项目适配结论 / 风险 / 可执行方案

- Query: 新路径「潜空间 NoiseLayer + 逐通道可学 σ + SGD×20epoch + λ(1−P)−H 修复型 loss」如何适配 npi 仓库 mild × latent-flow 线（θ 接口 / 数据口径 / 与信号空间 global-τ 管线契约关系 / 风险 / 首版落地方案）
- Scope: internal（代码 + spec + ckpt 实况）+ GPU 实测探针（只读，未改任何 repo 文件）
- Date: 2026-10-27
- 方法：通读 `oc_latent_flow.py:847-1193`、`ec_mild_unbiased.py`、`train_mild_depress.py`、`bisect_sigma.py`、`viz_mild.py`、design.md（本任务定稿权威）、`.trellis/spec/research/perturb-ec-pipeline.md` §8、`.trellis/spec/data/mild-depression.md`、`.trellis/research/environment.md`；并以 ec_mild_unbiased 同款装载路径在 fold0 test 首例（sorted(test)[0]，30 trial 全量）上跑了三轮只读探针（/tmp/probe{1,2,3}_*.py，未写 repo）：A) latent z 标度、B) σ 梯度量级、C) 计时/显存、D) CRN 恒等、E) 20-epoch SGD 动态模拟。

---

## 0. 实测探针摘要（本研究的决定性新证据）

环境：torch 2.8.0+cu128 / RTX 4080S 32GB / cudnn.deterministic=True / fp32；θ = `run_symw_fold0`（`--symw` 同款装载 `_model_args(True)`）；w500 口径 375/125/500、fold0 训练折归一化；n_mc=2（batch 60）为主配置。

| 探针 | 结果 |
|---|---|
| A. latent 标度 | z_obs(mu) 逐通道 std q0/50/100 = **0.553/0.607/0.720**（全局 0.613）；z_fc_target = 0.638/0.714/0.757。σ_max=0.48 ≈ **0.66–0.87× 逐通道 z std**（对照：信号空间 z-scored std≈1.03，0.48≈0.47×）。潜空间盒内相对幅度比信号空间更大，σ_max 不算过小 |
| C. 剂量响应 | 127 通道**联合**加噪 σ=0.48 时靶通道 P 仍 **0.985–0.990**（tgt0/tgt64）；P(σ) 全程 1.000→0.985，盒内 1−P ∈ [0, 0.015]——响应极平 |
| B. 梯度量级 | \|∂P/∂σ_i\|：σ=0.01 处 median≈0、max≈1e-4；σ=0.48 处 **median 3e-4 / q75 7e-4 / max 3.6e-3**。熵梯度 1/(127σ) = 0.787@0.01 → 0.0164@0.48，**全程比 pearson 梯度大 5–4 个数量级**。平衡点 σ*_med=1/(127λ\|∂P/∂σ\|) ≈ **31–1594（λ=1）≫ σ_max** ⇒ λ=1 时盒内无内部极小，存在解仅盒角 |
| E. SGD 20ep 动态 | **(lr=5e-3, λ=1)（design 默认）→ σ 全 127 通道收敛到 0.042 完全同值（零判别力；与熵 ODE 解析解 σ(n)=√(σ0²+2·lr·n/127) 吻合）**；(lr=0.5, λ=1) → **127/127 全钉上界 0.48**（零判别力）；(lr=0.5, **λ=30/100/300**) → interior **53/87/116**/127，λ=300 时 σ 分位 q0/25/50/75/100 = 0.059/0.150/0.254/0.397/0.480（判别结构出现）；(lr=1.0, λ=300) → interior 94、@lo 12（动态范围更大） |
| D. CRN | t 经独立 generator 固定后 origin 轨迹重复前向 **torch.equal 逐位恒等**（GPU 实测）——design §3.3 的缓存契约成立 |
| C. 计时/显存 | fp32 fwd+bwd（flow+decoder，batch 60）≈ **0.25 s/step**，peak **17.9 GiB**；bf16 ≈ 0.25 s、15.8 GiB（不省时）；**batch 120（n_mc=4×30）fp32 OOM**（32GB 卡）。20 epoch/靶 ≈ 5 s ⇒ 128 靶 ≈ **10.7 min/例**，8 例 ≈ 1.4 h，40 例 ≈ 7.2 h（对照 hist-block 主跑 6.5 min/例实测，subject_0.json elapsed_min=6.57） |
| bf16 对拍 | 同配置 bf16 autograd 的 \|∂P/∂σ\| 分位数与 fp32 一致（至 3–4 位）、P 差 1e-4 ⇒ bf16 骨干+fp32 σ/loss 结构数值可用（design §5 已定默认关，成立） |

**核心结论**：design §2.4 预留的「λ 重标定属主跑前拍板项」由实测升级为**硬门槛**——λ=1 在任何 lr 下都退化为零判别常行（近_init 均匀或全钉上界），可行窗 **λ≈100–300 × lr≈0.5–1.0**（fresh-ε 版本期望意义不变，仅轨迹噪声更大）。探针 E 即 interior 探针的现成模板。

---

## 1. 适配结论（逐项）

### 1.1 θ 代理脑选择与 ckpt 接口（研究问题 1）

**现役候选实况（文件已核存在）**：

| θ | ckpt 路径 | 构建旗标 | 状态 |
|---|---|---|---|
| **run_symw_fold0（建议首版默认）** | `/root/npi/outputs/10-25-latent-flow-exit-gnn-symw/run_symw_fold0/checkpoints/fold0_mild_flow_best_graph_mexhat.pt`（5.2MB） | `--symw` ⇒ `_model_args(True)` | 最近 hist-block 主跑（`outputs/10-27-ec-histblock-symw-f0/ec_histblock/`，8 例，provenance.ckpt_path 已核=此路径）所用；best 0.8054@E144 真收敛、幅度 0.858/0.848 三折稳健（context.md 路线表） |
| w500 基座 | `/root/npi/outputs/10-24-latent-flow-mild-w500/run_w500_fold0/checkpoints/fold0_mild_flow_best_graph_mexhat.pt`（+`fold0_mild_vae_best.pt`） | `_model_args(False)` | ec_mild_unbiased DEFAULT_CKPT（ec_mild_unbiased.py:120-121）；stage2 E150 顶格未收敛 caveat |
| exit-gnn | `/root/npi/outputs/10-24-latent-flow-exit-gnn/run_exitgnn_fold0/checkpoints/fold0_mild_flow_best_graph_mexhat.pt` | `--flow-exit-gnn` | 幅度欠校准回归 caveat |

**加载代码路径（零拷贝可复用，全部 file:line）**：
- 模型构建：`train_mild_depress.py:391-452 build_model_mild(device, args, stage=2, obs_len=375, forecast_len=125)`——graph_mode 走 mexhat，geometry 断言 `MILD_GEOMETRY_PATH = /root/autodl-tmp/Mild Depression/geometry_mild.pt`（train_mild_depress.py:129,400-415，E1..E128 名单断言，存在已核）。
- args 命名空间：`viz_mild.py:116-124 _model_args(flow_exit_gnn)`——mexhat/graph_k=2/alpha=0.5/gamma=0.1/gamma_m=1.0/beta_kl=1e-4。**symw 训练直用 FlowVelocityNetExitGNN（零架构差异，SymWAdam 仅 optimizer 尾部投影不进 ckpt）⇒ symw ckpt 用 `_model_args(True)` 装载**（ec_mild_unbiased.py:334-337 注释即此逻辑）。
- ckpt 装载：`viz_mild.py:127-134 load_flow_ckpt(model, ckpt)`——format v1 payload 解包 + `load_state_dict(strict=True)` + eval()。use_fc_decoder 全 mild ckpt 恒 False（train_mild_depress.py:424）⇒ 新脚本固定全窗 decoder 路径。
- **首版默认 θ 配置建议**：`--symw` 语义 + run_symw_fold0 ckpt + `_model_args(True)` + fold0；run_dir 段守卫（仿 ec_mild_unbiased.py:201-223 的 symw 分支：断言 ckpt 含 `10-25-latent-flow-exit-gnn-symw` 且 fold=0）。

### 1.2 数据口径（研究问题 2）

全部复用、零新实现：`load_manifest(MANIFEST_DEFAULT, fold)`（train_mild_depress.py:235-290，含 manifest SHA 断言 `4798790...5d881b`）；`compute_mild_train_norm(train_f, window_len=500)`（train_mild_depress.py:364-384，fold 键控 norm SHA 断言沿用 ec_mild_unbiased.py:104-109，fold0 = `d184e647...2758`）；`MildTrialDataset([file], (mean,std), obs_len=375, forecast_len=125, window_len=500)`（train_mild_depress.py:297-356）→ `x_obs = stack(t[:375])`、`x_fc = stack(t[375:500])`（ec_mild_unbiased.py:369-373 同款切片）。W500 硬三元组断言（ec_mild_unbiased.py:100,196-199）照抄。

**复用 vs 参数化**：8 例选择（fold0 test `sorted(fold_dict["test"])`，`--subjects`/`--subject-idx` 切片惯例）、30 trial（`--n-trials` 干跑参数）、`--seed 42` 沿用；靶通道集合首版 = 全 128 逐个循环（`--target-channel` 支持单靶干跑）。z_fc_target = `model._encode_obs(x_fc)`（oc_latent_flow.py:1108 同式）。

### 1.3 与现管线（信号空间 global-τ）的口径关系（研究问题 3，对照 perturb-ec-pipeline.md §8）

**可继承契约**：
1. σ 盒 [0.01,0.48] + init=下界（ec_mild_unbiased.py:112 惯例；bisect_sigma.py:82-83 同源）——数值域与现管线直接同型。
2. v4 公式 `−(ln σ* + ½ln 2πe) + 5/7`（ec_mild_unbiased.py:114-118；σ*∈盒 ⇒ EC∈[+0.0293, 3.9005] 全正（主会话 2026-10-27 对源码复核勘正；初稿 [1.471, 2.905] 系笔误））。
3. CRN 确定性：t generator 抽一次全程固定 ⇒ origin 逐位可复现（实测 §0-D）；origin/pert 同 t 配对 ⇒ 差异纯归 ε·σ。
4. 种子登记纪律：独立 `torch.Generator`（bisect_sigma.py:678-681 先例）或不落盘噪声数组的四前提配方重建（§8.4：同 seed/形状/设备/torch 版本）——design §3.3 已选「ε 逐步现抽 + 四前提登记」，注意 design §4 NoiseLayer 草案用 `randn_like`（全局流）——**建议改为可登记的独立 generator 或在 payload 如实登记全局流配方**（见风险 j）。
5. interior 率诊断口径（ec_mild_unbiased.py:598-607 的 n_at_min/max/interior/分位数块）。
6. 原子写 json/npz + 断点续跑 + BLAS 守卫 + outputs/<任务名>/ 纪律（environment.md）。

**必须新立（不可继承）**：
1. **语义差异 = 联合边际 vs 逐源隔离**：现管线每次只扰 1 源（σ*[a,b] 隔离剂量）；新方法 127 通道同时在场、每通道 σ_i 是「其余通道也在噪声中」的边际容忍度——跨通道噪声交互项混入边际梯度，与逐源 σ* 不等价。并置纪律 = 描述性并置不做等价判读（design §8-h 已同判）。
2. **潜空间 σ 标度差异**：σ 作用于 z（std≈0.55–0.72）而非 z-scored 信号（std≈1.0）——同数值 σ 在潜空间的相对幅度 ≈ 信号空间的 1.4–1.8×（0.48/0.61 vs 0.48/1.03）。**结论：无需 σ_ref 重标度**（design §1 已拍板不重标度、保持数值域可比；实测 per-ch std 范围紧 1.3× 也使相对化收益小，LatentProbeSite.absolute_sigma 先例 bisect_sigma.py:736-738,836-838 仅作参考），但 **z-std 诊断必须随 payload 登记**（design §5 payload 已列）。
3. **readout 语义**：pearson 软度量替代 MSE+τ+q ⇒ payload 全新 caliber 字段（无 tau_global/q 字段），method 串注明 single-step latent SGD pearson-entropy。
4. **方向 orientation**：优化产出 = 靶 b 对 127 源的 σ*_i ⇒ 落盘矩阵须**转置为房惯例行=源、列=靶**（perturb-ec spec §4 先例 `orientation` 字段），靶位 NaN/对角处理按 design §5。

---

## 2. 风险清单（a–g + 额外 i–l，逐项缓释）

| # | 风险 | 实测/判断 | 缓释 |
|---|---|---|---|
| a | z-score 后潜变量 std 与 σ 盒匹配 | **已实测**：z_obs 逐通道 std 0.553–0.720、z_fc_target 0.638–0.757；σ_max=0.48≈0.66–0.87× std——同量级、不算过小；真正问题是盒内 P 响应极平（1−P≤0.015） | 盒维持不动；缓存级输出 z-std 诊断入 payload（design §5 已列）；「σ_max 相对过小/过大」担忧解除，改判为 λ 标度问题（见 b） |
| b | λ=1 起步动态 | **已实测（决定性）**：λ=1 + lr=5e-3 → σ 全体均匀 0.042（零判别）；λ=1 + lr=0.5 → 全钉上界（零判别）；\|∂P/∂σ_i\| 比 1/(127σ) 小 5–4 个数量级、平衡点 ≫ 盒。可行窗 λ≈100–300 × lr≈0.5–1.0（λ=300/lr=0.5 时 interior 116/127、σ q0–q100 = 0.059–0.480） | ①冒烟 AC 保留「σ 有更新+诊断可判读」不预设 interior 比例（design §8-b），但**主跑前拍板必须换 λ/lr**（§3.6）；②epoch 0-1 即打印 σ 分位/@lo/@hi/interior + \|∂P/∂σ\| 分位 vs 1/(127σ) 对照行（探针 B 即模板）；③注记：λ 与 pearson 逐点计数 16×64=1024 的等效缩放语义（design §2.3）——换 λ 时与该注记联动登记 |
| c | pearson 梯度穿堂 | 冻结参数 requires_grad=False 但可训练前向**不可 no_grad/不可走 model.sample（@no_grad，oc_latent_flow.py:1155）**；fc_origin 须 no_grad+detach 缓存 | 构造检查：backward 后 σ.grad 非零（非靶位）、靶位恒 None/0；fc_origin 独立 no_grad 域 + `.detach()`（LatentProbeSite 同款 bisect_sigma.py:770-772 的 no_grad 教训：autocast 不禁 grad、24 次前向累积图 OOM 先例） |
| d | 内存（batch×n_mc×decoder 激活） | **已实测**：fp32 batch60（n_mc=2×30trial）fwd+bwd peak 17.9 GiB；batch120 fp32 **OOM**（32GB）；bf16 仅省 12%（15.8 GiB，激活主导） | 首版默认 n_mc=2×batch30（单 chunk/epoch）或 design 的 batch16×n_mc2（≈10 GiB）；n_mc=4 须 trial 分块（16×4=64 行/chunk）；可选后续优化：**逐通道解码**——ConvDecoder 逐通道独立（oc_latent_flow.py:699-752，C 折进 batch），靶读出只需解码靶通道 ⇒ decoder 计算理论 128×↓；但批形变化非逐位可比，须整 run 统一 decode 形状（origin 同形重算）且属口径变更须拍板 |
| e | 算力 20ep×40例×30trial | **已实测**：0.25 s/step（batch60 fp32）⇒ 20ep/靶 ≈5 s、128 靶/例 ≈10.7 min；8 例 ≈1.4 h、40 例 ≈7.2 h（对照 hist-block 6.5 min/例）——单步构造比 Euler-10 便宜但反向补回，总成本 ≈1.7× 现管线 | 冒烟计时外推入 experiment.md；主跑规模（8 vs 40 例）属拍板项；预估量级完全可行（nohup 一夜） |
| f | NaN 守卫 | fc 读出 nan_to_num（sample() 先例 oc_latent_flow.py:1188）+ loss/P/H 有限性断言；σ 盒投影天然防 σ≤0 进 log | design §5 已列；构造检查加「注入 NaN → 拒跑路径」断言 |
| g | bf16 与 σ/H fp32 混合精度边界 | **已实测**：bf16 骨干梯度分位与 fp32 一致、P 差 1e-4——数值可用；风险点在 σ·ε 极小时 bf16 加法下溢 | 首版默认 fp32（design §5 已定，bf16 可选不入验收）；若开 bf16：noise/z_pert 构造全程 fp32，仅 flow/decoder 调用进 autocast（探针 B-amp 即此结构），pearson/H/σ 恒 fp32 |
| i | **（额外）结构性零梯度通道**：\|∂P/∂σ_i\| q0=0.0000——mexhat 图先验（核零点 + K=2 消息传递局部性）下部分通道对靶 fc 影响严格≈0 ⇒ 这些 σ 任何 λ 下都钉上界 | EC 行将出现结构性 σ_max 平台 =「图先验不可达边」，与「弱影响边」在 EC 值上不可分 | payload/判读 caveat 登记；可另存 A_eff 可达集诊断（后续）；并置纪律描述性（§1.3-1） |
| j | **（额外）ε RNG 配方**：design §4 NoiseLayer 草案 `randn_like`（全局流）逐步现抽——复现依赖全局流状态与调用序，弱于 hist-block §8.3 独立 Generator 惯例 | 跨任务对拍时全局流配方脆弱 | 建议实现时 ε 用独立 `torch.Generator(device)`（seed+偏移正交登记，仿 `build_hist_block_xi` ec_mild_unbiased.py:262-282）；至少 payload `noise_model` 字段如实登记 RNG 配方 + 四前提 |
| k | **（额外）耦合语义登记**：潜空间扰动经 z_t 起点 + flow 条件两路进 GNN；decoder 的 z_obs 参数/skips 恒净（design §3.2 干净侧拍板）。实测补充：ConvDecoder 逐通道独立 ⇒ 靶通道解码只依赖靶通道 z_obs（本就无噪）——「decoder z_obs 用净缓存」与「用扰动缓存」对靶读出**逐位等价**，语义分叉仅是字面选择 | 语义与信号空间管线（encoder 全程重跑、skips 亦变）不同，须 caveat 登记 | design §3.2 已定干净侧；本等价性注记随档，构造检查加「净/扰动 z_obs 进 decoder 靶读出 allclose（fp32 逐位）」断言固化该语义 |
| l | **（额外）fresh-ε 方差**：design §3.3 拍板 ε 逐步现抽（梯度随机性来源）；探针 E 用固定 ε（更光滑）——期望动态不变，但 20 步内轨迹噪声更大、可能提前触界 | box 投影兜底触界；λ/lr 结论在期望意义下传递 | 冒烟同时输出逐 epoch \|σ−σ_prev\|（design §8-b 已列）；如判读受扰，`--eps-fixed` 旗标备选（口径登记） |

---

## 3. 可执行方案

### 3.1 文件布局与包体零改动判定

- 新文件唯一：`models/latent-flow-v1/ec_latent_sigma_learn.py`（BLAS 守卫头 + `sys.path.insert(VARIANT_DIR)` 惯例，仿 ec_mild_unbiased.py:56-74）。
- **包体 `latent_flow/` 零改动可行（判定：成立）**：所需接口全部现成公开——`model.encoder(x, return_skips=True)` / `model._encode_obs` / `model.flow(z_t, t, z_cond)` / `model.decoder(z_obs, z_fc_hat, skips=...)` / `viz_mild._model_args` / `viz_mild.load_flow_ckpt` / `train_mild_depress.{load_manifest, compute_mild_train_norm, build_model_mild, MildTrialDataset, _sha256_file}` / `latent_flow.training.seeding.set_seed`。NoiseLayer/单步构造/优化循环全部脚本内实现（design §3.2 复刻 oc_latent_flow.py:1111-1121）。无需向包内加任何东西。
- 输出：`outputs/10-27-ec-latent-sigma-learn/<run_id>/`（逐例 subject json + npz + 逐 epoch 诊断 jsonl，原子写）。

### 3.2 接口签名草案（与 design §4-§5 对齐）

```python
class NoiseLayer(nn.Module):
    # sigma: nn.Parameter (128,) init 0.01；mask buffer 靶=0 其余=1
    def __init__(self, n_channels=128, sigma_init=0.01, sigma_min=0.01, sigma_max=0.48): ...
    def forward(self, z_obs, eps):          # (B,128,16,64) × (B,128,16,64) → z_obs + (σ⊙mask)·eps
    @torch.no_grad()
    def project(self): ...                  # clamp_[0.01, 0.48]
    def entropy(self, target_ch): ...       # 非靶 127 通道 mean(½ln(2πeσ²))，排除靶（ln0=−∞）

def stage2_single_step(model, z_cond, z_fc_target, t):   # 复刻 :1111-1121；z_cond = 加噪侧
    # z_t=(1−t)z_cond+t·z_fc_target；z_fc_hat=z_t+(1−t)·flow(z_t,t,z_cond)

def build_origin_cache(model, x_obs, x_fc, *, t_seed):   # 冻结编码 + t 一次抽样 + origin fc 全程缓存
    # 返回 {z_obs, z_fc_target, skips, t, fc_origin (B,125,128)}（fc_origin no_grad+detach）

def optimize_target(model, cache, target_ch, *, lam, lr, epochs, n_mc,
                    batch_size, seed, device) -> dict:
    # 返回 {sigma (127,), conn (127,), P_trace, H_trace, sigma_traces (n_ep,127), diagnostics}
    # loss = lam·(1−P) − H；P = pearson scope=target（逐样本时间维、(n_mc·B) 平均，仿 _pearson_corr_fc eps=1e-8）

# main 循环：for sid → cache 一次 → for target in 128 → optimize_target →
#           σ*→EC_v4 行 → 转置成 [source,target] 落盘（行=源，orientation 字段）
```

CLI（design §6 基础上补实测驱动项）：`--mode {construct,smoke,main}`、`--ckpt`（默认 run_symw_fold0）、`--symw`（默认开 + run_dir 段守卫）、`--lam`（默认 1 保 design 起步档，**冒烟必跑 λ 梯子**）、`--lr`、`--epochs 20`、`--n-mc 2`、`--batch-size 30`（或 16）、`--pearson-scope {target,full}`、`--eps {fresh,fixed}`（默认 fresh）、`--bf16`（默认关）、`--subjects/--n-trials/--target-channel/--seed/--out-dir`。

### 3.3 构造检查清单（CPU 级，`--mode construct`，AC2 七项 + 本研补强）

1. σ 盒投影：构造越界更新 → clamp 回 [0.01,0.48]（含边界值本身）。
2. 熵解析梯度对拍：autograd ∂H/∂σ_i vs 解析 1/(127σ_i)（fp64 allclose）。
3. 靶通道零噪声：NoiseLayer 输出靶通道与输入 torch.equal；靶位 σ 无梯度（grad None/0）。
4. CRN：固定 t 两次 origin 前向 torch.equal 逐位恒等（CPU 断言；GPU 已实测过，见 §0-D）。
5. 单步构造复刻对拍：脚本 `stage2_single_step` vs 逐元素手写公式（小张量 fp64）allclose。
6. pearson 对拍：自实现 vs `scipy.stats.pearsonr` 逐样本（fp64）allclose；eps 分母护栏在常数序列上不 NaN。
7. 穿堂检查：backward 后非靶 σ.grad 全有限非零（合成可感场景）、全冻结参数 .grad 为 None；fc_origin 带 detach。
8. （补）λ(1−P)−H 驻点行为：合成 P(σ)（如 P=exp(−kσ²)）上验证 SGD 收敛到解析平衡点邻域、方向正确（σ*∝1/√λ 换 λ 斜率对拍）。
9. （补）NaN 守卫：注入 NaN fc → 断言拒跑路径触发。
10. （补）decoder 净/扰动 z_obs 等价性（风险 k）：靶读出 allclose（fp32 逐位）。

### 3.4 GPU 冒烟清单（2 例小 epoch，`--mode smoke`）

- `--subjects 2 --epochs 3 --target-channel {0,64}`：断言 σ∈盒、诊断块打印（σ 分位/@lo/@hi/interior、\|∂P/∂σ\| 分位 vs 1/(127σ) 对照、P/H、\|σ−σ_prev\|）。
- λ 梯子小试（本研究实证要求）：同 2 例 × λ∈{1,100,300} × lr∈{5e-3,0.5} 输出 interior 对照表（每格 <1 min）。
- 显存 peak < 24 GiB、逐例 < 5 min；CRN 恒等复验（GPU torch.equal）；json/npz 原子落盘、payload 键完整（hparams：lam/lr/epochs/n_mc/batch/eps 策略/t seed/sigma_bounds/scope/θ ckpt+SHA/fold/norm SHA/orientation/z-std 诊断/复现四前提串）。
- 同 seed 复跑 σ 逐位一致（fp32）。

### 3.5 验收标准建议

1. construct 10 项全绿；smoke 全部断言过 + λ 梯子对照表落 experiment.md。
2. 冒烟在选定 λ/lr 下 σ 分布非退化（interior>0 且非全同值——本研实测 λ=1 不满足此条，故此验收与 λ 拍板联动）。
3. 现役管线零影响：`git status` 仅新增 ec_latent_sigma_learn.py + outputs/；ec_mild_unbiased.py 与 latent_flow/ 零 diff。
4. 判读登记：联合边际语义 + 结构性 σ_max 平台 caveat + 潜空间标度注记随 payload。

### 3.6 主跑前须用户拍板项（实测数据支撑）

1. **λ 与 lr（最高优先）**：λ=1 实测不可用（零判别常行）；建议从 λ∈{100,300} × lr∈{0.5,1.0} 四格冒烟选定（探针 E：λ=300/lr=0.5 interior 116/127、λ=1 时 0 或全角）。λ 换档与 16×64 逐点计数缩放注记（design §2.3）联动登记。
2. θ 默认确认：建议 run_symw_fold0（现役 hist-block 主跑同源）；w500 备选。
3. ε 策略：fresh（design 拍板）vs fixed（CRN 更光滑）——默认 fresh，旗标备选。
4. n_mc/batch 组合：n_mc=2×batch30（17.9 GiB）或 batch16×n_mc4 分块；n_mc=4×batch30 不可（OOM）。
5. 逐通道解码优化（decoder 128×↓）是否启用：属口径变更（批形非逐位可比），默认不开。
6. 主跑规模：8 例（fold0 test，≈1.4–3 h）vs 40 例（≈7–15 h）。
7. 与 iid/hist-block 信号空间 EC 的并置口径：描述性并置（同 §1.3-1 纪律）。

## Caveats / Not Found

- 探针在 fold0 test 首例单被试、2 个靶通道（0/64）上测得；λ 可行窗的精确边界随被试/靶通道有波动（探针 E tgt64 λ=100 interior 75/127 vs tgt0 87/127），量级结论（λ≫1 必需）稳健。
- 探针用固定 ε；fresh-ε（design 拍板）的 20 步内轨迹方差未逐点模拟，期望动态不变（风险 l）。
- probe3 的 H 显示值含每项 +0.5 常数位移（显示笔误，不影响梯度/动态；解析值以 ½ln(2πeσ²)=−3.186@0.01 / +0.685@0.48 为准）。
- 未实测 w500/exit-gnn θ 下的梯度量级（λ 窗可能随 θ 平坦度漂移）；冒烟 λ 梯子以实际 θ 为准。
- 10-25-latent-flow-symw-f12 fold1/2 ckpt 存在但本任务 fold0 默认不涉及。
