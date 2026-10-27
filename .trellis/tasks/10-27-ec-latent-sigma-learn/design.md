# Design — EC 潜空间可学 σ 扰动（NoiseLayer·SGD·20epoch）首实现

> 任务：10-27-ec-latent-sigma-learn · study2 · 新路径（research/ec-latent-sigma-adaptation.md 适配研究随档）
> 交付边界：`models/latent-flow-v1/ec_latent_sigma_learn.py` + CPU 构造检查 + GPU 冒烟 2 例 × 3 epoch；全量主跑另立。

## 1. 定位与与现管线的关系

| 维度 | 现役信号空间 global-τ（ec_mild_unbiased.py） | 本任务潜空间可学 σ |
|---|---|---|
| 扰动位置 | 信号空间 x_obs 源通道（z-score 后入模前） | 潜空间 z_obs（stage-2 入口、GNN 之前） |
| 搜索策略 | log 网格 [0.01,0.48]×24 + global-τ σ* 判定 | 127 通道 σ_i 联合梯度优化（SGD×20epoch） |
| 响应度量 | 靶通道 MSE 类响应 + τ 阈值 | 靶通道 fc 段 pearson 保真（无硬阈值，λ 软化） |
| 采样口径 | Euler-10（sample） | 单步构造（forward_stage2 口径，:1111-1121） |
| σ 语义 | 逐源隔离（每次只扰 1 源） | **联合边际**（127 源同时在场的逐通道边际影响） |
| EC 映射 | v4 = −(ln σ*+0.5·ln2πe)+5/7 | **同式直出**（σ* 边界域同 [0.01,0.48] ⇒ 数值域同型；**并置 = 描述性**：EC 值仅同 run 内序数意义，跨 λ 比较=ln λ 平移混淆、跨管线数值不可直比） |

继承契约：σ 盒边界/init、v4 公式、种子登记与确定性重建（四前提）、interior 率诊断、路径隔离。新立契约：修复型 loss 语义（§2）、CRN = t 固定（§3.3）、潜空间 σ 标度注记（σ 相对 z_obs 逐通道 std 的含义，随 payload 登记 z-std 诊断；不引入 σ_ref 重标度，保持与现管线 σ 数值域直接可比）。

## 2. 目标函数良构化（本设计的核心裁决）

### 2.1 字面「loss = pearson − H」退化证明

基本事实：∂P/∂σ_i ≤ 0（噪声↑ ⇒ fc 解相关），∂H/∂σ_i = 1/(127σ_i) > 0。

- 最小化 P−H：两项同推 σ↑ ⇒ 全通道顶 σ_max；
- 最大化 P−H：两项同推 σ↓ ⇒ 全通道钉死初值 σ_min，一步不动；
- 内部驻点要求 ∂P/∂σ_i = +1/(127σ_i) > 0，左边 ≤ 0 右边 > 0，无解。

### 2.2 修复型目标（用户拍板 λ 默认 1）

**最小化 L = λ·(1 − P) − H** ⟺ 最大化 λP + H：

| 项 | 梯度 | 方向 |
|---|---|---|
| 保真项 λ(1−P) | ∂/∂σ_i = λ·\|∂P/∂σ_i\| ≥ 0 | 梯度下降推 σ↓（噪声伤输出时压回） |
| 容量项 −H | ∂/∂σ_i = −1/(127σ_i) < 0 | 推 σ↑（不伤输出的熵白拿） |

平衡点 **σ*_i ≈ 1/(127·λ·|∂P/∂σ_i|)**：影响力强 ⇒ σ* 小 ⇒ EC_v4 大（与 v4「σ* 小=连接强」同向）；无影响 ⇒ σ*→σ_max 钉上界（EC 地板 +0.029）；极强 ⇒ 钉下界（天花板 3.90，与网格档同域）。稳定性：−ln σ 凸 + 解相关随 σ 饱和 ⇒ 稳定内部极小（另一修复 min(P+H) 的盒内驻点为局部极大、收敛到顶点，弃用——论证随档）。

### 2.3 H 解析式与梯度回传

h(σ_i) = ½ln(2πeσ_i²) = ½ln(2πe) + ln σ_i；H = (1/127)·Σ_{i≠靶} h(σ_i)（**靶通道必须排除**：σ_靶≡0 时 ln0=−∞）；∂H/∂σ_i = 1/(127σ_i) 纯解析，σ 为叶参数，autograd 从 H 项直达，与 pearson 项梯度（重参数化 z̃=z+σ⊙ε ⇒ ∂z̃/∂σ_i=ε_i，穿堂流过冻结 stage-2/3）在 σ_i 上相加，一步 SGD 合并。每通道 16×64=1024 点的联合熵计数仅等效缩放 λ，结构不变（口径注记并入 λ 语义，不做数值放大）。

### 2.4 λ ↔ τ 对应与标定留白

λ↑ ≡ τ↓（更不容忍输出变化）⇒ 整体 σ*↓。λ=1 为起步档（用户拍板）；每 epoch 输出 interior 率（现管线口径），λ 重标定属主跑前拍板项，本任务只暴露诊断不搜索。

## 3. 架构接缝

### 3.1 冻结与梯度穿堂

`model.eval()` + 全参数 `requires_grad_(False)`；可训练前向**不进 no_grad**（梯度须经冻结权重回传到 z_noisy→σ）。fc_origin 侧全程 no_grad + 缓存（常数参考，pearson 梯度只走 fc_pert 支）。z_obs/z_fc_target 由冻结 encoder 编码，全程缓存。

### 3.2 单步构造复刻（非 Euler-10）

按 oc_latent_flow.py:1111-1121 逐式复刻为脚本内 `stage2_single_step(model, z_cond, z_fc_target, t)`：`t_e=t.view(-1,1,1,1)`；`z_t=(1−t_e)z_cond+t_e·z_fc_target`；`z_fc_hat=z_t+(1−t_e)·flow(z_t,t,z_cond)`。mild 线 symw/exit-gnn 的图混合在 `model.flow` 内部 ⇒ 对 z_cond 先加噪即满足「GNN 之前」。**decoder 干净侧语义**：全窗解码器路径下 decoder 的 z_obs/skips 参数用未加噪缓存（fc_pert 与 fc_origin 差异只经 z_fc_hat 传导，归因干净）；fc-only 解码器无此分叉。构造检查断言：同一 (z_cond, z_fc_target, t) 注入下脚本复刻与逐元素手写公式逐位恒等（注：forward_stage2 不暴露 z_fc_hat，属脚本公式 vs 手写公式的自参照对拍，局限随档）。

### 3.3 CRN 与缓存（潜空间口径的新契约）

单步构造唯一随机源 = t（z_fc_target=enc(x_fc) 确定性、无 x0 项——与标准 FM 采样不同，本构造是 z_obs→z_fc_target 线段上的单步）。**t 逐样本由 generator(seed) 抽一次并全程固定** ⇒ fc_origin 确定性可全程缓存；origin/pert 同 t ⇒ CRN 配对，pearson 差异纯归噪声 ε 与 σ。ε 逐步现抽（梯度随机性来源；逐位复现依赖同 seed/形状/设备/torch 版本四前提，与现管线契约同型）。缓存三级：z_obs/z_fc_target/skips（编码级，全程一次）、t（全程一次）、fc_origin（全程一次）；每 step 只跑一条 fc_pert 前向+反向。

### 3.4 数据口径与装载链（research 适配已核实，GPU 探针实测随档）

- **θ 装载（零拷贝复用）**：`train_mild_depress.build_model_mild(device, _model_args(True), stage=2, obs_len=375, forecast_len=125)` + `viz_mild.load_flow_ckpt(model, ckpt)`（ec_mild_unbiased.py:334-338 同款）；默认 ckpt = `outputs/10-25-latent-flow-exit-gnn-symw/run_symw_fold0/checkpoints/fold0_mild_flow_best_graph_mexhat.pt`（现役 symw，run_dir 段守卫仿 ec_mild_unbiased.py:201-223）；mild 全 ckpt `use_fc_decoder=False` ⇒ 固定全窗 decoder 路径；geometry_mild.pt 断言沿用（train_mild_depress.py:129,400-415）。
- **数据（全复用）**：`load_manifest`（SHA 断言 `4798790...5d881b`）+ `compute_mild_train_norm(train_f, 500)`（fold 键控 norm SHA，fold0=`d184e647...2758`）+ `MildTrialDataset`（obs 375/fc 125/window 500）；W500 硬三元组断言照抄；8 例 = fold0 test `sorted()[…]` 惯例 + 30 trial。
- **z_fc_target** = `model._encode_obs(x_fc)` 的 mu（oc_latent_flow.py:1108 同式）。
- **ε RNG（research 风险 j 采纳）**：ε 与 t 各用独立 `torch.Generator(device)`，**每 (被试,靶) 新建**（seed+键控偏移，仿 build_hist_block_xi :275-280），λ 梯子各格天然正交、payload 配方可登记；`--eps {fresh,fixed}` 默认 fresh（fixed 为 CRN 光滑备选，口径登记）。

## 4. NoiseLayer 模块

```python
class NoiseLayer(nn.Module):
    # n_channels=128, target_channel, sigma_init=0.01, sigma_min=0.01, sigma_max=0.48
    # mask: buffer, 靶=0 其余=1；sigma: nn.Parameter (128,) init 0.01
    def forward(z_obs):                    # (B,128,16,64)
        eps = torch.randn_like(z_obs)      # 逐点 iid（R1.7）
        gain = (sigma * mask).view(1,-1,1,1)
        return z_obs + gain * eps          # 重参数化，靶通道恒零
    @torch.no_grad() project():            # 投影 SGD 盒约束 clamp_[0.01,0.48]
    def entropy():                         # 127 通道 mean(0.5·ln(2πe)+ln σ)，排除靶
```

靶通道 σ 项：mask 恒 0 ⇒ 无梯度（单参数向量下靶元素 grad=0 而非 None，行为等价）⇒ SGD 不动；H 与 EC 输出均排除之。逐通道「独立可优化」= 单参数向量 elementwise 更新，无耦合（如需逐通道 lr 可切 param_groups，默认不开）。

## 5. 优化循环与数据流

**组织单位 = 被试 × 靶通道**（与现管线逐被试 EC 矩阵口径一致）：被试缓存一次（编码/t/fc_origin），128 靶逐个顺序优化（每靶独立 NoiseLayer 从 σ=0.01 重启），σ* 行转置为房惯例 **行=源、列=靶**（orientation 字段登记）后拼 EC 矩阵。实测 5 s/靶 ⇒ 128 靶 ≈10.7 min/例（单靶不做批打包已够廉价；多靶批打包留作后续优化）。

```
冻结+构建+装载（build_model_mild(_model_args(True)) / load_flow_ckpt / norm+manifest SHA 断言；cudnn.deterministic=True + benchmark=False + set_seed——ec_mild_unbiased.py:310-311,339 同款，AC6 复现依赖）
→ 被试级缓存: encode x_p→mu_p(skips), x_f→mu_f；t~U(0,1) 独立 generator 抽一次；origin 单步+decode→fc0 缓存(no_grad+detach)；z-std 诊断
→ for target_ch in range(128):
    noise = NoiseLayer(target_ch)（σ init 0.01）
    for epoch in range(20):
        for chunk in trials(分块):
            ε ~ 独立 generator（fresh 逐步现抽；fixed 则首抽复用）
            zc = chunk.repeat_interleave(n_mc)                    # MC 折叠进 batch 维
            z_noisy = noise(zc, ε)                                # 唯一梯度入口
            z_hat_p = stage2_single_step(model, z_noisy, rep(zt), rep(tb))
            fc_p = decode_fc(model, rep(zb), z_hat_p,
                            skips=sk.view(B,C,-1,-1).repeat_interleave(n_mc,dim=0).flatten(0,1))
            # skips 原形状 (B·C,ch,T')：naive repeat_interleave 会打乱 (b,c) 配对且无形状报错（review B MAJOR）——
            # 必须经 view(B,C,·,·) 维度感知 repeat；构造检查⑪锚点断言兜底
            P = pearson_fc(fc_p[:,:,[t_ch]], rep(fc0)[:,:,[t_ch]])    # scope=target；pearson_fc 返回 P 本身（≠_pearson_corr_fc 的 1−P 口径）
            H = noise.entropy()                                   # 127 通道（排除靶）
            loss = lam*(1-P) - H
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); noise.project()
        诊断: σ 分位/@lo/@hi/interior 率、|∂P/∂σ| 分位 vs 1/(127σ) 对照行、P/H、|σ−σ_prev|、逐靶 |∂P/∂σ| q0=0 掩码（结构性零梯度通道诊断，随 payload）→ 轨迹 jsonl
→ σ*→EC_v4 = −(ln σ*+0.5·ln 2πe)+5/7（σ*∈[0.01,0.48] ⇒ EC∈[+0.0293, 3.9005]；靶位 NaN）
→ 落盘: EC 矩阵（行=源）+ payload {seed, ε 配方(seed+偏移/策略), t seed, lam, lr, epochs, n_mc,
   batch, sigma_bounds, targets, scope, θ ckpt+SHA, fold, subjects, norm SHA, orientation,
   z-std 诊断, method="single-step latent SGD pearson-entropy", 复现四前提串}
```

pearson_fc：逐 (样本,通道) 时间维 pearson 再平均（仿 `_pearson_corr_fc`，eps=1e-8 分母护栏）；`--pearson-scope {target,full}` 默认 target。NaN 守卫：fc 读出 `nan_to_num`（sample() 先例）+ loss/P/H 有限性断言 + 注入 NaN 拒跑路径（构造检查⑨）。autocast bf16 为 CLI 可选项（默认关；实测 bf16 梯度与 fp32 一致至 3–4 位，σ·ε 构造全程 fp32，仅 flow/decoder 进 autocast）。

## 6. 代码布局与 CLI

单文件 `models/latent-flow-v1/ec_latent_sigma_learn.py`（变体层新增；BLAS 守卫 + `sys.path.insert(VARIANT_DIR)` 头部，仿 ec_mild_unbiased.py:56-74；**latent_flow/ 包体零改动成立**——research §3.1 判定，所需接口全部现成公开）。CLI：`--mode {construct,smoke,main}`、`--ckpt`（默认 run_symw_fold0）、`--symw`（默认开 + run_dir 段守卫）、`--fold`、`--subjects`、`--n-trials`、`--target-channel`（单靶干跑；缺省 = 全 128 循环）、`--lam`（默认 1 = 用户起步档拍板；冒烟必跑 λ 梯子，见 §8-b）、`--lr`（默认 0.5，research 可行窗内；5e-3 为对照档）、`--epochs 20`、`--n-mc 2`、`--batch-size 30`、`--pearson-scope {target,full}`、`--eps {fresh,fixed}`、`--bf16`（默认关）、`--seed 42`、`--out-dir`。输出落 `outputs/10-27-ec-latent-sigma-learn/<run_id>/`（原子写）。

## 7. 并行/效率设计（首版范围；实测数字来自 research §0-C）

- 向量化：ε 单 kernel 全张量 + σ 广播 view(1,C,1,1)，无逐通道 Python 循环（前提性优化）。
- n_mc 份独立 ε `repeat_interleave` 折进 batch 维（每 step 梯度 SNR ∝ √n_mc）。
- 被试级三级缓存（编码/t/fc_origin 全程一次；128 靶共享）⇒ 每靶仅 20 epoch × 单步前向+反向。
- 实测基准：fp32 batch 60（n_mc=2×30trial）fwd+bwd ≈0.25 s/step、峰值 17.9 GiB（32GB 卡内安全）；**batch 120（n_mc=4×30）fp32 OOM** ⇒ n_mc=4 须 trial 分块；bf16 仅省 12%（15.8 GiB，激活主导）不省时。量级：20ep/靶 ≈5 s ⇒ 128 靶 ≈10.7 min/例；8 例 ≈1.4 h、40 例 ≈7.2 h（对照 hist-block 主跑 6.5 min/例 ⇒ 总成本 ≈1.7×）。
- 后续扩展（不在首版）：多靶批打包（σ 参数升 (T,128)、样本副本 ×T、按副本 gather 掩码）、逐通道解码（ConvDecoder 逐通道独立 ⇒ 靶读出只解码靶通道，decoder 计算理论 128×↓；批形非逐位可比属口径变更须拍板）、DDP/按靶分片。

## 8. 风险与缓释（已合并 research 实测结论；全文见 research/ec-latent-sigma-adaptation.md §2）

| # | 风险 | 实测/判断 | 缓释 |
|---|---|---|---|
| a | 潜空间 z std 与 σ 盒匹配 | **已实测解除**：z_obs 逐通道 std 0.553–0.720（中位 0.607），σ_max=0.48≈0.66–0.87×std 同量级；真正问题是盒内 P 响应极平（σ=0.48 时 127 通道联合加噪下靶 P 仍 0.985–0.990） | 盒维持不动；z-std 诊断入 payload |
| b | **λ=1 动态（硬门槛，最高优先）** | **已实测（决定性）**：\|∂P/∂σ_i\|（σ=0.48 处 median 3e-4）比熵梯度 1/(127σ) 小 σ→0 端 ~4 个量级、σ_max 端 ~2 个量级（0.0164/3e-4≈55×）⇒ 平衡点 ≫ 盒（max 梯度 3.6e-3 ⇒ σ*≈2.2>0.48） ⇒ λ=1 任何 lr 下零判别（lr=5e-3 → σ 全体均匀 0.042 与熵 ODE 解析解吻合；lr=0.5 → 127/127 全钉上界）。可行窗 **λ≈100–300 × lr≈0.5–1.0**（λ=300/lr=0.5：interior 116/127、σ q0–q100=0.059–0.480） | ①λ 默认仍 1（用户起步档拍板），但**冒烟强制跑 λ 梯子 {1,100,300}×lr{5e-3,0.5,1.0} 出 interior 对照表**，λ/lr 终值主跑前拍板（§10）；②epoch 诊断含 \|∂P/∂σ\| 分位 vs 1/(127σ) 对照行（探针 B 模板）；③λ 换档与 §2.3 的 1024 点计数缩放注记联动登记 |
| c | 穿堂梯度被 no_grad/detach 误断（且不可走 model.sample——@no_grad 装饰） | 判断 | 构造检查⑦：backward 后非靶 σ.grad 有限非零、全冻结参数 grad=None；fc_origin 独立 no_grad+detach（autocast 不禁 grad 的 OOM 先例随档） |
| d | 显存 | **已实测**：n_mc=2×30trial fp32 峰值 17.9 GiB ✓；n_mc=4×30 OOM | 默认 n_mc=2；n_mc=4 走 trial 分块 |
| e | 算力 | **已实测**：0.25 s/step ⇒ 8 例≈1.4 h / 40 例≈7.2 h | 冒烟计时外推入 experiment.md；规模拍板 |
| f | NaN/发散 | 判断 | nan_to_num + 有限性断言 + 注入 NaN 拒跑路径（构造检查⑨） |
| g | bf16 数值 | **已实测**：梯度与 fp32 一致至 3–4 位 | 默认 fp32；bf16 可选不入验收 |
| h | 联合边际 vs 逐源隔离语义 | 判断 | payload/判读 caveat；描述性并置纪律 |
| i | **结构性零梯度通道**（mexhat 图先验不可达边） | **已实测**：\|∂P/∂σ_i\| q0=0.0000 ⇒ 这些 σ 任何 λ 下钉上界，EC 值上与弱影响不可分 | caveat 登记（σ_max 平台=图先验不可达边）；A_eff 可达集诊断留后续 |
| j | ε RNG 配方（全局流弱复现） | 判断 | 独立 Generator（seed+正交偏移登记，§3.4）；payload 如实登记配方+四前提 |
| k | decoder 净/扰动 z_obs 语义 | **已实测**：ConvDecoder 逐通道独立 ⇒ 靶读出两者逐位等价（语义分叉仅字面） | 沿 §3.2 干净侧拍板；构造检查⑩ 固化等价断言；与信号空间管线（encoder 重跑 skips 亦变）差异 caveat 登记 |
| l | fresh-ε 轨迹方差（探针用固定 ε） | 判断（期望动态不变） | box 投影兜底触界；逐 epoch \|σ−σ_prev\| 诊断；--eps fixed 备选 |

## 9. 兼容性与回滚

新增单文件，不触 ec_mild_unbiased.py 与 latent_flow/ 包体任何路径 ⇒ 现役管线零影响（research §3.1 判定包体零改动成立，所需接口全部现成公开）；回滚 = 删文件 + 删 outputs 目录。

## 10. 主跑前须用户拍板（本任务只暴露不决策；research §3.6 实测数据支撑）

1. **λ 与 lr（最高优先）**：λ=1 实测不可用（零判别常行）；建议从冒烟 λ 梯子 {1,100,300}×lr{5e-3,0.5,1.0} 对照表选定（探针 E：λ=300/lr=0.5 interior 116/127、lr=1.0 interior 94 动态范围最大）。λ 换档与 §2.3 逐点计数缩放注记联动登记。
2. θ 默认确认：建议 run_symw_fold0（现役 hist-block 主跑同源）；w500 备选。
3. ε 策略：fresh（默认，design 拍板）vs fixed（CRN 光滑）。
4. n_mc/batch：n_mc=2×batch30（17.9 GiB）默认；n_mc=4 须分块。
5. 逐通道解码优化（decoder 理论 128×↓）：批形非逐位可比，属口径变更，默认不开。
6. 主跑规模：8 例（≈1.4–3 h）vs 40 例（≈7–15 h）。
7. 与 iid/hist-block 信号空间 EC 的并置口径：描述性并置（§8-h 纪律）。
