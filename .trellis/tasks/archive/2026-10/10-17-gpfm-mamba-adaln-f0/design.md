# design.md — v4a-fmt adaLN 调制归一化变体（A 案 scale/shift）

> 依据：`research/adaln-adaptation.md`（1.0c 适配研究，结论与本 design 一致处不重复论证，锚点见研究 §9）；用户 Q1 拍板 A 案（2026-10-17）。

## §0 一句话

v4a-fmt 的 5 个 LayerNorm 位全部换成 `modulate(LN(x), s(t), b(t)) = (1+s(t)) ⊙ LN(x) + b(t)`，`(s,b)` 由 `time_embedding(t,16)` 经逐 site `SiLU→Linear(16,124)`（零初始化）生成——初始前向与 v4a-fmt 基线**逐位一致**，唯一实验变量 = norm 仿射从常量变为 t 条件（参数 130,366→140,906，+10,540 / +8.08%）。

## §1 改动面

### 1.1 v1 共享库 default-off 钩子（三文件 + 两构造点）

| 文件 | 改动 | 默认分支 |
|---|---|---|
| `models/gpfm-mamba-v1/mamba.py` | `MambaBlock.__init__` 加 `cond_adaln: str = "off", mod_dim: int \| None = None`；启用（`"scale_shift"`）时**在 `with torch.random.fork_rng(devices=[]):` 内**建 `self.mod = nn.Sequential(nn.SiLU(), nn.Linear(mod_dim, 2*d_model))` 并零初始化末层（weight=0, bias=0）——RNG 中性构造（见下契约行）；`forward(self, x, mod=None)`——`mod is None` 走原式 `x + self.mamba(self.norm(x))`（逐位不变），非 None 走 `x + self.mamba(modulate(self.norm(x), s, b))`（s,b 由 `self.mod(mod)` 自产切分）；`build_mamba_backbone` 尾部 kwargs 透传两参（v4b 六位置参调用向后兼容），返回仍 `nn.Sequential`（键名不变） | 零漂移 |
| `models/gpfm-mamba-v1/model.py` | `VelocityNetwork.__init__` 加 `cond_adaln: str = "off"`（排在 norm_type 后）；启用时**同样 fork_rng 内**建 `self.head_mod = nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, 2*d_model))`（零初始化）并透传 backbone；`forward` 签名**不变**——adaLN 分支：`temb = time_embedding(t, self.time_emb_dim)` → `for blk in self.blocks: h = blk(h, temb)`（Sequential 不能直传额外参，块循环；默认分支保留原 `self.blocks(h)` 单行）→ head 位 `self.head(modulate(self.head_norm(h), s_h, b_h))` | 零漂移（不建任何新**参数/子模块**——state_dict 键与结构零漂移；`self.cond_adaln` 模式字符串属性除外） |
| `models/gpfm-mamba-v1/train.py:279` / `sample.py:53` 后 | 各加一行 `cond_adaln=str(getattr(cfg, "cond_adaln", "off"))` | 旧 config 无字段 → "off" → 零漂移 |

- **RNG 流保持契约（review 双 Mandate 共同 Major，采纳 fork_rng 方案）**：全部调制 MLP 的构建必须包在 `torch.random.fork_rng(devices=[])` 内——`nn.Linear` 构建期 reset_parameters 无条件消耗全局 RNG，朴素内联构建会使同 seed 下 block i+1..3 与 head 的共享权重相对基线整体后移（零初始化不归还随机数）。fork_rng 隔离随机流 ⇒ `cond_adaln="off"` 与 `"scale_shift"` 同 seed 下**共享权重逐位相同** ⇒ 单测「同 seed 双构建逐位一致」成立，且 run 级兑现：seed 1234 的 adaLN 主 Run 初始权重 = v4afmt-main-01 初始权重，epoch-0 step-0 损失逐位相等，其后分叉恰为处理效应（真单变量）。已核：train.py/sample.py 均 CPU 构建后 `.to(device)`、loader 用独立 Generator，fork_rng 无交互破坏。

- **`modulate(x, s, b)`**：模块级函数（DiT 官方同式 `x*(1+s)+b`，s/b 形状 (B,2d) 切分后 (B,1,d) 广播）——放 mamba.py，model.py head 位复用。
- **否决 `norm_type="adaln"`**：adaLN 需 forward 期条件张量，与 build_norm 纯 nn.Module 钩子正交；adaLN 变体底层 norm **仍是 LayerNorm**（`norm_type="layernorm"` 保持），两字段两维度。
- **config 字段**：`cond_adaln: str = "off"`，取值 `"off" | "scale_shift" | "zero"`（枚举字符串，先例 lr_schedule/kernel_t；`"zero"` 本任务不启用、值域预留，非法值构建期 ValueError——先例 build_norm）。当前实现仅接 `"off"/"scale_shift"` 两支，`"zero"` 传入报 NotImplementedError（不实现未拍板路径）。
- **state_dict 键**（与 `blocks.{i}.norm.*`/`head_norm.*` 对仗）：`blocks.{i}.mod.1.{weight,bias}`（(124,16)/(124,)，i=0..3）+ `head_mod.1.{weight,bias}`。EMA/save_ckpt 全键泛型——零改动。
- **评估链路零改动**：`evaluate.py` 经 load_eval_model 复用（config_hash 严格校验天然隔离跨变体 ckpt）；`evaluate_internal_val`/`EulerSampler` 调用签名不变；`fc_metric.py` 不构建模型。CUDA fast path 零风险（调制不触碰 mamba_ssm 内核，输入形状/dtype/连续性不变）。

### 1.2 新薄变体目录 `models/gpfm-mamba-v4a-fmt-adaln/`

- **config**：v4a-fmt config 完整副本，唯差异字段：

| 字段 | v4a-fmt | 本变体 |
|---|---|---|
| `cond_adaln` | （无字段） | `"scale_shift"` |
| `variant_name` | `gpfm-mamba-v4a-fmt` | `gpfm-mamba-v4a-fmt-adaln`（checkpoint 主名） |
| `outputs_root` | `.../gpfm-mamba-fmt-f0` | `/root/npi/outputs/gpfm-mamba-adaln-f0`（新任务根） |
| `split_manifest_override` | v4a-fmt splits/ | 本目录 `splits/fold0_status_ok_seed42_val4.json`（首跑固化，与 v4a-fmt **逐字节一致**——同参数生成，断言） |

  其余逐字段一致（`norm_type="layernorm"` **保持**、bottleneck_rank=16、d_model=62、fs=250、period_p=25、gen_lowpass_hz=None、seed 1234、早停 10、恒 lr、EMA 0.999、batch 256）。
- **wrapper 三要素**（train/evaluate/fc_metric，v4a-fmt 先例）：config 单遮蔽（model/mamba 解析到 v1）+ `RUN_PREFIX = "v4afmt-adaln-"` 前缀守卫 + importlib 独立模块名加载 v1 入口；**无 --rank**（r=16 固定，wrapper 比 v4a-fmt 简一档）；fc_metric 注入 `--outputs-root` 新根。
- **前缀嵌套已知局限（review Minor，保留 D3 并登记）**：`v4afmt-adaln-` 嵌套于 v4a-fmt 的 `v4afmt-` 之下——若误经 v4a-fmt wrapper 传 adaln run_id，其守卫放行（晚失败：独立 outputs 根 FileNotFoundError + ckpt 名不同 + config_hash 校验三重兜底，非静默）。本变体 wrapper docstring 登记「`v4afmt-adaln-*` 归本任务根」；B3 反例测试含 --resume 派生名。
- **BLAS 护栏块**照抄（environment.md 惯例）。

## §2 语义与判读登记（design 级预登记，experiment.md 随附）

1. **t 双注入不对称**：输入侧 temb 仅 future 128 位置（现状），调制侧 (s,b)(t) 全 512 位置（含 past 段）——DiT 同款结构（timestep 既拼 token 也驱动 adaLN）。past 表征变为 t 依赖：与采样语义自洽（EulerSampler 每步对完整序列重新前向，v_θ 是 (x_past, x_t, t) 纯函数，无跨步缓存）。
2. **恒等初始化**：调制 MLP 末层 weight=0/bias=0 ⇒ s=b=0 ⇒ `(1+0)⊙LN(x)+0` 浮点逐位等于 LN(x)——**epoch-0 前向与基线逐位一致**（配 §1.1 fork_rng 契约，run 级同 seed 初始权重 = 基线），单测断言（同 seed 双构建对比式，先例 test_fmt_hooks.py ②；断言用 `torch.equal` 值相等语义——`+0.0` 会把 LN 输出的 −0.0 映射为 +0.0，值等字节异，review Suggestion）。
3. **EMA 影子滞后（review 定量化）**：930 窗/例 × 34 训练例 = 31,620 窗，batch 256 drop_last ⇒ **123 step/epoch**；EMA τ≈1000 step ≈ **8.1 epoch**——影子权重中调制参数从 0 起步滞后原始权重，早停 10 下 best 若落在 ~16 ep 内，EMA best ckpt 的调制可能未充分生效。**阴性预案（预登记）**：若结论为「无改善」且 best epoch ≲ 2×8.1≈16，追加一次**非 EMA last.pt** 补充评估（登记为补充 Run，不替换正式口径），防 EMA 滞后掩盖晚期生效的调制收益。t_loc=-0.8 偏态下 (s,b)(t) 在稀有 t>0.7 区为有界外推（Lipschitz ∝ ‖W‖，与输入侧 temb 同风险类）——判读留意 Euler 末端步行为。
4. **不做特殊处理**：weight_decay 对调制 W 同等对待（向恒等的轻微先验，登记）；grad_clip 全局范数语义不变。

## §3 评估口径（与 v4a-fmt 完全一致，复用不新立）

- Q1 质量 + 指标1 FC（fc_metric 新根）+ 指标2 视觉（250 Hz 标秒）+ 指标3 PSD（判读窗 1–30 Hz mean |Δlog10| ≤0.5、40–125 Hz 照登）；>40 Hz 功率占比观察（psd.npz 补算，脚本从归档任务**复制到本任务 `research/psd_band_metric3.py`** 改 run 路径后用——禁在归档目录内增改文件，review Suggestion）。
- **GC 阻塞延续**：`evaluate --skip-gc-only`，产物零 GC 对比文件（逐文件核验）。
- 对比表：`v4afmt-adaln-main-01` vs `v4afmt-main-01`（同 manifest/seed/纪律）——**单变量差异声明**：norm LayerNorm→adaLN(t)（5 位，A 案）+ 参数 130,366→140,906（+10,540）；结论标注「前置验证 · fold0 · 单 seed · 单次运行」（规范标签，review Suggestion）。**判读规则预登记**：单 seed 不设数值阈值，差异量级与既有跨变体单 seed 差异（fmt-f0 对比表）同框做方向性判读，防事后过度解读（review Suggestion）。

## §4 风险与回滚

| 风险 | 触发信号 | 处置 |
|---|---|---|
| 钩子改坏旧变体 | 六变体 hash pin 漂移 / mod=None 分支前向 ≠ 改前 | revert v1 三文件钩子，重设计 |
| adaLN 分支数值/结构错 | 恒等初始化等价断言红 / state_dict 键缺 | 修 §1.1 实现；对照研究 §4.3 键表 |
| dataloader 契约不符 | smoke 窗数/形状与 v4a-fmt 冒烟记录不符 | 停训练修接入层（数据本体禁改） |
| 主 Run 中断 | last.pt 存在 | --resume 同 run 目录续跑（前缀守卫兼容） |
| 训练不稳定（A 案概率低） | loss 尖峰 / val 崩 | 登记现象；如系统性失稳回炉讨论 B 案（须用户拍板，验收②改写） |

## §5 Run 纪律

experiment.md 先登记目的+配置再跑（reproducibility 契约）：smoke（每类 2 例 1ep + dataloader 契约核对）→ bench 1ep（显存/时长外推）→ 主 Run（nohup，≤100ep 早停 10，`--run-id v4afmt-adaln-main-01`）→ 正式评估。全程 fold0 单 seed 1234，不越 5-fold 门禁。
