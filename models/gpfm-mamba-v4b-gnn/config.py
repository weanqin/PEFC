"""gpfm-mamba-v4b-gnn 唯一配置源（薄变体，完整 config 副本，v4b d_ch=1 基座 + GraphHead 字段）。

**v4bg = H4 假设变体（任务 10-19-gpfm-mamba-v4b-gnn-f0，design §2/§3）**：
在 v4b **d_ch=1** 基座（H3 v4b-d1 臂）上，把 `_head` 的稠密 cMLP（W1/W2 einsum，62 子 MLP，
d1 口径 63,550 参数）替换为 **GraphHead**——latent-flow-v1 `LearnableMexHatMixer` 的适配移植
（`A_eff = γ_m·M(d;σ) + γ·W`：固定带通距离核钉住空间结构 + 可学习**有向**残差 W 零初始化），
检验「几何先验约束的跨通道混合位能否成为 GC 载体 / 改善传导特异性」（H4）。

- 基座不变部分（design §2.1，与 v4b-d1 逐位同构——同 seed 下 trunk 部分可复现对照）：
  逐通道特征 18 维 → 共享 Linear(18→1) → 4×MambaBlock(1) CI trunk → future 128 截取 +
  逐通道 RMSNorm(1)。
- 唯一结构变化（design §2.2，GraphHead）：per-channel Linear(1→h) → K×{A_eff 聚合 +
  per-channel Linear(h→h)+GELU} → per-channel Linear(h→1)；σ 可学习（ricker 核 log_sigma
  参数化）/ dog 4 参数备选；W (62,62) 有向零初始化（w_init 路径可注入）。
- 默认超参（对齐 10-19 edra_fmt stage2 run 实际口径 + design §3）：h=64、K=2、γ=0.1、
  γ_m=1.0、ricker、λ_u=λ_s=0（正则方法保留但不进损失）。
- **捆绑子变更（A-5，Run 块须照登）**：激活 SiLU→GELU；62 独立子 MLP → 通道共享权重 MLP；
  隐宽形态 d_z=16(per-target) → h=64(共享)；head 参数 63,550 → 12,358（design §2.4 预算）。
  v4 config 的 cMLP 专属字段 d_z/head_act 随头替换**移除**（模型无对应结构，保留即死字段），
  由 graph_* 字段取代（d_z/head_act 先例的延续）。
- 方向约定（A-1，design §2.2）：GraphHead 内部保持源语义 `A_eff[i,j] = 通道 j 驱动通道 i`
  （消息传递 `x @ A_effᵀ`）；一切对外读出统一转置为行=源、列=靶（S4 graph_readout 单点）。

**训练口径继承（design §3「继承 v4 全部字段」）**：数据根 `/root/autodl-tmp/edra_process_v4`
（旧范式 500 Hz、past 384/future 128、62ch）、划分 32/4/10（split_manifest_override 显式指向
v4 的 fold0 val4 manifest——与 v4a 逐字节相同，只读复用不新建）、batch 256、seed 1234、
EMA 0.999、早停 patience=10、lr 恒定 1e-3（无 cosine）、gen_lowpass_hz=40.0 正式口径。

依赖方式（path-isolation 薄变体契约 + v4 双遮蔽惯例）：本目录含 config.py + graph_head.py +
model.py（v4bg 新结构）+ train.py/evaluate.py 入口包装（config+model 双遮蔽，前缀守卫 `v4bg-`）；
mamba/data/gpr_prior/losses/train/sample/evaluate 等模块只读复用 v1（经 sys.path，v1 零改动）；
latent-flow-v1 全程只读（等价测经 sys.path 导入对照）。输出落同一任务 outputs 根
`/root/npi/outputs/gpfm-mamba-f0/`，run_id 一律前缀 `v4bg-`（包装脚本强制校验），
checkpoint 名 `gpfm-mamba-v4b-gnn-{best,last}.pt` 与 v1/v2/v4 天然隔离。
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class Config:
    # ---- 路径（path-isolation 契约；outputs 根与 v1/v2/v4 共享任务根，run_id 前缀隔离）----
    data_root: str = "/root/autodl-tmp/edra_process_v4"
    outputs_root: str = "/root/npi/outputs/gpfm-mamba-f0"
    variant_name: str = "gpfm-mamba-v4b-gnn"  # checkpoint 主名 + config_hash 组成部分（区别于 v1/v2/v4）

    # ---- 数据与划分（v4 系 2026-10-16 长期口径 7/1/2 = 32/4/10；manifest 只读复用 v4 的）----
    n_channels: int = 62
    past_len: int = 384
    future_len: int = 128
    stride: int = 64
    n_folds: int = 5
    fold: int = 0                 # study1 前置验证固定 fold0
    split_seed: int = 42          # 受试者级 5-fold 分层（MDD/HC）划分 seed
    subject_filter: str = "status_ok"  # 受试者过滤口径（2026-10-12 用户拍板 b）：'status_ok'（46 例）| 'qc_pass'（13 例）
    n_val_subjects: int = 4       # 训练折内内部验证受试者（分层每类 2 例）——2026-10-16 用户拍板长期口径
                                  # train/val/test=7/1/2（整数分配 32/4/10）；v1/v2 为 2（34/2/10）
    val_subject_seed: int = 42
    fixate_split: bool = True     # 划分固化：校验 v4 的固化 manifest（见 split_manifest_override，只读不新建）
    split_manifest_override: str = "/root/npi/models/gpfm-mamba-v4/splits/fold0_status_ok_seed42_val4.json"
                                  # **显式指向 v4 的 fold0 val4 manifest**（design §3/B-8：与 v4a 逐字节相同，
                                  # 单测断言；禁止缺省——v1 fixate_split 在 manifest 缺失时会向
                                  # models/gpfm-mamba-v1/splits/ 新写文件（零改动违规），故本字段必填）。
                                  # v1/v2/v3 无此字段 = 路径不变、零漂移。
                                  # 不可比注记：train 32 vs v1/v2 的 34 例；best 判据 val4 vs val2；test 折相同
    num_workers: int = 4

    # ---- GPR 条件先验（与 v1/v2/v4 一致）----
    kernel_t: str = "ou"          # 'ou'（默认）| 'matern32'（备选）
    gamma: float = 1.0
    period_p: int = 50            # tau_i = i*pi/p（alpha 节律周期，禁止整窗 [0,1] 归一）
    iso: float = 1e-2             # 白噪折叠进时间核：K'_t = K_t + iso*I（保持 Kronecker）
    pp_jitter: float = 1e-4       # K'_t,pp 求逆前额外 +1e-4*I
    sigma_s: float = 0.080        # Ricker 3D 空间核尺度（米，真实头皮坐标）
    jitter_s: float = 1e-3        # K_s <- K_s + jitter_s*I

    # ---- Mamba 速度网络 v4bg（v4b d_ch=1 基座；结构变化 = cMLP head → GraphHead）----
    d_model: int = 1              # v4-B 语义 = 逐通道独立宽度 d_ch。本变体固定 1（H3 v4b-d1 臂基座，
                                   # 用户 2026-10-19「开启」拍板确认；阴性结果按「限 d1 基座不支持」表述）
    n_layers: int = 4
    d_state: int = 16
    d_conv: int = 4
    expand: int = 2
    time_emb_dim: int = 16        # future token 的 t 正弦嵌入维度（逐通道特征 = 1+1+16=18）
    use_fast_path: bool = True    # mamba_ssm 官方 CUDA fast path

    # ---- GraphHead（v4bg 专属，design §3；d_z/head_act 先例延续：类默认 + 一致性断言双防线）----
    graph_kernel: str = "ricker"        # ricker | dog（距离核形状；σ/log_sigma 参数化同源组件）
    graph_k: int = 2                    # 消息传递轮数 K
    graph_hidden: int = 64              # per-channel 隐宽 h（通道共享权重）
    graph_gamma: float = 0.1            # γ（W 权重；buffer，可 ckpt 恢复）
    graph_gamma_m: float = 1.0          # γ_m（距离核权重；0 = 消融先验，buffer）
    graph_w_init: str = ""              # 空 = W 零初始化（源默认）；否则 npy/pt 路径注入 (62,62) 有向初值
    graph_lambda_uniform: float = 0.0   # uniformity 正则权重（默认关闭，对齐 10-19 edra_fmt run 实际口径；
                                        # 旋钮落地即不进训练损失——启用属口径变更须另登记）
    graph_lambda_sparse: float = 0.0    # sparsity 正则权重（同上，默认关闭）
    geometry_path: str = "/root/autodl-tmp/edra_processed_fmt/geometry_edra_fmt.pt"
                                        # 62ch 10-20 电极几何（10-19 D2/D4 拍板产物，只读）；
                                        # GraphHead 构造单一 choke point 加载 + 全 46 例通道名断言

    # ---- 复合损失（定义域 = future 128 列；与 v1/v2/v4 一致）----
    w_mae: float = 1.0
    w_cons: float = 1.0
    w_tv: float = 0.1             # 纯平滑惩罚，不对照 GT 差分
    w_corr: float = 0.1
    corr_eps: float = 1e-6

    # ---- 训练（与 v4 一致：恒定 lr + 早停 10）----
    seed: int = 1234              # 训练 seed（划分 seed 独立 = 42）
    batch_size: int = 256
    epochs: int = 100
    lr: float = 1e-3
    weight_decay: float = 1e-4
    lr_schedule: str = "constant"  # lr 恒定（2026-10-16 用户拍板；v1 train.py default-off 钩子透传）
    grad_clip: float = 0.5
    ema_decay: float = 0.999      # 短程训练口径（v1 登记的偏离论文 0.9999，不变）
    sigma_min: float = 1e-4
    t_loc: float = -0.8           # t = sigmoid(z), z ~ N(t_loc, t_scale^2)（logit-normal，不用 U[0,1]）
    t_scale: float = 0.8
    t_clamp_min: float = 1e-4
    t_clamp_max: float = 0.9999
    cudnn_deterministic: bool = True
    cudnn_benchmark: bool = False
    log_every: int = 1            # 每 step 进度行（R9）

    # ---- 内部验证 / best 判据（与 v1/v2/v4 一致）----
    val_t_grid: tuple = (0.1, 0.3, 0.5, 0.7, 0.9)
    val_eps_seed: int = 2026      # 固定 eps（确定性复算，跨 epoch 可比）
    use_ema_for_val: bool = True  # 评估用 EMA 权重
    early_stop_patience: int = 10  # 早停（2026-10-16 用户拍板）：val4 总 loss连续 10 epoch 无新 best 即停

    # ---- 采样（与 v1/v2/v4 一致）----
    euler_steps: int = 16
    n_samples: int = 16
    gen_lowpass_hz: float = 40.0  # 生成样本输出侧 40 Hz 砖墙低通（正式口径显式入 config）

    # ---- helper（与 v1/v2/v4 相同实现）----
    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    def config_hash(self) -> str:
        """resolved config 的 sha256 前 12 位，写入 checkpoint。"""
        blob = json.dumps(self.to_dict(), sort_keys=True, default=str)
        return hashlib.sha256(blob.encode()).hexdigest()[:12]

    def dump_yaml(self, path: str) -> None:
        """零依赖 YAML dump（flat 标量/元组，tuple -> JSON list）。"""
        lines = ["# gpfm-mamba-v4b-gnn (v4b d_ch=1 基座 + LearnableMexHatMixer 图头) resolved config（跑前 dump，reproducibility 契约）"]
        for k, v in self.to_dict().items():
            if isinstance(v, bool):
                lines.append(f"{k}: {str(v)}")
            elif isinstance(v, str):
                lines.append(f'{k}: "{v}"')
            elif isinstance(v, (tuple, list)):
                lines.append(f"{k}: {json.dumps(list(v))}")
            else:
                lines.append(f"{k}: {v}")
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")


def build_config(**overrides: Any) -> Config:
    cfg = Config()
    valid = {f.name for f in dataclasses.fields(Config)}
    for k, v in overrides.items():
        if k not in valid:
            raise KeyError(f"unknown config key: {k}")
        setattr(cfg, k, v)
    return cfg
