"""gpfm-mamba-v5-fmt 唯一配置源（v5 config 副本 + fmt 数据口径字段；任务 10-20-gpfm-mamba-v5-fmt-f0）。

**变体语义**：gpfm-mamba-v5 结构（v2 基座 d_model=62 × 4×[Mamba→GNN 空间块→LayerNorm]
轴分解交替，model/graph 自 v5 只读复制，结构零改动）× **edra_fmt 数据口径**——训练
`v5-fmt-01`（与 v5-main-01（edra_process_v4/500 Hz）并列，同 outputs 根），产出 θ* 候选
供 study2 NI-MSE-50-q0.90 主实例化（ni-mse-design.md v1.5）。

**与 gpfm-mamba-v5 的字段差异（其余逐字段一致；fmt 口径逐字段以 v1-fmt 先例为准）**：
- data_root: edra_process_v4 → **edra_processed_fmt**（250 Hz 新范式，48 例 HC24/MDD24）；
- variant_name: "gpfm-mamba-v5" → **"gpfm-mamba-v5-fmt"**（checkpoint 主名 + hash 组成部分，
  三处注册一致；与 v5-main-01 同 outputs 根下 checkpoint 主名天然隔离）；
- n_val_subjects: 2 → **4**（fmt 7/1/2 长期口径实际整数分配 34/4/10——与 v1-fmt manifest 一致，
  fixate_split 校验所需）；
- split_manifest_override:（v5 无此字段）→ **v1-fmt 固化 manifest 只读复用**（零重新划分）；
- period_p: 50 → **25**（fmt D3：tau_i=i·π/p 采样点网格，fs 减半 p 减半保物理时长）；
- norm_type:（v5 无此字段）→ **"layernorm"**（fmt 线 D4：5 个 norm 位 = 4×MambaBlock pre-norm
  + head norm 全换 LayerNorm，容量 +310 bias 参数（5×62）；v5-fmt model.py 钩子断言放行
  "rmsnorm"|"layernorm" 两值——v5 断言收窄的 fmt 最小放开，偏离登记见任务 experiment.md）；
- fs:（v5 无此字段）→ **250.0**（PSD 频轴/轨迹图标秒经 evaluate.py getattr 透传）；
- gen_lowpass_hz:（v5 无此字段）→ **None**（fmt D2 无低通：真实侧本就 1–40 Hz 带限，
  直接观察模型是否自行待在带内；旧范式 40 Hz 低通口径不变，跨范式谱类对比须登记差异）。

**训练纪律字段处置（2026-10-20 用户拍板更新）**：early_stop_patience=10 **已启用**（用户主 Run 前明示：
「cosine+早停10开启，仅修改这个配置」——cosine 为 v5 家族默认本就在用，早停 10 为本次唯一配置变更，
不走子代理测试）；lr_schedule 仍不自 v1-fmt 移植（v1-fmt 侧 "constant"，v5 家族默认 cosine）——
v5-fmt-01 实跑口径 = **cosine 退火 + 早停 patience 10**，与 v5-main-01（cosine + 早停禁用）的对照
差异集须随附本条（v5-main-01 跑满 100、best ckpt 语义等价早停产物）。

graph 七字段与 d_model=62 不变（geometry_path 本就 fmt 原生 geometry_edra_fmt.pt）。

依赖方式（path-isolation 契约 + v5 双遮蔽惯例）：本目录含 config.py / model.py / graph.py +
train.py/evaluate.py 双遮蔽入口包装；mamba/data/gpr_prior/losses/train/sample/evaluate/gc_psd
等模块**只读复用 v1**（经 sys.path）。输出落 `/root/npi/outputs/gpfm-mamba-f0/`（与 v1/v2/v5
同根），run_id 一律前缀 `v5-`（包装脚本强制校验），checkpoint 名 `gpfm-mamba-v5-fmt-{best,last}.pt`。

**250 Hz 已知坑（v1-fmt 先例）**：sample.py 低通 `brickwall_lowpass(out, cut_hz)` 用默认
fs=500——本 config gen_lowpass_hz=None 恒不触发；日后重开低通必须传 fs=250。
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class Config:
    # ---- 路径（path-isolation 契约；与 v1/v2/v5 同 outputs 根，run_id 前缀 + checkpoint 主名隔离）----
    data_root: str = "/root/autodl-tmp/edra_processed_fmt"       # fmt 新范式数据集（250 Hz，48 例）
    outputs_root: str = "/root/npi/outputs/gpfm-mamba-f0"        # 与 v5-main-01 同根并列（PRD R1）
    variant_name: str = "gpfm-mamba-v5-fmt"                      # checkpoint 主名 + config_hash 组成部分（三处注册一致）

    # ---- 数据与划分（fmt 口径：48 例 → 34/4/10；其余参数同 v5 → 经 override 校验 v1-fmt manifest）----
    n_channels: int = 62
    past_len: int = 384
    future_len: int = 128
    stride: int = 64
    n_folds: int = 5
    fold: int = 0                 # study1 前置验证固定 fold0
    split_seed: int = 42          # 受试者级 5-fold 分层（MDD/HC）划分 seed
    subject_filter: str = "status_ok"  # fmt manifest 48 行全 status=ok（qc_pass 留空不可用，v1-fmt 先例）
    n_val_subjects: int = 4       # fmt 7/1/2 长期口径实际整数分配 34/4/10（v1-fmt manifest 一致所需；
                                  # 旧 v5 为 2=34/2/10——跨数据范式对比须随附划分差异）
    val_subject_seed: int = 42
    fixate_split: bool = True     # 划分固化：同参数 → 校验 v1-fmt 的 splits/ manifest（只读共享，零重新划分）
    split_manifest_override: str = "/root/npi/models/gpfm-mamba-v1-fmt/splits/fold0_status_ok_seed42_val4.json"
                                  # S1.2：v1-fmt 固化 fold0 manifest 只读复用（v1 data.py default-off 钩子）；
                                  # 34/4/10 与本 config 同参数生成内容一致（fixate 校验通过即证）
    num_workers: int = 4

    # ---- GPR 条件先验（与 v5 一致，唯 p 减半）----
    kernel_t: str = "ou"          # 'ou'（默认）| 'matern32'（备选）
    gamma: float = 1.0
    period_p: int = 25            # fmt D3：tau_i = i*pi/p 采样点网格，fs 500→250 时 p 50→25 保物理时长
                                  # （OU e-folding = p/(pi*fs) 秒严格等价；先验与数据同时变，判读随附差异清单）
    iso: float = 1e-2             # 白噪折叠进时间核：K'_t = K_t + iso*I（保持 Kronecker）
    pp_jitter: float = 1e-4       # K'_t,pp 求逆前额外 +1e-4*I
    sigma_s: float = 0.080        # Ricker 3D 空间核尺度（米，真实头皮坐标）
    jitter_s: float = 1e-3        # K_s <- K_s + jitter_s*I

    # ---- Mamba 速度网络（v5 结构原样；d_model=62 轴分解不变量，唯 norm 类型换）----
    d_model: int = 62             # = n_channels（v5 轴分解不变量：模型维即通道维，零折叠/合并；模型构造断言）
    norm_type: str = "layernorm"  # fmt 线 D4：5 个 norm 位（4×MambaBlock pre-norm + head norm）全换
                                  # LayerNorm（容量 +310 = 5×62 bias）；v1 mamba.py/model.py default-off
                                  # 钩子 getattr 透传——v5 config 无此字段 = "rmsnorm"（v5 结构与
                                  # config_hash 零漂移）。v5-fmt model.py 钩子断言放行两值
    n_layers: int = 4             # 交替复合块数：4×[Mamba → 空间块 → LayerNorm]
    d_state: int = 16
    d_conv: int = 4
    expand: int = 2
    time_emb_dim: int = 16        # future token 的 t 正弦嵌入维度（保持 16，随 v2 不变）
    use_fast_path: bool = True    # mamba_ssm 官方 CUDA fast path

    # ---- v5 图空间块（自 v5 原样：七字段零改动；geometry_path 本就 fmt 原生）----
    graph_kernel: str = "ricker"  # 距离核：'ricker'（默认，σ 可学习 log_sigma 参数化）| 'dog'（4 参数重参数化）
    graph_gamma: float = 0.1      # γ：可学习有向残差 W 的权重
    graph_gamma_m: float = 1.0    # γ_m：距离核 M 的尺度（0 = 消融几何先验臂）
    graph_w_init: str = ""        # W 初始化："" = 零初始化（默认）；非空 = (62,62) 有向拓扑文件路径注入
    graph_lambda_uniform: float = 0.0   # λ_u：anti-uniformity 正则权重（默认 0 = 关闭，不进损失）
    graph_lambda_sparse: float = 0.0    # λ_s：L1 稀疏正则权重（默认 0 = 关闭，不进损失）
    geometry_path: str = "/root/autodl-tmp/edra_processed_fmt/geometry_edra_fmt.pt"  # 62ch 10-20 几何（只读，fmt 原生）

    # ---- 采样率（fmt 口径新增；PSD 频轴/轨迹图标秒经 evaluate.py getattr 透传）----
    fs: float = 250.0             # 旧范式无此字段 → getattr 默认 500.0（零漂移）；Nyquist 125 Hz

    # ---- 复合损失（定义域 = future 128 列；与 v5 一致）----
    w_mae: float = 1.0
    w_cons: float = 1.0
    w_tv: float = 0.1             # 纯平滑惩罚，不对照 GT 差分
    w_corr: float = 0.1
    corr_eps: float = 1e-6

    # ---- 训练（v5 家族默认 cosine 退火；早停 10 = 2026-10-20 用户拍板启用——见文件头注记）----
    seed: int = 1234              # 训练 seed（划分 seed 独立 = 42）
    batch_size: int = 256
    epochs: int = 100
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 0.5
    early_stop_patience: int = 10  # 连续 10 epoch 无新 val best 即停（v4-B 口径字段，v1 train.py getattr 透传；用户 2026-10-20 拍板启用）
    ema_decay: float = 0.999      # 短程训练口径（v1 登记的偏离论文 0.9999，不变）
    sigma_min: float = 1e-4
    t_loc: float = -0.8           # t = sigmoid(z), z ~ N(t_loc, t_scale^2)（logit-normal，不用 U[0,1]）
    t_scale: float = 0.8
    t_clamp_min: float = 1e-4
    t_clamp_max: float = 0.9999
    cudnn_deterministic: bool = True
    cudnn_benchmark: bool = False
    log_every: int = 1            # 每 step 进度行（R9）

    # ---- 内部验证 / best 判据（与 v5 一致）----
    val_t_grid: tuple = (0.1, 0.3, 0.5, 0.7, 0.9)
    val_eps_seed: int = 2026      # 固定 eps（确定性复算，跨 epoch 可比）
    use_ema_for_val: bool = True  # 评估用 EMA 权重

    # ---- 采样（与 v5 一致，唯无低通）----
    euler_steps: int = 16
    n_samples: int = 16
    gen_lowpass_hz: Optional[float] = None  # fmt D2：生成侧不加 40 Hz 低通（真实侧本就 1–40 Hz 带限；
                                            # 直接观察模型是否自行待在带内）。v5 经 getattr 默认 40.0。
                                            # 日后在 250 Hz 重开低通必须传 fs=250（sample.py 默认 fs=500 坑）

    # ---- helper（与 v5 相同实现）----
    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    def config_hash(self) -> str:
        """resolved config 的 sha256 前 12 位，写入 checkpoint。"""
        blob = json.dumps(self.to_dict(), sort_keys=True, default=str)
        return hashlib.sha256(blob.encode()).hexdigest()[:12]

    def dump_yaml(self, path: str) -> None:
        """零依赖 YAML dump（flat 标量/元组，tuple -> JSON list）。"""
        lines = ["# gpfm-mamba-v5-fmt（v5 轴分解交替 × edra_fmt/250 Hz：LayerNorm/p=25/无低通）"
                 " resolved config（跑前 dump，reproducibility 契约）"]
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
