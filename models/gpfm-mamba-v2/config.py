"""gpfm-mamba-v2 唯一配置源（薄变体，完整 config 副本）。

**单变量消融（用户 2026-10-12 指定）：唯一变化 = d_model 16 → 62**
——验证「62 通道压 16 维的跨通道瓶颈是否为 GC 偏弱（Run 007：GC r=0.282、幅值约真实一半）的主因」。
d_model=62 使 Mamba 残差流宽度 = 通道数（in_proj/head 仍为 62↔62，瓶颈仅指 d_model 内部宽度）。

其余一切与 gpfm-mamba-v1 完全一致（n_layers=4、d_state=16、d_conv=4、expand=2、
初始化方式、EMA 0.999、损失权重 {1,1,0.1,0.1}、t 分布 logit-normal(−0.8,0.8)、
数据口径 status_ok 46 例、划分 split_seed=42（读 v1 固化 manifest，跨变体共享只读）、
训练超参、评估口径）——不做恒等初始化等任何额外变量。

依赖方式（path-isolation 薄变体契约，design §7 v2 行）：
- 本目录仅 config.py + train.py/evaluate.py 入口包装；data/gpr_prior/mamba/model/losses/
  train/sample/evaluate/gc_psd 等模块**只读复用 v1**（经 sys.path，不复制代码）。
- 输出落同一任务 outputs 根 `/root/npi/outputs/gpfm-mamba-f0/`，run_id 一律前缀 `v2d62-`
  （包装脚本强制校验），checkpoint 名 `gpfm-mamba-v2d62-{best,last}.pt` 与 v1 天然隔离。
- 划分固化：subject_filter/split_seed 与 v1 相同 ⇒ data.fixate_split 校验到
  `models/gpfm-mamba-v1/splits/fold0_status_ok_seed42.json`（只读，不重写）。

v1 决策回填（与 v1 config 同源，未改动项不重复注释——见 models/gpfm-mamba-v1/config.py 文件头）。
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class Config:
    # ---- 路径（path-isolation 契约；outputs 根与 v1 同任务共享，run_id 前缀隔离）----
    data_root: str = "/root/autodl-tmp/edra_process_v4"
    outputs_root: str = "/root/npi/outputs/gpfm-mamba-f0"
    variant_name: str = "gpfm-mamba-v2d62"  # checkpoint 主名 + config_hash 组成部分（区别于 v1）

    # ---- 数据与划分（design §4；与 v1 逐字段一致 → 校验到 v1 固化 manifest）----
    n_channels: int = 62
    past_len: int = 384
    future_len: int = 128
    stride: int = 64
    n_folds: int = 5
    fold: int = 0                 # study1 前置验证固定 fold0
    split_seed: int = 42          # 受试者级 5-fold 分层（MDD/HC）划分 seed
    subject_filter: str = "status_ok"  # 受试者过滤口径（2026-10-12 用户拍板 b）：'status_ok'（46 例）| 'qc_pass'（13 例）
    n_val_subjects: int = 2       # 训练折内内部验证受试者（每类 1 例）
    val_subject_seed: int = 42
    fixate_split: bool = True     # 划分固化：与 v1 同参数 → 校验 v1 的 splits/ manifest（只读共享）
    num_workers: int = 4

    # ---- GPR 条件先验（design §2.1，时空可分 K_s x K_t；与 v1 一致）----
    kernel_t: str = "ou"          # 'ou'（默认）| 'matern32'（备选）
    gamma: float = 1.0
    period_p: int = 50            # tau_i = i*pi/p（alpha 节律周期，禁止整窗 [0,1] 归一）
    iso: float = 1e-2             # 白噪折叠进时间核：K'_t = K_t + iso*I（保持 Kronecker）
    pp_jitter: float = 1e-4       # K'_t,pp 求逆前额外 +1e-4*I
    sigma_s: float = 0.080        # Ricker 3D 空间核尺度（米，真实头皮坐标）
    jitter_s: float = 1e-3        # K_s <- K_s + jitter_s*I

    # ---- Mamba 速度网络（design §3；**单变量：d_model 16 → 62**）----
    d_model: int = 62             # ←←← 本变体唯一超参变化（v1=16；62=通道数，消跨通道瓶颈）
    n_layers: int = 4
    d_state: int = 16
    d_conv: int = 4
    expand: int = 2
    time_emb_dim: int = 16        # future token 的 t 正弦嵌入维度（保持 16，不随 d_model 变——单变量原则）
    use_fast_path: bool = True    # mamba_ssm 官方 CUDA fast path

    # ---- 复合损失（design §2.3，定义域 = future 128 列；与 v1 一致）----
    w_mae: float = 1.0
    w_cons: float = 1.0
    w_tv: float = 0.1             # 纯平滑惩罚，不对照 GT 差分
    w_corr: float = 0.1
    corr_eps: float = 1e-6

    # ---- 训练（design §2.2 / §5；与 v1 一致）----
    seed: int = 1234              # 训练 seed（划分 seed 独立 = 42）
    batch_size: int = 256
    epochs: int = 100
    lr: float = 1e-3
    weight_decay: float = 1e-4
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

    # ---- 内部验证 / best 判据（design §2.3；与 v1 一致）----
    val_t_grid: tuple = (0.1, 0.3, 0.5, 0.7, 0.9)
    val_eps_seed: int = 2026      # 固定 eps（确定性复算，跨 epoch 可比）
    use_ema_for_val: bool = True  # 评估用 EMA 权重

    # ---- 采样（§5/§6；与 v1 一致）----
    euler_steps: int = 16
    n_samples: int = 16

    # ---- helper（与 v1 相同实现）----
    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    def config_hash(self) -> str:
        """resolved config 的 sha256 前 12 位，写入 checkpoint。"""
        blob = json.dumps(self.to_dict(), sort_keys=True, default=str)
        return hashlib.sha256(blob.encode()).hexdigest()[:12]

    def dump_yaml(self, path: str) -> None:
        """零依赖 YAML dump（flat 标量/元组，tuple -> JSON list）。"""
        lines = ["# gpfm-mamba-v2 (d_model=62) resolved config（跑前 dump，reproducibility 契约）"]
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
