"""gpfm-mamba-v1 唯一配置源。

全部已定决策回填自 design.md §8（2026-10-12 用户拍板）：
- K_s = Ricker 三维正定形式 (1 - d^2/(3 sigma_s^2)) exp(-d^2/(2 sigma_s^2))，sigma_s=0.080 m + jitter_s=1e-3
  （严禁严格 1D Ricker (1-d^2/sigma^2)e^{-d^2/2sigma^2}——真实电极点集上实测非 PSD）
- K_t = OU（gamma=1, tau_i=i*pi/p, p=50, iso=1e-2 折叠进时间核，禁止整窗 [0,1] 归一）
- 损失权重 {mae:1, cons:1, tv:0.1, corr:0.1}；tv 纯平滑不对照 GT；corr 逐通道和式 Pearson
- t ~ sigmoid(N(-0.8, 0.8^2)) 逐窗，clamp [1e-4, 1-1e-4]；一步生成 x_hat1 = x_t + (1-t) v_theta
- best 判据 = 内部验证总 loss（固定 t 网格 {0.1,0.3,0.5,0.7,0.9} + 固定 eps 确定性复算）
- identity-K_s 消融臂不跑（需要归因时另建薄变体 gpfm-mamba-v1-ksid）
- 受试者过滤（2026-10-12 用户门禁拍板 b）：manifest status=='ok'（46 例 HC24/MDD22；
  qc_pass=True 仅 13 例——33 例挂 s7_interp_carry 软门但管线作者标 ok）
- 划分固化（用户要求）：fold0 划分 manifest 落 splits/ 目录，后续运行校验一致、漂移即报错
- EMA 0.999（登记偏离论文 0.9999：全程约 1.8 万步，tau=1/(1-decay)≈1000 步≈5.5 epoch 匹配短程训练；
  论文 0.9999 的 tau=1 万步在本程会让影子权重全程滞后）
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class Config:
    # ---- 路径（path-isolation 契约）----
    data_root: str = "/root/autodl-tmp/edra_process_v4"
    outputs_root: str = "/root/npi/outputs/gpfm-mamba-f0"
    variant_name: str = "gpfm-mamba-f0"

    # ---- 数据与划分（design §4）----
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
    fixate_split: bool = True     # 划分固化：首次落 manifest 到 splits/，此后校验一致、漂移报错
    num_workers: int = 4

    # ---- GPR 条件先验（design §2.1，时空可分 K_s x K_t）----
    kernel_t: str = "ou"          # 'ou'（默认）| 'matern32'（备选 -v2）
    gamma: float = 1.0
    period_p: int = 50            # tau_i = i*pi/p（alpha 节律周期，禁止整窗 [0,1] 归一）
    iso: float = 1e-2             # 白噪折叠进时间核：K'_t = K_t + iso*I（保持 Kronecker）
    pp_jitter: float = 1e-4       # K'_t,pp 求逆前额外 +1e-4*I
    sigma_s: float = 0.080        # Ricker 3D 空间核尺度（米，真实头皮坐标）
    jitter_s: float = 1e-3        # K_s <- K_s + jitter_s*I

    # ---- Mamba 速度网络（design §3）----
    d_model: int = 16
    n_layers: int = 4
    d_state: int = 16
    d_conv: int = 4
    expand: int = 2
    time_emb_dim: int = 16        # future token 的 t 正弦嵌入维度
    use_fast_path: bool = True    # mamba_ssm 官方 CUDA fast path

    # ---- 复合损失（design §2.3，定义域 = future 128 列）----
    w_mae: float = 1.0
    w_cons: float = 1.0
    w_tv: float = 0.1             # 纯平滑惩罚，不对照 GT 差分
    w_corr: float = 0.1
    corr_eps: float = 1e-6

    # ---- 训练（design §2.2 / §5）----
    seed: int = 1234              # 训练 seed（划分 seed 独立 = 42）
    batch_size: int = 256
    epochs: int = 100
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 0.5
    ema_decay: float = 0.999      # 2026-10-12 调整（论文 0.9999；理由见文件头注释）
    sigma_min: float = 1e-4
    t_loc: float = -0.8           # t = sigmoid(z), z ~ N(t_loc, t_scale^2)（logit-normal，不用 U[0,1]）
    t_scale: float = 0.8
    t_clamp_min: float = 1e-4
    t_clamp_max: float = 0.9999
    cudnn_deterministic: bool = True
    cudnn_benchmark: bool = False
    log_every: int = 1            # 每 step 进度行（R9）

    # ---- 内部验证 / best 判据（design §2.3）----
    val_t_grid: tuple = (0.1, 0.3, 0.5, 0.7, 0.9)
    val_eps_seed: int = 2026      # 固定 eps（确定性复算，跨 epoch 可比）
    use_ema_for_val: bool = True  # 评估用 EMA 权重

    # ---- 采样（§5/§6，本期不执行）----
    euler_steps: int = 16
    n_samples: int = 16

    # ---- helper ----
    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    def config_hash(self) -> str:
        """resolved config 的 sha256 前 12 位，写入 checkpoint。"""
        blob = json.dumps(self.to_dict(), sort_keys=True, default=str)
        return hashlib.sha256(blob.encode()).hexdigest()[:12]

    def dump_yaml(self, path: str) -> None:
        """零依赖 YAML dump（flat 标量/元组，tuple -> JSON list）。"""
        lines = ["# gpfm-mamba-v1 resolved config（跑前 dump，reproducibility 契约）"]
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
