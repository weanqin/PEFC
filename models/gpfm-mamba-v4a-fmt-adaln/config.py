"""gpfm-mamba-v4a-fmt-adaln 唯一配置源（薄变体，完整 config 副本；任务 10-17-gpfm-mamba-adaln-f0）。

**基线 = gpfm-mamba-v4a-fmt**（v4a 结构 + edra_processed_fmt 范式，fold0 最优候选
v4afmt-main-01）：本 config = v4a-fmt config 完整副本 + 唯 4 差异字段（design §1.2 表）——

**单变量差异声明（与 v4afmt-main-01 构成单变量对比）**：
norm LayerNorm→adaLN(t)（A 案 scale/shift，5 个 norm 位 = 4×MambaBlock pre-norm + head norm
全部换 `modulate(LN(x), s(t), b(t)) = (1+s)⊙LN(x)+b`，(s,b) 由 time_embedding(t,16) 经逐 site
Sequential(SiLU, Linear(16,124)) 零初始化生成）+ 参数量 130,366→**140,906（+10,540 / +8.08%）**。
其余逐字段一致（norm_type="layernorm" 保持——adaLN 变体底层 norm 仍是 LayerNorm，两字段两维度；
bottleneck_rank=16、d_model=62、fs=250、period_p=25、gen_lowpass_hz=None、seed 1234、
早停 10、恒 lr 1e-3、val4、EMA 0.999、batch 256）。

**差异字段（唯 4）**：
1. `cond_adaln="scale_shift"`（v1 mamba.py/model.py default-off 钩子启用；v4a-fmt 无此字段=off）
2. `variant_name="gpfm-mamba-v4a-fmt-adaln"`（checkpoint 主名）
3. `outputs_root="/root/npi/outputs/gpfm-mamba-adaln-f0"`（本任务新根，禁写旧根）
4. `split_manifest_override` 指本目录 splits/（首跑固化；与 v4a-fmt manifest **逐字节一致**——
   同参数决定性划分，单测断言）

依赖方式（path-isolation 薄变体契约，v4a-fmt 式单遮蔽）：本目录仅 config.py + train.py/
evaluate.py/fc_metric.py 入口包装（无 --rank——r=16 固定，比 v4a-fmt wrapper 简一档）；
model/mamba/data/gpr_prior/losses/train/sample/evaluate/gc_psd 只读复用 v1（经 sys.path）。
输出落本任务 outputs 根 `/root/npi/outputs/gpfm-mamba-adaln-f0/`，checkpoint 名
`gpfm-mamba-v4a-fmt-adaln-{best,last}.pt`。

**run_id 前缀守卫（RUN_PREFIX = `v4afmt-adaln-`）**：`v4afmt-adaln-*` 归本任务根；注意本前缀
嵌套于 v4a-fmt 的 `v4afmt-` 之下——若误经 v4a-fmt wrapper 传 adaln run_id 其守卫放行（已知局限，
晚失败兜底：独立 outputs 根 FileNotFoundError + ckpt 名不同 + config_hash 校验，非静默）。

**GC 阻塞延续（v4a-fmt D1 口径）**：评估经 evaluate 包装带 `--skip-gc-only`（PSD fs=250 与
空间归因照跑，无任何 GC 对比产物）。
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class Config:
    # ---- 路径（path-isolation 契约；本任务独立 outputs 根，与 v4a-fmt 完全隔离）----
    data_root: str = "/root/autodl-tmp/edra_processed_fmt"       # 新预处理范式数据集（250 Hz；与 v4a-fmt 同源）
    outputs_root: str = "/root/npi/outputs/gpfm-mamba-adaln-f0"  # 本任务 outputs 根（禁写旧根 gpfm-mamba-f0/fmt-f0）
    variant_name: str = "gpfm-mamba-v4a-fmt-adaln"               # checkpoint 主名 + config_hash 组成部分

    # ---- 数据与划分（与 v4a-fmt 逐字段一致；口径同 v4a 的 7/1/2 长期语义）----
    n_channels: int = 62
    past_len: int = 384
    future_len: int = 128
    stride: int = 64
    n_folds: int = 5
    fold: int = 0                 # study1 前置验证固定 fold0
    split_seed: int = 42          # 受试者级 5-fold 分层（MDD/HC）划分 seed
    subject_filter: str = "status_ok"  # fmt manifest 48 行全 status=ok（qc_pass 留空不可用）
    n_val_subjects: int = 4       # 训练折内内部验证受试者（分层每类 2 例）——7/1/2 长期口径语义
                                  # （实际整数分配以首跑固化 manifest 为准；与 v4a-fmt 同参数 ⇒ 34/4/10）
    val_subject_seed: int = 42
    fixate_split: bool = True     # 划分固化：首建写入本变体 splits/，之后校验一致、漂移报错
    split_manifest_override: str = "/root/npi/models/gpfm-mamba-v4a-fmt-adaln/splits/fold0_status_ok_seed42_val4.json"
                                  # 本目录自备 manifest（首跑固化；与 v4a-fmt 同参数生成 ⇒ 逐字节一致，单测断言）
    num_workers: int = 4

    # ---- GPR 条件先验（与 v4a-fmt 逐字段一致，唯 p 减半语义同基线）----
    kernel_t: str = "ou"          # 'ou'（默认）| 'matern32'（备选）
    gamma: float = 1.0
    period_p: int = 25            # tau_i = i*pi/p 采样点网格，fs 250 口径（与 v4a-fmt 保持）
    iso: float = 1e-2             # 白噪折叠进时间核：K'_t = K_t + iso*I（保持 Kronecker）
    pp_jitter: float = 1e-4       # K'_t,pp 求逆前额外 +1e-4*I
    sigma_s: float = 0.080        # Ricker 3D 空间核尺度（米，真实头皮坐标）
    jitter_s: float = 1e-3        # K_s <- K_s + jitter_s*I

    # ---- Mamba 速度网络（v4a-fmt 结构保持：低秩瓶颈 r=16 + trunk d_model=62 + LayerNorm；唯 norm 位换 adaLN）----
    d_model: int = 62             # trunk 宽度 = 通道数（v4a 保持）
    bottleneck_rank: int = 16     # v4a 架构字段：in_proj = Linear(79→16)→Linear(16→62) 纯线性低秩
    norm_type: str = "layernorm"  # 与 v4a-fmt 保持——adaLN 变体底层 norm 仍是 LayerNorm（两字段两维度）
    cond_adaln: str = "scale_shift"  # 本任务唯一结构变量（A 案）：5 个 norm 位全部换
                                  # modulate(LN(x), s(t), b(t))=(1+s)⊙LN(x)+b；(s,b) 由
                                  # time_embedding(t,16) 经逐 site Sequential(SiLU, Linear(16,124))
                                  # 生成（fork_rng 内构建 + 末层零初始化 ⇒ 初始严格等价纯 LayerNorm，
                                  # 同 seed 共享权重与 v4a-fmt 逐位相同）；v1 mamba.py/model.py
                                  # default-off 钩子 getattr 透传——旧变体无此字段 = "off"（零漂移）。
                                  # 参数 +10,540（5×(16·124+124)）→ 140,906（容量登记）
    n_layers: int = 4
    d_state: int = 16
    d_conv: int = 4
    expand: int = 2
    time_emb_dim: int = 16        # future token 的 t 正弦嵌入维度（调制信号复用该原始特征，不加投影）
    use_fast_path: bool = True    # mamba_ssm 官方 CUDA fast path（调制不触碰内核，零兼容风险）

    # ---- 采样率（fmt 口径；PSD 频轴/轨迹图标秒经 evaluate.py getattr 透传）----
    fs: float = 250.0             # 旧范式无此字段 → getattr 默认 500.0（零漂移）；Nyquist 125 Hz

    # ---- 复合损失（定义域 = future 128 列；与 v4a-fmt 一致）----
    w_mae: float = 1.0
    w_cons: float = 1.0
    w_tv: float = 0.1             # 纯平滑惩罚，不对照 GT 差分
    w_corr: float = 0.1
    corr_eps: float = 1e-6

    # ---- 训练（超参与纪律同 v4a-fmt：早停 10 + 恒定 lr 长期口径——单变量对比约束）----
    seed: int = 1234              # 训练 seed（划分 seed 独立 = 42）；fold0 单 seed（门禁：5-fold 须用户同意）
    batch_size: int = 256
    epochs: int = 100
    lr: float = 1e-3
    weight_decay: float = 1e-4
    lr_schedule: str = "constant"  # 同 v4a-fmt：lr 恒定 1e-3 无 cosine 退火（长期口径，保持）
    grad_clip: float = 0.5
    ema_decay: float = 0.999      # 短程训练口径（保持；EMA 影子滞后定量锚点见 design §2.3）
    sigma_min: float = 1e-4
    t_loc: float = -0.8           # t = sigmoid(z), z ~ N(t_loc, t_scale^2)（logit-normal，不用 U[0,1]）
    t_scale: float = 0.8
    t_clamp_min: float = 1e-4
    t_clamp_max: float = 0.9999
    cudnn_deterministic: bool = True
    cudnn_benchmark: bool = False
    log_every: int = 1            # 每 step 进度行（R9）

    # ---- 内部验证 / best 判据（同 v4a-fmt：val4 + 早停 10）----
    val_t_grid: tuple = (0.1, 0.3, 0.5, 0.7, 0.9)
    val_eps_seed: int = 2026      # 固定 eps（确定性复算，跨 epoch 可比）
    use_ema_for_val: bool = True  # 评估用 EMA 权重
    early_stop_patience: int = 10  # 同 v4a-fmt：best 判据连续 10 epoch 无新 best 即停（长期口径，保持）

    # ---- 采样（与 v4a-fmt 一致，唯无低通）----
    euler_steps: int = 16
    n_samples: int = 16
    gen_lowpass_hz: Optional[float] = None  # 与 v4a-fmt 同：生成侧不加 40 Hz 低通（直接观察模型
                                            # 是否自行待在带内）。日后在 250 Hz 重开低通必须传
                                            # fs=250（sample.py 默认 fs=500 坑）

    # ---- helper（与 v4a-fmt 相同实现）----
    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    def config_hash(self) -> str:
        """resolved config 的 sha256 前 12 位，写入 checkpoint。"""
        blob = json.dumps(self.to_dict(), sort_keys=True, default=str)
        return hashlib.sha256(blob.encode()).hexdigest()[:12]

    def dump_yaml(self, path: str) -> None:
        """零依赖 YAML dump（flat 标量/元组，tuple -> JSON list）。"""
        lines = ["# gpfm-mamba-v4a-fmt-adaln（edra_processed_fmt 250 Hz/LayerNorm/无低通，r=16+d_model=62，"
                 "5 norm 位 adaLN(t) scale/shift）resolved config（跑前 dump，reproducibility 契约）"]
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
