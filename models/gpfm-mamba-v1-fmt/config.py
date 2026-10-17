"""gpfm-mamba-v1-fmt 唯一配置源（薄变体，完整 config 副本；新预处理范式 fold0 训练任务 10-16-gpfm-mamba-fmt-f0）。

**数据范式**：`/root/autodl-tmp/edra_processed_fmt/`（EDRA release 自带 1–40 Hz@250 Hz 预处理，
48 例 HC24/MDD24，数据值零改动仅转 v4 契约格式；事实卡见任务 research/edra-fmt-dataset.md）。
身份审计确认与旧范式同源录制（outputs/edra-paradigm-compare/）。

**跨范式差异清单（多变量设计，非单变量；一切对比必随附——design §1.2/§2）**：
数据范式（fs 250/预处理 1–40 Hz 带限无 ICA·QC/48 例 vs 46）+ period_p 50→25（D3 保物理时长：
gpr_prior tau_i=i·π/p 为采样点网格，fs 减半 p 减半，OU e-folding 严格等价）+ 生成侧无 40 Hz
低通（D2：gen_lowpass_hz=None，真实侧本就带限，直接观察模型是否自行待在带内）+ norm
RMSNorm→LayerNorm（D4：5 个 norm 位 = 4×MambaBlock pre-norm + head norm 全换，容量 +80
bias 参数）+ 训练纪律长期口径（v1-fmt 侧叠加早停 10/恒定 lr/val4——对照纯净度不对称，跨范式
判读权重向 v4a-fmt 侧倾斜）。其余超参与 v1 逐字段一致（见 models/gpfm-mamba-v1/config.py 文件头）。

**划分固化（R2）**：48≠46 例 → 不可复用 v1/v4a 已固化 manifest（fixate_split 防漂移按设计报错）；
`split_manifest_override` 指向本目录 splits/ 自备 manifest（default-off 钩子），首跑固化
48 例 fold0 受试者级 5-fold 分层（seed 42）+ 内层 n_val_subjects=4（7/1/2 长期口径语义，
实际整数分配以固化 manifest 为准登记）；v4a-fmt 用同参数生成 ⇒ manifest 逐字节一致（单测断言）。
`subject_filter='status_ok'`（fmt manifest 48 行全 ok；qc_pass 字段留空——该口径本数据集不可用）。

**GC 阻塞（D1/R4）**：genGC vs real GC 一致性对比暂时不出现在评估产物——evaluate 包装带
`--skip-gc-only`（v1 evaluate.py default-off 钩子；PSD fs=250 与空间归因照跑，summary 记
"skipped(blocked)"，无任何 GC 对比文件）。

依赖方式（path-isolation 薄变体契约，v4a 式单遮蔽）：本目录仅 config.py + train.py/evaluate.py/
fc_metric.py 入口包装；model/mamba/data/gpr_prior/losses/train/sample/evaluate/gc_psd 等
模块只读复用 v1（经 sys.path）。输出落本任务 outputs 根 `/root/npi/outputs/gpfm-mamba-fmt-f0/`
（与 v4a-fmt 共享，run_id 前缀隔离），checkpoint 名 `gpfm-mamba-v1-fmt-{best,last}.pt`。

**250 Hz 已知坑（design 预登记）**：sample.py 低通 `brickwall_lowpass(out, cut_hz)` 用默认
fs=500——本 config gen_lowpass_hz=None 恒不触发；日后重开低通必须传 fs=250。diag_hf.py
（fs=500+旧 run id 硬编码）fmt 禁用。
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class Config:
    # ---- 路径（path-isolation 契约；本任务新 outputs 根，与 v4a-fmt 共享、run_id 前缀隔离）----
    data_root: str = "/root/autodl-tmp/edra_processed_fmt"       # 新预处理范式数据集（250 Hz）
    outputs_root: str = "/root/npi/outputs/gpfm-mamba-fmt-f0"    # 本任务 outputs 根（禁写旧根 gpfm-mamba-f0）
    variant_name: str = "gpfm-mamba-v1-fmt"                      # checkpoint 主名 + config_hash 组成部分

    # ---- 数据与划分（R2：48 例新固化；其余口径同 v1）----
    n_channels: int = 62
    past_len: int = 384
    future_len: int = 128
    stride: int = 64
    n_folds: int = 5
    fold: int = 0                 # study1 前置验证固定 fold0
    split_seed: int = 42          # 受试者级 5-fold 分层（MDD/HC）划分 seed
    subject_filter: str = "status_ok"  # fmt manifest 48 行全 status=ok（qc_pass 留空不可用）
    n_val_subjects: int = 4       # 训练折内内部验证受试者（分层每类 2 例）——长期口径 7/1/2 语义
                                  # （实际整数分配以首跑固化 manifest 为准；旧 v1 为 2=34/2/10）
    val_subject_seed: int = 42
    fixate_split: bool = True     # 划分固化：首建写入本变体 splits/，之后校验一致、漂移报错
    split_manifest_override: str = "/root/npi/models/gpfm-mamba-v1-fmt/splits/fold0_status_ok_seed42_val4.json"
                                  # v1 data.py::split_manifest_path 的 default-off 钩子——48≠46 例
                                  # 不可复用旧变体 manifest，须本目录首跑固化；v4a-fmt 同参数生成
                                  # ⇒ 内容逐字节一致（单测断言）
    num_workers: int = 4

    # ---- GPR 条件先验（design §2.1；与 v1 一致，唯 p 减半）----
    kernel_t: str = "ou"          # 'ou'（默认）| 'matern32'（备选）
    gamma: float = 1.0
    period_p: int = 25            # D3：tau_i = i*pi/p 采样点网格，fs 500→250 时 p 50→25 保物理时长
                                  # （OU e-folding = p/(pi*fs) 秒严格等价；先验与数据同时变，判读随附差异清单）
    iso: float = 1e-2             # 白噪折叠进时间核：K'_t = K_t + iso*I（保持 Kronecker）
    pp_jitter: float = 1e-4       # K'_t,pp 求逆前额外 +1e-4*I
    sigma_s: float = 0.080        # Ricker 3D 空间核尺度（米，真实头皮坐标）
    jitter_s: float = 1e-3        # K_s <- K_s + jitter_s*I

    # ---- Mamba 速度网络（design §3；结构同 v1，唯 norm 类型换）----
    d_model: int = 16
    norm_type: str = "layernorm"  # D4（用户 2026-10-16 拍板）：5 个 norm 位（4×MambaBlock pre-norm +
                                  # head norm）全部 RMSNorm→LayerNorm；v1 mamba.py/model.py default-off
                                  # 钩子 getattr 透传——旧变体无此字段 = "rmsnorm"（结构与 config_hash
                                  # 零漂移）。容量 +80 bias 参数（5×16）登记；语义 = 均值中心化归一
    n_layers: int = 4
    d_state: int = 16
    d_conv: int = 4
    expand: int = 2
    time_emb_dim: int = 16        # future token 的 t 正弦嵌入维度
    use_fast_path: bool = True    # mamba_ssm 官方 CUDA fast path

    # ---- 采样率（fmt 口径新增；PSD 频轴/轨迹图标秒经 evaluate.py getattr 透传）----
    fs: float = 250.0             # 旧范式无此字段 → getattr 默认 500.0（零漂移）；Nyquist 125 Hz

    # ---- 复合损失（design §2.3，定义域 = future 128 列；与 v1 一致）----
    w_mae: float = 1.0
    w_cons: float = 1.0
    w_tv: float = 0.1             # 纯平滑惩罚，不对照 GT 差分
    w_corr: float = 0.1
    corr_eps: float = 1e-6

    # ---- 训练（design §2.2 / §5；超参同 v1，调度与早停为长期口径）----
    seed: int = 1234              # 训练 seed（划分 seed 独立 = 42）；fold0 单 seed（门禁：5-fold 须用户同意）
    batch_size: int = 256
    epochs: int = 100
    lr: float = 1e-3
    weight_decay: float = 1e-4
    lr_schedule: str = "constant"  # 长期口径（v1 train.py default-off 钩子）：lr 恒定 1e-3 无 cosine
                                   # 退火（scheduler 对象保留，ckpt/resume 契约不变）；v1 默认 cosine
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

    # ---- 内部验证 / best 判据（design §2.3；判据同 v1，早停为长期口径）----
    val_t_grid: tuple = (0.1, 0.3, 0.5, 0.7, 0.9)
    val_eps_seed: int = 2026      # 固定 eps（确定性复算，跨 epoch 可比）
    use_ema_for_val: bool = True  # 评估用 EMA 权重
    early_stop_patience: int = 10  # 长期口径（v1 train.py default-off 钩子）：best 判据（val4 总 loss，
                                   # 固定 t 网格确定性复算+EMA）连续 10 epoch 无新 best 即停；v1 无字段=禁用

    # ---- 采样（§5/§6；与 v1 一致，唯无低通）----
    euler_steps: int = 16
    n_samples: int = 16
    gen_lowpass_hz: Optional[float] = None  # D2：生成侧不加 40 Hz 低通（真实侧本就 1–40 Hz 带限；
                                            # 直接观察模型是否自行待在带内）。v1 经 getattr 默认 40.0。
                                            # 日后在 250 Hz 重开低通必须传 fs=250（sample.py 默认 fs=500 坑）

    # ---- helper（与 v1 相同实现）----
    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    def config_hash(self) -> str:
        """resolved config 的 sha256 前 12 位，写入 checkpoint。"""
        blob = json.dumps(self.to_dict(), sort_keys=True, default=str)
        return hashlib.sha256(blob.encode()).hexdigest()[:12]

    def dump_yaml(self, path: str) -> None:
        """零依赖 YAML dump（flat 标量/元组，tuple -> JSON list）。"""
        lines = ["# gpfm-mamba-v1-fmt（新预处理范式 edra_processed_fmt，250 Hz/p=25/无低通/LayerNorm）"
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
