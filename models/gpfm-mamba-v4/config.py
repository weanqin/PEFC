"""gpfm-mamba-v4 唯一配置源（薄变体，完整 config 副本，v2 基础 + v4-B 结构字段）。

**v4-B = 用户 2026-10-16 指定架构（H3 跨通道交互假设变体，design §2）**：
- 通道独立（channel-independent, CI）编码 + 时序骨干：逐通道特征 18 维
  [本通道值 1 ‖ 段类型嵌入 1 ‖ t 正弦嵌入 16] → 共享 Linear(18→d_ch)（权重跨通道共享，
  PatchTST 式）→ 4×MambaBlock(d_ch, d_state=16, d_conv=4, expand=2) 在 (B×62, 512, d_ch)
  上逐通道独立 selective scan——trunk 严格零跨通道混合，通道身份保留为独立数据流。
- 顶层 cMLP 组件式混合读出头（**唯一跨通道混合层**，Tank 2021 / Li & Zhou 2025 移植）：
  截取 future 128 位置 + 逐通道 RMSNorm(d_ch) → 同时间步汇聚 h_t∈R^{62·d_ch} →
  每个目标通道 i 一个独立子 MLP：z_{t,i}=σ(Σ_j W^{(i),1}_{:,j} h_{t,j}+b¹_i)、
  v_θ(t,i)=W^{(i),2} z_{t,i}+b²_i；W¹ 按源通道列分块（可辨识有向读出）。
- 用户拍板超参（2026-10-16）：**d_ch=1（本 config 字段名沿用 `d_model`，语义=逐通道宽度
  d_ch——特例风险登记 design §9.10：RMSNorm(1) 退化为 sign(x)·w、trunk 容量塌缩）**、
  d_z=16、σ=SiLU、CI 权重共享、**不加 group lasso**（损失与 v1/v2 逐字一致；
  训后免费算内部 readout M[j,i]=‖W^{(i),1}_{:,j}‖_F 仅诊断）。
- 参数量（d_ch=1, d_z=16，design §3.2）：in_proj 19 + trunk 4×121=484 + head 前
  RMSNorm(1) 1 + head 62×1025=63,550 + seg 嵌入 2 = **64,056**（head 占 99.2%）。

**命名与别名**：目录 v4；用户口头「v3」「v4-B」均指本变体（仓库 v3 已被 v3lp 占用）。

训练/评估口径与 v2 逐字段一致（seed 1234、epochs 100、AdamW lr 1e-3 wd 1e-4、clip 0.5、
EMA 0.999、t~logit-normal(−0.8,0.8)、损失 {1,1,0.1,0.1}、status_ok 46 例）——结构变化
（CI 骨干 + cMLP head）之外，2026-10-16 另有三项口径变更登记（划分 32/4/10、早停
patience=10、lr 恒定，分别见下方各块与字段注释）。
**lr 调度口径变更（用户 2026-10-16 拍板：取消 cosine 退火，lr 恒定 1e-3）**：v1/v2 为
cosine（T_max=epochs）；实现 = v1 train.py default-off 钩子 `lr_schedule`（v1/v2/v3 config
无此字段 = "cosine"，行为与 config_hash 零漂移；"constant" = LambdaLR 恒 1.0，scheduler
对象保留兼容 ckpt/resume 状态契约）。不可比注记：优化轨迹与 v1/v2 不同（恒定 lr vs 退火
到 ~0），与早停/32/4 划分一并列入 H3 判读的口径差异清单。
**划分口径变更（用户 2026-10-16 拍板，长期口径 train/val/test = 7/1/2；本任务整数分配 32/4/10）**：
训练折 36 例内部由 train 34/val 2 改为 **train 32 / val 4**（分层 2 HC + 2 MDD，
val_subject_seed=42 不变）；fold0 外层与 test 10 例不动（保 v1/v2/v4 test 可比）。
实现 = v4 自备固化 manifest（v1 data.py `split_manifest_path` 的 default-off 钩子
`split_manifest_override` 指向本目录 splits/；v1/v2/v3 无此字段 = 路径与 config_hash 零漂移）。
**不可比注记**：① v4-B 训练数据 32 例 vs v1/v2 的 34 例（−6%）；② best 判据基于 val4
（v1/v2 为 val2）；③ test 折相同 ⇒ test 指标可比；④ 既有 bench（Run 002/003）val 数值为
val2 口径，与本口径 val 轨迹不直接可比。
正式口径显式化：`gen_lowpass_hz=40.0`（输出侧 40 Hz 低通，evaluation-calibers.md；
v1/v2 经 getattr 默认值透传，v4 起显式入 config 进 hash）。

依赖方式（path-isolation 薄变体契约 + 双遮蔽扩展，design §4.2）：
- 本目录含 config.py + model.py（v4-B 新结构；time_embedding/EMA 复制自 v1，勿单独演化）
  + train.py/evaluate.py 入口包装（config+model 双遮蔽）；mamba/data/gpr_prior/losses/
  train/sample/evaluate 等模块只读复用 v1（经 sys.path，v1 代码零改动）。
- 输出落同一任务 outputs 根 `/root/npi/outputs/gpfm-mamba-f0/`，run_id 一律前缀 `v4b-`
  （包装脚本强制校验），checkpoint 名 `gpfm-mamba-v4b-{best,last}.pt` 与 v1/v2 天然隔离。
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class Config:
    # ---- 路径（path-isolation 契约；outputs 根与 v1/v2 同任务共享，run_id 前缀隔离）----
    data_root: str = "/root/autodl-tmp/edra_process_v4"
    outputs_root: str = "/root/npi/outputs/gpfm-mamba-f0"
    variant_name: str = "gpfm-mamba-v4b"  # checkpoint 主名 + config_hash 组成部分（区别于 v1/v2）

    # ---- 数据与划分（design §4；2026-10-16 口径变更 32/4/10 → 校验 v4 自备固化 manifest，见下 override）----
    n_channels: int = 62
    past_len: int = 384
    future_len: int = 128
    stride: int = 64
    n_folds: int = 5
    fold: int = 0                 # study1 前置验证固定 fold0
    split_seed: int = 42          # 受试者级 5-fold 分层（MDD/HC）划分 seed
    subject_filter: str = "status_ok"  # 受试者过滤口径（2026-10-12 用户拍板 b）：'status_ok'（46 例）| 'qc_pass'（13 例）
    n_val_subjects: int = 4       # 训练折内内部验证受试者（分层每类 2 例）——用户 2026-10-16 拍板长期口径
                                  # train/val/test=7/1/2（整数分配 32/4/10）；此前 v1/v2 为 2（34/2/10）
    val_subject_seed: int = 42
    fixate_split: bool = True     # 划分固化：校验本变体 splits/ 自备 manifest（见 split_manifest_override）
    split_manifest_override: str = "/root/npi/models/gpfm-mamba-v4/splits/fold0_status_ok_seed42_val4.json"
                                  # v1 data.py::split_manifest_path 的 default-off 钩子（2026-10-16 口径变更）：
                                  # 指向 v4 自备 val4 固化 manifest；v1/v2/v3 无此字段 = 路径不变、零漂移。
                                  # 不可比注记：train 32 vs v1/v2 的 34 例；best 判据 val4 vs val2；test 相同
    num_workers: int = 4

    # ---- GPR 条件先验（design §2.1，时空可分 K_s x K_t；与 v1/v2 一致）----
    kernel_t: str = "ou"          # 'ou'（默认）| 'matern32'（备选）
    gamma: float = 1.0
    period_p: int = 50            # tau_i = i*pi/p（alpha 节律周期，禁止整窗 [0,1] 归一）
    iso: float = 1e-2             # 白噪折叠进时间核：K'_t = K_t + iso*I（保持 Kronecker）
    pp_jitter: float = 1e-4       # K'_t,pp 求逆前额外 +1e-4*I
    sigma_s: float = 0.080        # Ricker 3D 空间核尺度（米，真实头皮坐标）
    jitter_s: float = 1e-3        # K_s <- K_s + jitter_s*I

    # ---- Mamba 速度网络 v4-B（design §2/§3；结构变化 = CI 骨干 + cMLP head）----
    d_model: int = 1              # ←←← v4-B 语义 = 逐通道独立宽度 d_ch（用户拍板 2026-10-16；
                                  #      非 v1/v2 的全局混合宽度。d_ch 升档属口径变更须用户确认，
                                  #      且须 config+model 双处同改并重跑全部单测）
    n_layers: int = 4
    d_state: int = 16
    d_conv: int = 4
    expand: int = 2
    time_emb_dim: int = 16        # future token 的 t 正弦嵌入维度（逐通道特征 = 1+1+16=18）
    use_fast_path: bool = True    # mamba_ssm 官方 CUDA fast path
    d_z: int = 16                 # cMLP head 子 MLP 隐宽（用户拍板；与 model 类默认同值，双处一致性防线）
    head_act: str = "silu"        # head 激活 σ（用户拍板；与 model 类默认同值）

    # ---- 复合损失（design §2.3，定义域 = future 128 列；与 v1/v2 一致，不加 group lasso）----
    w_mae: float = 1.0
    w_cons: float = 1.0
    w_tv: float = 0.1             # 纯平滑惩罚，不对照 GT 差分
    w_corr: float = 0.1
    corr_eps: float = 1e-6

    # ---- 训练（design §2.2 / §5；与 v1/v2 一致，lr 调度除外——2026-10-16 口径变更恒定 lr）----
    seed: int = 1234              # 训练 seed（划分 seed 独立 = 42）
    batch_size: int = 256
    epochs: int = 100
    lr: float = 1e-3
    weight_decay: float = 1e-4
    lr_schedule: str = "constant"  # lr 调度（用户 2026-10-16 拍板：取消 cosine 退火，lr=1e-3 恒定不变）。
                                   # v1 train.py default-off 钩子 getattr 透传——v1/v2/v3 config 无此字段
                                   # = "cosine"（CosineAnnealingLR T_max=epochs，行为与 config_hash 零漂移）；
                                   # "constant" = LambdaLR 恒 1.0（scheduler 对象保留，ckpt/resume 状态契约不变）。
                                   # 不可比注记：优化轨迹与 v1/v2 不同（恒定 lr vs 退火到 ~0），随 H3 判读登记
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

    # ---- 内部验证 / best 判据（design §2.3；与 v1/v2 一致）----
    val_t_grid: tuple = (0.1, 0.3, 0.5, 0.7, 0.9)
    val_eps_seed: int = 2026      # 固定 eps（确定性复算，跨 epoch 可比）
    use_ema_for_val: bool = True  # 评估用 EMA 权重
    early_stop_patience: int = 10  # 早停（用户 2026-10-16 拍板）：best 判据（val4 总 loss，固定 t 网格
                                   # 确定性复算+EMA）连续 10 个 epoch 无新 best 即停止训练；v1/v2 config
                                   # 无此字段 = 禁用（v1 train.py default-off 钩子 getattr 透传，零漂移）。
                                   # （登记时的「cosine T_max=100 不随早停调整、提前停止 lr 未退火到底」
                                   # 交互注记，已随同日 lr 恒定口径变更（lr_schedule="constant"）作废：
                                   # lr 全程恒定，早停不再有 lr 未退火交互；best/last checkpoint 语义不变）

    # ---- 采样（§5/§6；与 v1/v2 一致）----
    euler_steps: int = 16
    n_samples: int = 16
    gen_lowpass_hz: float = 40.0  # 生成样本输出侧 40 Hz 砖墙低通（正式口径显式入 config；
                                  # v1/v2 经 sample.py getattr 默认 40.0 同值，仅此处进 hash）

    # ---- helper（与 v1/v2 相同实现）----
    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    def config_hash(self) -> str:
        """resolved config 的 sha256 前 12 位，写入 checkpoint。"""
        blob = json.dumps(self.to_dict(), sort_keys=True, default=str)
        return hashlib.sha256(blob.encode()).hexdigest()[:12]

    def dump_yaml(self, path: str) -> None:
        """零依赖 YAML dump（flat 标量/元组，tuple -> JSON list）。"""
        lines = ["# gpfm-mamba-v4 (v4-B: CI backbone d_ch=1 + cMLP head d_z=16) resolved config（跑前 dump，reproducibility 契约）"]
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
