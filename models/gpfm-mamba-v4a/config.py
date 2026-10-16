"""gpfm-mamba-v4a 唯一配置源（薄变体，完整 config 副本，v2 基础 + v4-A 低秩瓶颈字段）。

**v4-A = H3 假设的「接口强制混合 × 容量保持」缺失格变体（用户 2026-10-16 拍板复活，design §1/§2）**：
- 单变量设计：与 v2 唯一架构差异 = in_proj `Linear(79→62)` 满秩分解为
  `Linear(79→r=16) → Linear(16→62)` **纯线性低秩瓶颈**（两层间无激活——两层复合 = rank≤16
  线性映射，全部 62 通道逐 token 信息经 16 维共享潜表示，「接口强制跨通道混合」的干净单变量；
  加激活 = 1 隐层 MLP 第二语义，评审背书排除）；PyTorch 默认初始化（两层复合谱窄于单层为
  预登记风险项）；trunk（4×MambaBlock d_model=62）/ head（62→62）/ 其余全部与 v2 逐字段一致
  （容量不变：参数 132.4k→129.8k ≈ 0.98× v2，in_proj 4960→2334）。
- 瓶颈位置语义：v1 的 d16 贯穿 trunk 全深 vs v4-A 仅输入接口——GC 部分回升时残留归因
  「深度持续混合」（C 方案域，不追）。
- r 扫描 {8,32} 为条件扩展（主 Run 判读后议）：经 wrapper `--rank` 运行时改写本 config 实例
  （config.py 文件默认唯一 16，薄变体契约），run_id 须含匹配 `rXX` 标识（防 r 串用；
  resume/评估的 config_hash 校验天然防不同 r 的 ckpt 误用）。

**方向变更登记（用户 2026-10-16 拍板）：v4-B（CI 骨干 + cMLP head）放弃，代码与 Run 记录存档
不删（models/gpfm-mamba-v4/ 与 outputs/gpfm-mamba-f0/v4b-*）**。结论摘要：d_ch 1→8 剂量对照
（Run 006–010）证「容量必要而不充分」——GC 幅值/峰频/逐通道 r 恢复但 GC r 仅 0.133→0.163
（未达部分支持线 0.18）、指标 3 未达标（δ–β 0.519>0.5，生成过散 std_t +74%）、CRPS +8.2%。
v4-A 与 v4-B 互为对照臂（接口混合 vs 顶层显式读出），共享下述训练口径。

**口径继承清单（当前累计口径，均用户 2026-10-16 拍板；对 v1/v2 对比时必随附）**：
① 划分 32/4/10（7/1/2 长期口径，v1/v2 为 34/2/10——本 config `n_val_subjects=4` +
   `split_manifest_override` 指向本目录 splits/ 自备 val4 固化 manifest，内容与 v4b 的
   val4 manifest 逐字节一致由单测断言；v1/v2/v3 config 无 override 字段 = 路径与
   config_hash 零漂移）；② 早停 patience=10（v1 train.py default-off 钩子，v1/v2 恒 100ep）；
   ③ lr 恒定 1e-3 无 cosine（default-off 钩子同上）；④ gen_lowpass_hz=40.0 显式入 config
   （v1/v2 经 getattr 默认 40.0 同值，仅此处进 hash）。test 折与 v1 逐 ID 相同 ⇒ test 指标可比。

**命名与别名**：目录 v4a；用户口头「v4」「v4-A」均指本变体（仓库 v4 已被放弃存档的 v4-B 占用）。

依赖方式（path-isolation 薄变体契约，v2 式单遮蔽）：本目录仅 config.py + train.py/evaluate.py
入口包装（只遮蔽 config，模型经 v1 model.py 的 bottleneck_rank default-off 钩子构建）；
data/gpr_prior/mamba/model/losses/train/sample/evaluate 等模块只读复用 v1（经 sys.path）。
输出落同一任务 outputs 根 `/root/npi/outputs/gpfm-mamba-f0/`，run_id 一律前缀 `v4a-`
（包装脚本强制校验），checkpoint 名 `gpfm-mamba-v4a-{best,last}.pt` 与 v1/v2 天然隔离。
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
    variant_name: str = "gpfm-mamba-v4a"  # checkpoint 主名 + config_hash 组成部分（区别于 v1/v2）

    # ---- 数据与划分（design §4；2026-10-16 口径变更 32/4/10 → 校验 v4a 自备固化 manifest，见下 override）----
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
    split_manifest_override: str = "/root/npi/models/gpfm-mamba-v4a/splits/fold0_status_ok_seed42_val4.json"
                                  # v1 data.py::split_manifest_path 的 default-off 钩子（2026-10-16 口径变更）：
                                  # 指向 v4a 自备 val4 固化 manifest（首建写入、之后校验一致；内容与 v4b 的
                                  # val4 manifest 逐字节一致由单测 ⑧ 断言——同 split_fold 参数决定性输出）；
                                  # v1/v2/v3 无此字段 = 路径不变、零漂移。
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

    # ---- Mamba 速度网络（design §2；**单变量：in_proj 满秩 → r=16 纯线性低秩**，d_model=62 保持 v2 值）----
    d_model: int = 62             # trunk 宽度 = 通道数（与 v2 一致，容量不变——v4-A 只动输入接口）
    bottleneck_rank: int = 16     # ←←← 本变体唯一架构字段：in_proj = Linear(79→16)→Linear(16→62)
                                  #      纯线性低秩（无激活，rank≤16）；r 扫描 {8,32} 经 wrapper --rank
                                  #      运行时改写（config.py 文件默认唯一 16）；None 语义不用于本变体
    n_layers: int = 4
    d_state: int = 16
    d_conv: int = 4
    expand: int = 2
    time_emb_dim: int = 16        # future token 的 t 正弦嵌入维度（保持 16，不随 d_model 变——单变量原则）
    use_fast_path: bool = True    # mamba_ssm 官方 CUDA fast path

    # ---- 复合损失（design §2.3，定义域 = future 128 列；与 v1/v2 一致）----
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
                                   # lr 已恒定（lr_schedule="constant"），无早停-lr 交互注记；
                                   # best/last checkpoint 语义不变

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
        lines = ["# gpfm-mamba-v4a (v4-A: 低秩输入瓶颈 r=16, d_model=62) resolved config（跑前 dump，reproducibility 契约）"]
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
