#!/usr/bin/env python3
"""flow_exit_gnn_symw.py — W 强制对称化变体 optimizer（任务 10-25-latent-flow-exit-gnn-symw）.

语义（用户指定字面语义，prd R2）：每个 optimizer step 之后将可学习残差
矩阵 W 投影到对称子空间 W ← (W+Wᵀ)/2，训练全程维持 W 严格对称。
A_eff = γ_M·M + γ_W·W（LearnableMexHatMixer.effective_adjacency，
oc_latent_flow.py:426-441）：M 为对称距离核、W 为 128×128 可学习残差
（:414-424，零初始化、包内刻意不 symmetrize——A_eff 的全部有向自由度
都在 W 上）⇒ W 对称 ⇒ A_eff 对称 ⇒ 消息传递退化为无向图混合（本变体
的有意设计，非副作用；科学含义登记见 prd 背景节）。

为什么挂 optimizer 层（而非训练循环 / 包体回调）：
  - BaseTrainer 的 optimizer 为外部注入（trainer.py:180-181），train_epoch
    逐步调用 self.optimizer.zero_grad()/step()（:221,241）且无逐步回调
    钩子——逐步投影的正确挂点 = Adam 子类 step() 的尾部；
  - 经 optimizer 子类实现 ⇒ 训练循环零复制、包体 latent_flow/ 零改动
    （延续 exit-gnn R4 纪律），wrapper 只换注入的 optimizer 对象。

投影目标唯一 = model.flow.graph_mixer.W（本任务恰一个）：exit-gnn 变体
（flow_exit_gnn.py）下入口/出口共享单份 A_eff ⇒ 恰一个 W；gaussian 分支
（GraphChannelMixer）与 dense 分支无该参数 ⇒ wrapper 守卫互斥拒跑。

投影在 fp32 param 上精确对称：(a+b)/2 浮点加法交换律成立（IEEE 754
逐元素 a+b == b+a）⇒ 投影后 torch.equal(W, W.t()) 级严格对称（非数值
容差近似）。A_eff = γ_M·M + γ_W·W 在 M 逐元素对称时同样 torch.equal
级；注意 M 的对称性受 cdist 舍入影响——torch.cdist（CPU mm 路径，
P>25）对一般浮点坐标本身带 ~1 ulp 不对称（实测 pairwise_d 最大
|d-dᵀ| ≈ 4.8e-7），自验因此取小整数坐标（距离中间量全部精确可表示
⇒ cdist 逐位对称）；真实 geometry 训练中 A_eff 对称度受该 M 侧地板
约束（~2e-7），而 W 本体（R5 训后断言对象）始终 torch.equal 级严格
对称。

直接运行本模块 = CPU 快速自验（内存内随机张量，不读数据不训练；
冒烟/正式 Run 的训后 W 对称断言属 wrapper 侧 AC3/AC4，不在此处）：
  python3 models/latent-flow-v1/flow_exit_gnn_symw.py
"""
from __future__ import annotations

import os

# --- BLAS 守卫：必须在任何 numpy/scipy/torch 导入之前（npi 环境约束） ---
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "8")

import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import torch  # noqa: E402

VARIANT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(VARIANT_DIR))  # 使 latent_flow 可导入

from latent_flow.models.oc_latent_flow import (  # noqa: E402
    FlowVelocityNet,
    LearnableMexHatMixer,
)
from flow_exit_gnn import FlowVelocityNetExitGNN  # noqa: E402  (变体子类，同目录)


# =============================================================================
# SymWAdam：Adam 子类，step() 尾部对受管参数原位投影（断言即拒跑惯例）
# =============================================================================

class SymWAdam(torch.optim.Adam):
    """每步 Adam 更新后将受管参数投影到对称子空间 W ← (W+Wᵀ)/2。

    __init__ 的 sym_params = 受管投影参数列表（本任务恰一个
    graph_mixer.W），仅存 self._sym_params——不进 defaults、不进 state，
    避免污染 optimizer state 键面（state 按参数对象索引，持久化/装载
    行为与裸 Adam 一致）；step() = super().step(closure) 先完成正常
    Adam 更新，随后在 no_grad 下对每个受管参数
    p.data.copy_((p.data + p.data.t()) * 0.5) 原位投影，返回值（loss）
    原样透传。受管参数须为二维方阵（投影语义前提，构造时断言拒跑）。
    """

    def __init__(self, params, lr=1e-3, sym_params=None):
        self._sym_params = list(sym_params or [])
        for p in self._sym_params:
            assert p.dim() == 2 and p.shape[0] == p.shape[1], (
                f"sym_params 须为二维方阵参数（W ← (W+Wᵀ)/2 投影语义），"
                f"got shape {tuple(p.shape)}")
        super().__init__(params, lr=lr)

    def step(self, closure=None):
        loss = super().step(closure)
        with torch.no_grad():
            for p in self._sym_params:
                p.data.copy_((p.data + p.data.t()) * 0.5)
        return loss


# =============================================================================
# CPU 快速自验（prd AC2；内存内随机张量，不读数据、不训练——静态口径）
# =============================================================================

def _self_check() -> int:
    torch.manual_seed(42)
    # ① 随机非对称 w_init 起步（randn a.s. 非对称，先证伪前提），经一步
    #    SymWAdam（Adam 更新 + 尾部投影）后 W 严格对称。coords 取小整数
    #    （精确可表示 ⇒ cdist 逐位对称，见模块 docstring 的 cdist 1-ulp
    #    登记）——A_eff 的 torch.equal 断言才只检验 W 投影的传导，不被
    #    M 侧舍入地板污染
    w_init = torch.randn(128, 128)
    assert not torch.equal(w_init, w_init.t()), \
        "前提失效：randn w_init 竟然对称（a.s. 不可能，换 seed 重跑）"
    mixer = LearnableMexHatMixer(n_channels=128, latent_dim=64, hidden_ch=128,
                                 coords=torch.randint(0, 5, (128, 3)).float(),
                                 w_init=w_init)
    W = mixer.W
    assert not torch.equal(W, W.t()), "构造后 W 应保持 w_init 的非对称"
    # 随机掩码梯度 a.s. 非对称 → Adam 更新非对称 → 仅靠 step 尾部投影回对称
    (W * torch.randn_like(W)).sum().backward()
    SymWAdam([W], lr=1e-3, sym_params=[W]).step()
    assert torch.equal(W, W.t()), \
        "一步 SymWAdam 后 W 未严格对称（投影未生效）"
    # 投影发生在 Adam 更新之后而非替代更新：结果 ≠ 对初值的直接投影
    assert not torch.equal(W.detach(), (w_init + w_init.t()) * 0.5), \
        "W 等于对初值的直接投影——Adam 更新疑似未发生（step 尾部序错）"
    # ② A_eff = γ_M·M + γ_W·W：M 对称（距离核）+ W 对称 ⇒ A_eff 对称
    A = mixer.effective_adjacency()
    assert torch.equal(A, A.t()), \
        "A_eff 未对称（M 对称 + W 对称 ⇒ A_eff 必须对称）"
    # ③ 对照参数量：FlowVelocityNetExitGNN vs 包内 FlowVelocityNet 同参构建
    #    差恰 +33,024（exit-gnn 既有记账复验，架构面与本变体正交）
    coords = torch.randn(128, 3)
    base = FlowVelocityNet(graph_mode=True, graph_kernel="mexhat", coords=coords)
    net = FlowVelocityNetExitGNN(graph_mode=True, graph_kernel="mexhat",
                                 coords=coords)
    n_base = sum(p.numel() for p in base.parameters())
    n_exit = sum(p.numel() for p in net.parameters())
    assert n_exit - n_base == 33024, \
        f"参数净增 {n_exit - n_base} != 预期 33024（exit-gnn 记账）"
    # ④ SymWAdam 零参数增量：投影不注册任何新参数（step 前后模型参数量
    #    不变），受管引用即模型内那一份（_sym_params[0] is graph_mixer.W），
    #    整网语境下投影同样把 W 拉回严格对称
    n_before = sum(p.numel() for p in net.parameters())
    opt = SymWAdam([p for p in net.parameters() if p.requires_grad],
                   lr=1e-3, sym_params=[net.graph_mixer.W])
    assert opt._sym_params[0] is net.graph_mixer.W, \
        "_sym_params[0] 不是 graph_mixer.W 本体（受管引用错位）"
    (net.graph_mixer.W * torch.randn_like(net.graph_mixer.W)).sum().backward()
    opt.step()
    n_after = sum(p.numel() for p in net.parameters())
    assert n_after == n_before, \
        f"SymWAdam 引入参数增量 {n_after - n_before}（投影必须零参数增量）"
    assert torch.equal(net.graph_mixer.W, net.graph_mixer.W.t()), \
        "整网语境下投影后 W 未严格对称"
    print(f"[self-check] W 投影 OK：randn 非对称起 + 随机掩码梯度 → 一步 "
          f"SymWAdam（Adam 更新 + 尾部投影）后 torch.equal(W, W.t()) 严格对称")
    print(f"[self-check] A_eff OK：M（对称距离核）+ W（对称）⇒ "
          f"effective_adjacency() torch.equal 级对称")
    print(f"[self-check] params：base {n_base:,} -> exit-gnn {n_exit:,}"
          f"（净 +{n_exit - n_base:,}，预期 +33,024）")
    print(f"[self-check] SymWAdam 零参数增量 OK：step 前后模型参数量 "
          f"{n_before:,} 不变，_sym_params[0] is graph_mixer.W，整网投影后 "
          f"W 对称")
    return 0


if __name__ == "__main__":
    sys.exit(_self_check())
