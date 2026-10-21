"""GpfmProbeSite — 07-04 SignalProbeSite 契约的 gpfm-v5-main 适配（任务 10-21-ec-globaltau-v5main）。

与 07-04 `latent_flow/attribution/bisect_sigma.SignalProbeSite` 的逐字同构点（用户钉定「完全复用」）：
- 扰动算子：x^p 通道 a 加性高斯 σ·ξ_a + 固定方向偏置 μ_bias^a = 0.5·std_a（std 母体 = 实际入模
  x^p 窗集 std(dim=(0,1))，同 07-04 ch_std 定义）；偏置不随 σ 缩放。
- ξ 播种：seed+source（07-04 同式），同一 ξ 跨该源全部 σ 复用（幅度-only 比较）。
- 度量：per-target MSE / mean-shift，对 (窗, 时间) 取均值，对无扰动确定性基线；
  复用 07-04 原语 `per_target_mse` / `per_target_mean_diff`（(B,T,C) 约定，本适配在采样输出侧
  做一次 (B,C,F)→(B,F,C) 摆位，度量本身零改动）。
- 数值口径：bf16 autocast 域包裹采样前向（briefing 拍板，镜像 07-04）；扰动算术在输入侧 fp32。

代理脑特有契约（design §2/§4）：
- 生成噪声 (eps,eta) 被试级固定（seed*10000+sid 播种一次），跨源/跨 σ 逐位一致——由 runner 构造
  并注入，本类内断言其存在且不随调用改变；
- 入模 x_0 恒为 x^p_pert（扰动版）；x^f 仅由采样生成（不入模）。
"""

import torch

from latent_flow.attribution.bisect_sigma import per_target_mean_diff, per_target_mse


class GpfmProbeSite:
    """单被试探针位：扰动 x^p 通道 a → 采样 x^f_pert → 对基线逐靶 MSE / mean-shift。"""

    def __init__(self, sampler, x_p_subject: torch.Tensor,
                 eps: torch.Tensor, eta: torch.Tensor,
                 batch_cap: int = 1000, seed: int = 42,
                 use_biased_noise: bool = True, amp_dtype: torch.dtype = torch.bfloat16):
        self.sampler = sampler
        self.device = x_p_subject.device
        self.x_obs = x_p_subject                      # (N, 62, 384) fp32，z-score 后
        self.N, self.C, self.P = self.x_obs.shape
        self.F = self.sampler.prior.future_len        # 128
        self.batch_cap = batch_cap
        self.seed = seed
        self.amp_dtype = amp_dtype
        self._eps = eps                                # (N,1,62,128) 被试级固定
        self._eta = eta
        self._xi = None
        self._source = None
        self._use_biased_noise = use_biased_noise
        if use_biased_noise:
            ch_std = self.x_obs.float().std(dim=(0, 1))   # (C,) 母体 = 实际入模窗集（07-04 同定义）
            self._mu_bias = 0.5 * ch_std
        else:
            self._mu_bias = None
        self._x_fc_base = None

    # -- 基线（无扰动分支：x_0 ≡ x^p，同 (eps,eta)） ---------------------------

    @torch.no_grad()
    def compute_baseline(self) -> torch.Tensor:
        outs = []
        for c0 in range(0, self.N, self.batch_cap):
            c1 = min(c0 + self.batch_cap, self.N)
            with torch.amp.autocast("cuda", dtype=self.amp_dtype,
                                    enabled=self.device.type == "cuda"):
                out = self.sampler.sample(self.x_obs[c0:c1], 1,
                                          eps=self._eps[c0:c1], eta=self._eta[c0:c1])
            outs.append(out.squeeze(1).permute(0, 2, 1).float())   # (b,C,F)→(b,F,C)
        self._x_fc_base = torch.cat(outs, dim=0)                    # (N,128,62)
        return self._x_fc_base

    # -- 07-04 契约方法 ---------------------------------------------------------

    def set_source(self, source: int) -> None:
        """固定源通道并播种其噪声（07-04 同式：seed+source；同一 ξ 跨 σ 复用）。"""
        self._source = int(source)
        gen = torch.Generator(device=self.device).manual_seed(int(self.seed) + int(source))
        self._xi = torch.randn(self.N, self.P, device=self.device, generator=gen)

    @property
    def n_targets(self) -> int:
        return self.C

    @torch.no_grad()
    def perturb_and_measure(self, sigma: float, source: int):
        """x^p_pert = x^p + σ·ξ_a (+μ_bias^a) → 采样 → (mse_vec (62,), mean_diff_vec (62,))。

        扰动施加点 = z-score 后、入模前、x^p 段通道 a（design §2 数据流；
        其余通道逐位不动 —— 由 clone + 单通道赋值构造保证）。
        """
        if source != self._source:
            self.set_source(source)
        mse_sum = torch.zeros(self.C, device=self.device, dtype=torch.float32)
        md_sum = torch.zeros(self.C, device=self.device, dtype=torch.float32)
        for c0 in range(0, self.N, self.batch_cap):
            c1 = min(c0 + self.batch_cap, self.N)
            x_pert = self.x_obs[c0:c1].clone()            # fp32 输入侧（扰动算术不入 autocast）
            x_pert[:, source, :] = (
                x_pert[:, source, :] + float(sigma) * self._xi[c0:c1]
            )
            if self._mu_bias is not None:
                x_pert[:, source, :] = x_pert[:, source, :] + self._mu_bias[source]
            with torch.amp.autocast("cuda", dtype=self.amp_dtype,
                                    enabled=self.device.type == "cuda"):
                out = self.sampler.sample(x_pert, 1,
                                          eps=self._eps[c0:c1], eta=self._eta[c0:c1])
            out = out.squeeze(1).permute(0, 2, 1).float()          # (b,F,C)
            out = torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
            base = self._x_fc_base[c0:c1]
            mse_sum = mse_sum + per_target_mse(out, base) * (c1 - c0)
            md_sum = md_sum + per_target_mean_diff(out, base) * (c1 - c0)
        return mse_sum / self.N, md_sum / self.N
