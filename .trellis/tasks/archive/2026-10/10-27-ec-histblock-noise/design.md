# design.md — hist-block 历史段分组共享噪声变体

## 1. 边界与不变量

- **改动面**：仅 `models/latent-flow-v1/ec_mild_unbiased.py`（+任务档 `research/` 验证脚本）。
  `latent_flow` 包体零改动（mild 线纪律）。
- **不变量**（两模式下均保持）：
  - 扰动施加 = `perturb_channel` 逐字不改：`x_pert[:,:,src] = x_pert[:,:,src] + σ·ξ`，
    仅源通道、z-score 后、入模前；
  - CRN：同一源扫全部 σ 期间 ξ 不变；
  - μ=0（`use_biased_noise=False`，`_mu_bias is None` 断言保留）；
  - forecast / measure / τ / σ* / conn 计算路径零改动；
  - 代理脑确定性采样 ⇒ ξ 为唯一随机源。

## 2. 配置面（CLI，5 旗标）

```
--noise-mode {iid,hist-block}                 默认 iid（= 原实现逐字不变）
--hist-len INT                                默认 375（断言 == obs_len）
--hist-group-size INT                         默认 25（断言整除 hist-len；375/25=15 组）
--hist-share-channels / --no-hist-share-channels   默认共享（BooleanOptionalAction）
--hist-share-subjects  / --no-hist-share-subjects  默认共享
```

口径断言（`_caliber_asserts` 追加）：`noise-mode == hist-block` 时
`hist_len == obs_len(375)` 且 `hist_len % hist_group_size == 0`；iid 模式忽略全部 hist 旗标。

## 3. 噪声构造（唯一新函数）

```python
_HIST_XI_SEED_OFFSET = 9375      # = 375*25；与 seed+source∈[42,169]、seed*10000+sid 正交

def build_hist_block_xi(n_trials, hist_len, group_size, *, seed, device,
                        source_key=None, subject_key=None):
    """历史段分组噪声 ξ：(n_trials, G) i.i.d. N(0,1) → 组内复制 → (n_trials, hist_len)。
    source_key/subject_key 非 None 时卷入种子（= 对应共享开关关闭的退化路径）。"""
    assert hist_len % group_size == 0, (hist_len, group_size)
    n_groups = hist_len // group_size                      # 15
    s = int(seed) + _HIST_XI_SEED_OFFSET
    if source_key is not None:   s += 1009 * (source_key + 1)
    if subject_key is not None:  s += 100003 * (subject_key + 1)
    gen = torch.Generator(device=device).manual_seed(s)    # 独立 Generator，不受全局 RNG 影响
    base = torch.randn(n_trials, n_groups, device=device, generator=gen)  # (N, 15)
    return base.repeat_interleave(group_size, dim=1)       # (N, 375) 连续 25 点一组
```

- `repeat_interleave`（非 `repeat`）⇒ **连续** 25 点一组：`ξ[:, 25g:25(g+1)] ≡ base[:, g]`；
  组间来自不同 `randn` 抽样、相互独立（要求 2）。
- trial 维（行）独立抽样（拍板②）。
- 退化路径（AC6）：`source_key` / `subject_key` 键控偏移如上，两开关正交可叠加。

## 4. 探针子类（唯一新类）

```python
class HistBlockProbeSite(SignalProbeSite):
    """ξ 注入式探针：唯一差异 = set_source 不按 seed+source 重抽，改取注入构造。
    forecast / measure / CRN 全部继承零改动。"""

    def __init__(self, *, xi_provider, **kw):
        super().__init__(**kw)
        self._xi_provider = xi_provider

    def set_source(self, source: int) -> None:
        self._source = source
        xi = self._xi_provider(source)                     # 通道共享 ⇒ 忽略 source
        assert xi.shape == (self.N, self.x_obs.shape[1]), (xi.shape, self._xi.shape)
        self._xi = xi
```

安全性：父类 `__init__` 不调用 `set_source`（`_xi=None` 后直接 `_baseline()`），子类在
`super().__init__` 返回后才装 provider，首次 `set_source` 由调用方循环显式触发——与现有
stage1/stage2 调用序完全一致，无隐藏路径。

## 5. run_main 接线（3 处）

被试循环**外**（全共享路径只构造一次，要求 1 的"只生成一次"路线）：

```python
XI_HIST_GLOBAL = None
if a.noise_mode == "hist-block" and a.hist_share_channels and a.hist_share_subjects:
    n_max = a.n_trials if a.n_trials > 0 else TRIALS       # 数据集常量 30
    XI_HIST_GLOBAL = build_hist_block_xi(n_max, a.hist_len, a.hist_group_size,
                                         seed=a.seed, device=device)
```

每被试 site 构造处二选一（原 `SignalProbeSite(...)` 调用行在 iid 分支**逐字保留**）：

```python
common = dict(model=model, x_obs_subject=x_obs, n_steps=a.n_steps, use_proj=False,
              batch_size=a.batch_size, seed=a.seed, n_grid=a.n_grid,
              sigma_min=a.sigma_min, sigma_max=a.sigma_max, use_biased_noise=False)
if a.noise_mode == "hist-block":
    if a.hist_share_channels:
        xi = (XI_HIST_GLOBAL[:N] if a.hist_share_subjects else
              build_hist_block_xi(N, a.hist_len, a.hist_group_size, seed=a.seed,
                                  device=device, subject_key=subj_idx))
        provider = lambda src, _x=xi: _x                                 # 与源无关
    else:
        provider = lambda src: build_hist_block_xi(                      # 按源键控（可叠被试键）
            N, a.hist_len, a.hist_group_size, seed=a.seed, device=device,
            source_key=src, subject_key=None if a.hist_share_subjects else subj_idx)
    site = HistBlockProbeSite(xi_provider=provider, **common)
else:
    site = SignalProbeSite(**common)                                     # 原路径
```

stage1 / stage2 循环一行不动（接口未变）。run_main 头部打印追加 noise-mode 段。

## 6. 维度变换链

```
base     randn(N=30, G=15)                    每组独立采样一次（要求 2）
   │ repeat_interleave(25, dim=1)             组内复制 25 份
ξ_hist   (30, 375)                            t∈[25g, 25(g+1)) 恒等于 base[:,g]
   │ [:N_s] 被试切片（同一存储，零复制）        跨被试逐位一致（要求 1）
   │ set_source 忽略 source（同引用）          跨 128 源同一份（要求 1）
施加     x_pert[:, :, src] += σ·ξ_hist        perturb_channel 逐字不变；ξ 沿通道/被试维
                                            逻辑广播，物理零复制
```

边际方差仍为 σ²（组内常数不影响），τ / σ* 幅度口径与 iid 可比；hist-block 谱形为 15 段
阶跃（低频能量集中）——变体本意，口径差异由 payload `noise_model` 字段区分。

## 7. payload / 复现契约（拍板①：不落盘数组）

- hist-block 分支 `hparams` 追加：`noise_mode="hist-block"` / `hist_len` /
  `hist_group_size` / `hist_n_groups` / `hist_share_channels` / `hist_share_subjects` /
  `hist_xi_seed_offset=9375`；iid 分支键面不变（旗标关 = 旧 payload 先例）。
- `noise_model` 字符串分支：hist-block = `"unbiased_gaussian_mu0_hist_block (sigma*xi
  additive on z-scored x_obs (375); xi block-shared: 25-sample groups x15, shared
  across sources/subjects, same xi across sigma)"`；iid = 原串。
- 跨任务复现 = 确定性重建（种子+配方均随 payload 存档），**四前提**写入 docstring 与
  PRD：同 seed / 同 `(n_trials, 15)` 形状 / 同设备类型（CPU↔CUDA 流不同） / 同 torch
  版本（RNG 实现无跨版本契约）。

## 8. 验证设计（详见 implement.md §检查）

| # | 检查 | 手段 | 级别 |
|---|---|---|---|
| V1 | 同被试跨通道 ξ 逐元素相同 | provider 忽略源 + `site.set_source(0/1/127)` 后 `torch.equal` | 单元+集成 |
| V2 | 跨被试 ξ 逐元素相同 | 两次独立同参数重建 `torch.equal` + 双被试 site 对拍 | 单元+集成 |
| V3 | 组内 25 点恒等 | `ξ == ξ[:, ::25].repeat_interleave(25,1)` 逐组展开断言 | 单元 |
| V4 | 15 组、组间独立 | `G==15`；`ξ[:, ::25]==base` 构造性；N=200k 相关 sanity |offdiag|<0.01 | 单元 |
| V5 | 非历史段逻辑未变 | 默认 iid 改动前后 mini 干跑逐位对拍（基线先行）+ CRN digest + `_mu_bias is None` + diff 审查 | 集成 |

## 9. 风险与回滚

- **共享文件并发**：config-fix 会话同时在改 `ec_mild_unbiased.py`。缓解：改前落 iid 基线
  干跑 + 记录文件 hash；编辑窗口最短化；编辑前 `git diff` 复核无新外来改动；提交归属 3.4
  用户拍板。
- **GPU 争用**：干跑前 `nvidia-smi` 查占用；被试/trial 数最小化（1 例 × 8 源）。
- **回滚**：单文件改动，`git diff` 可逆；任务档验证脚本独立于包体，删除即净。
