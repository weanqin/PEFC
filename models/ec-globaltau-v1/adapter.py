"""gpfm-mamba-v5-main 代理脑加载适配层（任务 10-21-ec-globaltau-v5main）。

import-only 复用两源包（零逻辑改动）：
- `models/gpfm-mamba-v5`（config/model/graph，path 遮蔽优先）；
- `models/gpfm-mamba-v1`（data/gpr_prior/sample，v1 部分经 sys.path，sample.py 以独立模块名物化加载）。

加载模式沿 2026-10-21 已归档先例 `outputs/10-21-v5main-ec-explore/gen_ec_explore.py`，
差异（评审阻断 B1 修法）：θ\* 数值口径 = **EMA 正典权重**——本模块只负责装载（load_eval_model
本身已把 ck["ema"] 装入 EMA 对象），swap 由 runner 在全部前向的外层 `with ema.swap(model):`
统一执行（见 run_ec_globaltau.py）；explore 先例实跑未做 swap（raw weights），其数字不可与本路径直比。

数值精度（评审阻断 B2 + briefing 拍板 2026-10-21）：**bf16 autocast**，autocast 域由 probe_site
包裹采样前向（镜像 07-04 结构：扰动算术在输入侧 fp32），对 gpfm 正典 fp32 为登记偏离。
"""

import importlib.util
import sys
from pathlib import Path

V5 = "/root/npi/models/gpfm-mamba-v5"
V1 = "/root/npi/models/gpfm-mamba-v1"

DEFAULT_CKPT = "/root/npi/outputs/gpfm-mamba-f0/v5-main-01/checkpoints/gpfm-mamba-v5-best.pt"
DEFAULT_RUN_DIR = "/root/npi/outputs/gpfm-mamba-f0/v5-main-01"

# fold0 test 10 例（v1 固化 manifest；iterate_test_batches 内 fixate_split 校验防漂移）
FOLD0_TEST_IDS = [12, 17, 18, 24, 30, 38, 44, 55, 58, 61]


def load_surrogate(ckpt_path: str = DEFAULT_CKPT, run_dir: str = DEFAULT_RUN_DIR,
                   device: str = "cuda"):
    """装载 v5-main θ\*。返回 (cfg, model, ema, sampler, test_ds)。

    - cfg：v5 config（data_root=edra_process_v4，62ch，past 384/future 128/stride 64，fold0）；
    - model/ema：raw 权重模型 + EMA shadow（swap 由 runner 统一外层执行）；
    - sampler：EulerSampler(euler_steps=cfg.euler_steps=16, sigma_min=cfg.sigma_min=1e-4,
      gen_lowpass_hz=getattr(cfg,'gen_lowpass_hz',40.0))——explore 先例同构；
    - test_ds：EdraWindowsDataset（z-score 用 run 目录训练折统计——「z-score 之后」扰动约束的
      归一化出处），windows/subjects 元数据随附。
    """
    sys.path.insert(0, V1)
    sys.path.insert(0, V5)
    import config as C  # noqa: E402  -> v5（遮蔽）
    import model as _M  # noqa: E402  -> v5（遮蔽；graph 同路命中 v5）
    assert Path(C.__file__).parent == Path(V5), C.__file__
    assert Path(_M.__file__).parent == Path(V5), _M.__file__

    _spec = importlib.util.spec_from_file_location("gpfm_v1_sample", f"{V1}/sample.py")
    S = importlib.util.module_from_spec(_spec)  # noqa: E402
    _spec.loader.exec_module(S)  # 其 import config/model 命中 v5 缓存；data/gpr_prior 命中 v1

    cfg = C.build_config()
    model, ema, ck = S.load_eval_model(ckpt_path, cfg, device)
    prior = S.build_eval_prior(cfg, device)
    sampler = S.EulerSampler(model, prior, euler_steps=cfg.euler_steps,
                             sigma_min=cfg.sigma_min,
                             gen_lowpass_hz=getattr(cfg, "gen_lowpass_hz", 40.0))
    loader, test_ds = S.iterate_test_batches(cfg, Path(run_dir), batch_size=8)
    test_ids = sorted(int(Path(e["npz_path"]).name.split("_")[0]) for e in test_ds.subjects)
    assert test_ids == FOLD0_TEST_IDS, f"fold0 test 清单漂移: {test_ids}"
    return cfg, model, ema, sampler, test_ds


def subject_global_windows(test_ds, sid: int):
    """该受试者在 test_ds 中的全局窗索引（数据集顺序确定；explore 先例同款）。"""
    out = []
    for gi, (si, _) in enumerate(test_ds.windows):
        e = test_ds.subjects[si]
        if int(Path(e["npz_path"]).name.split("_")[0]) == sid:
            out.append(gi)
    return out
