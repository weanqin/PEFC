#!/usr/bin/env python3
"""_refs_worker —— 参考侧子进程 worker（refs_fmt.py 逐例调用；design §1 子进程隔离纪律）。

运行形态：fresh python3 进程，由 refs_fmt.py 以 cwd=<baseline-ec-v1>、env
PYTHONPATH=<baseline-ec-v1> 启动——本脚本自身目录（tools/，无 config.py）只占
sys.path[0]，baseline-ec-v1 是唯一仓库路径 ⇒ ec_refs/split_half/compare 模块级
`import config as C` 正常解析到 baseline config。**进程内调用不可行**：父进程
（perturb-ec-v1 目录）内 import 这些模块必被本变体 config.py 遮蔽击穿（design §1
已登记），故一律子进程 + npz 文件面交换。

职责（单例 x.npy → 单例 npz）：
  ① ec_refs.refs_for_subject(x, p)：ridge 条件 GC（行=源/列=靶/对角=0）+
     |gPDC|² 宽带/分带 + lb_frac；② split_half.ceiling_for_subject(x, p)：
     奇偶 2 s 段（250 样本 @125 Hz）两半同口径 EC 的 offdiag Spearman。
p 默认 = ec_refs.P_SHARED（8；10-13 任务定标单源，不在本侧复写常量）。

用法（仅经 refs_fmt.py 调用，不直接手跑）：
  python3 _refs_worker.py <x.npy> <out.npz> [--p 8]
"""

from __future__ import annotations

import os

# ---- BLAS 护栏（environment.md / _blas_guard 惯例：必须在 numpy 之前；----
# ---- 16 核 cgroup 限 8 线程；setdefault 幂等——环境继承下维持父进程值） ----
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "8")

import argparse  # noqa: E402
import json      # noqa: E402
import sys       # noqa: E402
import time      # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

# ---- 路径消毒（isolation 纪律的防御层；正常启动下为空操作）----
# 剔除继承路径中 /root/npi 下除 baseline-ec-v1 外的一切仓库目录（含 '' 相对项——
# cwd 已由启动器钉在 baseline 目录，resolve 后落在 baseline 内予以保留）。
_BASELINE = Path("/root/npi/models/baseline-ec-v1")
_REPO = Path("/root/npi")


def _sanitize_syspath():
    kept = []
    for p in list(sys.path):
        rp = Path(p).resolve() if p else Path.cwd()
        s = str(rp)
        if s.startswith(str(_REPO)) and not s.startswith(str(_BASELINE)):
            continue    # 其他仓库目录：剔除（防遮蔽）
        kept.append(p)
    sys.path[:] = kept
    if str(_BASELINE) not in sys.path:
        sys.path.insert(0, str(_BASELINE))


_sanitize_syspath()

import config as C  # noqa: E402
assert Path(C.__file__).resolve().parent == _BASELINE.resolve(), \
    f"config 遮蔽击穿：解析到 {C.__file__}（应归属 {_BASELINE}）——隔离纪律失败，中止"

import ec_refs        # noqa: E402
import split_half as SH  # noqa: E402
import gc_psd         # noqa: E402  （ec_refs 引入后缓存于 sys.modules）
assert Path(gc_psd.__file__).resolve().parent == Path(C.V1_DIR).resolve(), \
    f"gc_psd 解析异常：{gc_psd.__file__}（应归属 {C.V1_DIR}）"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("x_npy")
    ap.add_argument("out_npz")
    ap.add_argument("--p", type=int, default=ec_refs.P_SHARED)
    args = ap.parse_args()

    t0 = time.time()
    x32 = np.load(args.x_npy)
    x = np.asarray(x32, dtype=np.float64)   # 与 resample_io.load_cached 精度往返一致
    assert x.ndim == 2 and x.shape[0] == C.N_CHANNELS, f"x 形状异常: {x.shape}"
    t_load = time.time() - t0

    # ① 全记录 refs（ridge GC + gPDC + lb_frac；mne-c 本任务不做）
    t1 = time.time()
    r = ec_refs.refs_for_subject(x, args.p, mnec=False)
    t_refs = time.time() - t1

    # ② split-half 天花板（baseline-ec-v1 split_half.py 口径原样）
    t2 = time.time()
    c = SH.ceiling_for_subject(x, args.p)
    t_sh = time.time() - t2

    n_seg = x.shape[-1] // C.SPLIT_SEG_SAMPLES
    np.savez_compressed(
        args.out_npz,
        gc=r["gc"].astype(np.float32),
        gpdc_bb=r["gpdc_bb"].astype(np.float32),
        gpdc_bands=r["gpdc_bands"].astype(np.float32),
        lb_frac=r["lb_frac"], n=r["n"], p=r["p"], lam=r["lam"],
        sh_rho_gc=c["gc"], sh_rho_gpdc=c["gpdc"],
        n_samples=x.shape[-1], split_seg=C.SPLIT_SEG_SAMPLES, n_seg=n_seg,
    )
    # 单行 JSON 摘要（父进程解析进 run.log；数值事实以 npz 为准）
    print(json.dumps({
        "ok": True, "p": r["p"], "lam": r["lam"], "n": r["n"],
        "gc_max": float(r["gc"].max()), "gpdc_max": float(r["gpdc_bb"].max()),
        "lb_frac": round(r["lb_frac"], 4),
        "sh_rho_gc": round(c["gc"], 4), "sh_rho_gpdc": round(c["gpdc"], 4),
        "n_seg": n_seg, "t_load_s": round(t_load, 1),
        "t_refs_s": round(t_refs, 1), "t_sh_s": round(t_sh, 1),
        "config_file": str(Path(C.__file__).resolve()),
        "gc_psd_file": str(Path(gc_psd.__file__).resolve()),
    }))


if __name__ == "__main__":
    main()
