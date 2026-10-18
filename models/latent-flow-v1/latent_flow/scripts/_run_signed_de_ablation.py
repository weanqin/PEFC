"""DE 注入消融 signed DE 编排脚本（task 07-25-de-injection-ablation）。

对 3 变体 (nosc / now / nomexhat) × 5 fold 串行跑 run_oc_connectivity_global_tau
（有偏噪声、带符号）。带断点续跑：每组 10 个 subject_*.json 齐全则跳过。

必须用 conda npi 环境执行（CUDA）。从仓库根 D:/102-npi/ude-vae-npi 运行：
    /d/program/anaconda/envs/npi/python.exe -u -m \
        latent_flow.scripts._run_signed_de_ablation [--folds 0,1,2,3,4]

后台运行：
    nohup /d/program/anaconda/envs/npi/python.exe -u -m \
        latent_flow.scripts._run_signed_de_ablation \
        > logs/ablation/signed_de_multi.log 2>&1 &
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)

NPI_PY = r"D:\program\anaconda\envs\npi\python.exe"
CKPT_DIR = ROOT / "checkpoints"
OUT_DIR = ROOT / "results" / "oc_connectivity_global_tau_signed_ablation"

VARIANTS = ["nosc", "now", "nomexhat"]
N_SUBJECTS = 10
SOURCES = "128"
N_TRIALS = "32"
N_GRID = "12"   # default 24 too slow (17.4min/subject); 12 halves it (8.47min), see design.md


def log(msg):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def ckpt_path(fold, variant):
    return str(CKPT_DIR / f"fold{fold}_oc_latent_flow_flow_best_graph_mexhat_screg_l0.01_w0.1_{variant}.pt")


def out_path(fold, variant):
    return OUT_DIR / f"fold{fold}" / variant


def is_done(fold, variant):
    """Done iff all N_SUBJECTS subject_*.json present."""
    d = out_path(fold, variant)
    if not d.exists():
        return False
    return sum(1 for i in range(N_SUBJECTS) if (d / f"subject_{i}.json").exists()) >= N_SUBJECTS


def run_one(fold, variant):
    if is_done(fold, variant):
        log(f"[skip] fold{fold}/{variant} 已完成（{N_SUBJECTS} subjects）")
        return
    ckpt = ckpt_path(fold, variant)
    if not Path(ckpt).exists():
        log(f"!!! 缺失 ckpt: {ckpt}，跳过 fold{fold}/{variant}")
        return
    out = str(out_path(fold, variant))
    cmd = [NPI_PY, "-u", "-m", "latent_flow.scripts.run_oc_connectivity_global_tau",
           "--fold", str(fold), "--subjects", str(N_SUBJECTS),
           "--sources", SOURCES, "--n-trials", N_TRIALS, "--n-grid", N_GRID,
           "--checkpoint", ckpt, "--output-dir", out]
    log(f"=== fold{fold}/{variant} START ===\n  cmd: {' '.join(cmd)}")
    t0 = time.time()
    r = subprocess.run(cmd, cwd=str(ROOT))
    dt = (time.time() - t0) / 60
    if r.returncode != 0:
        log(f"!!! fold{fold}/{variant} FAILED (exit {r.returncode}, {dt:.1f} min)")
        sys.exit(r.returncode)
    log(f"=== fold{fold}/{variant} DONE ({dt:.1f} min) ===")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--folds", default="0,1,2,3,4")
    ap.add_argument("--variants", default=",".join(VARIANTS))
    args = ap.parse_args()

    folds = [int(x) for x in args.folds.split(",") if x.strip() != ""]
    variants = [v.strip() for v in args.variants.split(",") if v.strip() != ""]

    log(f"### signed DE 消融编排 START ###")
    log(f"folds={folds}  variants={variants}  subjects={N_SUBJECTS} sources={SOURCES} trials={N_TRIALS}")
    log(f"输出根目录: {OUT_DIR}")
    log(f"Python: {NPI_PY}")

    t0 = time.time()
    n_run = 0
    n_skip = 0
    for fold in folds:
        for variant in variants:
            if is_done(fold, variant):
                log(f"[skip] fold{fold}/{variant}")
                n_skip += 1
                continue
            run_one(fold, variant)
            n_run += 1

    dt = (time.time() - t0) / 60
    log(f"### 全部完成: 跑{n_run} 跳{n_skip} 总耗时 {dt:.1f} min ###")


if __name__ == "__main__":
    main()
