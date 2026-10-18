"""Graph module ablation pipeline, multi-fold (task 07-15-graph-module-ablation).

对每个指定 fold 串行执行完整 LOO 消融（基线 SC-reg screg_l0.01_w0.1）。
每个 fold 完整产物：ckpt + loss JSON + 训练日志 + A_eff + DE 结果 + GC/TE +
连接图 + 单 fold 汇总。全部 fold 完成后做跨 fold 汇总。

消融矩阵（留一，基线 = M + 0.1·W，SC 通过 --sc-reg-lambda 0.01 塑 W）：
  baseline  : M + SC-reg·W + W
  no-SC     : M + W                         (关 --sc-reg-lambda,不加 --sc-base)
  no-mexhat : 0·M + SC-reg·W + W             (--mexhat-gamma-m 0)
  no-W      : M + SC-reg·W(γ_w=0)            (--mexhat-gamma 0)

产物按 fold 子目录组织（见 prd/design）。

启动（从仓库根 D:/102-npi/ude-vae-npi）:
    # fold1-4 全流程（默认）
    nohup python -u scripts/run_ablation_pipeline.py \\
        > logs/ablation/pipeline_multi.log 2>&1 &
    # 含 fold0 重跑评估
    nohup python -u scripts/run_ablation_pipeline.py --folds 0,1,2,3,4 \\
        > logs/ablation/pipeline_multi.log 2>&1 &
查进度:
    tail -f logs/ablation/pipeline_multi.log
"""

import argparse
import subprocess
import sys
import os
import time
from pathlib import Path

# 项目根 = 包根（本脚本位于 latent_flow/scripts/，parents[1] = latent_flow/）
ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)

PY = sys.executable
CKPT_DIR = ROOT / "checkpoints"
LOG_DIR = ROOT / "logs" / "ablation"
RESULTS_DIR = ROOT / "results"
OUTPUTS_DIR = ROOT / "outputs"

VARIANT_NAMES = ["baseline_screg", "nosc", "nomexhat", "now"]
N_SUBJECTS = 10
DE_SOURCES = "128"
DE_N_TRIALS = "32"


def log(msg):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def run(cmd, desc):
    cmd = [str(c) for c in cmd]
    log(f"=== {desc} ===\ncmd: {' '.join(cmd)}")
    t0 = time.time()
    r = subprocess.run(cmd, cwd=str(ROOT))
    dt = (time.time() - t0) / 60
    if r.returncode != 0:
        log(f"!!! FAILED (exit {r.returncode}, {dt:.1f} min): {desc}")
        sys.exit(r.returncode)
    log(f"=== DONE {desc} ({dt:.1f} min) ===")


def exists(rel_path):
    return (ROOT / rel_path).exists()


def variant_specs(fold):
    """Return [(name, ckpt_basename, extra_train_flags, skip_train)] for a fold.

    skip_train=True when the checkpoint already exists (resume) — baseline and
    all trained variants resume by ckpt presence. fold0's baseline/trained ckpts
    already exist; fold1-4 will train them fresh on the first run.
    """
    base = f"fold{fold}_oc_latent_flow_flow_best_graph_mexhat_screg_l0.01_w0.1"
    vae = str(CKPT_DIR / f"fold{fold}_oc_latent_flow_vae_best.pt")
    common = ["--stage", "2", "--fold", str(fold), "--graph-mode", "--graph-kernel",
              "mexhat", "--epochs", "50", "--seed", "42",
              "--anti-uniform-lambda", "0.01", "--batch-size", "32",
              "--vae-ckpt", vae]
    specs = [
        ("baseline_screg", base,
         ["--sc-base", "--sc-reg-lambda", "0.01", "--gamma-sc", "0", "--mexhat-gamma", "0.1"]),
        ("nosc", base + "_nosc",
         ["--mexhat-gamma", "0.1"]),
        ("nomexhat", base + "_nomexhat",
         ["--mexhat-gamma-m", "0", "--sc-base", "--sc-reg-lambda", "0.01",
          "--gamma-sc", "0", "--mexhat-gamma", "0.1"]),
        ("now", base + "_now",
         ["--sc-base", "--sc-reg-lambda", "0.01", "--gamma-sc", "0", "--mexhat-gamma", "0"]),
    ]
    # skip_train iff ckpt already present (resume). The training step appends
    # extra flags + --save-path to `common`; here we return common separately.
    return base, vae, common, specs


def _skip(ckpt_rel):
    return exists(f"checkpoints/{ckpt_rel}.pt")


# ---------------------------------------------------------------------------
# Per-fold steps
# ---------------------------------------------------------------------------

def step0_stage1(fold, vae):
    """Stage-1 VAE training (fold0 skip — ckpt exists)."""
    log(f">>>>> [fold {fold}] Step 0: stage-1 VAE 训练 >>>>>")
    vae_rel = f"checkpoints/fold{fold}_oc_latent_flow_vae_best.pt"
    if exists(vae_rel):
        log(f"[skip-stage1] fold {fold} VAE 已存在")
        return
    run([PY, "-u", "-m", "latent_flow.scripts.train_oc_latent_flow",
         "--stage", "1", "--fold", str(fold), "--epochs", "50", "--seed", "42",
         "--batch-size", "32"],
        f"[fold {fold}] Step 0: stage-1 VAE")


def step_a0_baseline(fold, base, common, baseline_flags):
    """SC-reg baseline training (fold0 skip — ckpt exists)."""
    log(f">>>>> [fold {fold}] Step A0: SC-reg baseline 训练 >>>>>")
    if _skip(base):
        log(f"[skip-baseline] fold {fold} baseline 已存在")
        return
    save_path = str(CKPT_DIR / f"{base}.pt")
    run([PY, "-u", "-m", "latent_flow.scripts.train_oc_latent_flow"]
        + common + baseline_flags + ["--save-path", save_path],
        f"[fold {fold}] Step A0: SC-reg baseline")


def step_a_variants(fold, common, specs):
    """3 LOO variant training."""
    log(f">>>>> [fold {fold}] Step A: 3 变体训练 >>>>>")
    for name, ckpt_base, extra in specs:
        if name == "baseline_screg":
            continue  # handled in step_a0
        if _skip(ckpt_base):
            log(f"[skip-variant] fold {fold} {name} 已存在")
            continue
        save_path = str(CKPT_DIR / f"{ckpt_base}.pt")
        run([PY, "-u", "-m", "latent_flow.scripts.train_oc_latent_flow"]
            + common + extra + ["--save-path", save_path],
            f"[fold {fold}] Step A: 训练 {name}")


def step_b_aeff(fold, base):
    log(f">>>>> [fold {fold}] Step B: A_eff 提取 >>>>>")
    for name in VARIANT_NAMES:
        # map variant name -> checkpoint basename
        ckpt_bn = base if name == "baseline_screg" else f"{base}_{name}"
        out = f"results/graph_connectivity_ablation/fold{fold}/{name}"
        if exists(f"{out}/graph_connectivity_fold{fold}.json"):
            log(f"[skip-aeff] fold {fold} {name}")
            continue
        run([PY, "-u", "-m", "latent_flow.scripts.extract_graph_connectivity",
             "--checkpoint", str(CKPT_DIR / f"{ckpt_bn}.pt"), "--fold", str(fold),
             "--output-dir", out,
             "--outputs-dir", f"outputs/graph_connectivity_ablation/fold{fold}/{name}"],
            f"[fold {fold}] Step B: A_eff {name}")


def step_c_de(fold, base):
    log(f">>>>> [fold {fold}] Step C: DE 推断 >>>>>")
    for name in VARIANT_NAMES:
        if name == "baseline_screg":
            ckpt_bn = base
        else:
            ckpt_bn = f"{base}_{name}"
        out = f"results/oc_connectivity_relative_ablation/fold{fold}/{name}"
        if exists(f"{out}/subject_{N_SUBJECTS - 1}.json"):
            log(f"[skip-de] fold {fold} {name}")
            continue
        run([PY, "-u", "-m", "latent_flow.scripts.run_oc_connectivity_relative",
             "--fold", str(fold), "--subjects", str(N_SUBJECTS),
             "--sources", DE_SOURCES, "--n-trials", DE_N_TRIALS,
             "--checkpoint", str(CKPT_DIR / f"{ckpt_bn}.pt"),
             "--output-dir", out],
            f"[fold {fold}] Step C: DE 推断 {name}")


def step_d0_gcte(fold):
    """GC/TE baseline connectivity generation (per-fold dir, resumable)."""
    log(f">>>>> [fold {fold}] Step D0: GC/TE 数据生成 >>>>>")
    bl_dir = f"results/baseline_connectivity_fold{fold}"
    # done iff last subject exists for both gc and te
    if (exists(f"{bl_dir}/gc/subject_{N_SUBJECTS - 1}.json")
            and exists(f"{bl_dir}/te/subject_{N_SUBJECTS - 1}.json")):
        log(f"[skip-gcte] fold {fold} 已完成")
        return
    run([PY, "-u", "-m", "latent_flow.baseline.run_baseline_connectivity",
         "--fold", str(fold), "--subjects", str(N_SUBJECTS),
         "--methods", "gc,te", "--resume",
         "--results-dir", bl_dir,
         "--outputs-dir", f"outputs/baseline_connectivity_fold{fold}"],
        f"[fold {fold}] Step D0: GC/TE 生成")


def step_d_compare(fold):
    log(f">>>>> [fold {fold}] Step D: GC/TE 对比 >>>>>")
    bl_dir = str(ROOT / "results" / f"baseline_connectivity_fold{fold}")
    for name in VARIANT_NAMES:
        out = f"results/compare_params_ablation/fold{fold}/{name}"
        if exists(f"{out}/compare_fold{fold}.json"):
            log(f"[skip-compare] fold {fold} {name}")
            continue
        run([PY, "-u", "-m", "latent_flow.scripts.compare_params_vs_baseline",
             "--fold", str(fold),
             "--a-dir", f"results/graph_connectivity_ablation/fold{fold}/{name}",
             "--bl-dir", bl_dir,
             "--out-dir", out],
            f"[fold {fold}] Step D: GC/TE 对比 {name}")


def step_e_plots(fold):
    log(f">>>>> [fold {fold}] Step E: 连接图绘制 >>>>>")
    marker = f"outputs/ablation/fold{fold}/connectivity_plots.done"
    if exists(marker):
        log(f"[skip-plots] fold {fold} 已绘制")
        return
    run([PY, "-u", "-m", "latent_flow.scripts._plot_ablation_connectivity",
         "--fold", str(fold)],
        f"[fold {fold}] Step E: 连接图")
    (ROOT / "outputs" / "ablation" / f"fold{fold}").mkdir(
        parents=True, exist_ok=True)
    (ROOT / "outputs" / "ablation" / f"fold{fold}"
     / "connectivity_plots.done").touch()


def step_f_summarize(fold):
    log(f">>>>> [fold {fold}] Step F: 单 fold 汇总 >>>>>")
    if exists(f"results/ablation/fold{fold}/summary.json"):
        log(f"[skip-summary] fold {fold} 已生成")
        return
    run([PY, "-u", "-m", "latent_flow.scripts._summarize_ablation",
         "--fold", str(fold)],
        f"[fold {fold}] Step F: 汇总")


def run_fold(fold):
    """Run the full per-fold pipeline (Step 0 → F)."""
    log(f"############### FOLD {fold} START ###############")
    base, vae, common, specs = variant_specs(fold)
    baseline_flags = specs[0][2]  # baseline_screg extra flags
    step0_stage1(fold, vae)
    step_a0_baseline(fold, base, common, baseline_flags)
    step_a_variants(fold, common, specs)
    step_b_aeff(fold, base)
    step_c_de(fold, base)
    step_d0_gcte(fold)
    step_d_compare(fold)
    step_e_plots(fold)
    step_f_summarize(fold)
    log(f"############### FOLD {fold} DONE ###############")


def step_g_cross_fold_summary(folds):
    """Cross-fold aggregate summary after all folds complete."""
    log(">>>>>>> Step G: 跨 fold 汇总 >>>>>>>")
    run([PY, "-u", "-m", "latent_flow.scripts._summarize_ablation",
         "--cross-fold", ",".join(str(f) for f in folds)],
        "Step G: 跨 fold 汇总")


def parse_args():
    p = argparse.ArgumentParser(description="Multi-fold graph ablation pipeline.")
    p.add_argument("--folds", type=str, default="1,2,3,4",
                   help="Comma-separated fold list (default 1,2,3,4).")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    folds = [int(f) for f in args.folds.split(",") if f.strip() != ""]
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log(f"ablation pipeline START, root={ROOT}, folds={folds}, PY={PY}")
    for fold in folds:
        run_fold(fold)
    step_g_cross_fold_summary(folds)
    log(">>> 全流水线完成 <<<")
