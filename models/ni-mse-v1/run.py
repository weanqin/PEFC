#!/usr/bin/env python3
"""run.py — NI-MSE-50-q0.90 三期断点续跑 CLI（design.md §2 + §1b-7 续跑工程契约）。

用法（产物根 outputs/10-20-ni-mse-v5fmt-f0/<run-id>/：null/ perturb/ report/ manifest.json logs/）：
  python3 models/ni-mse-v1/run.py --stage null    --run-id ni-mse-v5fmt-01
  python3 models/ni-mse-v1/run.py --stage perturb --run-id ni-mse-v5fmt-01
  python3 models/ni-mse-v1/run.py --stage report  --run-id ni-mse-v5fmt-01
  # 冒烟小参数（Run 001）：
  python3 models/ni-mse-v1/run.py --stage null --run-id ni-mse-v5fmt-smoke \
      --config w_null=8 w_pert=4 k_resp=4 k_null=4 b_ref=4 c_arms=0.5 subjects=10

工程契约：
- manifest.json **原子写**（tmp + os.replace），每期增量更新：参数 / 种子树（含分支整数
  映射与 spawn_key 公式）/ 窗子集索引 / θ* 指纹（ckpt SHA256 + config_hash + epoch/
  best_val）/ env 快照 / 各期状态；
- R2 失败 = `.r2_failed` 标记文件（产物根）+ 非零退出；**后续 stage 拒跑**；不吞异常；
- report 前置断言：上游 null×|subjects| + perturb×|subjects|×|c 臂| 产物齐全且有限；
- θ*/refs/家族代码全程只读（path-isolation 契约）。
"""

from __future__ import annotations

import _blas_guard  # noqa: F401  必须最先：BLAS 五变量先于 numpy 装载

import argparse
import datetime
import hashlib
import json
import os
import socket
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import numpy as np        # noqa: E402  （BLAS guard 之后）
import torch              # noqa: E402

import noise as NZ        # noqa: E402
import stages as ST       # noqa: E402
import thostar as TH      # noqa: E402
import windows as WD      # noqa: E402

PRODUCTS_ROOT = Path("/root/npi/outputs/10-20-ni-mse-v5fmt-f0")
BLAS_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")

# --config 可覆盖键（冒烟缩参数用；正式 Run 一律默认钉定值）
_OVERRIDE_PARSERS = {
    "w_null": int, "w_pert": int, "k_resp": int, "k_null": int, "b_ref": int,
    "batch_size": int,
    "c_arms": lambda v: tuple(float(x) for x in v.split(",")),
    "subjects": lambda v: [int(x) for x in v.split(",")],
}


def parse_overrides(pairs: list) -> dict:
    out = {}
    for kv in pairs:
        if "=" not in kv:
            sys.exit(f"[run] --config 项格式应为 KEY=VALUE，got: {kv!r}")
        k, v = kv.split("=", 1)
        if k not in _OVERRIDE_PARSERS:
            sys.exit(f"[run] 不支持的 --config 键 {k!r}（允许：{sorted(_OVERRIDE_PARSERS)}）")
        out[k] = _OVERRIDE_PARSERS[k](v)
    subs = out.get("subjects")
    if subs is not None:
        for s in subs:
            if s not in TH.EXPECTED_TEST_IDS:
                sys.exit(f"[run] subjects 含非 fold0 test 名单成员 {s}（{TH.EXPECTED_TEST_IDS}）")
    return out


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def env_snapshot() -> dict:
    import scipy
    snap = {
        "hostname": socket.gethostname(),
        "python": sys.version.split()[0],
        "torch": torch.__version__, "cuda_available": bool(torch.cuda.is_available()),
        "numpy": np.__version__, "scipy": scipy.__version__,
        "blas_env": {v: os.environ.get(v) for v in BLAS_VARS},
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
    }
    if snap["cuda_available"]:
        snap["gpu"] = torch.cuda.get_device_name(0)
    import mamba_ssm
    snap["mamba_ssm"] = getattr(mamba_ssm, "__version__", "unknown")
    return snap


def theta_fingerprint() -> dict:
    return {
        "ckpt_path": TH.CKPT_PATH,
        "ckpt_sha256": sha256_file(Path(TH.CKPT_PATH)),
        "config_hash": TH.EXPECTED_CONFIG_HASH,
        "epoch": TH.EXPECTED_EPOCH, "best_val": TH.EXPECTED_BEST_VAL,
        "channel_mean_sha256": sha256_file(Path(TH.CHANNEL_MEAN_PATH)),
        "channel_std_sha256": sha256_file(Path(TH.CHANNEL_STD_PATH)),
        "split_manifest": TH.SPLIT_MANIFEST,
        "test_ids": TH.EXPECTED_TEST_IDS,
    }


def seed_tree_manifest() -> dict:
    return {
        "master_entropy": NZ.MASTER_ENTROPY,
        "branch_int_map": dict(NZ.BRANCH),
        "spawn_keys": {
            "eps_ref": "(10, sid)",
            "eta_ref": "(13, sid)",
            "eps_null": "(11, sid, w)",
            "eta_null": "(13, sid, w)",
            "xi_pert": "(12, sid, w, a, k)",
            "eta_pert": "(13, sid, w, a, k)",
            "eps_pert_fresh": "(11, sid, w, a, k)",
            "win_null": "(20, sid)", "win_pert": "(21, sid)",
        },
        "w_semantics": "被试窗全局索引（loader shuffle=False 确定性序，0..929）",
        "c_independence": "ξ/η/ε 均不含 c ⇒ 跨 c 臂 CRN 配对",
    }


def write_manifest_atomic(path: Path, manifest: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=1, ensure_ascii=False)
    os.replace(tmp, path)                       # §1b-7 原子写


def load_manifest(path: Path) -> dict:
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def guard_r2_marker(root: Path, stage: str) -> None:
    marker = ST.r2_marker_path(root)
    if marker.exists():
        sys.exit(f"[run] {stage} 拒跑：存在 R2 失败标记 {marker}——\n{marker.read_text()}")


def resolve_params(overrides: dict) -> ST.Params:
    p = ST.Params()
    for k in ("w_null", "w_pert", "k_resp", "k_null", "b_ref"):
        if k in overrides:
            setattr(p, k, overrides[k])
    if "c_arms" in overrides:
        p.c_arms = tuple(overrides["c_arms"])
    if p.w_pert > p.w_null:
        sys.exit(f"[run] w_pert({p.w_pert}) > w_null({p.w_null})——W_pert ⊂ W_null 契约破坏")
    return p


def build_manifest(run_id: str, root: Path, p: ST.Params, subjects, overrides: dict) -> dict:
    man = load_manifest(root / "manifest.json")
    man.setdefault("run_id", run_id)
    man["products_root"] = str(root)
    man["updated"] = datetime.datetime.now().isoformat(timespec="seconds")
    man["params"] = p.as_dict()
    man["params"]["subjects"] = [int(s) for s in subjects]
    man["params"]["n0"] = ST.n0_from_cfg(TH.load_cfg())
    man["params"]["cli_overrides"] = {k: (list(v) if isinstance(v, tuple) else v)
                                      for k, v in overrides.items()}
    man["seed_tree"] = seed_tree_manifest()
    man.setdefault("theta_star", theta_fingerprint())
    man.setdefault("env", env_snapshot())
    man.setdefault("stages", {})
    return man


def stage_null(root: Path, p: ST.Params, subjects, overrides: dict) -> None:
    man = _stage_common(root, p, subjects, overrides)
    th = TH.load_thostar()
    _, test_ds = TH.load_test_data(batch_size=overrides.get("batch_size", 4))
    plan = WD.build_window_plan(test_ds, p.w_null, p.w_pert)
    man["windows"] = {str(sid): {"wnull": plan[sid]["wnull"].tolist(),
                                 "wpert": plan[sid]["wpert"].tolist()} for sid in subjects}
    man["stages"]["null"] = {"status": "running",
                             "started": datetime.datetime.now().isoformat(timespec="seconds")}
    write_manifest_atomic(root / "manifest.json", man)
    t0 = time.time()
    summary = ST.null_stage(th, test_ds, plan, p, root / "null", subjects)
    man["stages"]["null"] = {"status": "done", "elapsed_s": round(time.time() - t0, 1),
                             "subjects": summary}
    man["updated"] = datetime.datetime.now().isoformat(timespec="seconds")
    write_manifest_atomic(root / "manifest.json", man)


def _stage_common(root: Path, p: ST.Params, subjects, overrides: dict) -> dict:
    run_id = root.name
    return build_manifest(run_id, root, p, subjects, overrides)


def stage_perturb(root: Path, p: ST.Params, subjects, overrides: dict) -> None:
    guard_r2_marker(root, "perturb")
    missing = [str(root / "null" / f"{sid}.npz") for sid in subjects
               if not (root / "null" / f"{sid}.npz").exists()]
    if missing:
        sys.exit(f"[run] perturb 前置断言失败：null 产物缺失 {missing}（先跑 --stage null）")
    man = _stage_common(root, p, subjects, overrides)
    th = TH.load_thostar()
    _, test_ds = TH.load_test_data(batch_size=overrides.get("batch_size", 4))
    plan = WD.build_window_plan(test_ds, p.w_null, p.w_pert)
    man["stages"]["perturb"] = {"status": "running",
                                "started": datetime.datetime.now().isoformat(timespec="seconds")}
    write_manifest_atomic(root / "manifest.json", man)
    t0 = time.time()
    summary = ST.perturb_stage(th, test_ds, plan, p, root, subjects)
    man["stages"]["perturb"] = {"status": "done", "elapsed_s": round(time.time() - t0, 1),
                                "subjects": summary}
    man["updated"] = datetime.datetime.now().isoformat(timespec="seconds")
    write_manifest_atomic(root / "manifest.json", man)


def stage_report(root: Path, p: ST.Params, subjects, overrides: dict) -> None:
    import report as RP
    guard_r2_marker(root, "report")
    man = _stage_common(root, p, subjects, overrides)
    man["stages"]["report"] = {"status": "running",
                               "started": datetime.datetime.now().isoformat(timespec="seconds")}
    write_manifest_atomic(root / "manifest.json", man)
    t0 = time.time()
    rep = RP.report_stage(root, p, subjects)    # 内含上游齐全有限前置断言
    r3_flag = "r3_triggered" if rep["flags"]["r3_any_triggered"] else "none"
    man["stages"]["report"] = {"status": "done", "elapsed_s": round(time.time() - t0, 1),
                               "flags": r3_flag}
    man["updated"] = datetime.datetime.now().isoformat(timespec="seconds")
    write_manifest_atomic(root / "manifest.json", man)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stage", required=True, choices=["null", "perturb", "report"])
    ap.add_argument("--run-id", default="ni-mse-v5fmt-01")
    ap.add_argument("--config", nargs="*", default=[], help="KEY=VALUE 覆盖（冒烟小参数）")
    args = ap.parse_args()
    overrides = parse_overrides(args.config)
    p = resolve_params(overrides)                 # 参数校验先于建目录（非法调用零落盘）
    subjects = overrides.get("subjects", TH.EXPECTED_TEST_IDS)
    root = PRODUCTS_ROOT / args.run_id
    (root / "logs").mkdir(parents=True, exist_ok=True)
    print(f"[run] stage={args.stage} run-id={args.run_id} root={root} "
          f"params={p.as_dict()} subjects={subjects}", flush=True)
    {"null": stage_null, "perturb": stage_perturb, "report": stage_report}[args.stage](
        root, p, subjects, overrides)
    print(f"[run] {args.stage} 期完成", flush=True)


if __name__ == "__main__":
    main()
