#!/usr/bin/env python3
"""run.py — latent-flow-v1 变体的 npi 侧包装入口（不属于 latent_flow 包本体）。

职责（design.md §3 包装层）：
  1. BLAS 线程守卫：在任何 numpy/torch 子进程启动前 setdefault OMP/OPENBLAS/MKL 等
     ≤8（本机 cgroup 有效 16 核、可见 128 核，防线程超订；外部 env 可覆盖）。
  2. 把首个位置参数（脚本名）转译为 `python -m latent_flow.<子包>.<脚本名>` 执行，
     其余参数原样透传（无 `--` 分隔，契约见任务 implement.md S2.1）。
  3. outputs 落点：对确认支持 `--output-dir` 类 flag 的脚本，默认注入
     `--output-root`（默认 /root/npi/outputs/latent-flow-port/）下的路径，使产物
     落 npi outputs 区而非包内相对路径 results/（design A-Minor-3）。
  4. baseline 并行上限：`--n-jobs` 上游默认 -1（全部可见核），本包装默认注入 8
     （16 核 cgroup 口径）；用户显式传 `--n-jobs` 时不注入。

用法：
  python3 run.py <scripts.name 之外的脚本名亦可，如 run_baseline_connectivity> [原样透传参数]
  python3 run.py <name> ... --output-root <dir>     # 覆盖输出根（run.py 消费该 flag）

注意：包本体零改动；直接 `-m` 直跑绕过本守卫时，须自行前置 env（见 README-port.md）。
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

# --- BLAS 守卫：必须在任何 numpy/scipy/torch 导入（含子进程）之前 ---
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "8")

VARIANT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(VARIANT_DIR))  # 使 latent_flow 可导入（供需要时内省）

DEFAULT_OUTPUT_ROOT = "/root/npi/outputs/latent-flow-port"

# 已核实支持输出类 flag 的脚本 → 注入 flag 名（来源：逐脚本 argparse 实测，2026-10-18）。
# 注入路径约定：--output-dir/--results-dir → <root>/<script>；--outputs-dir → <root>/<script>/png
OUTPUT_FLAGS: dict[str, tuple[str, ...]] = {
    "run_oc_connectivity_global_tau": ("--output-dir",),
    "run_oc_connectivity_relative": ("--output-dir",),
    "extract_graph_connectivity": ("--output-dir", "--outputs-dir"),
    "probe_oc_latent_flow_latent": ("--output-dir", "--outputs-dir"),
    "probe_component_ablation": ("--output-dir",),
    "run_baseline_connectivity": ("--results-dir", "--outputs-dir"),
}

# 并行度默认超限的脚本 → (flag, 注入值)。用户显式传 flag 时不注入。
N_JOBS_DEFAULTS: dict[str, tuple[str, str]] = {
    "run_baseline_connectivity": ("--n-jobs", "8"),
}

USAGE = (
    "usage: python3 run.py <script-name> [passthrough args...] [--output-root DIR]\n"
    "  script-name  latent_flow 子包下的模块名（scripts/ 优先，其次 baseline/ 等子包）\n"
    "  其余参数原样透传给 `python -m latent_flow.<subpkg>.<script-name>`"
)


def resolve_module(name: str) -> str | None:
    """按子包优先级把脚本名解析为 latent_flow 全限定模块名（文件系统探测，不导入）。"""
    for sub in ("scripts", "baseline", "training", "attribution", "viz", "data", "models"):
        if (VARIANT_DIR / "latent_flow" / sub / f"{name}.py").is_file():
            return f"latent_flow.{sub}.{name}"
    return None


def consume_output_root(args: list[str]) -> tuple[list[str], str]:
    """从透传参数中摘出 --output-root（子脚本无此 flag，由 run.py 消费）。"""
    out: list[str] = []
    root = DEFAULT_OUTPUT_ROOT
    i = 0
    while i < len(args):
        a = args[i]
        if a == "--output-root":
            if i + 1 >= len(args):
                sys.exit("run.py: --output-root requires a value")
            root = args[i + 1]
            i += 2
            continue
        if a.startswith("--output-root="):
            root = a.split("=", 1)[1]
            i += 1
            continue
        out.append(a)
        i += 1
    return out, root


def build_injections(name: str, passthrough: list[str], root: str) -> list[str]:
    """对 OUTPUT_FLAGS/N_JOBS_DEFAULTS 命中且用户未显式提供的 flag 注入默认值。"""
    inj: list[str] = []
    flags = OUTPUT_FLAGS.get(name, ())
    for flag in flags:
        if flag in passthrough or any(a.startswith(flag + "=") for a in passthrough):
            continue
        sub = "png" if flag == "--outputs-dir" else ""
        inj += [flag, os.path.join(root, name, sub) if sub else os.path.join(root, name)]
    nj = N_JOBS_DEFAULTS.get(name)
    if nj:
        flag, val = nj
        if flag not in passthrough and not any(a.startswith(flag + "=") for a in passthrough):
            inj += [flag, val]
    return inj


def main() -> int:
    argv = sys.argv[1:]
    if not argv or argv[0] in ("-h", "--help"):
        print(USAGE)
        return 0
    name = argv[0]
    module = resolve_module(name)
    if module is None:
        print(f"run.py: script not found under latent_flow/*/: {name!r}", file=sys.stderr)
        print(USAGE, file=sys.stderr)
        return 2
    passthrough, output_root = consume_output_root(argv[1:])
    injections = build_injections(name, passthrough, output_root)

    # 自检（G2 门禁判据：打印解析后的 OMP_NUM_THREADS ≤ 8）
    print(f"[run.py] module={module}", file=sys.stderr)
    print(f"[run.py] BLAS guard: OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS')} "
          f"OPENBLAS_NUM_THREADS={os.environ.get('OPENBLAS_NUM_THREADS')} "
          f"MKL_NUM_THREADS={os.environ.get('MKL_NUM_THREADS')}", file=sys.stderr)
    if injections:
        print(f"[run.py] injected: {' '.join(injections)}", file=sys.stderr)
    if passthrough:
        print(f"[run.py] passthrough: {' '.join(passthrough)}", file=sys.stderr)

    cmd = [sys.executable, "-m", module, *passthrough, *injections]
    # `-m` 依赖 sys.path 含变体目录：子进程经 PYTHONPATH 注入（保持调用方 cwd 不变，
    # 与任务 implement.md 的 import 冒烟口径 PYTHONPATH=/root/npi/models/latent-flow-v1 一致）。
    env = dict(os.environ)
    env["PYTHONPATH"] = (str(VARIANT_DIR) + os.pathsep + env["PYTHONPATH"]
                         if env.get("PYTHONPATH") else str(VARIANT_DIR))
    return subprocess.call(cmd, cwd=os.getcwd(), env=env)


if __name__ == "__main__":
    sys.exit(main())
