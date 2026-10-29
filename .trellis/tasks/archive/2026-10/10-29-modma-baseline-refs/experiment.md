# 实验日志 — 10-29-modma-baseline-refs

> 归档注册任务（无 GPU 跑批），Run 001 = 三方法 × 5 折注册入 refs。

## Run 001 — MODMA 5 折 baseline（gc/gpd/te）注册入 refs（2026-10-30）

### 目的与假设

- 用户指令：把 MODMA 5 折 baseline 注册到 refs（其下载的旧库原版 zip 为源）。

### 配置

- 源：`/root/autodl-tmp/modma-gc-ec.zip` 解包树（旧库原版矩阵 2026-07-18/07-23；gc fold0 三方逐位验证在先）
- 目标：`refs/{granger_bivariate_spectral/gpdc_pairwise_bivariate/symbolic_transfer_entropy}/modma5fold-*/`（refs 规范：独立子目录 + 原文件名 + SHA256SUMS + meta.json + 可重放 register_refs.py）
- 命令：各档 `python3 register_refs.py`（--src 默认解包根，可重放）

### 结果

- 三档注册完成：gc 50 文件（每折 10）、gpd **53 文件（fold2-4 各 11）**、te 50 文件；`sha256sum -c` 全过（0 非-OK）。
- gpd 比 gc 多出的 3 例 = gc 档缺的各折第 11 例（fold2 02020014 / fold3 02020022 / fold4 02020025）——已验证并在 meta/MANIFEST 注记。
- `refs/MANIFEST.md` 新增「旧库 baseline 三方法（MODMA 5 折）」节 + 三行档案表 + 用途边界。
- 产物：`refs/{三方法}/modma5fold-*/`（153 json + 3×{register_refs.py, register.log, SHA256SUMS, meta.json}）。

### 结论

- 注册合规（refs 建档约定逐项满足：只读原则保持、原文件名、meta 含被试映射与三方验证溯源链）；MODMA 线后续 5 折对照直接读 refs，不再依赖 autodl-tmp 解包树。
