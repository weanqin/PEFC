# consent — 10-29-modma-baseline-refs

- **时间**：2026-10-30（会话 62da5e76）
- **同意凭据**：用户明确指令（2026-10-30）：「将modma数据集上5个fold上的baseline都注册到refs当中」——点名 refs 注册的执行令（本会话 10-28/10-29/10-30 三次同模式先例）。
- **briefing 要点**：
  1. 数据源 = 用户下载的旧库原版 zip（`/root/autodl-tmp/modma-gc-ec.zip` → 解包 `modma-gc-ec/`），含 gc/gpd/te × fold0-4 矩阵；gc fold0 已三方逐位验证（zip = 旧库残留 = 10-27 现算）。
  2. 注册形态 = refs 规范三件套：`refs/<方法名>/<参数配置>/`（独立子目录、原文件名、fold 子目录）+ SHA256SUMS + meta.json（被试映射/口径/溯源）+ 可重放 register_refs.py；MANIFEST.md 追加三行。
  3. 风险 = 无（纯归档增量，不触既有档；refs 只读原则保持——新增另建子目录）。
  4. 验证 = 计数对表（gc 10×5、gpd 10+10+11×3、te 10×5）、抽查校验和、MANIFEST 行齐。
