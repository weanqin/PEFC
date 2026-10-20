# npi/refs — baseline 常用参考结果归档

> 目的：将已计算完成的参考结果（baseline refs）集中归档，供跨任务复用。
> 组织规则（2026-10-20 建档约定）：**`refs/<方法名>/<参数配置>/` 独立子目录**，不同方法/不同参数的结果分目录存放、不混用、不覆盖；保留原文件名；每档必含 `meta.json`（来源路径 + 方法/参数 + 文件清单 + 校验和文件）。
> 参考为只读：新增参考一律另建子目录并在本清单追加一行。

## granger_bivariate_spectral（latent_flow 原库 GC）

| 子目录 | 数据 | 关键参数 | 来源 | 归档日 |
|---|---|---|---|---|
| `edrafmt48-cont-zscore-fs250-fixed8-f1-125-nf65/` | edra_processed_fmt 48 例（HC 24/MDD 24）全长连续 60001 样本/例，逐通道 z-score | 固定阶 8、n_freq 65、1–125 Hz、fs 250、梯形积分（Geweke 1982 / Dhamala 2008） | `outputs/10-19-latent-flow-edra-fmt-train/run_01/gc_oldlib_refs`（任务 10-19-latent-flow-edra-fmt-train，2026-10-19 生成） | 2026-10-20 |

- 内容：48 × `refs_s{id}.npz`（`ec` float32 (62,62)，**ec[a,b]=a→b 行=源/列=靶**，对角 0）+ `gen_refs.py`/`gen_refs.log`（生成溯源）+ `SHA256SUMS` + `meta.json`。
- 用途：原库 GC 登记（用户 2026-10-20 拍板）为 **GC 计算主路径**，本档为其在 fmt 数据上的现成基线参考；与 A_eff（行=靶/列=源）对照的方向对齐主口径 = |A_eff| vs `ec.T`。
