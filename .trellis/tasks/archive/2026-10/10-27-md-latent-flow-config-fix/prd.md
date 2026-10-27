# md-latent-flow 线配置固定：refs 取消 v2 gc-ec 1-125hz 对比项 + 默认公式切 v4

## 子研究路由（1.0b / 1.0c）

<!-- trellis-research-routing -->

- **分类结果**：`study2`（EC 推断算法线的配置/工具维护——不动研究结论，只固定对比配置与交付公式默认值）
- **依据**：任务主体 = md-latent-flow 线 EC 对比流水（refs 双 GC 对照 + conn 公式默认值）的配置清理；无训练、无推断重跑、无新结论产出。
- **新路径/方法检测（1.0c）**：`已有路径：recompute_conn_v4.py 双 GC 对照 / ec_mild_unbiased.py 主跑 conn 公式（多先例在档）`——无新方法，research adaptation 不触发。
- **research 产物**：`无（配置固定，不改研究记忆结论；v2 档注销事实随本任务 journal 登记）`
- **路由备注**：v4 公式（−(ln σ*+0.5·ln2πe)+5/7）为 2026-10-24 用户拍板交付公式，本任务将其升为线默认；v2 refs 档（mild40 aic4-25-f1-125）物理文件用户已删，本任务清理配置残留引用。

## Goal

用户 2026-10-27 拍板两项配置固定，仅围绕该配置修改：

1. **refs 取消 v2 gc-ec 1-125hz 对比项**：`refs/granger_bivariate_spectral/mild40-trial500-zscore-fs250-aic4-25-f1-125-favg/`（v2 档，1–125 Hz AIC 4–25）物理文件已在用户工作区删除；清理代码/清单中的全部对比引用，不影响 v1 mild 档（1–45 Hz fixed8）与 edra 两档。
2. **默认计算公式切 v4**：`ec_mild_unbiased.py` 主跑 conn 由房内 v1 式 `−(ln σ*+ln2πe)+1` 切换为 v4 式 `−(ln σ*+0.5·ln2πe)+5/7`（计算逻辑、配置常量、公式声明串、口径注记全部指向 v4）；相关注释/诊断/汇总措辞同步。

## Requirements（改动点定位，侦察在档）

v2 档残留引用共 4 文件；v1→v4 默认公式切换点集中 1 文件：

| 文件 | 现状 | 改动 |
|---|---|---|
| `models/latent-flow-v1/recompute_conn_v4.py` | `REFS` 字典含 v2 条目（L55–57）；docstring 双 GC 段（L14–18）；`refs_caliber` 双分支（L183–187）；json rounding 诊断假定输入 conn 为 v1（L108/L123，默认切 v4 后将失配） | 删 v2 条目与分支；docstring 改单参考；rounding 诊断按档 `connectivity_formula` 声明自适应（旧档 v1 / 新档 v4），键名 `v1_json_f32_rounding_maxabs` → `json_f32_rounding_maxabs` |
| `models/latent-flow-v1/ec_mild_unbiased.py` | conn 计算 L314 `−(ln σ*+ln2πe)+1`；公式串 L439；summary caliber L593；gccorr protocol L687；docstring L7 | 四处 + 计算行全部切 v4；新增 `HALF_LOG_2PIE`/`V4_OFFSET` 常量（与 recompute_conn_v4.py 命名对齐） |
| `models/latent-flow-v1/crossfold_summary_f12.py` | `REFS_KEYS` 含 v2（L91）；summary refs 描述块（L232–235）；控制台 `v2 GC rho` 行（L273–275）；`asserts_passed == 40` 硬断言（L132）；docstring/CALIBER_NOTE「双参考 16/16 / 双 GC」措辞 | REFS_KEYS 收 v1 单键；删 v2 汇总行；断言放宽 `in (32, 40)`（40=双参考时代在档档、32=v1 单参考新配置复跑）；措辞同步注明 2026-10-27 配置固定 |
| `models/latent-flow-v1/viz_ec_pert.py` | `REFS_DIRS` 含 v2 档（L62–64）；`ANCHOR_GROUPMEAN_RHO` 含 v2 锚（L68）；图 2 三联契约（docstring L14–16 + `fig_group_vs_gc` 1×3 布局 L189） | 删 v2 条目/锚；图 2 改二联（1×len(panels) 布局）；断言只留 v1 锚；契约措辞同步 |
| `refs/MANIFEST.md` | v2 档表格行（L24）+ 两条 aic4-25（v2）bullet（L29–30） | 删行删 bullet，加注销注记（2026-10-27 用户拍板，文件已删；其余各档不动） |

**不受影响（已核对）**：`viz_symw_ec_pairs.py`（仅 v1 refs）；`symmetry_diag_symw.py`（v4 已是主读出、v1 为在档并置描述，非默认配置）；`models/ec-globaltau-v1/*`（命中的 f1-125 为 edra 档，非 md 线）；`.trellis/tasks` 归档与 `outputs/` 历史产物（在档 `gccorr_dual_summary.json` 的 v2 块为历史数据，保留不重生成）。

## Acceptance Criteria

- [ ] **AC1 refs 无 v2 对比项**：`grep -rn "v2_1_125_aic4_25\|aic4-25\|EC_GC_1_125" models/ refs/MANIFEST.md` 的命中仅限「注销记注」本身（执行期修订：3 处注释/清单注记保留取消事实与目录名供溯源，非活跃对比引用）；活跃引用（REFS 字典条目 / REFS_KEYS 成员 / REFS_DIRS 条目 / npz 加载路径 / MANIFEST 表格行）零残留（edra 档名 `edrafmt48...f1-125` / `edrav4-46...f1-125` 不属本项、允许保留；产物文件名 `gccorr_dual_summary.json` 为在档 outputs 契约名，不改名）。
- [ ] **AC2 默认公式 = v4**：`ec_mild_unbiased.py` conn 计算行与全部公式声明串（payload `connectivity_formula`、summary caliber、gccorr protocol、docstring）均为 v4 式；代码内不再默认出现 v1 式计算路径。
- [ ] **AC3 其他对比项不受影响**：v1 mild 档引用（`ec_mild_unbiased.py` REFS_DIR、`viz_symw_ec_pairs.py`、`viz_ec_pert.py` v1 条目）逐处保留；edra 两档引用不动。
- [ ] **AC4 语法/性质可验证**：4 个 .py 全过 `python3 -m py_compile`；v4 性质复核（网格 [0.01,0.48] 上 conn 全正 min=+0.0293、与 1/(σ*+1e-4) 重映射 Spearman 不变性保持）以只读一行式验证（不跑 GPU、不写 outputs）。
- [ ] **AC5 配置残留核对（用户核对项）**：refs 中不存在 v2 gc-ec 1-125hz 对比项；md-latent-flow 线默认 v4；注释/示例/回退配置同步（含 recompute 诊断自适应、crossfold 断言两态、MANIFEST 注销注记）。

## 边界（不做）

- 不跑任何 GPU 主跑/重算；不重生成 `outputs/` 历史档（含 crossfold_summary.json 在档版本）。
- 不动 `.trellis/tasks/` 归档任务记录（历史 v2 引用属任务档案，非活跃配置）。
- 不动两个遗留 in_progress 任务（10-24-latent-flow-exit-gnn / 10-24-latent-flow-mild-w500）的档位。
- 研究记忆（progress.md）中 v2 refs 相关历史结论段不回写改写（历史事实），仅在 wrap-up 时登记配置固定事件。
