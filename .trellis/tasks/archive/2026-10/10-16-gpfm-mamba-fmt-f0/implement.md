# implement.md — 执行计划

## 前置（start 前）

- [x] 1.4.3 用户明确同意（ briefing 后「确认开始」）
- [x] experiment.md 首个 Run 块模板就位（目的/配置/结果/结论四区）

## A. 共享模块 default-off 钩子（v1 目录；review 回填后范围）

- [x] A1 `gc_psd.py::run_all` 加 `skip_gc=False` 参数：True 分支跳过 BIC/VAR/GC 一致性/bootstrap，PSD + 空间归因照跑，summary 记 `"gc": "skipped(blocked)"`，**不落任何 GC 产物**；默认 False 路径逐位不变；docstring 频率注释顺带 fs 参数化
- [x] A2 `evaluate.py` 三点：①run_all 调用点 fs → `getattr(cfg, "fs", 500.0)`；②`--skip-gc-only` 与 `--skip-gc` 入 mutually exclusive group；③`_traj_fig_data` `"fs": 500` → `getattr(cfg, "fs", 500.0)`（review Major：指标2 轨迹图时间轴）
- [x] A3 `fc_metric.py`：`--outputs-root` 参数（默认 = 原硬编码路径）
- [x] A4 **norm 类型钩子**（D4）：`mamba.py`（MambaBlock + build_mamba_backbone 加 `norm_type="rmsnorm"` 参数）+ `model.py`（`__init__` 参数 → head_norm 与 backbone 透传）+ `train.py:278`/`sample.py:52` 构造点 `getattr(cfg, "norm_type", "rmsnorm")` 透传；默认分支 = nn.RMSNorm 原结构逐位一致；docstring 两处 RMSNorm 表述参数化
- [x] A5 单测（放 v1-fmt tests，参照 v4a 八测风格）：
  - 旧变体 hash 零漂移：v1/v2/v4a config_hash 与冻结期望值逐位一致（钩子未改任何旧 config）
  - **norm 钩子**：默认 norm_type 下模型结构与 state_dict 键与改前逐位一致（RMSNorm）；fmt config（layernorm）下 5 个 norm 位全部 nn.LayerNorm 且 state_dict 含 `*.norm.bias`/`head_norm.bias` 键
  - `run_all(skip_gc=True)` 合成小输入冒烟：产出 PSD/空间产物；**无 gc_real/gc_gen/gc_F_p_* npy、结果 json 无 gc 数组/图引用且记 `"gc": "skipped(blocked)"`**（review：断言面补全）；`skip_gc` 默认 False 签名兼容
  - `fc_metric` 默认 outputs 根 = 原值；`_traj_fig_data` 默认 fs=500 不漂移
- [x] A6 既有单测全数复跑通过（回归，**含 `models/gpfm-mamba-v4/tests`**——review 补）

## B. v1-fmt 变体目录

- [x] B1 `config.py` 完整副本 + design §1.2 字段差异（data_root/outputs/variant_name/p=25/gen_lowpass_hz=None/fs=250/n_val=4/override/早停10/恒 lr/norm_type="layernorm"）
- [x] B2 `train.py` / `evaluate.py` 包装：单遮蔽（config→本目录、model→v1）+ run_id 前缀守卫 `v1fmt-` + 独立模块名加载 v1 入口（无 --rank）
- [x] B3 `fc_metric.py` 薄包装或直接用 v1 脚本 + `--outputs-root` 传参（二选一，实现时按 wrapper 惯例定）
- [x] B4 tests：config 字段断言 + 前缀守卫断言
- [x] B5 smoke：`--smoke --run-id v1fmt-smoke-01`（每类 2 例 1ep）→ dataloader 契约核对写细（review 盲区回填）：每例 930 窗、`data (62,60001)` float32、X_past (B,62,384)/X_future (B,62,128)、z-score 输出有限性
- [x] B6 fold0 manifest 固化核对 + 登记：**固化发生在 B5 首次 smoke**（`fixate_split` 先于小子集截取，写入完整 48 例口径 34/4/10；review 实测模拟 test=10 例 ids [10,11,12,17,24,28,30,44,61,69]——以实际落盘为准登记）；commit ② 含 splits/ 新文件
- [x] B7 bench：`--bench --epochs 1 --run-id v1fmt-bench-1ep` → 显存/时长外推 100ep
- [x] B8 主 Run：nohup `--run-id v1fmt-main-01`（≤100ep 早停 10）→ best/last ckpt + metrics 落盘
- [x] B9 评估：`evaluate.py --run-id v1fmt-main-01 --skip-gc-only` → Q1 + PSD(fs=250) + 空间归因；`fc_metric --outputs-root .../gpfm-mamba-fmt-f0 --run-id v1fmt-main-01` → 指标1；指标3 判读按 design §1.4-4 预登记口径补算（1–30 Hz mean |Δlog10|，psd.npz 数据源）；确认产物无任何 GC 对比文件

## C. v4a-fmt 变体目录（镜像 B）

- [x] C1 config（v4a 副本 + 同款差异字段；bottleneck_rank=16 保持）
- [x] C2 wrappers（含 --rank {8,16,32} 透传对齐 v4a；前缀 `v4afmt-`）
- [x] C3 tests（含：首跑后与 v1-fmt manifest test 折逐 ID 一致断言）
- [x] C4 smoke → C5 manifest 校验复用（与 B6 逐字节一致）→ C6 bench → C7 主 Run（`v4afmt-main-01`）→ C8 评估（同 B9）

## D. 判读与收尾

- [x] D1 experiment.md 全 Run 块回填（结果+结论；标注「单次运行」；design §1.4-5 三条判读注记随附）
- [x] D2 跨范式对比表：fmt vs 旧 v1/v4a（Q1/FC/PSD），随附差异清单（fs/预处理/48例/p=25/无低通/训练纪律）+ **对照纯净度不对称注记**（v4a 侧权重更高——review Mandate A）
- [x] D3 生成侧 >40 Hz 功率占比观察（无低通口径下模型是否自行待在带内——D2 拍板的观察项；**数据源 `eval/psd.npz` 小脚本，禁用 v1 diag_hf.py**——review：fs=500+旧 run id 硬编码）
- [x] D4 指标2 视觉图交用户人工判（轨迹图时间轴经 A2③ 修正后为 250 Hz 正确标秒）
- [x] D5 5-fold 启动建议留待用户拍板（门禁：前置验证经用户同意）
- [x] D6 trellis-check 全量检查
- [x] D7 spec 回填（3.3）：`.trellis/spec/data/` 新增 edra-fmt 数据集契约；evaluation-calibers + surrogate-brain-qualification 指标3 频段定义修正（1–30 Hz 判读窗 / 40–125 Hz 观察段，fs 耦合适配）+ 新范式口径登记（无低通 + GC 阻塞 + p=25）；context.md 技术路线表现更新；progress.md 当前状态同步
- [ ] D8 journal + 归档（/finish-work）

## 验证命令

```bash
# 单测（A5/A6/B4/C3 全绿为门禁；v4/tests 为 review 补入的回归面）
python3 -m pytest models/gpfm-mamba-v1/tests models/gpfm-mamba-v4a/tests \
  models/gpfm-mamba-v4/tests models/gpfm-mamba-v1-fmt/tests \
  models/gpfm-mamba-v4a-fmt/tests -q
# 冒烟与训练（经各变体包装入口，run_id 前缀守卫生效；wrapper 保留 BLAS 护栏块）
python3 models/gpfm-mamba-v1-fmt/train.py --smoke --run-id v1fmt-smoke-01
# 评估（GC 阻塞 + 无低通 + fs=250）
python3 models/gpfm-mamba-v1-fmt/evaluate.py --run-id v1fmt-main-01 --skip-gc-only
```

## 风险文件与回滚点

| 风险 | 触发信号 | 回滚 |
|---|---|---|
| 共享钩子改坏旧变体行为 | A5/A6 单测红、旧 config_hash 漂移 | revert v1 目录钩子（逐处），重设计 |
| dataloader 与 fmt 数据契约不匹配 | B5 smoke 形状/窗数不符 | 停训练，修数据接入层（禁改 fmt 数据本体） |
| 250 Hz 遗留 fs 假设 | PSD 频轴/图异常、低通误触发 | 核对 fs 透传链；低通本任务恒 None |
| 主 Run 中断 | last.pt 存在 | --resume 沿用同 run 目录续跑（前缀守卫兼容） |

## 提交点约定（对齐 10-16 惯例）

A+B1-B4+C1-C3（代码+单测）→ commit ①；smoke/bench → commit ②；两主 Run+评估 → commit ③；判读+spec 回填 → commit ④；归档+journal → ⑤
