# implement.md — 10-17-perturb-ec-f0 执行计划

> 前置：prd.md / design.md 已定稿（含双 mandate review 回填）；θ\* ckpt 与 fold0 manifest 已核。每步产物落 `/root/npi/outputs/perturb-ec-f0/<run-id>/`；experiment.md Run 块**先登记后跑**（experiment-template.md）。

## B0 脚手架 + 单测（提交点①）

- [ ] `models/perturb-ec-v1/`：config.py（全部扰动侧口径常量：ckpt 路径、R_GRID、K=8、N_w=32、τ 分位 {90,95,99}、seeds 2026、方向约定注释；R_NEXT/翻转常数由 R_GRID 推导禁字面量）/ loader.py（importlib 唯一名装载，**sys.modules 先注册后 exec**，design §1 四行配方；`split_manifest_override` 存在性前置断言）/ perturb.py（A 案增益 + B 案 default-off）/ response.py / readout.py 骨架。所有入口（run.py、tests 启动器、子进程启动器）头部 BLAS setdefault 块（numpy 前）。
- [ ] tests/（①②③惯例，rng 2026）：
  - ① CRN 恒等：r=0 变体与 baseline **同一 sample() 调用同批内**相邻元素逐位相等；扰动仅改源通道输入行；同 eps/eta 两次调用逐位相同
  - ② 采样器口径：`sampler.gen_lowpass_hz is None`；装载 config_hash == 05d9bb875a2c；`sys.modules['config']` 未被本任务占据且 sample/data/gpr_prior `__file__` 归属 v1 目录；`posterior_mean` 扰动差 B≠A 通道严格为零（先验逐通道事实断言）
  - ③ readout 数学（合成数据）：D̄(r)=2r → slope=2、AUC 解析值；永不达 τ → r\*=R_NEXT（右删失）→ strength=0；翻转/R_NEXT 由 R_GRID 推导；load_ec_pert 契约 npz 往返（方向/非负断言）
- [ ] 验证：单测全绿 + ckpt 装载 GPU 冒烟（1 batch 采样 shape/finite 断言）。
- [ ] 回滚：单测红 → 停在装载/采样层，不进 B1。

## B1 smoke：1 例 × 2 窗全链（首 Run 块登记）

- [ ] experiment.md Run 块（smoke-01）登记**全部四要素口径**（AC1：P/D/τ 池化域=逐被试分池/eps-eta 共享范围/网格/K/N_w/τ 分位/方向约定/gen_lowpass=None 实现口径声明）——先登记后跑。
- [ ] `run.py --stage smoke`：2 窗 × 62 源 × r 网格 × K → D 曲线 + null + τ + readout 骨架全链跑通；抽 4 条 (A→B) 曲线快图人工核形态。
- [ ] 检查点：曲线无 NaN；A=B 自响应显著大于 A≠B（定性）；CRN 恒等复验。异常 → design §8 回滚点②。

## B2 bench：1 例 N_w=32 计时（档位定版）

- [ ] `run.py --stage bench --subjects 1`：N_w=32 扰动链全 62 源计时（每窗单次 sample()、变体并批 B=497）→ 10 例外推时长登记；GPU 显存峰值登记。
- [ ] N_w=32 vs 930 全集对照抽查（baseline 侧可得量的受试者级 median 稳定性）；**档位调整最多一次**（如调则登记新旧不可比部分 + 外推警示重登记）。

## B3 主 Run：10 例扰动批采样（提交点②）

- [ ] Run 块（main-01）登记后 `mkdir -p outputs/perturb-ec-f0/main-01/logs/` 再 nohup：10 例 × N_w=32 × (62 源 × 8 档 + baseline) × K=8 CRN 采样（每窗一次调用）。
- [ ] 产物：`outputs/perturb-ec-f0/main-01/`：窗级 D 曲线 npz（10×62×62×9）、null 样本池（逐被试）、per-subject D̄ 曲线、τ 三分位（逐被试）、config.yaml + env_snapshot.txt、logs/。
- [ ] 完整性断言：shape/finite/例数/源数齐全；进度日志逐批。

## B4 readout + 有效路径 + 诊断图（提交点③）

- [ ] 三 readout（slope 主 / AUC 次 / r\* 右删失+敏感性）→ 受试者级 EC 矩阵 npz + load_ec_pert 契约 npz ×3 readout。
- [ ] 诊断七件套（design §6）：曲线样例、单调性/R² 分布、地板分离+删失率、空间衰减、对角 vs 跨通道（含先验自响应分量解析量化）、矩阵热图+top-100 路径（figs/，common/visualizer 纯增量扩展）。
- [ ] 「流程可跑通 + 响应真实性」结论登记（AC4）。

## B5 参考侧 + 比较器（并入提交点③）

- [ ] `run.py --stage refs`：fmt 10 例 → 125 Hz 重采样+z-score → **子进程**调 `refs_for_subject(p=8)` → gc/gpdc npz（fresh python、sys.path 只含 baseline 目录、npz 文件面交换）。
- [ ] `run.py --stage split-half`：fmt 10 例天花板重算（**必做**，review A 对齐预登记）。
- [ ] `run.py --stage compare`：EC_pert ×3 readout vs {gc, gpdc} → 主比较（slope × 2 法 Holm）+ 上下文（auc/rstar、未偏 ρ、天花板比值）→ comparison.json/matrices.npz。
- [ ] AC5/AC6 判定登记：初步证据门槛（ρ̄>0 且 Holm p<0.05）达成与否；未达 → 缺口诊断（删失率/地板贴合度/形态异常分项）+ 下一步选项；阴性=证据不足非证伪（power 注记随附）。

## B6 质检 + 收尾（提交点④）

- [ ] trellis-check 全量（含 experiment.md 完整性、路径隔离扫描——common/visualizer.py 增量扩展为 design 声明的已知写入点、产物布局）。
- [ ] 3.3 spec 更新：Study II 四要素首版流程规范沉淀候选（evaluation-calibers Study II 节回填实际口径 或 新 spec 文件，按 check 建议）；research 记忆回填（context.md Study II 行状态、progress.md 节点）。
- [ ] journal 记录 + `task.py finish` + archive。

## 验证命令速查

```bash
python3 models/perturb-ec-v1/tests/run_tests.py            # 单测（或 pytest）
python3 models/perturb-ec-v1/run.py --stage smoke          # 1 例 2 窗全链
python3 models/perturb-ec-v1/run.py --stage bench --subjects 1
mkdir -p outputs/perturb-ec-f0/main-01/logs && \
  nohup python3 models/perturb-ec-v1/run.py --stage main \
  > outputs/perturb-ec-f0/main-01/logs/run.log 2>&1 &
python3 models/perturb-ec-v1/run.py --stage readout / refs / split-half / compare / figs
```
