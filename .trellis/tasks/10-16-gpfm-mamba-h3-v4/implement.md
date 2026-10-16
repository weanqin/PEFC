# Implement — H3 v4-A 低秩输入瓶颈（执行计划 · v4-B 放弃后复活版）

> v4-B 各步已完成并存档（Run 001–010）；本计划自 v4-A 起接续，Run 编号续 011。当前累计口径：32/4/10、早停 10、恒定 lr 1e-3。

## 1. v1 default-off 钩子 + v4a 目录（design §4）

- [x] v1 `model.py`（bottleneck_rank 参数/条件 in_proj）+ `train.py`/`sample.py`（getattr 透传）
- [x] `models/gpfm-mamba-v4a/`：config（v2 副本 + 当前口径字段，无 d_z/head_act）+ train/evaluate wrapper（单遮蔽 config、`v4a-` 前缀、`--rank` 覆写）+ tests 八项
- [x] v1 七测回归 + 三变体 hash 零漂移
- [x] **提交点 A'**（c329013）

## 2. 冒烟 + 1-epoch bench（Run 011）

- [x] `--smoke --run-id v4a-smoke-a` → `--epochs 1 --bench --run-id v4a-r16-bench-1ep`（实测 8.6 s/ep、1.95 GiB、损失 1.4982→1.1484 正常下降）
- [x] 用户「尝试 v4a」已授权方向：bench 无异常即直接进主 Run（experiment.md Run 011 留痕）
- [x] **提交点 B'**（c329013）

## 3. 主 Run（Run 012）

- [x] 跑前登记；nohup 脱离式启动（`--run-id v4a-r16-main-01`，早停 73ep / best ep63，best val 0.77901）
- [x] 监控（周期巡检）；无中断，未需 `--resume`
- [x] **提交点 C'**（349aedd）

## 4. 正式评估（Run 013）

- [x] evaluate + fc_metric（Run 017 同口径）；七维度五列对比（v1/v2/v4b-d1/v4b-d8/v4a）+ 三检验 + R1 + std_t
- [x] **提交点 D'**（349aedd）

## 5. H3 判读与收尾

- [x] 三态判读建议 + 与 v4b 剂量对照的联合陈述 → 交用户（Run 013 结论区）
- [x] 用户判定留痕（experiment.md「任务收尾」：H3 落「不支持」侧；5-fold 候选集 {v1, v4a}；r 扫描不采纳）
- [x] 2.2 trellis-check（Agent 形式）；3.3 spec 回填（context.md 变体注记 v4a/v4b、data-pipeline 7/1/2 长期口径、reproducibility 早停/恒定 lr 钩子惯例）；progress.md 同步；3.4 提交；finish-work

## 验证命令汇总

```bash
python3 models/gpfm-mamba-v4a/tests/test_v4a.py
for t in test_model test_sample test_gpr test_losses test_data test_gc_psd test_bandlimit; do \
  python3 models/gpfm-mamba-v1/tests/$t.py || exit 1; done
python3 models/gpfm-mamba-v4a/train.py --smoke --run-id v4a-smoke-a
python3 models/gpfm-mamba-v4a/train.py --epochs 1 --bench --run-id v4a-r16-bench-1ep
nohup python3 models/gpfm-mamba-v4a/train.py --run-id v4a-r16-main-01 > …/v4a-r16-main-01/logs/train.log 2>&1 &
python3 models/gpfm-mamba-v4a/evaluate.py --run-id v4a-r16-main-01
python3 models/gpfm-mamba-v1/fc_metric.py --run-id v4a-r16-main-01
```
