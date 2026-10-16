# Implement — H3 v4-A 低秩输入瓶颈（执行计划 · v4-B 放弃后复活版）

> v4-B 各步已完成并存档（Run 001–010）；本计划自 v4-A 起接续，Run 编号续 011。当前累计口径：32/4/10、早停 10、恒定 lr 1e-3。

## 1. v1 default-off 钩子 + v4a 目录（design §4）

- [ ] v1 `model.py`（bottleneck_rank 参数/条件 in_proj）+ `train.py`/`sample.py`（getattr 透传）
- [ ] `models/gpfm-mamba-v4a/`：config（v2 副本 + 当前口径字段，无 d_z/head_act）+ train/evaluate wrapper（单遮蔽 config、`v4a-` 前缀、`--rank` 覆写）+ tests 八项
- [ ] v1 七测回归 + 三变体 hash 零漂移
- [ ] **提交点 A'**

## 2. 冒烟 + 1-epoch bench（Run 011）

- [ ] `--smoke --run-id v4a-smoke-a` → `--epochs 1 --bench --run-id v4a-r16-bench-1ep`（预期 ~9 s/ep、~2 GiB、损失走势对照 v2 bench）
- [ ] 用户「尝试 v4a」已授权方向：bench 无异常（无 NaN/显存安全/损失正常下降）即直接进主 Run，不再等待；异常则停下汇报
- [ ] **提交点 B'**

## 3. 主 Run（Run 012）

- [ ] 跑前登记；**nohup 脱离式启动**（`--run-id v4a-r16-main-01`，100ep 上限 + 早停 10）
- [ ] 监控（周期巡检，勿用可取消的阻塞等待）；中断则 `--resume`
- [ ] **提交点 C'**

## 4. 正式评估（Run 013）

- [ ] evaluate + fc_metric（Run 017 同口径）；七维度**五列**对比（v1/v2/v4b-d1/v4b-d8/v4a）+ 三检验 + R1 + std_t
- [ ] **提交点 D'**

## 5. H3 判读与收尾

- [ ] 三态判读建议 + 与 v4b 剂量对照的联合陈述（接口混合 vs 顶层显式读出 vs 损失侧缺口）→ 交用户
- [ ] 2.2 trellis-check（Agent 形式）；3.3 spec 回填（context.md 变体注记 v4a/v4b、data-pipeline 7/1/2 长期口径、早停/恒定 lr 钩子惯例）；progress.md 同步；3.4 提交；finish-work

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
