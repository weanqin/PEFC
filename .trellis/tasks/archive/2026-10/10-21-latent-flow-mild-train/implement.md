# implement — 10-21-latent-flow-mild-train

> 执行清单。每阶段末跑「验证命令」，全绿才进下一阶段；S4 前置 = 1.4 用户同意 + `task.py start`。

## 阶段 S1 — manifest + 数据契约 spec

- [ ] 写 `models/latent-flow-v1/make_mild_manifest.py`（design §2.2：BLAS 守卫、RandomState(42) 分层 5-fold、全量断言、已存在拒跑）。
- [ ] 运行生成 `models/latent-flow-v1/splits/mild_5fold_seed42.json`，登记 5 折 test 名单与各折 28/4/8 计数。
- [ ] 写 `.trellis/spec/data/mild-depression.md`（按 dataset-contract-template：路径/例册/shape/fs/label 语义/通道假设链/划分算法+manifest 哈希）。
- [ ] 回填 `.trellis/research/environment.md` 数据落点一行（Mild Depression：路径/40 例 HC20-MDD20/来源 zip，评审 B-F7）。

验证：`python3 models/latent-flow-v1/make_mild_manifest.py`（重复运行应拒跑）；抽查 fold0/4 名单与计数。

## 阶段 S2 — geometry（D1 拍板后）

- [ ] 写 `models/latent-flow-v1/build_mild_geometry.py`（design §3：MNE GSN-HydroCel-128 → coords(128,3)+ch_names → geometry_mild.pt；与 MODMA geometry 模板一致性校验（防代码 bug，非数据侧证据）；`--out` 可改落点）。
- [ ] 运行构建（D1=数据根豁免案时落 `/root/autodl-tmp/Mild Depression/geometry_mild.pt`）；`load_geometry` 回读断言 shape 128/3。

验证：构建日志含 128 点/旁证比对行；`python3 -c "load_geometry(...)"` 回读成功。

## 阶段 S3 — 训练入口 wrapper + 冒烟

- [ ] 写 `models/latent-flow-v1/train_mild_depress.py`（design §5：fold 必选、manifest 断言、MildTrialDataset、wrapper 归一化、build_model_mild、smoke 1125/375、config JSON 落盘）。
- [ ] 冒烟 1：`--fold 0 --stage 1 --limit-subjects 2 --epochs 2 --save-path <run>/smoke_vae.pt`（断言窗数 30×2、损失有限值下降、config 落盘）。
- [ ] 冒烟 2：`--fold 0 --stage 2 --epochs 2 --limit-subjects 2 --vae-ckpt <smoke ckpt> --save-path <run>/smoke_flow.pt`（断言非 flow 键零失配、trainable=flow、history 复制进 run 目录）。
- [ ] 冒烟 3（dense 反切开关可用性）：`--dense-fallback --epochs 1` 一次前向。

验证：三条冒烟日志齐；`git status` 确认包体 `latent_flow/` 零 diff、无冒烟产物残留于包内 checkpoints/。

## 阶段 S4 — 正式 5-fold 训练（nohup 后台，顺序执行）

- [ ] **跑前登记**（experiment-log 契约）：nohup 启动前先落 experiment.md Run 块「目的+配置」区（5 折共块；评审 A-F5/B-F6）。
- [ ] fold0 stage1 → stage2（`nohup ... > run_fold0/stage{1,2}.log &`，逐个等待完成再启下一个）。
- [ ] （D2=b 案时）fold0 完成即暂停，向用户呈 fold0 结果（best/smoke/耗时），确认后续 1–4 折再继续。
- [ ] fold1–4 同流程（可 `for f in 1 2 3 4` 串行脚本；任一 stage 失败即停、登记、不上溯）。
- [ ] 每折 stage2 日志尾核对 smoke 行（MAE/std_ratio）；fold0 实测单 stage 耗时回填 experiment.md（修正预估值）。

验证：10 份 nohup 日志全部收尾无 traceback；每折 run 目录三件套（configs/checkpoints/logs）齐全。

## 阶段 S5 — 核对与收尾（对应 AC5–AC8）

- [ ] R5 一致性核对五项（阶段顺序/stage2 加载路径证据/trainable=flow/无 gpfm import/包体零 diff）。
- [ ] 全量 pytest 冻结口径复核（67P/0F/17S，逐目录单跑纪律不适用于本变体——latent-flow 套件本身单命令）。
- [ ] `summary.md`：逐折 stage1/stage2 best val、best epoch、早停触发与否、smoke、耗时 + 汇总统计。
- [ ] `experiment.md` Run 块（四区契约；5 折共块 + 逐折表）。
- [ ] 3.3 spec 更新（mild-depression.md 已在 S1 落；按需补 evaluation-calibers / context.md 技术路线表回填——latent-flow 行增 Mild Depression 线）。
- [ ] 3.4 commit。

验证：trellis-check 全项过；`git status` 干净。

## 验证命令速查

```bash
# manifest
python3 /root/npi/models/latent-flow-v1/make_mild_manifest.py
# geometry
python3 /root/npi/models/latent-flow-v1/build_mild_geometry.py
# 冒烟（fold0）
python3 /root/npi/models/latent-flow-v1/train_mild_depress.py --fold 0 --stage 1 \
  --limit-subjects 2 --epochs 2 --save-path /root/npi/outputs/10-21-latent-flow-mild-train/smoke/smoke_vae.pt
# 正式（fold N 两阶段；nohup 由执行时包裹）
python3 /root/npi/models/latent-flow-v1/train_mild_depress.py --fold 0 --stage 1
python3 /root/npi/models/latent-flow-v1/train_mild_depress.py --fold 0 --stage 2
# 回归
python3 -m pytest /root/npi/models/latent-flow-v1/latent_flow/tests/ -q
```

## 回滚点

新增文件均为未跟踪文件，按清单手动删除（评审 B-F4，`git checkout` 对未跟踪文件无效）：
- S1 后：删 `models/latent-flow-v1/splits/` + `make_mild_manifest.py` + spec 文件 + environment.md 回填行。
- S2 后：加删 `build_mild_geometry.py` + geometry_mild.pt。
- S3 后：加删 `train_mild_depress.py` + 任务 outputs 目录。
- S4 中单折失败：删该折 run 目录重跑，不影响他折。
