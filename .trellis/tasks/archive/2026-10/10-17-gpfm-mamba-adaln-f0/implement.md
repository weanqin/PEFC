# implement.md — 执行计划

## 前置（start 前）

- [ ] 1.4.3 用户明确同意（briefing 后「确认开始」）
- [ ] experiment.md 首个 Run 块模板就位（目的/配置/结果/结论四区）
- **硬性顺序约束（review Minor）**：A4 零漂移断言的「git HEAD 改前对比」**必须在 commit ① 之前运行并留输出**（钩子一旦提交，HEAD 即改后代码，对比退化为恒真）；commit ① 之后仅 A5 回归面有效

## A. v1 共享库 default-off 钩子（design §1.1）

- [ ] A1 `mamba.py`：模块级 `modulate(x, s, b)`（DiT 式 `(1+s)⊙x+b`，(B,2d)→切分 (B,1,d) 广播）；`MambaBlock.__init__` 加 `cond_adaln="off", mod_dim=None`（启用 `"scale_shift"` 时**在 `with torch.random.fork_rng(devices=[]):` 内**建 `self.mod = Sequential(SiLU(), Linear(mod_dim, 2*d_model))`，末层 weight=0/bias=0——**RNG 中性构造，design §1.1 契约**；`"zero"` 传 NotImplementedError；非法值 ValueError）；`forward(x, mod=None)`——`mod is None` 原式逐位不变；`build_mamba_backbone` 尾部 kwargs 透传两参（返回仍 nn.Sequential 键名不变）
- [ ] A2 `model.py`：`VelocityNetwork.__init__` 加 `cond_adaln="off"`（norm_type 后）；启用时**fork_rng 内**建 `head_mod`（零初始化）+ backbone 透传；`forward` adaLN 分支（temb 重算 → 块循环 `h = blk(h, temb)` → head 位 modulate），默认分支（`self.blocks(h)` 单行 + `head(head_norm(h))`）逐字节保持
- [ ] A3 `train.py:279` / `sample.py:53` 后各一行：`cond_adaln=str(getattr(cfg, "cond_adaln", "off"))`
- [ ] A4 单测（放本变体 tests，对仗 test_v4afmt.py 四段式 + test_fmt_hooks 零漂移式）：
  - **零漂移**：六旧变体（v1/v2/v3/v4a/v1-fmt/v4a-fmt）config_hash 与冻结 pin 逐位一致（v2/v3 无测试套件，由 pin 断言覆盖；**v4/v4b 自有 model.py 但经 v1 mamba.py 构建骨干，共享钩子影响面含之——经 A5 test_v4b 回归覆盖**，review Minor）；`cond_adaln="off"` 下模型结构与 state_dict 键与改前（**commit ① 前**的 git HEAD 子进程干净副本）逐位一致、`mod=None` 前向逐位一致
  - **adaLN 分支**：5 调制位键齐全（`blocks.{i}.mod.1.{weight,bias}` ×4 + `head_mod.1.{weight,bias}`）且形状 (124,16)/(124,)；参数 pin **140,906**（基线 130,366 + 10,540）；**恒等初始化下同 seed 同输入前向与 `norm_type="layernorm"` + cond_adaln="off" 基线逐位一致（`torch.equal` 值相等语义——±0.0 字节异例外，review Suggestion；fork_rng 契约下共享权重同基线）**；手工设非零调制权重后输出确实变化（调制路径活体断言）；`mod=None` 分支与旧 MambaBlock 逐位一致
  - **构造链 round-trip（review Minor）**：经 `sample.load_eval_model` 同款 kwargs 路径构建（覆盖 sample.py:53 透传行真实执行面），断言加载模型含全部 mod 键；tiny EulerSampler 数步 CUDA 冒烟（adaLN 前向分支首触达前移，不等 B7）
  - 非法 `cond_adaln` 值构建期 ValueError；`"zero"` NotImplementedError
- [ ] A5 既有单测全数复跑通过（回归：v1/v4a/v4/v1-fmt/v4a-fmt 五套）

## B. 变体目录 `models/gpfm-mamba-v4a-fmt-adaln/`

- [ ] B1 `config.py`：v4a-fmt 完整副本 + design §1.2 差异表四字段（cond_adaln="scale_shift" / variant_name / outputs_root / split_manifest_override 本目录）；文件头差异清单注释（单变量声明 + 参数增量登记）
- [ ] B2 wrappers：train/evaluate/fc_metric（v4a-fmt 先例三要素；RUN_PREFIX `v4afmt-adaln-`；无 --rank；BLAS 护栏块；docstring 登记「`v4afmt-adaln-*` 归本任务根」+ 嵌套局限注记——design §1.2）
- [ ] B3 tests：config 字段断言（含 hash pin）+ 前缀守卫断言（train/evaluate/fc_metric 反例，**含 --resume 派生 run 名反例**——path-isolation 协议要求，review Minor）+ **与 v4a-fmt splits manifest 逐字节一致断言**
- [ ] B4 smoke：`--smoke --run-id v4afmt-adaln-smoke-01`（每类 2 例 1ep）→ dataloader 契约核对（930 窗/例、data (62,60001) float32、X_past (B,62,384)/X_future (B,62,128)、z-score 有限性）；manifest 固化（首跑写本目录 splits/，与 v4a-fmt 逐字节一致登记）
- [ ] B5 bench：`--bench --epochs 1 --run-id v4afmt-adaln-bench-1ep` → 显存/时长外推 100ep
- [ ] B6 主 Run：nohup `--run-id v4afmt-adaln-main-01`（≤100ep 早停 10）→ best/last ckpt + metrics 落盘
- [ ] B7 评估：`evaluate.py --run-id v4afmt-adaln-main-01 --skip-gc-only` → Q1 + PSD(fs=250) + 空间归因；`fc_metric --outputs-root /root/npi/outputs/gpfm-mamba-adaln-f0 --run-id v4afmt-adaln-main-01` → 指标1；指标3 预登记口径补算（1–30 Hz mean |Δlog10|，psd.npz 数据源；脚本**从归档任务复制到本任务 `research/psd_band_metric3.py`** 改 run 路径后用，禁在归档目录增改——review Suggestion）；确认产物无任何 GC 对比文件（逐文件核验）

## C. 判读与收尾

- [ ] C1 experiment.md 全 Run 块回填（结果+结论；标签「前置验证 · fold0 · 单 seed · 单次运行」；design §2 机制登记随附——EMA 影子滞后定量锚点 123 step/epoch ≈ 8.1 ep + 阴性预案、t>0.7 外推）
- [ ] C2 对比表：v4afmt-adaln-main-01 vs v4afmt-main-01（Q1 全维度/FC/指标3/逐通道 r/>40Hz 占比/参数量 +10,540）+ 单变量差异声明（norm LayerNorm→adaLN(t)，唯一变量）；判读规则 = design §3 预登记（方向性、不设阈值、与 fmt-f0 跨变体差异同框）；**若「无改善」且 best epoch ≲16 → 追加非 EMA last.pt 补充评估（补充 Run 登记，不替换正式口径）**
- [ ] C3 指标2 视觉图交用户人工判（figs/ 轨迹图，250 Hz 标秒）
- [ ] C4 trellis-check 全量检查
- [ ] C5 spec 回填（3.3）：reproducibility 钩子链追加 `cond_adaln`；context.md 技术路线表登记 adaln 变体（研究结论采纳后回填 registry）；evaluation-calibers 视新口径；progress.md 当前状态同步
- [ ] C6 journal + 归档（/finish-work 惯例）

## 验证命令

```bash
# 单测（A4/A5/B3 全绿为门禁）
python3 models/gpfm-mamba-v4a-fmt-adaln/tests/test_v4afmt_adaln.py
python3 models/gpfm-mamba-v1-fmt/tests/test_fmt_hooks.py   # 回归（零漂移面）
python3 models/gpfm-mamba-v4a-fmt/tests/test_v4afmt.py     # 回归（v4a-fmt pin）
# 冒烟与训练（经变体包装入口，前缀守卫生效）
python3 models/gpfm-mamba-v4a-fmt-adaln/train.py --smoke --run-id v4afmt-adaln-smoke-01
# 评估（GC 阻塞延续）
python3 models/gpfm-mamba-v4a-fmt-adaln/evaluate.py --run-id v4afmt-adaln-main-01 --skip-gc-only
```

## 风险文件与回滚点

| 风险 | 触发信号 | 回滚 |
|---|---|---|
| 共享钩子改坏旧变体 | A4 零漂移断言红、旧 config_hash pin 漂移 | revert v1 三文件钩子（逐处），重设计 |
| adaLN 数值/结构错 | 恒等初始化等价断言红 / state_dict 键缺 | 对照 design §1.1 键表修实现 |
| dataloader 契约不符 | B4 smoke 形状/窗数与 v4a-fmt 记录不符 | 停训练，修接入层（禁改数据本体） |
| 主 Run 中断 | last.pt 存在 | --resume 沿用同 run 目录续跑 |
| A 案系统性失稳 | loss 尖峰 / val 崩（登记现象） | 回炉讨论 B 案（须用户拍板 + 验收改写） |

## 提交点约定（对齐 fmt-f0 惯例）

A+B1-B3（代码+单测）→ commit ①；smoke/bench（含 splits/）→ commit ②；主 Run+评估 → commit ③；判读+spec 回填 → commit ④；归档+journal → ⑤
