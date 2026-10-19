# implement — 10-19-latent-flow-edra-fmt-train

> 执行清单。前置：`task.py start`（1.4 用户批准后）。路径隔离：本任务新增代码仅 `models/latent-flow-v1/`（包装层 + 三处包内 additive）；产物仅 `outputs/10-19-latent-flow-edra-fmt-train/run_01/`。

## 阶段 S1 — 数据 shim（包内 additive）

- [ ] S1.1 `latent_flow/config.py`：追加 `EDRA_FMT_DATA_ROOT`（env 同名）与 `EDRA_FMT_GEOMETRY_PATH` 常量
- [ ] S1.2 `latent_flow/data/datasets.py`：`_REGISTRY` 追加 `"edra_fmt"` Descriptor
- [ ] S1.3 `latent_flow/data/modma.py`：`_load_subject` 追加 `"npz"` 分支
- [ ] S1.4 冒烟验证（不训练）：`get_dataset("edra_fmt")` 解析、`list_hc/list_mdd` 24+24 例、单例 `MODMADataset(subject_files=[一例], dataset="edra_fmt", window_size=512, step=64, obs_size=384)` 窗数 == 930、数据形状 (384,62)/(512,62)
- [ ] S1.5 回归：全量 pytest 维持 67P/0F/17S（`cd latent_flow && OMP…=8 python3 -m pytest tests/ -q`）

**Gate G1**：S1.4/S1.5 全绿才进 S2。

## 阶段 S2 — geometry（仅 D1=mexhat 时执行；D1=dense 则跳过）

- [ ] S2.1 `build_edra_fmt_geometry.py`（包装层）：import 复用包内 MNE 解析逻辑，从 npz `channel_names` 生成 `geometry_edra_fmt.pt`；**落点随 D2 拍板**（数据根豁免案 / run_01 + env 覆盖案），wrapper 内 `load_geometry(path=…)` 显式传路径（`load_geometry` 的 dataset 分派不识别 "edra_fmt"，不得传 dataset key）
- [ ] S2.2 交叉验证（判定门）：生成坐标 vs `chan_coords_62.npy` 逐通道方向余弦 ≥ **0.99** 且欧氏距离差异登记；不达标 → **中止并问用户**（不得自行放行）

## 阶段 S3 — 训练入口（包装层）

- [ ] S3.1 `models/latent-flow-v1/train_edra_fmt.py`：按 design §4（BLAS 守卫 / manifest 断言 / loaders / 模型 384/128 / 复用 `_run_stage1/_run_stage2` / 配置 JSON 落盘 / history：stage1 由 wrapper 自写、stage2 包内副本复制）
- [ ] S3.2 `.gitignore`（变体级）追加 `latent_flow/logs/`；README-port.md 补「edra_fmt 训练入口 + logs 说明」两行
- [ ] S3.3 冒烟（两步，确保 stage2 冒烟真正走到 ckpt 加载与冻结路径）：
  - stage1 冒烟：`--stage 1 --epochs 1 --patience 1 --limit-subjects 2 --save-path …/run_01/checkpoints/smoke_tmp_vae.pt`（patience≥1 使 ckpt 落盘；断言全过；RAM/耗时观察登记）
  - stage2 冒烟：`--stage 2 --epochs 1 --patience 1 --limit-subjects 2 --vae-ckpt …/run_01/checkpoints/smoke_tmp_vae.pt --save-path …/run_01/checkpoints/smoke_tmp_flow.pt`（验证非 flow 键零失配断言 + VAE 冻结 + trainable 参数打印；D1=mexhat 时同步验证 geometry 加载；save-path 防冒烟权重写入正式 ckpt 名）
  - 冒烟后删除 `smoke_tmp_vae.pt` 与 `smoke_tmp_flow.pt`（仍在 run_01 隔离区内；知情项：stage2 冒烟 patience>0 会在包内 logs/edra_fmt/ 写冒烟 history，属已声明写入区、正式跑同 stem 覆盖）
- [ ] S3.4 冒烟后核对：无任何文件写入包内 checkpoints/、run_01 外零产物（`git status` + find 核对）

**Gate G2**：冒烟双阶段退出码 0 + 产物隔离核对通过才进正式训练。

## 阶段 S4 — 正式训练（nohup 后台）

- [ ] S4.1 Stage1：`nohup python3 train_edra_fmt.py --stage 1 > …/run_01/logs/stage1.log 2>&1 &`；监控至完成（早停或 50 epoch）；登记 best epoch / best val loss / 总耗时
- [ ] S4.2 Stage2：`--stage 2 --vae-ckpt …/run_01/checkpoints/fold0_edra_fmt_vae_best.pt`（D1 变体 flag）；同样监控登记；确认日志三行证据（加载 ckpt 路径 / 非 flow 键零失配 / trainable 参数数 = flow only）
- [ ] S4.3 每阶段完成后：history JSON 复制核对、config JSON 与实际参数一致性抽查

## 阶段 S5 — 核对与收尾

- [ ] S5.1 **R5 一致性核对**（逐条登记进 experiment.md）：① 阶段顺序（stage2 日志中 ckpt 路径指向 stage1 产物文件）；② 模型类型（stage2 = 潜空间 CFM：t~U 线性插值 + 速度场 + Euler 采样、VAE 冻结）；③ 无 gpfm：`grep -rn "gpfm" models/latent-flow-v1/` 对新增文件零命中（包本体沿用移植期零命中事实）
- [ ] S5.2 experiment.md Run 块四区补全（两阶段配置/划分引用/窗口参数/ckpt 与日志路径/耗时/偏差）
- [ ] S5.3 AC1–AC7 自检清单逐项打勾
- [ ] S5.4 trellis-check 全范围复核（派发，含 AC7 pytest 回归复跑）
- [ ] S5.5 3.3 spec update（environment.md 登记新入口/geometry 若生成；context.md 若有状态变化）→ 3.4 commit → wrap-up

## 验证命令速查

```bash
# G1 回归
cd /root/npi/models/latent-flow-v1/latent_flow && \
  OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 \
  MODMA_DATA_ROOT=/root/autodl-tmp/modma python3 -m pytest tests/ -q   # 期望 67P/0F/17S
# 冒烟（两步，见 S3.3；stage2 按拍板的 D1 变体加 flag）
python3 /root/npi/models/latent-flow-v1/train_edra_fmt.py --stage 1 --epochs 1 --patience 1 \
  --limit-subjects 2 --save-path …/run_01/checkpoints/smoke_tmp_vae.pt
python3 /root/npi/models/latent-flow-v1/train_edra_fmt.py --stage 2 --epochs 1 --patience 1 \
  --limit-subjects 2 --vae-ckpt …/run_01/checkpoints/smoke_tmp_vae.pt \
  --save-path …/run_01/checkpoints/smoke_tmp_flow.pt
# 正式（D1=mexhat 示例 stage2 flag：--graph-mode --graph-kernel mexhat）
```

## 回滚点

- S1 后回滚：`git checkout -- models/latent-flow-v1/latent_flow/`（三文件 additive 撤销）
- S3 后回滚：删 `train_edra_fmt.py`/`build_edra_fmt_geometry.py` + checkout 三文件与 .gitignore
- S4 产物回滚：删 `outputs/10-19-latent-flow-edra-fmt-train/` 整目录（无既有数据可被覆盖）
