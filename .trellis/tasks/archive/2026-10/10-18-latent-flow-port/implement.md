# Implement：latent_flow 移植执行计划（10-18-latent-flow-port）

> 前置：design.md 经评审修订、§9 开放决策已拍板后 `task.py start` 方可执行。源库 `/root/autodl-tmp/latent_flow` 全程只读。
> 基线锚点：源库 HEAD `9efe1e3` + 工作树内容；`data/` 加载器经 `git show HEAD:data/<f>` 物化恢复。
> 评审修订已并入：黑名单式裁剪语义、冒烟改道 shape_check、sed 替代 dos2unix、rsync 源路径写死、G1 双判据、run.py 参数契约定死。

## 阶段 S0：前置（用户拍板项落地）

- [x] S0.1 源库 bundle 备份：`git -C /root/autodl-tmp/latent_flow bundle create /root/autodl-tmp/latent_flow.bundle --all`（含被裁线的完整历史；验证 `git bundle verify` 通过）
- [x] S0.2 MODMA zip 可读性复测（用户提示数据在位）：`head -c 4 /root/autodl-tmp/modda_resting_process.zip`（PK 头）+ `unzip -l` 抽查——可读则解压 `HC/ MDD/` 到 `/root/autodl-tmp/modma/`（数据盘），README-port 登记 `MODMA_DATA_ROOT=/root/autodl-tmp/modma`，S1.6 预期口径相应上调（`test_subject_classification.py` 4 失败转通过）；仍 ENOENT 则按边界登记不动
- [x] S0.3（可选随 S0.2）geometry.pt 检查：若 zip/源库有 `geometry.pt` 或可由脚本生成则挂到 MODMA_DATA_ROOT（`test_graph_utils.py` 2 skip 转通过）；无则保持 skip 如实登记

## 阶段 S1：落位与恢复（回滚点：`rm -rf models/latent-flow-v1`）

- [x] S1.1 rsync 落位（**源路径写死防双层嵌套——源库是仓库根=包根布局**）：
  ```bash
  rsync -a /root/autodl-tmp/latent_flow/ /root/npi/models/latent-flow-v1/latent_flow/ \
    --exclude={.git,.trellis,.zcode,.claude,.agents,.codex,archive,checkpoints,results,outputs,logs,experiments,.obsidian,__pycache__,.pytest_cache} \
    --exclude={'AGENTS.md','.gitignore','.gitattributes'} --exclude='*.pdf'
  ```
  （README.md 保留在包内——env var 文档源）
- [x] S1.2 物化恢复 `data/` 9 个加载器（清单见 design §2 内联）：`git -C /root/autodl-tmp/latent_flow show HEAD:data/<f> > /root/npi/models/latent-flow-v1/latent_flow/data/<f>`（`__init__.py` 空 blob 属正常）
- [x] S1.3 判决笔记独立拷贝：`mkdir -p /root/npi/models/latent-flow-v1/latent_flow/docs/experiments-notes && cp /root/autodl-tmp/latent_flow/experiments/*.md $_`（20 份，含中文名 `实验二.md`；若用户裁定不进 git 则跳过并在 README-port 登记）
- [x] S1.4 行尾统一 LF + 落 `.gitattributes`（`*.py text eol=lf`，md/csv/yaml 同）+ 变体级 `.gitignore`（`results/`、`.pytest_cache/`）：
  ```bash
  find /root/npi/models/latent-flow-v1 -type f \( -name '*.py' -o -name '*.md' -o -name '*.csv' -o -name '*.yaml' \) -exec sed -i 's/\r$//' {} +
  ```
  （GNU sed 在位；dos2unix 未装不安装；`git show` 恢复文件天然 LF，此步对它们为 no-op）
- [x] S1.4b 恢复盘逐字节核验：对 9 个文件 `git -C /root/autodl-tmp/latent_flow show HEAD:data/<f> | sed 's/\r$//' | diff - <dest>`，全部零 diff 才继续
- [x] S1.5 若拍板裁剪 A（黑名单式）：**先冻结 keep/delete 两列逐文件清单**（对 scripts/ 与 tests/ 逐文件判归属：de/dysib/fm_gate/dcgnn 线删，OC 簇及共享依赖留——`viz/probe.py`、`run_oc_connectivity_{global_tau,relative}.py`、`build_edra_geometry.py`、`coord_audit.py`、`models/graph_utils.py`、`norms/` 必留；`models/{de_vae,dysib,dcgnn_classifier,fm_model}.py`、`analysis/{de,fm_gate}/`、`training/train_dysib*.py` 必删），再执行删除，清单存 `experiment.md`；删除后以实际删除集与冻结 delete 列 `comm`/`diff` 核对零差（证「没多留」）
- [x] S1.5b 冻结 G1 预期口径：对保留的每个 tests/*.py 逐文件 `python3 -m pytest tests/<f> -q` 记 per-file 计数（全量基线 293 passed / 4 failed〔`test_subject_classification.py`，MODMA 数据缺〕/ 5 skipped〔`test_coord_audit.py` 3 项 TODO + `test_graph_utils.py` 2 项 geometry.pt 缺〕；裁剪 A 下 4 失败保留〔data/ 全留〕、3 个 coord_audit skip 随其归属定去留）；per-file 总和即 G1 预期值
- [x] S1.6 验证（G1 门禁，双判据）：
  ```bash
  cd /root/npi/models/latent-flow-v1/latent_flow
  OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 python3 -m pytest tests/ -q   # 对齐 S1.5b 冻结口径
  cd /root/npi/models/latent-flow-v1   # import 冒烟须从包父目录跑（包内跑会因 sys.path 缺父目录 100% 假失败）
  for m in latent_flow/scripts/*.py; do PYTHONPATH=/root/npi/models/latent-flow-v1 python3 -c "import latent_flow.scripts.$(basename $m .py)" || echo "FAIL $m"; done  # 零失败
  grep -rIl $'\r' /root/npi/models/latent-flow-v1 | wc -l   # = 0（行尾硬检查）
  ```
- **门禁 G1**：三项全过才进 S2；失败逐条归因（环境差异→登记处置；闭包缺件→回 S1.5 补齐）

## 阶段 S2：包装层（回滚点：删除新增包装文件）

- [x] S2.1 `run.py`：顶部 BLAS 守卫（numpy 导入前 setdefault OMP/MKL/OPENBLAS ≤8）→ `sys.path.insert` 变体目录 → 转译执行 `python -m latent_flow.scripts.<name>`。**参数契约定死：首个位置参数=脚本名，其余参数原样透传，无 `--` 分隔**；自检打印解析后的 OMP_NUM_THREADS；对有 `--output-dir` 类 flag 的脚本支持 `--output-root /root/npi/outputs/latent-flow-port/` 透传；baseline 子命令默认 `--n_jobs 8`
- [x] S2.2 `README-port.md`：两种入口（run.py 推荐 + `-m` 直跑 env 前置写法）、env var 表（MODMA/EDRA_DATA_ROOT 本机指向）、baseline 相对路径产物警告（勿裸跑，产物会落 cwd 的 results/）、旧 θ\* checkpoints 原位清单引用、裁剪口径与源锚点 `9efe1e3`、Study I/II 边界声明（design §5 摘要）、FOUNDATION 定量基线数字
- [x] S2.3 验证（G2 门禁）：`python3 /root/npi/models/latent-flow-v1/run.py shape_check_oc_latent_flow` 退出码 0 且打印 OMP_NUM_THREADS ≤8

## 阶段 S3：运行验证

- [x] S3.1 GPU 冒烟 = S2.3 的 shape_check（合成数据 `torch.randn` 前向，秒级，承担 AC3）。**train `--smoke` 登记为数据阻塞项**（`--dataset` 仅 modma/edra，本机均不可加载；待 MODMA zip 修复或 EDRA shim 后补跑，写入 experiment.md 与 README-port）
- [x] S3.2 git 干净性：
  ```bash
  git -C /root/npi status --short | grep -v 'latent-flow-v1\|\.trellis'   # 应为空
  find /root/npi/models/latent-flow-v1 -name '*.pt' | wc -l                # = 0
  find /root/npi/models/latent-flow-v1 -name .pytest_cache -exec rm -rf {} +  # 清理防入库
  ```
- [x] S3.3 experiment.md 登记执行 Run 块（目的/配置含精确命令与环境/结果数值/产物路径与结论；含 G1 冻结清单与口径）

## 阶段 S4：质量与收尾

- [x] S4.1 派发 `trellis-check`（全范围：spec 合规/路径隔离/一致性/评审条件落实复核）；整改至通过
- [x] S4.2 spec update（3.3）：context.md 技术路线表回填 latent-flow 条（路由按拍板结果）；environment.md 登记移植事实、dos2unix 未装（sed 替代惯例）、MODMA zip 异常
- [x] S4.3 commit（信息注明源锚点 `9efe1e3`、7z 来源、载荷与裁剪口径；**不并入**既有未提交的 ` M .trellis/research/environment.md` 前会话遗留）；journal 记录；任务归档

## 验证命令速查

```bash
cd /root/npi/models/latent-flow-v1/latent_flow && OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 python3 -m pytest tests/ -q
python3 /root/npi/models/latent-flow-v1/run.py shape_check_oc_latent_flow
git -C /root/npi status --short | grep -v 'latent-flow-v1\|\.trellis'      # 应为空
```

## 回滚总案

任一阶段失败且不可修：`rm -rf /root/npi/models/latent-flow-v1 /root/npi/outputs/latent-flow-port` + revert 相关提交；experiment.md 以「中止登记」Run 块收尾（append-only 不删除）；已做的 3.3 编辑（context.md/environment.md）手工还原；源库与既有变体不受影响。
