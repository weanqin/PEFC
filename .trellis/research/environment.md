# 环境契约（事实档案 · 写/跑代码前必读）

> **常驻规则**：任何未记录在案的环境场景，必须先问用户确认——不得猜测 env、不得自行安装依赖。环境变化（新 env / 新机器 / 新依赖）须当次回填本文件。
> **初始化**：2026-10-10 由 AI 按用户指示直接感知当前云服务器配置填写。

## 本地（= 当前云服务器；用户本地经 SSH 连接本机访问 `/root/npi`）

- 系统：Ubuntu 22.04.5 LTS（内核 5.15.0-117），seetacloud（AutoDL 系）实例
- 硬件：128 核 CPU / 503 GB 内存 / 1× NVIDIA GeForce RTX 4080 SUPER（32 GB 显存，驱动 595.71.05）
- 磁盘：系统盘 `/` 30 GB（余 ~29 GB）；数据盘 `/root/autodl-tmp` 50 GB（RAID，余 ~50 GB）——数据集与 outputs 等大文件应落数据盘
- Python：miniconda3（`/root/miniconda3`），**仅 base 环境**，Python 3.12.3
- 深度学习：PyTorch 2.8.0+cu128（base 已装；`torch.cuda.is_available()`=True；CUDA 12.8 运行时随 torch 自带；自带 triton 3.4.0；einops 0.8.2 已装）。**勘误（2026-10-12）**：`/usr/local/cuda-12.8/bin/nvcc`（V12.8.93）实际存在（`/usr/local/cuda` 软链指向它），只是不在 PATH——源码编译 CUDA 扩展时需 `export CUDA_HOME=/usr/local/cuda` 与 `PATH=$CUDA_HOME/bin:$PATH`
- Mamba 栈（2026-10-12 受用户委派安装完成，GPU 前向/反向实测通过）：**mamba_ssm 2.3.1 + causal_conv1d 1.7.0**（均为 cu12torch2.8cxx11abiTRUE-cp312 预编译 wheel，与 torch 2.8 的 cxx11abi=TRUE 匹配）；附带安装 transformers 5.19.0 / ninja 1.13.2 / huggingface-hub 2.2.0。**版本锁定事实**：mamba-ssm ≥2.3.2.post1 要求 `triton>=3.5.0` 且引入 tilelang/quack-kernels 重依赖，与 torch 2.8 钉死的 `triton==3.4.0` 硬冲突——**torch 升级到 2.9+ 之前不得升级 mamba-ssm**；Mamba2 的 Triton 路径运行时必需 causal_conv1d（未装时报 `'NoneType' object is not callable`）。wheel 存档：`/root/autodl-tmp/wheels/`（mamba_ssm-2.3.1 533 MB、causal_conv1d-1.7.0 196 MB，可复装）
- 科学计算（2026-10-13 更新）：numpy 2.3.2 / matplotlib 3.10.5 已装；**pymupdf 1.28.2（2026-10-12 用户要求安装，建 PDF+文本抽取实测通过）**；**scipy 1.18.1 / scikit-learn 1.9.1 / mne 1.13.2（2026-10-13 用户批准安装，pip 实测通过，含 KSG-TE cKDTree Chebyshev 基准）**；**mne-connectivity 0.9.0 / pandas 3.0.6（2026-10-13 用户批准安装，baseline-ec-v1 用）**；**pytest 9.1.1（2026-10-18 用户要求安装，实测通过；随装 iniconfig 2.3.1，pluggy 1.0.0→1.7.0 升级）**；**statsmodels / gpytorch 仍未安装；idtxl 不装（TE 全系取消）**；既有评估代码保持 numpy-only 口径不动，新 baseline-EC 管线可用新依赖
- **有效算力事实（2026-10-13 实测登记）**：cgroup `cpu.max=1600000/100000` → **有效 16 核**（`nproc` 显示 128 是假象；并行吞吐实测 16 进程满效、32 进程崩至 24%）；numpy/OpenBLAS 默认按 128 可见核起线程会超订打满整机 CPU——**所有入口脚本须在 numpy 导入前设 BLAS 线程上限**（baseline-ec-v1 的 `_blas_guard.py` 惯例，限 8 线程，setdefault 可覆盖）
- 工具：git / nohup / rsync 可用；**tmux 未安装**（需要时先问用户）；**dos2unix 未装**（行尾统一用 GNU sed：`sed -i 's/\r$//'`，2026-10-18 latent-flow 移植惯例）
- 数据落点：`/root/autodl-tmp/`（2026-10-10 用户指定；原 `/root/npi/dataset/` 占位目录已删除）。**outputs 根（2026-10-12 用户指定）：`/root/npi/outputs/<任务名>/`**——按任务名建独立目录、落仓库内但被 `.gitignore` 排除（轻量产物：配置/日志/图/小 checkpoint）；数据集等大文件仍一律在 `/root/autodl-tmp`
- **latent_flow 源库与移植（2026-10-18 登记）**：先前研究代码库存档 `/root/autodl-tmp/latent_flow.7z`（5.4 GB）+ 解压副本 `/root/autodl-tmp/latent_flow/`（git 锚点 `9efe1e3`，**全程只读**）+ 备份 `/root/autodl-tmp/latent_flow.bundle`（1.1 GB，--all，verify 通过）；旧 θ\* checkpoints 558 MB 原位不随移。移植产物在仓库 `models/latent-flow-v1/`（study1 新路径，详见该目录 README-port.md）。**勘误（2026-10-18）**：MODMA resting 数据真实文件名 `modda`→`modma_resting_process.zip`（1.07 GB，PK 头有效可读；先前 ENOENT 系文件名误引），已解压 `/root/autodl-tmp/modma/{HC 29, MDD 24}`（.mat，键 `data`、(128,~42501) float32）——MODMA 线数据在位
- 依赖协调（2026-10-12）：mamba-ssm 安装委派**已完成**（见上方「Mamba 栈」条目，含版本锁定事实与 wheel 存档位置），任务侧 Mamba 相关运行步骤可解除挂起
- 网络与装包惯例（2026-10-12 实测）：GitHub release 资产直连会超时，需 `source /etc/network_turbo`（AutoDL 学术加速，仅限 GitHub/HuggingFace）+ `curl -C -` 断点续传；pip 大包用清华镜像 `-i https://pypi.tuna.tsinghua.edu.cn/simple`（快），默认阿里云镜像对大文件慢且连接易挂死；mamba-ssm 裸 `pip install` 会因构建隔离缺 torch 报 `ModuleNotFoundError`，应直接装预编译 wheel（`pip install --no-deps <wheel>`）

## 远程算力

| 别名 | 地址 | 环境 | 规格 | 备注 |
|---|---|---|---|---|
| （无） | | | | 当前服务器即唯一已知算力；如新增远程机器由用户回填本表 |

## 派发惯例

- 长跑后台化：nohup（tmux 不可用）
- 日志落点：随任务落 `outputs/<任务名>/run_<id>/logs/`（2026-10-12 随 outputs 根一并确定）
- 版本快照：`/root/npi` 已于 2026-10-10 `git init`（分支 `main`）；`.gitignore` 默认排除 outputs/、checkpoint 与 Python 缓存；git 身份已配置在本仓库（`--local`：weanqin / 1984536775@qq.com）；远端 `github.com/weanqin/PEFC`（HTTPS，2026-10-10 首推 main）。推送凭据明文落盘在本机 `~/.git-credentials`（credential.helper store，仓库内无令牌，权限 600）；令牌更换时更新该文件即可
- 代码/数据同步：代码可经远端仓库 `github.com/weanqin/PEFC`（git push/pull）同步；数据仍在 `/root/autodl-tmp`、不经 git，跨机数据同步方式未定
