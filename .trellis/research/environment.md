# 环境契约（事实档案 · 写/跑代码前必读）

> **常驻规则**：任何未记录在案的环境场景，必须先问用户确认——不得猜测 env、不得自行安装依赖。环境变化（新 env / 新机器 / 新依赖）须当次回填本文件。
> **初始化**：2026-10-10 由 AI 按用户指示直接感知当前云服务器配置填写。

## 本地（= 当前云服务器；用户本地经 SSH 连接本机访问 `/root/npi`）

- 系统：Ubuntu 22.04.5 LTS（内核 5.15.0-117），seetacloud（AutoDL 系）实例
- 硬件：128 核 CPU / 503 GB 内存 / 1× NVIDIA GeForce RTX 4080 SUPER（32 GB 显存，驱动 595.71.05）
- 磁盘：系统盘 `/` 30 GB（余 ~29 GB）；数据盘 `/root/autodl-tmp` 50 GB（RAID，余 ~50 GB）——数据集与 outputs 等大文件应落数据盘
- Python：miniconda3（`/root/miniconda3`），**仅 base 环境**，Python 3.12.3
- 深度学习：PyTorch 2.8.0+cu128（base 已装；`torch.cuda.is_available()`=True；CUDA 12.8 运行时随 torch 自带，系统无独立 nvcc toolkit）
- 工具：git / nohup / rsync 可用；**tmux 未安装**（需要时先问用户）
- 数据落点：`/root/autodl-tmp/`（2026-10-10 用户指定；原 `/root/npi/dataset/` 占位目录已删除，具体子目录结构待数据集选定后回填）

## 远程算力

| 别名 | 地址 | 环境 | 规格 | 备注 |
|---|---|---|---|---|
| （无） | | | | 当前服务器即唯一已知算力；如新增远程机器由用户回填本表 |

## 派发惯例

- 长跑后台化：nohup（tmux 不可用）
- 日志落点：未定（建议随任务落在 `outputs/<variant>/logs/`，待用户确认）
- 版本快照：`/root/npi` 已于 2026-10-10 `git init`（分支 `main`）；`.gitignore` 默认排除 outputs/、checkpoint 与 Python 缓存；git 身份已配置在本仓库（`--local`：weanqin / 1984536775@qq.com）；远端 `github.com/weanqin/PEFC`（HTTPS，2026-10-10 首推 main）。推送凭据明文落盘在本机 `~/.git-credentials`（credential.helper store，仓库内无令牌，权限 600）；令牌更换时更新该文件即可
- 代码/数据同步：代码可经远端仓库 `github.com/weanqin/PEFC`（git push/pull）同步；数据仍在 `/root/autodl-tmp`、不经 git，跨机数据同步方式未定
