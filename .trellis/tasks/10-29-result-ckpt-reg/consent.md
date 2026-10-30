# consent — 10-29-result-ckpt-reg

- **时间**：2026-10-30（本会话）
- **同意凭据**：用户原始指令点名 result 目录（result-dir-policy 写入许可情形 1）：
  > "筛选出所有完整跑完 5 折训练以及 5 折交叉评估（5-fold EC）的模型，收集它们对应的 checkpoint（ckpt）文件，并将这些 checkpoint 注册（登记）到 result 中。"
  
  加 AskUserQuestion 两问拍板（同轮明确选择）：
  > Q1「是否创建轻量任务执行注册」→ **「创建任务并注册（推荐）」**（选项描述含：立轻量任务 → PRD → briefing 征得同意后 start → 10 ckpt 复制入 result/ + 清单登记 sha256/来源任务/口径摘要）
  > Q2「modma·0705 fold0 EC 在途如何处理」→ **「本次仅注册 mild（推荐）」**
- **briefing**（已在 Q1 选项与筛选报告中向用户呈现）：
  - 改动点 = `result/checkpoints/mild/latent_flow_symw/fold{0..4}/` 新增 10 个 ckpt 副本（约 130MB）+ 新增 `result/checkpoints/MANIFEST.md` + `result/README.md` 当前状态段追加注册记录；
  - 关键逻辑 = `cp -p` 只读复制 + `cmp` 逐位校验 + sha256 随档登记（fold0 VAE 借用 w500 基座作注记）；
  - 潜在风险 = 无（纯增量，不动 outputs/ 原件，不干扰运行中 modma fold0 训练进程）；
  - 验证 = AC1 cmp 逐位一致 / AC2 sha256 复核 / AC4 原件不动。
- **结论**：review（筛选核查全量在案）→ briefing → 用户明确同意，三件齐备，可 `task.py start`。

## 追加同意凭据（同任务第二轮，2026-10-30）

- **追加指令 1**（任务暂不收尾 + EC 注册范围）：
  > "暂不收尾任务，将mild latent-flow-symw-anti0.2-gnn-exit5折上的EC_pert结果、逐被试EC_pert与GC-EC相关性对比、组平均EC_pert与GC-EC相关性（注册入 result）"
- **追加指令 2**（版本口径收紧）：
  > "fold0/1/2仅将recompute_v4版本的结果注册result"
- **执行解读**：任务保持 in_progress 不归档不提交；第二轮注册对象 = 同款双全模型五折 EC 三类产物；fold0/1/2 版本 = recompute_v4 目录产物（v4 公式），其 v1 原版/curves/summary 与混编 crossfold_summary 不注册。写入许可仍为情形 1（指令语境延续首轮 result 注册令）。

## 追加同意凭据（同任务第三轮，2026-10-30）

- **追加指令 3**（损失数据注册）：
  > "将mild-lf-symw-anti0.2-gnn-exit变体5fold上的训练时的损失数据文件和测试集上的损失数据文件注册到result当中"
- **执行解读**：延续既有任务（未新立任务，`task.py start` 恢复指针）；注册物 = 五折训练逐 epoch 损失历史 ×10。经代码走读确认：测试集（8 例测试折）在训练管线中无损失数据文件（history `val` = 4 例验证划分；测试侧仅 3-batch smoke 控制台行）——已抄录进 MANIFEST 并在交付报告中说明；best-ckpt 全量测试折损失现算属新实验，未擅自执行，待令。

## 追加同意凭据（同任务第四轮，2026-10-30）

- **追加指令 4**（modma 补注册）：
  > "10-29-modma-ec-te-corr 实验已经完成；MODMA 数据集 fold0–4 的模型训练已完成。请整理并汇总 modma stage-1 和 stage-2 在 fold0–4 上的以下内容注册到result当中：1. 模型权重文件（各 fold ckpt）；2. 训练过程曲线（train/val/test 损失曲线文件）；3. EC_pert 连接矩阵结果（sigma\*、tau、逐被试/组平均 EC×GC-EC 与 TE-EC 相关性）。按 stage-1/stage-2、fold0–4 分层组织，列出文件路径清单。"
- **执行解读**：即首轮登记的「modma·0705 待 fold0 EC 完成后补注册」挂起项，前提（te-corr 完成归档、fold0 训练+EC 落盘）核验成立后执行；写入许可情形 1（点名 result）。stage1 ckpt 为仓库外旧库预训件，随档复制入库并注记来源；stage1 无训练曲线、test 无损失数据文件、fold0 EC×GC 相关无文件——三处缺口如实登记不虚构。
