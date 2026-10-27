# consent — 10-27-ec-histblock-symw-f0

## 1.4 同意记录（implementation consent gate）

- **时间**：2026-10-27
- **用户确认原话（同意链）**：
  1. 原始指令（本会话工作令，完整明确五要素）：「请使用 hist-block 噪声构造范式，在
     fold0 上对全部 8 个样例运行扰动归因 EC 分析；代理脑使用 10-25-md-lf-anti0.2-
     gnnexit。请严格保留上述范式、fold 划分、样例数量、分析类型和代理脑标识，并按
     该配置完成分析。」
  2. 先例链：本会话同日 hist-block 变体任务（10-27-ec-histblock-noise）经
     AskUserQuestion 明确同意创建并实施后全流程交付归档；此前 EC 三跑/可视化均以
     直接工作令 + consent 先例链推进（10-25-symw-ec-f12 consent.md 登记模式）。
- **确认对象**：prd.md（轻量 PRD-only：执行型任务，方法零改动、口径逐字钉死）。
- **流程凭据**：implement.jsonl / check.jsonl 已配；review = 会话内代理脑身份对拍
  （全库唯一匹配侦察在档）+ 五要素逐项落 prd，trellis-review dispatch 按轻量先例
  跳过（口径 100% 复用已验证 hist-block 开关 + symw EC 先例）。
- **关键口径钉死清单（执行不得偏离）**：
  - 命令面 = `--mode main --symw --fold 0 --noise-mode hist-block --subjects 8
    --task-label 10-27-ec-histblock-symw-f0`，其余默认（q=0.5/σ_ref=0.3/
    log[0.01,0.48]×24/seed 42/30 trial 全量/hist-block 全共享默认）；
  - ckpt = run_symw_fold0（用户标识 10-25-md-lf-anti0.2-gnnexit 的全库唯一匹配，
    对拍见 prd），--symw 守卫与 norm SHA fold0 键必须原样通过，断言失败即停不绕；
  - 方法代码零改动（发现 hist-block 实现缺陷 → 停下报告，不顺手改）；
  - 产物只落 outputs/10-27-ec-histblock-symw-f0/；
  - hist-block 与 iid 各跑的数值差异含噪声谱形因素，判读按 prd AC4 描述性口径。
