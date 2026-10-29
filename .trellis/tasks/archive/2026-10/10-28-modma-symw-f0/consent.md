# consent — 10-28-modma-symw-f0

## 同意记录（1.4 门）

- **时间**：2026-10-28（会话 62da5e76）
- **用户确认原话**：「同意」
- **针对的 briefing**（上一轮消息全文呈现，含改动点/关键逻辑/风险/路径与环境/验证方案）：
  1. 任务理解：mild 线 10-25-symw-gnn-exit 变体移植到 MODMA resting fold0 训练 → v4 conn EC → 与 10-27 现算 GC（AIC 口径）做逐被试 + 组平均相关。
  2. 变体口径：exit-GNN 出口流 + SymWAdam W 对称化 + decoder 联训（encoder/proj 冻结）+ anti-uniform λ=0.2；lr 1e-3 / 150 ep / batch 64 / patience 30 / seed 42。
  3. 数据：包内 MODMA 管线（500 窗/step250/obs375/fc125、逐被试 z-score、5 折 seed42、5 例 held-out val）；stage-1 复用旧库 fold0 VAE（只读）。
  4. EC：07-05 家规参数（q0.5/σ_ref0.3/grid log[0.01,0.3]×16/32 窗/seed42）+ v4 conn 公式。
  5. 改动点：新增变体层两文件 `train_modma_symw.py` / `ec_modma_symw_v4.py`（包体零改动）；产物只落 `outputs/10-28-modma-symw-f0/`。
  6. 风险与成本：GPU 约 2–3 小时（训练 ~90min + EC ~45min）；bf16 AMP 非逐位确定；新变体首训无对拍锚点，结果为新口径登记；描述性并置不构成代理脑资格判定。
  7. 验证方案：冒烟（2 例×2 epoch + EC mini）→ 正式训练 → EC 10 例（P(resp>τ)=0.500 逐例构造性校验）→ gccorr 双级读数 → experiment.md 记录归档。
- **门禁背景**：models 树写入被编辑门拦截（无 in_progress 任务），向用户呈现三选项（立任务 / TRELLIS_HOOKS=0 / 停止），用户选择立任务并回复「同意」。
