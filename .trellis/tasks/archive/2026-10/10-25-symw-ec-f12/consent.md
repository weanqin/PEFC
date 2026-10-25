# consent — 10-25-symw-ec-f12

## 1.4 同意记录（implementation consent gate）

- **时间**：2026-10-25（方案提交同日）
- **用户确认原话（同意链）**：
  1. 原始指令（本会话首条消息，完整明确工作令）：「在新训练好的 gnnexit-symw 模型的 fold1
     和 fold2 数据划分上，使用与 fold0 完全相同的扰动归因（perturbation attribution）方法
     计算 EC 指标的归因结果」+ 三项规格（①与 fold0 一致的扰动策略/步长/指标/流程可直接
     对比；②加载 fold1/fold2 训练权重；③输出各 fold 归因结果 + 配置注明便于与 fold0 汇总）。
  2. 本会话前序两次同构工作令（EC 三跑、可视化）均经任务制执行并交付；第二次起用户以
     直接工作令 + 既有 consent 先例链推进（本条延续该模式，处置透明）。
- **确认对象**：prd.md + implement.md（轻量：design 不适用——口径 = fold0 逐字复用，
  适配点钉到行级）。
- **流程凭据**：implement.jsonl 6 条 / check.jsonl 4 条已配；1.4.1 review dispatch 按
  轻量规则跳过（设计 100% 复用三先例；用户三项规格即验收基准）。
- **关键口径钉死清单（执行不得偏离）**：
  - 口径 = fold0（10-25-symw-ec-unbiased）逐字相同：无偏 μ=0 σ·ξ、q=0.5、σ_ref=0.3、
    log[0.01,0.48]×24、各折 test 8 例 × 30 trial、seed 42、bf16、τ 分位含对角、
    conn 交付 v4；
  - 代理脑 = f12 fold1 best 0.8264@E116 / fold2 best 0.8433@E113（唯一 ckpt 路径见
    prd 关键事实）；
  - **归因算法逻辑零改动**：适配仅 norm SHA fold 键控（值 = 训练侧 configs 登记值）+
    守卫扩展（f12 run_dir + fold 前缀断言）+ payload 元数据分支；计算路径逐行不动，
    旗标关 fold0 逐位对拍兜底；
  - 跨折（0/1/2）并置为描述性（各折 θ 不同代理脑实例），不作复现性判读；EC 结果
    provisional 随 θ 拍板联动。
