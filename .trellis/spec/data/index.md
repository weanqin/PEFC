# 数据契约包（Data Specs）

**作用**：每个数据集一份「实测事实契约」——来源、口径、字段、产物位置、已知坑。下游一切消费（训练 / 评估 / 统计）以契约为准；**契约未记录的事实不得凭印象使用**。

## 当前契约

| 数据集 | 契约文件 | 一句话说明 |
|---|---|---|
| edra | [edra.md](./edra.md) | HC/MDD 静息态 EEG，62 通道 500 Hz，受试者级 npz（HC 24 / MDD 22），时间窗 `[past=384, future=128]` |
| edra-fmt | [edra-fmt.md](./edra-fmt.md) | 新预处理范式（release 1–40 Hz@250 Hz 零改动转换），48 例 HC24/MDD24，同窗契约，930 窗/例，fs 耦合口径见契约 §3 |

新数据集接入：复制 [dataset-contract-template.md](./dataset-contract-template.md) 为 `<数据集名>.md`，据实填写并加入上表。契约只记**实测事实**（跑过验证 / 核对过原始文件），不确定的写「待验证 + 日期」。
