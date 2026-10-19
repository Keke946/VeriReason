# Day 5A：M4 正式 QLoRA SFT 预注册报告

## 一页结论

M4 的主实验冻结为 **full5364×1 epoch、671 updates**。它是独立成立的 E1 正式 baseline：
完整一轮训练后安全停止并发布 adapter。此前 `1852fc4` 的 2048×2 草案和 `278330f` 的全量
修订历史均保留；本次不把 1% loss 变化做成机械的第二轮开关。update512/671 的 dev256
teacher-forced loss 只形成透明的 decision packet；是否延长到 epoch2 必须由控制端看真实证据、
时间和费用后另行批准，并另冻 LR/scheduler/预算。

Day 5A 只做本地方案、代码和测试：没有云、GPU、模型/数据下载或 heldout 读取（0/0），M4
仍是 pre-cloud pending。

## P0：梯度累积语义审计修复

控制审计发现一个不能忽略的框架边界：冻结的 `transformers==5.14.1` 会看到
Qwen3 `forward(..., **kwargs)`，从而把 `model_accepts_loss_kwargs` 推断为 true。它的
`training_step` 在这个分支不会再除以当前 gradient accumulation steps。M4 的自定义
`compute_loss` 没有消费 `num_items_in_batch`；若不处理，普通八条窗口和末尾窗口的总梯度都可能
整体放大 8 倍。此前的纯代数 mock 只证明了 loss multiplier 的公式，不能证明真实 Trainer
走了哪一条分支。

修复在构造自定义 Trainer 后 fail-closed：先断言精确 5.14.1 推断确实为 true，再显式设为
false 并再次断言。于是 Trainer 负责按 `current_gradient_accumulation_steps=8` 缩放：正常窗口
得到八条均值；最后四条的 `2×` loss 加四个 `0×` 完整 forward 得到四条均值。实际 torch+
Transformers 回归测试使用带 `**kwargs` 的最小模型，运行真实 `Trainer.train()` 并比较参数更新：
八条窗口为 8 条均值、末尾窗口为 4 条均值，同时断言 8 次 forward、GAS=8、512/671 同类强制
save/log 证据读取。它锁定 wheel 版本为 5.14.1，并检查 `training_step` 源码包含
`model_accepts_loss_kwargs`、`num_items_in_batch` 与 `current_gradient_accumulation_steps`；云端
preflight 会把 wheel 版本、torch 版本和该源码的 SHA-256 写入 `trainer-contract.json`。

本机没有 pinned torch/Transformers wheel，所以本地 `m4-verify` 会明确打印
`unavailable_not_verified`，而不是把这条真实回归悄悄算作通过；云端 orchestrator 在读数据、
token preflight 和训练前以 required 模式执行相同检查，版本/源码/回归不满足即停止。这是保留的
云端 tiny preflight 门禁，不是开始正式 M4 训练的授权。

### Preflight closure：真实回归夹具路径修复

Day 5B 精确环境的两条真实 Trainer 回归都在 `required_update_observations(update=1)` 停止。
根因是测试夹具把 `TrainingArguments.output_dir` 设为 `tmp_path`，实际保存为
`tmp_path/checkpoint-1`，却把同一 `tmp_path` 作为正式 outer output 交给读取函数；正式代码的
合同是 outer `output/checkpoints/checkpoint-1`。因此读取函数必然找不到 checkpoint，独立于
loss/grad_norm/LR/elapsed 的值。这是测试夹具 bug，不是正式 `train_formal_adapter` 的路径 bug。
修复让夹具使用同一 `checkpoint_directory(outer_output)` helper，并新增测试断言正式训练也使用该
helper；fail-closed 的 log、elapsed、m4-contract 与 manifest 非空检查保持不变。精确环境仍须在
CPU 上重跑两条真实 Trainer 测试，才能确认所有运行时字段均已通过。

## Day 1–4 已经准备了什么

我们固定了 Qwen3-4B-Base 的精确 revision、单卡 RTX 4090 D 48GB 和 M2 的 governed BIRD
数据：59 个 train DB 的 5,364 条 eligible 样本、10 个 validation DB 的 1,209 条样本；500 条
heldout 隔离。M3 又用同一 prompt/masking/runtime 完成 8/8 tiny 垂直切片、两次更新、adapter
重载和公平同卷评测。它给出真实工程证据（峰值 12,278 MiB、8 microstep/6.101470 s），但不是
正式精度结论。

## 为什么从 2048×2 改为 5364×1

| 路线 | 唯一样本 | epoch | example-passes | updates | 解释性 |
|---|---:|---:|---:|---:|---|
| M4 主实验 | 5,364 | 1 | 5,364 | 671 | 全部 59 DB、无 selector 抽样偏差 |
| 旧草案 | 2,048 | 2 | 4,096 | 512 | 子集重复两遍、遗漏 3,316 条 |
| 1024 | — | — | — | — | 不再是 M4 路线；仅未来另立必要性说明 |

主实验只比旧草案多 `5364/4096-1=30.96%` 训练 passes，却看到每一条 eligible SQL 一次，避免
“哈希刚好抽到什么”影响研究结论。full1209 的评测成本和训练集是 2048 还是5364无关，不能用它
为抽样训练辩护。单条 microbatch 的显存形状也不会因数据行数增长；M3 没有显示 48GB 障碍。

## 一条 BIRD 样本怎样进入训练

样本先是 M2 manifest 中 split=train、operational eligible、非 heldout 的记录。真实 token
preflight 要求全 5,364 条都能在 2,048 tokens 内按 M3 规则准备，否则失败，不静默删行。程序
把数据库 schema（不含 sample values）和问题拼成 SQL-only prompt；gold SQL+EOS 才有 label，
system/user/padding 都是 -100。模型预测这些 answer tokens，loss 就是它与 gold SQL 的差距。

## epoch、batch、LoRA 与 GPU 中发生的事

epoch 是把训练集看一遍；batch=1 是每次放一条长 schema；gradient accumulation=8 是累计八次
梯度才让 optimizer 更新，等效 batch=8。最后 4 条不能丢：各自 loss 乘2，追加 4 个 loss=0 的
padding forward，正常除8后得到四条的平均，因此第671次更新真实且无 padding 梯度。

基座以 NF4 4-bit 保存、bf16 计算；LoRA rank16/alpha32/dropout0.05 只训练注意力和 MLP 投影的
小 adapter。paged AdamW 8-bit 节省 optimizer 显存；learning rate 1e-4 经过 21 步 warmup 后按
**一轮 671 steps** cosine 下降。gradient checkpointing 用计算换激活显存，`use_cache=false`。
checkpoint 是可恢复快照（adapter、optimizer、scheduler、RNG、Trainer state）。

## 两个 dev loss 是什么、又不是什么

固定 dev256 在 update512 和671 各做一次 teacher-forced answer-only token-weighted mean loss；
packet 同时记录训练 loss window、gradient norm、LR、NaN/Inf、耗时和 checkpoint hash，并计算
`relative_dev_loss_drop=(loss512-loss671)/loss512`。1% 是帮助后续人阅读曲线的 advisory reference，
不是自动 gate；无论它高低，Day 5A 程序都在671停止。它不做 early stopping、不选 checkpoint、
不改学习率，也不生成 SQL，因此便宜且与正式 generation 分开。

## Base 与 SFT 如何公平比较

预注册 dev256、互斥 final953 及其并集 full1209 的 IDs/hash。E0 Base、E1 SFT 各对 full1209
**仅生成一次**，共用同一有序 ID、prompt/token hash、greedy decoding 和 M2 evaluator。full1209
是主报告；final953 是从相同 full 原始行扣除 dev256 后聚合的独立视图，不重跑。报告 behavior-
compatible、strict、SQL-only、valid SQL、失败分类、吞吐、显存、时间和实际费用。heldout 仍0/0。

## 成功、输出、风险与恢复

最小成功是 update671 adapter 和 decision packet；推荐成功再加同卷 full1209 E0/E1。输出包括 token
report、selection/prepared hashes、512/671 checkpoints、`epoch-extension-decision.json`、两个 full
generation 原始证据和 SQL-free result summary。NaN/Inf、identity drift、heldout 非零、一次安全重试
后 OOM、成本/时限都会停止；保留完整 checkpoint 后恢复，绝不删四条或临时换超参。

M3 测得训练 0.762684 s/microstep，因此训练仅线性估计 1.14 h；full1209 的唯一 E0/E1 generation
按 M3 测得分别约8.16h与1.82h。两次 teacher-forced dev 没有直接测量，保守用训练 microstep 代理
约≤0.11h；裸合计约11.22h。准备/IO/重载未知，建议12h、hard stop14h，历史费率2.59 CNY/h 下为
CNY31.08/36.26，开跑前必须重查价格并获控制批准。

## M4 后与 RL 的连接、学习清单与面试讲法

M4 证明的是固定 SFT 数据与公平 validation 下 4B adapter 相比 Base 的变化；它不证明 heldout
泛化、reward 安全或 RL 有效。成功 adapter 是后续误差分析/RL 的干净起点，不自动进入它们。

应掌握 `prepare_example`（SQL-only mask）、`select_formal_rows`（冻结 IDs）、
`build_accumulation_plan`/`scaled_compute_loss`（最后四条）、`train_formal_adapter`（checkpoint/resume）、
`evaluate_m4_teacher_loss.py`（诊断）和 `validate_comparison`（同卷）。面试可说：我保留了不获
批准的抽样历史，改用全量一轮；处理了不能整除 batch 的真实梯度语义；把 dev 曲线做成证据而非
自动化 p-hacking；并把 full1209 generation 去重，得到可恢复、可审计的正式 SFT 基线。
