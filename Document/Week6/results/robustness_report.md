# Week 6 无 GUI 端到端故障演示

2026 年 10 月 24 日，在 `codex/week6-delivery` 分支运行
`scripts/week6_robustness_demo.py`。真实使用生产代码中的 `WorkflowRunner`、`TaskRunner`、
`ActionAdapter`、`Verifier` 和 `TaskRecorder`；仅观察、规划、执行后端和时钟由确定状态机提供。
模拟执行器只修改内存中的测试应用状态，没有调用桌面输入、OCR、模型、GPU 或消息发送。

最终批次为 `robustness_20261024_attempt005/`：**11/11 场景、218/218 自动核对项通过**，
源文件在运行前后 SHA256 一致，未加载 `torch`、`paddle` 或 `transformers`。
批次场景运行墙钟时间约 10.569 秒；每个场景另记录虚拟时间，用于确定性地触发 deadline。
虚拟时间不作为系统性能实测。初次批次 `robustness_20261024_attempt001/` 亦完整保留；
后续批次保留驱动及生产代码快照，第五批次对应最终截图时限和根监测修复；此前四批分别保存，不覆盖历史记录。

| 场景 | 实际整体状态 | 已确认结果调用数 | 未确认调用 | 规划调用 | 恢复次数 | 子目标状态 | fresh 总体检查 |
|---|---|---:|---:|---:|---:|---|---|
| 无故障三子目标 | succeeded | 3 | 0 | 3 | 0 | 完成、完成、完成 | passed |
| 第2子目标目标短暂缺失 | succeeded | 3 | 0 | 4 | 1 | 完成、完成、完成 | passed |
| 第2子目标一次明确执行失败 | succeeded | 4 | 0 | 4 | 1 | 完成、完成、完成 | passed |
| 持续执行失败至共享规划上限 | failed | 3 | 0 | 3 | 1 | 完成、失败、未开始 | 未运行 |
| 共享动作预算耗尽 | blocked | 2 | 0 | 2 | 0 | 完成、完成、未开始 | 未运行 |
| 执行返回时总时间耗尽 | timed_out | 1 | 0 | 1 | 0 | 超时、未开始、未开始 | 未运行 |
| 第2子目标规划中取消 | cancelled | 1 | 0 | 2 | 0 | 完成、取消、未开始 | 未运行 |
| 子目标全过、fresh 总体规则未过 | failed | 3 | 0 | 3 | 0 | 完成、完成、完成 | failed |
| dry-run | dry_run_completed | 3 | 0 | 3 | 0 | 均为 dry_run_completed | 未运行 |
| 消息工作流 | rejected | 0 | 0 | 0 | 0 | 均未开始 | 未运行 |
| 执行器改变模拟状态后抛异常 | failed | 1 | 1 | 2 | 0 | 完成、失败、未开始 | 未运行 |

“已确认结果调用数”沿用生产 `action_count`：有完整 `ActionResult` 的调用，包括明确失败和
dry-run 校验；它不等于成功的实际动作数。dry-run 的三次调用没有改变模拟状态，也不记任务成功。
异常场景中第2子目标的调用没有返回完整结果，因此单列为一次未确认调用，后续子目标停止，
`accounting_complete=false`，没有自动重规划或重放该调用。

目标缺失场景在第2子目标动作前的当前帧移除目标，真实解析器拒绝派发；随后目标移动到新位置。
恢复反馈的原因记录为 `target_resolution`，恢复点击使用新位置，目标 ID 属于实际派发所用的新帧。
第1子目标只执行一次。一次明确执行失败的场景返回失败 `ActionResult`，随后从当前帧恢复，
四次已确认调用中有一次失败；失败调用原样保留。

所有已派发坐标均自动对照当前模拟帧；记录中的解析 ID 必须存在于该派发帧的 JSON。
总体检查必须比所有子任务验证再获取一个新观察 ID；即使三个子目标都通过，缺失总体标记也会失败。
消息工作流在 `WorkflowSpec` 校验时被拒绝，未创建任何子运行器。取消在模拟规划器中注入
`KeyboardInterrupt`，真实取消路径保留第1子目标已完成的证据。

`on_checkpoint` 回调在每次子目标状态持久化后读取当时的 `workflow_summary.json`，
核对已完成数、动作数和规划数；回调事件必须与追加日志前缀完全一致。
最终终态另核对 `checkpoints.jsonl` 末行及 `workflow_summary.json`。
回调当前不接收 `finally` 中的最后一条终态事件，因此不能把回调最后一次状态误称为总体结论。

每个场景拥有独立目录，包含规范、选项、模拟观察/规划/执行轨迹、真实子任务证据、恢复记录、
实时监测、检查点回调快照和期望/实际逐项核对结果。批次保留实际源文件快照及前后 SHA256。
再次指定已有输出目录会被拒绝；已实际验证拒绝后原批次所有文件字节不变，
证据见 `robustness_refusal_audit.json`。

在仓库根目录复现到一个新的批次：

```powershell
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'
python -X utf8 scripts/week6_robustness_demo.py --output-dir outputs/week6_robustness_new
```

也可省略 `--output-dir`，默认在 `Document/Week6/results/` 创建带 UTC 时间和随机后缀的新目录。
任何自动核对失败都会令 CLI 返回非零退出码，同时保留本次所有场景记录。
本演示证明的是生产运行、验证、恢复和记账路径在这些确定故障下符合预期，
不提供真实桌面任务或真实模型的成功率，也不能替代实际应用验收。
