# Week 6 高级功能与鲁棒性优化

系统版本为 `2.0.0`，开发分支为 `codex/week6-delivery`。本轮从第一阶段提交
`9149e064d90ab6796e7f25ece6cbbd026785b26c` 继续，其父提交为 Week 5 正式提交
`d180710d7efcfdd6930065cdba6abf882a52328f`。Week 5 权重、历史结果和文件校验清单保持原样。

本周实现子目标分解执行、异常恢复、感知处理优化和运行状态监测，交付系统 v2.0、复现入口与鲁棒性报告。
故障注入、合成截图感知实测和真实本地模型的保存截图规划分别记录，不合并为真实桌面任务成功率。

## 子目标与总体检查

`WorkflowSpec` 由调用方明确提供最多 32 个有序子目标，每个子目标仍交给原 `TaskRunner` 规划和执行。
子目标必须有非空 `expect_text` 或 `forbid_text`。通过实际检查后才进入下一阶段；全部阶段结束后重新观察，
用独立 `final_check` 判断总体目标。模型说完成、返回 `finish` 或动作派发成功都不能单独完成子目标。

各阶段共享动作额度、规划器调用次数和单调时钟时限。失败、取消、未知执行结果或预算耗尽时停止后续阶段。
已验证阶段保留，不重放。根目录必须新建；旧日志无论完整或损坏都不能触发恢复执行。
dry-run 可以检查各阶段计划，但完成子目标数保持 0，结果为 `dry_run_completed`。

本版最终总体检查仅支持显式文字规则，不支持进程退出规则。工作流拒绝消息任务，防止多阶段之间重置发送保护。
原单任务的消息前后检查和发送后禁止重试规则继续保留；本周没有新增发送额度。

## 运行入口

先验证定义，不加载模型、不捕获屏幕、不创建运行目录：

```powershell
python scripts/week6_agent_cli.py --workflow configs/week6_example_workflow.json --validate-only
```

下面命令连接已启动的本机 Ollama OpenAI 兼容端点。客户端要求 API key；本机 Ollama 可使用占位值 `ollama-local`，它不是托管服务凭据。
默认 dry-run。真实执行仍要求交互终端、逐阶段展示计划和确认，没有 `--yes`：

```powershell
$env:GUI_AGENT_API_KEY = "ollama-local"
python scripts/week6_agent_cli.py --workflow configs/week6_example_workflow.json --provider openai_compatible --base-url http://127.0.0.1:11434/v1 --model qwen2.5vl:3b --ocr-engine tesseract --planning-attempts 6 --max-actions 12 --task-timeout 240
python scripts/week6_agent_cli.py --workflow configs/week6_example_workflow.json --provider openai_compatible --base-url http://127.0.0.1:11434/v1 --model qwen2.5vl:3b --ocr-engine tesseract --planning-attempts 6 --max-actions 12 --task-timeout 240 --execute
```

示例对应 [本地演示页面](demo/local_editor.html)：打开页面、打开编辑区、输入演示标记并点击本地完成按钮。
页面仅保留内存状态，不发送消息、不写文件。可用浏览器直接打开 HTML，或在仓库根目录启动只监听本机的静态服务：

```powershell
python -m http.server 8766 --bind 127.0.0.1 --directory Document/Week6/demo
```

访问 `http://127.0.0.1:8766/local_editor.html`，将页面放在前台后运行 CLI。
示例不是已测得的真实桌面成功记录；验证范围见下文。

## 故障恢复与中断记录

重规划收到 `context.recovery`，包含最近一轮失败分类、最多五个动作结果、累计与剩余动作数和剩余时间，
不携带旧坐标、输入文字、模型原文或自由文本错误。恢复仍受原次数、动作和时限约束。

派发前记录无输入内容的 `dispatch_intents.jsonl`。进入执行器后异常或中断且无完整结果时，
`unconfirmed_action_attempts` 单列未知次数，不当作成功或零动作，并停止自动重试。工作流确认动作数此时只是已知下界。
监测和意图记录写入后再次核验时限与需要的前台身份，避免记录开销跨过最后检查。
最终截图和证据保存后再次检查时限，耗尽预算后不启动新的任务评估。

取消摘要读取步骤与恢复日志中的完整有效记录，注明损坏行并保留原始字节。
自然语言成功条件、空文字检查、带错误的观察，以及仅有窗口类名的空画面不能证明任务完成。

## 运行状态

每个任务和 CLI 工作流根目录有原子替换的 `run_status.json` 与顺序追加的 `status_events.jsonl`。
它们记录阶段、子目标、已验证数、动作与规划额度、相对耗时、进程身份和真实终态，不保存任务正文、输入或坐标。

```powershell
python -S scripts/week6_status.py --run-dir outputs/week6/your_run_id
python -S scripts/week6_status.py --run-dir outputs/week6/your_run_id --watch --interval 1
```

读取工具只用标准库，不启动模型或桌面。它核对快照与日志的序列及内容；损坏、缺失、超过 30 秒未更新、
进程消失或 PID 复用分别显示诊断。`finished` 表示终态记录可信，具体成功与否仍看 `task_status`。
阶段更新不能作为独立心跳证明长模型调用仍在推进。
根监测的过程动作数来自已确认结果；终态计入未确认派发并显示 `unconfirmed_dispatch`。
`workflow_summary.json` 仍分别保留确认数与未知次数，未知调用不计为成功。

## 测量与验证

- [感知测量](results/perception_report.md)：4 张固定合成图、100 组交替顺序配对计时，完整有序候选一致。
  UI 提取中位耗时 14.421→12.185 ms；简单深色图略慢，整帧处理未证明加速。
  英文标签 20/20，中文 0/8；本机 Tesseract 仅有 eng/osd。
- [故障演示](results/robustness_report.md)：真实运行器、适配器、验证器、记录器和工作流配合明确模拟后端。
  各批次保留原始日志、检查点和源码快照，结果以对应汇总为准。
- [真实模型规划](results/model_recovery_20261024_001/probe_notes.md)：固定本机 `qwen2.5vl:3b`，
  3 次规划调用、5 次推理请求。两个目标对照正确；双同名按钮未拒绝选择，三个最终计划均缺少提示要求的 `finish`。
  2/3 仅表示目标对照，不代表完整协议遵循率或桌面任务成功率。

复现时选择新输出目录，不覆盖历史批次：

```powershell
python scripts/week6_benchmark_perception.py --help
python scripts/week6_robustness_demo.py --help
python scripts/week6_model_recovery_probe.py --help
```

Windows CPU 检查统一父子 Python 的 UTF-8 编码：

```powershell
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
python -m pytest -q
python -m ruff check .
```

[第一阶段记录](Week6_Phase1_Test_Report.md) 原样保留：首次全量 2667 通过、6 项编码失败，随后六项复测通过；
当时没有一次全部通过的完整运行。本次全量记录独立保存，不重写历史。

## 能力边界

子目标及总体文字规则由调用方定义，系统不会自行把自然语言句子变成可信验收器。
当前截图中的有效 ID 仅证明目标属于这一帧，不能独立证明模型消除了语义歧义。
模型规划质量、中文 OCR 支持和真实桌面任务完成率仍需分别测量。本周不追加 Week 5 消息发送，
不改写原 C11 的训练、质量或资源补测结果。
