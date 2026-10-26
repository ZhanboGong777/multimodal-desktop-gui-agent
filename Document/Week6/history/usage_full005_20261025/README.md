# Week 6 高级功能与鲁棒性优化（验收待完成）

**完整桌面验收尚未完成，当前不是正式完善版。** 系统版本为 `2.0.0`，开发在 `codex/week6-delivery` 分支进行。当前 full005 完整 CPU 回归为 **2927 通过、0 失败**，但最新实现尚无完整、可归因的真实窗口通过记录。模型对操作后画面的预测仍有问题，正常 009 与瞬态恢复 010 尚未运行。

本周对照四项要求：复杂任务自动拆解与分步执行、错误检测与重试、UI 识别准确率与速度、实时监测与日志。Week 7 的综合任务集、跨应用、跨分辨率及竞品评估另行开展。本周不新增 Week 5 消息发送，不改 C11 权重、训练、质量或资源补测结论。

## 自动拆解与固定工作流

`--instruction` 接收整体自然语言请求；调用方另用 `--expect-text`、`--forbid-text` 定义不可由模型改写的最终文字规则。分解器根据当前截图和候选元素生成有过程检查的子目标，阶段执行后重新观察；后续页面尚不可见时，可以在共享额度内继续滚动拆解。模型输出 `finish`、动作派发成功或过程检查通过，均不能单独证明整体任务完成，最终仍须独立检查调用方规则。

`--workflow` 保留调用方预先定义 `WorkflowSpec` 的入口，与 `--instruction` 互斥。固定工作流最多 32 个有序阶段，每阶段必须有非空的文字检查。两种模式在阶段失败、取消、未知派发、额度或时限耗尽时停止后续执行；dry-run 不把阶段计作实际完成。每次须使用新的运行目录，已有日志不能触发自动恢复执行。

过程检查是合取：全部 `expect_text` 应在操作后出现，全部 `forbid_text` 应在操作后消失。`forbid_text` 不表示“不要点击”或“已经完成”。当前实现拒绝整体条件在执行前已经满足的提案；例如当前同时显示 ALPHA 与 BETA，`expect ALPHA + forbid BETA` 目前为假，可以表示保留 ALPHA 并移除 BETA，不能仅因 ALPHA 已可见就拒绝。模型不能禁止最终必需文字；调用方最终规则及终态复核继续保留。

已通过阶段会反馈给模型；与已通过阶段完全相同的指令，或相同的文字检查组合，会被拒绝。这只是字符串与规则组合检查，**不能保证语义等价的改写不会重复操作**。分解的首次请求与至多两次约束修正均计入请求额度；默认共至多三次，修正不能重置共享额度。

## 运行入口与额度

先在仓库根目录验证整体请求；这一步不调用模型、不截屏、不输入、不创建运行目录：

```powershell
python scripts/week6_agent_cli.py --instruction "Prepare the ALPHA report with the GREEN theme and commit it locally in the open lab window" --expect-text "LOCAL COMPLETE" --expect-text "Project ALPHA" --expect-text "Theme GREEN" --validate-only
python scripts/week6_agent_cli.py --workflow configs/week6_example_workflow.json --validate-only
```

以下自动模式示例连接已启动的本机 Ollama OpenAI 兼容端点。`ollama-local` 是该本机端点的占位 API key。命令默认 dry-run，仍会观察当前桌面并调用模型；它不是已通过的完整桌面示例。准备好专用窗口后才运行：

```powershell
$env:GUI_AGENT_API_KEY = "ollama-local"
python scripts/week6_agent_cli.py --config configs/week6_accuracy.yaml --instruction "Prepare the ALPHA report with the GREEN theme and commit it locally in the open lab window" --expect-text "LOCAL COMPLETE" --expect-text "Project ALPHA" --expect-text "Theme GREEN" --provider openai_compatible --base-url http://127.0.0.1:11434/v1 --model qwen2.5vl:7b --planning-attempts 16 --model-requests 24 --max-actions 16 --task-timeout 900 --decomposition-rounds 4
```

真实执行须添加 `--execute`，并使用交互终端逐计划确认，没有 `--yes`。CLI 默认配置仍为 `configs/week4.yaml`；使用双语 screen 配置须显式传入 `--config configs/week6_accuracy.yaml`。模型是否可用、服务上下文大小和任务时限须按实际环境检查；CPU 回归与输入验证不证明模型运行可完成。

| 参数 | 默认或范围 | 作用 |
| --- | --- | --- |
| `--model-requests` | 24，正整数 | 自动拆解、格式修正与动作规划共享的实际 HTTP 请求上限 |
| `--decomposition-rounds` | 4，1 至 8 | 根据新观察继续拆解的轮数上限 |
| `--max-subgoals` | 8，1 至 32 | 分解器可提出的子目标数量上限 |
| `--planning-attempts` | 来自配置；自动模式至少 2 | 包含自动拆解在内的共享规划调用额度 |
| `--max-actions`、`--task-timeout` | 来自配置，可显式覆盖 | 全流程动作额度与单调时钟时限 |
| `--expect-text`、`--forbid-text` | 可重复传入，自动模式须有规则 | 调用方最终文字规则；固定模式从工作流定义读取 |
| `--output-directory`、`--run-id` | `outputs/week6`、自动生成 ID | 新运行目录的父路径与名称 |

自动入口要求能明确计数的 transport；OpenAI 兼容客户端按实际 HTTP 派发计数，关闭 SDK 隐式重试，超时随剩余时限收缩。固定工作流保留原有 provider 选择。请求数、规划调用数和确认动作数是不同口径，不能互相替代。

## 双语 OCR 资源

`configs/week6_accuracy.yaml` 选择 Tesseract 的 `screen` 配置，`en/zh` 双语、2 倍缩放、置信度阈值 0.5。资源、来源、SHA256 与 Apache 2.0 许可证见 [OCR 资源说明](resources/tesseract/README.md)；这里使用现成模型，没有训练或微调 OCR 模型。

本次 Windows 后端不能读取含非 ASCII 字符的 tessdata 路径，自定义路径也须无空格。先把仓库的两份资源复制到 ASCII 无空格目录，并仅为当前 PowerShell 会话设置环境变量：

```powershell
$week6OcrData = 'D:/week6-ocr/tessdata'
New-Item -ItemType Directory -Path $week6OcrData -Force | Out-Null
Copy-Item -Path 'Document/Week6/resources/tesseract/*.traineddata' -Destination $week6OcrData
$env:TESSDATA_PREFIX = $week6OcrData
```

原 `native` 配置和默认英文模式继续可用。screen 用额外 CPU 处理时间换取当前留出集的小字与中文识别效果，不能据此声称整个智能体更快。

## 专用离线窗口与验收限制

`scripts/week6_local_lab.py` 提供项目、主题、审阅、本地确认四个阶段的离线 Tk 窗口，只改变自身测试状态。窗口状态与事件文件用于独立验收，不交给分解器、动作规划器或观察器。

```powershell
python scripts/week6_local_lab.py --directory outputs/week6/lab_fixture_new --mode normal
```

旧 [HTML 演示页面](demo/local_editor.html) 及 `configs/week6_example_workflow.json` 仍保留，用于固定工作流示例。它们不是完整真实任务通过证据。

最新真实窗口批次 [正常 008](results/live_auto_20261024_normal_008/acceptance.json) 为 ALPHA + GREEN，整体 `failed`：两个阶段分别归因到两个已确认输入，审阅与本地确认未完成；6 次模型请求、6 次规划调用、2 次确认动作、0 次未知派发，总耗时 776.391 秒，含激活及桥接等待。第三轮检查存在矛盾且未派发；直接停止原因为约 90 秒未取得窗口激活回执。它使用合取修复前的实现，不能当作当前版本通过，也不能把停止原因改写为新校验生效。

当前实现的正常 009、瞬态恢复 010 尚缺证据。每个完整验收须同时满足运行器成功、独立终态检查、四个语义转换各自对应不同输入回执、源码前后未变。瞬态批次还须如实保留一次审阅拒绝及后续恢复。持续故障、歧义停止目前只有模拟覆盖。

真实开发批次使用受限 Sky 输入桥接与专用窗口独立状态；普通 CLI 使用仓库执行器。桥接批次不能替代普通执行器的任意应用端到端验证。窗口增加的 400 ms 跨页防重入属于实验环境修复，不计为模型质量提升。

## 错误恢复、状态与日志

重规划上下文含最近失败分类、最多五个动作结果、累计和剩余额度、剩余时间；不携带旧坐标、输入文字、模型原文或自由文本错误。派发前保留 `dispatch_intents.jsonl`。进入执行器后异常或中断且无完整结果时，未知次数单列为 `unconfirmed_action_attempts`，停止自动重试，不能记成功或零动作。

根目录及任务目录保留原子替换的 `run_status.json`、追加的 `status_events.jsonl`，记录阶段、真实终态、额度、耗时和进程身份；不保存任务正文、输入或坐标。

```powershell
python -S scripts/week6_status.py --run-dir outputs/week6/your_run_id
python -S scripts/week6_status.py --run-dir outputs/week6/your_run_id --watch --interval 1
```

状态读取只使用标准库。快照与日志矛盾、损坏、缺失、超过 30 秒未更新、进程消失或 PID 复用分别诊断；`finished` 表示终态记录可信，成功与否仍看 `task_status`。阶段更新不是独立心跳，不能证明长模型调用仍推进；确认数与未知派发保持分别记录。

## 已确认的测量与历史

- [冻结留出准确率](results/perception_accuracy_20261024/README.md)：6 帧、48 个唯一标签。native 英文召回 58.33%，native 双语 62.50%，screen 双语 95.83%，后者精度及 F1 均为 95.83%。screen 与同双语 native 相比，解码、OCR、UI 总中位耗时 342.302→375.676 ms，增加 9.75%。每臂 18 条相关重复，不能当成 18 个独立画面；错误保留。
- [真实像素独立验证](results/perception_accuracy_live_20261024/external_001/summary.json)：一张保存的真实英文画面，冻结 ROI 的 5 个标签、2 个按钮三臂均识别正确，无准确率增益。全图未穷尽标注，不能报告全图精度或桌面执行成功。
- [原感知配对计时](results/perception_report.md)：4 张合成图、100 组配对，完整有序候选一致；UI 提取中位 14.421→12.185 ms，整帧处理未证明加速。旧英文 20/20、中文 0/8 属于当时仅英文资源环境，保留历史，不能覆盖为新双语结果。
- [故障模拟](results/robustness_20261024_attempt005/summary.json)：11 场景、218 项检查通过。实际运行器、适配器、验证器、记录器和工作流配合明确模拟后端；不能据此计算真实桌面恢复成功率。
- [早期真实模型规划](results/model_recovery_20261024_001/probe_notes.md) 及 [真实窗口开发审计](results/live_development_audit_20261024/summary.json) 保留输出截断、歧义、环境中断和跨页误触等失败，不把开发中断当模型质量零分，也不合并为任务成功率。
- 保存图开发推理共 9 次：旧生产分解 3、显式多轮 2、字段含义 2、新生产 1、schema 示例变体 1。前 7 次由旧检查拒绝，其中一例存在合取误拦；后两次生成同样的 `expect Review report + forbid Draft ready`，当前结构允许，但窗口审阅后原按钮会消失，后态可实现性仍有疑点。没有新截图、输入或执行，没有真实新 GUI 验收；示例变体未集成。原记录见 [当前生产诊断](results/saved_decomposition_20261025_002/summary.json) 与 [示例变体诊断](results/saved_forbid_example_20261025_001/summary.json)。

当前 [full005 完整回归](results/release_validation_completed_20261025_005/validation.json) 为 2927 通过、0 失败，pytest 428.96 秒、封装 431.515 秒、exit 0，241 份前后及当前文件哈希一致。分解实现 SHA256 为 `3bc8ff2accfb32d5e4f58a5461fec0bfd97ee0bc14b5434d047632c5ef3e7a18`。这证明该版本通过对应回归，不代替真实窗口验收。

Windows CPU 检查统一父子 Python 的 UTF-8 编码：

```powershell
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
python -m pytest -q
python -m ruff check .
```

[第一阶段](Week6_Phase1_Test_Report.md)、旧 [v2 报告](Week6_Robustness_Report.md) 与旧 [发布说明](Release_Notes.md) 原样保留，是各自日期的历史阶段材料；不用于声明当前完成。旧使用说明按原字节存入 [初始版本归档](history/v2_initial_20261024/README.md)，哈希另存同目录。当前仅有 full005 验收待完成稿；正式报告、源码与 wheel 对照及发布审计须等缺失的真实验收完成后再闭合。
