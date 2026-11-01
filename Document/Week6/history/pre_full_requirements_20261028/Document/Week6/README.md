# Week 6 高级功能与鲁棒性优化

系统版本 `2.0.0`，开发分支 `codex/week6-delivery`。本版按受限工程范围对照四项要求：整体指令下的逐动作在线规划、错误检测与重试、UI识别准确率与速度、实时监测与日志。正常017和瞬态015的严格受控窗口验收通过，完整CPU回归为3027项通过、0失败，pytest 535.12秒，封装墙钟566.027秒，真实exit 0；244份文件的before、after与当前SHA完全一致。本版按上述受限工程范围闭合实现与证据；报告和分发文件按独立生成与发布清单核验。

层次化语义子任务生成及依赖管理未实现。`CALLER_GOAL`是本地执行容器，模型产生的是当前截图下的下一动作计划；受控四阶段窗口、两组固定目标和一次审阅拒绝不能推出任意应用成功率。Week 7 的综合任务集、跨应用、跨分辨率及竞品评估另行开展。Week 5 C11权重、历史训练、质量和资源补测结论保留，本周不新增训练或消息发送。

## 整体指令与固定工作流

CLI 的 `--instruction` 默认选择 `rolling_actions`，接收整体自然语言任务。调用方用可重复的 `--expect-text`、`--forbid-text` 提供不可由模型改写的最终文字规则。每次真实模型计划硬限 `max_steps=1`，输入后重新观察；最终规则未满足时在共享额度内继续规划。动作、规划、格式尝试和实际HTTP请求分别计数。

`week6.rolling_actions.v2` 在输入前编译记录中设置 `gate_on_step_verification=false`。模型自写的 step expected_result 仅作诊断，原始 passed、failed 或 inconclusive 均保留；调用方目标检查和独立新帧 final 才决定整体完成。执行器失败、未知派发、取消和超时继续按原运行器分支处理，不能把缺失结果改写成成功。

工作流仍要求显式屏幕文字检查，不支持消息发送工作流或进程谓词作为总体final。既有输出目录会被拒绝，原日志不授权恢复执行。

Python `AutomaticWorkflowRunner` API 默认仍为 `observed_departure`；CLI可用 `--automatic-protocol observed_departure` 显式选旧路径。它的marker/A过程检查属于兼容实现，不是017/015的实际验收模式。滚动动作模式没有模型生成的语义已完成阶段台账，不能把动作或固定容器数量解释成语义子任务数。

`--workflow` 保留调用方预写 `WorkflowSpec` 的入口，与 `--instruction` 互斥。固定工作流最多32个有序阶段，逐阶段严格检查行为保留。已有目录会被拒绝；旧日志只供检查，不触发自动恢复。默认dry-run不改变应用状态，也不把阶段计为实际完成。

## 运行入口与预算

先在仓库根目录检查输入定义，这一步不调用模型、不截屏、不输入、不创建运行目录：

```powershell
python scripts/week6_agent_cli.py --instruction "Prepare the ALPHA report with the GREEN theme and commit it locally in the open lab window" --expect-text "LOCAL COMPLETE" --expect-text "Project ALPHA" --expect-text "Theme GREEN" --validate-only
python scripts/week6_agent_cli.py --workflow configs/week6_example_workflow.json --validate-only
```

下面连接已启动的本机Ollama OpenAI兼容端点。`ollama-local`是该本机服务的占位API key。命令默认dry-run，会观察并调用模型；准备好对应专用窗口后才运行。使用普通CLI的生产执行器与下述Sky桥接验收是不同的派发环境。

```powershell
$env:GUI_AGENT_API_KEY = "ollama-local"
python scripts/week6_agent_cli.py --instruction "Prepare the ALPHA report with the GREEN theme and commit it locally in the open lab window" --expect-text "LOCAL COMPLETE" --expect-text "Project ALPHA" --expect-text "Theme GREEN" --provider openai_compatible --base-url http://127.0.0.1:11434/v1 --model qwen2.5vl:7b --planning-attempts 12 --model-requests 24 --max-actions 12 --task-timeout 1200
```

真实输入添加 `--execute`，仍需交互终端逐计划确认，没有 `--yes`。CLI默认配置为 `configs/week4.yaml`；双语screen配置须显式传 `--config configs/week6_accuracy.yaml`。`--model-requests`默认24，限制实际HTTP；`--planning-attempts`、`--max-actions`和`--task-timeout`限制全流程。`--decomposition-rounds`、`--max-subgoals`主要供旧observed-departure路径，不能当成滚动动作的语义完成数量。

自动入口要求可明确计数的transport。OpenAI兼容客户端按实际HTTP派发事件计数，关闭SDK隐式重试，单次超时随剩余时限缩小，迟到结果不继续执行。无法明确计数的LangChain transport不用于自动入口。请求数、规划调用、格式尝试和确认动作数不能互相替代。

## 双语OCR资源

`configs/week6_accuracy.yaml` 选择Tesseract的screen配置、en/zh双语、2倍缩放和0.5置信度阈值。来源、SHA256、许可证见 [OCR资源说明](resources/tesseract/README.md)。本次使用现成语言模型，没有训练或微调OCR。

该Windows后端的自定义tessdata路径须为无空格ASCII路径。复制仓库资源并仅设置当前PowerShell会话：

```powershell
$week6OcrData = 'D:/week6-ocr/tessdata'
New-Item -ItemType Directory -Path $week6OcrData -Force | Out-Null
Copy-Item -Path 'Document/Week6/resources/tesseract/*.traineddata' -Destination $week6OcrData
$env:TESSDATA_PREFIX = $week6OcrData
```

原native及默认英文模式保留。冻结留出集6帧48个唯一标签中，screen双语为46/48，精确率、召回率与F1均95.83%；同双语native为30/48。screen整帧处理中位342.302→375.676 ms，增加9.75%；18次相关重复不是18个独立画面。真实英文保存图三臂均5/5标签、2/2按钮，没有准确率增益，整帧未完整标注。

此前轮廓去重配对中，UI提取中位14.421→12.185 ms，完整有序候选一致；整帧处理未证明普遍加速。详见 [感知实测](results/perception_report.md)、[冻结留出说明](results/perception_accuracy_20261024/README.md) 和 [真实像素验证](results/perception_accuracy_live_20261024/external_001/summary.json)。

## 真实受控窗口与失败保留

`scripts/week6_local_lab.py` 只修改自身离线测试状态，提供项目、主题、审阅、本地提交四阶段。独立状态和事件文件只用于验收对照，不作为模型输入。

```powershell
python scripts/week6_local_lab.py --directory outputs/week6/lab_fixture_new --mode normal
```

| 新源码实际批次 | 条件 | HTTP 规划 输入 | 运行秒 | 严格结果 |
| --- | --- | --- | --- | --- |
| [正常017](results/live_auto_20261026_normal_017/acceptance.json) | ALPHA GREEN | 4 4 4 | 652.750 | 全部通过 |
| [瞬态015](results/live_auto_20261025_transient_015/acceptance.json) | BETA BLUE 一次审阅拒绝 | 5 5 5 | 275.703 | 全部通过 |

瞬态拒绝来自应用审阅层，两批所有ActionResult均success、已知执行器失败0；执行器失败分支另有CPU场景证据。两批的运行器成功、独立终态、不同输入回执逐项归因、源码前后未变和真实owner exit0共同核验，均为 `executed_caller_goal`、0未知派发、完整计数。瞬态原step诊断为4 passed、1 failed，提交预测句子未出现的failed原样保留，调用方与独立final通过。原局部passed也不证明语义阶段通过。

本机qwen2.5vl:7b经实际截图、CPU OCR和生产规划/运行器执行，输入走限制于专用窗口的Sky桥接。不同语义转换各归属于不同回执；400 ms跨页防重入属于实验环境。该证据不替代普通CLI/PyAutoGUI执行器的任意应用验证。两批分日运行，经过时间包含激活和operator桥接等待，不作正常与恢复速度的因果比较。

历史001至011的输出、环境和归因失败保留；012只在旧v1源码下通过；013为受控停止真实exit15，不补写完整acceptance。014有2个已知输入和1未知派发；016虽独立终态通过，但提交无对应输入回执，运行器又在派发前激活超时，整体failed。缺少的输入、摘要和退出码保持原样，不合并开发成功率。

## 状态与日志

根目录提供整体生命周期与真实终态；逐动作规划、输入、恢复主要在 `workflow/001_CALLER_GOAL` 子目录。根状态可能不逐步更新，不能以根动作0推断子任务无输入。

```powershell
python -S scripts/week6_status.py --run-dir outputs/week6/your_run_id
python -S scripts/week6_status.py --run-dir outputs/week6/your_run_id/workflow/001_CALLER_GOAL --watch --interval 1
```

只读工具仅用标准库，区分快照/日志矛盾、缺失、损坏、过期、进程消失和PID复用。`finished`表示终态记录可信，成功与否仍看task_status。长模型调用未生成阶段事件时，过期不能当成退出。

`dispatch_intents.jsonl`是意图，不证明操作已发生。执行器进入后缺完整结果记unknown并停止自动重放；派发前激活失败可能保持unknown=0。原生回复、单步上限、步骤、前后帧、调用方终态及owner真实退出分开保存。

## 检查与交付范围

完整CPU回归为3027项通过、0失败，pytest 535.12秒，封装墙钟566.027秒，真实exit 0；244份文件的before、after与当前SHA完全一致。

```powershell
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
python -m pytest -q
python -m ruff check src tests scripts
```

全量原件选择 [2026年10月26日full006](results/release_validation_completed_20261026_006/validation.json)，不沿用旧full005或尚未执行的旧日期目的地。故障模拟 [11场景218项](results/robustness_20261024_attempt005/summary.json)、保存图规划和真实窗口结果分别解释。

[第一阶段](Week6_Phase1_Test_Report.md)、旧报告/发布说明及旧包按原日期保留；初始使用说明见 [历史归档](history/v2_initial_20261024/README.md)。当前更新报告为 [实验报告](Week6_Robustness_Report.md)。交付检查覆盖相同completion字节、最终DOCX/PDF页图、数字链接及源码、wheel和压缩包字节；结果由随包校验清单记录。该检查不等于干净环境的全部依赖安装已经验证。
