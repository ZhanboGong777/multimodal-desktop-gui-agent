# Week 6 高级功能与鲁棒性优化

系统版本 `2.1.0`，开发分支 `codex/week6-full-requirements`。本轮补齐整体指令到语义子任务序列的自动生成、逐子任务执行和复核，并增加持久化 OCR 后端与当前计划进度记录。v5首轮冻结源已完成full003回归，normal007仍未通过；**后续提示修订的冻结与回归、真实窗口验收、报告及分发文件仍待完成**，不能沿用改动前的回归结论。

新证据统一存放于 [full_requirements_20261028](results/full_requirements_20261028/)。[初始语义实现原件](results/full_requirements_20261028/semantic_implementation_001/) 保留原版本范围；[v5首轮冻结记录](results/full_requirements_20261028/semantic_caller_predicates_001/final_source_freeze_001.json)对应normal007所用源码。该版的[285项定向CPU检查](results/full_requirements_20261028/semantic_caller_predicates_001/cpu_validation_001/validation.json)包含SDK/MockTransport请求体和独立审计边界，pytest91.62秒、实际exit0，Ruff实际exit0。[完整收集](results/full_requirements_20261028/semantic_caller_predicates_001/full_collection_001/collection.json)本身只记录3234项数量；其后的[full003实际运行](results/full_requirements_20261028/full_validation_003/validation.json)才记录3234项通过、pytest/Ruff实际退出码0。新提示修订不沿用这些通过结果，回归也不能替代真实模型效果和实机验收。

本版实现有序语义子任务，不要求调用方手写分解，也不把旧 `CALLER_GOAL` 滚动容器当作语义序列。复杂依赖图、任意应用成功率及跨应用、跨分辨率的系统评估不在已验证结论中。Week 7 的综合评估另行开展；Week 5 C11 权重、历史训练、质量和资源补测结论保留，本轮不新增训练或消息发送。

## 自动语义子任务与兼容入口

CLI 的 `--instruction` 默认选择 `semantic_subtasks`。调用方只提供整体自然语言任务，以及可重复的 `--expect-text`、`--forbid-text` 最终文字规则。模型根据整体意图和这些规则提出有序目标，每项包含 `goal_id`、目标说明及完成判据。初始计划仍须原生给出至少两个不同语义目标；它们不是本地固定步骤表，也不能只是整段原请求的重复包装或裸动作名。每个目标的动作规划仍收到完整调用方请求。

当前协议为 `week6.semantic_subtasks.v5`。高层采用 `input_modality="caller_intent_only"`：只接收完整原请求、不可变的调用方最终文字规则，以及已验证目标和失败类别的逻辑记录，不发送当前OCR、控件几何、窗口snapshot或图片；`image_path=None`、`image_attached=false`。原始snapshot、OCR和像素继续归档，供运行范围、输入前检查及事后复核使用，不能因文件留存就声称高层模型见过它。低层 `TaskPlanner` 仍接收逐次新截图，定位并检查每个实际输入。

阶段 `expect_text` 只能逐字选择调用方原始 `expect_text`，`forbid_text` 只能逐字选择原始 `forbid_text`，保留正反极性。相同候选写入模型可读格式说明和传输JSON schema；本地再检查原生字符串的精确成员关系，不靠截断、去空格或改字放行。模型可把相关操作合成一个有可信完成边界的目标，不能预测新页面文案来凑阶段数。至少两个不同调用方谓词只是候选来源的必要条件：同义、子串或只在终页同时出现的文字，仍可能无法区分实际阶段；谓词数量本身不是分解成功证据。

高层按顺序规划请求中的未来状态变化。后续页面控件尚未出现在初屏，不等于用户缺少参数，也不使明确要求变成可选项；到达对应阶段后，动作层才复核真实控件，不可用时停止。真正缺失、矛盾的参数或无法形成可验证条件时仍可澄清，安全限制仍可拒绝，不强制ready。高层不再根据当前屏幕消歧，纯图形理解、任意未来文案和任意应用的可验证分解没有验收结论。

本地前置检查会拒绝协议明确保留的未填写标记，例如 `[not specified]`、`[unknown]` 和 `<TBD>`，保留具体原因及0次HTTP、0输入。它不是模型学会澄清的证据，也不是完整的自然语言缺参检测；没有这些标记不证明参数齐全，把保留标记当作字面操作内容的任务可能不受此入口支持。

完整原请求继续传给每个子目标的动作规划器。进入目标前重新观察，检查窗口身份、范围与文字证据，并在相关输入前保存该目标的固定后置条件；当前画面已满足的一组宽泛文字不能单独授予完成。每次动作计划最多一个步骤，一个语义目标可包含多次动作计划。执行后用可信检查器复核固定目标条件，再用独立新帧确认该目标；所有语义目标通过后，才另取新帧检查原调用方 final。模型无法改写调用方最终规则。

模型自写的动作 `expected_result` 是诊断信息，原始 passed、failed、inconclusive 均保留；它不能替代固定语义目标检查。目标未通过时，只有计数完整、没有未知派发或取消等停止条件的已知失败，才允许在剩余额度内重新观察并生成未完成目标的新计划。旧失败及旧条件保留，新条件在下一次输入前冻结；已验证目标不会自动重放，事后出现总体 final 也不能把失败语义目标改判通过。字面目标去重不能证明所有改写后的动作都具有幂等性。

Python `AutomaticWorkflowRunner` API 默认仍为 `observed_departure`。需要新路径时显式传 `protocol_mode="semantic_subtasks"`；CLI 可用 `--automatic-protocol observed_departure` 或 `rolling_actions` 选择兼容模式。旧 marker/A 过程检查属于 observed 路径。`week6.rolling_actions.v2` 保留整体目标下逐动作规划和独立 final，但没有模型生成的语义目标序列台账，历史017/015使用的是该模式。

`--workflow` 保留调用方预写 `WorkflowSpec` 的入口，与 `--instruction` 互斥，最多32个有序阶段，原严格检查行为不变。自动模式和手写模式都要求显式文字规则，不支持消息发送工作流或进程谓词作为总体 final。既有输出目录会被拒绝，旧日志只供检查，不触发自动恢复；dry-run 不授予实际执行完成。

## 运行入口与共享预算

先在仓库根目录检查输入定义。这一步不调用模型、不截屏、不输入，也不创建运行目录：

```powershell
python scripts/week6_agent_cli.py --instruction "Prepare the ALPHA report with the GREEN theme and commit it locally in the open lab window" --expect-text "LOCAL COMPLETE" --expect-text "Project ALPHA" --expect-text "Theme GREEN" --validate-only
python scripts/week6_agent_cli.py --workflow configs/week6_example_workflow.json --validate-only
```

下面连接已启动的本机 Ollama OpenAI 兼容端点，并显式选择持久化 OCR 配置。`ollama-local` 是本机服务的占位 API key。命令默认 dry-run，会观察并调用模型；对应专用窗口、DLL 和语言资源准备好后才运行。普通 CLI 的生产执行器与历史 Sky 桥接验收属于不同派发环境。

```powershell
$env:GUI_AGENT_API_KEY = "ollama-local"
python scripts/week6_agent_cli.py --config configs/week6_accuracy_persistent.yaml --instruction "Prepare the ALPHA report with the GREEN theme and commit it locally in the open lab window" --expect-text "LOCAL COMPLETE" --expect-text "Project ALPHA" --expect-text "Theme GREEN" --provider openai_compatible --base-url http://127.0.0.1:11434/v1 --model qwen2.5vl:7b --planning-attempts 12 --model-requests 24 --max-actions 12 --task-timeout 1200
```

真实输入添加 `--execute`，仍需在交互终端逐计划确认，没有 `--yes`。CLI 默认配置仍为 `configs/week4.yaml`；持久化双语 screen 配置必须显式传入。`--max-subgoals` 默认8、最多32，语义初始计划至少2；`--decomposition-rounds` 默认4、范围1至8，约束未完成目标的修订次数。

`--model-requests` 默认24，按实际 HTTP 派发计数。`--planning-attempts` 包含语义分解、格式修正和动作规划，`--max-actions` 与 `--task-timeout` 覆盖全流程；子任务和修订不重置这些额度，记录 I/O 与观察也占用同一截止时间。OpenAI 兼容客户端关闭 SDK 隐式重试，并随剩余时限缩小单次超时；迟到结果不继续执行。无法明确计数的 LangChain transport 不用于自动入口。请求数、规划调用、格式尝试与确认动作数分别记录。

## 持久化双语 OCR 的复现条件

`configs/week6_accuracy_persistent.yaml` 显式选择 Tesseract C API 的 `persistent` transport：en/zh 双语、screen 配置、2倍缩放、0.5置信度阈值，与 screen 子进程路径使用相同像素处理和语言资源。它复用 API 实例，在每帧之间清理结果和自适应状态；不是训练或微调 OCR。原 subprocess、native 配置和默认英文模式保留。

Windows 需要已安装的 Tesseract 可执行文件、匹配的 `libtesseract-5.dll` 及其依赖 DLL。示例配置指向 `C:/Program Files/Tesseract-OCR/libtesseract-5.dll`，语言目录为 `D:/week6-full-ocr/tessdata`；换机器后须按真实安装位置调整。缺失 DLL、C API 函数或语言文件时明确失败，不静默改用其他 transport 或下载资源。仓库只附语言模型与许可证，不附系统 DLL。

语言文件来源、SHA256 和许可证见 [OCR 资源说明](resources/tesseract/README.md) 及其中的 `provenance.json`。本机复现可从仓库根目录复制到无空格 ASCII 路径：

```powershell
$week6OcrData = 'D:/week6-full-ocr/tessdata'
New-Item -ItemType Directory -Path $week6OcrData -Force | Out-Null
Copy-Item -Path 'Document/Week6/resources/tesseract/*.traineddata' -Destination $week6OcrData
```

该命令不修改系统 Tesseract 安装；持久化示例通过配置中的 `tesseract_tessdata_dir` 显式传目录。若复现旧 `configs/week6_accuracy.yaml` 子进程实验，仍按资源说明设置当前会话的 `TESSDATA_PREFIX`。现有 Windows 后端的自定义目录使用无空格 ASCII 路径。

新版持久化性能原件位于 [primary](results/full_requirements_20261028/perception_holdout_primary_001/) 和 [replication](results/full_requirements_20261028/perception_holdout_replication_001/)，最终汇总与发布数字待验收清单核定。旧版6帧48标签留出结果仍为：screen 双语46/48（精确率、召回率、F1均95.83%），同双语 native 30/48；旧 screen 整帧中位342.302→375.676 ms，增加9.75%。这些是历史配置对照，不是持久化后端的新加速结论。18次相关重复不等于18个独立画面，历史真实英文保存图三臂均5/5标签、2/2按钮，没有准确率增益。

旧轮廓去重配对中，UI 提取中位14.421→12.185 ms，完整有序候选一致；这不证明整帧或整个智能体普遍加速。原件见 [历史感知实测](results/perception_report.md)、[冻结留出说明](results/perception_accuracy_20261024/README.md) 和 [真实像素验证](results/perception_accuracy_live_20261024/external_001/summary.json)。

## 新验收状态与历史真实窗口

**v2.1.0 的语义子任务正常/瞬态真实验收仍待完成，v5首轮回归通过仅对应修改前源码。** 新窗口的原始请求、模型原生语义计划、逐目标合同、前后观察、动作回执、独立事件核对与真实 owner 退出结果分别保存于 [本轮 live 目录](results/full_requirements_20261028/live/)。最终报告、源码包和分发清单也仍待更新；不能把“实现冻结”或“回归通过”写成“全部交付通过”。

本轮[尝试索引](results/full_requirements_20261028/live_attempts_index_004.json)保留normal001至006：001初始请求超时，002理由循环至输出长度截断，003、004及006误判调用方缺信息，均未产生输入；005在模型调用前连续画面桥接超时、缺少激活收据，不记为v4模型语义失败。随后[三次模型开发诊断](results/full_requirements_20261028/semantic_model_diagnostic_001/diagnostic_analysis_001.json)没有GUI、输入或目标完成：A重复检查条件被拒，B结构允许但预测文案未获执行验证，C缺项目参数仍返回ready，负对照失败。诊断脚本exit0只表示脚本结束，不能代替两批实机验收。

[normal007](results/full_requirements_20261028/live/normal_007/run/runtime_result.json)的3次模型回复完全相同，审阅与提交目标使用重复的完成条件，在所有输入前被拒绝；3次规划、3次HTTP、0输入、0未知，实际子退出码1。接下来只修订通用提示与错误反馈，让模型合并共享完成条件的相关操作并保留全部要求，不降低检查门槛，也不在本地写入实验室步骤。

`scripts/week6_local_lab.py` 只修改自身离线测试状态，提供项目、主题、审阅、本地提交四阶段。独立状态和事件仅用于验收对照，不作为模型输入：

```powershell
python scripts/week6_local_lab.py --directory outputs/week6/lab_fixture_new --mode normal
```

下表仅为 **v2.0 rolling_actions.v2 历史验收**，不验证 v2.1.0 语义序列：

| 历史批次 | 条件 | HTTP / 规划 / 输入 | 运行秒 | 当时严格结果 |
| --- | --- | --- | --- | --- |
| [正常017](results/live_auto_20261026_normal_017/acceptance.json) | ALPHA GREEN | 4 / 4 / 4 | 652.750 | 全部通过 |
| [瞬态015](results/live_auto_20261025_transient_015/acceptance.json) | BETA BLUE，一次审阅拒绝 | 5 / 5 / 5 | 275.703 | 全部通过 |

瞬态拒绝来自应用审阅层，两批所有 ActionResult 均 success、已知执行器失败0；执行器失败分支另有 CPU 场景证据。两批运行器、独立终态、不同输入回执归因、源码前后未变和真实 owner exit0 共同核验，均为 `executed_caller_goal`、0未知派发、完整计数。瞬态原 step 诊断4 passed、1 failed；提交预测未出现的 failed 原样保留。当时这些通过不代表模型生成的语义目标通过。

历史验收使用本机 qwen2.5vl:7b、实际截图、CPU OCR 和生产规划/运行器，输入走限制于专用窗口的 Sky 桥接；400 ms 跨页防重入属于实验环境。两组固定目标不构成任意应用成功率，也不替代普通 CLI/PyAutoGUI 的跨应用验证。分日经过时间包含激活和 operator 桥接等待，不据此作恢复速度的因果比较。

历史001至011的输出、环境和归因失败保留；012只在旧 v1 源码下通过；013受控停止真实 exit15，不补完整 acceptance。014有2个已知输入和1未知派发；016虽独立终态通过，但提交无对应输入回执，派发前又激活超时，整体 failed。缺失输入、摘要和退出码保持原样，不拼接开发成功率。

## 当前计划进度与日志

语义模式根监控使用独立 schema `week6.run_status.semantic.v1`，记录当前语义目标、累计已验证完成数和当前计划总数。计划修订时，总数为“已验证完成 + 新的剩余目标”，可在分解事件中调整，不能低于已完成数；旧失败目标继续保留在台账，不能从历史中消失。旧 `week6.run_status.v1` 固定总数行为保持兼容。更新由阶段、输入和检查事件驱动，不承诺每毫秒刷新。

```powershell
python -S scripts/week6_status.py --run-dir outputs/week6/your_run_id
python -S scripts/week6_status.py --run-dir outputs/week6/your_run_id --watch --interval 1
```

只读工具仅用标准库，区分快照/日志矛盾、缺失、损坏、过期、进程消失和 PID 复用。`finished` 表示终态记录可信，成功与否仍看 task_status；长模型调用期间的过期不能当成退出。旧 rolling 根状态主要反映生命周期，动作明细仍在其 `workflow/001_CALLER_GOAL`，不能以旧根动作0推断子任务无输入。

语义记录包括 `semantic_request.json`、每轮 `native_proposal.json` 与 `compiled_workflow.json`、输入前的 `goal_contract_*.json`、`semantic_goal_ledger.json`、逐目标 `semantic_goal_audit.json`、完整原生动作计划和 `automatic_workflow_summary.json`。分解原回复、格式错误、修订原因、原始动作诊断与最终检查分开保留。

v5 的高层 `protocol.json`、每次分解请求和结果记录意图模态、未附图及snapshot未发送标志；`snapshot_archive.json`、原像素副本和 `context_source_archive.json` 单独保存出处。`native_proposal.json` 与 `compiled_workflow.json` 保留原生指令、所选调用方谓词和编译上下文的对应关系；低层计划仍保存当次图片参数。真实传输日志没有单独捕获wire请求体时，不声称逐字节核对过真实wire；SDK/MockTransport的请求体测试另有CPU证据。v1至v4的超时、冗长回复和误澄清按各自源码及原生回复保留，通路故障与模型效果分别解释。

`dispatch_intents.jsonl` 是意图，不证明操作已发生。执行器进入后缺完整结果记 unknown 并停止自动重放；派发前激活失败可能保持 unknown=0。动作计数、真实 HTTP、窗口观察、固定目标判定、调用方终态及 owner 真实退出分开审计，未知记录不填0。

## 回归与交付范围

v5首轮的[full003](results/full_requirements_20261028/full_validation_003/validation.json)已完成3234项通过，pytest报告761.44秒，拥有pytest子进程的运行器等待766.360秒，完整验证运行器耗时766.968秒；pytest与Ruff实际退出码均为0。[只读复核](results/full_requirements_20261028/full_validation_003/root_audit_001.json)确认当时256份代码的测试前后、v5首轮冻结及复核时字节一致；`README.tested.md`保留测试时根README原件。后续提示源码改变后必须重新做完整回归，真实窗口验收与发布清单也仍待完成。复查命令为：

```powershell
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
python -m pytest -q
python -m ruff check src tests scripts
```

**历史 v2.0 full006** 为3027项通过、0失败，pytest535.12秒、封装墙钟566.027秒、真实 exit0；其244份文件 before、after 和当时 Git 字节一致。原件见 [2026年10月26日 full006](results/release_validation_completed_20261026_006/validation.json)，不称为当前 v2.1.0 suite 结果。[11场景218项故障模拟](results/robustness_20261024_attempt005/summary.json)、保存图规划和真实窗口结果分别解释。

v4的[235项定向检查](results/full_requirements_20261028/semantic_sequential_prompt_001/cpu_validation_001/validation.json)及[full002](results/full_requirements_20261028/full_validation_002/validation.json)保留为同版历史：full002为3184项通过、pytest634.52秒，pytest/Ruff实际退出码0，255份代码前后相同。该结论不能转移到已修改的v5。[full001](results/full_requirements_20261028/full_validation_001/validation.json)的1失败、1225通过、1958项设置错误也保留；设置错误来自验证脚本漏建basetemp父目录，另1项为README数量声明断言，不归作模型质量零分。

三份更新前说明已按字节保存在 [pre_full_requirements_20261028](history/pre_full_requirements_20261028/)，附 SHA256 清单。[第一阶段](Week6_Phase1_Test_Report.md)、[初始使用说明](history/v2_initial_20261024/README.md)、[旧实验报告](Week6_Robustness_Report.md) 和旧 DOCX/PDF、源码包均保留其原版本范围。v2.1.0 的报告和分发文件须待本轮验收后另行更新；现有旧报告不当作新版本交付证明。
