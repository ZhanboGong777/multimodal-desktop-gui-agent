# Week 6 使用说明

系统版本为 `2.1.0`，分支为 `codex/week6-full-requirements`。同一286份源码已通过正常与故障恢复两组受控窗口验收。完整回归3803项通过，pytest 664.89秒，pytest与Ruff实际退出码均为0。实际过程和限制见报告及原始记录。

## 任务分解与执行

自然语言 CLI 默认使用 `semantic_subtasks`，协议为 `week6.semantic_subtasks.v6`。用户提供完整任务和最终文字规则；高层模型生成至少两个有顺序、含义不同的目标。程序保留原生指令与条件，不在本地补写实验室步骤。Python API 默认仍为 `observed_departure`，使用新路径时须显式传入 `protocol_mode="semantic_subtasks"`。

高层分解接收任务、不可变的最终条件及有限的逻辑反馈，不接收当前截图、OCR、坐标或窗口资料。动作规划层则使用每次输入前的新截图定位控件。原始图像和观察记录仍保存，供运行检查和复查。

模型预测的阶段文字不是可信事实。整个阶段条件在输入前必须尚未成立，完成计数还须有已确认输入及后续新观察支持。用户最终条件独立检查。文字验证采用大小写不敏感的子串匹配，不能理解通用否定语义或证明操作因果；真实验收另外核对原生目标、输入、应用事件与前后画面。

目标条件在操作前已成立时，程序拒绝该目标的输入和完成计数，在原预算内请求修订剩余工作。已执行目标因条件未满足而失败时，下一轮只收到原生目标和原条件中缺失、禁止出现的文字；核对信息不全时保留 `null`。原失败状态和条件不改写，后来出现终点文字也不补记旧目标成功。已确认完成的目标不自动重放。

动作层校验目标文字与控件 ID 在原帧及刷新帧中的一致性，不自动替换模型给出的 ID。无 ID 查询只在当前标签及有限词组中生成；无可读 OCR 的图形候选保留视觉定位路径。未知输入、取消和计数不完整会停止推进。格式修正、分解、动作规划及恢复共用请求、动作和时间预算。

## 安装

支持Python3.11至3.13，本轮实际使用3.12。先在源码根目录创建虚拟环境并安装requirements.txt，再安装随附wheel。wheel元数据不自动安装全部运行依赖；CLI、配置和实验材料从源码目录使用。分发检查使用已有依赖环境，不能代替干净机器的全依赖安装。

```powershell
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt
.venv/Scripts/python.exe -m pip install path/to/multimodal_desktop_gui_agent-2.1.0-py3-none-any.whl
```

## 首次运行

先在源码根目录检查输入。`--validate-only` 不调用模型、截屏或输入，也不创建运行目录：

```powershell
python scripts/week6_agent_cli.py --instruction "Prepare the GREEN report for project ALPHA, review it and commit it locally in Week6 Local Lab" --expect-text "LOCAL COMPLETE" --expect-text "Project ALPHA" --expect-text "Theme GREEN" --validate-only
python scripts/week6_agent_cli.py --workflow configs/week6_example_workflow.json --validate-only
```

实际模型依赖本机 Ollama 服务和 `qwen3-vl:8b-instruct`。本轮固定 digest 为 `70e58e37beb15464e19aa903dedb2cba828b89c0fb186f75ff2738c7347b96eb`；复现时应核对完整 digest，标签本身不能保证字节不变。任务分解和动作规划均使用该模型；`--text-model`可显式指定任务分解模型，两层共用预算。模型权重不在 wheel 或源码包中，也不是 Week 5 C11 adapter。

```powershell
ollama pull qwen3-vl:8b-instruct
$env:GUI_AGENT_API_KEY = "ollama-local"
python scripts/week6_agent_cli.py --config configs/week6_role_models.yaml --instruction "Prepare the GREEN report for project ALPHA, review it and commit it locally in Week6 Local Lab" --expect-text "LOCAL COMPLETE" --expect-text "Project ALPHA" --expect-text "Theme GREEN" --provider openai_compatible --base-url http://127.0.0.1:11434/v1 --model qwen3-vl:8b-instruct --text-model qwen3-vl:8b-instruct --planning-attempts 24 --model-requests 32 --max-actions 20 --task-timeout 7200
```

这条命令默认为 dry-run，仍会观察界面并调用模型。真实输入需添加 `--execute`，并在交互终端逐计划确认，没有 `--yes`。CLI 的生产输入后端与本轮专用窗口 Sky 桥接验收是两个派发环境。既有输出目录不能覆盖，旧日志不会自动触发恢复。

`--workflow` 是调用方预写的有序流程入口，与 `--instruction` 互斥。旧 `observed_departure`、`rolling_actions` 保留兼容用途；后者的逐动作规划不等于模型生成语义目标序列。

## OCR 环境与测量范围

`configs/week6_accuracy_persistent.yaml` 显式启用持久化 Tesseract C API，采用英中资源、screen 处理、2 倍缩放和 0.5 置信度阈值。每帧结束清理识别结果与自适应状态。默认配置仍为 `configs/week4.yaml`，未安装 DLL 的机器不会默认进入持久化路径。

Windows 示例依赖 `C:/Program Files/Tesseract-OCR/libtesseract-5.dll`、其依赖 DLL 及 `D:/week6-full-ocr/tessdata` 中的固定语言文件。换机器须修改配置中的真实路径。缺少文件或 C API 函数时明确失败，不静默换后端。语言文件、来源与许可证保存在 `Document/Week6/resources/tesseract/`，系统 DLL 不随包分发。

新留出含 8 张完整标注合成画面、64 个文字标签和 48 个矩形。候选文字 F1 为 93.6508%；持久化路径的两批预热完整帧中位数分别为 108.291、109.888 ms。完整帧口径从 PNG 解码到预处理、OCR 及坐标/轮廓输出，不含系统截屏、模型和输入。冷启动仍比英文基线慢。另有一张真实保存图的配对测量，其标注区域各方案均识别 5/5 标签、2/2 按钮，没有准确率提升；持久化路径也不普遍快于英文路径。

## 状态与证据

```powershell
python -S scripts/week6_status.py --run-dir outputs/week6/your_run_id
python -S scripts/week6_status.py --run-dir outputs/week6/your_run_id --watch --interval 1
```

状态包含当前目标、累计已验证目标数、当前计划总数、动作与规划用量。顶层转发活动子任务的计数，完成数只来自外层验证台账。更新由事件触发；长模型等待可能显示 stale，不能因此推断进程退出。

复查时同时阅读原生方案、固定目标条件、逐目标台账、输入回执、应用事件、最终观察和真实进程退出记录。脚本正常退出、记录格式通过、目标成立和实际业务操作完成分别判断。历史失败、无效条件、未知输入及未通过审计的终点画面全部保留，不与后续尝试拼接成成功。

本轮不包含 Week 7 的跨应用、跨分辨率或竞品综合评估，不新增训练或外部消息发送。两组受控任务的通过不能推导任意应用成功率。

## 可选图形候选细节

configs/week6_role_models.yaml显式开启planning.visual_candidate_details，默认PlanningConfig仍为false。该路径在完整原图后附上当前无OCR匿名轮廓的原像素裁图，最多8张、总面积100万像素，不补写标签。可读控件必须在原生回复中提供非空target_text，刷新时同时核对文字与ID。附图处理和模型推理开销不在原OCR基准耗时中，实际任务墙钟已包含这些开销。

## 本轮实际验收

| 场景 | 已确认输入 | 规划 | HTTP | 未确认输入 | 失败目标尝试 | 墙钟秒 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| normal_048 | 4 | 5 | 5 | 0 | 0 | 385.157 |
| transient_027 | 5 | 6 | 6 | 0 | 0 | 366.703 |

[正常场景](results/full_requirements_20261028/review/normal048_root_acceptance_003.json)与[恢复场景](results/full_requirements_20261028/review/transient027_root_acceptance_003.json)使用相同冻结源码和模型digest。恢复场景实际经历应用审阅拒绝和重新观察后的重试，再完成本地提交。两组均为一个高层分解轮次、两个成功目标，无失败语义目标；一次审阅拒绝和重试发生在第二目标内。正常场景第二步辅助时效检查超限后仍输入的操作偏差已披露。动作提示性检查与真实应用结果分别保留。原生目标、实际输入、前后画面和应用事件由Codex代理核对，不代表独立人工签字。

[完整回归](results/full_requirements_20261028/full_validation_007/root_audit_001.json)、[冻结记录](results/full_requirements_20261028/auditor_prompt_sync_001/final_source_freeze_001.json)、[报告](Week6_Robustness_Report.md)与[开发尝试索引](results/full_requirements_20261028/live_attempts_index_016.json)分别保存。旧文档见[历史清单](history/pre_final_acceptance_20261101_016/manifest.json)。
