# Week 6 真实本地模型恢复规划探测

2026-10-24（UTC 04:11；Australia/Sydney 15:11）。本次使用项目的 OpenAICompatibleClient、TaskPlanner 和 ActionAdapter，对保存的合成界面执行真实模型规划与目标解析。没有运行桌面执行器，没有派发鼠标或键盘输入。

## 固定设置与全部结果

- Ollama：`http://127.0.0.1:11434/v1`；唯一模型 `qwen2.5vl:3b`，Q4_K_M，digest `fb90415cde1ef08aa669ae74b082d49b158729b6db1ab183c941417d507e71a1`。
- 使用现有系统提示，temperature 0、max_tokens 2048、运输重试 0；每条件只调用一次规划器，保留其原有最多一次格式重试。每条件整体时限 120 秒。未修改提示、重跑条件或更换模型。
- 三张 800×520 英文界面由 Pillow 绘制并逐张查看；当前元素来自手工 ground truth，未运行 OCR；ground truth 期望行为未发送给模型。

| 预先固定条件 | 实际推理请求 | 规划调用经过时间 | 当前目标对照 |
|---|---:|---:|---|
| 唯一 Run analysis 按钮 | 1 | 17,284.296 ms | 通过；当前 `obs-0001-e000`，解析坐标 (220,318) |
| target_resolution 反馈后按钮移动 | 2 | 3,018.992 ms | 通过；当前文字唯一匹配到 `obs-0002-e001`，解析坐标 (595,418) |
| 两个同名 Run analysis 按钮 | 2 | 2,707.094 ms | **失败**；模型没有请求澄清，选择左侧 `obs-0003-e000` |

合计 3 次规划器调用、5 次真实推理 HTTP 请求。读取模型目录及驻留状态的本地元数据 GET 不计入推理请求数。第一次调用包含客户端初始化和当时的服务器状态开销；这些数值不是纯 GPU 计算时间，也不是稳态推理延迟分布。

恢复条件和歧义条件的首个回复均使用了 schema 不接受的 `status="success"`，被现有解析器拒绝；其内部格式重试结果和原文全部保留。第二次恢复回复没有提供 element_id，ActionAdapter 使用当前帧唯一文字匹配解析，而非复用原坐标。所有最终计划均没有旧 frame id 或显式坐标参数；也都省略了提示要求的最后 finish 标记。因此“2/3”仅指本次固定的目标对照，不能解释为全部提示协议遵循率。

## 失败与适用范围

歧义负例揭示了真实限制：有效的当前 element_id 只能说明候选存在，不能说明语义已经消歧。ActionAdapter 能解析模型选中的左按钮，本次语义对照仍判失败。原始失败未被隐藏，也未通过追加试验替代。

这是“真实本地模型 + 保存的合成界面图”的规划探测，不是真实桌面复杂任务，也没有执行后成功验证。三个小型英文界面不能代表真实应用、中文 OCR、长任务或真实错误恢复分布。恢复反馈由脚本注入，未实际先执行一次失败。使用的是 Ollama 已安装基础模型，与 Week 5 已接受的 C11 LoRA 分开；本次不读取、加载或修改该权重。

## 证据与复现

`summary.json` 保存完整条件结果、模型 digest、提示及源码哈希；`input_01.json` 至 `input_03.json` 保存输入及 ground truth；`request_001.json` 至 `request_005.json` 保存全部 SDK 原始回复和独立解析检查；`condition_01.json` 至 `condition_03.json` 保存规划器实际解析结果及当前帧解析对照；`frames/` 保存三张原图。`artifact_audit.json` 保存证据完整性审计和实际逐图检查记录。所用相关源码在试验前后哈希一致。

复现命令（必须使用新的输出目录；已有目录被拒绝）：

```powershell
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:NO_PROXY = "127.0.0.1,localhost"
python scripts/week6_model_recovery_probe.py --output Document/Week6/results/model_recovery_new_run
```

该命令会发起新模型请求；本次结果完成后没有再次运行它。
