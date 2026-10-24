# 系统 v2.0.0 发布说明

日期：2026-10-24（Australia/Sydney）。开发分支：`codex/week6-delivery`。
最终全量检查：`2840 项通过，0 失败，pytest 443.05 秒，退出码 0`；原始验证记录：`results/release_validation_20261024_attempt004/validation.json`。

## 本版变化

- 调用方定义有序子目标和独立总体文字检查，各阶段共用动作、规划调用及时间预算。已验证阶段不重放，旧日志不能触发自动恢复执行。
- 重规划获得结构化失败反馈；执行异常产生未知派发记录时停止自动重试，确认动作数只作为下界。
- 感知记录各处理阶段耗时，并优化候选框去重。配对测量保留完整候选与计时；没有证明整个观察流程普遍加速。
- 新增运行状态快照、事件日志及只读状态工具。Windows 原子替换遇短暂 PermissionError 时仅有限重试文件操作，不重复动作或事件追加。

工作流本版需要显式文字检查，不接受消息工作流；总体检查不支持进程规则。原单任务执行边界继续保留。Week 5 的 C11 权重、历史训练与资源结果独立保留，本版不追加训练或消息发送。

## 从源码安装

支持范围由 `pyproject.toml` 声明为 Python 3.11–3.13；本次验证环境是 Windows、Python 3.12。其他操作系统尚未形成同等的本版验证记录。

在源码包根目录创建环境并安装：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install -e .
python -c "import gui_agent; print(gui_agent.__version__)"
```

版本输出应为 `2.0.0`。项目尚未将运行依赖写入 wheel 的安装元数据，因此只安装 wheel 不会自动安装 `requirements.txt`。CLI、配置、示例页面和实验记录从完整源码包使用；本版没有声明独立的已安装命令入口。

真实 OCR 还需要本机 Tesseract 可执行文件及所需语言包。当前测量使用英文配置，本机没有中文语言数据，不能据此声称支持中文界面。默认工作流不要求安装训练框架；数据集或 LangChain 相关功能另见 `requirements-agent.txt`。

## 首次检查与演示

```powershell
python scripts/week6_agent_cli.py --workflow configs/week6_example_workflow.json --validate-only
python scripts/week6_robustness_demo.py --output-dir outputs/week6_first_check
```

第一条只验证工作流定义；第二条在模拟后端上运行真实生产运行器，不操作桌面、不加载模型。输出目录必须是新的。示例页面和真实执行命令见 [使用说明](README.md)；默认 dry-run，真实执行仍需要交互终端和各阶段计划确认。

只读检查已有运行状态：

```powershell
python -S scripts/week6_status.py --run-dir outputs/week6/your_run_id
```

该命令只读取指定目录，不启动模型或桌面。`finished` 表示终态记录可信，具体成功与否仍看 `task_status`。

## 回归检查

开发测试需要 `requirements-dev.txt`；涉及数据集、LangChain 等可选模块的检查还需要对应依赖。完整数量以记录的环境为准，缺少可选模块产生的跳过不能当作完整全量通过。

```powershell
python -m pip install -r requirements-dev.txt -r requirements-agent.txt
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
python -m pytest -q
python -m ruff check .
```

## 已知限制

真实本地模型探测只有三个固定合成界面条件。两个目标对照正确，但双同名按钮未请求澄清；三个最终计划均缺少提示要求的 `finish`。当前 ID 有效不代表语义已消歧。该结果不是真实桌面任务成功率，也不证明模型质量提升。

本轮没有真实应用中的复杂任务执行结果。故障模拟、固定图像 OCR/候选测量与本地模型保存图规划分别解释，详情及保留的失败见 [鲁棒性报告](Week6_Robustness_Report.md)。
