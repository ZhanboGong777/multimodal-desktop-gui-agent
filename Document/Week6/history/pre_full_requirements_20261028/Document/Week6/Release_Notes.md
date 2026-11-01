# 系统 v2.0.0 发布说明

日期：2026年10月26日 Australia/Sydney。开发分支：`codex/week6-delivery`。本次按受限工程范围整理Week6交付，完整CPU回归为3027项通过、0失败，pytest 535.12秒，封装墙钟566.027秒，真实exit 0；244份文件的before、after与当前SHA完全一致。本版按上述受限工程范围闭合实现与证据；报告和分发文件按独立生成与发布清单核验。

## 当前能力与范围

CLI整体指令入口默认迁移到 `rolling_actions`。模型依据整体调用方请求与当前截图逐动作在线规划，每次计划硬限 `max_steps=1`；输入后重新观察，最终文字规则由调用方保有。`CALLER_GOAL`是本地容器，层次化语义子任务生成与依赖管理未实现。Python API默认仍保留 `observed_departure`；固定 `--workflow` 有序阶段入口和严格检查行为不变。

`week6.rolling_actions.v2` 在输入前声明 `gate_on_step_verification=false`。模型自写的step预测只作原始诊断，failed与inconclusive均保留；新帧调用方规则与另一张独立final帧决定任务完成。实际执行失败、未知派发、取消和超时继续按原运行器分支处理。预算覆盖动作、规划、实际HTTP与截止时间，恢复不重置额度。

重规划读取当前帧和结构化失败反馈；进入执行器后无完整结果时停止自动重放。状态快照、事件、输入意图和真实终态分别记录，只读状态工具核验进程身份与数据可信度。逐输入详情位于 `workflow/001_CALLER_GOAL`，根状态主要反映整体生命周期。

双语screen OCR保留现成英中语言资源与Apache 2.0许可证。冻结6帧留出中识别46/48，精确率、召回率、F1均95.83%；同双语native为30/48，但screen整帧中位耗时增加9.75%。局部轮廓提取配对中位14.421→12.185 ms，未证明整个智能体普遍加速。错误与真实英文保存图无增益结果均保留。

## 真实验证与历史失败

同份冻结源下，正常017完成ALPHA/GREEN，4次HTTP、4规划、4输入、652.750秒；瞬态015完成BETA/BLUE，首次审阅拒绝后恢复，5次HTTP、5规划、5输入、275.703秒。两批运行器、独立终态、不同输入回执逐项归因、源码未变和真实owner exit0共同通过，未知派发0。瞬态提交后的模型step预测仍failed，原始记录未改写，调用方final通过。

瞬态拒绝发生在应用审阅层，两批已知执行器失败数均为0；原step failed不等于执行器失败，执行器失败恢复分支另以CPU场景解释。

这两个受控专用窗口案例不构成任意应用成功率；经过时间包含截屏、OCR、模型和operator桥接等待，不能据分日耗时比较恢复性能。验收使用限制于专用窗口的Sky桥接，不能替代普通PyAutoGUI执行器的跨应用验证。

旧开发尝试、保存图探针和源码阶段全部保留。012是旧v1源码下的正常通过；013受控停止真实exit15且缺完整acceptance；014有2个已知输入和1未知派发；016独立终态通过但提交无输入归因，随后激活超时，整体失败。旧阶段CPU通过不拼接为新源码全量通过，也不合并开发成功率。

完整CPU回归为3027项通过、0失败，pytest 535.12秒，封装墙钟566.027秒，真实exit 0；244份文件的before、after与当前SHA完全一致。

## 从源码安装与首次检查

声明支持Python 3.11至3.13，本次验证为Windows和Python 3.12。其他操作系统未形成相同范围的实测记录。在源码包根目录创建环境：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install -e .
python -c "import gui_agent; print(gui_agent.__version__)"
```

版本输出应为2.0.0。wheel元数据没有完整运行依赖，单装wheel不会自动安装requirements.txt；CLI、配置及实验记录从完整源码使用。本次从已有项目环境构建和检查wheel，不声明干净依赖环境安装已经验证。

真实OCR需要Tesseract可执行文件及语言数据；双语资源与Windows路径设置见 [使用说明](README.md)。默认工作流不要求训练框架；数据集或LangChain功能另见requirements-agent.txt。

```powershell
python scripts/week6_agent_cli.py --instruction "Complete the local task" --expect-text "Expected final marker" --validate-only
python scripts/week6_robustness_demo.py --output-dir outputs/week6_first_check
python -S scripts/week6_status.py --run-dir outputs/week6/your_run_id
```

输入验证不调用模型或桌面；故障演示使用模拟后端；状态命令只读指定目录。默认dry-run会观察和规划，真实输入需加execute并在交互终端逐计划确认。

## 保留与后续评估

Week5 C11权重、原训练、质量及资源补测结论不变，本周不新增训练或消息发送。Week7的综合任务集、跨应用、分辨率及竞品比较另行开展。层次化语义计划、依赖管理、广泛应用质量和一般任务成功率尚未实现或未测。

原报告、DOCX/PDF、源码包及初始发布说明保留为历史。更新结果与具体证据见 [实验报告](Week6_Robustness_Report.md)，最终源码/报告/wheel/压缩包字节由交付审计单独核验。
