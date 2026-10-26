# Week 6 只读发布审查材料

这是已完成审查材料的原字节归档，不是正式完成记录。`copy_manifest_001.json` 列出每个原路径、归档路径、字节数及两端 SHA256；大批原图、训练或桌面日志不在此重复复制，仍由原实验目录保留。

材料范围：

- `collection/`：2026-10-25 的 CPU 收集原 stdout 和 JSON，真实结果为 3027 tests collected，exit 0；这不是 3027 项测试通过。原命令通过内联 Python 执行，当时没有单独保存的 harness 脚本。JSON 中的 `command` 是原执行参数；本文和字节验证程序是事后准备，不能冒充原 harness。
- `independent/`：冻结源码 `706d98f6156f01224c3fcb99b971623553f91e732249818c029b46665efd6940` 的 23 项独立 CPU 反例。使用实际 TaskRunner、虚构观察和离线执行器；实际模型、HTTP、截图、系统输入均为 0，不能证明桌面任务成功。原脚本按字节保存为 `.py.txt`。
- `gate/`：集中只读门槛快照 `c0ce78004529b3624030b5a0f65401d73b887012177816088f586339720372c6`、实际 preflight 的 11 项拒绝检查及原脚本。10 个无效选择只在内存构造，未写入有效完成记录；另 1 个为真实历史待完成记录。
- `positive/`：已结束的 transient015 和 normal017 集中审查原件。它们分别核验真实原 acceptance、退出收据、模型请求、已确认输入、语义转换及独立最终规则，不构成任意应用成功率。
- `negative/`：014 桥接超时并有未知派发，以及 016 终态转换未归因的独立审计和原构建脚本。两者都不计为成功；016 的独立终态通过不能代替运行器或归因通过。
- `observations/`：2026-10-26 full006 的被动进程身份观察。实际测试 CPython、venv launcher 与 harness 分别记录。此快照只证明观察当时的身份，不证明退出、测试通过或完整回归完成。

rolling v2 采用固定调用方整体目标下的动作级滚动规划，每次模型计划最多一步。它未实现层次化语义子任务生成或依赖管理。原模型 step 预测的 failed/inconclusive 保持原值；最终结果由不变调用方规则及独立新帧核验。所有前代失败仍由原目录保留。

只读核验归档字节（普通 Python、无需项目依赖）：

```powershell
python verify_bundle.py.txt
```

在原工作区还可同时复核所有原文件仍与副本相同：

```powershell
python verify_bundle.py.txt --check-original-sources
```

使用原项目 Python 对已结束的 015/017 重跑同一集中门槛（只读取原实验，不创建模型、GUI、输入或测试）：

```powershell
& 'D:/Developer/multimodal-desktop-gui-agent/.venv-week5/Scripts/python.exe' verify_bundle.py.txt --recheck-selected-gates
```

以上程序不会生成报告、打包、修改原件，也不会宣称 full006 已完成。项目依赖和原实验绝对路径须仍存在；分发后的副本可单独核验归档字节，原路径重审须在原工作区进行。原 CPU 反例和拒绝检查脚本保留其原路径约定，归档目录不是它们的执行目录。

完整交付还需实际 full006 的真实退出审计、前后及当前源码 SHA 一致、新报告逐页实际查看、干净提交与 wheel/ZIP 字节核验。它们不由本材料提前宣告。
