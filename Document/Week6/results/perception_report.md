# Week 6 感知性能与保真检查

2026 年 10 月 24 日，在 `codex/week6-delivery` 分支实测。基线固定为提交
`9149e064d90ab6796e7f25ece6cbbd026785b26c` 的观察服务和轮廓检测代码。
原始源码、四张测试图、ground truth、全部 OCR 输出、候选框和逐次耗时均保留在
`perception_benchmark/`；汇总可由 `perception_analysis.py` 从原始 JSON 重新生成。

基线运行时未另存当时的基准驱动，后续仅做了导入方式和 lint 清理。
现已逆向恢复该次清理前的完整字节，归档到
`perception_benchmark/source_snapshot/baseline_week6_benchmark_perception.py.txt`；
SHA256 为 `0c8162553c8235b08c944a46fe0a10e60ae070783316aa104247aa422c59d9d1`，
与基线 JSON 的原始记录精确一致。恢复过程和原归档缺失情况见
`perception_benchmark/baseline_driver_reconstruction.json`。当前驱动、原始 JSON 和测量结果均未改写，
也未重跑基准；恢复归档的时间不冒充基线执行时已保存的时间。

## 改动及依据

先运行基线，再改代码。基线对密集细边框图做三次 cProfile：总计 0.134 秒，
其中 `_deduplicate` 累计 0.087 秒；这类图的主要 Python 开销是重复计算候选框的包含关系。
cProfile 会增加开销，这组数字只用于定位热点，性能结论采用未剖析的配对计时。

改动限制在当前帧：相同坐标的重复边框只进入后处理一次；去重时每个框的面积和坐标只取一次，
先检查面积条件，再计算相交面积；轮廓集合采用迭代器连接。两个边缘检测分支、细边框的
0.90 矩形度要求、OCR 排除规则、优先区域和候选预算均保留。没有缓存上一帧的目标，
没有缩小图像、跳过 OCR 或放宽操作验证。

`ObservationSnapshot` 新增可选的 `stage_timings_ms`，旧记录缺失时默认为空字典。
每次观察都记录以下不重叠的墙钟时间，单位为毫秒：

| 字段 | 口径 |
|---|---|
| `capture` | `capture_monitor` 整个调用；启用保存时包含保存图片。本次基准替换为 PNG 解码，不代表真实截屏耗时。 |
| `engine_init` | OCR 工厂调用；同一观察服务复用引擎后为 0。后端首次 `recognize` 的延迟加载计入 OCR。 |
| `ocr` | 当前图片的完整 `recognize` 调用，包含后端延迟初始化、推理和结果解析。 |
| `ui_detection` | OCR 排除框整理与当前图片的轮廓提取；关闭时为 0，异常时保留已耗时间。 |
| `selection` | 排序、截断及当前帧 `ElementRef` 构造。 |

`processing_time_ms` 继续保留观察入口到快照构造前的总时间。它包含阶段之间的 Python 开销；
快照模型自身校验和最终 JSON 序列化不在上述阶段计时内。失败仍写入原有 `errors`，
计时不把失败改成成功。引擎工厂失败时 OCR 阶段为 0，表示未调用识别，不表示识别成功。
基线 JSON 的 `instrumented_stage_ms` 来自基准包装器，其中 `selection` 只计 `_select`；
新版快照内建的同名阶段还包括引用构造，两者不直接做阶段加速比较。UI 独立配对则采用相同边界。

## 样本、配置与结果

四张 960×640 的合成界面覆盖浅色、深色、密集控件和低对比度一像素边框。每张含五个英文标签、
两个中文标签和六个已知矩形控件；密集图另含未完整标注的小控件，用于暴露去重成本。
图片、Arial/微软雅黑字体文件的 SHA256 和选择理由保存在 `fixture_manifest.json`。
这些图由脚本生成，未捕获用户桌面，也未加入公开网页截图。

实际运行 Tesseract 5.5.3.20260724 CPU OCR，`languages=[en]`、最低置信度 0.5，未安装或调用 GPU 模型。
本机只安装 `eng` 和 `osd` 语言数据，不能据此声称支持中文。
每个版本运行四图各五次观察；配对候选提取另运行每图 25 次，共 100 组，交替先运行基线或新版。
配对使用该图片实际 OCR 得到的排除框，图片解码不计入 UI 独立耗时。OpenCV 在线程数为 1 的
基准进程内测量，生产配置的默认线程数未改动。

| 合成图 | 基线 UI 中位/ms | 新版 UI 中位/ms | 变化 | 候选数 |
|---|---:|---:|---:|---:|
| light_form | 8.569 | 8.417 | 减少 1.77% | 14 |
| dark_form | 8.003 | 8.220 | 增加 2.71% | 13 |
| dense_dashboard | 22.238 | 16.912 | 减少 23.95% | 122 |
| thin_border | 22.631 | 16.150 | 减少 28.64% | 118 |
| 四图合计，100 次 | 14.421 | 12.185 | 减少 15.51% | 各图逐项一致 |

100 组的输出列表完全一致，包括候选顺序、坐标、中心、来源和置信度。
四图的 24 个已知矩形控件均有 IoU≥0.8 的候选，基线和新版相同。
密集图中的其他候选未全部标注，因此不报告候选精确率。

按各图第一次观察的已知标签计，英文为 20/20，中文为 0/8；五次重复中的结果一致。
标签判定要求在对应 ground-truth 区域内出现完全相同的规范化文字，不采用模糊匹配。
中文失败原样保留，不以英文语言包的结果代替中文识别结果。

跨时段顺序运行的热观察总时间中位数为 221.848→228.817 毫秒，增加约 3.14%；
未证明端到端观察加速。OCR 约占本次观察的主要时间，轮廓优化对简单界面也未稳定提速。
两个版本的首次观察为 515.544 和 281.452 毫秒，首次导入和 OS 缓存状态不同，
不能把这两个数作为引擎初始化优化证据。测试未运行规划器、动作执行、GUI 或消息发送，
这里的标签/矩形覆盖率不等于真实桌面任务成功率。

## 验证与复现

现有 UI 检测、观察服务和新增 Week 6 感知测试合计 **95 passed，0 failed，2.75 秒**。
新增测试覆盖冷/热初始化拆分、阶段失败与禁用、旧快照加载、连续帧不复用坐标、
随机重叠框的旧版去重等价性，以及四张测量图在多种过滤配置下的候选逐项等价性。
原有细边框、开口 L 形、噪点、OCR 排除和前景区域优先测试仍通过。

在仓库根目录，使用现有项目依赖执行：

```powershell
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'
python -X utf8 scripts/week6_benchmark_perception.py --phase baseline
python -X utf8 scripts/week6_benchmark_perception.py --phase optimized
python -X utf8 Document/Week6/results/perception_analysis.py
python -X utf8 -m pytest -q tests/test_week6_perception.py tests/test_ui_detection.py tests/test_runtime_observation.py
```

默认脚本需要本地 Git 基线对象、Windows Arial/微软雅黑字体和可执行的 Tesseract。
实际执行使用 `D:/Developer/multimodal-desktop-gui-agent/.venv-week5/Scripts/python.exe`，
Python 3.12.4、OpenCV 4.10.0、NumPy 2.3.5、Pillow 12.3.0、pytesseract 0.3.13。
重新运行会覆盖对应基准 JSON，原交付结果应先复制到独立目录。
