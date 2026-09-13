# 第一周实验报告：GUI 智能体技术调研与开发环境搭建

## 1. 基本信息

- 项目名称：基于多模态大模型的桌面 GUI 智能体开发与优化
- 实验阶段：Week 1
- 实验周期：2026-09-08 至 2026-09-14
- 本周主题：技术调研、开发环境搭建与基础工具验证
- 计划投入时间：15 小时

## 2. 实验目标

本周的主要目标是了解桌面 GUI 智能体的技术路线，完成基础开发环境搭建，并验证后续开发所需的屏幕感知、图像处理、文字识别及桌面控制工具。具体目标如下：

1. 调研 UI-TARS、Claude Computer Use 和 ScreenAgent 等代表性 GUI 智能体方案。
2. 总结 GUI 智能体的核心架构、关键模块与主要技术挑战。
3. 在 MacBook 上搭建 Python、PyTorch、Git 和 VS Code 开发环境。
4. 验证屏幕截图、OCR、OpenCV 图像处理、鼠标与键盘控制等基础功能。
5. 制定 MacBook 与 Windows GPU 设备协同开发方案。

## 3. 实验环境

| 项目 | 配置 |
| --- | --- |
| 开发设备 | MacBook Air M2，24 GB 统一内存 |
| 操作系统 | macOS 26.3.1，Apple Silicon ARM64 |
| 开发工具 | Visual Studio Code、GitHub Desktop |
| Python | 3.12.10 |
| 虚拟环境 | Python venv（`.venv`） |
| 深度学习框架 | PyTorch 2.14.0 |
| Mac 加速后端 | MPS |
| OCR 工具 | Tesseract 5.5.3、pytesseract |
| 图像与桌面工具 | MSS、Pillow、OpenCV、PyAutoGUI、Pynput |
| 测试与代码检查 | Pytest、Ruff |
| 版本控制 | Git、GitHub |

MacBook 不支持 NVIDIA CUDA，因此本周在 Mac 端使用 PyTorch MPS 完成加速验证。Windows Y9000P 将作为后续 CUDA 推理、模型微调和批量评测设备，其 CUDA 环境将在实际使用该设备时单独配置和验证。

## 4. 技术调研结果

### 4.1 GUI 智能体基本流程

通过对相关论文、技术文档和开源项目的调研，将桌面 GUI 智能体的基本运行流程归纳为：

```text
用户指令 → 屏幕感知 → 任务拆解与规划 → 动作生成
        → 鼠标键盘执行 → 结果反馈 → 错误检测与重试 → 日志记录
```

系统需要持续获取桌面状态，由多模态模型理解当前界面并生成动作，再由本地控制模块执行动作，最后根据新截图判断任务是否成功或是否需要调整计划。

### 4.2 代表性方案

- UI-TARS：重点关注基于截图的界面理解、坐标定位、统一动作表示和原生 GUI 操作能力。
- Claude Computer Use：重点关注模型与本地计算机工具之间的循环调用方式，以及敏感操作确认、环境隔离等安全机制。
- ScreenAgent：重点关注任务规划、动作执行、结果反思和交互轨迹记录。

### 4.3 关键技术挑战

- 准确识别文字、按钮、输入框等 UI 元素，并获得可靠坐标。
- 将自然语言任务拆解为可执行、可验证的桌面操作步骤。
- 适配不同操作系统、应用程序、窗口尺寸和屏幕分辨率。
- 处理界面加载、弹窗、操作失败和任务状态变化。
- 防止智能体执行未经授权或具有风险的桌面操作。

完整调研内容见 [Week1_GUI_Agent_Technical_Research.md](./Week1_GUI_Agent_Technical_Research.md)。

## 5. 开发环境与项目初始化

本周完成了以下环境与工程配置：

1. 将 GitHub 仓库克隆到 MacBook，并建立 GitHub Desktop 与 VS Code 协同工作流程。
2. 创建项目专用 `.venv` 虚拟环境，并在 VS Code 中选择对应 Python 解释器。
3. 安装 PyTorch、OpenCV、MSS、Pillow、Tesseract OCR、PyAutoGUI、Pynput、Pytest 和 Ruff 等依赖。
4. 创建 `src/gui_agent`、`tests`、`scripts` 和 `configs` 等基础目录。
5. 创建通用、开发及 Mac 端依赖清单，并配置 `.env.example`、`.gitignore` 和 `.gitattributes`。
6. 创建环境检测、屏幕感知、图像处理和桌面控制测试脚本。

详细部署步骤见 [Week1_Development_Environment_and_Dual_Device_Workflow.md](./Week1_Development_Environment_and_Dual_Device_Workflow.md)。

## 6. 实验过程与结果

### 6.1 屏幕截图与 OCR 测试

运行命令：

```bash
python scripts/smoke_test_perception.py
```

测试成功截取 `1920 × 1080` 桌面图像，并保存至 `outputs/week1/screenshot_test.png`。Tesseract 能够从截图中识别部分英文界面文字，证明“屏幕截图 → 图像转换 → OCR”基础链路可以正常运行。

由于当前测试直接识别整张屏幕，且只安装了英文语言数据，识别结果存在少量乱码。后续可通过界面区域裁剪、图像缩放、二值化及补充中文 OCR 模型提高准确率。

### 6.2 OpenCV 图像处理测试

运行命令：

```bash
python scripts/smoke_test_image_processing.py
```

测试成功读取桌面截图，完成灰度化、高斯模糊和 Otsu 二值化，并生成：

- `outputs/week1/screenshot_gray.png`
- `outputs/week1/screenshot_binary.png`

实验表明 OpenCV 图像读取、处理和保存功能运行正常，可用于后续 OCR 预处理和 UI 元素检测。

### 6.3 鼠标与键盘控制测试

首先运行只读检查：

```bash
python scripts/smoke_test_control.py
```

首次检测发现 macOS 辅助功能权限未开启。为 Visual Studio Code 开启辅助功能权限并重启后，权限状态变为 `True`。

随后运行实际控制测试：

```bash
python scripts/smoke_test_control.py --execute
```

脚本完成了一次约 40 像素的可逆鼠标移动，并按下和释放一次 `Shift` 键。测试过程中未执行点击或输入文字，鼠标和键盘事件均成功发送。

## 7. 本周交付物

- GUI 智能体技术调研报告。
- 文献和相关 GitHub 项目索引。
- Mac 与 Windows 双设备开发方案及环境配置文档。
- 项目基础目录、依赖清单与配置文件。
- 环境检测、截图/OCR、图像处理和桌面控制测试脚本。
- 第一周工作日志及本实验报告。
- GitHub 项目仓库。

## 8. 实验结论

本周完成了 GUI 智能体核心技术路线调研，并在 MacBook Air M2 上建立了可复现的 Python 开发环境。PyTorch MPS、屏幕截图、英文 OCR、OpenCV 图像处理以及鼠标键盘控制均通过基础测试，项目代码结构和双设备协同方案也已建立。因此，当前环境能够支持下一阶段的桌面感知与控制模块开发。
