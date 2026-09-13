# Week 1 Experiment Report

## 1. Basic Information

| Item | Details |
| --- | --- |
| Project | Development and Optimization of a Multimodal LLM-Powered Desktop GUI Agent |
| Stage | Week 1 |
| Period | 7 September 2026 to 13 September 2026 |
| Topic | Technical research, development environment setup, and basic tool verification |
| Planned time | 15 hours |

## 2. Objectives

The main objectives for this week were to understand the technical route of desktop GUI agents, establish the basic development environment, and verify the screen-perception, image-processing, OCR, and desktop-control tools required for later development.

1. Review representative GUI agent approaches, including UI-TARS, Claude Computer Use, and ScreenAgent.
2. Summarize the core architecture, key modules, and main technical challenges of GUI agents.
3. Configure Python, PyTorch, Git, and Visual Studio Code on the MacBook and Windows Y9000P.
4. Verify screenshot capture, OCR, OpenCV image processing, and mouse and keyboard control.
5. Define a collaborative workflow for the MacBook and Windows GPU machine.

## 3. Experimental Environments

### 3.1 Mac Environment

| Item | Configuration |
| --- | --- |
| Device | MacBook Air M2 with 24 GB unified memory |
| Operating system | macOS 26.3.1, Apple Silicon ARM64 |
| Development tools | Visual Studio Code and GitHub Desktop |
| Python | 3.12.10 |
| Virtual environment | Python venv (`.venv`) |
| Deep-learning framework | PyTorch 2.14.0 |
| Acceleration backend | MPS |
| OCR | Tesseract 5.5.3 and pytesseract |
| Image and desktop tools | MSS, Pillow, OpenCV, PyAutoGUI, and Pynput |
| Testing and code quality | Pytest and Ruff |
| Version control | Git and GitHub |

### 3.2 Windows GPU Environment

| Item | Configuration |
| --- | --- |
| Device | Lenovo Legion Y9000P with RTX 4060 Laptop GPU and 8 GB VRAM |
| Operating system | Windows 11 |
| Python | 3.12.4 in the project `.venv` |
| Deep-learning framework | PyTorch 2.14.0+cu126 and torchvision 0.29.0+cu126 |
| CUDA | CUDA Runtime 12.6 and NVIDIA driver 560.94 |
| OCR | Tesseract 5.5.3 and pytesseract |

The MacBook does not support NVIDIA CUDA, so the Mac environment uses PyTorch MPS when available. The Windows Y9000P will handle later CUDA inference, small-scale model fine-tuning, and batch evaluation. Both machines have completed the basic Week 1 environment setup and verification.

## 4. Technical Research Summary

This week covered GUI agent technologies, representative approaches, and key technical issues. See [Week1_GUI_Agent_Technical_Research.md](../Week1_GUI_Agent_Technical_Research.md) for the full report.

## 5. Development Environment and Project Initialization

The following environment and engineering tasks were completed:

1. Cloned the GitHub repository to the MacBook and established a workflow using GitHub Desktop and Visual Studio Code.
2. Created a project-specific `.venv` and selected its Python interpreter in Visual Studio Code.
3. Installed PyTorch, OpenCV, MSS, Pillow, Tesseract OCR, PyAutoGUI, Pynput, Pytest, Ruff, and related dependencies.
4. Created the basic `src/gui_agent`, `tests`, `scripts`, and `configs` directories.
5. Created common, development, and Mac dependency lists and configured `.env.example`, `.gitignore`, and `.gitattributes`.
6. Added environment, screen-perception, image-processing, and desktop-control test scripts.
7. Configured PyTorch CUDA on the Windows Y9000P and verified GPU detection and basic GPU computation.

See [Week1_Development_Environment_and_Dual_Device_Workflow.md](../Week1_Development_Environment_and_Dual_Device_Workflow.md) for environment details and verification results.

## 6. Experimental Process and Results

### 6.1 Screenshot and OCR Test

| Check | Command | Result |
| --- | --- | --- |
| Screenshot and OCR | `python scripts/smoke_test_perception.py` | Captured a 1920 x 1080 desktop screenshot and saved it to `outputs/week1/screenshot_test.png`. Tesseract recognized part of the English interface text, confirming that the screenshot-to-OCR pipeline runs successfully. |

The initial test processed the full screen and used only English language data, so the OCR output contained a small amount of incorrect text. Region cropping, image scaling, binarization, and a Chinese OCR model can improve accuracy.

Tesseract was used to verify the initial OCR pipeline. PaddleOCR will be introduced according to the project outline and evaluated on mixed Chinese and English desktop interfaces.

### 6.2 OpenCV Image-Processing Test

| Check | Command | Result |
| --- | --- | --- |
| OpenCV image processing | `python scripts/smoke_test_image_processing.py` | Read the desktop screenshot, applied grayscale conversion, Gaussian blur, and Otsu binarization, and generated `outputs/week1/screenshot_gray.png` and `outputs/week1/screenshot_binary.png`. |

The result confirms that OpenCV image loading, processing, and saving work correctly and can support OCR preprocessing and later UI element detection.

### 6.3 Mouse and Keyboard Control Test

| Check | Command | Result |
| --- | --- | --- |
| Read-only permission check | `python scripts/smoke_test_control.py` | The first run detected that macOS Accessibility permission was disabled. After enabling permission for Visual Studio Code and restarting it, the permission status became `True`. |
| Reversible control test | `python scripts/smoke_test_control.py --execute` | Moved the pointer by approximately 40 pixels and pressed and released the Shift key. The test did not click or type text, and both events were sent successfully. |

### 6.4 Windows Environment and CUDA Test

| Check | Command or Test | Result |
| --- | --- | --- |
| CUDA and dependency check | `python scripts/check_environment.py --gpu` | All basic dependencies passed, `torch.cuda.is_available()` returned `True`, and the RTX 4060 Laptop GPU was detected. Matrix multiplication passed on `cuda:0`, confirming that PyTorch can use the GPU. |
| Basic tools and code quality | Screenshot, OCR, image processing, Pytest, and Ruff | The Windows machine captured a 2560 x 1600 screenshot and passed English OCR, OpenCV processing, Pytest, and Ruff checks. The machine is ready for later model inference, small-scale fine-tuning, and batch experiments. |

## 7. Week 1 Deliverables

- GUI agent technical research report.
- Mac and Windows dual-device workflow and environment configuration document.
- Basic project directories, dependency lists, and configuration files.
- Environment, screenshot and OCR, image-processing, and desktop-control test scripts.
- Week 1 work log and this experiment report.
- GitHub project repository.

## 8. Conclusion

This week completed the research on the core technical route of GUI agents and established Python development environments on the MacBook Air M2 and Windows Y9000P. The Mac environment passed basic tests for screenshot capture, English OCR, OpenCV image processing, and mouse and keyboard control. The Windows environment passed PyTorch CUDA, GPU computation, and the same basic tool checks. The project structure and dual-device workflow are now in place for the next stage of screen-perception and control development.
