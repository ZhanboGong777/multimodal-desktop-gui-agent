# Week 1 Development Environment and Dual-Device Workflow

This project uses a MacBook Air M2 as the primary development machine and a Lenovo Legion Y9000P as the Windows development and GPU compute machine. The Mac environment includes a Python virtual environment, screen-perception tools, and desktop-control tools. The Windows environment includes a Python virtual environment, PyTorch with CUDA, Tesseract, screenshot capture, OCR, image processing, and basic tests. Model serving and fine-tuning experiments can later be extended on the Y9000P. WSL2 and Ubuntu will be configured only if vLLM or another Linux-specific toolchain is required.

## 1. Device Responsibilities

| Device or Service | Main Uses | Acceleration | Current Status |
| --- | --- | --- | --- |
| MacBook Air M2 | Daily development, screenshots, OCR, desktop control, and API calls | MPS | Basic environment completed |
| Lenovo Legion Y9000P | Windows development, CUDA inference, small-scale LoRA experiments, and batch testing | NVIDIA CUDA | Windows CUDA environment completed |
| Cloud model service | Multimodal model access while travelling or when the GPU machine is offline | API | Connected as required by experiments |

Model inference and desktop control remain separate. The model receives the user instruction and screenshot and returns a structured action. Screenshot capture and mouse and keyboard actions are executed locally on the machine that is running the agent. This design allows the project to switch between a cloud model, a local Mac model, and a Y9000P model service without changing the main workflow.

## 2. Mac Development Environment

| Item | Configuration |
| --- | --- |
| Device and operating system | MacBook Air M2, 24 GB unified memory; macOS 26.3.1, ARM64 |
| Development tools | Visual Studio Code and GitHub Desktop |
| Python | 3.12.10 in the project `.venv` |
| Deep-learning framework | PyTorch 2.14.0 with MPS support built in |
| Basic tools | MSS, Pillow, OpenCV, Tesseract, PyAutoGUI, Pynput, Pytest, and Ruff |

## 3. Mac Environment Verification

| Check | Command or Script | Result |
| --- | --- | --- |
| Dependencies | `python scripts/check_environment.py` | Main Python packages and Tesseract are available |
| Screenshot and OCR | `python scripts/smoke_test_perception.py` | Saved a 1920 x 1080 screenshot and recognized English interface text |
| Image processing | `python scripts/smoke_test_image_processing.py` | Generated grayscale and binary images successfully |
| Desktop control | `python scripts/smoke_test_control.py --execute` | Accessibility permission was available; mouse movement and the Shift key event passed |
| Unit tests | `pytest -q` | One test passed |
| Code quality | `ruff check .` | All checks passed |

The PyTorch build includes MPS support, but `torch.backends.mps.is_available()` returned `False` in the background test environment. This result should be checked again from the Visual Studio Code terminal before local model inference on the Mac. It does not affect the current screenshot, OCR, image-processing, or control modules.

## 4. Windows CUDA Development Environment

| Item | Configuration |
| --- | --- |
| Device and operating system | Lenovo Legion Y9000P, Windows 11, NVIDIA GeForce RTX 4060 Laptop GPU |
| Development tools | Visual Studio Code, GitHub Desktop, and Windows PowerShell |
| Python | 3.12.4 in the project `.venv` |
| Deep-learning framework | PyTorch 2.14.0+cu126 and torchvision 0.29.0+cu126 |
| GPU and driver | RTX 4060 Laptop GPU, 8 GB VRAM; NVIDIA driver 560.94 |
| CUDA | CUDA Runtime 12.6; Compute Capability 8.9 |
| OCR | Tesseract 5.5.3.20260724 |
| WSL | WSL2 enabled; only `docker-desktop` is currently present, and Ubuntu is not installed |

## 5. Windows Environment Verification

| Check | Command or Script | Result |
| --- | --- | --- |
| CUDA availability | `torch.cuda.is_available()` | Returned `True` and detected the RTX 4060 Laptop GPU |
| Dependencies and acceleration | `python scripts/check_environment.py --gpu` | All basic dependencies passed; CUDA Runtime 12.6 and the `cuda:0` matrix operation passed |
| GPU computation | 2048 x 2048 matrix multiplication | Passed on `cuda:0` in 0.049 seconds |
| Driver information | `nvidia-smi` | RTX 4060 Laptop GPU, driver 560.94, 8188 MiB VRAM |
| Dependency consistency | `pip check` | No dependency conflicts |
| Python packages | Import check | All main Python packages imported successfully |
| Screenshot and OCR | `python scripts/smoke_test_perception.py` | 2560 x 1600 screenshot and English OCR passed |
| Image processing | `python scripts/smoke_test_image_processing.py` | Grayscale and binary images generated successfully |
| Unit tests | `pytest` | One test passed |
| Code quality | `ruff check .` | All checks passed |

The native Windows environment satisfies the Week 1 requirements for CUDA, screenshot capture, OCR, image processing, and basic tests. Tesseract with English language data is currently used to verify the OCR pipeline, so Chinese interfaces may be recognized incorrectly. The OCR engine will be updated in the next stage.

## 6. Next Steps

- Install WSL2 Ubuntu only if vLLM or another Linux-specific toolchain is required.
- Continue using the same test suite after the basic Windows compatibility and dual-device acceleration checks.
- Replace Tesseract with PaddleOCR during Week 2 and record the updated dependencies and verification results on both machines.
- Keep this document synchronized with changes to the environments, dependencies, and verification results.
- Continue excluding `.venv`, `.env`, and `outputs` from Git; synchronize only source code and documentation between devices.
