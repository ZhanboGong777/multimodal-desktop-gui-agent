# Week 1 Development Environment and Dual-Device Workflow

## 1. Recommended Approach

This project will use a **MacBook Air M2 as the primary development machine and the Lenovo Legion Y9000P as a remote GPU compute node**.

The MacBook will handle daily coding, screen capture, OCR, mouse and keyboard control, agent framework development, API integration, logging, and macOS task testing. The Y9000P will mainly handle CUDA-based model inference, model fine-tuning, and batch performance evaluation. When the Y9000P is unavailable, the project can switch to a cloud multimodal-model API so that development can continue while travelling.

A MacBook Air M2 with 24 GB of unified memory is sufficient for most development tasks in this project. However, Apple Silicon uses MPS/MLX rather than NVIDIA CUDA. CUDA-specific operators, official CUDA training scripts, and sustained heavy training workloads should be run on the Y9000P.

## 2. Dual-Device Architecture

```text
MacBook Air - primary development machine
├── User instruction input
├── Screen capture and OCR
├── UI element parsing
├── Task decomposition and agent loop
├── Mouse and keyboard execution
├── Logging, error detection, and retry
└── Model client
     ├── Cloud multimodal-model API
     ├── Local Mac MPS/MLX model
     └── Remote Y9000P model service
                         ↓
Y9000P - home GPU node
├── WSL2 and Ubuntu
├── NVIDIA CUDA and PyTorch
├── Transformers or vLLM model service
├── LoRA or QLoRA fine-tuning
└── Batch testing and performance evaluation
```

Model inference and desktop control should be separated. The Mac sends the user instruction and required screenshots to the selected model service. The model returns a structured action, while the mouse and keyboard action is executed locally on the Mac.

## 3. Week 1 Environment Objectives

Week 1 should complete the following setup tasks:

1. Configure the macOS development tools and a Python virtual environment.
2. Install PyTorch and verify MPS acceleration.
3. Install screen capture, image-processing, OCR, and mouse/keyboard-control libraries.
4. Install multimodal-model API clients.
5. Install the browser-automation environment.
6. Configure macOS Screen Recording and Accessibility permissions.
7. Test each basic function and record environment versions.
8. Prepare a unified model interface for later connection to the Y9000P.

## 4. MacBook Air Deployment Steps

### Step 1: Check the System

macOS 14 or later is recommended.

```bash
sw_vers
uname -m
```

`uname -m` should return `arm64`.

### Step 2: Install Xcode Command Line Tools

```bash
xcode-select --install
```

Verify the installation:

```bash
xcode-select -p
```

### Step 3: Install Homebrew

If Homebrew is not already installed, use the command provided on its official website:

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

Follow the terminal instructions to add Homebrew to `PATH`, and then verify it:

```bash
brew --version
```

Official reference: [Homebrew Installation](https://docs.brew.sh/Installation)

### Step 4: Install the Base Development Tools

```bash
brew install git git-lfs python@3.11 node pnpm tesseract
git lfs install
```

Verify the tools:

```bash
git --version
python3.11 --version
node --version
pnpm --version
tesseract --version
```

UI-TARS Desktop officially requires Node.js 20 or later and pnpm 9 or later:

- [UI-TARS Desktop Contributing Guide](https://github.com/bytedance/UI-TARS-desktop/blob/main/CONTRIBUTING.md)

### Step 5: Clone the Repository and Create a Virtual Environment

Run the following commands in the location where the project code should be stored:

```bash
git clone https://github.com/ZhanboGong777/multimodal-desktop-gui-agent.git
cd multimodal-desktop-gui-agent
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Activate the environment whenever starting a development session:

```bash
source .venv/bin/activate
```

Python recommends isolated virtual environments for individual projects. The `.venv` directory should not be committed to Git:

- [Python venv Documentation](https://docs.python.org/3/library/venv.html)

### Step 6: Install PyTorch and Common Scientific Libraries

```bash
pip install torch torchvision torchaudio
pip install numpy pandas pillow opencv-python
```

Verify PyTorch and MPS:

```bash
python -c "import torch; print('PyTorch:', torch.__version__); print('MPS available:', torch.backends.mps.is_available())"
```

`MPS available: True` indicates that PyTorch can use the M2 GPU.

Apple Silicon uses the MPS backend and does not require CUDA:

- [Accelerated PyTorch Training on Mac](https://developer.apple.com/metal/pytorch/)
- [PyTorch MPS Backend](https://docs.pytorch.org/docs/main/notes/mps.html)

### Step 7: Install Desktop Perception and Control Libraries

```bash
pip install mss pyautogui pynput pytesseract
```

Main purposes:

- `mss`: fast screen capture.
- `Pillow` and `OpenCV`: image loading, resizing, annotation, and preprocessing.
- `pytesseract`: OCR through Tesseract.
- `pyautogui`: mouse, keyboard, and basic screenshot control.
- `pynput`: mouse and keyboard event monitoring or control.

### Step 8: Install Model API and Configuration Libraries

```bash
pip install openai anthropic httpx python-dotenv pydantic
```

Create a local `.env` file in the repository root:

```env
MODEL_PROVIDER=api
MODEL_BASE_URL=
MODEL_API_KEY=
```

Never upload `.env` to GitHub. A safe `.env.example` without real credentials can be committed instead.

### Step 9: Install Testing and Code-Quality Tools

```bash
pip install pytest pytest-asyncio ruff
```

These packages provide unit testing, asynchronous testing, formatting, and code-quality checks.

### Step 10: Install Browser Automation

```bash
pip install playwright pytest-playwright
playwright install chromium
```

Playwright supports macOS and requires a separate browser-binary installation:

- [Playwright Python Installation](https://playwright.dev/python/docs/intro)

### Step 11: Configure macOS Permissions

Open:

```text
System Settings -> Privacy & Security
```

Enable the following permissions for the application that runs the code:

- Accessibility
- Screen Recording
- Automation when requested by macOS

The relevant application may be Terminal, VS Code, PyCharm, or another terminal. Permissions should be checked again when the execution application changes.

### Step 12: Run Basic Verification Tests

Test Python imports:

```bash
python -c "import torch, cv2, mss, pyautogui, pytesseract; print('Python libraries: OK')"
```

Test screen capture:

```bash
python -c "import pyautogui; pyautogui.screenshot('test_screenshot.png'); print('Screenshot saved')"
```

Read the current cursor position:

```bash
python -c "import pyautogui; print('Cursor position:', pyautogui.position())"
```

Test OCR:

```bash
python -c "from PIL import Image; import pytesseract; print(pytesseract.image_to_string(Image.open('test_screenshot.png'))[:500])"
```

Test Playwright:

```bash
python -c "from playwright.sync_api import sync_playwright; p=sync_playwright().start(); b=p.chromium.launch(); print('Chromium:', b.version); b.close(); p.stop()"
```

Record the installed versions and test results for the development-environment document and the Week 1 experimental report.

## 5. Recommended Repository Structure

```text
multimodal-desktop-gui-agent/
├── src/
│   └── gui_agent/
│       ├── perception/       # Screenshot, OCR, and UI parsing
│       ├── planning/         # Task decomposition and action planning
│       ├── actions/          # Mouse and keyboard operations
│       ├── models/           # API, local, and remote model clients
│       ├── evaluation/       # Result verification and metrics
│       └── logging/          # Trajectories, errors, and runtime logs
├── tests/
├── scripts/
├── configs/
├── docs/
├── data/
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

## 6. GitHub File Management

Recommended `.gitignore` entries:

```gitignore
.venv/
.env
__pycache__/
.pytest_cache/
.ruff_cache/
.DS_Store
models/
checkpoints/
outputs/
logs/
data/raw/
```

GitHub should contain source code, configuration templates, tests, documentation, and small example data. Model weights, large datasets, API keys, checkpoints, and large collections of screenshots should not be committed directly.

## 7. Model Interface Design

The agent should not be tightly coupled to a single model provider. A common interface can be defined as follows:

```python
class ModelClient:
    def predict(self, instruction: str, screenshot_path: str) -> dict:
        raise NotImplementedError
```

Later implementations can include:

```text
CloudAPIClient   # Cloud API while travelling
MacLocalClient   # Local MPS/MLX experiments
RemoteGPUClient  # Home Y9000P model service
```

This allows the model backend to be changed through configuration without changing the main agent loop.

## 8. Y9000P GPU Node Deployment

Basic inspection of the Y9000P can be completed during Week 1. The full CUDA model service can be introduced gradually between Weeks 3 and 5.

### Step 1: Confirm the GPU and VRAM

Run in Windows PowerShell:

```powershell
nvidia-smi
```

Record the GPU model, VRAM capacity, and driver version. Model selection and fine-tuning settings should be based on the actual VRAM.

### Step 2: Install WSL2 and Ubuntu

Run in an administrator PowerShell window:

```powershell
wsl --install -d Ubuntu-24.04
```

Restart Windows after installation and create the Ubuntu user account.

### Step 3: Install Linux Development Tools

Run inside WSL Ubuntu:

```bash
sudo apt update
sudo apt install -y git git-lfs python3-venv python3-pip build-essential
git lfs install
```

### Step 4: Create the GPU Python Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Use the [official PyTorch installation selector](https://pytorch.org/get-started/locally/) to generate a CUDA installation command compatible with the current NVIDIA driver. Do not permanently hard-code a CUDA version before checking the machine.

Verify CUDA after installation:

```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
```

### Step 5: Install Model Serving and Fine-Tuning Libraries

Install packages gradually according to the selected model:

```bash
pip install transformers accelerate peft datasets
```

vLLM can be added when an OpenAI-compatible local model server is needed. On Windows, vLLM is generally used through WSL2/Linux:

- [vLLM Installation](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/index.html)

### Step 6: Start a Local Model Service

After preparing a compatible model, run a service that initially listens only on the local machine:

```bash
vllm serve <model-name-or-path> --host 127.0.0.1 --port 8000
```

If VRAM is limited, use a smaller or quantized model. An RTX 4060/4070 with 8 GB of VRAM is more suitable for small-model inference, quantized models, and limited LoRA/QLoRA experiments than for full fine-tuning of a large multimodal model.

### Step 7: Establish a Secure Remote Connection

Do not expose port `8000` directly to the public internet. Connect to the home computer through a private VPN or SSH and then create a tunnel:

```bash
ssh -L 8000:127.0.0.1:8000 <user>@<home-gpu-host>
```

After the tunnel is established, the Mac can access the service at:

```text
http://127.0.0.1:8000/v1
```

The application should fall back to a cloud API whenever the Y9000P is offline.

## 9. Eight-Week Device Allocation

| Week | Primary Device | Work |
| --- | --- | --- |
| Week 1 | Mac | Technical research, complete environment setup, and basic tool verification |
| Week 2 | Mac | Screen capture, OCR, UI grounding, and mouse/keyboard control |
| Week 3 | Mac | Agent framework, model APIs, task decomposition, and remote-GPU interface preparation |
| Week 4 | Mac | System integration, command-line interface, and basic task tests |
| Week 5 | Mac + Y9000P | Data and prompt work on Mac; inference and fine-tuning experiments on Y9000P |
| Week 6 | Both | Grounding optimization, error recovery, logging, and cross-platform testing |
| Week 7 | Mainly Y9000P | Batch tests and performance evaluation, with supplementary macOS testing on Mac |
| Week 8 | Mac | Code organization, technical report, and system demonstration |

## 10. Week 1 Completion Criteria and Deliverables

Week 1 environment setup is complete when:

- The Python virtual environment can be created and activated.
- PyTorch runs and the MPS result has been recorded.
- A screenshot can be captured and saved successfully.
- OCR can extract text from a test screenshot.
- The cursor position can be read and macOS permissions are configured correctly.
- Playwright can launch Chromium.
- Model API packages can be imported and credentials are managed through `.env`.
- The Git repository and `.gitignore` are configured.
- Python, PyTorch, Node.js, pnpm, and Tesseract versions are recorded.
- The GUI agent technical research has been completed.

Recommended Week 1 deliverables:

1. GUI Agent Technical Research Report
2. Development Environment Configuration Guide
3. Basic verification code or screenshots
4. Environment version and test-result record
5. Week 1 Experimental Report

---

Document date: 8 September 2026  
Note: The commands in this document are deployment instructions. Confirm the current directory, operating-system version, and actual project dependencies before running them.
