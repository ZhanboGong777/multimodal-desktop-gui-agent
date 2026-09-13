# Work Log

## Week 1 - 8 September 2026 to 13 September 2026

### Completed

- Reviewed GUI agent technologies, papers, technical documentation, and open-source repositories.
- Defined the collaborative development workflow for the MacBook and Windows Y9000P.
- Cloned the GitHub repository to the MacBook and configured Visual Studio Code and the project virtual environment.
- Initialized the core source, configuration, script, and test directories.
- Installed PyTorch, OpenCV, MSS, Tesseract OCR, PyAutoGUI, Pynput, and other basic dependencies.
- Added environment, screenshot and OCR, image-processing, and desktop-control test scripts.
- Configured `.gitignore`, `.gitattributes`, dependency lists, and the environment-variable template.
- Configured PyTorch CUDA on the Windows Y9000P.
- Prepared the Week 1 technical research, environment configuration, and experiment reports.

### Verification Results

- The Mac Python 3.12.10 virtual environment is available.
- Screenshot capture and English OCR passed on the Mac.
- OpenCV grayscale conversion and binarization passed on the Mac.
- macOS Accessibility permission was configured, and the reversible mouse and keyboard test passed.
- The PyTorch build includes MPS support; local availability will be checked again from the Visual Studio Code terminal before Mac model inference.
- The Windows Python 3.12.4 virtual environment and project dependencies are available.
- PyTorch detected the RTX 4060 Laptop GPU, and the CUDA matrix operation passed on `cuda:0`.
- Windows screenshot capture, English OCR, OpenCV processing, Pytest, and Ruff checks passed.

### Next Steps

- Introduce PaddleOCR and evaluate mixed Chinese and English desktop interfaces.
- Continue developing the screen-perception and desktop-control core modules.
- Define a common model interface for cloud APIs, Mac-local models, and the Windows GPU service.
- Continue recording environment changes and experimental results in the project documentation.
