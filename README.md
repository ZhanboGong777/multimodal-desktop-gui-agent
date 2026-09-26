# Multimodal Desktop GUI Agent

### Project Overview

This is a focused on developing and optimizing a desktop GUI agent powered by multimodal large language models. The agent is expected to understand natural-language instructions, perceive text and interface elements on the screen, plan task steps, and perform desktop operations through mouse and keyboard controls.

### Expected Development Progress

| Week | Planned Work |
|---|---|
| Week 1 | Technical research and development environment setup |
| Week 2 | Desktop perception and control module development |
| Week 3 | GUI dataset processing and multimodal Agent framework setup |
| Week 4 | End-to-end system integration and basic desktop task testing |
| Week 5 | Multimodal model fine-tuning and prompt optimization |
| Week 6 | Advanced features and system robustness optimization |
| Week 7 | System evaluation and performance analysis |
| Week 8 | Code cleanup, technical report, and system demonstration |

### Current Status

The project has completed Week 3. On top of the Week 2 perception and control
modules, the repository now contains adapters that normalise the ScreenAgent,
Mind2Web and WebArena datasets into one `GUITaskSample` format with a validated
JSONL export, a provider-independent multimodal model interface with an offline
mock backend and an OpenAI-compatible client, and a planner that turns an
instruction into a Pydantic-validated `TaskPlan`. Nothing in Week 3 executes a
plan; that is Week 4.

```bash
python scripts/week3_prepare_dataset.py --dataset webarena \
    --input data/raw/webarena/test.raw.json --output data/processed/webarena.jsonl --limit 20
python scripts/week3_model_demo.py --provider mock
python scripts/week3_planning_demo.py --provider mock --instruction "Open the browser"
```

### Week 2

The project has completed Week 2. The repository now contains reusable screen
capture, image preprocessing, two OCR backends (PaddleOCR with a Tesseract
fallback), non-text UI candidate detection, bounding-box annotation, text
grounding, screenshot-to-control coordinate mapping, dry-run-first desktop
control with drag support, per-run recording, and 165 unit tests at 88% coverage.

Progress and project deliverables will be updated throughout the development
process.
