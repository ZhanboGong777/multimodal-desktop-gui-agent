# GUI Agent Literature Index

This folder contains the main papers reviewed during the Week 1 technical research. The papers are grouped by research area and were downloaded from their official arXiv pages. Each filename includes its arXiv identifier for easier retrieval and citation.

## 01 Core GUI Agents

### UI-TARS: Pioneering Automated GUI Interaction with Native Agents

- Local paper: [01_UI-TARS_2501.12326.pdf](01_Core_GUI_Agents/01_UI-TARS_2501.12326.pdf)
- Online paper: [arXiv](https://arxiv.org/abs/2501.12326)
- Repository: [bytedance/UI-TARS](https://github.com/bytedance/UI-TARS)
- Summary: Studies how a native visual GUI agent understands screenshots, plans tasks, and generates mouse and keyboard operations. Its action representation, coordinate processing, and reasoning workflow are particularly relevant to this project.

### ScreenAgent: A Vision Language Model-driven Computer Control Agent

- Local paper: [02_ScreenAgent_2402.07945.pdf](01_Core_GUI_Agents/02_ScreenAgent_2402.07945.pdf)
- Online paper: [arXiv](https://arxiv.org/abs/2402.07945)
- Repository: [niuzaisheng/ScreenAgent](https://github.com/niuzaisheng/ScreenAgent)
- Summary: Controls a computer through planning, action, and reflection. It provides a useful reference for modular desktop-agent design, trajectory recording, and result-feedback mechanisms.

## 02 GUI Understanding and Element Grounding

### SeeClick: Harnessing GUI Grounding for Advanced Visual GUI Agents

- Local paper: [03_SeeClick_2401.10935.pdf](02_GUI_Understanding_and_Element_Grounding/03_SeeClick_2401.10935.pdf)
- Online paper: [arXiv](https://arxiv.org/abs/2401.10935)
- Repository: [njucckevin/SeeClick](https://github.com/njucckevin/SeeClick)
- Summary: Focuses on GUI grounding, which maps natural-language descriptions to the correct buttons, icons, or input fields in screenshots. It also introduces the ScreenSpot benchmark.

### OmniParser for Pure Vision Based GUI Agent

- Local paper: [04_OmniParser_2408.00203.pdf](02_GUI_Understanding_and_Element_Grounding/04_OmniParser_2408.00203.pdf)
- Online paper: [arXiv](https://arxiv.org/abs/2408.00203)
- Repository: [microsoft/OmniParser](https://github.com/microsoft/OmniParser)
- Summary: Parses interface screenshots into interactive regions and semantic descriptions, supporting UI element detection, icon understanding, and click-position prediction.

## 03 Datasets and Benchmarks

### OSWorld: Benchmarking Multimodal Agents for Open-Ended Tasks in Real Computer Environments

- Local paper: [05_OSWorld_2404.07972.pdf](03_Datasets_and_Benchmarks/05_OSWorld_2404.07972.pdf)
- Online paper: [arXiv](https://arxiv.org/abs/2404.07972)
- Repository: [xlang-ai/OSWorld](https://github.com/xlang-ai/OSWorld)
- Summary: Provides tasks, environment initialization, and execution-based evaluation in real operating systems. It is useful for designing basic tasks and end-to-end tests in later project stages.

### Mind2Web: Towards a Generalist Agent for the Web

- Local paper: [06_Mind2Web_2306.06070.pdf](03_Datasets_and_Benchmarks/06_Mind2Web_2306.06070.pdf)
- Online paper: [arXiv](https://arxiv.org/abs/2306.06070)
- Repository: [OSU-NLP-Group/Mind2Web](https://github.com/OSU-NLP-Group/Mind2Web)
- Summary: Contains natural-language tasks and human action trajectories from real websites. Its data structure, action sequences, and dataset splits are useful references for GUI dataset processing.

### WebArena: A Realistic Web Environment for Building Autonomous Agents

- Local paper: [07_WebArena_2307.13854.pdf](03_Datasets_and_Benchmarks/07_WebArena_2307.13854.pdf)
- Online paper: [arXiv](https://arxiv.org/abs/2307.13854)
- Repository: [web-arena-x/webarena](https://github.com/web-arena-x/webarena)
- Summary: Provides reproducible web tasks and evaluates functional correctness using the final environment state. It is useful for studying long-horizon tasks and result verification.

## 04 Official Technical Documentation

The reviewed Claude Computer Use resources do not include a standalone official research paper, so they are stored separately as a technical-resource index:

- [Claude Computer Use Official Resources](04_Official_Technical_Documentation/Claude_Computer_Use_Official_Resources.md)

## Recommended Reading Order

1. UI-TARS
2. Claude Computer Use official resources
3. ScreenAgent
4. SeeClick and OmniParser
5. OSWorld
6. Mind2Web and WebArena

---

English version prepared: 8 September 2026

