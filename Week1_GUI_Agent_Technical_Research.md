# Week 1 Technical Research: Multimodal LLM-Powered Desktop GUI Agents

## 1. Research Objective

This research focuses on desktop GUI agents powered by multimodal large language models. The main objective is to understand how these agents perform screen perception, UI element grounding, task decomposition and planning, mouse and keyboard control, result verification, and error recovery. Relevant datasets, evaluation environments, and open-source implementations are also reviewed to support the selection of a technical approach for subsequent development.

## 2. Related Papers and Technical Resources

### 2.1 UI-TARS: Pioneering Automated GUI Interaction with Native Agents

- Paper: [UI-TARS on arXiv](https://arxiv.org/abs/2501.12326)
- GitHub: [bytedance/UI-TARS](https://github.com/bytedance/UI-TARS)

UI-TARS is a native visual GUI agent that understands interfaces from screenshots and generates mouse and keyboard actions. The paper discusses unified action representation, task reasoning, GUI element grounding, and iterative improvement through interaction trajectories. It is closely aligned with the objective of this project and is particularly useful for studying model input and output formats, action parsing, and coordinate processing.

### 2.2 Claude Computer Use

- Official documentation: [Claude Computer Use Tool](https://platform.claude.com/docs/en/agents-and-tools/tool-use/computer-use-tool)
- Official example: [Anthropic Computer Use Demo](https://github.com/anthropics/anthropic-quickstarts/tree/main/computer-use-demo)

Claude Computer Use controls a desktop through a loop in which the model requests an action, the local application executes it, and an updated screenshot is returned to the model. The official example includes an agent loop, screenshot capture, mouse and keyboard tools, and a container-based isolated environment. It is useful for understanding the interface between a multimodal model and local control tools, as well as security considerations such as environment isolation, confirmation of sensitive actions, and prompt-injection protection.

### 2.3 ScreenAgent: A Vision Language Model-driven Computer Control Agent

- Paper: [ScreenAgent on arXiv](https://arxiv.org/abs/2402.07945)
- GitHub: [niuzaisheng/ScreenAgent](https://github.com/niuzaisheng/ScreenAgent)

ScreenAgent uses screenshots as input and performs computer tasks through mouse and keyboard actions. Its main workflow includes planning, action execution, and reflection. The open-source project provides data, model services, and a desktop control client. It is useful for studying prototype architecture, interaction trajectory recording, and adjustment of future actions based on execution results.

### 2.4 OSWorld: Benchmarking Multimodal Agents for Open-Ended Tasks in Real Computer Environments

- Paper: [OSWorld on arXiv](https://arxiv.org/abs/2404.07972)
- GitHub: [xlang-ai/OSWorld](https://github.com/xlang-ai/OSWorld)

OSWorld provides desktop tasks, environment initialization, and automated evaluation in real computer environments. Its tasks cover web applications, office software, file operations, and cross-application workflows. It is useful for designing later test tasks and for studying evaluation based on task success, execution behaviour, and final environment state.

### 2.5 SeeClick: Harnessing GUI Grounding for Advanced Visual GUI Agents

- Paper: [SeeClick on arXiv](https://arxiv.org/abs/2401.10935)
- GitHub: [njucckevin/SeeClick](https://github.com/njucckevin/SeeClick)

SeeClick treats GUI grounding - locating the correct interface element from a natural-language instruction - as a key capability of visual GUI agents. The project also introduces the ScreenSpot benchmark, which includes desktop, web, and mobile interfaces. This work is relevant to coordinate prediction and grounding-accuracy evaluation for buttons, input fields, icons, and other UI elements.

### 2.6 OmniParser for Pure Vision Based GUI Agent

- Paper: [OmniParser on arXiv](https://arxiv.org/abs/2408.00203)
- GitHub: [microsoft/OmniParser](https://github.com/microsoft/OmniParser)

OmniParser converts interface screenshots into structured interactive regions with semantic descriptions, helping multimodal models understand and locate screen elements more accurately. It can complement OCR-based perception and can be used to compare a screenshot-only approach with an approach that augments screenshots with structured UI information.

### 2.7 Mind2Web: Towards a Generalist Agent for the Web

- Paper: [Mind2Web on arXiv](https://arxiv.org/abs/2306.06070)
- GitHub: [OSU-NLP-Group/Mind2Web](https://github.com/OSU-NLP-Group/Mind2Web)

Mind2Web is a dataset and evaluation project for general-purpose web agents. It contains natural-language tasks and human action trajectories collected from a wide range of real websites. Although it focuses on web environments, its task descriptions, action sequences, and dataset organization provide useful references for GUI dataset processing and simple task decomposition in this project.

### 2.8 WebArena: A Realistic Web Environment for Building Autonomous Agents

- Paper: [WebArena on arXiv](https://arxiv.org/abs/2307.13854)
- GitHub: [web-arena-x/webarena](https://github.com/web-arena-x/webarena)

WebArena provides a reproducible web-interaction environment with long-horizon tasks and evaluates completion using the final state of the environment. Its main value to this project is its evaluation design: an agent should be assessed not only on whether it performed a click, but also on whether the resulting state actually satisfies the user's objective.

## 3. Key GitHub Repositories

| Repository | Main Content | Relevance to This Project |
| --- | --- | --- |
| [bytedance/UI-TARS](https://github.com/bytedance/UI-TARS) | GUI agent models, inference deployment, action parsing, and coordinate processing | Reference for model invocation and action output formats |
| [bytedance/UI-TARS-desktop](https://github.com/bytedance/UI-TARS-desktop) | A desktop application for controlling local computers and browsers | Reference for overall desktop architecture and user interaction |
| [anthropics/anthropic-quickstarts](https://github.com/anthropics/anthropic-quickstarts/tree/main/computer-use-demo) | Claude Computer Use desktop-control example | Reference for the agent loop, tool wrappers, and environment isolation |
| [niuzaisheng/ScreenAgent](https://github.com/niuzaisheng/ScreenAgent) | Planning, action, reflection, datasets, and control client | Reference for modular design and trajectory recording |
| [xlang-ai/OSWorld](https://github.com/xlang-ai/OSWorld) | Real desktop environments, task sets, and automated evaluation | Reference for test design and evaluation metrics |
| [microsoft/OmniParser](https://github.com/microsoft/OmniParser) | Screenshot parsing, interactive-region detection, and icon understanding | Reference for screen perception and UI element parsing |
| [njucckevin/SeeClick](https://github.com/njucckevin/SeeClick) | GUI grounding model, training data, and ScreenSpot benchmark | Reference for UI coordinate grounding and accuracy evaluation |
| [OSU-NLP-Group/Mind2Web](https://github.com/OSU-NLP-Group/Mind2Web) | Real-world web tasks, action trajectories, and evaluation code | Reference for GUI dataset structure and task decomposition |
| [web-arena-x/webarena](https://github.com/web-arena-x/webarena) | Reproducible web environment and end-to-end task evaluation | Reference for result verification and long-horizon task testing |

## 4. Initial Findings

Most current GUI agents follow the closed-loop workflow below:

**User instruction -> screen capture and interface parsing -> task decomposition and action planning -> mouse and keyboard execution -> updated screen and result feedback -> error detection and retry**

Based on the project objective, the initial prototype can adopt a modular design. Screenshot capture and OCR/interface parsing can provide screen perception; a multimodal large language model can perform task planning and action generation; mouse and keyboard tools can execute the actions; and a unified logging module can record screenshots, actions, execution time, results, and errors at each step.

During the first stage, UI-TARS, the Claude Computer Use Demo, and ScreenAgent should be studied and tested first. OSWorld, SeeClick, OmniParser, Mind2Web, and WebArena can then be used as references during evaluation and optimization.

## 5. Recommended Reading Order

1. UI-TARS: understand the overall technical approach of a visual GUI agent.
2. Claude Computer Use: understand the interaction loop between a model and desktop-control tools.
3. ScreenAgent: study planning, execution, reflection, and trajectory recording.
4. SeeClick and OmniParser: study UI element grounding and screenshot parsing.
5. OSWorld, Mind2Web, and WebArena: study dataset formats, task design, and evaluation methods.

---

Research date: 7 September 2026  
Note: All links in this document point to original papers, official technical documentation, or official project repositories.

