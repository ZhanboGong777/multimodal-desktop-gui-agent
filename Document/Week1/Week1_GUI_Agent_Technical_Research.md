# Week 1 GUI Agent Technical Research Report

This week, I reviewed papers, technical documentation, and open-source implementations related to multimodal large language model powered desktop GUI agents. The review focused on UI-TARS, Claude Computer Use, and ScreenAgent. Although their implementation details differ, all three approaches require screen understanding, task planning, action execution, and result feedback. The project should first establish stable screenshot capture, OCR, and mouse and keyboard control modules before connecting a local model or an API-based model.

## 1. Research Objectives

This research addresses three questions: which modules form a desktop GUI agent, how existing systems connect multimodal models to desktop-control tools, and which implementation sequence is suitable for this eight-week project. The primary scope follows the project outline and covers UI-TARS, Claude Computer Use, and ScreenAgent, with additional work on interface grounding, datasets, and evaluation.

## 2. Basic Architecture of a GUI Agent

A GUI agent receives a user task and the current screen state, then produces mouse or keyboard actions that a local program can execute. After each action, the system captures a new screenshot and checks whether the interface changed as expected. This creates a closed loop:

**User instruction -> screen perception -> task planning -> action execution -> result feedback**

1. **Screen perception** captures screenshots and identifies interface elements such as text, buttons, input fields, and icons.
2. **Task planning** generates the next action from the user instruction and current interface, including clicking, typing, scrolling, and waiting.
3. **Control** converts model output into mouse and keyboard operations while constraining the action range and execution order.
4. **Feedback** checks the new screenshot, replans when an action fails or the interface changes unexpectedly, and records the interaction log.

## 3. Representative Technical Approaches

### 3.1 UI-TARS

UI-TARS directly interprets and grounds interface elements from screenshots, then generates mouse and keyboard actions. It combines perception, reasoning, and action generation in one model, reducing its dependence on the DOM or accessibility tree. Its coordinate representation, action format, and interaction-trajectory training are closely related to this project's desktop-control objective. Related implementations include UI-TARS Desktop and Midscene for browser automation.

- GitHub: https://github.com/bytedance/UI-TARS
- Paper: https://arxiv.org/abs/2501.12326
- Desktop implementation: https://github.com/bytedance/UI-TARS-desktop
- Browser automation: https://github.com/web-infra-dev/midscene

### 3.2 Claude Computer Use

Claude Computer Use follows a loop between the model and local tools. The model reads a screenshot and requests an action, the local program executes the action, and a new screenshot is returned until the task is complete. This approach clearly separates the model from the executor and emphasizes confirmation for sensitive actions, environment isolation, and prompt-injection protection.

- GitHub: https://github.com/anthropics/claude-quickstarts/tree/main/computer-use-demo
- Official documentation: https://platform.claude.com/docs/en/agents-and-tools/tool-use/computer-use-tool

### 3.3 ScreenAgent

ScreenAgent divides computer control into planning, execution, and reflection and records the complete interaction trajectory. The open-source project provides model services, data, and a desktop-control client. Its trajectory records preserve screenshots, actions, and results at each step, which helps identify the cause of a failed task.

- GitHub: https://github.com/niuzaisheng/ScreenAgent
- Paper: https://arxiv.org/abs/2402.07945

### 3.4 Comparison

| Approach | Interface Understanding | Action Execution | Main Strength |
| --- | --- | --- | --- |
| UI-TARS | Direct screenshot understanding and element grounding | The model generates coordinates and actions | End-to-end visual operation |
| Claude Computer Use | Screenshots are provided as model input | Local tools execute actions in a loop | Clear interface and safety mechanisms |
| ScreenAgent | Screenshots and interaction history | Planning, execution, and reflection | Modular design and complete trajectory records |

## 4. Interface Grounding and Evaluation

Interface grounding determines whether an agent can click the correct location. SeeClick treats GUI grounding as a separate task and predicts control coordinates from a textual instruction. OmniParser first converts a screenshot into semantic interactive regions and then provides the parsed result to a planning model. These approaches represent direct coordinate prediction and structured interface parsing and can support later grounding experiments.

- OmniParser paper: https://arxiv.org/abs/2408.00203
- OmniParser GitHub: https://github.com/microsoft/OmniParser
- SeeClick paper: https://arxiv.org/abs/2401.10935
- SeeClick GitHub: https://github.com/njucckevin/SeeClick

For evaluation, OSWorld provides tasks and automated validation in real desktop environments. Mind2Web provides natural-language web tasks and human action trajectories. WebArena provides a reproducible web environment and evaluates whether a task is complete from the final environment state. These projects show that a GUI agent should be evaluated on successful task completion, not merely on whether it produced a click action.

- OSWorld paper: https://arxiv.org/abs/2404.07972
- OSWorld GitHub: https://github.com/xlang-ai/OSWorld
- Mind2Web paper: https://arxiv.org/abs/2306.06070
- Mind2Web GitHub: https://github.com/OSU-NLP-Group/Mind2Web
- WebArena paper: https://arxiv.org/abs/2307.13854
- WebArena GitHub: https://github.com/web-arena-x/webarena

## 5. Current Technical Challenges

- Screenshots contain text, icons, and visually similar buttons, so the model must understand both semantics and coordinates.
- Long tasks contain multiple intermediate states, and one failed action can affect all later steps.
- Window scaling, screen resolution, and operating-system differences change element positions and require consistent coordinate and screenshot processing.
- Page loading, pop-up windows, and input focus introduce uncertainty and require waiting, detection, and retry logic.
- Desktop control can affect real files and accounts, so high-risk actions must be restricted and execution records must be retained.

## 6. Initial Technical Route

The first prototype will use a modular architecture. The perception layer will capture screenshots with MSS and use OpenCV and OCR to extract text and interface regions. The planning layer will define a common model interface that converts the user instruction and interface state into structured actions. The execution layer will use PyAutoGUI and Pynput for mouse and keyboard operations. The feedback and logging module will record each screenshot, action, execution time, and result. Tesseract is used for the initial OCR pipeline test; PaddleOCR will be introduced later for comparison and optimization on Chinese and English desktop interfaces.

The model interface will support both local deployment and cloud APIs so that the project is not tied to a single model provider. The MacBook will be used for daily development and MPS tests, while the Windows GPU machine will be used for CUDA inference and small-scale LoRA experiments. After the basic closed loop is complete, a fixed task set will be used to compare task success rates across perception methods and models.

For cross-platform support, the perception layer will represent positions with logical coordinates and convert them to global coordinates using the target display's origin offset. The macOS control layer requires Accessibility and Screen Recording permissions, while the Windows implementation must account for display scaling. Platform-specific permissions and dependencies will be checked through delayed imports so that the same codebase can run on macOS and Windows. Linux support remains a later objective.

## 7. Research Conclusions

The review confirms that a desktop GUI agent depends on a continuous perception, planning, execution, and feedback loop. UI-TARS demonstrates end-to-end visual operation, Claude Computer Use provides a clear interface between a model and local tools, and ScreenAgent demonstrates planning, reflection, and trajectory recording. These findings support a modular prototype with interchangeable model backends, local desktop execution, and detailed logging.

## References

1. UI-TARS: Pioneering Automated GUI Interaction with Native Agents. Paper: https://arxiv.org/abs/2501.12326; GitHub: https://github.com/bytedance/UI-TARS
2. Claude Computer Use. Documentation: https://platform.claude.com/docs/en/agents-and-tools/tool-use/computer-use-tool; Example: https://github.com/anthropics/claude-quickstarts/tree/main/computer-use-demo
3. ScreenAgent: A Vision Language Model-driven Computer Control Agent. Paper: https://arxiv.org/abs/2402.07945; GitHub: https://github.com/niuzaisheng/ScreenAgent
4. SeeClick: Harnessing GUI Grounding for Advanced Visual GUI Agents. Paper: https://arxiv.org/abs/2401.10935; GitHub: https://github.com/njucckevin/SeeClick
5. OmniParser for Pure Vision Based GUI Agent. Paper: https://arxiv.org/abs/2408.00203; GitHub: https://github.com/microsoft/OmniParser
6. OSWorld: Benchmarking Multimodal Agents for Open-Ended Tasks in Real Computer Environments. Paper: https://arxiv.org/abs/2404.07972; GitHub: https://github.com/xlang-ai/OSWorld
7. Mind2Web: Towards a Generalist Agent for the Web. Paper: https://arxiv.org/abs/2306.06070; GitHub: https://github.com/OSU-NLP-Group/Mind2Web
8. WebArena: A Realistic Web Environment for Building Autonomous Agents. Paper: https://arxiv.org/abs/2307.13854; GitHub: https://github.com/web-arena-x/webarena
