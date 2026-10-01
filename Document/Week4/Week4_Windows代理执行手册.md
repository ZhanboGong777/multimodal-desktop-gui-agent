# Week 4：Windows 端执行手册（写给执行代理）

> **这份文档写给在 Windows 机器上干活的 AI 代理。** 它不需要读别的文档，也不需要访问 Mac：
> 所有命令、期望输出、失败处理和回报格式都在这里。仓库里另有一份 `Week4_Windows复核手册.md`
> （给人类复核方看的），内容一致，但**以本文为准**。

---

## 0. 三条铁律

先读这三条，它们比后面所有命令都重要。

1. **不许编造任何结果。** 你报告里的每一个数字、状态、时间，都必须来自你**实际执行过的命令的
   实际输出**。没跑过的，写"未执行"；跑失败的，照原样报失败。
2. **不许把"没测"写成"通过"。** 这个项目的全部价值建立在"声称"与"证据"分得开。五个任务里
   任何一个没跑成，就如实写没跑成——**空着比填错好，填错比编造好**。
3. **不许修改证据。** `Document/Week4/evidence/` 下的文件是运行产物，只能新增、不能编辑。
   失败记录**不得删除**，也不得为了让成功率好看而重跑覆盖。

如果你在某个环节卡住并且无法解决，**停下来报告卡在哪一步、报错原文是什么**，不要绕过它继续。

---

## 1. 前提检查（先做这一步）

```powershell
cd D:\Developer\multimodal-desktop-gui-agent
git status
git log --oneline -3
Test-Path scripts\week4_warmup.py
```

**期望**：

- `git status` 干净（或只有 `outputs/` 等未跟踪目录）；
- `Test-Path` 返回 **`True`**。

**如果 `Test-Path` 返回 `False`**，说明推送还没到这台机器。停下来，告诉用户：

> 仓库里还没有 `scripts/week4_warmup.py`，本地代码是旧版本（`origin/main` 停在第一轮那个
> "五个用例定位全部失败"的版本）。请先在 Mac 侧 `git push`，这里再 `git pull`。

**不要**在旧版本上继续跑后面的步骤——那样跑出来的失败没有任何意义。

记录下 `git log --oneline -1` 的**完整提交号**，回报时要用。

---

## 2. 安装与环境自检

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements-agent.txt
pip install -r requirements-dev.txt
```

> 若 `.venv` 不存在：`python -m venv .venv` 后再激活。

### 2.1 必须安装 `tesseract` 可执行文件

**这一步漏掉的话，五个任务会全部失败，而且原因和代码无关。**

`configs/week4.yaml` 里选的是 `ocr.engine: tesseract`。`pip` 装的 `pytesseract` **只是外壳**，
真正的识别程序是那个 exe，它不在 requirements 里。

```powershell
winget install --id UB-Mannheim.TesseractOCR
# 装完【重开一个 PowerShell】，然后确认：
tesseract --version
```

若 `tesseract` 仍不在 PATH：把它所在目录加进 PATH，或者改用 PaddleOCR
（`pip install -r requirements-ocr-gpu.txt`，然后把 `configs/week4.yaml` 里 `ocr.engine` 改成
`paddleocr`）。**不要同时装 `paddlepaddle` 和 `paddlepaddle-gpu`。**

### 2.2 环境自检

```powershell
python scripts/check_environment.py
```

**期望结尾**：`Week 1 environment is ready.`，并且中间有一行 **`Screen capture: OK (<宽>x<高>)`**。

| 你看到的 | 含义 | 处理 |
|---|---|---|
| `Screen capture: OK (2560x1600)` | 截图链路正常 | 继续 |
| `Screen capture: NO DISPLAY VISIBLE` | 这个进程看不到显示设备 | **停下来报告**。远程会话或权限问题，后面每一步都会失败 |
| `Tesseract CLI: MISSING` | 第 2.1 步没生效 | 回 2.1，重开终端 |
| `macOS Accessibility permission: N/A` | Windows 上正常 | 忽略 |

---

## 3. 离线验证（不碰桌面、不碰模型）

三条命令，逐条跑、逐条核对。**任何一条对不上都先报告，不要继续往下走。**

### 3.1 全量测试

```powershell
python -m pytest -q
```

**期望：`575 passed`**（Mac 上就是这个数，见 §9 若数字不同的处理）。

### 3.2 十二个新文件的测试

```powershell
python -m pytest -q tests/test_action_adapter.py tests/test_runtime_runner.py `
    tests/test_runtime_verification.py tests/test_runtime_recording.py `
    tests/test_runtime_observation.py tests/test_week4_cli.py `
    tests/test_week4_prompts.py tests/test_week4_integration.py `
    tests/test_week4_cases.py tests/test_week4_evidence.py `
    tests/test_week4_warmup.py tests/test_week4_demo.py
```

**期望：`217 passed`。**

### 3.3 离线闭环演示

```powershell
python scripts/week4_offline_demo.py
```

**期望**（三行都要对上）：

```
  status      : succeeded
  actions     : 4 dispatched
  verification: passed - all success rules matched against the current screen
```

这个脚本不需要桌面也不需要模型，它证明的是回路机制本身。

---

## 4. 模型服务

> **先调大上下文窗口。** 2560×1600 的截图加元素清单实测 **7 517 tokens**，而 Ollama 默认只给
> 4096，会直接 400 失败。这是**服务端设置，改程序没用**。

```powershell
# 1) 先设变量。OLLAMA_CONTEXT_LENGTH 必须由【服务进程】继承才生效。
$env:GUI_AGENT_API_KEY     = "ollama"
$env:GUI_AGENT_BASE_URL    = "http://127.0.0.1:11434/v1"
$env:OLLAMA_CONTEXT_LENGTH = "16384"

# 2) 看服务是不是还在跑。有输出就说明它是在设变量【之前】启动的。
Get-Process ollama*, llama-server -ErrorAction SilentlyContinue |
    Select-Object Id, ProcessName, @{n='MemMB';e={[math]::Round($_.WorkingSet64/1MB)}}

#    停：托盘图标 -> Quit，或  Stop-Process -Name ollama -Force
#    起：从【刚设好变量的这个窗口】启动，新进程才继承得到

# 3) 确认服务活着、模型在
D:\Developer\Ollama\ollama.exe list
```

**怎么知道变量真的进去了**：做完 §6 的预演后，如果它仍然报
`exceeds the available context size (4096 tokens)`，就是没生效——**回头把重启完整做掉**，
不要带着 4096 往下跑，否则五个用例会全部失败，而原因和代码无关。

---

## 5. 预热（必须在五个任务之前）

`ollama run ... "ok"` 是**纯文本**预热，它通过**不能**说明视觉路径可用，而且什么都不记录。
这一步用**任务将要用的同一个客户端**发文本 + 一张**真实截图**，并把冷/暖状态、内存、耗时、
失败原因落盘。

```powershell
python scripts/week4_warmup.py --config configs/week4.yaml `
    --provider openai_compatible --model qwen2.5vl:7b `
    --json outputs\week4\warmup.json --repeat 2
```

**期望**：三行 probe 都是 `ok`，最后打印 `ready : the model answered text and image`。

**为什么必须在任务之前**：第一次请求一台冷服务要 8–10 秒（内存紧张时约 72 秒）。
不预热的话，**第一个任务的耗时会把模型加载算成自己的规划耗时**——那张结果表里第一个数就会是
事后无法解释的那个。

若输出里出现 `OLLAMA_CONTEXT_LENGTH` 字样 → 回 §4 做完整重启。

记录：`text cold-or-idle` 那一行的毫秒数，回报时要用。

---

## 6. 预演五个用例（不发送任何真实事件）

```powershell
python scripts/week4_agent_cli.py --list-cases
foreach ($c in "T01","T02","T03","T04","T05") {
    python scripts/week4_agent_cli.py --case $c --quiet
}
```

**期望**：五个都是 `Task status : dry_run_completed`、`actions : 1`。

这一轮跑的是**真实截图、真实 OCR、真实提示词、真实适配器、真实记录**，只有模型的判断是模拟的。
它验证的是管线，不是能力。

> 屏幕上必须有可识别的文字。如果某个用例停在第 1 步并提示 `the first frame had no readable
> text`，**先看它上面那行 `observation: ...`**：写着 `ocr unavailable` 就是引擎没起来（回 §2.1）；
> 没有任何 `observation:` 行才是锁屏、黑屏或整屏图片——停在一个有文字的界面再跑。

---

## 7. 五个真实任务（W4-11）

> ### ⚠️ 这一步必须有真人在真实终端上操作
>
> `--execute` 会要求**交互确认**（还要输入倒计时期间的确认），CLI 明确检查
> `sys.stdin.isatty()`：**没有真终端就直接拒绝并退出 2**，管道喂 `y` 也不行——这是刻意的设计，
> "同意"永远不能被假定。
>
> 所以分工是：**你（代理）准备好一切、逐个跑完后的核对与收集由你做；那五条 `--execute` 命令
> 必须由人在一个真实终端窗口里执行**（人可以打字回答 `y`）。若你确实运行在交互式终端里，可以自己跑；
> 若不是，请把命令交给用户执行，然后由你读取产物。
>
> **不要把"我无法交互"当成失败理由去伪造结果。**

### 7.1 先造 T03 需要的测试文件

```powershell
New-Item -ItemType Directory -Force -Path "$env:USERPROFILE\Desktop\week4_test" | Out-Null
@"
WEEK4-OPEN-FILE-OK

This file exists so task T03 has a real target.
"@ | Set-Content "$env:USERPROFILE\Desktop\week4_test\week4_sample.txt"
```

### 7.2 执行顺序与每个用例的前置状态

**建议顺序：T01 → T02 → T03 → T05 → T04（最后，它会真的发消息）。**

命令模板（把 `T01` 换成对应用例）：

```powershell
python scripts/week4_agent_cli.py --case T01 `
    --provider openai_compatible --model qwen2.5vl:7b --execute
```

| 用例 | 动手之前必须先满足 | 算成功 | **不算成功** |
|---|---|---|---|
| **T01** 打开浏览器 | **把所有浏览器窗口关掉**；桌面可见、启动入口没被遮住 | 前台是浏览器窗口，有地址栏或标签栏 | 只按了快捷键、搜索框里出现浏览器名、本来就开着浏览器 |
| **T02** 搜索 | 浏览器窗口已打开**并获得焦点** | 结果页已加载，查询词可见 | 文字只在输入框里、没提交 |
| **T03** 打开文件 | `week4_sample.txt` 存在；同名文件未打开 | 文件已在应用中打开 | 文件只是被选中、开了同名的另一个文件 |
| **T04** 发消息 | 测试会话已打开、不含本次标记；**操作者同意真实发送** | 唯一标记出现在**正确的会话**里 | 只是草稿、旧消息有同样文字、发错会话 |
| **T05** 关闭应用 | **先把 `week4_sample.txt` 打开**（让标记出现在屏幕上）；窗口获得焦点 | 目标窗口消失，其他应用未受影响 | 只是最小化、关错了窗口 |

**T01 和 T05 有执行前的前置状态检查**：如果任务目标在动手之前就已经成立，运行会直接
`blocked`，既不调模型也不点任何东西（文案：`the success rule already holds on the untouched
screen`）。因为这两条的规则"什么都不做也能满足"——而这两种恰好是判定标准里写明不算成功的情况。
**这不是 bug，不要试图绕过它**（这个检查在命令行上关不掉，`require_preconditions` 既不是 CLI
参数也不是配置项，这是刻意的）。

### 7.3 T04 的标记每次运行都不同

CLI 启动时会打印一行：

```
  marker     : WEEK4_MESSAGE_CHECK_<时间戳>
```

**那就是本次要发送、也是本次要核对的文本**，请抄下来。固定的标记会让这个用例只能用一次
（上次的消息还在会话里，规则已成立，前置条件又会拒绝重试）。所以 T04 的"成功"指的是**这个
时间戳的标记**出现在正确的会话里，不是任何旧消息。

**若你或用户不同意真实发送消息**：把该行记作"未实测"并说明原因。**不要**用草稿测试冒充已完成发送。

### 7.4 每个用例跑完后立刻做两件事

```powershell
# 1) 把证据收进仓库
python scripts/week4_collect_evidence.py --latest T01

# 2) 读汇总
Get-Content outputs\week4\T01_*\task_summary.json | Select-String '"status"|"execute"|"execution_ms"|"verification"'
```

**记录每个用例的**：`status`、`execute`（必须是 `true`）、`execution_ms`（或 `elapsed_ms`）、
`verification.outcome`、`run_id`（= 运行目录名）、以及 `notes` 里有没有异常。

> `status=succeeded` **且** `execute=true` 才算真实成功。**预演再干净也不算。**
>
> **第一轮失败也要保留**：调试后重测，**不要删除失败记录**来提高成功率。一个用例跑两次就是
> "两次尝试"，只有其中 `succeeded` 的那次算成功。

---

## 8. 回填报告

把结果填进 **`Document/Week4/Week4_Basic_Task_Test_Report.md`** 的 `## Results` 表格。
列的含义：`Attempts`（正式尝试次数）、`Successes`（`succeeded` 且 `execute=true` 的次数）、
`Status`、`Verification method`（规则怎么判的）、`Timing basis`（用 `execution_ms`）、
`Evidence`（`Document/Week4/evidence/<run id>/`）。

同一时间把中文报告 `Week4_中文实验报告.md` 第 5 节的结果表也填上（如果那份文档在你手上）。

**只填你亲眼看到的值。** 没有跑的行保持 `not run` / `未实测`。

---

## 9. 故障速查

| 现象 | 成因 | 处理 |
|---|---|---|
| `Test-Path scripts\week4_warmup.py` 为 False | 推送还没到 | 停下，让用户先 push |
| 测试数不是 `575` / `217` | 代码版本不对（可能是旧提交） | 报出 `git log --oneline -1` 的提交号，**不要**自行解释成"环境差异" |
| `monitor_index 1 is out of range (available 1..0)` | **两个成因**：屏幕休眠/锁定，或进程根本看不到屏幕 | 判据是 `available 1..0`（只剩索引 0 的聚合伪显示器且尺寸 `0x0`）：那是权限/远程会话问题，**唤醒屏幕没用**。先跑 §2.2 |
| `the first frame had no readable text` | 先看它**上面那行** `observation: ...` | 写着 `ocr unavailable` / `tesseract is not installed` → 回 §2.1 装 exe（唤醒屏幕没用）；没有 `observation:` 行才是锁屏/黑屏 |
| `ocr unavailable: Tesseract failed: ...` | OCR 引擎没起来 | 回 §2.1 |
| `exceeds the available context size (4096 tokens)` | `OLLAMA_CONTEXT_LENGTH` 没被服务进程继承 | 回 §4 做**完整重启**（托盘中重新加载不算） |
| `the success rule already holds on the untouched screen` | 前置状态没满足（T01 有浏览器开着 / T05 文件没打开） | 按 §7.2 调整，重新跑 |
| `no element matches '<文本>'` | 屏幕上找不到那个元素 | 打开 `outputs\week4\<run>\obs-*.json` 看那条元素的文本，**原文报回来** |
| `is not from observation ... and the step names no text target` | 模型只给了元素编号、没给文字目标（重观察后编号必然过期） | **原文报回来**，这是模型行为问题 |
| `--execute needs an interactive terminal` | 非交互环境 | 见 §7 顶部的说明：必须真人在真实终端执行 |
| 退出码 `2` 且 `status=blocked` | 被拦下（前置状态/无规则/无法确认） | 看 `notes` 里的原因 |
| 退出码 `130` | 操作者在确认处回答了 `n` | 正常，说明拒绝路径有效 |

**退出码对照**：`0` 完成（看 `status`）、`1` 失败、`2` 被拦下或参数错误、`3` 超时、`130` 取消。

---

## 10. 回报格式

**只填你实际看到的。** 没有的字段写 `未执行`。

```
【环境】
  提交号               : <git log --oneline -1 的完整输出>
  Python / 解释器      : <sys.version; 是否是 .venv 里的>
  check_environment    : Screen capture = ___ / Tesseract CLI = ___
  显示缩放             : ___%（设置 -> 系统 -> 显示 -> 缩放）

【离线】
  pytest 全量          : ___ passed
  十二文件             : ___ passed
  offline_demo         : status=___ actions=___ verification=___
  CLI --list-cases     : ___（是否列出 T01..T05）
  五个预演             : T01=___ T02=___ T03=___ T04=___ T05=___（各自 status/actions）

【预热】
  warmup ready         : ___   冷启动耗时 = ___ ms   暖启动 = ___ ms

【真实任务】每个用例一行
  T0x: status=___ execute=___ execution_ms=___ verification=___ run_id=___
       attempts=___ successes=___ 备注=___

【异常】
  出现过哪些报错？（原文抄写，不要转述）
  有没有被前置状态检查拦下的？哪一个？
  有没有需要人工接管的步骤？
```

**回报里不要带**：真实 API key、个人聊天内容、未处理的桌面截图。

---

## 11. 明确禁止的事项

1. **不要**把没有执行的命令写成已执行。
2. **不要**把 `blocked`、`failed`、`timed_out` 写成"通过"或以任何方式美化。
3. **不要**删除或编辑 `Document/Week4/evidence/` 里已有的任何文件。
4. **不要**为了让数字对上而去改测试、改报告或改代码。**数字对不上本身就是一条要报告的结果。**
5. **不要**关闭或绕过安全机制：前置状态检查、两次确认、倒计时、fail-safe。
   若某条检查挡住了你，**报告它**，不要绕过。
6. **不要**把预演（dry run）或模拟结果当作真实能力证据。
7. **不要**在未获同意的情况下真实发送消息（T04）。

---

## 12. 已知限制（不是缺陷，不要当 bug 报）

- **五类任务在 Mac 上从未跑过**，所以这是**首次真实测试**，不是复现。第一轮失败是正常的。
- 只支持**主显示器**；多显示器与负坐标未实现。
- `type_text` 用 `pyautogui.typewrite`，**只能输入 ASCII**，中文输入未实现也未声称支持。
- **没有重规划**：某一步失败就停止，不自动恢复（属于 Week 6）。
- **没有自动焦点检查**：目标会在新画面上重新定位，但没有任何代码读取前台窗口来确认键盘焦点。
  缓解是流程性的（中高风险任务会带着"将要输入的文本"再确认一次）。**本项目不声称能自动防止所有误输入。**
- T01/T05 的自动规则比判定标准**弱**（只能看屏幕上有没有某段文字），靠前置状态检查补上另一半。
- 失败分类建议用这几个词：焦点不对、目标歧义、输入不完整、未提交、加载超时、验证不充分。
