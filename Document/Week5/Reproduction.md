# Week5复现与权重加载

## 环境

只检查交付完整性时可直接运行`python scripts/week5_verify_submission.py`，无需安装ML依赖；这项哈希检查不重新证明模型质量或桌面成功。

使用Python 3.12和独立虚拟环境。原实验记录的版本为torch 2.14.0+cu126、transformers 4.57.6、peft 0.18.1、accelerate 1.12.0、bitsandbytes 0.49.0、Pillow 12.3.0。`requirements-week5.txt`固定Python依赖，CUDA版PyTorch按目标机器单独安装。CPU测试不需要下载基座模型。

```powershell
python -m venv .venv-week5
.\.venv-week5\Scripts\python.exe -m pip install -r requirements-dev.txt -r requirements-week5.txt
.\.venv-week5\Scripts\python.exe -m pytest -q
```

## 现有LoRA

权重位于`artifacts/week5/c11/adapter_model.safetensors`，SHA256为`58b005e9fa13e20df15f28ca09c71c0d61c71617abfeb96d17456982a810a54b`，288个FP32张量、3,686,400参数。它需要`Qwen/Qwen2.5-VL-3B-Instruct`固定revision `66285546d2b821cf421d4f5eb2576359d3770cd3`。

基座不在仓库内。取得该固定版本后保存在`models/Qwen2.5-VL-3B-Instruct/`，包括模型、tokenizer及processor文件。可用huggingface_hub显式下载：

```python
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="Qwen/Qwen2.5-VL-3B-Instruct",
    revision="66285546d2b821cf421d4f5eb2576359d3770cd3",
    local_dir="models/Qwen2.5-VL-3B-Instruct",
)
```

加载实际最终权重：

```python
from gui_agent.finetuning.c11_training import recipe
from gui_agent.finetuning.training import load_model_and_processor
model, processor = load_model_and_processor(
    recipe("models/Qwen2.5-VL-3B-Instruct"),
    adapter_path="artifacts/week5/c11", for_training=False,
)
```

先执行`python -m pip install -e .`，或把`src`加入PYTHONPATH。加载会使用本地文件、NF4双重量化、BF16、SDPA，不会自动下载，也不操作桌面。公开adapter_config把原本机路径改为模型ID并补上revision；权重字节没有改动，两份配置哈希分别记录在summary中。使用前同时核对基座上游许可。

## 数据与新训练

原C11为1458条保留数据加1024条ShowUI新增标注，共2482条；其中ShowUI共2048条、101张公开截图。ShowUI固定源revision为`0d190dfee8ca2cb24f1a7157f8bdde52845d5348`。本仓库不重新分发原始训练截图、微信适配截图、已暴露评估输入或私有receipt，因此不能仅靠一次clone逐字节重建原训练集或重新证明原16项门槛。

恢复的`week5_download_assets.py`、`week5_prepare_data.py`及`gui_agent.finetuning.data`用于早期ScreenAgent流程。它们的默认Qwen2-VL配置是历史候选，不能当成最终C11命令。最终C11几何、损失、内存实现分别保存在`resized_contract.py`、`structured_ce.py`、`embedding_memory.py`，与实验使用源码逐字节相同。

新增可移植入口`week5_c11.py`把这些数值实现接到独立的新数据资格检查和训练流程。它是本次提交整理后的入口，已经做CPU检查；本次没有重新执行其完整GPU训练。原实验的机器绑定owner、签名和桌面激活通道仍只属于私有历史凭据。

准备自己的JSONL训练集和独立验证集。每行含`sample_id`、`group_id`、`source`、`source_split`、`instruction`、`image_path`、`image_width`、`image_height`、`actions`；图像路径相对该JSONL。actions使用原始截图像素坐标。验证集在sample、group、图像RGB哈希上均须与训练集分离。训练集必须恰好2482行且source_split为train；原测试数据不得混入。

```json
{"sample_id":"example-train-001","group_id":"example-image-001","source":"own_dataset","source_split":"train","instruction":"Click the search control","image_path":"images/example.png","image_width":780,"image_height":600,"actions":[{"action_type":"click","x":100,"y":120}]}
```

以下命令默认只打印计划；加`--execute`才实际准备或训练。输出目录必须不存在，不覆盖、不续训、不跳过失败行：

```powershell
python scripts/week5_c11.py prepare --train-jsonl data/processed/week5/train.jsonl --validation-jsonl data/processed/week5/validation.jsonl --base-model models/Qwen2.5-VL-3B-Instruct --output outputs/week5_portable_prepared --execute
python scripts/week5_c11.py train --prepared outputs/week5_portable_prepared --output outputs/week5_portable_train --execute
```

prepare仅加载CPU processor，对每行分别编码无标签prefix及完整assistant目标，保留完整JSON/EOS、不截断，核验真实图像网格，保存token资格及固定权重。train重新检查基座树和数据字节，按seed42+epoch顺序跑8轮；微批1、累积8，最后两行按2行平均。结构6／坐标4／其余活动token与EOS2，分母为原shifted活动token数。实际新训练的结果单独记录，不冒充历史C11的权重或通过凭据。

## 历史验收的边界

历史离线16项全部通过，开发bbox为19/24，C11封存留出bbox为15/24。留出Windows图标2/6恰在原最低边界；留出现在已经暴露，不能再作为新留出。公开summary是从原闭合记录提取的历史结果，不是本次CPU测试重新跑出的模型成绩。

桌面原12条消息已全部消耗，不允许重复发送。本公开版本不提供自动发送微信消息的脚本；离线生成不能替代SDK执行、坐标映射、取消能力与实际气泡核验。

最终C17/C19自然语言提示保存在`gui_agent.finetuning.t04_prompts`，只返回文字，不含原标记或执行权限。提示输入接到`resized_contract.build_messages`时仍只暴露请求、阶段与原生图像信息，不加入答案坐标。
