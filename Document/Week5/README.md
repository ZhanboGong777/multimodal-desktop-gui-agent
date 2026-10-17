# Week5 GitHub 提交

Week5已完成原约定范围。本目录是适合GitHub提交的公开版本，完整私有实验记录继续保存在本机交付包。

- [最终报告](Week5_Final_Report_20261017.md)：训练前后对照、原16项离线门槛、12条实机范围与失败诊断。
- [复现与加载](Reproduction.md)：依赖、数据格式、C11训练入口和LoRA加载。
- [公开结果](results/summary.json)：实际训练计数、离线分项、实机候选划分与权重SHA256。
- [来源索引](results/provenance.json)：恢复源码及私有历史凭据的哈希。
- [LoRA](../../artifacts/week5/c11/)：实际C11保存权重，基座模型另行准备。
- [GitHub提交步骤](GitHub_Submission.md)。

clone后可先运行`python scripts/week5_verify_submission.py`，只用Python标准库核对本次交付文件的SHA256，不加载模型、不操作桌面。

最终基座为Qwen2.5-VL-3B-Instruct。恢复的历史源码还包含早期Qwen2-VL、pointer与spatial head方案；这些方案的测试不等于最终C11训练或桌面验收。最终12条由3版提示、5个有成功结果的执行候选分段核验，不能写成单一提示12/12。
