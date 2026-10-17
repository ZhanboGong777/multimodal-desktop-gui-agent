---
base_model: Qwen/Qwen2.5-VL-3B-Instruct
library_name: peft
tags:
  - lora
  - gui-agent
  - image-text-to-text
---

# Week5 C11 LoRA

Actual adapter saved after the completed 2,482-row, eight-epoch C11 run.
This directory contains the adapter only. The pinned base revision is
`66285546d2b821cf421d4f5eb2576359d3770cd3`.

The weight file is 14,787,968 bytes, with SHA256
`58b005e9fa13e20df15f28ca09c71c0d61c71617abfeb96d17456982a810a54b`.
It contains 288 FP32 LoRA tensors / 3,686,400 parameters. Training used language
q/k/v/o projections, rank8, alpha16, dropout0.05; the vision encoder was frozen.

The public config replaces the original local base directory with the upstream
model ID and supplies its revision. The adapter weight bytes are unchanged.
Both config hashes are recorded in the [public summary](../../../Document/Week5/results/summary.json).

The original 16 offline gates passed. Desktop verification covered the original
12 distinct messages across three prompt versions and five execution candidates;
it is not a single fixed prompt scoring12/12. The holdout is now exposed and the
desktop evaluation is in-domain adaptation. This model card does not grant any
desktop messaging authorization.

See the [report](../../../Document/Week5/Week5_Final_Report_20261017.md) and
[loading instructions](../../../Document/Week5/Reproduction.md). The base model
and its upstream terms are available from
[Qwen](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct/tree/66285546d2b821cf421d4f5eb2576359d3770cd3).
