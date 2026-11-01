#!/usr/bin/env python
"""Build a compact Week 5 report from recorded evidence using only the stdlib.

No model, CUDA, GUI, downloads, or model-weight copies are used. --output is
the Markdown filename; facts.json defaults to the evaluation directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

CELLS = ("base_original", "base_improved", "adapter_original", "adapter_improved")
RATE_KEYS = (
    "strict_json_valid",
    "strict_action_exact",
    "action_type_accuracy",
    "text_exact_match",
    "click_bbox_hit",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_hash(value: Any) -> str:
    text = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object: {path}")
    return payload


def cell(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ").replace("\r", " ")


def number(value: Any, *, decimals: int = 3) -> str:
    if value is None:
        return "—"
    if type(value) not in (int, float, bool) or not math.isfinite(value):
        raise ValueError("report numeric field must be finite or null")
    return f"{value:.{decimals}f}"


def rate(metric: dict[str, Any]) -> str:
    correct, total, value = metric["correct"], metric["total"], metric["rate"]
    if not 0 <= correct <= total:
        raise ValueError("inconsistent metric numerator/denominator")
    if total == 0:
        if value is not None:
            raise ValueError("empty metric denominator must have null rate")
        return "— (0/0)"
    if value is None or not math.isclose(value, correct / total, rel_tol=1e-10, abs_tol=1e-12):
        raise ValueError("metric rate differs from numerator/denominator")
    return f"{100 * value:.2f}% ({int(correct)}/{total})"


def table(headers: list[str], rows: list[list[Any]]) -> str:
    return "\n".join(
        [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join("---" for _ in headers) + " |",
            *("| " + " | ".join(cell(value) for value in row) + " |" for row in rows),
        ]
    )


def link(path: Path, label: str) -> str:
    return f"[{label}](<{path.resolve().as_posix()}>)"


def extract(args: argparse.Namespace) -> dict[str, Any]:
    training_dir, data_dir, evaluation_dir = (
        Path(args.training_dir).resolve(),
        Path(args.data_dir).resolve(),
        Path(args.evaluation_dir).resolve(),
    )
    paths = {
        "training_summary": training_dir / "training_summary.json",
        "training_config": training_dir / "training_config.json",
        "data_manifest": data_dir / "manifest.json",
        "evaluation_summary": evaluation_dir / "summary.json",
        "evaluation_metadata": evaluation_dir / "metadata.json",
        "evaluation_raw": evaluation_dir / "raw.jsonl",
    }
    inputs = {name: {"path": str(path), "sha256": sha256(path)} for name, path in paths.items()}
    train = read_json(paths["training_summary"])
    config = read_json(paths["training_config"])
    prep = read_json(paths["data_manifest"])
    summary = read_json(paths["evaluation_summary"])
    metadata = read_json(paths["evaluation_metadata"])
    if summary["schema_version"] != "week5.offline-evaluation.v1":
        raise ValueError("unsupported evaluation schema")
    if (
        summary["scope"] != "offline_gui_action_generation"
        or summary["desktop_task_e2e"] is not False
    ):
        raise ValueError("report refuses to mix offline generation with desktop task E2E")
    if train["config"] != config or metadata["inference_config"]["training_config"] != config:
        raise ValueError("training and evaluation effective configurations differ")
    if prep["schema_version"] != 1:
        raise ValueError("unsupported preparation schema")
    splits = {}
    for name in ("train", "validation", "test"):
        output = prep["outputs"][name]
        path = data_dir / f"{name}.jsonl"
        digest = sha256(path)
        if digest != output["sha256"]:
            raise ValueError(f"{name} JSONL differs from preparation manifest")
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if (
            len(rows) != output["samples"]
            or len({row["group_id"] for row in rows}) != output["groups"]
        ):
            raise ValueError(f"{name} counts differ from preparation manifest")
        splits[name] = {
            "samples": output["samples"],
            "groups": output["groups"],
            "sha256": digest,
            "path": str(path),
        }
        inputs[f"data_{name}"] = {"path": str(path), "sha256": digest}
    if sum(item["samples"] for item in splits.values()) != prep["retained_samples"]:
        raise ValueError("retained sample count differs from split counts")
    if prep["input_records"] != prep["retained_samples"] + prep["excluded_samples"]:
        raise ValueError("preparation input/exclusion denominator does not close")
    if sum(prep["exclusion_counts"].values()) != prep["excluded_samples"]:
        raise ValueError("exclusion reasons do not close")
    if metadata["selection"]["dataset_sha256"] != splits["test"]["sha256"]:
        raise ValueError(
            "evaluation selection does not come from the manifest's official test JSONL"
        )
    for field, split in (("train_data", "train"), ("validation_data", "validation")):
        if train.get(field) is not None and train[field]["jsonl_sha256"] != splits[split]["sha256"]:
            raise ValueError(f"{field} snapshot differs from the data manifest")
    raw = [
        json.loads(line)
        for line in paths["evaluation_raw"].read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    requests = [row for row in raw if row["record_type"] == "request"]
    verdicts = {row["run_id"]: row for row in raw if row["record_type"] == "provenance_verdict"}
    selected = {row["sample_id"] for row in metadata["selection"]["samples"]}
    grouped: dict[str, list[dict[str, Any]]] = {}
    identities = set()
    for request in requests:
        identity = (request["cell"], request["sample_id"])
        if identity in identities or request["sample_id"] not in selected:
            raise ValueError("duplicate or unregistered raw evaluation sample")
        identities.add(identity)
        if request["study_id"] != metadata["study_id"] or request["run_id"] != metadata["run_id"]:
            raise ValueError("raw request and metadata cohort identities differ")
        grouped.setdefault(request["cell"], []).append(request)
    metric_cells = []
    for recorded in summary["cells"]:
        name = recorded["cell"]
        if name not in CELLS or recorded["study_id"] != metadata["study_id"]:
            raise ValueError("unknown cell or mixed study in evaluation summary")
        jobs = grouped.pop(name, [])
        if len(jobs) != recorded["requests"]:
            raise ValueError(f"raw request denominator differs for {name}")
        attempted = sum(row["inference_attempted"] for row in jobs)
        statuses = Counter()
        for row in jobs:
            verdict = verdicts.get(row["run_id"])
            statuses[
                "provenance_pending"
                if verdict is None
                else "provenance_changed"
                if not verdict["unchanged"]
                else row["status"]
            ] += 1
        if dict(statuses) != recorded["statuses"] or attempted != recorded["inference_attempted"]:
            raise ValueError(f"raw statuses/attempt counts differ for {name}")
        for key in RATE_KEYS:
            rate(recorded[key])
        for key in ("strict_json_valid", "strict_action_exact"):
            if recorded[key]["total"] != len(jobs):
                raise ValueError(f"{key} denominator differs from all requests for {name}")
        duration = recorded["percall_e2e_ms_all_recorded_requests"]
        if duration["n"] + recorded["duration_missing"] != len(jobs):
            raise ValueError("duration coverage differs from request denominator")
        for key in ("percall_e2e_ms_all_recorded_requests", "percall_e2e_ms_strict_action_success"):
            if recorded[key]["n"] == 0 and any(
                recorded[key][p] is not None for p in ("p50", "p90", "p95")
            ):
                raise ValueError("zero latency sample count must have null percentiles")
        metric_cells.append(recorded)
    if grouped:
        raise ValueError("raw contains cells absent from the summary")
    source_before, source_after = train["source_hashes"], train.get("source_hashes_after")
    evidence = train.get("artifact_file_evidence", {})
    checkpoint = {
        name: {
            "path": item.get("path"),
            "required_files_present": item.get("required_files_present"),
            "reload_verified_by_training": item.get("reload_verified"),
            "file_count": len(item.get("files", {})),
            "file_manifest_sha256": json_hash(item.get("files", {})),
        }
        for name, item in evidence.items()
        if name in ("adapter", "processor", "partial_adapter")
    }
    steps = train["step_metrics"]
    training_facts = {
        key: train.get(key)
        for key in (
            "status",
            "run_kind",
            "started_at",
            "finished_at",
            "termination_reason",
            "optimizer_steps",
            "optimizer_attempts",
            "skipped_optimizer_steps",
            "micro_steps",
            "samples_seen",
            "epochs_completed",
            "elapsed_seconds",
            "model_load_seconds",
            "trainable_parameters",
            "frozen_visual_parameters",
            "peak_cuda_allocated_bytes",
            "adapter_parameters_changed",
            "initial_adapter_parameter_sha256",
            "final_adapter_parameter_sha256",
            "loss_scope",
            "accumulation_objective",
            "error",
        )
    }
    training_facts.update(
        {
            "config": config,
            "versions": train["versions"],
            "source_hashes": source_before,
            "source_hashes_after": source_after,
            "source_unchanged": source_before == source_after if source_after is not None else None,
            "first_group_mean_example_loss": steps[0]["mean_example_loss"] if steps else None,
            "last_group_mean_example_loss": steps[-1]["mean_example_loss"] if steps else None,
            "validation_metrics": train["validation_metrics"],
            "checkpoint_evidence": checkpoint,
            "base_model_provenance_status": train.get("base_model", {}).get("provenance_status"),
        }
    )
    facts = {
        "schema_version": "week5.report-facts.v1",
        "generated_at": datetime.now(UTC).isoformat(),
        "scope": "offline_gui_action_generation",
        "desktop_task_e2e": False,
        "input_files": inputs,
        "training": training_facts,
        "data": {
            key: prep[key]
            for key in (
                "seed",
                "validation_fraction",
                "target_format",
                "image_hash",
                "split_policy",
                "input_records",
                "retained_samples",
                "excluded_samples",
                "exclusion_counts",
            )
        },
        "evaluation": {
            key: summary[key]
            for key in (
                "schema_version",
                "scope",
                "quantile_method",
                "strict_success_definition",
                "latency_scope",
                "warmup_calls",
                "comparisons",
                "paired_comparisons",
                "missing_distance_penalty",
                "startup_errors",
            )
        },
    }
    facts["data"].update(
        {
            "splits": splits,
            "sources": [
                {
                    key: source.get(key)
                    for key in ("dataset", "source_split", "origin_url", "revision", "fingerprint")
                }
                for source in prep["sources"]
            ],
        }
    )
    facts["evaluation"].update(
        {
            "study_id": metadata["study_id"],
            "run_id": metadata["run_id"],
            "cells": metric_cells,
            "selection": {
                key: metadata["selection"][key]
                for key in ("seed", "cap", "dataset_sha256", "selection_method")
            },
            "selected_samples": len(selected),
            "raw_requests": len(requests),
            "raw_warmup_calls": sum(row["record_type"] == "warmup" for row in raw),
            "runtime_versions": metadata["runtime_versions"],
            "inference_config": metadata["inference_config"],
            "percall_timeout_seconds": metadata["percall_timeout_seconds"],
            "load_timeout_seconds": metadata["load_timeout_seconds"],
            "selection_frozen_before_generation": metadata["selection_frozen_before_generation"],
            "prompt_version": metadata["prompt_version"],
            "prompts": metadata["prompts"],
            "provenance_verdicts": list(verdicts.values()),
            "artifact_sha256": {
                key: metadata[key]
                for key in (
                    "base_artifact_sha256",
                    "adapter_artifact_sha256",
                    "processor_artifact_sha256",
                )
            },
            "source_manifest_sha256": json_hash(metadata["source_files"]),
            "raw_audit": "request identities, counts, statuses and duration coverage checked; metric values read from evaluator summary",
        }
    )
    if facts["evaluation"]["raw_warmup_calls"] != summary["warmup_calls"]:
        raise ValueError("warmup denominator differs between raw and summary")
    # Verify snapshots again before writing: a running trainer/evaluator is not
    # allowed to silently yield a report assembled from inconsistent moments.
    if any(sha256(Path(item["path"])) != item["sha256"] for item in inputs.values()):
        raise ValueError("an input evidence file changed while building the report")
    return facts


def render(facts: dict[str, Any], facts_path: Path) -> str:
    training, data, evaluation = facts["training"], facts["data"], facts["evaluation"]
    config = training["config"]
    lines = [
        "# Week 5 LoRA 实验报告",
        "",
        f"生成时间：{facts['generated_at']}。训练状态 `{training['status']}`，运行类型 `{training['run_kind']}`。",
        (
            f"本报告口径为 **offline GUI action generation**：评测留证包含 {evaluation['raw_requests']} 条请求，"
            f"每组固定候选列表包含 {evaluation['selected_samples']} 个官方 test 样本。"
        ),
        (
            "评测没有执行桌面动作；请求级推理耗时不等于教师要求的真实桌面 CLI 完整 E2E。"
            "本实验是同一 2B 基座与其 LoRA 的对照，不能据此声称 Week4 7B 或 T04 已改善。"
        ),
        "",
        "## 训练与数据",
        "",
        (
            f"基座 `{config['base_model_id']}`，revision `{config['base_model_revision']}`。"
            f"4bit={config['load_in_4bit']}，dtype={config['compute_dtype']}，attention={config['attention_implementation']}，"
            f"LoRA rank/alpha={config['lora_rank']}/{config['lora_alpha']}，学习率={config['learning_rate']}。"
            f"训练仅使用 `{config['prompt_variant']}` prompt，改进 prompt 仅用于固定对照。"
        ),
        "",
        table(
            ["分区", "样本", "session groups", "JSONL SHA256"],
            [
                [name, split["samples"], split["groups"], split["sha256"]]
                for name, split in data["splits"].items()
            ],
        ),
        "",
        (
            f"原始记录 {data['input_records']}，保留 {data['retained_samples']}，排除 {data['excluded_samples']}。"
            f"划分规则：{data['split_policy']}；seed={data['seed']}。"
        ),
        "",
        table(
            ["排除原因", "记录数"],
            [[reason, count] for reason, count in data["exclusion_counts"].items()],
        ),
        "",
        table(
            ["训练指标", "记录值"],
            [
                [key, training[key]]
                for key in (
                    "optimizer_steps",
                    "optimizer_attempts",
                    "skipped_optimizer_steps",
                    "micro_steps",
                    "samples_seen",
                    "epochs_completed",
                    "trainable_parameters",
                    "frozen_visual_parameters",
                    "adapter_parameters_changed",
                )
            ],
        ),
        "",
        (
            f"总训练留证耗时 {number(training['elapsed_seconds'])} 秒，模型加载 {number(training['model_load_seconds'])} 秒；"
            f"峰值 CUDA allocated {training['peak_cuda_allocated_bytes']} bytes。"
            f"首组/末组 mean-example loss 为 {number(training['first_group_mean_example_loss'], decimals=6)} / "
            f"{number(training['last_group_mean_example_loss'], decimals=6)}，仅作轨迹描述。"
        ),
        "",
        table(
            ["验证 epoch", "样本", "mean-example loss", "token-weighted loss"],
            [
                [
                    row["epoch"],
                    row["examples"],
                    number(row["mean_example_loss"], decimals=6),
                    number(row["token_weighted_loss"], decimals=6),
                ]
                for row in training["validation_metrics"]
            ],
        ),
        "",
        "## 固定四组离线对照",
        "",
        (
            f"seed={evaluation['selection']['seed']}，预注册上限={evaluation['selection']['cap']}。"
            "模型输入仅有截图、任务指令和原始图像尺寸；不提供动作标签、点击框、sample ID 或源文件名。"
            "strict success 定义为完整 native-pixel 动作数组与标签完全相等。bbox 命中单独报告，不能替代整动作 exact。"
        ),
        f"固定选样方法：{evaluation['selection']['selection_method']}。生成前冻结={evaluation['selection_frozen_before_generation']}。",
        (
            f"max_new_tokens={evaluation['inference_config']['max_new_tokens']}，"
            f"单请求 hard timeout={evaluation['percall_timeout_seconds']} 秒，"
            f"worker 加载 timeout={evaluation['load_timeout_seconds']} 秒。"
        ),
        "",
    ]
    by_cell = {row["cell"]: row for row in evaluation["cells"]}
    rows = []
    for name in CELLS:
        if name not in by_cell:
            rows.append([name, "未记录", *(["—"] * 6)])
            continue
        row = by_cell[name]
        errors = (
            ", ".join(f"{key}={value}" for key, value in row["statuses"].items() if key != "ok")
            or "0"
        )
        rows.append([name, row["requests"], *(rate(row[key]) for key in RATE_KEYS), errors])
    lines += [
        table(
            [
                "cell",
                "请求 N",
                "strict JSON",
                "whole action exact",
                "action type",
                "text exact",
                "bbox hit",
                "非 ok 状态",
            ],
            rows,
        ),
        "",
        (
            "非法格式、异常、超时和取消保留在请求分母；未启动请求单列，不能伪造推理耗时。"
            "action-type 以索引对齐，分母为每样本 max(标签动作数,有效预测动作数)，额外动作计错。"
            "text exact 保留空格和 Unicode；bbox 分母只含真实标注框，边界采用半开区间。"
        ),
        "",
        "## 请求级推理耗时",
        "",
        "下表单位毫秒，采用 nearest-rank ceil(p×N)。模型加载及预热另计；N=0 的分位数留空。",
        "",
    ]
    for key, label in (
        ("percall_e2e_ms_all_recorded_requests", "全部有计时请求"),
        ("percall_e2e_ms_strict_action_success", "whole-action strict success 请求"),
    ):
        lines += [
            f"{label}：",
            "",
            table(
                ["cell", "计时 N", "P50", "P90", "P95", "缺失耗时"],
                [
                    [
                        row["cell"],
                        row[key]["n"],
                        *(number(row[key][p]) for p in ("p50", "p90", "p95")),
                        row["duration_missing"],
                    ]
                    for row in evaluation["cells"]
                ],
            ),
            "",
        ]
    lines += [
        f"独立预热调用 {evaluation['raw_warmup_calls']} 次，不计上述请求指标。",
        "",
        table(
            [
                "cell",
                "坐标距离 observed N",
                "observed mean",
                "含缺失处罚 mean",
                "缺失预测",
                "无真实 bbox",
            ],
            [
                [
                    row["cell"],
                    row["click_distance_observed"]["n"],
                    number(row["click_distance_observed"]["mean"], decimals=6),
                    number(row["click_distance_all_targets_penalty_sqrt2"]["mean"], decimals=6),
                    row["click_distance_missing"],
                    row["click_bbox_absent"],
                ]
                for row in evaluation["cells"]
            ],
        ),
        "",
        (
            "归一化距离为 hypot(dx/width,dy/height)。全目标距离另含缺失预测 sqrt(2) 处罚；"
            "处罚不代表观测到的位置，不用仅有有效预测的均值掩盖缺失。"
        ),
        "",
        "## 配对变化与出处",
        "",
        "delta 为 after − before 的比例变化；这里只描述相同样本对照，不声称统计显著。",
        "",
        table(
            ["同样本集", "完整 2×2", "四组交集 N", "并集 N"],
            [
                [
                    row["same_recorded_sample_sets"],
                    row["complete_2x2_same_samples"],
                    row["shared_sample_count"],
                    row["union_sample_count"],
                ]
                for row in evaluation["comparisons"]
            ],
        ),
        "",
        table(
            ["比较", "配对 N", "可比较", "strict exact delta", "strict JSON delta", "改善/恶化"],
            [
                [
                    row["comparison"],
                    row["paired_samples"],
                    row["comparison_eligible"],
                    number(row["strict_success_rate_delta_after_minus_before"], decimals=6),
                    number(row["strict_json_rate_delta_after_minus_before"], decimals=6),
                    f"{row['outcomes'].get('improved', 0)}/{row['outcomes'].get('worsened', 0)}",
                ]
                for row in evaluation["paired_comparisons"]
            ],
        ),
        "",
        (
            f"study ID：`{evaluation['study_id']}`。训练源文件前后相同：{training['source_unchanged']}。"
            f"基座下载出处状态：{training['base_model_provenance_status']}。"
        ),
        "",
        table(
            ["评测冻结对象", "SHA256"],
            [[key, value] for key, value in evaluation["artifact_sha256"].items()],
        ),
        "",
        (
            f"评测源码 manifest 摘要 SHA256：`{evaluation['source_manifest_sha256']}`。"
            f"final provenance unchanged：{[row['unchanged'] for row in evaluation['provenance_verdicts']]}。"
        ),
        "",
        (
            "训练保存文件是否齐全与文件哈希来自 checkpoint 留证；训练侧 reload_verified 字段仍按原值保留。"
            "评测重新加载的 adapter 与其推理记录另查 raw/metadata，不能把保存成功当作行为成功。"
        ),
        "",
        table(
            ["数据来源", "split", "revision", "fingerprint"],
            [
                [row["dataset"], row["source_split"], row["revision"], row["fingerprint"]]
                for row in data["sources"]
            ],
        ),
        "",
        table(
            ["软件", "评测版本"],
            [[key, value] for key, value in evaluation["runtime_versions"].items()],
        ),
        "",
        "## 限制与 Week4 T04 P0 后续",
        "",
        (
            "离线截图的 next-action 标签与真实桌面完整任务不同；坐标 exact 会拒绝框内其他可接受点，因此 bbox hit 同时列出。"
            "样本规模、数据领域和硬件限制了结论外推；零成功、缺失耗时或负 delta 都应保留，不能据 loss 下降宣称 GUI 可靠性提高。"
        ),
        "",
        "- P0：继续用独立正确会话、完整 marker、已发送状态和空编辑区做 T04 真桌面验收，保留失败与 blocked 分母。",
        "- P0：验证前台准备及指针遮挡，保持 editor/send 角色授权、当前截图与发送前精确像素检查，禁止误重发。",
        "- P0：保持草稿与消息区域严格读取，复核模型漏字、旧消息和组合态误读；单独记录模型与桌面条件的失败原因。",
        "- P0：用冻结源码/配置/预热口径重新收集连续样本，独立报告 CLI 全 E2E、人工确认和系统 P50/P90/P95；不与本表合并。",
        "",
        "## 可复核文件",
        "",
    ]
    for name, evidence in facts["input_files"].items():
        lines += [f"- {link(Path(evidence['path']), name)}；SHA256 `{evidence['sha256']}`。"]
    lines += [f"- {link(facts_path, 'facts.json')} 保存结构化数字、配置和出处摘要。", ""]
    return "\n".join(lines)


def write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".report-tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-dir", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--evaluation-dir", required=True)
    parser.add_argument("--output", required=True, help="Markdown report filename")
    parser.add_argument("--facts", help="default <evaluation-dir>/facts.json")
    args = parser.parse_args(argv)
    try:
        output = Path(args.output).resolve()
        facts_path = (
            Path(args.facts).resolve()
            if args.facts
            else Path(args.evaluation_dir).resolve() / "facts.json"
        )
        facts = extract(args)
        protected = {Path(evidence["path"]).resolve() for evidence in facts["input_files"].values()}
        if output == facts_path or output in protected or facts_path in protected:
            raise ValueError("report output must not replace an input evidence file")
        document = render(facts, facts_path)
        write_atomic(output, document)
        facts["report"] = {"path": str(output), "sha256": sha256(output)}
        write_atomic(
            facts_path, json.dumps(facts, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        )
        print(f"Report: {output}\nFacts: {facts_path}")
        return 0
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"report error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
