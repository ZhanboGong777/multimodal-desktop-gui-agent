"""Measure the CLI's complete lifetime; never dispatch desktop actions by default.

``plan`` prints commands without starting a child. ``run`` launches dry-run CLI
children unless --execute is explicit; execute children inherit interactive stdin
and keep both CLI confirmations. ``aggregate`` only reads saved benchmark JSON.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import signal
import subprocess
import sys
import threading
import time
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "week4.e2e.v1"
CASE_IDS = ("T01", "T02", "T03", "T04", "T05")
STATUS_EXITS = {
    "succeeded": 0,
    "dry_run_completed": 0,
    "failed": 1,
    "blocked": 2,
    "timed_out": 3,
    "cancelled": 130,
}


def canonical_hash(value: Any) -> str:
    data = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def source_snapshot(repo: Path) -> dict[str, Any]:
    """Identify dirty builds too, without recording source/credentials in metadata."""

    def git(*args: str) -> bytes:
        return subprocess.check_output(["git", *args], cwd=repo, stderr=subprocess.DEVNULL)

    commit = git("rev-parse", "HEAD").decode("ascii").strip()
    status = git("status", "--porcelain", "-z")
    diff = git("diff", "--binary", "HEAD")
    digest = hashlib.sha256(status + diff)
    for name in sorted(git("ls-files", "--others", "--exclude-standard", "-z").split(b"\0")):
        if not name:
            continue
        path = repo / os.fsdecode(name)
        digest.update(name)
        if path.is_file():
            digest.update(hashlib.sha256(path.read_bytes()).digest())
    source_digest = hashlib.sha256()
    names = set(git("ls-files", "-z").split(b"\0"))
    names.update(git("ls-files", "--others", "--exclude-standard", "-z").split(b"\0"))
    for name in sorted(names):
        if not name:
            continue
        relative = os.fsdecode(name).replace("\\", "/")
        if not (
            relative.startswith(("src/", "scripts/", "configs/", "requirements"))
            or relative == "pyproject.toml"
        ):
            continue
        path = repo / os.fsdecode(name)
        if path.is_file():
            source_digest.update(name)
            source_digest.update(hashlib.sha256(path.read_bytes()).digest())
    return {
        "commit": commit,
        "dirty": bool(status),
        "working_tree_sha256": digest.hexdigest(),
        "source_tree_sha256": source_digest.hexdigest(),
        "dirty_diff_sha256": hashlib.sha256(diff).hexdigest(),
    }


def resolve_configuration(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, str]]:
    """Use the same dotenv, model precedence and overrides as the child CLI."""
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from dotenv import dotenv_values

    from gui_agent.config import load_config, resolve_model_config

    child_env = dict(os.environ)
    for key, value in dotenv_values(REPO_ROOT / ".env").items():
        if value is not None:
            child_env.setdefault(key, value)
    # The benchmark only addresses this controlled conversation. A user's .env
    # override must not silently aim the repeated send case at somebody else.
    child_env["GUI_AGENT_MESSAGE_CONVERSATION"] = "文件传输助手"
    # An absent key would be reloaded from .env by the child. Set a safe explicit
    # value instead; the CLI replaces this case's base marker with --message-marker.
    child_env["GUI_AGENT_MESSAGE_INSTRUCTION"] = (
        "Type WEEK4_MESSAGE_CHECK_001 into the message box of the conversation named "
        '"文件传输助手", which is already open, then click the send button. '
        "Do not press Enter at any point."
    )
    child_env["PYTHONIOENCODING"] = "utf-8"
    child_env["PYTHONUNBUFFERED"] = "1"
    config = load_config(args.config)
    resolve_model_config(
        config,
        provider=args.provider,
        model=args.model,
        base_url=args.base_url,
        environ=child_env,
    )
    for arg, attribute in (
        ("max_actions", "max_actions"),
        ("task_timeout", "task_timeout_seconds"),
        ("planning_attempts", "max_planning_attempts"),
    ):
        value = getattr(args, arg)
        if value is not None:
            setattr(config.execution, attribute, value)
    if args.ocr_engine:
        config.perception.ocr.engine = args.ocr_engine
        config.perception.ocr.fallback_engine = args.ocr_engine
    if args.ocr_min_confidence is not None:
        config.perception.ocr.min_confidence = args.ocr_min_confidence
    effective = config.model_dump(mode="json")
    # Output folders are per request; they are not a benchmark configuration change.
    for section in ("output", "logging"):
        effective[section]["directory"] = "<per-run-output>"
    return effective, child_env


def new_identity(case_id: str) -> tuple[str, str]:
    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S_%f")
    suffix = uuid4().hex[:8]
    return (
        f"{case_id}_{stamp}_{suffix}",
        f"WEEK4_MESSAGE_CHECK_{stamp[:15]}_{suffix}" if case_id == "T04" else "",
    )


def child_command(
    args: argparse.Namespace,
    config: dict[str, Any],
    case_id: str,
    run_id: str,
    marker: str,
    directory: Path,
    frozen_config: Path | None = None,
) -> list[str]:
    model = config["model"]
    command = [
        args.python,
        "-u",
        str(REPO_ROOT / "scripts" / "week4_agent_cli.py"),
        "--config",
        str(frozen_config or Path(args.config).resolve()),
        "--case",
        case_id,
        "--run-id",
        run_id,
        "--output-directory",
        str(directory / "cli"),
        "--provider",
        model["provider"],
        "--model",
        model["model_name"],
    ]
    if model.get("base_url"):
        command += ["--base-url", model["base_url"]]
    if marker:
        command += ["--message-marker", marker]
    for flag in (
        "max_actions",
        "task_timeout",
        "planning_attempts",
        "ocr_engine",
        "ocr_min_confidence",
    ):
        if getattr(args, flag) is not None:
            command += ["--" + flag.replace("_", "-"), str(getattr(args, flag))]
    if args.execute:
        command.append("--execute")
    return command


def _terminate_tree(process: subprocess.Popen[bytes]) -> None:
    """Stop this measured child and its descendants, never another application."""
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    else:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        if os.name != "nt":
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        process.kill()
        process.wait()


def measure_process(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    timeout: float,
    output: Path,
    tee: bool = True,
) -> dict[str, Any]:
    """Time launch, imports, requests, retries, confirmations, printing and exit.

    stdin is inherited, rather than a pipe that would make --execute noninteractive.
    Waiting for the output reader also includes receiving the child's final bytes.
    Parent-side JSON aggregation is outside this request latency.
    """
    started_at = utc_now()
    started = time.perf_counter_ns()
    timed_out = False
    cancelled = False
    process = None
    reader_error: list[str] = []
    try:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            stdin=None,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            start_new_session=os.name != "nt",
        )

        def read_output() -> None:
            assert process is not None and process.stdout is not None
            try:
                with output.open("wb") as log:
                    while chunk := process.stdout.read1(8192):
                        log.write(chunk)
                        log.flush()
                        if tee:
                            buffer = getattr(sys.stdout, "buffer", None)
                            if buffer is not None:
                                buffer.write(chunk)
                            else:
                                sys.stdout.write(chunk.decode("utf-8", errors="replace"))
                            sys.stdout.flush()
            except Exception as exc:  # noqa: BLE001
                reader_error.append(f"{type(exc).__name__}: {exc}")

        reader = threading.Thread(target=read_output, daemon=True)
        reader.start()
        try:
            remaining = max(0, timeout - (time.perf_counter_ns() - started) / 1_000_000_000)
            process.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            timed_out = True
            _terminate_tree(process)
        except KeyboardInterrupt:
            cancelled = True
            _terminate_tree(process)
        reader.join(timeout=10)
        if reader.is_alive():
            # A descendant retaining stdout is an incomplete end-to-end response.
            # Do not turn a successful parent exit into a completed measurement.
            reader_error.append("stdout did not reach EOF after child exit")
        elapsed = (time.perf_counter_ns() - started) / 1_000_000
        return {
            "started_at": started_at,
            "finished_at": utc_now(),
            "outer_e2e_ms": elapsed,
            "returncode": process.returncode,
            "outer_timeout": timed_out,
            "cancelled": cancelled,
            "output_error": "; ".join(reader_error) or None,
            "launch_error": None,
        }
    except OSError as exc:
        return {
            "started_at": started_at,
            "finished_at": utc_now(),
            "outer_e2e_ms": (time.perf_counter_ns() - started) / 1_000_000,
            "returncode": None,
            "outer_timeout": False,
            "cancelled": False,
            "output_error": None,
            "launch_error": f"{type(exc).__name__}: {exc}",
        }


def classify_summary(
    summary: dict[str, Any] | None,
    measurement: dict[str, Any],
    *,
    case_id: str,
    execute: bool,
    marker: str,
) -> tuple[str, str]:
    """Use recorded verdict + exit code; terminal prose is never a success signal."""
    if measurement.get("outer_timeout"):
        return "outer_timed_out", "outer deadline exceeded; includes termination time"
    if measurement.get("cancelled"):
        return "cancelled", "operator interrupted the measured child"
    if measurement.get("launch_error") or measurement.get("output_error"):
        return "harness_error", measurement.get("launch_error") or measurement["output_error"]
    if summary is None:
        if measurement["returncode"] == 2:
            return "blocked", "CLI refused before writing a task summary"
        return "missing_summary", "no uniquely identified task_summary.json"
    if not isinstance(summary, dict):
        return "invalid_summary", "task summary is not a JSON object"
    if (summary.get("case_id") or summary.get("task_id")) != case_id:
        return "invalid_summary", "case differs from requested case"
    if summary.get("execute") is not execute:
        return "invalid_summary", "execution mode differs from requested mode"
    status = summary.get("status", "")
    if not isinstance(status, str):
        return "invalid_summary", "task status is not a string"
    if status not in STATUS_EXITS or measurement["returncode"] != STATUS_EXITS[status]:
        return "invalid_summary", "task status and process exit code disagree"
    if status == "succeeded":
        verification = summary.get("verification") or {}
        if not isinstance(verification, dict):
            return "invalid_summary", "verification is not a JSON object"
        if not execute or verification.get("outcome") != "passed":
            return "invalid_summary", "success lacks real execution and passed verification"
        if case_id == "T04":
            evidence = verification.get("evidence") or {}
            if not isinstance(evidence, dict):
                return "invalid_summary", "verification evidence is not a JSON object"
            assessed = evidence.get("vision_assessment") or {}
            if not isinstance(assessed, dict):
                return "invalid_summary", "T04 assessment is not a JSON object"
            if not (
                assessed.get("marker") == marker
                and assessed.get("conversation") == "文件传输助手"
                and assessed.get("composer_empty") is True
                and assessed.get("outgoing") is True
                and assessed.get("status") == "sent"
                and assessed.get("send_state") == "sent"
            ):
                return "invalid_summary", "T04 lacks exact sent marker in 文件传输助手"
    return status, "task summary and process exit agree"


def nearest_rank(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    return sorted(values)[math.ceil(percentile * len(values)) - 1]


def configuration_check(
    args: argparse.Namespace, cohort: dict[str, Any]
) -> tuple[bool, str | None]:
    try:
        matches = (
            canonical_hash(resolve_configuration(args)[0]) == cohort["config_sha256"]
            and hashlib.sha256(Path(args.config).read_bytes()).hexdigest()
            == cohort["config_file_sha256"]
        )
        return matches, None
    except Exception as error:  # noqa: BLE001 - invalid edits must retain the in-flight record
        return False, f"{type(error).__name__}: {error}"


def configuration_unchanged(args: argparse.Namespace, cohort: dict[str, Any]) -> bool:
    return configuration_check(args, cohort)[0]


def check_provenance_after(
    record: dict[str, Any],
    args: argparse.Namespace,
    source: dict[str, Any],
    frozen_path: Path,
    frozen_sha: str,
) -> None:
    errors = []
    try:
        record["source_after"] = source_snapshot(REPO_ROOT)
        source_changed = record["source_after"] != source
    except Exception as error:  # noqa: BLE001 - preserve every request even when git/files fail
        source_changed = True
        record["source_after"] = None
        errors.append(f"source: {type(error).__name__}: {error}")
    config_matches, config_error = configuration_check(args, record["cohort"])
    record["config_changed"] = not config_matches
    if config_error:
        errors.append(f"config: {config_error}")
    try:
        frozen_changed = hashlib.sha256(frozen_path.read_bytes()).hexdigest() != frozen_sha
    except Exception as error:  # noqa: BLE001
        frozen_changed = True
        errors.append(f"frozen config: {type(error).__name__}: {error}")
    record["provenance_changed"] = source_changed or not config_matches or frozen_changed
    record["provenance_error"] = "; ".join(errors) or None


def apply_provenance_verdict(record: dict[str, Any]) -> None:
    if record.get("provenance_changed"):
        record.setdefault("status_before_provenance_check", record["status"])
        record["status"] = "provenance_changed"
        record["status_detail"] = "source or original configuration changed during the request"


def warmup_metadata(
    path: Path | None, config: dict[str, Any], state: str, raw: bytes | None = None
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "warm_state_evidence": "declared_unverified" if state != "unknown" else "unknown",
    }
    if path is None:
        return metadata
    raw = path.read_bytes() if raw is None else raw
    record = json.loads(raw.decode("utf-8"))
    if not isinstance(record, dict):
        raise TypeError("warmup record must be a JSON object")
    model = config["model"]
    matches = (
        record.get("provider") == model["provider"]
        and record.get("model_name") == model["model_name"]
        and record.get("base_url") == model.get("base_url")
    )
    metadata.update(
        {
            "warmup_record_sha256": hashlib.sha256(raw).hexdigest(),
            "warmup_ready": record.get("ready") is True,
            "warmup_model_matches": matches,
            # week4_warmup.py's captured_at is recorded after its probes finish.
            "warmup_finished_at": record.get("finished_at") or record.get("captured_at"),
        }
    )
    if state == "warm" and record.get("ready") is True and matches:
        metadata["warm_state_evidence"] = "matching_ready_probe_record"
    return metadata


def latency_stats(records: list[dict[str, Any]], field: str) -> dict[str, Any]:
    values = [
        float(record[field])
        for record in records
        if isinstance(record.get(field), (int, float))
        and not isinstance(record[field], bool)
        and math.isfinite(record[field])
        and record[field] >= 0
    ]
    return {
        "n": len(values),
        "p50_ms": nearest_rank(values, 0.50),
        "p90_ms": nearest_rank(values, 0.90),
        "p95_ms": nearest_rank(values, 0.95),
        "max_ms": max(values) if values else None,
    }


def aggregate_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Separate versions/configs/warm state, keeping every recorded request."""
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen: set[str] = set()
    for record in records:
        # Normalize earlier v1 records too, without modifying the saved input.
        record = dict(record)
        if record.get("schema") != SCHEMA:
            raise ValueError(
                "input is not an outer E2E record; runtime-only timings are incompatible"
            )
        elapsed = record.get("outer_e2e_ms")
        if (
            not isinstance(elapsed, (int, float))
            or isinstance(elapsed, bool)
            or not math.isfinite(elapsed)
            or elapsed < 0
        ):
            raise ValueError("every request must retain a finite nonnegative outer_e2e_ms")
        apply_provenance_verdict(record)
        identity = record["run_id"]
        if identity in seen:
            raise ValueError(f"duplicate run_id: {identity}")
        seen.add(identity)
        key = canonical_hash(record["cohort"])
        groups[key].append(record)
    result: list[dict[str, Any]] = []
    for key, cohort_records in sorted(groups.items()):
        for case_id in ["ALL", *sorted({r["case_id"] for r in cohort_records})]:
            subset = [r for r in cohort_records if case_id == "ALL" or r["case_id"] == case_id]
            statuses = Counter(r["status"] for r in subset)
            success = [r for r in subset if r["status"] == "succeeded"]
            result.append(
                {
                    "cohort_id": key,
                    "cohort": subset[0]["cohort"],
                    "case_id": case_id,
                    "requests": len(subset),
                    "succeeded": len(success),
                    "success_rate": len(success) / len(subset),
                    "status_counts": dict(statuses),
                    "timeout_count": sum(
                        r["status"] in ("timed_out", "outer_timed_out")
                        or bool(r.get("outer_timeout"))
                        or r.get("observed_task_status") == "timed_out"
                        or r.get("status_before_provenance_check")
                        in ("timed_out", "outer_timed_out")
                        for r in subset
                    ),
                    "all_request_end_latency": latency_stats(subset, "outer_e2e_ms"),
                    "success_completion_latency": latency_stats(success, "outer_e2e_ms"),
                    "all_request_system_latency": latency_stats(subset, "system_e2e_ms"),
                    "success_system_latency": latency_stats(success, "system_e2e_ms"),
                    "by_status": {
                        status: latency_stats(
                            [r for r in subset if r["status"] == status], "outer_e2e_ms"
                        )
                        for status in sorted(statuses)
                    },
                    "provenance_changed_count": sum(
                        bool(r.get("provenance_changed")) for r in subset
                    ),
                }
            )
    return {
        "schema": SCHEMA,
        "quantile_method": "nearest-rank ceil(p*n)",
        "latency_unit": "ms",
        "record_count": len(records),
        "groups": result,
    }


def report_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# 固定版本外层端到端测量",
        "",
        "计时：启动 CLI 前至进程退出且终端输出接收完成。包含启动、重试、确认、落盘和退出。",
        "分位数采用 nearest-rank（ceil(p × n)）；失败、阻塞、超时均保留在请求分母中。",
        "system = outer E2E − task_summary.confirmation_ms；缺失确认时长时留空，不填 0。",
        "这是扣除人工确认与执行倒计时后的派生指标；原始 E2E 仍完整保留。",
        "",
        "| Cohort | Case | N | 成功 | 成功率 | 超时 | 结束时延 P50/P90/P95/max (s) | 成功完成 P50/P90/P95/max (s) |",
        "|---|---|---:|---:|---:|---:|---|---|",
    ]

    def quantiles(stats: dict[str, Any]) -> str:
        if not stats["n"]:
            return "— (n=0)"
        return (
            "/".join(
                f"{stats[name] / 1000:.3f}" for name in ("p50_ms", "p90_ms", "p95_ms", "max_ms")
            )
            + f" (n={stats['n']})"
        )

    for group in result["groups"]:
        lines.append(
            f"| {group['cohort_id'][:8]} | {group['case_id']} | {group['requests']} | "
            f"{group['succeeded']} | {group['success_rate']:.1%} | {group['timeout_count']} | "
            f"{quantiles(group['all_request_end_latency'])} | "
            f"{quantiles(group['success_completion_latency'])} |"
        )
    lines += [
        "",
        "小样本分位数不能用于稳定性结论；n=1 的 P50/P90/P95 等于该单次耗时。",
        "不同 commit、工作树哈希、配置、模型、冷热状态、执行模式或场景分组，不混算。",
        "stdout/启动/缺失摘要/摘要不一致等异常仍计入请求分母，独立状态见 aggregate.json。",
    ]
    for group in result["groups"]:
        if group["case_id"] == "ALL":
            lines += [
                "",
                f"## Cohort {group['cohort_id'][:8]}",
                "",
                "```json",
                json.dumps(group["cohort"], ensure_ascii=False, indent=2),
                "```",
                f"测量期间来源变化：{group['provenance_changed_count']} 次。",
            ]
    return "\n".join(lines) + "\n"


def positive(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("must be finite and greater than zero")
    return parsed


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("plan", "run"):
        command = commands.add_parser(name)
        command.add_argument("--case", choices=CASE_IDS, nargs="+", default=["T04"])
        command.add_argument("--repeats", type=positive_int, default=1)
        command.add_argument("--config", default=str(REPO_ROOT / "configs" / "week4.yaml"))
        command.add_argument("--python", default=sys.executable)
        command.add_argument("--output-directory", default=str(REPO_ROOT / "outputs" / "e2e"))
        command.add_argument("--provider", choices=("mock", "openai_compatible", "langchain"))
        command.add_argument("--model")
        command.add_argument("--base-url")
        command.add_argument(
            "--execute", action="store_true", help="real CLI actions; retains confirmations"
        )
        command.add_argument("--outer-timeout", type=positive, default=1200)
        command.add_argument("--task-timeout", type=positive)
        command.add_argument("--max-actions", type=positive_int)
        command.add_argument("--planning-attempts", type=positive_int)
        command.add_argument("--ocr-engine", choices=("tesseract", "paddleocr"))
        command.add_argument("--ocr-min-confidence", type=float)
        command.add_argument("--warm-state", choices=("cold", "warm", "unknown"), default="unknown")
        command.add_argument("--warmup-record", type=Path)
        command.add_argument("--scenario", default="baseline")
        command.add_argument("--environment-note", default="")
        command.add_argument(
            "--pause-between-runs",
            action="store_true",
            help="wait for READY after each request so desktop preconditions can be restored",
        )
    command = commands.add_parser("aggregate")
    command.add_argument(
        "--input",
        nargs="+",
        type=Path,
        required=True,
        help="record file(s) or directories containing e2e_run.json",
    )
    command.add_argument("--output-directory", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args(argv)
    if args.command == "aggregate":
        paths = set()
        for path in args.input:
            paths.update(path.rglob("e2e_run.json") if path.is_dir() else [path])
        if not paths:
            raise ValueError("no e2e_run.json records found")
        records = [json.loads(path.read_text(encoding="utf-8")) for path in sorted(paths)]
        result = aggregate_records(records)
        write_json(args.output_directory / "aggregate.json", result)
        (args.output_directory / "report.md").write_text(report_markdown(result), encoding="utf-8")
        print(f"Aggregated {len(records)} requests to {args.output_directory}")
        return 0

    config, env = resolve_configuration(args)
    if args.ocr_min_confidence is not None and not 0 <= args.ocr_min_confidence <= 1:
        raise ValueError("--ocr-min-confidence must be between 0 and 1")
    if len(set(args.case)) != len(args.case):
        raise ValueError("duplicate --case values")
    if args.warm_state == "cold" and args.repeats * len(args.case) > 1:
        raise ValueError(
            "cold measurement requires one request; reset the model externally before each run"
        )
    if args.warmup_record and not args.warmup_record.is_file():
        raise ValueError("--warmup-record must name an existing record")
    warmup_bytes = args.warmup_record.read_bytes() if args.warmup_record else None
    source = source_snapshot(REPO_ROOT)
    cohort = {
        **source,
        "config_sha256": canonical_hash(config),
        "config_file_sha256": hashlib.sha256(Path(args.config).read_bytes()).hexdigest(),
        "provider": config["model"]["provider"],
        "model": config["model"]["model_name"],
        "warm_state": args.warm_state,
        "execute": args.execute,
        "scenario": args.scenario,
        "outer_timeout_seconds": args.outer_timeout,
        "environment_note": args.environment_note,
        "platform": sys.platform,
        "python": str(Path(args.python).resolve()),
        "message_conversation": "文件传输助手",
        **warmup_metadata(args.warmup_record, config, args.warm_state, raw=warmup_bytes),
    }
    batch = Path(args.output_directory).resolve() / (
        "batch_" + datetime.now(UTC).strftime("%Y%m%d_%H%M%S_%f") + "_" + uuid4().hex[:8]
    )
    frozen_config_path = batch / "frozen_config.json"
    if args.command == "plan":
        preview = []
        for repetition in range(args.repeats):
            for case_id in args.case:
                run_id, marker = new_identity(case_id)
                preview.append(
                    {
                        "repetition": repetition + 1,
                        "run_id": run_id,
                        "case_id": case_id,
                        "marker": marker,
                        "command": child_command(
                            args,
                            config,
                            case_id,
                            run_id,
                            marker,
                            batch / run_id,
                            frozen_config_path,
                        ),
                    }
                )
        print(
            json.dumps(
                {"will_launch": False, "cohort": cohort, "requests": preview},
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    batch.mkdir(parents=True, exist_ok=False)
    write_json(batch / "effective_config.json", config)
    frozen_config = json.loads(json.dumps(config))
    frozen_config["output"]["directory"] = str(batch / "cli_defaults")
    frozen_config["logging"]["directory"] = str(batch / "cli_defaults")
    write_json(frozen_config_path, frozen_config)
    frozen_config_sha256 = hashlib.sha256(frozen_config_path.read_bytes()).hexdigest()
    write_json(
        batch / "batch_metadata.json",
        {
            "schema": SCHEMA,
            "cohort": cohort,
            "planned_requests": args.repeats * len(args.case),
            "pause_between_runs": args.pause_between_runs,
            "started_at": utc_now(),
            "frozen_config_path": str(frozen_config_path),
            "frozen_config_file_sha256": frozen_config_sha256,
        },
    )
    if warmup_bytes is not None:
        (batch / "warmup.json").write_bytes(warmup_bytes)
    records = []
    for repetition in range(args.repeats):
        for case_id in args.case:
            if records and args.pause_between_runs:
                # Preparation is outside the next measured process lifetime.
                # A failed run can leave a draft or dialog; do not turn that
                # into the next independent request's initial state by accident.
                try:
                    ready = (
                        input(
                            f"Prepare {case_id}'s desktop preconditions; type READY to continue, "
                            "or STOP to end this batch: "
                        )
                        .strip()
                        .casefold()
                    )
                except EOFError:
                    print("No interactive preparation input; retaining completed records.")
                    return 2
                if ready != "ready":
                    print("Batch stopped before launching another request; records retained.")
                    return 130
            current_source = source_snapshot(REPO_ROOT)
            if (
                current_source != source
                or not configuration_unchanged(args, cohort)
                or hashlib.sha256(frozen_config_path.read_bytes()).hexdigest()
                != frozen_config_sha256
            ):
                print(
                    "Source/config changed during this batch; refusing to start another request.",
                    file=sys.stderr,
                )
                return 2
            run_id, marker = new_identity(case_id)
            directory = batch / run_id
            directory.mkdir()
            if warmup_bytes is not None:
                (directory / "cli").mkdir()
                (directory / "cli" / "warmup.json").write_bytes(warmup_bytes)
            command = child_command(
                args, config, case_id, run_id, marker, directory, frozen_config_path
            )
            print(f"\n[{repetition + 1}/{args.repeats}] {case_id} run_id={run_id}", flush=True)
            measurement = measure_process(
                command,
                cwd=REPO_ROOT,
                env=env,
                timeout=args.outer_timeout,
                output=directory / "terminal.log",
            )
            # Persist before any post-run reads. A crash or an invalid/deleted
            # config cannot erase a request that already occurred, and an unchecked
            # record cannot be mistaken for a fixed-version success on recompute.
            write_json(
                directory / "e2e_run.json",
                {
                    "schema": SCHEMA,
                    "run_id": run_id,
                    "case_id": case_id,
                    "marker": marker,
                    "cohort": cohort,
                    **measurement,
                    "status": "pending_provenance",
                    "status_detail": "post-run validation incomplete",
                    "observed_task_status": None,
                },
            )
            summaries = list((directory / "cli").rglob("task_summary.json"))
            summary = None
            summary_error = None
            if len(summaries) == 1:
                try:
                    summary = json.loads(summaries[0].read_text(encoding="utf-8"))
                except (OSError, ValueError) as exc:
                    summary_error = str(exc)
            status, detail = classify_summary(
                summary, measurement, case_id=case_id, execute=args.execute, marker=marker
            )
            summary_fields = summary if isinstance(summary, dict) else {}
            confirmation = summary_fields.get("confirmation_ms")
            if (
                not isinstance(confirmation, (int, float))
                or isinstance(confirmation, bool)
                or not math.isfinite(confirmation)
                or not 0 <= confirmation <= measurement["outer_e2e_ms"]
            ):
                confirmation = None
            record = {
                "schema": SCHEMA,
                "run_id": run_id,
                "case_id": case_id,
                "marker": marker,
                "conversation": "文件传输助手" if case_id == "T04" else None,
                "repetition": repetition + 1,
                "cohort": cohort,
                "command": command,
                **measurement,
                "status": status,
                "status_detail": detail,
                "observed_task_status": summary_fields.get("status"),
                "confirmation_ms": confirmation,
                "system_e2e_ms": measurement["outer_e2e_ms"] - confirmation
                if confirmation is not None
                else None,
                "runtime_elapsed_ms": summary_fields.get("elapsed_ms"),
                "planner_ms": summary_fields.get("planner_ms"),
                "planning_calls": summary_fields.get("planning_calls"),
                "model_requests": summary_fields.get("model_requests"),
                "screen": summary_fields.get("screen"),
                "python_version": summary_fields.get("python_version"),
                "os_version": summary_fields.get("os_version"),
                "task_summary_path": str(summaries[0]) if len(summaries) == 1 else None,
                "summary_error": summary_error,
                "frozen_config_file_sha256": frozen_config_sha256,
            }
            record["status_before_provenance_check"] = status
            record["status"] = "pending_provenance"
            write_json(directory / "e2e_run.json", record)
            check_provenance_after(record, args, source, frozen_config_path, frozen_config_sha256)
            record["status"] = status
            apply_provenance_verdict(record)
            record["warmup_age_seconds"] = None
            if cohort.get("warmup_finished_at"):
                try:
                    recorded = datetime.fromisoformat(cohort["warmup_finished_at"])
                    started = datetime.fromisoformat(measurement["started_at"])
                    record["warmup_age_seconds"] = (started - recorded).total_seconds()
                except (ValueError, TypeError):
                    pass
            write_json(directory / "e2e_run.json", record)
            records.append(record)
            result = aggregate_records(records)
            write_json(batch / "aggregate.json", result)
            (batch / "report.md").write_text(report_markdown(result), encoding="utf-8")
            print(
                f"Recorded {record['status']}: {measurement['outer_e2e_ms'] / 1000:.3f}s; {directory}",
                flush=True,
            )
            if measurement["cancelled"] or record["provenance_changed"]:
                return 130 if measurement["cancelled"] else 2
    print(f"\nComplete: {batch / 'report.md'}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, TypeError, OSError) as error:
        print(f"benchmark error: {error}", file=sys.stderr)
        raise SystemExit(2) from None
