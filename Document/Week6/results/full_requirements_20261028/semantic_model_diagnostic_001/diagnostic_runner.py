"""Private serial development probes; never creates a GUI runner or executor."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import psutil

sys.stdout.reconfigure(encoding="utf-8")


def save(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def identity():
    process = psutil.Process()
    parents = []
    for parent in process.parents():
        try:
            parents.append({"pid": parent.pid, "create_time": parent.create_time(), "exe": parent.exe()})
        except psutil.Error as exc:
            parents.append({"pid": parent.pid, "identity_error": type(exc).__name__})
    return {"pid": process.pid, "create_time": process.create_time(), "exe": process.exe(),
            "cmdline": process.cmdline(), "parents": parents}


def pins(plan):
    root = Path(plan["repository"])
    return {relative: hashlib.sha256((root / relative).read_bytes()).hexdigest()
            for relative in plan["code_pins"]}


def worker(plan, directory):
    sys.path.insert(0, str(Path(plan["repository"]) / "src"))
    from gui_agent.models.openai_compatible import OpenAICompatibleClient
    from gui_agent.planning.semantic_workflow import SemanticDecomposer, SemanticPlan
    from gui_agent.planning.workflow_decomposer import BudgetedModelClient, WorkflowRequest, _unique_keys
    from gui_agent.runtime.schemas import ObservationSnapshot

    save(directory / "worker_start.json", identity())
    if pins(plan) != plan["code_pins"]:
        raise RuntimeError("Production pins differ before diagnostic; no HTTP permitted")
    baseline = json.loads((directory / "normal006_archive/attempt_01_request.json").read_text("utf-8"))
    archived_snapshot = ObservationSnapshot.model_validate(baseline["context"]["current_observation"])
    transport = plan["transport"]
    client = OpenAICompatibleClient(model_name=transport["model"], base_url=transport["base_url"],
        api_key="ollama-local", temperature=0, max_tokens=4096, max_retries=0,
        timeout_seconds=900, environ={})
    deadline = time.monotonic() + 2700
    results, actual_requests = [], 0
    for case in plan["cases"]:
        if actual_requests >= 3:
            raise RuntimeError("Three-HTTP hard cap exhausted")
        case_dir = directory / case["id"]
        case_dir.mkdir(exist_ok=False)
        save(case_dir / "request_arguments.json", case["arguments"])
        save(case_dir / "messages_pre_registered.json", case["messages"])
        budgeted = BudgetedModelClient(client, max_requests=1, deadline=deadline,
            audit_path=case_dir / "transport_requests.jsonl")
        started = time.monotonic()
        args = case["arguments"]
        response = budgeted.generate_multimodal(args["instruction"], system=args["system"],
            context=args["context"], image_path=None, response_format=args["response_format"])
        actual_requests += budgeted.request_count
        save(case_dir / "model_response.json", {**response.as_dict(), "content": response.content,
                                               "raw_response": response.raw_response})
        record = budgeted.transport_records[-1] if budgeted.transport_records else {}
        if record.get("messages") != case["messages"]:
            raise RuntimeError("Actual pre-wire messages differ from registration")
        bodies = [item["body"] for item in record.get("http_responses", [])
                  if item.get("body_encoding") == "json"]
        if bodies and bodies[-1].get("choices"):
            provider_content = bodies[-1]["choices"][0]["message"]["content"]
            with (case_dir / "provider_native_content.txt").open("x", encoding="utf-8", newline="") as handle:
                handle.write(provider_content)
        result = {"case": case["id"], "response_ok": response.ok, "requests": budgeted.request_count,
            "requests_cumulative": actual_requests, "elapsed_seconds": time.monotonic() - started,
            "native_status": None, "schema_valid": False, "archived_snapshot_local_constraints_valid": False,
            "validation_error": "", "operational_error": response.error, "executed": False}
        if response.ok:
            try:
                native = json.loads(response.content, object_pairs_hook=_unique_keys)
                save(case_dir / "native_proposal.json", native)
                proposal = SemanticPlan.model_validate(native, strict=True)
                result.update(native_status=proposal.status, schema_valid=True, reason=proposal.reason,
                              goal_count=len(proposal.subgoals))
                caller = WorkflowRequest.model_validate(args["context"]["original_caller_request"])
                SemanticDecomposer(max_subgoals=8, max_format_retries=0)._validate(
                    proposal, caller, archived_snapshot, [], False)
                result["archived_snapshot_local_constraints_valid"] = True
            except ValueError as exc:
                result["validation_error"] = str(exc)
        save(case_dir / "result.json", result)
        results.append(result)
        temporary = directory / "worker_status.tmp"
        temporary.write_text(json.dumps({"completed": len(results), "current": case["id"],
                             "requests": actual_requests}, ensure_ascii=False), encoding="utf-8")
        temporary.replace(directory / "worker_status.json")
        print(json.dumps(result, ensure_ascii=False), flush=True)
    save(directory / "worker_summary.json", {"cases": results, "requests_total": actual_requests,
        "production_pins_after": pins(plan), "new_gui_capture_input_executor_calls": 0,
        "meaning": "Development-only model diagnostics; no plan was executed and no live task success is established."})
    return 0


def owner(plan, directory, prereg_path, prereg_sha):
    before = pins(plan)
    save(directory / "code_pins_before.json", before)
    if before != plan["code_pins"]:
        raise RuntimeError("Registered code pins changed; no child/HTTP permitted")
    save(directory / "owner_start.json", identity())
    argv = [sys.executable, "-X", "utf8", str(Path(__file__).resolve()), "--worker", str(prereg_path), prereg_sha]
    started = time.monotonic()
    with (directory / "worker_stdout.txt").open("wb") as stdout, (directory / "worker_stderr.txt").open("wb") as stderr:
        child = subprocess.Popen(argv, cwd=plan["repository"], stdin=subprocess.DEVNULL,
            stdout=stdout, stderr=stderr, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        process = psutil.Process(child.pid)
        save(directory / "child_launch.json", {"argv": argv, "pid": child.pid,
            "create_time": process.create_time(), "exe": process.exe(), "owner_pid": os.getpid()})
        print(json.dumps({"owner_pid": os.getpid(), "launched_pid": child.pid, "preregistration_sha256": prereg_sha}), flush=True)
        delivered = None
        while True:
            try:
                code = child.wait(timeout=5)
                break
            except subprocess.TimeoutExpired:
                status_path = directory / "worker_status.json"
                if status_path.is_file():
                    status = status_path.read_text("utf-8")
                    if status != delivered:
                        print(status, flush=True)
                        delivered = status
    after = pins(plan)
    save(directory / "code_pins_after.json", after)
    worker_path = directory / "worker_start.json"
    summary_path = directory / "worker_summary.json"
    summary = json.loads(summary_path.read_text("utf-8")) if summary_path.is_file() else None
    receipt = {"owner": identity(), "preregistration_sha256": prereg_sha,
        "waited_launcher_pid": child.pid, "child_wait_exit_code": code,
        "worker_actual_identity": json.loads(worker_path.read_text("utf-8")) if worker_path.is_file() else None,
        "elapsed_seconds": time.monotonic() - started, "code_pin_count": len(before),
        "code_unchanged": before == after == plan["code_pins"],
        "summary": summary, "meaning": "Exit0 means diagnostic harness completion, not GUI/semantic-task success."}
    save(directory / "owner_receipt.json", receipt)
    print(json.dumps({"child_wait_exit_code": code, "code_unchanged": receipt["code_unchanged"],
                      "elapsed_seconds": receipt["elapsed_seconds"]}, ensure_ascii=False), flush=True)
    return code or (0 if receipt["code_unchanged"] else 1)


if __name__ == "__main__":
    mode, filename, expected_sha = sys.argv[1:]
    preregistration = Path(filename).resolve()
    raw = preregistration.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha:
        raise RuntimeError("Registration hash mismatch")
    configuration = json.loads(raw)
    if hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != configuration["runner_sha256"]:
        raise RuntimeError("Diagnostic runner differs from registration")
    if len(configuration["cases"]) != 3:
        raise RuntimeError("Registration must contain exactly three bounded cases")
    raise SystemExit(worker(configuration, preregistration.parent) if mode == "--worker"
                     else owner(configuration, preregistration.parent, preregistration, expected_sha))
