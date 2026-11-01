"""CPU synthetic regression fixtures, never actual model/GPU inference."""
import importlib.util
import json
from argparse import Namespace
from pathlib import Path

import pytest
from PIL import Image

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/week5_targeted_probe.py"
SPEC = importlib.util.spec_from_file_location("targeted_probe_test", SCRIPT)
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


@pytest.mark.parametrize("variant", ["current_step", "conditional_actor"])
def test_current_step_cli_uses_the_same_registered_prompt_as_training(monkeypatch, variant):
    captured = {}

    def capture(args):
        captured.update(vars(args))
        return 0

    monkeypatch.setattr(probe, "run", capture)
    args = []
    for name in ("train-jsonl", "validation-jsonl", "selection-manifest", "training-config", "adapter", "output"):
        args.extend(["--" + name, "unused-synthetic-path"])
    assert probe.main(args + ["--prompt-variant", variant, "--partition", "train"]) == 0
    assert captured["prompt_variant"] == probe.TrainingConfig(prompt_variant=variant).prompt_variant
    assert captured["partition"] == "train"


def record(status="ok", generation=None, **extra):
    actions = [{"action_type": "click", "x": 4, "y": 5},
               {"action_type": "type_text", "text": " A\nB "}]
    return {"record_type": "request", "split": "train16", "stratum": "literal_text_2_actions",
            "sample_id": "synthetic", "status": status, "inference_attempted": True,
            "duration_ms": 2, "target": {"actions": actions, "action_annotations": [{}, {}],
            "image_width": 20, "image_height": 20},
            "generation": generation if generation is not None else json.dumps(actions), **extra}


def test_extra_action_does_not_pass_complete_text_capacity_gate():
    row = record()
    predicted = row["target"]["actions"] + [{"action_type": "key_press", "key": "enter"}]
    row["generation"] = json.dumps(predicted)
    score = probe.score_row(row, unchanged=True)
    assert score["text_exact_correct"] == 1  # original aligned metric remains intact
    assert not score["action_type_sequence_exact"] and not score["row_all_text_exact"]


def test_text_in_wrong_action_position_is_not_credited():
    row = record()
    row["generation"] = json.dumps(list(reversed(row["target"]["actions"])))
    score = probe.score_row(row, unchanged=True)
    assert score["text_exact_correct"] == 0 and not score["row_all_text_exact"]


@pytest.mark.parametrize("text", ["A\nB", " A B ", " A\nB", " A\nB\t"])
def test_text_whitespace_is_never_normalized(text):
    row = record()
    predicted = [dict(x) for x in row["target"]["actions"]]
    predicted[1]["text"] = text
    row["generation"] = json.dumps(predicted)
    assert not probe.score_row(row, unchanged=True)["row_all_text_exact"]


@pytest.mark.parametrize("status", ["error", "timeout", "cancelled", "not_started"])
def test_failed_partial_answers_keep_full_target_denominators(status):
    s = probe.summarize([record(status)], unchanged=True)["splits"][0]
    assert s["requests"] == 1
    assert s["metrics"]["type_sequence_exact"] == {"correct": 0, "total": 1, "rate": 0}
    assert s["metrics"]["text_exact"] == {"correct": 0, "total": 1, "rate": 0}


def test_nontext_rows_are_not_free_text_successes():
    row = record()
    row["target"]["actions"] = row["target"]["actions"][:1]
    row["target"]["action_annotations"] = [{}]
    row["generation"] = json.dumps(row["target"]["actions"])
    s = probe.summarize([row], unchanged=True)["splits"][0]
    assert s["metrics"]["row_all_text_exact"] == {"correct": 0, "total": 0, "rate": None}


@pytest.fixture
def cpu_run(tmp_path, monkeypatch):
    config = probe.TrainingConfig(base_model_path=str(tmp_path / "base"))
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config.to_dict()), encoding="utf8")
    manifest = {"based_on_model_outputs": False, "test_read": False, "selections": {}}
    actions_by_image = {}
    for split, offset in [("train16", 0), ("validation16", 16)]:
        rows, identities = [], []
        for index in range(16):
            image = tmp_path / f"image-{offset+index}.png"
            Image.new("RGB", (20, 20), (offset+index, 70, 80)).save(image)
            actions = [{"action_type": "click", "x": 4, "y": 5},
                       {"action_type": "type_text", "text": "literal A\nB "}]
            if index >= 8:
                actions.append({"action_type": "key_press", "key": "enter"})
            row = {"sample_id": f"{split}-{index}", "group_id": f"{split}-group-{index}",
                   "source": "synthetic CPU fixture", "source_split": "train", "image_path": str(image),
                   "image_width": 20, "image_height": 20, "instruction": "Enter literal A\nB ",
                   "actor_step": "PRE_REQUEST_STEP", "current_task": "PRIVATE_RAW_FIELD",
                   "send_prompt": "PRIVATE_SOURCE_PROMPT",
                   "actions": actions, "action_annotations": [{} for _ in actions]}
            rows.append(row)
            actions_by_image[str(image)] = actions
        path = tmp_path / f"{split}.jsonl"
        path.write_text("\n".join(json.dumps(r) for r in rows)+"\n", encoding="utf8")
        for row in probe.load_prepared_records(path):
            identities.append({"sample_id": row["sample_id"], "group_id": row["group_id"],
                "source": row["source"], "source_split": row["source_split"],
                "image_path": row["image_path"], "image_rgb_sha256": row["image_sha256"],
                "image_file_sha256": row["image_file_sha256"], "stratum": "literal_text"})
        identity = {"path": str(path), "sha256": probe.file_sha256(path)}
        manifest["selections"][split] = {"jsonl": identity, "source": identity, "samples": identities}
    selection = tmp_path / "selection.json"
    selection.write_text(json.dumps(manifest), encoding="utf8")
    args = Namespace(train_jsonl=str(tmp_path / "train16.jsonl"), validation_jsonl=str(tmp_path / "validation16.jsonl"),
        selection_manifest=str(selection), training_config=str(config_path), adapter=str(tmp_path / "adapter"),
        output=str(tmp_path / "output"), prompt_variant="targeted_sequence", partition="both",
        timeout=1, load_timeout=1, max_new_tokens=256)
    monkeypatch.setattr(probe, "artifact_manifest", lambda path: {"path": str(path), "synthetic": True})
    monkeypatch.setattr(probe, "sources", lambda: {"synthetic_source": "frozen"})
    state = {"calls": 0, "closed": 0, "prompts": []}

    def install(error=None, create_error=False, close_error=False):
        class SyntheticWorker:
            def __init__(self, config, **kwargs):
                assert "actions" not in config.training_config
                if create_error:
                    raise RuntimeError("synthetic load failure")
                self.ready = {"scope": "synthetic CPU fixture; not actual GPU inference"}

            def request(self, image, prompt, **kwargs):
                state["calls"] += 1
                state["prompts"].append(prompt)
                assert "action_annotations" not in prompt and "target_bbox" not in prompt
                assert "sample_id" not in prompt and '"x": 4' not in prompt
                if error is not None and state["calls"] == 2:
                    if isinstance(error, BaseException):
                        raise error
                    return error
                return {"status": "ok", "generation": json.dumps(actions_by_image[image]), "duration_ms": 2}

            def close(self):
                state["closed"] += 1
                if close_error:
                    raise RuntimeError("synthetic close failure")
        monkeypatch.setattr(probe, "HFWorker", SyntheticWorker)

    def output():
        root = Path(args.output)
        return json.loads((root / "summary.json").read_text(encoding="utf8")), \
            [json.loads(x) for x in (root / "raw.jsonl").read_text(encoding="utf8").splitlines()]
    return args, state, install, output


def test_conditional_actor_cli_and_warmup_use_explicit_step_without_raw_teacher_prompt(cpu_run):
    args, state, install, output = cpu_run
    args.prompt_variant = "conditional_actor"
    install()
    assert probe.run(args) == 0
    summary, raw = output()
    assert summary["complete"] and len(state["prompts"]) == 33
    assert all(prompt.endswith("Current subtask: PRE_REQUEST_STEP") for prompt in state["prompts"])
    assert all("PRIVATE" not in prompt for prompt in state["prompts"])
    assert all(row["prompt"] in state["prompts"] for row in raw if row["record_type"] == "request")


def test_constraint_opt_in_is_frozen_and_retained_in_each_raw_planned_row(cpu_run):
    args, state, install, output = cpu_run
    args.output_constraint = "json_actions"
    install()
    assert probe.run(args) == 0
    _summary, raw = output()
    meta = json.loads((Path(args.output) / "metadata.json").read_text(encoding="utf8"))
    descriptor = meta["frozen_before_generation"]["output_constraint"]
    assert descriptor["mode"] == "json_actions" and len(descriptor["schema_sha256"]) == 64
    assert meta["provenance_after"]["output_constraint"] == descriptor
    assert all(row["output_constraint"] == descriptor for row in raw if row["record_type"] == "request")
    assert state["calls"] == 33  # One warmup, one call per planned row, no repair/retry.


@pytest.mark.parametrize("partition,requests", [("both", 32), ("train", 16), ("validation", 16)])
def test_partition_sends_only_image_prompt_and_scopes_denominators(cpu_run, partition, requests):
    args, state, install, output = cpu_run
    args.partition = partition
    install()
    assert probe.run(args) == 0
    summary, raw = output()
    assert summary["planned_requests"] == requests and summary["complete"]
    assert sum(s["requests"] for s in summary["splits"]) == requests
    assert len([r for r in raw if r["record_type"] == "request"]) == requests
    assert state["calls"] == requests+1 and state["closed"] == 1
    if partition == "train":
        assert summary["splits"][1]["requests"] == 0


@pytest.mark.parametrize("error", [RuntimeError("synthetic request failure"), KeyboardInterrupt(),
                                  {"status": "timeout", "generation": "", "duration_ms": 3}])
def test_abort_retains_all32_planned_rows_without_invented_timings(cpu_run, error):
    args, _state, install, output = cpu_run
    install(error)
    assert probe.run(args) == 1
    summary, raw = output()
    rows = [r for r in raw if r["record_type"] == "request"]
    assert len(rows) == 32 and not summary["complete"]
    assert sum(r["status"] == "not_started" for r in rows) == 31
    assert all(r["duration_ms"] is None for r in rows if r["status"] == "not_started")
    assert all(s["metrics"]["whole_action_exact"]["correct"] == 0 for s in summary["splits"])


def test_load_failure_preserves_full_denominator_and_no_fake_attempts(cpu_run):
    args, _state, install, output = cpu_run
    install(create_error=True)
    assert probe.run(args) == 1
    summary, _raw = output()
    assert sum(s["requests"] for s in summary["splits"]) == 32
    assert sum(s["inference_attempted"] for s in summary["splits"]) == 0


def test_changed_provenance_invalidates_all_success_credit(cpu_run, monkeypatch):
    args, _state, install, output = cpu_run
    install()
    calls = iter([{"source": "before"}, {"source": "after"}])
    monkeypatch.setattr(probe, "sources", lambda: next(calls))
    assert probe.run(args) == 1
    summary, _raw = output()
    assert not summary["provenance_unchanged"]
    assert all(s["metrics"]["whole_action_exact"]["correct"] == 0 for s in summary["splits"])


@pytest.mark.parametrize("error", [OSError("synthetic artifact read failure"), KeyboardInterrupt()])
def test_initial_provenance_failure_retains_planned32_before_any_worker(cpu_run, monkeypatch, error):
    args, state, install, output = cpu_run
    install()

    def fail(_path):
        raise error

    monkeypatch.setattr(probe, "artifact_manifest", fail)
    assert probe.run(args) == 1
    summary, raw = output()
    assert sum(s["requests"] for s in summary["splits"]) == 32
    assert state["calls"] == 0
    assert len([r for r in raw if r["record_type"] == "request"]) == 32
    assert all(s["metrics"]["whole_action_exact"]["correct"] == 0 for s in summary["splits"])


def test_completed_token_limit_keeps_worker_but_answer_scores_zero(cpu_run):
    args, state, install, output = cpu_run
    install({"status": "error", "error_kind": "token_limit", "generated_tokens": 256,
             "generation": "[]", "duration_ms": 2})
    assert probe.run(args) == 0  # complete ledger, not a passed model gate
    summary, _raw = output()
    assert summary["splits"][0]["metrics"]["whole_action_exact"]["correct"] == 15
    assert state["calls"] == 33


@pytest.mark.parametrize("extra", [{"generated_tokens": True}, {"generated_tokens": 2}, {"error_kind": "transport"}])
def test_token_limit_reuse_requires_typed_completed_forward(extra):
    assert not probe.completed_token_limit({"status": "error", "error_kind": "token_limit", "generated_tokens": 256, **extra}, 256)
