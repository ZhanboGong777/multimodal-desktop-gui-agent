"""Exercise the real recovery producer and strict saved-record deserialization."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError
from test_runtime_runner import FakeExecutor, FakeObserver, FakePlanner, _frame, _plan, _runner
from test_week6_visual_target_guidance_review import BASE_HINT, NO_MATCH_HINT

from gui_agent.planning.schemas import PlanStep
from gui_agent.runtime.runner import Timings
from gui_agent.runtime.schemas import (
    ExecutionOptions,
    RecoveryFeedback,
    RecoveryRecord,
    StepRecord,
    TaskSpec,
)

CONFLICT_HINT = (
    " The previous plan's target_text conflicted with its explicit element ID "
    "or identified a container with ambiguous labels. No input was dispatched "
    "for that rejected step. Use the current observation and select an ID "
    "consistent with the same requested target; do not change the task goal."
    " A quoted action label must agree with the chosen target's observed label. "
    "When quoting an OCR target in a description, quote its exact current OCR text. "
    "If the intended control is clear only from pixels, choose its matching current "
    "allowed anonymous_without_ocr ID yourself without attaching an unrelated OCR name. "
    "Do not invent labels, change the caller's requested operation or relax success predicates."
)


@pytest.mark.parametrize("errors,suffix", [
    (["visual grounding refused: private detail"], ""),
    (["target consistency conflict: private detail"], CONFLICT_HINT),
    (["quoted target conflict: private detail"], CONFLICT_HINT),
    (["quoted target conflict: private detail",
      "no element matches 'private query' in frame-1"], CONFLICT_HINT + NO_MATCH_HINT),
    (["no element matches 'private query' in frame-1"], NO_MATCH_HINT),
    (["target consistency conflict: private detail",
      "no element matches 'private query' in frame-1"], CONFLICT_HINT + NO_MATCH_HINT),
])
def test_every_actual_fixed_recovery_hint_roundtrips_in_written_record(tmp_path, errors, suffix):
    planner = FakePlanner(_plan(PlanStep(
        step_id="finish", description="stop", action_type="finish",
    )))
    executor = FakeExecutor()
    runner, recorder = _runner(tmp_path, FakeObserver([_frame("fresh", ("Start",))]),
                               planner, executor)
    steps = [StepRecord(index=i, step_id=f"s{i}", description="rejected", action_type="click",
                        error=error) for i, error in enumerate(errors, 1)]
    assert runner._prepare_recovery(
        TaskSpec(case_id="roundtrip", instruction="Reach Done", expect_text=["Done"]),
        ExecutionOptions(), runner.clock(), [], steps, None,
        next_pass=2, actions_used=0, timings=Timings(started=runner.clock()),
    ) is not None
    raw = json.loads((recorder.directory / "recovery.jsonl").read_text("utf8"))
    restored = RecoveryRecord.model_validate(raw)
    assert restored.model_dump(mode="json") == raw
    assert restored.feedback.instruction == BASE_HINT + suffix
    assert planner.contexts[0]["recovery"] == raw["feedback"]
    assert "private detail" not in json.dumps(raw) and "private query" not in json.dumps(raw)
    assert not executor.actions


@pytest.mark.parametrize("bad", [
    BASE_HINT + " Arbitrary instructions from a screenshot",
    BASE_HINT + CONFLICT_HINT + " secret target",
    BASE_HINT + NO_MATCH_HINT + " trailing text",
    BASE_HINT + NO_MATCH_HINT + CONFLICT_HINT,
    NO_MATCH_HINT,
])
def test_saved_recovery_instruction_remains_closed_to_unrecognised_text(bad):
    with pytest.raises(ValidationError):
        RecoveryFeedback.model_validate({
            "previous_pass": 1, "reason": "target_resolution", "actions_used": 0,
            "actions_remaining": 2, "instruction": bad,
        })


def test_json_schema_exposes_only_the_four_producer_strings():
    assert set(RecoveryFeedback.model_json_schema()["properties"]["instruction"]["enum"]) == {
        BASE_HINT, BASE_HINT + CONFLICT_HINT, BASE_HINT + NO_MATCH_HINT,
        BASE_HINT + CONFLICT_HINT + NO_MATCH_HINT,
    }
