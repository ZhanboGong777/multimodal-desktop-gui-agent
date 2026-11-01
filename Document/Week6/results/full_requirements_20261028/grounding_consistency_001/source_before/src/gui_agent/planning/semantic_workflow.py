"""Model-generated semantic objectives, fixed before input and verified in order.

The model proposes a complete objective sequence, not a primitive action list.
Its literal postconditions remain untrusted proposals: the ordinary verifier
tests them on real observations, and unchanged caller checks decide completion.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
from collections.abc import Callable
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import Field, field_validator, model_validator

from ..models.base import ModelClient
from ..models.openai_compatible import OpenAICompatibleClient
from ..runtime.monitoring import RunMonitor
from ..runtime.runner import TaskRunner
from ..runtime.schemas import ExecutionOptions, ObservationSnapshot, TaskSpec, VerificationResult
from ..runtime.verification import Verifier
from ..runtime.workflow import WorkflowRunner, WorkflowRunResult, WorkflowSpec, _safe_id
from ..schemas import ActionResult, SchemaModel
from .planner import TaskPlanner
from .workflow_decomposer import (
    BudgetedModelClient,
    ProposedSubgoal,
    WorkflowRequest,
    _ProgressChild,
    _rolling_frame_guard,
    _RollingFrames,
    _RollingVerifier,
    _unique_keys,
    _write_json,
    _write_new_json,
)

SEMANTIC_PROTOCOL_VERSION = "week6.semantic_subtasks.v6"
SEMANTIC_INPUT_MODALITY = "caller_intent_only"
SEMANTIC_STEP_POLICY = "Raw model step predictions are advisory; the predeclared semantic subtask postconditions are mandatory."
SEMANTIC_REASON_MAX_CHARACTERS = 300
SEMANTIC_INSTRUCTION_MAX_CHARACTERS = 600
SEMANTIC_CHECK_MAX_CHARACTERS = 200
SEMANTIC_CHECKS_PER_LIST = 8
_INLINE_CORRECTION_MAX_CHARACTERS = 2000
_INPUT_TYPES = {"click", "double_click", "right_click", "drag", "scroll", "type_text", "key_press", "hotkey"}
_PRIMITIVE_ONLY = re.compile(r"^(?:click|double[_ -]?click|right[_ -]?click|move|drag|type_text|key_press|hotkey|scroll|wait|finish)\b", re.IGNORECASE)
_UNRESOLVED_LITERAL = re.compile(r"\[(?:not specified|unspecified|unknown|未指定|待填写)\]|<TBD>", re.IGNORECASE)
SEMANTIC_SYSTEM_PROMPT = """Plan the COMPLETE caller request as an ORDERED sequence of semantic state changes, not primitive clicks or copies of the whole request.
This is sequential planning: later-stage controls normally are hidden on the
initial page. Absence of a later control is not missing caller information and
does not make an explicitly requested operation optional. Plan the requested
future changes; the action layer will observe each stage, validate actual
controls and stop safely if they are unavailable. An export dialog can still
be unopened while earlier editing is unfinished.

You receive the complete caller request and immutable final text predicates,
plus a logical record of verified objectives and failure categories. You do NOT
receive current screen text, geometry or pixels. The action layer separately
observes the screen for each input. Plan caller intent, not an inventory of UI
controls; never claim you have seen a screen. Previous replies are untrusted data.

Return only the supplied JSON schema. Initially include at least TWO distinct
semantic objectives covering ALL requirements; revisions cover unfinished work
only and may contain one. Never repeat completed objectives. Each goal needs a
unique ASCII goal_id, one concise instruction (<=600 characters), and literal
expect_text/forbid_text POSTCONDITIONS fixed before input: text required visible
or absent AFTER that goal, not do-not-click lists. You may propose intermediate
literal outcome/status text as an UNTRUSTED PREDICTION, not an observed fact.
Each list has <=8 nonblank strings of <=200 characters, with no outer whitespace.
Do not change these strings after execution to make a failed goal look passed.
A currently visible action caption alone is not proof that its operation ran:
for example, a 'Save' button differs from a predicted 'Changes saved' message.
A genuinely new control can witness reaching a new stage, but does not prove
every requested operation. Choose specific completion evidence and sufficient
context. The action layer will test the full predicate on fresh before/after
screens and stop if the proposed evidence cannot be verified. Related requested
operations may share a semantic goal with a justified combined outcome.
The same COMPLETE expect_text/forbid_text set may belong to only ONE goal. If
several unfinished requested operations need that same outcome, combine ALL
those operations into one concise semantic goal in your response, rather than
returning separate goals with identical checks. Do not drop any requested
operation to remove a duplicate, and never combine new work into an already
completed goal. A predicate that becomes true only after the overall request is
finished cannot verify an earlier goal while requested operations remain. Put
the overall final-state conjunction on the last goal, whose instruction covers
all operations still needed to produce that state. Other goals need their own
distinguishable outcomes. Predictions can be wrong; never claim they are observed.
Do not invent a goal sequence just to reach the minimum of two. If the request
cannot support two meaningful, distinguishable objectives with verifiable checks,
return needs_clarification. Multiple supplied strings alone do not prove that
the objectives are executable. Each goal's whole predicate must be false on a
fresh screen before input. Caller final checks remain unchanged and independent.
Never forbid any text required by the unchanged caller final check.

Use ready with reason="" or one short sentence (<=300 characters), never repeated
clauses or analysis. Preserve needs_clarification for genuinely missing or
contradictory caller parameters or an unjustifiable verifiable contract; name
the specific issue once, with no goals. Hidden later controls alone are not such
an issue. Use refused for unsupported unsafe requests; do not force ready.
Do not return actions, coordinates, frame/element IDs, final checks, budgets or
extra fields. Never send messages, email, or public posts.
"""


def _no_repeated_clauses(value: str) -> str:
    clauses = [" ".join(part.casefold().split())
               for part in re.split(r"[.!?。！？;；\n]+", value)]
    substantial = [part for part in clauses if len(part) >= 8]
    if len(substantial) != len(set(substantial)):
        raise ValueError("use a concise sentence without repeated clauses")
    return value


def _caller_contract_issue(request: WorkflowRequest) -> str:
    """Reserved-marker/final-consistency checks, not general ambiguity detection."""
    for value in [request.instruction, *request.expect_text, *request.forbid_text]:
        marker = _UNRESOLVED_LITERAL.search(value)
        if marker:
            return (f"reserved unresolved literal {marker.group()!r} must be resolved before semantic planning; "
                    "using these reserved forms as literal content is unsupported")
    if any(forbidden.casefold() in expected.casefold()
           for forbidden in request.forbid_text for expected in request.expect_text):
        return "caller positive and negative predicates contradict the literal verifier"
    return ""


SemanticCheckText = Annotated[str, Field(strict=True, min_length=1, max_length=SEMANTIC_CHECK_MAX_CHARACTERS)]


class SemanticGoal(ProposedSubgoal):
    goal_id: str = Field(strict=True)
    instruction: str = Field(strict=True, min_length=1, max_length=SEMANTIC_INSTRUCTION_MAX_CHARACTERS)
    expect_text: list[SemanticCheckText] = Field(default_factory=list, max_length=SEMANTIC_CHECKS_PER_LIST)
    forbid_text: list[SemanticCheckText] = Field(default_factory=list, max_length=SEMANTIC_CHECKS_PER_LIST)

    @field_validator("instruction")
    @classmethod
    def _concise_instruction(cls, value: str) -> str:
        return _no_repeated_clauses(value)

    @field_validator("goal_id")
    @classmethod
    def _identity(cls, value: str) -> str:
        return _safe_id(value)

    @model_validator(mode="after")
    def _semantic_objective(self) -> SemanticGoal:
        if _PRIMITIVE_ONLY.search(self.instruction):
            raise ValueError("describe the semantic state change, not a primitive action instruction")
        return self


class SemanticPlan(SchemaModel):
    status: Literal["ready", "needs_clarification", "refused"]
    reason: str = Field(strict=True, max_length=SEMANTIC_REASON_MAX_CHARACTERS)
    subgoals: list[SemanticGoal] = Field(max_length=32)

    @field_validator("reason")
    @classmethod
    def _concise_reason(cls, value: str) -> str:
        return _no_repeated_clauses(value)

    @model_validator(mode="after")
    def _shape(self) -> SemanticPlan:
        if self.status == "ready" and not self.subgoals:
            raise ValueError("ready semantic plan needs executable objectives")
        if self.status != "ready" and (self.subgoals or not self.reason.strip()):
            raise ValueError("non-ready plan needs a reason and no executable objectives")
        return self


def _semantic_response_format(_request: WorkflowRequest, max_subgoals: int) -> dict[str, Any]:
    schema = SemanticPlan.model_json_schema()
    schema["properties"]["subgoals"]["maxItems"] = max_subgoals
    goal = schema["$defs"]["SemanticGoal"]
    goal["required"] = ["goal_id", "instruction", "expect_text", "forbid_text"]
    return {"type": "json_schema", "json_schema": {
        "name": "semantic_subtask_sequence", "strict": True, "schema": schema}}


def _validate_native_predicates(native: Any, _request: WorkflowRequest) -> None:
    """Check raw strings before inherited validators can strip whitespace."""
    if not isinstance(native, dict) or not isinstance(native.get("subgoals"), list):
        raise ValueError("native semantic response must contain a subgoals list")  # noqa: TRY004 - bounded schema repair
    for goal in native["subgoals"]:
        if not isinstance(goal, dict):
            raise ValueError("native semantic objective must be an object")  # noqa: TRY004 - bounded schema repair
        instruction = goal.get("instruction")
        if (type(instruction) is not str or not instruction or instruction != instruction.strip()
                or len(instruction) > SEMANTIC_INSTRUCTION_MAX_CHARACTERS):
            raise ValueError("native instruction must be a bounded nonblank string preserved without stripping")
        for field in ("expect_text", "forbid_text"):
            values = goal.get(field)
            if (not isinstance(values, list) or len(values) > SEMANTIC_CHECKS_PER_LIST
                    or any(type(value) is not str or not value or value != value.strip()
                           or len(value) > SEMANTIC_CHECK_MAX_CHARACTERS for value in values)):
                raise ValueError(f"native {field} must contain bounded nonblank strict strings preserved without stripping")


def _logical_completed(completed: list[dict[str, Any]]) -> list[dict[str, Any]]:
    # Runtime verification/evidence objects, paths and screen content are never
    # sent back to the high-level intent planner through the completion ledger.
    return [{key: entry[key] for key in ("goal_id", "instruction", "expect_text", "forbid_text")}
            for entry in completed]


def _logical_failure(failure: dict[str, Any] | None) -> dict[str, str] | None:
    if failure is None:
        return None
    status = failure.get("status")
    category = ("semantic_postconditions_unsatisfied" if status == "failed" else
                "caller_final_unsatisfied" if "reason" in failure else "previous_objective_not_completed")
    return {"category": category}


class SemanticPlanResult(SchemaModel):
    schema_version: Literal["week6.semantic_subtasks.v6"] = SEMANTIC_PROTOCOL_VERSION
    input_modality: Literal["caller_intent_only"] = SEMANTIC_INPUT_MODALITY
    image_attached: Literal[False] = False
    status: str = "failed"
    proposal: SemanticPlan | None = None
    workflow: WorkflowSpec | None = None
    planning_calls: int = 0
    model_requests: int = 0
    elapsed_ms: float = 0
    evidence_directory: str
    error: str = ""


def _signature(goal: SemanticGoal) -> tuple[Any, ...]:
    return (goal.instruction.casefold(), tuple(sorted(x.casefold() for x in goal.expect_text)),
            tuple(sorted(x.casefold() for x in goal.forbid_text)))


def _goal_task(goal: SemanticGoal, request: WorkflowRequest) -> TaskSpec:
    return TaskSpec(
        case_id=goal.goal_id, risk="medium", gate_on_step_verification=False,
        instruction=("Complete ONLY the current semantic subtask, then stop for verification. "
                     "Do not execute later objectives or repeat completed selections.\n"
                     f"Current semantic subtask: {goal.instruction}\n"
                     f"Complete original caller request: {request.instruction}"),
        expect_text=list(goal.expect_text), forbid_text=list(goal.forbid_text),
    )


def _final_task(request: WorkflowRequest) -> TaskSpec:
    return TaskSpec(case_id="FINAL", instruction="Independently read unchanged caller final predicates",
                    expect_text=list(request.expect_text), forbid_text=list(request.forbid_text))


def _initial_guard(frame: ObservationSnapshot) -> str:
    if (frame.errors or not frame.image_path or not frame.window_id or not frame.foreground_stable
            or frame.window_bounds is None or not any(x.text.strip() for x in frame.elements)):
        return "semantic execution requires readable saved pixels and stable explicit window identity/geometry"
    return ""


class SemanticDecomposer:
    """Strict full-sequence proposal parsing; no caller-written subtask list."""

    def __init__(self, *, max_subgoals: int = 8, max_format_retries: int = 2,
                 clock: Callable[[], float] = time.monotonic) -> None:
        if type(max_subgoals) is not int or not 2 <= max_subgoals <= 32:
            raise ValueError("semantic max_subgoals must be an integer in 2..32")
        if type(max_format_retries) is not int or not 0 <= max_format_retries <= 2:
            raise ValueError("semantic format retries must be an integer in 0..2")
        self.max_subgoals, self.max_format_retries, self.clock = max_subgoals, max_format_retries, clock

    def decompose(self, request: WorkflowRequest, *, evidence_directory: Path,
                  budgeted_client: BudgetedModelClient, initial_observation: ObservationSnapshot,
                  completed_subgoals: list[dict[str, Any]], max_calls: int, deadline: float,
                  revision: bool = False, failure: dict[str, Any] | None = None) -> SemanticPlanResult:
        directory = Path(evidence_directory)
        directory.mkdir(parents=True, exist_ok=False)
        started, before_requests = self.clock(), budgeted_client.request_count
        result = SemanticPlanResult(evidence_directory=str(directory.resolve()))
        try:
            issue = _caller_contract_issue(request)
            if issue:
                result.status, result.error = "needs_clarification", issue
                return result
            if _initial_guard(initial_observation):
                raise ValueError(_initial_guard(initial_observation))
            response_format = _semantic_response_format(request, self.max_subgoals)
            schema_text = json.dumps(response_format["json_schema"]["schema"], ensure_ascii=False, sort_keys=True)
            system = (SEMANTIC_SYSTEM_PROMPT + "\nExact response JSON schema (also supplied as response_format):\n"
                      + schema_text)
            context: dict[str, Any] = {
                "original_caller_request": request.model_dump(mode="json"),
                "minimum_objectives": 1 if revision else 2, "maximum_objectives": self.max_subgoals,
                "revision": revision, "completed_subgoals": _logical_completed(completed_subgoals),
                "failure": _logical_failure(failure),
                "input_modality": SEMANTIC_INPUT_MODALITY, "image_attached": False,
            }
            _write_new_json(directory / "snapshot_archive.json", initial_observation.model_dump(mode="json"))
            pixels = Path(initial_observation.image_path)
            pixel_archive = None
            if pixels.is_file():
                pixel_archive = directory / ("snapshot_pixels" + pixels.suffix)
                with pixel_archive.open("xb") as handle:
                    handle.write(pixels.read_bytes())
            _write_new_json(directory / "context_source_archive.json", {
                "completed_subgoals": completed_subgoals, "failure": failure,
                "snapshot_archive": "snapshot_archive.json",
                "snapshot_pixels_archive": pixel_archive.name if pixel_archive else None,
                "snapshot_pixels_sha256": hashlib.sha256(pixel_archive.read_bytes()).hexdigest() if pixel_archive else None,
                "snapshot_pixels_source_status": "copied_byte_exact" if pixel_archive else "source_not_available",
                "sent_to_high_level_model": False,
            })
            provider = budgeted_client.client
            if (not isinstance(provider, OpenAICompatibleClient)
                    and getattr(provider, "supports_strict_json_schema", False) is not True):
                raise ValueError("semantic_subtasks requires strict JSON schema transport; no fallback")
            _write_new_json(directory / "protocol.json", {
                "schema_version": SEMANTIC_PROTOCOL_VERSION, "response_format": response_format,
                "caller_request": request.model_dump(mode="json"), "revision": revision,
                "input_modality": SEMANTIC_INPUT_MODALITY, "image_attached": False,
                "snapshot_image_path_provenance": initial_observation.image_path,
                "snapshot_sent_to_model": False,
                "postcondition_source": "untrusted_model_prediction", "final_source": "unchanged_caller",
                "output_constraints": {
                    "schema_version": "week6.semantic_output_constraints.v1",
                    "reason_max_characters": SEMANTIC_REASON_MAX_CHARACTERS,
                    "instruction_max_characters": SEMANTIC_INSTRUCTION_MAX_CHARACTERS,
                    "check_max_characters": SEMANTIC_CHECK_MAX_CHARACTERS,
                    "checks_per_list_max": SEMANTIC_CHECKS_PER_LIST,
                    "inline_correction_max_characters": _INLINE_CORRECTION_MAX_CHARACTERS,
                    "oversized_previous_response": "archived_in_full_not_repeated_in_correction_context",
                },
            })
            for attempt in range(min(max_calls, self.max_format_retries + 1)):
                if self.clock() >= deadline:
                    result.status, result.error = "timed_out", "deadline exhausted before semantic decomposition"
                    break
                result.planning_calls += 1
                _write_new_json(directory / f"attempt_{attempt + 1:02d}_request.json", {
                    "instruction": request.instruction, "system": system,
                    "context": context, "image_path": None,
                    "input_modality": SEMANTIC_INPUT_MODALITY, "image_attached": False,
                    "snapshot_image_path_provenance": initial_observation.image_path,
                    "response_format": response_format,
                })
                response = budgeted_client.generate_multimodal(
                    request.instruction, system=system, context=context,
                    image_path=None, response_format=response_format,
                )
                _write_new_json(directory / f"attempt_{attempt + 1:02d}_response.json", {
                    **response.as_dict(), "content": response.content, "raw_response": response.raw_response})
                if self.clock() >= deadline:
                    result.status, result.error = "timed_out", "semantic response returned after shared deadline"
                    break
                if not response.ok:
                    raise ValueError(response.error or "empty semantic response")
                try:
                    native = json.loads(response.content, object_pairs_hook=_unique_keys)
                    _validate_native_predicates(native, request)
                    proposal = SemanticPlan.model_validate(native, strict=True)
                    self._validate(proposal, request, initial_observation, completed_subgoals, revision)
                except ValueError as exc:
                    result.error = str(exc)
                    _write_new_json(directory / f"attempt_{attempt + 1:02d}_error.json", {"error": str(exc)})
                    context = {**context, "constraint_error": str(exc),
                               "previous_response_record": str((directory / f"attempt_{attempt + 1:02d}_response.json").resolve()),
                               "previous_response_characters": len(response.content)}
                    context.pop("previous_response", None)
                    if len(response.content) <= _INLINE_CORRECTION_MAX_CHARACTERS:
                        context["previous_response"] = response.content
                    continue
                result.status, result.error, result.proposal = proposal.status, proposal.reason, proposal
                _write_new_json(directory / "native_proposal.json", native)
                if proposal.status == "ready":
                    result.workflow = WorkflowSpec(workflow_id=request.workflow_id, instruction=request.instruction,
                        subgoals=[_goal_task(goal, request) for goal in proposal.subgoals], final_check=_final_task(request))
                    _write_new_json(directory / "compiled_workflow.json", {
                        "schema_version": SEMANTIC_PROTOCOL_VERSION, "native_proposal": native,
                        "workflow": result.workflow.model_dump(mode="json"),
                        "mapping": [{"goal_id": goal.goal_id,
                                     "native_instruction": native["subgoals"][number]["instruction"],
                                     "validated_instruction": goal.instruction,
                                     "compiled_instruction": task.instruction,
                                     "expect_text": list(goal.expect_text), "forbid_text": list(goal.forbid_text)}
                                    for number, (goal, task) in enumerate(zip(proposal.subgoals, result.workflow.subgoals, strict=True))],
                    })
                break
        except (KeyboardInterrupt, Exception) as exc:  # noqa: BLE001 - retain native evidence and real failures
            result.status = "cancelled" if isinstance(exc, KeyboardInterrupt) else "failed"
            result.error = f"{type(exc).__name__}: {exc}"
        finally:
            result.model_requests = budgeted_client.request_count - before_requests
            result.elapsed_ms = max(0, (self.clock() - started) * 1000)
            try:
                _write_new_json(directory / "result.json", result.model_dump(mode="json"))
            except (KeyboardInterrupt, Exception) as exc:  # noqa: BLE001 - preserve already spent planning and HTTP calls
                result.status = "cancelled" if isinstance(exc, KeyboardInterrupt) else "failed"
                result.error += f"; semantic decomposition result persistence failed: {type(exc).__name__}"
        return result

    def _validate(self, proposal: SemanticPlan, request: WorkflowRequest, before: ObservationSnapshot,
                  completed: list[dict[str, Any]], revision: bool) -> None:
        if proposal.status != "ready":
            return
        if not (1 if revision else 2) <= len(proposal.subgoals) <= self.max_subgoals:
            raise ValueError("initial semantic sequence needs 2..maximum objectives; a revision needs 1..maximum")
        ids, instructions, signatures = set(), set(), set()
        check_owners: dict[tuple[Any, ...], str] = {}
        done_ids = {x["goal_id"].casefold() for x in completed}
        done_instructions = {x["instruction"].casefold() for x in completed}
        done_checks = {(tuple(sorted(s.casefold() for s in x["expect_text"])),
                        tuple(sorted(s.casefold() for s in x["forbid_text"]))) for x in completed}
        for goal in proposal.subgoals:
            key, instruction = goal.goal_id.casefold(), goal.instruction.casefold()
            signature = _signature(goal)
            if instruction == request.instruction.casefold():
                raise ValueError("a semantic subtask cannot be a copy of the overall caller request")
            if key in done_ids or instruction in done_instructions or signature[1:] in done_checks:
                raise ValueError("completed semantic objective/checks cannot be replayed; keep completed objectives unchanged")
            if key in ids or instruction in instructions:
                raise ValueError("duplicate semantic goal_id/instruction cannot be replayed; use a distinct unfinished objective")
            if signature[1:] in signatures:
                prior = check_owners[signature[1:]]
                raise ValueError(
                    f"duplicate semantic postconditions for goals {prior!r} and {goal.goal_id!r}: "
                    "one complete expect_text/forbid_text set can verify only one goal. "
                    "In the model response, combine their unfinished required operations into one semantic goal; "
                    "preserve every requested operation, do not merge into completed objectives, and do not use "
                    "an overall-final outcome for an earlier goal with requested operations still remaining."
                )
            if any(forbidden.casefold() in final.casefold() for forbidden in goal.forbid_text for final in request.expect_text):
                raise ValueError("semantic forbidden postcondition conflicts with immutable caller final text")
            ids.add(key)
            instructions.add(instruction)
            signatures.add(signature[1:])
            check_owners[signature[1:]] = goal.goal_id
        if Verifier().check_task(_goal_task(proposal.subgoals[0], request), before).passed:
            raise ValueError("first semantic goal postconditions already hold; existing context cannot prove execution")


class SemanticWorkflowResult(SchemaModel):
    schema_version: Literal["week6.semantic_workflow.v1"] = "week6.semantic_workflow.v1"
    decomposition_protocol_mode: Literal["semantic_subtasks"] = "semantic_subtasks"
    workflow_id: str
    status: str = "failed"
    plans: list[SemanticPlanResult] = Field(default_factory=list)
    goal_outcomes: list[dict[str, Any]] = Field(default_factory=list)
    completed_subgoals: list[dict[str, Any]] = Field(default_factory=list)
    semantic_goals_total: int = 0
    workflow_result: WorkflowRunResult | None = None
    workflow_results: list[WorkflowRunResult] = Field(default_factory=list)
    final_verification: VerificationResult | None = None
    final_observation_reference: str = ""
    total_actions: int = 0
    planning_calls: int = 0
    model_requests: int = 0
    unconfirmed_action_attempts: int = 0
    accounting_complete: bool = True
    elapsed_ms: float = 0
    evidence_directory: str
    error: str = ""
    notes: list[str] = Field(default_factory=list)


class _SemanticVerifier(_RollingVerifier):
    def check_task(self, task: TaskSpec, observation: ObservationSnapshot | None) -> VerificationResult:
        result = super().check_task(task, observation)
        result.evidence["verification_source"] = "frozen_model_semantic_subtask_postconditions"
        return result


class _SemanticChild:
    def __init__(self, child: Any, task: TaskSpec, before: ObservationSnapshot) -> None:
        if (not isinstance(child, TaskRunner) or not isinstance(getattr(child, "planner", None), TaskPlanner)
                or type(child.planner.max_steps) is not int or child.planner.max_steps < 1):
            raise ValueError("semantic execution requires a dedicated real TaskRunner and TaskPlanner")
        self.child, self.task = child, task.model_copy(deep=True)
        self.observer = _RollingFrames(child.observer, before)
        self.verifier = _SemanticVerifier(child.verifier, self.observer)
        self.recorder, self.planning_limit_reference = child.recorder, ""
        self.planning_protocol_version = SEMANTIC_PROTOCOL_VERSION
        self.step_verification_policy = SEMANTIC_STEP_POLICY
        self.known_outcome = None
        child.observer, child.verifier = self.observer, self.verifier

    def run(self, task: TaskSpec, options: ExecutionOptions, **callbacks: Any) -> Any:
        if task.model_dump() != self.task.model_dump():
            raise ValueError("frozen semantic task changed before execution")
        outcome = _ProgressChild._run_single_step(self, task, options, **callbacks)
        self.known_outcome = outcome
        try:
            limit = json.loads(Path(self.planning_limit_reference).read_text(encoding="utf8"))
            post_id = outcome.verification.evidence.get("observation_id") if outcome.verification else None
            post = self.observer.frames.get(post_id)
            checked = Verifier().check_task(self.task, post)
            inputs, valid = 0, True
            for step in outcome.steps:
                if step.action_result is None:
                    continue
                try:
                    receipt = ActionResult.model_validate(step.action_result)
                    valid &= receipt.action is not None and receipt.action.action_type == step.action_type
                    if receipt.success and not receipt.dry_run and step.action_type in _INPUT_TYPES:
                        inputs += 1
                except ValueError:
                    valid = False
            plans_complete = (limit["protocol_version"] == SEMANTIC_PROTOCOL_VERSION and limit["restored"]
                              and len(limit["calls"]) == outcome.planning_calls
                              and all(x["max_steps"] == 1 and x["returned_steps"] in {None, 0, 1} for x in limit["calls"]))
            eligible = bool(outcome.execute and outcome.status == "succeeded" and outcome.verification
                            and outcome.verification.passed and checked.passed and inputs > 0 and valid
                            and not self.observer.failure and post is not None
                            and plans_complete and outcome.unconfirmed_action_attempts == 0)
            _write_new_json(self.recorder.directory / "semantic_goal_audit.json", {
                "schema_version": SEMANTIC_PROTOCOL_VERSION, "frozen_task": self.task.model_dump(mode="json"),
                "child_status": outcome.status, "child_verification": outcome.verification.model_dump(mode="json") if outcome.verification else None,
                "semantic_postcondition_verification": checked.model_dump(mode="json"),
                "post_observation_id": post_id, "confirmed_input_count": inputs,
                "planning_accounting_complete": plans_complete, "eligible_for_semantic_credit": eligible,
                "planning_limit_reference": self.planning_limit_reference,
                "steps": [x.model_dump(mode="json") for x in outcome.steps],
                "meaning": "The original model-proposed literal semantic postconditions were independently checked; no raw failed step was rewritten.",
            })
            if outcome.execute and outcome.status == "succeeded" and not eligible:
                return outcome.model_copy(update={"status": "blocked", "verification": VerificationResult(outcome="inconclusive"),
                                                  "error": "semantic objective evidence is unqualified"})
            return outcome
        except (KeyboardInterrupt, Exception) as exc:  # noqa: BLE001 - preserve already known receipts and counts
            return outcome.model_copy(update={"status": "cancelled" if isinstance(exc, KeyboardInterrupt) else "failed",
                "verification": VerificationResult(outcome="inconclusive"), "error": f"semantic audit failed: {type(exc).__name__}"})


class SemanticWorkflowRunner:
    def __init__(self, directory: str | Path, client: ModelClient, runner_factory: Callable, *,
                 initial_observer: Callable[[Path], ObservationSnapshot] | None,
                 max_model_requests: int = 24, max_subgoals: int = 8, max_format_retries: int = 2,
                 max_decomposition_rounds: int = 4, max_goal_planning_calls: int = 3,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.decomposer = SemanticDecomposer(max_subgoals=max_subgoals, max_format_retries=max_format_retries, clock=clock)
        if type(max_decomposition_rounds) is not int or not 1 <= max_decomposition_rounds <= 8:
            raise ValueError("semantic revision rounds must be in 1..8")
        if type(max_goal_planning_calls) is not int or max_goal_planning_calls <= 0:
            raise ValueError("semantic per-goal planning cap must be positive")
        self.directory, self.client, self.runner_factory = Path(directory), client, runner_factory
        self.initial_observer, self.clock = initial_observer, clock
        self.max_model_requests, self.max_rounds = max_model_requests, max_decomposition_rounds
        self.max_goal_planning_calls = max_goal_planning_calls

    def run(self, request: WorkflowRequest, options: ExecutionOptions, **callbacks: Any) -> SemanticWorkflowResult:
        request = WorkflowRequest.model_validate(request.model_dump())
        options = ExecutionOptions.model_validate(options.model_dump())
        if not math.isfinite(options.task_timeout_seconds) or options.max_planning_attempts < 2:
            raise ValueError("semantic execution needs a finite deadline and decomposition plus action planning budget")
        started = self.clock()
        if not math.isfinite(started):
            raise ValueError("semantic monotonic clock must be finite")
        deadline = started + options.task_timeout_seconds
        self.directory.mkdir(parents=True, exist_ok=False)
        monitor = RunMonitor(self.directory, clock=self.clock)
        monitor.start(case_id=request.workflow_id, execute=options.execute, max_actions=options.max_actions,
                      max_planning_attempts=options.max_planning_attempts, dynamic_subgoals=True)
        result = SemanticWorkflowResult(workflow_id=request.workflow_id, evidence_directory=str(self.directory.resolve()))
        client, wrapper, inflight = None, None, False
        on_decomposition = callbacks.pop("on_decomposition", None)
        on_checkpoint = callbacks.pop("on_checkpoint", None)
        seen_plans: set[str] = set()
        failed_objectives: set[tuple[Any, ...]] = set()
        latest: ObservationSnapshot | None = None

        def observe(name: str) -> ObservationSnapshot:
            nonlocal latest
            if self.clock() >= deadline:
                raise TimeoutError("shared semantic deadline exhausted before observation")
            path = self.directory / name
            path.mkdir(exist_ok=False)
            frame = self.initial_observer(path)
            frame = ObservationSnapshot.model_validate(frame.model_dump())
            _write_new_json(path / "observation.json", frame.model_dump(mode="json"))
            error = _initial_guard(frame) or (_rolling_frame_guard(latest, frame) if latest is not None else "")
            if error:
                raise ValueError(error)
            if self.clock() >= deadline:
                raise TimeoutError("shared semantic deadline exhausted during observation")
            latest = frame
            return frame

        try:
            _write_new_json(self.directory / "semantic_request.json", {
                "schema_version": SEMANTIC_PROTOCOL_VERSION, "caller_request": request.model_dump(mode="json"),
                "options": options.model_dump(mode="json"), "absolute_deadline": deadline,
                "max_model_requests": self.max_model_requests, "max_goal_planning_calls": self.max_goal_planning_calls,
            })
            issue = _caller_contract_issue(request)
            _write_new_json(self.directory / "semantic_preflight.json", {
                "schema_version": SEMANTIC_PROTOCOL_VERSION, "supported": not bool(issue), "issue": issue,
                "scope": "declared unresolved literal forms and contradictory caller final predicates only",
            })
            if issue:
                result.status, result.error = "needs_clarification", issue
                return result
            if self.initial_observer is None:
                raise ValueError("semantic_subtasks requires an explicit initial observation provider")
            client = BudgetedModelClient(self.client, max_requests=self.max_model_requests, deadline=deadline,
                                         clock=self.clock, audit_path=self.directory / "transport_requests.jsonl")
            before = observe("initial_observation")
            if Verifier().check_task(_final_task(request), before).passed:
                result.status, result.error = "blocked", "caller final state already holds; no semantic execution is credited"
                return result
            failure = None
            for round_index in range(1, self.max_rounds + 1):
                if result.planning_calls + 2 > options.max_planning_attempts or result.total_actions >= options.max_actions:
                    result.status, result.error = "blocked", "shared action/planning budget exhausted before semantic proposal"
                    break
                monitor.update("decomposing", planning_pass=result.planning_calls)
                if on_decomposition:
                    on_decomposition({"stage": "semantic_decomposing", "round": round_index, "workflow_id": request.workflow_id})
                plan = self.decomposer.decompose(request,
                    evidence_directory=self.directory / f"semantic_plan_round_{round_index:03d}",
                    budgeted_client=client, initial_observation=before, completed_subgoals=result.completed_subgoals,
                    max_calls=options.max_planning_attempts - result.planning_calls - 1, deadline=deadline,
                    revision=round_index > 1, failure=failure)
                result.plans.append(plan)
                result.planning_calls += plan.planning_calls
                if plan.status != "ready" or plan.workflow is None or plan.proposal is None:
                    result.status, result.error = plan.status, plan.error
                    break
                fingerprint = json.dumps([_signature(goal) for goal in plan.proposal.subgoals], sort_keys=True)
                if fingerprint in seen_plans or any(_signature(goal) in failed_objectives for goal in plan.proposal.subgoals):
                    result.status, result.error = "blocked", "semantic replanning repeated an unchanged plan"
                    break
                seen_plans.add(fingerprint)
                result.semantic_goals_total = len(result.completed_subgoals) + len(plan.proposal.subgoals)
                if result.semantic_goals_total > 32:
                    result.status, result.error = "blocked", "cumulative semantic goal bound of 32 would be exceeded"
                    break
                monitor.update("decomposing", planning_pass=result.planning_calls,
                               subgoals_total=result.semantic_goals_total,
                               subgoals_completed=len(result.completed_subgoals),
                               detail_code=f"semantic_plan_revision_{round_index}")
                failed = False
                for index, (goal, task) in enumerate(zip(plan.proposal.subgoals, plan.workflow.subgoals, strict=True)):
                    if self.clock() >= deadline:
                        raise TimeoutError("shared semantic deadline exhausted before goal")
                    if result.total_actions >= options.max_actions or result.planning_calls >= options.max_planning_attempts:
                        result.status, result.error = "blocked", "shared semantic execution budget exhausted"
                        return result
                    before = observe(f"contract_before_{round_index:03d}_{index + 1:03d}")
                    if Verifier().check_task(task, before).passed:
                        result.status, result.error = "blocked", "semantic goal checks already hold before input; no completion credit"
                        return result
                    goal_dir = self.directory / f"goal_{round_index:03d}_{index + 1:03d}_{goal.goal_id}"
                    _write_new_json(self.directory / f"goal_contract_{round_index:03d}_{index + 1:03d}.json", {
                        "schema_version": SEMANTIC_PROTOCOL_VERSION, "round": round_index, "goal": goal.model_dump(mode="json"),
                        "task": task.model_dump(mode="json"), "before": before.model_dump(mode="json"),
                        "caller_final": _final_task(request).model_dump(mode="json"), "postconditions_fixed_before_input": True,
                    })
                    actions_start, calls_start = result.total_actions, result.planning_calls

                    def checkpoint(event: dict[str, Any], actions_start: int = actions_start,
                                   calls_start: int = calls_start, goal_id: str = goal.goal_id) -> None:
                        monitor.update("executing", actions_used=actions_start + event["actions"],
                                       planning_pass=calls_start + event["planning_calls"], subgoal_id=goal_id,
                                       subgoals_completed=len(result.completed_subgoals))
                        if on_checkpoint:
                            on_checkpoint({**event, "actions": actions_start + event["actions"],
                                           "planning_calls": calls_start + event["planning_calls"],
                                           "completed": len(result.completed_subgoals), "total": result.semantic_goals_total})

                    def factory(current: TaskSpec, _index: int, directory: Path,
                                task: TaskSpec = task, before: ObservationSnapshot = before) -> _SemanticChild:
                        nonlocal wrapper
                        try:
                            child = self.runner_factory(current, len(result.goal_outcomes), directory, client)
                            if current.model_dump() != task.model_dump():
                                raise ValueError("factory changed the frozen semantic task")
                            if getattr(getattr(child, "planner", None), "client", None) is not client:
                                raise ValueError("semantic action planner must use the shared BudgetedModelClient")
                            if isinstance(child, TaskRunner):
                                child.absolute_deadline = deadline if child.absolute_deadline is None else min(deadline, child.absolute_deadline)
                            wrapper = _SemanticChild(child, task, before)
                            return wrapper
                        except (KeyboardInterrupt, Exception) as exc:
                            _write_new_json(directory / "semantic_child_setup_error.json", {
                                "type": type(exc).__name__, "error": str(exc)})
                            raise

                    local = WorkflowSpec(workflow_id=request.workflow_id, instruction=request.instruction,
                                         subgoals=[task], final_check=task.model_copy(deep=True))
                    wrapper, inflight = None, True
                    outcome = WorkflowRunner(goal_dir, factory, clock=self.clock).run(local,
                        options.model_copy(update={"max_actions": options.max_actions - result.total_actions,
                            "max_planning_attempts": min(self.max_goal_planning_calls, options.max_planning_attempts - result.planning_calls),
                            "task_timeout_seconds": max(0.001, deadline - self.clock())}),
                        absolute_deadline=deadline, on_checkpoint=checkpoint, **callbacks)
                    inflight = False
                    result.workflow_result = outcome
                    result.workflow_results.append(outcome)
                    result.total_actions += outcome.total_actions
                    result.planning_calls += outcome.planning_calls
                    result.unconfirmed_action_attempts += outcome.unconfirmed_action_attempts
                    result.accounting_complete &= outcome.accounting_complete
                    record = {"round": round_index, "goal_id": goal.goal_id, "instruction": goal.instruction,
                              "expect_text": list(goal.expect_text), "forbid_text": list(goal.forbid_text),
                              "status": outcome.status, "evidence_directory": str(goal_dir.resolve()),
                              "verification": outcome.final_verification.model_dump(mode="json") if outcome.final_verification else None,
                              "task_verification": outcome.subgoals[0].verification.model_dump(mode="json")
                              if outcome.subgoals[0].verification else None}
                    result.goal_outcomes.append(record)
                    _write_json(self.directory / "semantic_goal_ledger.json", result.goal_outcomes)
                    if wrapper is not None:
                        latest = wrapper.observer.last
                    if outcome.status != "succeeded":
                        failed_objectives.add(_signature(goal))
                        result.status, result.error = outcome.status, outcome.error or "semantic goal did not complete"
                        # Never use replanning to rescue unknown, unreadable, cancelled,
                        # late or incomplete execution. Original failed entries stay failed.
                        if (options.execute and outcome.status == "failed" and outcome.accounting_complete
                                and outcome.unconfirmed_action_attempts == 0 and wrapper is not None
                                and not wrapper.observer.failure and self.clock() < deadline
                                and outcome.subgoals[0].verification is not None
                                and outcome.subgoals[0].verification.outcome == "failed"):
                            before = observe(f"revision_observation_{round_index:03d}")
                            if not Verifier().check_task(_final_task(request), before).passed:
                                failure, failed = record, True
                                break
                        return result
                    result.completed_subgoals.append(record)
                    monitor.update("subgoal_verification", actions_used=result.total_actions,
                                   planning_pass=result.planning_calls, subgoal_id=goal.goal_id,
                                   subgoals_completed=len(result.completed_subgoals))
                    if on_checkpoint:
                        on_checkpoint({"case_id": goal.goal_id, "status": "completed", "actions": result.total_actions,
                                       "planning_calls": result.planning_calls,
                                       "completed": len(result.completed_subgoals), "total": result.semantic_goals_total})
                if failed:
                    continue
                if not options.execute:
                    result.status = "dry_run_completed"
                    break
                final = observe(f"caller_final_round_{round_index:03d}")
                result.final_verification = Verifier().check_task(_final_task(request), final)
                result.final_observation_reference = str((self.directory / f"caller_final_round_{round_index:03d}" / "observation.json").resolve())
                if result.final_verification.passed and result.accounting_complete and not result.unconfirmed_action_attempts:
                    result.status, result.error = "succeeded", ""
                    break
                before, failure = final, {"reason": "completed objectives did not satisfy unchanged caller final predicates"}
                result.status, result.error = "failed", "independent caller final check failed"
            else:
                result.status, result.error = "blocked", "bounded semantic proposal rounds exhausted"
        except (KeyboardInterrupt, Exception) as exc:  # noqa: BLE001 - retain exact failure and known count lower bounds
            result.status = ("cancelled" if isinstance(exc, KeyboardInterrupt) else
                             "timed_out" if isinstance(exc, TimeoutError) or self.clock() >= deadline else "failed")
            result.error = f"{type(exc).__name__}: {exc}"
            if inflight:
                result.accounting_complete = False
                if wrapper is not None:
                    known = wrapper.known_outcome
                    counts = ((known.action_count, known.planning_calls, known.unconfirmed_action_attempts) if known is not None
                              else WorkflowRunner._read_interrupted_counts(wrapper.recorder.directory, wrapper.task.case_id))
                    result.total_actions += counts[0]
                    result.planning_calls += counts[1]
                    result.unconfirmed_action_attempts += counts[2]
                result.notes.append("Interrupted child counts are confirmed lower bounds; no replay or final success is permitted.")
        finally:
            result.model_requests = client.request_count if client else 0
            result.elapsed_ms = max(0, (self.clock() - started) * 1000)
            if self.clock() >= deadline and result.status not in {"cancelled", "timed_out"}:
                result.status, result.error = "timed_out", "global semantic deadline exhausted; " + result.error
            try:
                monitor.update("task_verification", planning_pass=result.planning_calls,
                               subgoals_completed=len(result.completed_subgoals))
                monitor.finish("blocked" if result.status in {"refused", "needs_clarification"} else result.status,
                               actions_used=result.total_actions + result.unconfirmed_action_attempts,
                               subgoals_completed=len(result.completed_subgoals))
            except (KeyboardInterrupt, Exception) as exc:  # noqa: BLE001 - persistence cannot erase known receipts
                result.status = "cancelled" if isinstance(exc, KeyboardInterrupt) else "failed"
                result.error += f"; semantic monitor persistence failed: {type(exc).__name__}"
            try:
                _write_json(self.directory / "automatic_workflow_summary.json", result.model_dump(mode="json"))
            except (KeyboardInterrupt, Exception) as exc:  # noqa: BLE001 - return known accounting even when disk is unavailable
                result.status = "cancelled" if isinstance(exc, KeyboardInterrupt) else "failed"
                result.error += f"; semantic summary persistence failed: {type(exc).__name__}"
        return result
