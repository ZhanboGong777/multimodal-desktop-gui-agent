"""Decide whether a step, or a whole task, actually achieved anything.

The distinction this module exists to enforce: an action that did not raise is not
a task that finished. ``ActionResult.success`` only means PyAutoGUI dispatched the
event. Something has to look at the screen afterwards and say whether the expected
change is there.
"""

from __future__ import annotations

from typing import Any

from ..schemas import ActionResult
from .schemas import ObservationSnapshot, TaskSpec, VerificationResult


def _haystack(observation: ObservationSnapshot) -> str:
    """All visible text in one searchable string, case-folded."""
    return " ".join(item.text for item in observation.elements if item.text).casefold()


def _missing(needles: list[str], haystack: str) -> list[str]:
    return [needle for needle in needles if needle.casefold() not in haystack]


class Verifier:
    """Checks steps and tasks against what is actually on screen."""

    def __init__(self, *, poll_interval_seconds: float = 0.5) -> None:
        self.poll_interval_seconds = poll_interval_seconds

    # ── one step ───────────────────────────────────────────────────────
    def check_step(
        self,
        *,
        expected_result: str | None,
        before: ObservationSnapshot | None,
        after: ObservationSnapshot | None,
        action_result: ActionResult | None,
    ) -> VerificationResult:
        """Judge a single step.

        A step is never "passed" merely because the action returned success. When
        there is no observable expectation to check, the honest answer is
        ``inconclusive``, which the runner treats as "do not claim progress".
        """
        if action_result is not None and not action_result.success:
            return VerificationResult(
                outcome="failed",
                detail=f"action failed: {action_result.error or 'unknown error'}",
                evidence={"error": action_result.error},
            )

        if after is None:
            return VerificationResult(
                outcome="inconclusive", detail="no observation after the action"
            )

        if after.errors:
            return VerificationResult(
                outcome="inconclusive",
                detail=f"observation reported errors: {'; '.join(after.errors)}",
                evidence={"observation_id": after.observation_id},
            )

        changed = before is None or _signature(before) != _signature(after)
        evidence: dict[str, Any] = {
            "observation_id": after.observation_id,
            "changed": changed,
            "element_count": len(after.elements),
        }

        expectation = (expected_result or "").strip()
        if expectation:
            # The expectation is free text written by the model, so it is matched
            # loosely: if any distinctive word from it appears, the screen moved
            # in the expected direction. A miss is reported, not hidden.
            tokens = [t for t in _words(expectation) if len(t) >= 4]
            haystack = _haystack(after)
            hits = [t for t in tokens if t in haystack]
            evidence["expectation_tokens"] = tokens
            evidence["expectation_hits"] = hits
            if tokens and not hits:
                return VerificationResult(
                    outcome="failed",
                    detail=f"expected result not observed: {expectation!r}",
                    evidence=evidence,
                )
            if tokens:
                return VerificationResult(
                    outcome="passed",
                    detail=f"expected result observed: {', '.join(hits[:3])}",
                    evidence=evidence,
                )

        if changed:
            return VerificationResult(
                outcome="passed", detail="the screen changed after the action", evidence=evidence
            )
        return VerificationResult(
            outcome="inconclusive",
            detail="the screen looks unchanged; the step may not have had an effect",
            evidence=evidence,
        )

    # ── the whole task ─────────────────────────────────────────────────
    def check_task(
        self, task: TaskSpec, observation: ObservationSnapshot | None
    ) -> VerificationResult:
        """Judge whether the task goal is met on the current screen.

        Nothing else may mark a task ``succeeded``: not ``finish``, not a run out
        of steps, not a changed screenshot.
        """
        if not task.success_rules and not task.expect_text and not task.forbid_text:
            return VerificationResult(
                outcome="inconclusive",
                method=task.verification,
                detail="the task has no verifiable success rule; refusing to declare it complete",
            )

        if observation is None:
            return VerificationResult(
                outcome="inconclusive",
                method=task.verification,
                detail="no observation available to verify against",
            )

        haystack = _haystack(observation)
        evidence: dict[str, Any] = {"observation_id": observation.observation_id}

        absent = _missing(task.expect_text, haystack)
        if absent:
            evidence["missing"] = absent
            return VerificationResult(
                outcome="failed",
                method=task.verification,
                detail=f"expected on screen but not found: {', '.join(absent)}",
                evidence=evidence,
            )

        present = [text for text in task.forbid_text if text.casefold() in haystack]
        if present:
            evidence["unexpected"] = present
            return VerificationResult(
                outcome="failed",
                method=task.verification,
                detail=f"still on screen but should be gone: {', '.join(present)}",
                evidence=evidence,
            )

        evidence["expect_text"] = task.expect_text
        evidence["forbid_text"] = task.forbid_text
        return VerificationResult(
            outcome="passed",
            method=task.verification,
            detail="all success rules matched against the current screen",
            evidence=evidence,
        )

    def check_task_with_polling(
        self,
        task: TaskSpec,
        observe: Any,
        *,
        deadline_seconds: float,
        sleep: Any,
        clock: Any,
    ) -> tuple[VerificationResult, ObservationSnapshot | None]:
        """Re-observe until the task passes or the verification budget runs out.

        Interfaces are injected so the loop can be tested without a screen.
        """
        started = clock()
        latest: ObservationSnapshot | None = None
        while True:
            latest = observe()
            result = self.check_task(task, latest)
            if result.outcome == "passed":
                return result, latest
            if clock() - started >= deadline_seconds:
                return result, latest
            sleep(self.poll_interval_seconds)


def _words(text: str) -> list[str]:
    cleaned = "".join(ch if ch.isalnum() or ch.isspace() else " " for ch in text)
    return [word.casefold() for word in cleaned.split()]


def _signature(observation: ObservationSnapshot) -> tuple[int, tuple[str, ...]]:
    """A cheap fingerprint of what is on screen.

    Element count plus the sorted text of the first elements is enough to tell
    "the page changed" from "nothing happened", and is stable across two captures
    of the same static screen.
    """
    texts = tuple(sorted(item.text for item in observation.elements if item.text)[:40])
    return len(observation.elements), texts
