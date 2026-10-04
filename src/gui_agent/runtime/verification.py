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
    """Everything the frame says, in one searchable string, case-folded.

    The window title and its class are included alongside the OCR text, and they are the
    more reliable half. Measured on T02: a real Google results page carried
    `google.com/search?q=...&gs_lcrp=...` in its address field and the element list for that
    frame did not contain it, while the window title `GUI agent research - Google` was read
    on one frame of the run and missed on the next. The title is read from the window, so it
    does not depend on the OCR engine seeing a narrow field at all - and a rule may
    reasonably name it.

    "" rather than an exception when a session has no window: an empty title contributes
    nothing and the OCR text still decides, which is exactly the behaviour that existed
    before this was added.
    """
    parts = [item.text for item in observation.elements if item.text]
    if observation.window_title:
        parts.append(observation.window_title)
    if observation.window_class:
        parts.append(observation.window_class)
    return " ".join(parts).casefold()


def _missing(needles: list[str], haystack: str) -> list[str]:
    return [needle for needle in needles if needle.casefold() not in haystack]


#: Actions that are not meant to change the screen. 11.1.6 allows them to be
#: verified by their own execution condition and forbids reading a page change as
#: their success; there is nothing else to check, and saying so is the honest
#: answer rather than implying the screen confirmed anything.
_DISPATCH_ONLY_ACTIONS = frozenset({"move", "wait"})


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
        action_type: str | None = None,
    ) -> VerificationResult:
        """Judge a single step.

        A step is never "passed" merely because the action returned success, and
        never because the screen changed - 11.1.5 names that inference as the one
        thing a step check must not make. With no expectation to look for, the
        answer is ``inconclusive``: the frame did move, and that is recorded, but
        nothing here can say the step did what it intended.

        ``move`` and ``wait`` are the exception 11.1.6 allows: they are not meant
        to change the screen at all, so their verification is the action's own
        completion, reported as such rather than as a screen observation.
        """
        if action_result is not None and not action_result.success:
            return VerificationResult(
                outcome="failed",
                detail=f"action failed: {action_result.error or 'unknown error'}",
                evidence={"error": action_result.error},
            )

        when = _DISPATCH_ONLY_ACTIONS if action_type in _DISPATCH_ONLY_ACTIONS else None
        if when is not None and not (expected_result or "").strip():
            if action_result is None:
                return VerificationResult(
                    outcome="inconclusive",
                    detail=f"{action_type}: no result recorded for the action",
                )
            return VerificationResult(
                outcome="passed",
                detail=(
                    f"{action_type} completed; it does not change the screen, so this is "
                    "the action's own completion and not an observation of a result"
                ),
                evidence={"action_type": action_type},
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
            # 11.1.5: a changed screenshot is not success. It is worth recording -
            # the action did have some effect - but it does not say the step did
            # what it was for, and no expectation was given to check it against.
            return VerificationResult(
                outcome="inconclusive",
                detail=(
                    "the screen changed after the action, which is not evidence that the "
                    "step did what it intended; no expected result was supplied to check"
                ),
                evidence=evidence,
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

        Interfaces are injected so the loop can be tested without a screen. A
        failure to observe is not a failure of the task: the screen went away, and
        the honest verdict for a screen nobody can look at is ``inconclusive``.
        Letting that exception out ended the whole run as a traceback, which is what
        a sleeping display used to do to a run that had otherwise finished.
        """
        started = clock()
        latest: ObservationSnapshot | None = None
        while True:
            try:
                latest = observe()
            except Exception as exc:  # noqa: BLE001 - the screen went away
                return (
                    VerificationResult(
                        outcome="inconclusive",
                        method=task.verification,
                        detail=(
                            "could not look at the screen to verify: "
                            f"{type(exc).__name__}: {exc}"
                        ),
                    ),
                    latest,
                )
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
