"""Warm the model up before a week's real tasks, and record what it cost.

    13.4 asks for this and nothing provided it: the warmup has to happen before
    the tasks, its time must not land inside a warm-start task's timing, and the
    probe has to include an image - a text-only warmup passes while the vision
    path is still cold. It also asks for cold/warm state, available memory, the
    request time, a failure classification and the retry count to be recorded
    rather than remembered.

    python scripts/week4_warmup.py --config configs/week4.yaml
    python scripts/week4_warmup.py --config configs/week4.yaml --image shot.png
    python scripts/week4_warmup.py --config configs/week4.yaml --json warmup.json

Separate process on purpose. The first request a freshly started server sees pays
for loading weights and warming the vision encoder, and on the Windows box that is
the difference between about 8 s and about 72 s; a task whose timing includes it
looks like a model problem when it is a scheduling one. Running this first means
the five task runs are all warm starts, and the cold number is recorded here where
it belongs.

The probes go through the project's own client, not curl. 13.4.6 is explicit that
a long-timeout probe succeeding says nothing about the client the tasks use, and
that a wall clock over the timeout is not proof of failure either - the only
answer that settles it is the client's own, which is what this measures.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from week4_agent_cli import load_environment

from gui_agent.config import Config, load_config, resolve_model_config
from gui_agent.models import CLIENTS, create_model_client
from gui_agent.runtime.runner import explain_model_failure

#: The text probe. Short on purpose: it is here to load the weights, and a longer
#: answer only adds decode time to a number whose meaning is "how long until the
#: server was ready".
TEXT_PROBE = "Reply with the single word: ready"

#: The image probe. It asks about the picture rather than chatting, so an answer
#: that ignores the image is visibly wrong rather than plausibly generic.
IMAGE_PROBE = (
    "In one short sentence, what is the dominant colour or background of this "
    "screenshot? Answer from the image."
)


@dataclass
class Probe:
    """One request's outcome, in the shape 13.4.3 asks to record."""

    kind: str
    state: str
    ok: bool
    latency_ms: float
    timeout_seconds: float
    error: str | None = None
    failure_class: str | None = None
    content: str = ""
    usage: dict[str, Any] = field(default_factory=dict)

    def describe(self) -> str:
        verdict = "ok" if self.ok else "FAILED"
        line = (
            f"  {self.kind:<5} {self.state:<9} {self.latency_ms:>9.1f} ms  "
            f"{verdict:<6} (timeout {self.timeout_seconds:g}s)"
        )
        if self.error:
            line += f"\n        {self.failure_class}"
        return line


def available_memory_mb() -> float | None:
    """Free memory in MB, or None when it cannot be read.

    `psutil` arrives transitively through PaddleOCR's optional extras, so it is
    used when present and reported as unknown otherwise - a warmup that refused to
    run without a statistics library would be a worse trade than one missing a
    column.
    """
    try:
        import psutil
    except ImportError:
        return None
    return round(psutil.virtual_memory().available / 1e6, 1)


def _classify(message: str) -> str:
    """The project's own classifier, so a warmup failure reads like a run failure."""
    return explain_model_failure(message)


def probe(client: Any, *, kind: str, state: str, image: str | None) -> Probe:
    """One request through the client, timed from the caller's side.

    The client reports its own latency, but a request that never reached the model
    has none, and the interesting number on a cold start is the whole wait.
    """
    timeout = float(getattr(client, "timeout_seconds", 0.0))
    started = time.perf_counter()
    if kind == "image":
        response = client.generate_multimodal(IMAGE_PROBE, image_path=image)
    else:
        response = client.generate_text(TEXT_PROBE)
    elapsed_ms = (time.perf_counter() - started) * 1000.0

    if not response.ok:
        return Probe(
            kind=kind,
            state=state,
            ok=False,
            latency_ms=round(elapsed_ms, 1),
            timeout_seconds=timeout,
            error=response.error or "the model returned nothing usable",
            failure_class=_classify(response.error or "the model returned nothing usable"),
        )
    return Probe(
        kind=kind,
        state=state,
        ok=True,
        latency_ms=round(elapsed_ms, 1),
        timeout_seconds=timeout,
        content=response.content.strip()[:200],
        usage=dict(response.usage),
    )


def _screenshot(destination: Path, monitor_index: int) -> Path:
    """Capture the screen through the same layer the runtime uses.

    Not a test image and not a stock picture: 13.4.2 wants the vision path proved
    with the kind of request the tasks will send, and the expensive part is
    encoding a full-resolution frame, which a small fixture would not exercise.
    """
    from gui_agent.perception.capture import capture_monitor

    capture = capture_monitor(monitor_index, output_directory=destination, save=True)
    if capture.image_path is None:
        raise RuntimeError("the capture produced no file to send")
    return Path(capture.image_path)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", default=str(REPO_ROOT / "configs" / "week4.yaml"))
    parser.add_argument("--provider", choices=sorted(CLIENTS), default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument(
        "--image",
        default=None,
        help="screenshot for the vision probe; without it a fresh one is captured",
    )
    parser.add_argument(
        "--json", default=None, help="write the record here instead of only printing it"
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=0,
        help="extra image probes, to show the warm steady state next to the cold one",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    dotenv_loaded = load_environment()
    args = parse_args(argv)

    config: Config = load_config(args.config)
    resolve_model_config(
        config,
        provider=args.provider,
        model=args.model,
        base_url=args.base_url,
    )

    image = args.image
    captured: Path | None = None
    if image is None:
        try:
            # Its own folder, not the directory the sessions live in: the probe
            # frame belongs to no run, and dropping it among the session folders
            # made the output tree read as though a run had produced it.
            captured = _screenshot(
                REPO_ROOT / "outputs" / "week4" / "warmup", config.perception.monitor_index
            )
        except Exception as exc:  # noqa: BLE001 - any capture failure is the same answer
            print(f"warmup     : could not capture the screen ({type(exc).__name__}: {exc})")
            print("             pass --image with a real screenshot instead; a text-only")
            print("             warmup would leave the vision path cold (13.4.2).")
            return 2
        image = str(captured)

    memory_before = available_memory_mb()
    client = create_model_client(config.model)

    print("warmup     : 13.4 - run this before the week's tasks, not as part of them")
    print(f"provider   : {client.name}")
    print("env file   : ./.env" if dotenv_loaded else "env file   : none in this directory")
    print(f"model      : {client.model_name}")
    print(f"base_url   : {getattr(client, 'base_url', None) or '(provider default)'}")
    print(f"screenshot : {image}")
    print(f"timeout    : {config.model.timeout_seconds:g}s per request")
    print(f"max_retries: {config.model.max_retries}")
    print(
        "memory     : "
        + (f"{memory_before} MB available" if memory_before is not None else "unknown")
    )
    print()

    # The first probe of a fresh server is the cold one. It is labelled that way
    # because that is what it is for this process, not because the server's state
    # was inspected - a server already holding the model turns this into a second
    # warm sample, which the record shows by the two numbers agreeing.
    probes: list[Probe] = [
        probe(client, kind="text", state="cold-or-idle", image=None),
        probe(client, kind="image", state="vision-path", image=image),
    ]
    for index in range(max(0, args.repeat)):
        state = "warm" if index else "warm-repeat"
        probes.append(probe(client, kind="image", state=state, image=image))

    for entry in probes:
        print(entry.describe())
        if entry.ok and entry.content:
            print(f"        {entry.content}")

    memory_after = available_memory_mb()
    print()
    print(
        "memory     : "
        + (f"{memory_after} MB available after" if memory_after is not None else "unknown")
    )

    text_probe = probes[0]
    image_probe = probes[1]
    ready = text_probe.ok and image_probe.ok
    if ready:
        print()
        print("ready      : the model answered text and image; task runs will be warm starts")
        print(
            "note       : these times are this machine, this load and this model. "
            "13.4.4 - they are not a fixed cost of the model."
        )
    else:
        print()
        print("not ready  : fix this before running the tasks. A task run started now")
        print("             would record a cold or failed request as its own planning time.")

    record = {
        "captured_at": datetime.now(UTC).isoformat(),
        "provider": client.name,
        "model_name": client.model_name,
        "base_url": getattr(client, "base_url", None),
        "screenshot": image,
        "screenshot_captured_here": captured is not None,
        "timeout_seconds": config.model.timeout_seconds,
        "max_retries_configured": config.model.max_retries,
        "memory_available_mb_before": memory_before,
        "memory_available_mb_after": memory_after,
        "ready": ready,
        "probes": [asdict(entry) for entry in probes],
    }
    if args.json:
        destination = Path(args.json)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        print(f"record     : {destination}")

    print()
    print(json.dumps({"ready": ready, "probes": len(probes)}, ensure_ascii=False))
    return 0 if ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
