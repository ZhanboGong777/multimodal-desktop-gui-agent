"""Exercise the model client without a network or an API key.

    python scripts/week3_model_demo.py --provider mock
    python scripts/week3_model_demo.py --provider openai_compatible --prompt "hello"

The mock path must work offline - it is what the tests and the Windows machine use.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from gui_agent.config import load_config
from gui_agent.models import create_model_client
from gui_agent.models.base import ModelError


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(REPO_ROOT / "configs" / "default.yaml"))
    parser.add_argument("--provider", choices=["mock", "openai_compatible"], default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--prompt", default="Describe what a GUI agent does in one sentence.")
    parser.add_argument("--image", default=None, help="image path passed as multimodal context")
    parser.add_argument("--health-check", action="store_true", help="probe the backend first")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    if args.provider:
        config.model.provider = args.provider
    if args.model:
        config.model.model_name = args.model
    if args.base_url:
        config.model.base_url = args.base_url

    client = create_model_client(config.model)
    print(f"provider   : {client.name}")
    print(f"model      : {client.model_name}")
    print(f"base_url   : {getattr(client, 'base_url', None) or '(provider default)'}")
    print()

    if args.health_check:
        ok = client.health_check()
        print(f"health     : {'ok' if ok else 'unavailable'}")
        if not ok:
            print("            (the mock backend always reports ok; a real one needs")
            print("             GUI_AGENT_API_KEY and a reachable base_url)")

    try:
        response = client.generate_multimodal(args.prompt, image_path=args.image)
    except ModelError as exc:
        print(f"error      : {exc}")
        return 1

    print()
    print(f"latency    : {response.latency_ms:.1f} ms")
    print(f"usage      : {json.dumps(response.usage, ensure_ascii=False) or '{}'}")
    print(f"ok         : {response.ok}")
    if response.error:
        print(f"error      : {response.error}")
        return 1

    print()
    print("--- content ---")
    print(response.content)
    print()
    print("--- log-safe view (no credentials, no raw payload) ---")
    print(json.dumps(response.as_dict(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
