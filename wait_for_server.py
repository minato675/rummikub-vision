"""Wait until the local Roboflow Inference HTTP service is ready."""

from __future__ import annotations

import argparse
import time

import requests

from rummikub_inference import load_settings


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeout", type=float, default=180)
    args = parser.parse_args()
    settings = load_settings(require_api_key=False)
    deadline = time.monotonic() + args.timeout
    last_error = "not started"
    while time.monotonic() < deadline:
        try:
            response = requests.get(f"{settings.server_url}/", timeout=3)
            if response.status_code < 500:
                print(f"Inference server ready: {settings.server_url} (HTTP {response.status_code})")
                return 0
            last_error = f"HTTP {response.status_code}"
        except requests.RequestException as exc:
            last_error = str(exc)
        time.sleep(2)
    raise TimeoutError(f"Inference server did not become ready: {last_error}")


if __name__ == "__main__":
    raise SystemExit(main())

