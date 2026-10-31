"""Wait for the Docker stack and send a smoke-test image to the API."""

from __future__ import annotations

import argparse
import base64
import os
import sys
import time
from pathlib import Path

import requests


DEFAULT_IMAGE = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk"
    "YAAAAAYAAjCB0C8AAAAASUVORK5CYII="
)


def load_dotenv(path: Path) -> dict[str, str]:
    """Read simple KEY=VALUE entries without overriding shell variables."""
    values: dict[str, str] = {}
    if not path.is_file():
        return values
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values.setdefault(key.strip(), value.strip().strip("\"'"))
    return values


def wait_for_health(url: str, api_key: str, timeout: float, interval: float) -> None:
    deadline = time.monotonic() + timeout
    last_error = "no response"
    headers = {"X-API-Key": api_key}
    while time.monotonic() < deadline:
        try:
            response = requests.get(f"{url}/health", headers=headers, timeout=5)
            if response.ok:
                body = response.json()
                if body.get("status") == "ok":
                    print(f"API is healthy: {body}")
                    return
                last_error = f"unexpected health response: {body}"
            else:
                last_error = f"HTTP {response.status_code}: {response.text}"
        except requests.RequestException as error:
            last_error = str(error)
        time.sleep(interval)
    raise TimeoutError(f"API did not become healthy within {timeout:g}s: {last_error}")


def analyze_sample(url: str, api_key: str, image_path: Path | None) -> None:
    image = image_path.read_bytes() if image_path else DEFAULT_IMAGE
    filename = image_path.name if image_path else "facesense-smoke-test.png"
    response = requests.post(
        f"{url}/analyze",
        headers={"X-API-Key": api_key},
        files={"image": (filename, image, "image/png")},
        timeout=30,
    )
    if not response.ok:
        raise RuntimeError(f"/analyze failed with HTTP {response.status_code}: {response.text}")
    result = response.json()
    if not isinstance(result.get("faces"), list):
        raise RuntimeError(f"/analyze returned an unexpected response: {result}")
    print(f"/analyze succeeded: {len(result['faces'])} face(s) detected")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--image", type=Path, help="Optional JPEG/PNG/WebP sample image")
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--interval", type=float, default=2)
    args = parser.parse_args()
    if args.timeout <= 0 or args.interval <= 0:
        parser.error("--timeout and --interval must be positive")

    dotenv = load_dotenv(args.env_file)
    api_key = os.getenv("FACESENSE_API_KEY") or dotenv.get("FACESENSE_API_KEY")
    if not api_key or api_key.startswith("replace-with-"):
        print("Set FACESENSE_API_KEY in .env or the shell before running.", file=sys.stderr)
        return 2
    base_url = (
        os.getenv("FACESENSE_API_URL")
        or dotenv.get("FACESENSE_API_URL")
        or f"http://localhost:{os.getenv('API_PORT') or dotenv.get('API_PORT', '8000')}"
    ).rstrip("/")

    try:
        wait_for_health(base_url, api_key, args.timeout, args.interval)
        analyze_sample(base_url, api_key, args.image)
    except (OSError, RuntimeError, TimeoutError, ValueError) as error:
        print(f"Stack smoke test failed: {error}", file=sys.stderr)
        return 1
    print("Stack smoke test passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
