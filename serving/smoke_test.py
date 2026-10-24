"""Optional local-server smoke check. Does nothing unless --hit is passed."""

from __future__ import annotations

import argparse
import sys

import httpx2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="GET /v1/models on a local vLLM server.")
    parser.add_argument("--hit", action="store_true", help="Actually call the server.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    args = parser.parse_args(argv)
    if not args.hit:
        print("smoke_test: not asked to hit the server.")
        return 0
    response = httpx2.get(f"{args.base_url.rstrip('/')}/models", timeout=10)
    print(response.status_code)
    print(response.text[:500])
    return 0 if response.status_code == 200 else 1


if __name__ == "__main__":
    sys.exit(main())
