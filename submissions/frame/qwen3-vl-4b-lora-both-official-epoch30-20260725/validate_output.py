#!/usr/bin/env python3
"""Validate a FOCUS answer.json against its request.json."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--requests", required=True, type=Path)
    parser.add_argument("--answers", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    requests = json.loads(args.requests.read_text(encoding="utf-8"))
    answers = json.loads(args.answers.read_text(encoding="utf-8"))
    errors: list[str] = []

    if not isinstance(requests, list):
        errors.append("request.json is not a list")
        requests = []
    if not isinstance(answers, list):
        errors.append("answer.json is not a list")
        answers = []

    expected = [row.get("qID") for row in requests if isinstance(row, dict)]
    actual = [row.get("qID") for row in answers if isinstance(row, dict)]
    if len(answers) != len(requests):
        errors.append(f"response count {len(answers)} != request count {len(requests)}")
    if len(set(actual)) != len(actual):
        errors.append("response qIDs are not unique")
    if set(actual) != set(expected):
        errors.append(
            f"response qID set differs: missing={sorted(set(expected) - set(actual))} "
            f"extra={sorted(set(actual) - set(expected))}"
        )

    for index, row in enumerate(answers):
        if not isinstance(row, dict):
            errors.append(f"response {index} is not an object")
            continue
        if not isinstance(row.get("qID"), str) or not row["qID"]:
            errors.append(f"response {index} has invalid qID")
        if not isinstance(row.get("content"), str):
            errors.append(f"response {index} has non-string content")
        elif len(row["content"].strip()) > 300:
            errors.append(f"response {index} content exceeds 300 characters")
        latency = row.get("latency")
        if (
            not isinstance(latency, (int, float))
            or isinstance(latency, bool)
            or not math.isfinite(float(latency))
            or latency < 0
        ):
            errors.append(f"response {index} has invalid latency")

    if errors:
        print("FAIL")
        for error in errors:
            print(f"- {error}")
        return 1
    print(f"PASS: {len(answers)} responses; qIDs/count/types/latencies valid")
    return 0


if __name__ == "__main__":
    sys.exit(main())
