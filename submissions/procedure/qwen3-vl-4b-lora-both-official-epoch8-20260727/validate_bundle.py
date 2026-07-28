#!/usr/bin/env python3
"""Validate the static structure of an ORena FOCUS submission bundle."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

TRACKS = {"frame", "segment", "procedure"}
REQUIRED_FILES = {
    ".dockerignore",
    ".gitignore",
    "Dockerfile",
    "LICENSE",
    "NOTICE",
    "apptainer.def",
    "do_build.sh",
    "do_build_apptainer.sh",
    "do_save.sh",
    "do_test_apptainer.sh",
    "do_test_run.sh",
    "guidance.md",
    "inference.py",
    "provenance.json",
    "requirements.txt",
    "validate_bundle.py",
    "validate_output.py",
}
EXECUTABLE_FILES = {
    "do_build.sh",
    "do_build_apptainer.sh",
    "do_save.sh",
    "do_test_apptainer.sh",
    "do_test_run.sh",
    "validate_bundle.py",
    "validate_output.py",
}
FORBIDDEN_ROOT_SUFFIXES = (".tar", ".tar.gz", ".tgz")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--track", required=True, choices=sorted(TRACKS))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    bundle = args.bundle.resolve()
    errors: list[str] = []
    provenance_value: dict[str, object] = {}

    if not bundle.is_dir():
        errors.append(f"bundle is not a directory: {bundle}")
    if bundle.parent.name != args.track:
        errors.append(
            f"bundle must be directly under submissions/{args.track}/, "
            f"not {bundle.parent.name!r}"
        )

    for relative in sorted(REQUIRED_FILES):
        path = bundle / relative
        if not path.is_file():
            errors.append(f"missing required file: {relative}")

    for directory in ("resources", "test/input/interface_1"):
        path = bundle / directory
        if not path.is_dir() or not any(path.iterdir()):
            errors.append(f"missing or empty directory: {directory}")

    media = "frames" if args.track == "frame" else "plain"
    media_dir = bundle / "test" / "input" / "interface_1" / media
    if not media_dir.is_dir() or not any(media_dir.iterdir()):
        errors.append(f"missing or empty test media directory: {media_dir.relative_to(bundle)}")

    for relative in EXECUTABLE_FILES:
        path = bundle / relative
        if path.is_file() and not os.access(path, os.X_OK):
            errors.append(f"script is not executable: {relative}")

    for path in bundle.rglob("*"):
        relative = path.relative_to(bundle)
        if path.is_symlink():
            errors.append(f"symlink is forbidden in transferable bundle: {relative}")
        if path.is_file() and path.suffix.lower() == ".sif":
            errors.append(f"Apptainer image must not be bundled: {relative}")
        if (
            path.is_file()
            and path.parent == bundle
            and path.name.lower().endswith(FORBIDDEN_ROOT_SUFFIXES)
        ):
            errors.append(f"Docker/container archive must not be bundled: {relative}")

    provenance = bundle / "provenance.json"
    if provenance.is_file():
        try:
            value = json.loads(provenance.read_text())
            if not isinstance(value, dict):
                errors.append("provenance.json root must be an object")
            else:
                provenance_value = value
                for key in ("track", "algorithm", "source_model", "source_checkpoint"):
                    if not value.get(key):
                        errors.append(f"provenance.json lacks non-empty {key!r}")
                if value.get("track") != args.track:
                    errors.append("provenance.json track does not match --track")
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"invalid provenance.json: {exc}")

    container = provenance_value.get("container", {})
    if isinstance(container, dict) and container.get("dtype") == "bfloat16":
        bf16_requirements = {
            "Dockerfile": ("FOCUS_MODEL_DTYPE=bfloat16",),
            "inference.py": ("FOCUS_MODEL_DTYPE", '"float16": torch.float16'),
            "do_test_run.sh": ("FOCUS_TEST_MODEL_DTYPE", "--gpus=all"),
            "do_save.sh": ("FOCUS_MODEL_DTYPE=bfloat16",),
            "guidance.md": ("V100", "FP16", "BF16"),
        }
        for relative, fragments in bf16_requirements.items():
            path = bundle / relative
            if not path.is_file():
                continue
            text = path.read_text(encoding="utf-8")
            for fragment in fragments:
                if fragment not in text:
                    errors.append(
                        f"{relative} lacks BF16-final/V100-test safeguard: {fragment}"
                    )

    for name in ("request.json", "FO_definitions.json", "batch.json"):
        path = bundle / "test" / "input" / "interface_1" / name
        if not path.is_file():
            errors.append(f"missing test fixture: {path.relative_to(bundle)}")

    if errors:
        print("FAIL")
        for error in errors:
            print(f"- {error}")
        return 1

    regular_files = sum(path.is_file() for path in bundle.rglob("*"))
    total_bytes = sum(path.stat().st_size for path in bundle.rglob("*") if path.is_file())
    print(f"PASS: {bundle}")
    print(f"track={args.track} regular_files={regular_files} bytes={total_bytes}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
