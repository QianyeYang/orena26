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
    "resources/weights.sha256",
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
FORBIDDEN_CACHE_DIRS = {"__pycache__", ".cache", ".pytest_cache"}
FORBIDDEN_TRAINING_FILES = {
    "optimizer.pt",
    "scheduler.pt",
    "rng_state.pth",
    "trainer_state.json",
    "training_args.bin",
}


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
        if any(part in FORBIDDEN_CACHE_DIRS for part in relative.parts):
            errors.append(f"cache path is forbidden in transferable bundle: {relative}")
        if path.is_file() and path.name in FORBIDDEN_TRAINING_FILES:
            errors.append(f"training-only state is forbidden: {relative}")
        if path.is_file() and path.suffix.lower() == ".sif":
            errors.append(f"Apptainer image must not be bundled: {relative}")
        if (
            path.is_file()
            and path.parent == bundle
            and path.name.lower().endswith(FORBIDDEN_ROOT_SUFFIXES)
        ):
            errors.append(f"Docker/container archive must not be bundled: {relative}")

    dockerignore = bundle / ".dockerignore"
    if dockerignore.is_file():
        patterns = {
            line.strip()
            for line in dockerignore.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        }
        if "*.tar" in patterns and "!resources/python-vendor.tar" not in patterns:
            errors.append(
                ".dockerignore excludes resources/python-vendor.tar required by Dockerfile"
            )

    dockerfile = bundle / "Dockerfile"
    docker_test = bundle / "do_test_run.sh"
    if dockerfile.is_file() and docker_test.is_file():
        if "USER algorithm" in dockerfile.read_text(encoding="utf-8"):
            test_text = docker_test.read_text(encoding="utf-8")
            if 'chmod a+rwx "${OUTPUT_DIR}"' not in test_text:
                errors.append(
                    "non-root Docker image lacks a writable-output preparation step"
                )

    adapter_config = bundle / "resources" / "adapter" / "adapter_config.json"
    if adapter_config.is_file():
        adapter_text = adapter_config.read_text(encoding="utf-8")
        for prefix in ("/dev/shm/", "/datasets/", "/users/"):
            if prefix in adapter_text:
                errors.append(
                    f"adapter_config.json leaks source-cluster path prefix {prefix!r}"
                )

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

    fixture_root = bundle / "test" / "input" / "interface_1"
    request_path = fixture_root / "request.json"
    batch_path = fixture_root / "batch.json"
    if request_path.is_file() and batch_path.is_file():
        try:
            request_rows = json.loads(request_path.read_text(encoding="utf-8"))
            batch_value = json.loads(batch_path.read_text(encoding="utf-8"))
            if not isinstance(request_rows, list) or not request_rows:
                errors.append("test request.json must be a non-empty list")
            else:
                fixture_qids = [
                    row.get("qID") for row in request_rows if isinstance(row, dict)
                ]
                if len(fixture_qids) != len(request_rows):
                    errors.append("every test request must be an object with a qID")
                if (
                    any(not isinstance(qid, str) or not qid for qid in fixture_qids)
                    or len(set(fixture_qids)) != len(fixture_qids)
                ):
                    errors.append("test request qIDs must be non-empty and unique")
                extension = ".png" if args.track == "frame" else ".mp4"
                for qid in fixture_qids:
                    media_path = media_dir / f"{qid}{extension}"
                    if not media_path.is_file() or media_path.stat().st_size == 0:
                        errors.append(
                            f"missing or empty fixture media for qID {qid!r}: "
                            f"{media_path.relative_to(bundle)}"
                        )
                if not isinstance(batch_value, dict):
                    errors.append("test batch.json must be an object")
                else:
                    if batch_value.get("qIDs") != fixture_qids:
                        errors.append("batch.json qIDs do not match request.json order")
                    if batch_value.get("batch_size") != len(fixture_qids):
                        errors.append("batch.json batch_size does not match request count")
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"invalid test request/batch fixture: {exc}")

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
