#!/usr/bin/env python3
"""Local HTTP server for the FOCUS Frame annotation tool."""

from __future__ import annotations

import argparse
import csv
import json
import mimetypes
import os
import re
import sys
import tempfile
import threading
import traceback
from collections import Counter, defaultdict
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from guide import CLASS_NAMES, FO_GUIDE
from validation import (
    AnnotationError,
    empty_annotation,
    normalise_annotation,
    validate_annotation,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
APP_ROOT = Path(__file__).resolve().parents[1]
STATIC_ROOT = APP_ROOT / "static"
DEFAULT_DATA_ROOT = REPO_ROOT / "data/annotations/frame-annotation-tool"
_CASE_ID = re.compile(r"^[a-z]+-[0-9a-f]{16}$")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        os.fchmod(fd, 0o664)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    except BaseException:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise


class AnnotationStore:
    def __init__(self, data_root: Path) -> None:
        self.data_root = data_root
        self.pool_path = data_root / "pool.json"
        self.records_dir = data_root / "records"
        self.exports_dir = data_root / "exports"
        self.audit_path = data_root / "audit.jsonl"
        self.lock = threading.RLock()
        if not self.pool_path.is_file():
            raise FileNotFoundError(
                f"annotation pool not found: {self.pool_path}; run build_pool.py"
            )
        self.pool = json.loads(self.pool_path.read_text(encoding="utf-8"))
        self.cases = self.pool["cases"]
        self.case_by_id = {case["id"]: case for case in self.cases}
        self.guide_case_by_id = {
            case["id"]: case for case in self.pool.get("extra_guide_cases", [])
        }
        self.all_cases = self.case_by_id | self.guide_case_by_id
        if len(self.case_by_id) != len(self.cases):
            raise ValueError("pool contains duplicate case IDs")
        self.records_dir.mkdir(parents=True, exist_ok=True)
        self.exports_dir.mkdir(parents=True, exist_ok=True)
        self._status_cache = {
            case_id: {
                "status": "unstarted",
                "revision": 0,
                "box_count": 0,
                "updated_at": None,
            }
            for case_id in self.case_by_id
        }
        for path in self.records_dir.glob("*.json"):
            if path.stem not in self.case_by_id:
                continue
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise RuntimeError(
                    f"cannot read annotation record {path}: {exc}"
                ) from exc
            self._status_cache[path.stem] = self._status(record)

    @staticmethod
    def _status(record: dict[str, Any]) -> dict[str, Any]:
        return {
            "status": record["status"],
            "revision": record["revision"],
            "box_count": len(record.get("boxes", [])),
            "updated_at": record.get("updated_at"),
        }

    def _record_path(self, case_id: str) -> Path:
        if case_id not in self.case_by_id or not _CASE_ID.fullmatch(case_id):
            raise KeyError(case_id)
        return self.records_dir / f"{case_id}.json"

    def read(self, case_id: str) -> dict[str, Any]:
        path = self._record_path(case_id)
        if not path.is_file():
            return empty_annotation(case_id)
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"cannot read annotation record {path}: {exc}") from exc
        return value

    def status_map(self) -> dict[str, dict[str, Any]]:
        with self.lock:
            return {
                case_id: dict(status)
                for case_id, status in self._status_cache.items()
            }

    def save(
        self,
        case_id: str,
        raw: dict[str, Any],
        *,
        expected_revision: int,
        client: str,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        case = self.case_by_id.get(case_id)
        if case is None:
            raise KeyError(case_id)
        with self.lock:
            previous = self.read(case_id)
            if previous["revision"] != expected_revision:
                raise RevisionConflict(previous)
            record = normalise_annotation(raw, case, previous=previous)
            now = _utc_now()
            record["revision"] = previous["revision"] + 1
            record["created_at"] = previous.get("created_at") or now
            record["updated_at"] = now
            _atomic_json(self._record_path(case_id), record)
            self._status_cache[case_id] = self._status(record)
            report = validate_annotation(record, case)
            audit = {
                "timestamp": now,
                "case_id": case_id,
                "revision": record["revision"],
                "previous_status": previous["status"],
                "status": record["status"],
                "box_count": len(record["boxes"]),
                "client": client,
            }
            with self.audit_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(audit, ensure_ascii=False) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            return record, report

    def case_payload(self, case_id: str) -> dict[str, Any]:
        case = self.all_cases.get(case_id)
        if case is None:
            raise KeyError(case_id)
        annotation = (
            self.read(case_id)
            if case_id in self.case_by_id
            else empty_annotation(case_id)
        )
        return {
            "case": case,
            "annotation": annotation,
            "validation": validate_annotation(annotation, case),
            "read_only": case_id not in self.case_by_id,
        }

    def progress(self) -> dict[str, Any]:
        statuses = self.status_map()
        totals = Counter(item["status"] for item in statuses.values())
        batches: dict[int, Counter[str]] = defaultdict(Counter)
        datasets: dict[str, Counter[str]] = defaultdict(Counter)
        for case in self.cases:
            status = statuses[case["id"]]["status"]
            batches[int(case["batch"])][status] += 1
            datasets[case["dataset"]][status] += 1

        def serialise(counter: Counter[str]) -> dict[str, int]:
            values = {
                status: int(counter.get(status, 0))
                for status in (
                    "complete",
                    "needs_review",
                    "in_progress",
                    "skipped",
                    "unstarted",
                )
            }
            values["done"] = values["complete"]
            values["started"] = sum(
                value
                for status, value in values.items()
                if status not in {"unstarted", "done", "started"}
            )
            return values

        return {
            "total": len(self.cases),
            "statuses": serialise(totals),
            "batches": [
                {
                    "batch": batch,
                    "size": sum(counter.values()),
                    **serialise(counter),
                }
                for batch, counter in sorted(batches.items())
            ],
            "datasets": {
                dataset: serialise(counter)
                for dataset, counter in sorted(datasets.items())
            },
            "updated_at": max(
                (
                    item["updated_at"]
                    for item in statuses.values()
                    if item["updated_at"]
                ),
                default=None,
            ),
        }

    def export(self) -> dict[str, Any]:
        with self.lock:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            export_dir = self.exports_dir / stamp
            suffix = 1
            while export_dir.exists():
                export_dir = self.exports_dir / f"{stamp}-{suffix}"
                suffix += 1
            export_dir.mkdir(parents=True)

            records: list[tuple[dict[str, Any], dict[str, Any]]] = []
            progress_rows: list[dict[str, Any]] = []
            for case in self.cases:
                record = self.read(case["id"])
                report = validate_annotation(record, case)
                progress_rows.append(
                    {
                        "rank": case["rank"],
                        "batch": case["batch"],
                        "case_id": case["id"],
                        "dataset": case["dataset"],
                        "status": record["status"],
                        "box_count": len(record["boxes"]),
                        "revision": record["revision"],
                        "updated_at": record.get("updated_at") or "",
                        "qa_mismatch_count": len(report["qa_mismatches"]),
                    }
                )
                if record["status"] == "complete":
                    records.append((case, record))

            canonical_path = export_dir / "annotations-complete.jsonl"
            with canonical_path.open("w", encoding="utf-8") as handle:
                for case, record in records:
                    handle.write(
                        json.dumps(
                            {
                                "case_id": case["id"],
                                "dataset": case["dataset"],
                                "video": case["video"],
                                "timestamp": case["timestamp"],
                                "image_relative_path": case[
                                    "image_relative_path"
                                ],
                                "image_width": case["image_width"],
                                "image_height": case["image_height"],
                                "no_foreign_objects": record[
                                    "no_foreign_objects"
                                ],
                                "boxes": record["boxes"],
                                "notes": record["notes"],
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )

            categories = [
                {"id": index, "name": name}
                for index, name in enumerate(CLASS_NAMES, start=1)
            ]
            category_ids = {item["name"]: item["id"] for item in categories}
            images = []
            annotations = []
            annotation_id = 1
            for image_id, (case, record) in enumerate(records, start=1):
                images.append(
                    {
                        "id": image_id,
                        "file_name": case["image_relative_path"],
                        "width": case["image_width"],
                        "height": case["image_height"],
                        "case_id": case["id"],
                        "dataset": case["dataset"],
                        "video": case["video"],
                        "timestamp": case["timestamp"],
                    }
                )
                for box in record["boxes"]:
                    annotations.append(
                        {
                            "id": annotation_id,
                            "image_id": image_id,
                            "category_id": category_ids[box["class_name"]],
                            "bbox": [
                                box["x"],
                                box["y"],
                                box["width"],
                                box["height"],
                            ],
                            "area": round(box["width"] * box["height"], 3),
                            "iscrowd": 0,
                            "difficult": box["difficult"],
                        }
                    )
                    annotation_id += 1
            _atomic_json(
                export_dir / "coco-complete.json",
                {
                    "info": {
                        "description": (
                            "FOCUS Frame human bounding-box annotations"
                        ),
                        "created_at": _utc_now(),
                        "pool_selection_version": self.pool[
                            "selection_version"
                        ],
                    },
                    "images": images,
                    "annotations": annotations,
                    "categories": categories,
                },
            )

            progress_path = export_dir / "progress.csv"
            with progress_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle, fieldnames=list(progress_rows[0].keys())
                )
                writer.writeheader()
                writer.writerows(progress_rows)

            manifest = {
                "created_at": _utc_now(),
                "pool_selection_version": self.pool["selection_version"],
                "pool_source_fingerprint": self.pool["source_fingerprint"],
                "complete_case_count": len(records),
                "complete_box_count": len(annotations),
                "files": [
                    "annotations-complete.jsonl",
                    "coco-complete.json",
                    "progress.csv",
                ],
            }
            _atomic_json(export_dir / "manifest.json", manifest)
            try:
                display_directory = str(export_dir.relative_to(REPO_ROOT))
            except ValueError:
                display_directory = str(export_dir)
            return {"directory": display_directory, **manifest}


class RevisionConflict(RuntimeError):
    def __init__(self, current: dict[str, Any]) -> None:
        super().__init__("annotation was changed in another browser tab")
        self.current = current


class Handler(BaseHTTPRequestHandler):
    server: "AnnotationServer"
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args: Any) -> None:
        sys.stderr.write(
            f"[{self.log_date_time_string()}] {self.address_string()} "
            f"{fmt % args}\n"
        )

    def _send_json(
        self, value: Any, status: HTTPStatus = HTTPStatus.OK
    ) -> None:
        body = json.dumps(value, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def _send_error_json(
        self,
        status: HTTPStatus,
        message: str,
        *,
        code: str | None = None,
        details: Any = None,
    ) -> None:
        payload: dict[str, Any] = {
            "error": message,
            "code": code or status.phrase.casefold().replace(" ", "_"),
        }
        if details is not None:
            payload["details"] = details
        self._send_json(payload, status)

    def _read_json(self) -> dict[str, Any]:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError as exc:
            raise AnnotationError("invalid Content-Length") from exc
        if length <= 0 or length > 2_000_000:
            raise AnnotationError("request body must be between 1 byte and 2 MB")
        body = self.rfile.read(length)
        try:
            value = json.loads(body)
        except json.JSONDecodeError as exc:
            raise AnnotationError("request body is not valid JSON") from exc
        if not isinstance(value, dict):
            raise AnnotationError("request body must be a JSON object")
        return value

    def _serve_file(
        self,
        path: Path,
        *,
        cache: str = "no-store",
        content_type: str | None = None,
    ) -> None:
        if not path.is_file():
            self._send_error_json(HTTPStatus.NOT_FOUND, "file not found")
            return
        data = path.read_bytes()
        guessed = content_type or mimetypes.guess_type(path.name)[0]
        self.send_response(HTTPStatus.OK)
        self.send_header(
            "Content-Type", guessed or "application/octet-stream"
        )
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", cache)
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = unquote(parsed.path)
        try:
            if path == "/api/meta":
                pool = self.server.store.pool
                self._send_json(
                    {
                        "name": "Frame Annotation Tool",
                        "pool": {
                            "selection_version": pool["selection_version"],
                            "generated_at": pool["generated_at"],
                            "selection_policy": pool["selection_policy"],
                            "summary": pool["summary"],
                        },
                        "classes": FO_GUIDE,
                        "guide_examples": pool.get("guide_examples", {}),
                    }
                )
                return
            if path == "/api/pool":
                statuses = self.server.store.status_map()
                cases = []
                for case in self.server.store.cases:
                    cases.append(
                        {
                            "id": case["id"],
                            "rank": case["rank"],
                            "batch": case["batch"],
                            "priority_score": case["priority_score"],
                            "priority_tier": case["priority_tier"],
                            "priority_reasons": case["priority_reasons"],
                            "dataset": case["dataset"],
                            "dataset_name": case["dataset_name"],
                            "video": case["video"],
                            "timestamp": case["timestamp"],
                            "procedure_type": case["procedure_type"],
                            "qa_count": len(case["questions"]),
                            **statuses[case["id"]],
                        }
                    )
                self._send_json({"cases": cases})
                return
            if path == "/api/progress":
                self._send_json(self.server.store.progress())
                return
            if path.startswith("/api/case/"):
                case_id = path.removeprefix("/api/case/")
                self._send_json(self.server.store.case_payload(case_id))
                return
            if path.startswith("/image/"):
                case_id = path.removeprefix("/image/")
                case = self.server.store.all_cases.get(case_id)
                if case is None:
                    raise KeyError(case_id)
                image_path = (
                    REPO_ROOT / case["image_relative_path"]
                ).resolve()
                image_path.relative_to(REPO_ROOT.resolve())
                self._serve_file(
                    image_path,
                    cache="private, max-age=3600",
                    content_type="image/jpeg",
                )
                return
            if path in {"/", "/index.html"}:
                self._serve_file(STATIC_ROOT / "index.html")
                return
            relative = path.removeprefix("/")
            if not relative or "/" in relative or relative.startswith("."):
                self._send_error_json(HTTPStatus.NOT_FOUND, "route not found")
                return
            self._serve_file(STATIC_ROOT / relative)
        except KeyError:
            self._send_error_json(HTTPStatus.NOT_FOUND, "unknown case")
        except Exception as exc:
            traceback.print_exc()
            self._send_error_json(
                HTTPStatus.INTERNAL_SERVER_ERROR, f"server error: {exc}"
            )

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = unquote(parsed.path)
        try:
            if path.startswith("/api/annotation/"):
                case_id = path.removeprefix("/api/annotation/")
                payload = self._read_json()
                if "expected_revision" not in payload:
                    raise AnnotationError("expected_revision is required")
                try:
                    expected_revision = int(payload["expected_revision"])
                except (TypeError, ValueError) as exc:
                    raise AnnotationError(
                        "expected_revision must be an integer"
                    ) from exc
                record, report = self.server.store.save(
                    case_id,
                    payload.get("annotation", {}),
                    expected_revision=expected_revision,
                    client=self.client_address[0],
                )
                self._send_json({"annotation": record, "validation": report})
                return
            if path == "/api/export":
                # Require a JSON request to reduce accidental GET-like exports.
                self._read_json()
                self._send_json(self.server.store.export(), HTTPStatus.CREATED)
                return
            self._send_error_json(HTTPStatus.NOT_FOUND, "route not found")
        except KeyError:
            self._send_error_json(HTTPStatus.NOT_FOUND, "unknown case")
        except RevisionConflict as exc:
            self._send_error_json(
                HTTPStatus.CONFLICT,
                str(exc),
                code="revision_conflict",
                details={"current": exc.current},
            )
        except AnnotationError as exc:
            status = (
                HTTPStatus.UNPROCESSABLE_ENTITY
                if exc.code == "completion_blocked"
                else HTTPStatus.BAD_REQUEST
            )
            self._send_error_json(
                status, str(exc), code=exc.code, details=exc.details
            )
        except Exception as exc:
            traceback.print_exc()
            self._send_error_json(
                HTTPStatus.INTERNAL_SERVER_ERROR, f"server error: {exc}"
            )


class AnnotationServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(
        self, address: tuple[str, int], store: AnnotationStore
    ) -> None:
        self.store = store
        super().__init__(address, Handler)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8770)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    args = parser.parse_args()
    store = AnnotationStore(args.data_root.resolve())
    server = AnnotationServer((args.host, args.port), store)
    print(
        f"Frame Annotation Tool: http://{args.host}:{args.port}\n"
        f"Pool: {len(store.cases)} ranked training frames\n"
        f"Records: {store.records_dir}",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping.", flush=True)
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
