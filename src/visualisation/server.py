"""Stdlib HTTP server + JSON API for the ORena FOCUS prediction visualiser.

No external web framework — just ``http.server``. Loads selected result roots
once at startup, then serves a small single-page app plus a JSON API and frame
images. Designed to be viewed inside VSCode's built-in Simple Browser.

Usage (from repo root)::

    python -m src.visualisation.server --port 8765 --run-root <results>
"""

from __future__ import annotations

import argparse
import json
import logging
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from src import paths
from src.visualisation.capabilities import CAPABILITY_GROUPS, TAXONOMY_SOURCE
from src.visualisation.loader import Dataset, build_dataset

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"
_CONTENT_TYPES = {".html": "text/html", ".js": "text/javascript", ".css": "text/css"}

# Populated in main(); read-only after startup so threads can share it freely.
DATASET: Dataset | None = None
FRAME_ROOTS: tuple[Path, ...] = ()


# --------------------------------------------------------------------------- #
# API payload builders
# --------------------------------------------------------------------------- #
def _runs_payload() -> list[dict]:
    return [r.as_dict() for r in DATASET.runs]


def _capabilities_payload() -> dict:
    return {"source": TAXONOMY_SOURCE, "groups": CAPABILITY_GROUPS}


def _questions_payload(track: str, split: str, dataset: str) -> dict:
    """Compact per-question rows for the matrix view (filtering happens client-side)."""
    sd = DATASET.split(track, split, dataset)
    if sd is None:
        return {"runs": [], "questions": []}
    questions = []
    for qid in sd.order:
        info = sd.questions[qid]
        preds = {}
        for rid in sd.run_ids:
            p = sd.preds.get(rid, {}).get(qid)
            if p is not None:
                preds[rid] = {"prediction": p["prediction"], "correct": p["correct"]}
        questions.append(
            {
                "qID": qid,
                "question": info["question"],
                "gt": info["gt"],
                "answer_format": info["answer_format"],
                "primary_capability": info["primary_capability"],
                "secondaries": info["secondaries"],
                "ood": info["ood"],
                "clinical": info["clinical"],
                "video": info["video"],
                "has_image": qid in sd.image_path,
                "preds": preds,
            }
        )
    return {"runs": sd.run_ids, "questions": questions}


def _question_detail(track: str, split: str, dataset: str, qid: str) -> dict | None:
    sd = DATASET.split(track, split, dataset)
    if sd is None or qid not in sd.questions:
        return None
    info = sd.questions[qid]
    runs = []
    for rid in sd.run_ids:
        p = sd.preds.get(rid, {}).get(qid)
        if p is None:
            continue
        run = next((r for r in DATASET.runs if r.id == rid), None)
        runs.append(
            {
                "run_id": rid,
                "label": run.label if run else rid,
                "raw_output": p["raw_output"],
                "prediction": p["prediction"],
                "correct": p["correct"],
                "latency": p["latency"],
                "error": p["error"],
                "system_prompt": p.get("system_prompt"),
            }
        )
    return {
        "qID": qid,
        "question": info["question"],
        "prompt": info["prompt"],
        "gt": info["gt"],
        "answer_format": info["answer_format"],
        "primary_capability": info["primary_capability"],
        "secondaries": info["secondaries"],
        "ood": info["ood"],
        "clinical": info["clinical"],
        "video": info["video"],
        "ts_start": info["ts_start"],
        "ts_end": info["ts_end"],
        "has_image": qid in sd.image_path,
        "runs": runs,
    }


def _resolve_image(track: str, split: str, dataset: str, qid: str) -> Path | None:
    """Return a frame only when it is inside an approved dataset frame tree."""
    sd = DATASET.split(track, split, dataset)
    if sd is None:
        return None
    raw = sd.image_path.get(qid)
    if not raw:
        return None
    p = Path(raw).resolve()
    if not any(p.is_relative_to(root) for root in FRAME_ROOTS):
        logger.warning("rejected out-of-tree image path: %s", p)
        return None
    return p if p.exists() else None


# --------------------------------------------------------------------------- #
# HTTP handler
# --------------------------------------------------------------------------- #
class Handler(BaseHTTPRequestHandler):
    server_version = "OrenaViz/1.0"

    def log_message(self, fmt, *args):  # quieter logs
        logger.debug("%s - %s", self.address_string(), fmt % args)

    # -- response helpers --------------------------------------------------- #
    def _send(
        self,
        code: int,
        body: bytes,
        content_type: str,
        cache_control: str | None = None,
    ):
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        if cache_control:
            self.send_header("Cache-Control", cache_control)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _json(self, obj, code: int = 200):
        self._send(code, json.dumps(obj, default=str).encode("utf-8"), "application/json")

    def _static(self, name: str):
        path = (STATIC_DIR / name).resolve()
        try:
            path.relative_to(STATIC_DIR.resolve())
        except ValueError:
            return self._send(403, b"forbidden", "text/plain")
        if not path.exists():
            return self._send(404, b"not found", "text/plain")
        ct = _CONTENT_TYPES.get(path.suffix, "application/octet-stream")
        self._send(200, path.read_bytes(), ct, cache_control="no-store")

    # -- routing ------------------------------------------------------------ #
    def do_HEAD(self):
        self.do_GET()

    def do_GET(self):
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        route = u.path
        try:
            if route == "/" or route == "/index.html":
                return self._static("index.html")
            if route.startswith("/static/"):
                return self._static(route[len("/static/"):])
            if route == "/api/runs":
                return self._json(_runs_payload())
            if route == "/api/capabilities":
                return self._json(_capabilities_payload())
            if route == "/api/questions":
                return self._json(
                    _questions_payload(
                        q.get("track", ""),
                        q.get("split", ""),
                        q.get("dataset", ""),
                    )
                )
            if route.startswith("/api/question/"):
                qid = route[len("/api/question/"):]
                detail = _question_detail(
                    q.get("track", ""),
                    q.get("split", ""),
                    q.get("dataset", ""),
                    qid,
                )
                return self._json(detail) if detail else self._json({"error": "not found"}, 404)
            if route == "/img":
                img = _resolve_image(
                    q.get("track", ""),
                    q.get("split", ""),
                    q.get("dataset", ""),
                    q.get("qid", ""),
                )
                if img is None:
                    return self._send(404, b"no image", "text/plain")
                return self._send(200, img.read_bytes(), "image/jpeg")
            return self._send(404, b"not found", "text/plain")
        except BrokenPipeError:
            pass  # client (browser) closed the connection mid-response
        except Exception:  # noqa: BLE001
            logger.exception("error handling %s", self.path)
            try:
                self._json({"error": "internal"}, 500)
            except Exception:  # noqa: BLE001
                pass


# --------------------------------------------------------------------------- #
# Entrypoint
# --------------------------------------------------------------------------- #
def main() -> None:
    global DATASET, FRAME_ROOTS
    ap = argparse.ArgumentParser(description="ORena FOCUS prediction visualiser")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--repo-root", default=str(paths.REPO_ROOT))
    ap.add_argument(
        "--run-root",
        action="append",
        default=[],
        help="load only results under this repository-relative or absolute directory",
    )
    ap.add_argument("--include-smoke", action="store_true", help="include logs/smoke/* runs")
    a = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    repo_root = Path(a.repo_root).resolve()
    FRAME_ROOTS = tuple(
        paths.frames_root(dataset=dataset).resolve()
        for dataset in paths.DATASETS
    )
    if a.run_root:
        logger.info("loading selected result root(s): %s", ", ".join(a.run_root))
    else:
        logger.info("loading runs under %s ...", repo_root)
    DATASET = build_dataset(
        repo_root,
        include_smoke=a.include_smoke,
        run_roots=a.run_root or None,
    )
    by_split: dict = {}
    for r in DATASET.runs:
        by_split.setdefault((r.track, r.split, r.dataset), 0)
        by_split[(r.track, r.split, r.dataset)] += 1
    logger.info(
        "loaded %d runs across %d (track,split,dataset) groups",
        len(DATASET.runs),
        len(by_split),
    )
    for (tk, sp, dataset), n in sorted(by_split.items()):
        logger.info("  %s/%s/%s: %d run(s)", tk, sp, dataset, n)

    httpd = ThreadingHTTPServer((a.host, a.port), Handler)
    url = f"http://localhost:{a.port}"
    print(f"\n  ORena visualiser ready -> {url}")
    print("  VSCode: Cmd/Ctrl-Shift-P -> 'Simple Browser: Show' -> paste the URL")
    print("  (Ctrl-C to stop)\n")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n  stopping ...")
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()
