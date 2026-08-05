from pathlib import Path
import re
from types import SimpleNamespace

import pytest

from src import paths
from src.visualisation import loader
from src.visualisation import server
from src.visualisation.loader import Dataset, SplitData


@pytest.mark.parametrize(
    ("header", "size", "expected"),
    [
        (None, 100, None),
        ("bytes=0-9", 100, (0, 9)),
        ("bytes=90-", 100, (90, 99)),
        ("bytes=-10", 100, (90, 99)),
        ("bytes=95-200", 100, (95, 99)),
        ("bytes=-200", 100, (0, 99)),
    ],
)
def test_parse_byte_range(header, size, expected):
    assert server._parse_byte_range(header, size) == expected


@pytest.mark.parametrize(
    "header",
    [
        "items=0-9",
        "bytes=",
        "bytes=100-",
        "bytes=20-10",
        "bytes=-0",
        "bytes=0-1,4-5",
    ],
)
def test_parse_byte_range_rejects_invalid_or_unsatisfiable_ranges(header):
    with pytest.raises(ValueError):
        server._parse_byte_range(header, 100)


def test_dataset_is_inferred_from_result_path():
    assert loader._infer_dataset(Path("track-procedure/logs/eval_epoch_8/heico")) == "heico"
    assert (
        loader._infer_dataset(Path("track-procedure/logs/eval_epoch_8/lapchole"))
        == "lapchole"
    )
    assert loader._infer_dataset(Path("track-procedure/logs/eval_epoch_8")) is None


@pytest.mark.parametrize("track", ["segment", "procedure"])
def test_video_resolution_stays_inside_dataset_tree(track, tmp_path, monkeypatch):
    focus_root = tmp_path / "focus"
    video_root = focus_root / "heico" / "videos"
    video_root.mkdir(parents=True)
    video = video_root / "case.mp4"
    video.write_bytes(b"video")

    split = SplitData(track=track, split="test", dataset="heico")
    split.questions = {
        "safe": {"video": "case.mp4"},
        "escape": {"video": "../../../outside.mp4"},
    }
    dataset = Dataset(splits={(track, "test", "heico"): split})
    monkeypatch.setattr(paths, "FOCUS_ROOT", focus_root)
    monkeypatch.setattr(server, "DATASET", dataset)

    assert server._resolve_video(track, "test", "heico", "safe") == video
    assert server._resolve_video(track, "test", "heico", "escape") is None
    assert server._resolve_video("frame", "test", "heico", "safe") is None


@pytest.mark.parametrize("track", ["segment", "procedure"])
def test_video_prefers_browser_proxy(track, tmp_path, monkeypatch):
    focus_root = tmp_path / "focus"
    video_root = focus_root / "heico" / "videos"
    proxy_root = focus_root / "heico" / "video-proxies-480p"
    video_root.mkdir(parents=True)
    proxy_root.mkdir(parents=True)
    source = video_root / "case.avi"
    proxy = proxy_root / "case.mp4"
    source.write_bytes(b"source")
    proxy.write_bytes(b"proxy")

    split = SplitData(track=track, split="test", dataset="heico")
    split.questions = {"safe": {"video": "case.avi"}}
    dataset = Dataset(splits={(track, "test", "heico"): split})
    monkeypatch.setattr(paths, "FOCUS_ROOT", focus_root)
    monkeypatch.setattr(server, "DATASET", dataset)

    assert server._resolve_video(track, "test", "heico", "safe") == proxy


def test_video_element_has_no_initial_source():
    html = (Path(server.__file__).parent / "static" / "index.html").read_text()
    match = re.search(r"<video\s+id=\"trackVideo\"[^>]*>", html)
    assert match is not None
    assert 'preload="none"' in match.group(0)
    assert not re.search(r"\ssrc=", match.group(0))


def test_unified_ui_exposes_every_track_module():
    html = (Path(server.__file__).parent / "static" / "index.html").read_text()
    for track in paths.TRACKS:
        assert f'data-track="{track}"' in html


def test_config_reports_available_tracks_and_valid_initial_track(monkeypatch):
    dataset = SimpleNamespace(
        runs=[SimpleNamespace(track="segment"), SimpleNamespace(track="procedure")]
    )
    monkeypatch.setattr(server, "DATASET", dataset)
    monkeypatch.setattr(server, "INITIAL_TRACK", "frame")

    assert server._config_payload() == {
        "initial_track": "segment",
        "available_tracks": ["segment", "procedure"],
    }


def test_server_can_fall_forward_when_requested_port_is_occupied():
    occupied = server.ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
    requested_port = occupied.server_address[1]
    replacement = None
    try:
        replacement = server._bind_http_server(
            "127.0.0.1",
            requested_port,
            auto_port=True,
        )
        assert replacement.server_address[1] > requested_port
    finally:
        occupied.server_close()
        if replacement is not None:
            replacement.server_close()


def test_server_preserves_address_in_use_error_without_auto_port():
    occupied = server.ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
    try:
        with pytest.raises(OSError, match="Address already in use"):
            server._bind_http_server(
                "127.0.0.1",
                occupied.server_address[1],
                auto_port=False,
            )
    finally:
        occupied.server_close()
