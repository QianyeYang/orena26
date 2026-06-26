"""Shared, model-agnostic utilities for the ORena FOCUS challenge.

- :mod:`src.data`    — parquet -> focus Request/Reference (offline, parquet-direct)
- :mod:`src.frames`  — extract only referenced frames + resolve their paths
- :mod:`src.adapter` — constrain raw VLM text to format-parseable answers
- :mod:`src.paths`   — repo path resolution
"""

from .adapter import (
    MAX_TEXT_LEN,
    build_response,
    is_parseable,
    normalize_answer,
    parse_mc_options,
)
from .data import (
    load_split,
    read_parquet,
    row_to_reference,
    row_to_request,
    save_split,
)
from .frames import (
    BASE_FPS,
    extract_frames,
    extract_split,
    frame_index,
    needed_frames,
    request_frame_paths,
)

__all__ = [
    # data
    "read_parquet", "load_split", "save_split", "row_to_request", "row_to_reference",
    # frames
    "needed_frames", "extract_frames", "extract_split", "request_frame_paths",
    "frame_index", "BASE_FPS",
    # adapter
    "normalize_answer", "build_response", "parse_mc_options", "is_parseable", "MAX_TEXT_LEN",
]
