"""Multi-view counting for FRAME: tiles, augmentations, and count-set routing.

Counting carries ~38% of the final score and is the model's worst capability:
exact 0.478 against within-±1 0.798, with predictions capping at 5 while ground
truth reaches 12. The failure is range compression — the model regresses to the
marginal instead of enumerating — so the fix is to stop asking for a single
gestalt number over the whole frame.

Three mechanisms, all of which fit inside the unspent latency budget (we use
15 s of a 220 s pooled allowance per 20-question batch):

**Tiling.** Split the frame into a 2×2 grid, upscale each tile back to roughly
the full-frame pixel budget, and ask for a count per tile under a *centre-in-
tile* rule so an object straddling a seam is counted once. Summing four counts
of 0–3 lands in the regime where the model scores 0.79–0.47, instead of asking
for one count of 8 in the regime where it scores 0.01. Tiles are upscaled
because the extracted frames are only 960×540 (HeiCo) / 720×576 (LapChole) —
already under the 602,112-pixel cap — so a raw tile would carry *fewer* tokens
than the whole frame, not more.

**Augmentation.** Horizontal and vertical flips and a centre zoom, aggregated
across views. Safe here because these views are only ever used for `number`
questions; they would destroy the quadrant semantics of spatial questions.

**Routing.** "How many different foreign object *classes*" is answered by
taking the size of the predicted class *set*, not by predicting a number:
measured 0.840 versus 0.752 on saved epoch-30 predictions, because |set| can be
right when the set itself is wrong. Class counts are unioned across views, never
summed — distinct classes do not add.

Views are stored per-view so combination rules can be swept offline against one
expensive inference pass. `combine_counts` is deliberately a pure function of a
per-view table.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
import statistics

from PIL import Image

from src.counting import CountMode, CountingQuestion
from src.prompts import FO_NAMES

#: Names used in prompts and stored in the per-view table.
TILE_NAMES: tuple[str, ...] = ("top-left", "top-right", "bottom-left", "bottom-right")

#: How a family of views is reduced to one count. Tiles are summed (each object
#: falls in exactly one tile under the centre rule); whole-frame variants are a
#: repeated measurement of the same quantity and so are voted on.
REDUCE_SUM = "sum"
REDUCE_VOTE = "vote"


@dataclass(frozen=True)
class View:
    """One rendered view of a frame, plus how it should be reduced."""

    view_id: str
    kind: str  # "whole" | "tile" | "augment"
    image: Image.Image
    reduce: str
    tile_name: str | None = None


def _resize_to_budget(img: Image.Image, target_pixels: int) -> Image.Image:
    """Scale *img* so its area is about *target_pixels*, preserving aspect ratio."""
    w, h = img.size
    if w * h <= 0:
        return img
    scale = (target_pixels / (w * h)) ** 0.5
    if scale <= 1.0:
        return img
    return img.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.LANCZOS)


def build_views(
    image_path: str,
    *,
    tiles: bool = True,
    augment: bool = True,
    target_pixels: int = 602112,
) -> list[View]:
    """Render the view set for one frame.

    The whole frame is always the first view, so a caller that budgets out can
    truncate the list and still hold a usable answer.
    """
    base = Image.open(image_path).convert("RGB")
    views: list[View] = [View("whole", "whole", base, REDUCE_VOTE)]

    if augment:
        views.append(View("hflip", "augment", base.transpose(Image.FLIP_LEFT_RIGHT), REDUCE_VOTE))
        views.append(View("vflip", "augment", base.transpose(Image.FLIP_TOP_BOTTOM), REDUCE_VOTE))
        w, h = base.size
        m = 0.1  # 10% centre zoom: drops frame edges, magnifies the surgical field
        zoom = base.crop((round(w * m), round(h * m), round(w * (1 - m)), round(h * (1 - m))))
        views.append(View("zoom", "augment", _resize_to_budget(zoom, target_pixels), REDUCE_VOTE))

    if tiles:
        w, h = base.size
        halves = ((0, 0), (w // 2, 0), (0, h // 2), (w // 2, h // 2))
        for name, (x, y) in zip(TILE_NAMES, halves):
            tile = base.crop((x, y, x + w // 2, y + h // 2))
            views.append(
                View(f"tile:{name}", "tile", _resize_to_budget(tile, target_pixels), REDUCE_SUM, name)
            )
    return views


def _target_phrase(cq: CountingQuestion) -> str:
    if cq.mode is CountMode.TARGET and cq.target:
        return f'instances of "{cq.target}"'
    return "foreign-object instances, of any class"


def build_view_instruction(question: str, cq: CountingQuestion, view: View) -> str:
    """Instruction for one view.

    For a tile the question is restated against the crop with the centre rule,
    because the original wording ("in this frame") would otherwise invite the
    model to reason about parts of the frame it cannot see.
    """
    if cq.mode is CountMode.CLASSES:
        names = ", ".join(FO_NAMES)
        scope = (
            f"This image is the {view.tile_name} quadrant of a surgical video frame. "
            "Name only classes with at least one instance whose centre lies inside "
            "this crop."
            if view.kind == "tile"
            else "Name every foreign-object class visible in this surgical video frame."
        )
        return (
            f"{scope}\n\nUse these names: {names}. Separate multiple names with a "
            "comma and a space, and name each class at most once. If none is "
            "visible, answer: none. Output only the name(s)."
        )

    if view.kind == "tile":
        return (
            f"This image is the {view.tile_name} quadrant of a surgical video frame.\n\n"
            f"Count the {_target_phrase(cq)} whose centre lies inside this crop. "
            "An object only partly visible at the crop edge counts only if its "
            "centre is inside.\n\nAnswer with a single non-negative integer "
            "(digits only, no words)."
        )
    return (
        f"{question}\n\nCount carefully, then answer with a single non-negative "
        "integer (digits only, no words)."
    )


def _mode_or_median(values: Sequence[int]) -> int:
    """Modal value, ties broken towards the median — never invents a new value."""
    if not values:
        return 0
    counts: dict[int, int] = {}
    for v in values:
        counts[v] = counts.get(v, 0) + 1
    best = max(counts.values())
    tied = sorted(v for v, c in counts.items() if c == best)
    if len(tied) == 1:
        return tied[0]
    med = statistics.median(values)
    return min(tied, key=lambda v: (abs(v - med), v))


#: Combination rules over one row's views. Swept offline against saved views;
#: `whole` is the current production behaviour and the control.
COMBINE_RULES: tuple[str, ...] = (
    "whole",          # ignore the extra views (baseline)
    "tiles",          # sum the four tiles
    "vote",           # vote over whole + augmentations
    "max_whole_tiles",  # tiles undercount less at high counts; whole is safer low
    "tiles_if_high",  # trust the tile sum only once it disagrees upward
)


def combine_counts(
    per_view: dict[str, int],
    rule: str = "max_whole_tiles",
    *,
    augment_ids: Iterable[str] = ("whole", "hflip", "vflip", "zoom"),
    high_threshold: int = 4,
) -> int:
    """Reduce one row's per-view counts to a single answer under *rule*."""
    if rule not in COMBINE_RULES:
        raise ValueError(f"unknown combine rule: {rule}")

    whole = per_view.get("whole")
    tile_vals = [per_view[f"tile:{n}"] for n in TILE_NAMES if f"tile:{n}" in per_view]
    tile_sum = sum(tile_vals) if len(tile_vals) == len(TILE_NAMES) else None
    votes = [per_view[v] for v in augment_ids if v in per_view]

    if rule == "whole":
        return whole if whole is not None else 0
    if rule == "tiles":
        return tile_sum if tile_sum is not None else (whole or 0)
    if rule == "vote":
        return _mode_or_median(votes) if votes else (whole or 0)
    if rule == "max_whole_tiles":
        base = _mode_or_median(votes) if votes else (whole or 0)
        return max(base, tile_sum) if tile_sum is not None else base
    # tiles_if_high
    base = _mode_or_median(votes) if votes else (whole or 0)
    if tile_sum is not None and tile_sum >= high_threshold and tile_sum > base:
        return tile_sum
    return base


def combine_class_sets(per_view: dict[str, frozenset[str]]) -> frozenset[str]:
    """Union class sets across views — distinct classes union, they never sum."""
    out: set[str] = set()
    for value in per_view.values():
        out |= set(value)
    out.discard("None")
    return frozenset(out)
