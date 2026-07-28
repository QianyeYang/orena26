"""Canonical FOCUS classes and annotation guidance."""

from __future__ import annotations

import re
from typing import Any

FO_GUIDE: tuple[dict[str, Any], ...] = (
    {
        "name": "Sponge",
        "color": "#e85d75",
        "description": (
            "Soft absorbent material used to soak up fluids. It is commonly white "
            "when fresh and can become red-brown when saturated with blood."
        ),
        "look_for": (
            "Folded or crumpled woven fabric; soft irregular edges; white, pink, "
            "red, or brown depending on blood saturation."
        ),
        "do_not_count": (
            "Loose tissue, fat, blood clots, glare, or mesh-like anatomy. Count "
            "separate visible sponge pieces as separate instances."
        ),
        "confusions": ("Mesh", "Absorbable Hemostatic Agent", "Specimen"),
    },
    {
        "name": "Clip",
        "color": "#ffb000",
        "description": (
            "Small metal or polymer device used to seal a vessel or duct. A Clip "
            "becomes a foreign object only after it is placed in the abdomen."
        ),
        "look_for": (
            "Small bright, dark, or coloured V/U-shaped pieces, often clustered "
            "along a vessel, duct, or staple line."
        ),
        "do_not_count": (
            "Do not count clips still loaded inside a clip applier. Box every "
            "placed clip separately, including adjacent clips."
        ),
        "confusions": ("Needle",),
    },
    {
        "name": "Specimen Bag",
        "color": "#00b3b8",
        "description": (
            "Sterile pouch used to collect and retrieve resected tissue or an "
            "organ from the body cavity."
        ),
        "look_for": (
            "Thin translucent, white, blue, or purple pouch; may be collapsed, "
            "open, folded, or filled with tissue."
        ),
        "do_not_count": (
            "The attached drawstring is part of the bag, not another object. A "
            "Specimen inside remains a separate Specimen annotation."
        ),
        "confusions": ("Specimen", "Sponge"),
    },
    {
        "name": "Silicone Loop",
        "color": "#6f7bf7",
        "description": (
            "Soft flexible band used to encircle and control a vessel or "
            "anatomical structure for isolation or traction."
        ),
        "look_for": (
            "Smooth narrow loop or band, usually white, that curves around tissue "
            "and may cross itself."
        ),
        "do_not_count": (
            "Do not label sutures, ordinary instrument cables, or a bag string as "
            "a Silicone Loop."
        ),
        "confusions": ("External Drain", "Needle"),
    },
    {
        "name": "External Drain",
        "color": "#3aa76d",
        "description": (
            "Clear or fluid-filled tube used to evacuate fluid from the surgical "
            "site to outside the body."
        ),
        "look_for": (
            "Smooth tubular structure, often transparent or pale, with a rounded "
            "tip or side holes; only part of it may be visible."
        ),
        "do_not_count": (
            "Do not label trocars, suction/irrigation instruments, or other rigid "
            "tools that remain connected to the external environment."
        ),
        "confusions": ("Silicone Loop",),
    },
    {
        "name": "Needle",
        "color": "#c77dff",
        "description": (
            "Sharp straight or curved metal needle used for placing sutures."
        ),
        "look_for": (
            "Thin reflective curved or straight metal body, frequently attached "
            "to a suture and partly held by a needle driver."
        ),
        "do_not_count": (
            "A visible suture thread alone is not a Needle. Label only when the "
            "metal needle itself is visible."
        ),
        "confusions": ("Clip", "Silicone Loop"),
    },
    {
        "name": "Gallstone",
        "color": "#d6c94f",
        "description": (
            "Roundish calcified concretion originating from the gallbladder, "
            "typically white, yellow, tan, brown, or green."
        ),
        "look_for": (
            "Discrete pebble-like round or faceted body. Several stones may be "
            "clustered together."
        ),
        "do_not_count": (
            "Do not label fat lobules, cautery debris, or tissue fragments. A "
            "stone inside a Specimen Bag is still a Gallstone; label both objects."
        ),
        "confusions": ("Specimen",),
    },
    {
        "name": "Specimen",
        "color": "#f27649",
        "description": (
            "Excised biological tissue or organ that must be retrieved after all "
            "connections to the patient's anatomy have been cut."
        ),
        "look_for": (
            "Discrete tissue mass separated from surrounding anatomy, possibly "
            "held by an instrument or contained in a bag."
        ),
        "do_not_count": (
            "Do not label attached anatomy, ordinary fat, or blood. A Specimen "
            "inside a Specimen Bag remains a separate annotation."
        ),
        "confusions": ("Specimen Bag", "Sponge", "Gallstone"),
    },
    {
        "name": "Mesh",
        "color": "#64b5f6",
        "description": (
            "Screen-like implantable patch used to reinforce a weak area of the "
            "body wall."
        ),
        "look_for": (
            "Broad flat sheet with a regular woven, porous, or grid texture. It "
            "may be folded during insertion and later spread over tissue."
        ),
        "do_not_count": (
            "A Sponge is cloth-like and absorbent; an absorbable hemostatic agent "
            "is usually frizzy or fleece-like and applied to a bleeding surface."
        ),
        "confusions": ("Sponge", "Absorbable Hemostatic Agent"),
        "external_reference": {
            "label": "Clinical laparoscopic mesh reference",
            "url": "https://www.surgicaloasis.com/is-it-safe-to-repair-hernia-during-gall-bladder-surgery/",
        },
    },
    {
        "name": "Absorbable Hemostatic Agent",
        "color": "#f2f2f2",
        "description": (
            "Resorbable material applied to a bleeding surface to promote "
            "clotting and intended to be absorbed."
        ),
        "look_for": (
            "White or pale-yellow frizzy fleece, fibrous pad, powder, or soft "
            "mesh-like material conforming directly to a tissue surface."
        ),
        "do_not_count": (
            "Distinguish it from a woven implantable Mesh and from a removable "
            "absorbent Sponge. If unsure, mark the box uncertain."
        ),
        "confusions": ("Sponge", "Mesh"),
        "external_reference": {
            "label": "Authoritative absorbable-hemostat reference",
            "url": (
                "https://www.jnjmedtech.com/en-US/products/surgery/biosurgery/"
                "surgicel-snow-absorbable-hemostat/"
            ),
        },
        "external_image": (
            "https://images.contentstack.io/v3/assets/blt6442fb89e58ceab5/"
            "blt819f9fbc4f73c33b/69cd66492c747b95ba576353/"
            "US_SRG_BIOS_126842.1_-_SURGICEL_SNoW_hero_1.png"
            "?format=webp&quality=85&width=1200"
        ),
    },
)

CLASS_NAMES: tuple[str, ...] = tuple(item["name"] for item in FO_GUIDE)
CLASS_BY_CASEFOLD = {name.casefold(): name for name in CLASS_NAMES}
CLASS_COLORS = {item["name"]: item["color"] for item in FO_GUIDE}

_ALIASES = {
    "sponge": "Sponge",
    "sponges": "Sponge",
    "clip": "Clip",
    "clips": "Clip",
    "specimen bag": "Specimen Bag",
    "specimen bags": "Specimen Bag",
    "silicone loop": "Silicone Loop",
    "silicone loops": "Silicone Loop",
    "external drain": "External Drain",
    "external drains": "External Drain",
    "needle": "Needle",
    "needles": "Needle",
    "gallstone": "Gallstone",
    "gallstones": "Gallstone",
    "specimen": "Specimen",
    "specimens": "Specimen",
    "mesh": "Mesh",
    "meshes": "Mesh",
    "absorbable hemostatic agent": "Absorbable Hemostatic Agent",
    "absorbable hemostatic agents": "Absorbable Hemostatic Agent",
}


def canonical_class(value: Any) -> str | None:
    """Return a canonical FOCUS name for a class-like scalar."""
    text = " ".join(str(value).strip().split()).casefold()
    return _ALIASES.get(text)


def classes_in_answer(value: Any) -> tuple[str, ...]:
    """Extract canonical class names from a comma-separated answer."""
    text = " ".join(str(value).strip().split())
    found: list[str] = []
    for part in text.split(","):
        canonical = canonical_class(part)
        if canonical and canonical not in found:
            found.append(canonical)
    if found:
        return tuple(found)

    lower = text.casefold()
    for name in sorted(CLASS_NAMES, key=len, reverse=True):
        if re.search(rf"(?<!\w){re.escape(name.casefold())}(?!\w)", lower):
            found.append(name)
    return tuple(found)
