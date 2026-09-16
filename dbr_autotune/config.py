"""Static configuration: license, file types, tuning defaults."""
from __future__ import annotations

import os

# Public trial license shipped with the DBR samples. Network access required.
# Override with --license or the DBR_LICENSE environment variable.
DEFAULT_LICENSE = "DLS2eyJoYW5kc2hha2VDb2RlIjoiMjAwMDAxLTEwNTI2NzQwMSJ9"

SINGLE_PAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".jpe", ".jfif"}
MULTI_PAGE_EXT = {".pdf", ".tif", ".tiff"}
IMAGE_EXT = SINGLE_PAGE_EXT | MULTI_PAGE_EXT

# Formats whose symbology has weak or optional error detection. A result in one
# of these is only trusted as pseudo ground truth when several independent
# configurations agree on it.
RISKY_1D_FORMATS = {
    "BF_CODE_39", "BF_CODE_39_EXTENDED", "BF_ITF", "BF_CODABAR",
    "BF_CODE_11", "BF_MSI_CODE", "BF_INDUSTRIAL_25", "BF_CODE_93",
    "BF_PHARMACODE", "BF_PHARMACODE_ONE_TRACK", "BF_PHARMACODE_TWO_TRACK",
    "BF_PATCHCODE",
}

# Barcode formats the tuner is allowed to narrow down to, grouped for probing.
FORMAT_GROUPS = {
    "1d": ["BF_ONED"],
    "2d": ["BF_QR_CODE", "BF_DATAMATRIX", "BF_PDF417", "BF_AZTEC",
           "BF_MICRO_QR", "BF_MICRO_PDF417", "BF_MAXICODE", "BF_DOTCODE",
           "BF_GS1_DATABAR", "BF_GS1_COMPOSITE"],
    "postal": ["BF_POSTALCODE"],
}


def resolve_license(cli_value: str | None) -> str:
    return cli_value or os.environ.get("DBR_LICENSE") or DEFAULT_LICENSE


# ``BarcodeResultItem.get_format_string()`` returns the enum name without its
# ``BF_`` prefix ("DATAMATRIX", "PHARMACODE_ONE_TRACK"), while templates and the
# format enum use the prefixed spelling. Everything downstream - the truth set,
# symbology narrowing, the weak-checksum guard - compares against template
# spellings, so normalise at the boundary.
def _known_formats() -> frozenset:
    try:
        from .sdk import EnumBarcodeFormat
        return frozenset(n for n in dir(EnumBarcodeFormat) if n.startswith("BF_"))
    except Exception:
        return frozenset()


KNOWN_FORMATS = _known_formats()


def canonical_format(name: str) -> str:
    """Map an SDK result format string onto its ``BF_`` template spelling."""
    if not name:
        return "BF_NULL"
    if name.startswith("BF_"):
        return name
    prefixed = "BF_" + name
    if not KNOWN_FORMATS or prefixed in KNOWN_FORMATS:
        return prefixed
    return name


def is_weak_checksum(fmt: str) -> bool:
    """True for symbologies whose decode is not self-validating."""
    return canonical_format(fmt) in RISKY_1D_FORMATS
