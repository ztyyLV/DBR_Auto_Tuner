"""The search space: seed configurations and per-knob candidate ladders.

Ladders are ordered cheapest-first where that makes sense, but the search tries
every level of a knob it touches, so ordering only affects tie-breaking.
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

# Every symbology the discovery probe is allowed to look for. BF_DEFAULT
# deliberately excludes the expensive long-tail formats (DotCode, Pharmacode,
# postal codes), so discovery has to ask for them by name.
DISCOVERY_FORMATS = [
    "BF_ONED", "BF_GS1_DATABAR", "BF_GS1_COMPOSITE", "BF_PATCHCODE",
    "BF_QR_CODE", "BF_MICRO_QR", "BF_DATAMATRIX", "BF_PDF417", "BF_MICRO_PDF417",
    "BF_AZTEC", "BF_MAXICODE", "BF_DOTCODE", "BF_POSTALCODE", "BF_PHARMACODE",
]

ALL_DEBLUR = [
    "DM_BASED_ON_LOC_BIN", "DM_THRESHOLD_BINARIZATION", "DM_DIRECT_BINARIZATION",
    "DM_SMOOTHING", "DM_MORPHING", "DM_DEEP_ANALYSIS", "DM_SHARPENING",
    "DM_GRAY_EQUALIZATION", "DM_SHARPENING_SMOOTHING", "DM_NEURAL_NETWORK",
]

ALL_LOCALIZATION = [
    "LM_CONNECTED_BLOCKS", "LM_LINES", "LM_STATISTICS", "LM_STATISTICS_MARKS",
    "LM_NEURAL_NETWORK", "LM_SCAN_DIRECTLY", "LM_CENTRE",
]


def seed_configs(formats: List[str]) -> List[Tuple[str, Dict[str, Any]]]:
    """Starting points, spanning fast/default/thorough and the awkward cases."""
    return [
        ("seed:speed-first", {
            "formats": formats, "expected_count": 1,
            "localization_modes": ["LM_SCAN_DIRECTLY", "LM_CONNECTED_BLOCKS"],
            "grayscale_transform": ["GTM_ORIGINAL"],
            "resist_deformation": False, "complement_barcode": False,
            "scale_barcode_image": False, "region_predetect": True,
            "scale_image": {"type": "ST_SCALE_DOWN", "edge": 1440},
            "timeout_ms": 5000,
        }),
        ("seed:default", {
            "formats": formats, "timeout_ms": 10000,
        }),
        ("seed:read-rate", {
            "formats": formats, "expected_count": 0,
            "localization_modes": ["LM_CONNECTED_BLOCKS", "LM_LINES", "LM_STATISTICS"],
            "grayscale_transform": ["GTM_ORIGINAL", "GTM_INVERTED"],
            "timeout_ms": 30000,
        }),
        ("seed:dpm", {
            "formats": formats, "expected_count": 0,
            "dpm_modes": ["DPMCRM_GENERAL"],
            "localization_modes": ["LM_STATISTICS_MARKS", "LM_CONNECTED_BLOCKS"],
            "grayscale_transform": ["GTM_ORIGINAL", "GTM_INVERTED"],
            "timeout_ms": 30000,
        }),
        ("seed:dotcode", {
            "formats": formats, "expected_count": 0,
            "localization_modes": ["LM_STATISTICS_MARKS"],
            "grayscale_transform": ["GTM_ORIGINAL", "GTM_INVERTED"],
            "binarization": {"block": 39, "fill": 0, "compensation": 10},
            "deblur_modes": ["DM_BASED_ON_LOC_BIN", "DM_THRESHOLD_BINARIZATION",
                             "DM_DEEP_ANALYSIS"],
            "timeout_ms": 30000,
        }),
        ("seed:deep", {
            "formats": formats, "expected_count": 0,
            "dpm_modes": ["DPMCRM_GENERAL"],
            "localization_modes": ALL_LOCALIZATION,
            "grayscale_transform": ["GTM_ORIGINAL", "GTM_INVERTED"],
            "deblur_modes": ALL_DEBLUR,
            "barcode_scale": {"module_size_threshold": 4, "target_module_size": 8},
            "timeout_ms": 20000,
        }),
        ("seed:deep-upscaled", {
            "formats": formats, "expected_count": 0,
            "dpm_modes": ["DPMCRM_GENERAL"],
            "localization_modes": ALL_LOCALIZATION,
            "grayscale_transform": ["GTM_ORIGINAL", "GTM_INVERTED"],
            "deblur_modes": ALL_DEBLUR,
            "grayscale_enhance": ["GEM_SHARPEN_SMOOTH"],
            "binarization": {"block": 31, "fill": 0, "compensation": 10},
            "region_predetect": False,
            "timeout_ms": 20000,
        }),
    ]


def ladders(profile: Dict[str, Any]) -> List[Tuple[str, List[Any]]]:
    """Ordered (knob, levels) pairs for coordinate ascent.

    ``profile`` carries what the probe learned - notably the median short edge,
    used to pick sensible image-scaling thresholds.
    """
    short_edge = profile.get("median_short_edge") or 1080
    down = sorted({v for v in (2400, 1800, 1440, 1080, 720) if v < short_edge})
    up = sorted({v for v in (1080, 1440, 1800) if v > short_edge})

    scale_levels: List[Any] = [None]
    scale_levels += [{"type": "ST_SCALE_DOWN", "edge": v} for v in reversed(down)]
    scale_levels += [{"type": "ST_SCALE_UP", "edge": v} for v in up]

    return [
        ("dpm_modes", [None, ["DPMCRM_GENERAL"], ["DPMCRM_AUTO"]]),
        ("grayscale_transform", [
            ["GTM_ORIGINAL"],
            ["GTM_ORIGINAL", "GTM_INVERTED"],
            ["GTM_INVERTED"],
        ]),
        ("localization_modes", [
            ["LM_CONNECTED_BLOCKS", "LM_LINES", "LM_STATISTICS"],
            ["LM_CONNECTED_BLOCKS"],
            ["LM_STATISTICS_MARKS"],
            ["LM_STATISTICS_MARKS", "LM_CONNECTED_BLOCKS"],
            ["LM_CONNECTED_BLOCKS", "LM_STATISTICS", "LM_STATISTICS_MARKS"],
            ["LM_NEURAL_NETWORK", "LM_CONNECTED_BLOCKS"],
            ["LM_SCAN_DIRECTLY", "LM_CONNECTED_BLOCKS"],
            ["LM_CENTRE", "LM_CONNECTED_BLOCKS"],
            ALL_LOCALIZATION,
        ]),
        ("binarization", [
            None,
            {"block": 0, "fill": 1, "compensation": 10},
            {"block": 15, "fill": 0, "compensation": 10},
            {"block": 23, "fill": 0, "compensation": 10},
            {"block": 31, "fill": 0, "compensation": 10},
            {"block": 39, "fill": 0, "compensation": 10},
            {"block": 55, "fill": 0, "compensation": 10},
            {"block": 31, "fill": 1, "compensation": 20},
            {"block": 39, "fill": 0, "compensation": -10},
        ]),
        ("deblur_modes", [
            None,
            ["DM_BASED_ON_LOC_BIN", "DM_THRESHOLD_BINARIZATION"],
            ["DM_BASED_ON_LOC_BIN", "DM_THRESHOLD_BINARIZATION", "DM_DEEP_ANALYSIS"],
            ["DM_BASED_ON_LOC_BIN", "DM_THRESHOLD_BINARIZATION", "DM_DEEP_ANALYSIS",
             "DM_NEURAL_NETWORK"],
            ["DM_DIRECT_BINARIZATION", "DM_THRESHOLD_BINARIZATION", "DM_SHARPENING"],
            ["DM_SHARPENING_SMOOTHING", "DM_MORPHING", "DM_DEEP_ANALYSIS"],
            ALL_DEBLUR,
        ]),
        ("grayscale_enhance", [
            None,
            ["GEM_GENERAL"],
            ["GEM_GRAY_SMOOTH"],
            ["GEM_SHARPEN_SMOOTH"],
            ["GEM_GRAY_EQUALIZE"],
            ["GEM_GRAY_SMOOTH", "GEM_SHARPEN_SMOOTH"],
        ]),
        ("scale_image", scale_levels),
        ("barcode_scale", [
            None,
            {"module_size_threshold": 4, "target_module_size": 6},
            {"module_size_threshold": 4, "target_module_size": 8},
            {"module_size_threshold": 2, "target_module_size": 6},
            {"module_size_threshold": 6, "target_module_size": 10},
        ]),
        ("colour_conversion", [
            None,
            {"r": 100, "g": 0, "b": 0},
            {"r": 0, "g": 100, "b": 0},
            {"r": 0, "g": 0, "b": 100},
            {"r": 0, "g": 50, "b": 50},
        ]),
        ("mirror_mode", [None, "MM_BOTH"]),
        ("texture_detection", [None, {"sensitivity": 5}, {"sensitivity": 9}]),
        ("region_predetect", [True, False]),
        ("text_detect", [False, True]),
    ]


# Knobs the speed-trim phase tries to switch off, cheapest win first.
TRIM_ORDER: List[Tuple[str, Any]] = [
    ("text_detect", False),
    ("texture_detection", None),
    ("colour_conversion", None),
    ("resist_deformation", False),
    ("complement_barcode", False),
    ("scale_barcode_image", False),
    ("grayscale_enhance", None),
    ("mirror_mode", None),
    ("binarization", None),
    ("region_predetect", False),
    ("dpm_modes", None),
]
