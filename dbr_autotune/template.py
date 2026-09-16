"""Knob vector -> DBR 11.x CaptureVision template JSON.

A *knob vector* is a plain dict. Every key is optional; ``DEFAULTS`` supplies the
rest. ``build()`` turns one into the template JSON that the SDK consumes, and
that is the only place in this project that knows the template schema.

Every key emitted here was validated against the installed SDK via
``init_settings`` - see ``selfcheck.py``.
"""
from __future__ import annotations

import copy
import hashlib
import json
from typing import Any, Dict

TEMPLATE_VERSION = "5.3"

DEFAULTS: Dict[str, Any] = {
    # --- task level -------------------------------------------------------
    "formats": ["BF_DEFAULT"],
    "expected_count": 0,            # 0 = unknown / read everything found
    "timeout_ms": 10000,
    "max_parallel_tasks": 0,        # 0 = let the SDK decide
    "max_threads_in_one_task": 0,
    "dpm_modes": None,              # e.g. ["DPMCRM_GENERAL"]
    "min_result_confidence": None,
    "min_barcode_text_length": None,
    # --- localization -----------------------------------------------------
    "localization_modes": ["LM_CONNECTED_BLOCKS", "LM_LINES", "LM_STATISTICS"],
    "region_predetect": True,
    # --- decoding ---------------------------------------------------------
    "deblur_modes": None,           # None = SDK default set
    "resist_deformation": True,
    "complement_barcode": True,
    "scale_barcode_image": True,
    "barcode_scale": None,          # {"module_size_threshold":4,"target_module_size":6}
    # --- image pipeline ---------------------------------------------------
    "colour_conversion": None,      # {"r":-1,"g":-1,"b":-1}
    "grayscale_transform": ["GTM_ORIGINAL"],
    "grayscale_enhance": None,      # list of GEM_* names
    "binarization": None,           # {"block":31,"compensation":10,"fill":1}
    "scale_image": None,            # {"type":"ST_SCALE_DOWN","edge":2000}
    "texture_detection": None,      # {"sensitivity":5}
    "text_detect": False,
    # --- per-format specification ----------------------------------------
    "mirror_mode": None,            # "MM_NORMAL" | "MM_BOTH"
    "module_size_range": None,      # {"min":1,"max":1000}
}

_ROI = "roi_autotuned"
_TASK = "task_autotuned"
_IP = "ip_autotuned"
_BFS = "bfs_autotuned"


def normalize(knobs: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Fill in defaults and return a fresh, complete knob vector."""
    out = copy.deepcopy(DEFAULTS)
    for key, value in (knobs or {}).items():
        if key not in DEFAULTS:
            raise KeyError(f"unknown knob {key!r}")
        out[key] = copy.deepcopy(value)
    return out


def with_knobs(knobs: Dict[str, Any], **changes: Any) -> Dict[str, Any]:
    """Return a copy of ``knobs`` with ``changes`` applied."""
    out = copy.deepcopy(knobs)
    for key, value in changes.items():
        if key not in DEFAULTS:
            raise KeyError(f"unknown knob {key!r}")
        out[key] = copy.deepcopy(value)
    return out


def _modes(names) -> list:
    return [{"Mode": name} for name in names]


def _image_parameter(k: Dict[str, Any]) -> Dict[str, Any]:
    stages: list = []

    if k["colour_conversion"]:
        cc = k["colour_conversion"]
        stages.append({
            "Stage": "SST_CONVERT_TO_GRAYSCALE",
            "ColourConversionModes": [{
                "Mode": "CICM_GENERAL",
                "RedChannelWeight": cc.get("r", -1),
                "GreenChannelWeight": cc.get("g", -1),
                "BlueChannelWeight": cc.get("b", -1),
            }],
        })

    if k["scale_image"]:
        si = k["scale_image"]
        stages.append({
            "Stage": "SST_SCALE_IMAGE",
            "ImageScaleSetting": {
                "ScaleType": si.get("type", "ST_SCALE_DOWN"),
                "ReferenceEdge": si.get("edge_ref", "RE_SHORTER_EDGE"),
                "EdgeLengthThreshold": si["edge"],
            },
        })

    if k["texture_detection"]:
        stages.append({
            "Stage": "SST_DETECT_TEXTURE",
            "TextureDetectionModes": [{
                "Mode": "TDM_GENERAL_WIDTH_CONCENTRATION",
                "Sensitivity": k["texture_detection"].get("sensitivity", 5),
            }],
        })

    if k["text_detect"]:
        stages.append({
            "Stage": "SST_DETECT_TEXT_ZONES",
            "TextDetectionMode": {"Mode": "TTDM_LINE", "Direction": "UNKNOWN", "Sensitivity": 3},
        })
        stages.append({"Stage": "SST_REMOVE_TEXT_ZONES_FROM_BINARY", "IfEraseTextZone": 1})

    if k["grayscale_transform"]:
        stages.append({
            "Stage": "SST_TRANSFORM_GRAYSCALE",
            "GrayscaleTransformationModes": _modes(k["grayscale_transform"]),
        })

    if k["grayscale_enhance"]:
        stages.append({
            "Stage": "SST_ENHANCE_GRAYSCALE",
            "GrayscaleEnhancementModes": _modes(k["grayscale_enhance"]),
        })

    if k["binarization"]:
        b = k["binarization"]
        mode = {
            "Mode": "BM_LOCAL_BLOCK",
            "BlockSizeX": b.get("block", 0),
            "BlockSizeY": b.get("block", 0),
            "EnableFillBinaryVacancy": b.get("fill", 1),
            "ThresholdCompensation": b.get("compensation", 10),
        }
        # Apply the same binarization to both binary surfaces, otherwise the
        # texture-removed path silently keeps the SDK defaults.
        stages.append({"Stage": "SST_BINARIZE_IMAGE", "BinarizationModes": [mode]})
        stages.append({"Stage": "SST_BINARIZE_TEXTURE_REMOVED_GRAYSCALE",
                       "BinarizationModes": [copy.deepcopy(mode)]})

    return {"Name": _IP, "ApplicableStages": stages}


def _task(k: Dict[str, Any]) -> Dict[str, Any]:
    predetect_stages = [{"Stage": "SST_PREDETECT_REGIONS"}] if k["region_predetect"] else []

    localize_stage: Dict[str, Any] = {"Stage": "SST_LOCALIZE_CANDIDATE_BARCODES"}
    if k["localization_modes"]:
        localize_stage["LocalizationModes"] = _modes(k["localization_modes"])

    decode_stages: list = []
    if k["resist_deformation"]:
        decode_stages.append({"Stage": "SST_RESIST_DEFORMATION"})
    if k["complement_barcode"]:
        decode_stages.append({"Stage": "SST_COMPLEMENT_BARCODE"})
    if k["scale_barcode_image"]:
        stage: Dict[str, Any] = {"Stage": "SST_SCALE_BARCODE_IMAGE"}
        if k["barcode_scale"]:
            bs = k["barcode_scale"]
            stage["BarcodeScaleModes"] = [{
                "Mode": "BSM_LINEAR_INTERPOLATION",
                "ModuleSizeThreshold": bs.get("module_size_threshold", 4),
                "TargetModuleSize": bs.get("target_module_size", 6),
            }]
        decode_stages.append(stage)
    decode_final: Dict[str, Any] = {"Stage": "SST_DECODE_BARCODES"}
    if k["deblur_modes"]:
        decode_final["DeblurModes"] = _modes(k["deblur_modes"])
    decode_stages.append(decode_final)

    task: Dict[str, Any] = {
        "Name": _TASK,
        "ExpectedBarcodesCount": k["expected_count"],
        "BarcodeFormatIds": list(k["formats"]),
        "MaxThreadsInOneTask": k["max_threads_in_one_task"],
        "SectionArray": [
            {"Section": "ST_REGION_PREDETECTION", "ImageParameterName": _IP,
             "StageArray": predetect_stages},
            {"Section": "ST_BARCODE_LOCALIZATION", "ImageParameterName": _IP,
             "StageArray": [localize_stage, {"Stage": "SST_LOCALIZE_BARCODES"}]},
            {"Section": "ST_BARCODE_DECODING", "ImageParameterName": _IP,
             "StageArray": decode_stages},
        ],
    }
    if k["dpm_modes"]:
        task["DPMCodeReadingModes"] = _modes(k["dpm_modes"])
    if k["min_result_confidence"] is not None:
        task["MinResultConfidence"] = k["min_result_confidence"]
    if k["min_barcode_text_length"] is not None:
        task["MinBarcodeTextLength"] = k["min_barcode_text_length"]
    return task


def _format_spec(k: Dict[str, Any]) -> Dict[str, Any] | None:
    spec: Dict[str, Any] = {"Name": _BFS, "BarcodeFormatIds": list(k["formats"])}
    touched = False
    if k["mirror_mode"]:
        spec["MirrorMode"] = k["mirror_mode"]
        touched = True
    if k["module_size_range"]:
        r = k["module_size_range"]
        spec["ModuleSizeRangeArray"] = [{"MinValue": r["min"], "MaxValue": r["max"]}]
        touched = True
    return spec if touched else None


def build(knobs: Dict[str, Any], name: str = "AutoTuned") -> Dict[str, Any]:
    """Build a complete, standalone template document from a knob vector."""
    k = normalize(knobs)
    task = _task(k)
    doc: Dict[str, Any] = {
        "Version": TEMPLATE_VERSION,
        "CaptureVisionTemplates": [{
            "Name": name,
            "ImageROIProcessingNameArray": [_ROI],
            "MaxParallelTasks": k["max_parallel_tasks"],
            "Timeout": k["timeout_ms"],
        }],
        "TargetROIDefOptions": [{"Name": _ROI, "TaskSettingNameArray": [_TASK]}],
        "BarcodeReaderTaskSettingOptions": [task],
        "ImageParameterOptions": [_image_parameter(k)],
    }
    spec = _format_spec(k)
    if spec:
        doc["BarcodeFormatSpecificationOptions"] = [spec]
        task["BarcodeFormatSpecificationNameArray"] = [_BFS]
    return doc


def to_json(knobs: Dict[str, Any], name: str = "AutoTuned", indent: int | None = None) -> str:
    return json.dumps(build(knobs, name), indent=indent, ensure_ascii=False)


def fingerprint(knobs: Dict[str, Any]) -> str:
    """Stable hash of the *effective* template, used to memoise trial runs."""
    payload = json.dumps(build(knobs, "fp"), sort_keys=True, ensure_ascii=False)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def describe(knobs: Dict[str, Any]) -> Dict[str, Any]:
    """Knob vector minus everything left at its default - i.e. what was tuned."""
    k = normalize(knobs)
    return {key: value for key, value in k.items() if value != DEFAULTS[key]}


def build_multi(variants: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Merge several knob vectors into one document with one CVT per variant.

    Internal option names are suffixed per variant so a single file can be
    loaded once and the templates selected by name at capture time.
    """
    doc: Dict[str, Any] = {
        "Version": TEMPLATE_VERSION,
        "CaptureVisionTemplates": [],
        "TargetROIDefOptions": [],
        "BarcodeReaderTaskSettingOptions": [],
        "ImageParameterOptions": [],
    }
    specs: list = []
    for index, (name, knobs) in enumerate(variants.items()):
        part = build(knobs, name)
        suffix = f"_{index}"
        rename = {_ROI: _ROI + suffix, _TASK: _TASK + suffix,
                  _IP: _IP + suffix, _BFS: _BFS + suffix}

        def fix(value):
            if isinstance(value, str):
                return rename.get(value, value)
            if isinstance(value, list):
                return [fix(v) for v in value]
            if isinstance(value, dict):
                return {k: fix(v) for k, v in value.items()}
            return value

        part = fix(part)
        doc["CaptureVisionTemplates"].extend(part["CaptureVisionTemplates"])
        doc["TargetROIDefOptions"].extend(part["TargetROIDefOptions"])
        doc["BarcodeReaderTaskSettingOptions"].extend(part["BarcodeReaderTaskSettingOptions"])
        doc["ImageParameterOptions"].extend(part["ImageParameterOptions"])
        specs.extend(part.get("BarcodeFormatSpecificationOptions", []))
    if specs:
        doc["BarcodeFormatSpecificationOptions"] = specs
    return doc
