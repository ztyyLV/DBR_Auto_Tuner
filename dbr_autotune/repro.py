"""Reproducibility: what it takes for someone else to get your numbers.

A tuning result is only worth comparing if the other side can tell what produced
it. Every ``results.json`` carries an environment fingerprint and the full
request, so a run can be repeated with one command:

    dbr-autotune replay out/results.json
"""
from __future__ import annotations

import platform
import sys
from typing import Any, Dict

from . import __version__


def sdk_version() -> str:
    """Version of the installed Dynamsoft bundle, or why it could not be read."""
    try:
        import dynamsoft_capture_vision_bundle as bundle
    except Exception as exc:                                  # not installed
        return f"unavailable ({type(exc).__name__})"
    for attr in ("__version__", "version", "VERSION"):
        value = getattr(bundle, attr, None)
        if isinstance(value, str) and value:
            return value
    # The package does not always expose a version attribute; fall back to the
    # distribution metadata, which is what pip actually installed.
    try:
        from importlib.metadata import version
        return version("dynamsoft-capture-vision-bundle")
    except Exception:
        return "unknown"


def environment_fingerprint() -> Dict[str, Any]:
    """Everything that can move the numbers between two machines."""
    import os
    return {
        "autotune": __version__,
        "dbr_bundle": sdk_version(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
    }


def compare_environments(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    """Field-by-field differences between two fingerprints.

    Timing differences are expected whenever ``platform``, ``machine`` or
    ``cpu_count`` differ; a differing ``dbr_bundle`` can move read rates too.
    """
    keys = sorted(set(a) | set(b))
    return {key: {"a": a.get(key), "b": b.get(key)}
            for key in keys if a.get(key) != b.get(key)}


def describe_drift(differences: Dict[str, Any]) -> str:
    if not differences:
        return "environments match"
    notes = []
    if "dbr_bundle" in differences:
        notes.append("the SDK version differs, so read rates may differ too")
    if {"platform", "machine", "cpu_count"} & set(differences):
        notes.append("the hardware or OS differs, so timings are not comparable")
    if "autotune" in differences:
        notes.append("the tuner version differs, so the search space may have changed")
    return "; ".join(notes) or "environments differ in: " + ", ".join(differences)
