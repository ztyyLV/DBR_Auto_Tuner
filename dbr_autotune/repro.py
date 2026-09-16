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


def sdk_info() -> tuple:
    """(distribution name, version) of whichever Dynamsoft bundle is in use."""
    try:
        from .sdk import DISTRIBUTION, sdk_version
    except ImportError as exc:                       # SDK not installed at all
        return "none", f"unavailable ({exc.__class__.__name__})"
    return DISTRIBUTION, sdk_version()


def environment_fingerprint() -> Dict[str, Any]:
    """Everything that can move the numbers between two machines."""
    import os
    distribution, version = sdk_info()
    return {
        "autotune": __version__,
        "sdk": distribution,
        "sdk_version": version,
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
    if {"sdk", "sdk_version"} & set(differences):
        notes.append("the SDK version differs, so read rates may differ too")
    if {"platform", "machine", "cpu_count"} & set(differences):
        notes.append("the hardware or OS differs, so timings are not comparable")
    if "autotune" in differences:
        notes.append("the tuner version differs, so the search space may have changed")
    return "; ".join(notes) or "environments differ in: " + ", ".join(differences)
