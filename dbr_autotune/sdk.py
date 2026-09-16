"""The one place that imports the Dynamsoft SDK.

Dynamsoft ships the same Python API under two distribution names:

* ``dynamsoft-barcode-reader-bundle`` — barcode only, versioned to match the
  Barcode Reader itself (11.x). This is what the project depends on.
* ``dynamsoft-capture-vision-bundle`` — the full Capture Vision suite, which
  also does document normalisation and label recognition, versioned separately
  (3.x).

Both expose ``CaptureVisionRouter``, ``LicenseManager`` and the enums this tool
uses, so either satisfies it. Anyone who already has the larger bundle
installed should not be made to install a second copy of the same native
libraries, so this module takes whichever is present.

Import the SDK from here, never directly, so that rule has exactly one home.
"""
from __future__ import annotations

from types import ModuleType
from typing import List, Tuple

# Preferred first. Order matters only when both are installed.
_CANDIDATES: List[Tuple[str, str]] = [
    ("dynamsoft_barcode_reader_bundle", "dynamsoft-barcode-reader-bundle"),
    ("dynamsoft_capture_vision_bundle", "dynamsoft-capture-vision-bundle"),
]


class SDKNotInstalled(ImportError):
    """Neither Dynamsoft bundle could be imported."""


def _load() -> Tuple[ModuleType, str, str]:
    problems = []
    for module_name, distribution in _CANDIDATES:
        try:
            module = __import__(module_name)
        except ImportError as exc:
            problems.append(f"{distribution}: {exc}")
            continue
        return module, module_name, distribution
    raise SDKNotInstalled(
        "The Dynamsoft SDK is not installed. Install the barcode bundle:\n"
        "    pip install dynamsoft-barcode-reader-bundle\n"
        "(the full dynamsoft-capture-vision-bundle also works).\n"
        "Tried:\n  " + "\n  ".join(problems))


_module, MODULE_NAME, DISTRIBUTION = _load()


def sdk_version() -> str:
    """Version of whichever bundle is in use."""
    for attribute in ("__version__", "version", "VERSION"):
        value = getattr(_module, attribute, None)
        if isinstance(value, str) and value:
            return value
    # The bundles do not always carry a version attribute; the distribution
    # metadata is what pip actually installed, so fall back to that.
    try:
        from importlib.metadata import version
        return version(DISTRIBUTION)
    except Exception:
        return "unknown"


def __getattr__(name: str):
    """Re-export the SDK's public names: ``from .sdk import CaptureVisionRouter``."""
    try:
        return getattr(_module, name)
    except AttributeError:
        raise AttributeError(
            f"{name!r} is not provided by {DISTRIBUTION}. If this worked with "
            f"another Dynamsoft bundle, the two have diverged.") from None
