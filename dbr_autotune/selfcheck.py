"""Validate every level of the search space against the installed SDK.

Run with ``python -m dbr_dbr_autotune.selfcheck``. Each knob level is compiled into a
template and handed to ``init_settings``; anything the SDK rejects is reported.
Use it after an SDK upgrade to catch renamed or removed parameters.
"""
from __future__ import annotations

import sys

from . import space, template as T
from .config import resolve_license


def main() -> int:
    from .sdk import CaptureVisionRouter, LicenseManager

    code, message = LicenseManager.init_license(resolve_license(None))
    if code not in (0, -10077):
        print(f"license error ({code}): {message}")
        return 2
    router = CaptureVisionRouter()

    failures = []
    checked = 0

    def check(label: str, knobs: dict) -> None:
        nonlocal checked
        checked += 1
        ec, em = router.init_settings(T.to_json(T.normalize(knobs), "SelfCheck"))
        if ec not in (0, -10077):
            failures.append(f"{label}: ({ec}) {em}")
            print(f"  FAIL  {label}\n        ({ec}) {em}")

    print("seed configurations")
    for name, knobs in space.seed_configs(space.DISCOVERY_FORMATS):
        check(name, knobs)

    print("knob ladders")
    profile = {"median_short_edge": 1200}
    for knob, levels in space.ladders(profile):
        for level in levels:
            check(f"{knob}={level!r}", {knob: level})

    print("trim levels")
    for knob, value in space.TRIM_ORDER:
        check(f"trim {knob}={value!r}", {knob: value})

    print("discovery formats")
    for fmt in space.DISCOVERY_FORMATS:
        check(f"format {fmt}", {"formats": [fmt]})

    print(f"\n{checked} template(s) checked, {len(failures)} rejected")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
