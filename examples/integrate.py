"""Using dbr-autotune from inside another Python project.

Run it:  python examples/integrate.py ./images

Three things an integrator normally wants, in order:

1. tune a folder and get the template back as a dict, without touching the CLI;
2. hand that template straight to the SDK and decode with it;
3. re-check a stored template later, as a regression gate.

Nothing here writes to stdout on your behalf or reads argv beyond this file --
the library is a library.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from dbr_autotune import TuneRequest, evaluate, tune


def step1_tune(folder: str) -> dict:
    """Tune on a folder and return the chosen template document."""
    request = TuneRequest(
        images=[folder],
        out=None,             # None = compute everything, write nothing to disk
        rounds=1,             # keep the example quick
        max_trials=40,
        # jobs=1 until the intermittent SDK crash under concurrency is fixed;
        # see "Known issue" in the README.
        jobs=1,
    )

    # `log` gets human-readable lines, `progress` gets structured events. Both
    # are optional -- pass neither and tune() is silent.
    def on_progress(event: dict) -> None:
        if event.get("event") == "trial":
            print(f"    [{event['index']:>3}/{event['total']}] {event['phase']:<8} "
                  f"coverage {event['page_coverage']:.0%}", flush=True)

    outcome = tune(request, progress=on_progress)

    score = outcome.final
    print(f"\n  page coverage : {score.page_coverage:.1%} "
          f"({score.pages_partial}/{score.pages_total} images produced a read)")
    print(f"  mean decode   : {score.mean_ms:.0f} ms")
    if score.pages_undetermined:
        print(f"  ! {score.pages_undetermined} image(s) were never read by anything. "
              f"If they do hold barcodes, coverage is the number to trust, not recall.")

    # A plain dict, ready for InitSettings in any edition of the SDK.
    return outcome.templates()["AutoTuned_MaxRecall"]


def step2_decode_with_it(template: dict, folder: str) -> None:
    """Use the tuned template directly, the way production code would."""
    from dbr_autotune.sdk import CaptureVisionRouter, LicenseManager
    from dbr_autotune.config import resolve_license

    LicenseManager.init_license(resolve_license(None))
    router = CaptureVisionRouter()
    name = template["CaptureVisionTemplates"][0]["Name"]
    code, message = router.init_settings(json.dumps(template))
    if code not in (0, -10077):
        raise RuntimeError(f"template rejected ({code}): {message}")

    for path in sorted(Path(folder).iterdir())[:5]:
        if path.is_dir():
            continue
        result = router.capture(str(path), name)
        items = result.get_items() or []
        text = items[0].get_text()[:28] if items else "-"
        print(f"    {path.name:<28} {len(items)} code(s)  {text}")


def step3_regression_gate(template_path: str, folder: str) -> bool:
    """Score a stored template against images, e.g. in CI after an SDK upgrade."""
    score = evaluate([folder], template_path, jobs=1)
    passed = score.page_coverage >= 0.80
    print(f"    coverage {score.page_coverage:.1%}  p95 {score.p95_ms:.0f} ms  "
          f"-> {'PASS' if passed else 'FAIL'}")
    return passed


def main() -> int:
    folder = sys.argv[1] if len(sys.argv) > 1 else "images"
    if not Path(folder).is_dir():
        print(f"usage: python examples/integrate.py <image folder>   "
              f"(got {folder!r})")
        return 2

    print("1. tuning")
    template = step1_tune(folder)

    print("\n2. decoding with the tuned template")
    step2_decode_with_it(template, folder)

    print("\n3. using it as a regression gate")
    out = Path("example_template.json")
    out.write_text(json.dumps(template, indent=2), encoding="utf-8")
    ok = step3_regression_gate(str(out), folder)
    out.unlink(missing_ok=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
