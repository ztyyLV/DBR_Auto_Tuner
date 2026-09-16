"""Command line front end.

    dbr-autotune ui                       open the local web UI
    dbr-autotune tune ./images -o out     tune once (the default verb)
    dbr-autotune cv ./images -k 5         K-fold cross-validation
    dbr-autotune evaluate t.json ./images score an existing template
    dbr-autotune replay out/results.json  repeat a recorded run
    dbr-autotune selfcheck                validate the search space against the SDK

``dbr-autotune ./images`` still works: an unrecognised first argument is taken
as the start of a ``tune``.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional

from . import __version__, api, template as T
from .api import TuneRequest, tune
from .scoring import Score
from .repro import compare_environments, describe_drift, environment_fingerprint

VERBS = {"ui", "tune", "cv", "evaluate", "replay", "selfcheck", "version"}


# --- shared flags ---------------------------------------------------------

def _add_tuning_flags(p: argparse.ArgumentParser, with_images: bool = True) -> None:
    if with_images:
        p.add_argument("images", nargs="+",
                       help="image files or folders (jpg/png/bmp/pdf/tif...)")
    p.add_argument("-o", "--out", default="autotune_out", help="output directory")
    p.add_argument("--license", default=None,
                   help="DBR license key (or set DBR_LICENSE); defaults to the public trial key")
    p.add_argument("--jobs", type=int, default=0,
                   help="parallel decode workers during search (default: half the cores)")
    p.add_argument("--no-recursive", action="store_true", help="do not descend into subfolders")
    p.add_argument("--limit", type=int, default=None,
                   help="evenly subsample the dataset to at most N pages")
    p.add_argument("--seed", type=int, default=0,
                   help="seed for the holdout / fold split (default 0, so splits are reproducible)")

    g = p.add_argument_group("objective")
    g.add_argument("--speed-weight", type=float, default=0.0,
                   help="0 = recall first, speed only breaks ties (default). Above 0, "
                        "each halving of decode time is worth SPEED_WEIGHT*10 recall points")
    g.add_argument("--recall-tolerance", type=float, default=0.0,
                   help="recall fraction that may be sacrificed for a faster template")

    g = p.add_argument_group("budget")
    g.add_argument("--max-trials", type=int, default=200, help="maximum configurations to evaluate")
    g.add_argument("--time-budget", type=float, default=0.0, metavar="SECONDS",
                   help="stop searching after this many seconds (0 = unlimited)")
    g.add_argument("--rounds", type=int, default=2, help="coordinate-ascent sweeps (default 2)")
    g.add_argument("--quick", action="store_true", help="shorthand for --rounds 1 --max-trials 60")
    g.add_argument("--skip-trim", action="store_true", help="skip the speed-trimming phase")

    g = p.add_argument_group("priors")
    g.add_argument("--formats", nargs="+", default=None, metavar="BF_...",
                   help="restrict to these symbologies instead of discovering them")
    g.add_argument("--expected-count", type=int, default=None,
                   help="barcodes per page, if known (0 = unknown)")
    g.add_argument("--ground-truth", default=None, metavar="FILE",
                   help="CSV (file,text[,format]) or JSON with the expected codes")
    g.add_argument("--freeze-truth", action="store_true",
                   help="do not add later discoveries to the inferred ground truth")
    g.add_argument("--min-agree", type=int, default=2, metavar="N",
                   help="configurations that must agree before a weak-checksum 1D read "
                        "(Code 39, ITF, Pharmacode...) is believed (default 2)")
    g.add_argument("--min-weak-length", type=int, default=4, metavar="N",
                   help="shortest payload believed in a weak-checksum symbology (default 4)")

    p.add_argument("--no-report", action="store_true", help="skip the HTML report")
    p.add_argument("--quiet", action="store_true", help="only print the summary")


def _request(args: argparse.Namespace, holdout: float = 0.0) -> TuneRequest:
    rounds, max_trials = args.rounds, args.max_trials
    if args.quick:
        rounds, max_trials = 1, min(max_trials, 60)
    return TuneRequest(
        images=args.images, out=args.out, license=args.license, jobs=args.jobs,
        recursive=not args.no_recursive, limit=args.limit, holdout=holdout,
        seed=args.seed,
        speed_weight=args.speed_weight, recall_tolerance=args.recall_tolerance,
        max_trials=max_trials, time_budget_s=args.time_budget, rounds=rounds,
        skip_trim=args.skip_trim, formats=args.formats,
        expected_count=args.expected_count, ground_truth=args.ground_truth,
        freeze_truth=args.freeze_truth, min_agree=args.min_agree,
        min_weak_length=args.min_weak_length, write_report=not args.no_report,
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="dbr-autotune",
        description="Derive a high-recall, low-latency Dynamsoft Barcode Reader 11.x "
                    "template from a folder of images.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""examples:
  dbr-autotune ui
  dbr-autotune ./images -o out
  dbr-autotune cv ./images -k 5 --quick
  dbr-autotune evaluate out/AutoTuned_MaxRecall.json ./more-images
  dbr-autotune replay out/results.json
""")
    p.add_argument("--version", action="version", version=f"dbr-autotune {__version__}")
    subs = p.add_subparsers(dest="verb")

    ui = subs.add_parser("ui", help="open the local web UI")
    ui.add_argument("--port", type=int, default=8765)
    ui.add_argument("--host", default="127.0.0.1", help="loopback only, by design")
    ui.add_argument("--no-browser", action="store_true")

    t = subs.add_parser("tune", help="tune once (the default verb)")
    _add_tuning_flags(t)
    t.add_argument("--holdout", type=float, default=0.0, metavar="FRACTION",
                   help="hold out this fraction of the files and report on them separately")

    cv = subs.add_parser("cv", help="K-fold cross-validation")
    _add_tuning_flags(cv)
    cv.add_argument("-k", "--folds", type=int, default=5, help="number of folds (default 5)")

    ev = subs.add_parser("evaluate", help="score an existing template on a folder")
    ev.add_argument("template", help="template JSON file")
    ev.add_argument("images", nargs="+")
    ev.add_argument("--license", default=None)
    ev.add_argument("--ground-truth", default=None)
    ev.add_argument("--json", action="store_true", help="print machine-readable output")
    g = ev.add_argument_group(
        "regression gates - any breach exits 1, so this works as a CI check")
    g.add_argument("--min-coverage", type=float, default=None, metavar="FRACTION",
                   help="fail if fewer than this fraction of pages produced a read "
                        "(e.g. 0.95)")
    g.add_argument("--min-recall", type=float, default=None, metavar="FRACTION",
                   help="fail if recall falls below this; only meaningful together "
                        "with --ground-truth")
    g.add_argument("--max-mean-ms", type=float, default=None, metavar="MS",
                   help="fail if the mean decode time exceeds this")
    g.add_argument("--max-p95-ms", type=float, default=None, metavar="MS",
                   help="fail if the p95 decode time exceeds this")

    rp = subs.add_parser("replay", help="repeat a run recorded in results.json")
    rp.add_argument("results", help="a results.json written by an earlier run")
    rp.add_argument("-o", "--out", default=None, help="override the output directory")
    rp.add_argument("--images", nargs="+", default=None,
                    help="override the image paths (they may live elsewhere on this machine)")
    rp.add_argument("--license", default=None)

    subs.add_parser("selfcheck", help="validate every search-space level against the SDK")
    return p


# --- verbs ----------------------------------------------------------------

def _print_tune_summary(outcome: api.TuneOutcome) -> None:
    s, final = outcome.summary, outcome.final
    base = outcome.scores.get("baseline")
    print("\n" + "=" * 72)
    truth = s["ground_truth"]
    print(f"  ground truth      {truth['total']} codes "
          f"({s['request']['ground_truth'] or 'inferred'})")
    if base:
        print(f"  SDK default       {base.recall:6.1%} recall   {base.mean_ms:8.1f} ms/page")
    print(f"  auto-tuned        {final.recall:6.1%} recall   {final.mean_ms:8.1f} ms/page"
          f"   (p95 {final.p95_ms:.0f} ms)")
    print(f"  page coverage     {final.page_coverage:6.1%}          "
          f"{final.pages_partial}/{final.pages_total} pages produced a read")
    if base and base.mean_ms and final.mean_ms:
        ratio = base.mean_ms / final.mean_ms
        print(f"  change            {(final.recall - base.recall) * 100:+.1f} recall pts, "
              f"{ratio:.2f}x {'faster' if ratio >= 1 else 'slower'}")
    if s["holdout"]:
        print(f"  holdout           {s['holdout']['page_coverage']:6.1%} coverage on "
              f"{s['holdout']['pages']} unseen page(s)")
    print(f"  evaluated         {len(s['trials'])} configurations in {s['elapsed_s']:,.0f}s")

    if s["undetermined_pages"] and not s["request"]["ground_truth"]:
        print("-" * 72)
        print(f"  ! {len(s['undetermined_pages'])} page(s) were never decoded by ANY configuration, so the")
        print("    inferred truth set records them as carrying no barcode. If they do")
        print(f"    carry one, the real recall is at most {final.recall_floor:.1%}, not {final.recall:.1%} -")
        print("    the denominator above only counts codes something managed to read.")
        print("    Check these pages by eye; supply --ground-truth to score them properly:")
        for label in s["undetermined_pages"]:
            print(f"      {label}")
    print("-" * 72)
    for key, path in outcome.paths.items():
        print(f"  {key:<12} -> {path}")
    print("=" * 72)


def cmd_tune(args: argparse.Namespace) -> int:
    log = (lambda *_a: None) if args.quiet else print
    outcome = tune(_request(args, holdout=args.holdout), log=log)
    _print_tune_summary(outcome)
    return 0


def cmd_cv(args: argparse.Namespace) -> int:
    from .crossval import cross_validate
    log = (lambda *_a: None) if args.quiet else print
    outcome = cross_validate(_request(args), k=args.folds, log=log)

    path = ""
    if args.out:
        os.makedirs(args.out, exist_ok=True)
        path = os.path.join(args.out, "cross-validation.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(outcome.to_json(), handle, indent=2, ensure_ascii=False)

    print("\n" + "=" * 72)
    for line in outcome.summary_lines():
        print(line)
    print("-" * 72)
    print("  Every fold was validated on files it never saw. The spread is the number")
    print("  that matters: a wide one means the result depends on which images you")
    print("  happened to tune on, and more images would help more than more tuning.")
    if path:
        print(f"  details      -> {path}")
    print("=" * 72)
    return 0


def _gate_failures(args: argparse.Namespace, score: Score) -> List[str]:
    """Which declared thresholds this run breached, in plain words."""
    checks = [
        ("page coverage", args.min_coverage, score.page_coverage, "below", "{:.1%}"),
        ("recall", args.min_recall, score.recall, "below", "{:.1%}"),
        ("mean decode", args.max_mean_ms, score.mean_ms, "above", "{:.0f} ms"),
        ("p95 decode", args.max_p95_ms, score.p95_ms, "above", "{:.0f} ms"),
    ]
    failures = []
    for label, limit, actual, direction, fmt in checks:
        if limit is None:
            continue
        breached = actual < limit if direction == "below" else actual > limit
        if breached:
            failures.append(f"{label} {fmt.format(actual)} is {direction} the "
                            f"required {fmt.format(limit)}")
    return failures


def cmd_evaluate(args: argparse.Namespace) -> int:
    score = api.evaluate(args.images, args.template, license=args.license,
                         ground_truth=args.ground_truth)
    failures = _gate_failures(args, score)

    if args.json:
        print(json.dumps({
            "template": args.template, "images": args.images,
            "recall": round(score.recall, 4),
            "page_coverage": round(score.page_coverage, 4),
            "pages_read": score.pages_partial, "pages_total": score.pages_total,
            "pages_undetermined": score.pages_undetermined,
            "mean_ms": round(score.mean_ms, 1), "p95_ms": round(score.p95_ms, 1),
            "passed": not failures, "failures": failures,
            "environment": environment_fingerprint(),
        }, indent=2))
        return 1 if failures else 0

    print(f"  template          {args.template}")
    print(f"  pages            {score.pages_total}")
    print(f"  page coverage    {score.page_coverage:6.1%}  "
          f"({score.pages_partial}/{score.pages_total} produced a read)")
    print(f"  recall           {score.recall:6.1%}  ({score.found}/{score.expected} codes)")
    print(f"  mean decode      {score.mean_ms:8.1f} ms   p95 {score.p95_ms:.0f} ms")
    if not args.ground_truth:
        print("  note: without --ground-truth the truth set is this template's own output,")
        print("        so recall is 100% by construction. Page coverage is the real number.")
    if failures:
        print("\n  FAILED:")
        for failure in failures:
            print(f"    - {failure}")
        return 1
    if any(v is not None for v in (args.min_coverage, args.min_recall,
                                   args.max_mean_ms, args.max_p95_ms)):
        print("\n  all declared thresholds met")
    return 0


def cmd_replay(args: argparse.Namespace) -> int:
    with open(args.results, encoding="utf-8") as handle:
        recorded = json.load(handle)
    if recorded.get("schema") not in ("dbr-autotune/1", None):
        print(f"warning: unfamiliar schema {recorded.get('schema')!r}", file=sys.stderr)

    request = TuneRequest.from_json(recorded.get("request") or {})
    if args.images:
        request.images = args.images
    if args.out:
        request.out = args.out
    if args.license:
        request.license = args.license

    then = recorded.get("environment") or {}
    now = environment_fingerprint()
    drift = compare_environments(then, now)
    print(f"replaying {args.results}")
    print(f"  images   {', '.join(request.images)}")
    if drift:
        print(f"  ! environment differs: {describe_drift(drift)}")
        for key, values in drift.items():
            print(f"      {key:<12} recorded {values['a']!r}  now {values['b']!r}")
    else:
        print("  environment matches the recorded run")

    missing = [p for p in request.images if not os.path.exists(p)]
    if missing:
        print(f"error: image path(s) not found: {', '.join(missing)}\n"
              f"       pass --images to point at where they live here.", file=sys.stderr)
        return 2

    outcome = tune(request, log=print)
    _print_tune_summary(outcome)

    before = (recorded.get("selected") or {}).get("page_coverage")
    if before is not None:
        after = outcome.final.page_coverage
        print(f"\n  recorded page coverage {before:.1%}  ->  this run {after:.1%}"
              f"   ({(after - before) * 100:+.1f} pts)")
    return 0


def cmd_ui(args: argparse.Namespace) -> int:
    from .server import serve
    serve(host=args.host, port=args.port, open_browser=not args.no_browser)
    return 0


def cmd_selfcheck(_args: argparse.Namespace) -> int:
    from .selfcheck import main as run_selfcheck
    return run_selfcheck()


# --- entry point ----------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    # `dbr-autotune ./images` - an unrecognised first token means "tune".
    if argv and argv[0] not in VERBS and not argv[0].startswith("-"):
        argv.insert(0, "tune")
    args = build_parser().parse_args(argv)

    verb = getattr(args, "verb", None)
    if verb is None:
        build_parser().print_help()
        return 1
    handler = {"tune": cmd_tune, "cv": cmd_cv, "evaluate": cmd_evaluate,
               "replay": cmd_replay, "ui": cmd_ui, "selfcheck": cmd_selfcheck}[verb]
    try:
        return handler(args)
    except FileNotFoundError as exc:
        print(f"error: path not found: {exc}", file=sys.stderr)
        return 2
    except (ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
