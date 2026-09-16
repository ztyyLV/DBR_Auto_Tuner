"""Public programmatic API.

Everything the command line and the web UI do goes through here, so an
integrator gets the same capability without shelling out:

    from dbr_autotune import TuneRequest, tune

    outcome = tune(TuneRequest(images=["./photos"], out="./out"))
    print(outcome.summary["selected"]["recall"])
    template_json = outcome.templates()["AutoTuned_MaxRecall"]

``tune()`` is synchronous and single-shot. Pass ``log=`` for human-readable
lines and ``progress=`` for structured events (the web UI streams those).
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional

from . import dataset as ds, report as rp, template as T, truth as gt
from .config import resolve_license
from .engine import Engine, image_stats
from .repro import environment_fingerprint
from .scoring import Score, score as score_run
from .search import TuneOptions, TuneResult, Tuner

VARIANT_TEMPLATE_NAMES = {
    "max_recall": "AutoTuned_MaxRecall",
    "balanced": "AutoTuned_Balanced",
    "fastest": "AutoTuned_Fastest",
}

Log = Callable[[str], None]
Progress = Callable[[Dict[str, Any]], None]


@dataclass
class TuneRequest:
    """Everything one tuning run needs. Field names match the CLI flags."""

    images: List[str]
    out: Optional[str] = None                 # None = do not write anything
    license: Optional[str] = None
    jobs: int = 0                             # 0 = half the cores
    recursive: bool = True
    limit: Optional[int] = None
    holdout: float = 0.0
    seed: int = 0                             # governs the holdout/fold split

    speed_weight: float = 0.0
    recall_tolerance: float = 0.0

    max_trials: int = 200
    time_budget_s: float = 0.0
    rounds: int = 2
    skip_trim: bool = False

    formats: Optional[List[str]] = None
    expected_count: Optional[int] = None
    ground_truth: Optional[str] = None
    freeze_truth: bool = False
    min_agree: int = 2
    min_weak_length: int = 4

    write_report: bool = True

    def resolved_jobs(self) -> int:
        return self.jobs or max(1, (os.cpu_count() or 2) // 2)

    def to_options(self) -> TuneOptions:
        return TuneOptions(
            speed_weight=self.speed_weight,
            recall_tolerance=self.recall_tolerance,
            max_trials=self.max_trials,
            time_budget_s=self.time_budget_s,
            rounds=self.rounds,
            grow_truth=not self.freeze_truth,
            min_agree_risky=self.min_agree,
            min_weak_length=self.min_weak_length,
            formats=self.formats,
            expected_count=self.expected_count,
            skip_trim=self.skip_trim,
        )

    def to_json(self) -> Dict[str, Any]:
        """Serialisable form - this is what makes a run replayable."""
        return asdict(self)

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> "TuneRequest":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass
class TuneOutcome:
    request: TuneRequest
    result: TuneResult
    scores: Dict[str, Score]                  # keyed by variant, plus best/baseline
    emit_knobs: Dict[str, Dict[str, Any]]
    outputs: Dict[str, str] = field(default_factory=dict)      # variant -> filename
    paths: Dict[str, str] = field(default_factory=dict)        # artefact -> full path
    holdout: Optional[Score] = None
    summary: Dict[str, Any] = field(default_factory=dict)

    @property
    def final(self) -> Score:
        return self.scores.get("best", self.result.best.score)

    def templates(self) -> Dict[str, Dict[str, Any]]:
        """Template name -> template document, ready for ``InitSettings``."""
        return {VARIANT_TEMPLATE_NAMES.get(key, key): T.build(knobs,
                                                              VARIANT_TEMPLATE_NAMES.get(key, key))
                for key, knobs in self.emit_knobs.items()}

    def combined_template(self) -> Dict[str, Any]:
        """All variants in one document, selectable by name at capture time."""
        return T.build_multi({VARIANT_TEMPLATE_NAMES.get(k, k): v
                              for k, v in self.emit_knobs.items()})


def discover(request: TuneRequest) -> ds.Dataset:
    return ds.discover(request.images, recursive=request.recursive, limit=request.limit)


def estimate_trials(request: TuneRequest, profile: Dict[str, Any]) -> int:
    """Roughly how many configurations a run will evaluate, for progress bars."""
    from . import space
    seeds = len(space.seed_configs(space.DISCOVERY_FORMATS))
    per_round = sum(len(levels) - 1 for _k, levels in space.ladders(profile))
    trim = 0 if request.skip_trim else len(space.TRIM_ORDER) + 2
    estimate = seeds + 1 + per_round * request.rounds + trim
    return min(estimate, request.max_trials) if request.max_trials else estimate


def fit_timeout(knobs: Dict[str, Any], measured: Score) -> Dict[str, Any]:
    """Propose a ``Timeout`` of twice the slowest page actually observed.

    Only a proposal: the caller must re-measure with it and keep the original
    if anything stopped decoding. ``Timeout`` is not a pure wall-clock cap --
    the SDK budgets its decoding effort against it, so lowering it can cost a
    page that had finished well inside the old limit.

    The point of fitting it at all is that an exhaustive probe configuration may
    have been allowed 30 seconds, and shipping that to production means one
    pathological image can block a caller for half a minute.
    """
    tail = max(measured.max_ms, measured.p95_ms * 1.5, 250.0)
    fitted = int(min(60000, max(500, round(tail * 2.0 / 100.0) * 100)))
    return T.with_knobs(knobs, timeout_ms=fitted)


def tune(request: TuneRequest, log: Optional[Log] = None,
         progress: Optional[Progress] = None) -> TuneOutcome:
    """Run the full pipeline. Writes to ``request.out`` when it is set."""
    log = log or (lambda _msg: None)
    emit = progress or (lambda _event: None)
    started = time.perf_counter()
    licence = resolve_license(request.license)

    data = discover(request)
    if not len(data):
        raise ValueError("no supported images found")
    tune_set, holdout = (data.split(request.holdout, seed=request.seed)
                         if request.holdout else (data, ds.Dataset([])))

    log(f"dbr-autotune  |  {len(data)} page(s) from {', '.join(request.images)}")
    if data.skipped:
        log(f"  {len(data.skipped)} unsupported file(s) ignored")
    if data.uncounted:
        log(f"  WARNING: could not count pages in {len(data.uncounted)} file(s); they are "
            f"treated as single-page, so per-page recall and timings will be wrong. "
            f"Install pypdf for PDF page counts.")
    if len(holdout):
        log(f"  tuning on {len(tune_set)} page(s), holding out {len(holdout)} for validation")

    profile = image_stats(tune_set)
    if profile:
        log("  " + "  ".join(f"{k}={v}" for k, v in profile.items()))
    jobs = request.resolved_jobs()
    log(f"  {jobs} parallel worker(s); search timings are relative, "
        f"the final numbers are measured single-threaded")

    total_estimate = estimate_trials(request, profile)
    emit({"event": "start", "pages": len(data), "tuning_pages": len(tune_set),
          "holdout_pages": len(holdout), "profile": profile,
          "estimated_trials": total_estimate})

    engine = Engine(tune_set, licence, jobs=jobs)
    options = request.to_options()
    tuner = Tuner(engine, options, log=log)

    def on_trial(trial) -> None:
        emit({"event": "trial", "index": len(tuner.trials), "total": total_estimate,
              "phase": trial.phase, "name": trial.name,
              "recall": trial.score.recall, "mean_ms": trial.score.mean_ms,
              "page_coverage": trial.score.page_coverage,
              "accepted": trial.accepted})
    tuner.on_trial = on_trial

    if request.ground_truth:
        tuner.truth = gt.load(request.ground_truth, tune_set)
        tuner.options.grow_truth = False
        log(f"  ground truth loaded from {request.ground_truth}: {tuner.truth.total} codes")

    result = tuner.tune(profile)

    # -- final measurement, single threaded --------------------------------
    log("\nRe-measuring the selected templates single-threaded")
    emit({"event": "phase", "phase": "measure"})
    serial_engine = Engine(tune_set, licence, jobs=1)
    scores: Dict[str, Score] = {}
    emit_knobs: Dict[str, Dict[str, Any]] = {}
    for key, trial in result.variants.items():
        name = VARIANT_TEMPLATE_NAMES.get(key, key)
        run = serial_engine.run(trial.knobs, name=name, serial=True)
        measured = score_run(run, result.truth)
        emit_knobs[key], scores[key] = trial.knobs, measured
        note = ""

        # Fitting the timeout down is only safe if it is then *verified*.
        # Timeout is not a pure wall-clock cap: the SDK budgets its decoding
        # effort against it, so a lower value can stop a page decoding even
        # though that page finished far inside the old limit. Emitting a
        # template scored under a timeout it does not carry would overstate it.
        candidate = fit_timeout(trial.knobs, measured)
        fitted = candidate["timeout_ms"]
        if fitted != trial.knobs["timeout_ms"]:
            check = score_run(serial_engine.run(candidate, name=name, serial=True),
                              result.truth)
            if (check.found >= measured.found
                    and check.pages_partial >= measured.pages_partial):
                emit_knobs[key], scores[key] = candidate, check
                note = f"   Timeout -> {fitted}ms (verified)"
            else:
                note = (f"   Timeout kept at {trial.knobs['timeout_ms']}ms: "
                        f"{fitted}ms lost "
                        f"{measured.pages_partial - check.pages_partial} page(s)")
        log(f"    {key:<12} {scores[key].summary()}{note}")
    if "max_recall" in scores:
        scores["best"] = scores["max_recall"]
    final = scores.get("best", result.best.score)
    if result.baseline is not None:
        run = serial_engine.run(result.baseline.knobs, name="Baseline", serial=True)
        scores["baseline"] = score_run(run, result.truth)
        log(f"    {'baseline':<12} {scores['baseline'].summary()}")

    # -- holdout -----------------------------------------------------------
    holdout_score = None
    if len(holdout):
        log("\nValidating on the held-out pages")
        emit({"event": "phase", "phase": "holdout"})
        holdout_score = evaluate_on(holdout, result.best.knobs, licence, options,
                                    probe_knobs=[t.knobs for t in result.trials
                                                 if t.phase == "probe"])
        log(f"    holdout      {holdout_score.summary()}")

    outcome = TuneOutcome(request=request, result=result, scores=scores,
                          emit_knobs=emit_knobs, holdout=holdout_score)
    outcome.summary = _summary(outcome, data, tune_set, holdout, profile,
                               time.perf_counter() - started)

    if request.out:
        _write(outcome, request.out)
    emit({"event": "done", "summary": outcome.summary, "paths": outcome.paths})
    return outcome


def evaluate_on(pages: ds.Dataset, knobs: Dict[str, Any], licence: str,
                options: TuneOptions,
                probe_knobs: Optional[List[Dict[str, Any]]] = None) -> Score:
    """Score one configuration on pages the search never saw.

    The truth set for these pages is rebuilt from scratch - the probe
    configurations plus the candidate itself - so a code only present here still
    counts, and the candidate is not simply graded against its own output.
    """
    engine = Engine(pages, licence, jobs=1)
    probe_runs = [engine.run(k, name=f"probe{i}")
                  for i, k in enumerate(probe_knobs or [])]
    truth = gt.infer(probe_runs, options.min_agree_risky, options.min_weak_length)
    run = engine.run(knobs, name="Evaluation", serial=True)
    truth.absorb(run, options.min_agree_risky, options.min_weak_length)
    return score_run(run, truth)


def evaluate(images: List[str], template: str | Dict[str, Any],
             license: Optional[str] = None, jobs: int = 1,
             ground_truth: Optional[str] = None) -> Score:
    """Score an existing template file against a folder of images.

    ``template`` is a path to a template JSON file or an already-parsed
    document. Use it to check a template that came from somewhere else, or to
    re-check one after an SDK upgrade.
    """
    data = ds.discover(images)
    if not len(data):
        raise ValueError("no supported images found")
    document = json.load(open(template, encoding="utf-8")) if isinstance(template, str) \
        else template
    name = document["CaptureVisionTemplates"][0]["Name"]

    engine = Engine(data, resolve_license(license), jobs=jobs)
    run = engine.run_document(document, name=name, serial=jobs == 1)
    if ground_truth:
        truth = gt.load(ground_truth, data)
    else:
        # Truth inferred from a single run, so the corroboration rule that
        # protects against spurious weak-checksum reads cannot be satisfied by
        # anything -- requiring two agreeing configurations here would silently
        # discard every Code 39 / Codabar / ITF read and report the template as
        # having missed those pages. Judge this run on its own output; the
        # minimum-payload-length guard still applies.
        truth = gt.infer([run], min_agree_risky=1)
    return score_run(run, truth)


# --- artefacts ------------------------------------------------------------

def _summary(outcome: TuneOutcome, data: ds.Dataset, tune_set: ds.Dataset,
             holdout: ds.Dataset, profile: Dict[str, Any], elapsed: float) -> Dict[str, Any]:
    result, scores, final = outcome.result, outcome.scores, outcome.final
    return {
        "schema": "dbr-autotune/1",
        "elapsed_s": round(elapsed, 1),
        "environment": environment_fingerprint(),
        "request": outcome.request.to_json(),
        "dataset": {"roots": outcome.request.images, "pages": len(data),
                    "tuned_on": len(tune_set), "holdout": len(holdout),
                    "profile": profile},
        "ground_truth": result.truth.to_json(),
        "withheld": [
            {"page": os.path.basename(key), "format": entry.format, "text": entry.text,
             "agreements": entry.agreements, "confidence": entry.confidence,
             "reason": "weak-checksum symbology, not corroborated"}
            for key, entry in result.truth.unconfirmed()],
        "undetermined_pages": [info["label"] for info in final.per_page.values()
                               if info["expected"] == 0],
        "selected": {
            "template_name": VARIANT_TEMPLATE_NAMES["max_recall"],
            "knobs": T.describe(outcome.emit_knobs["max_recall"]),
            "recall": round(final.recall, 4),
            "recall_floor": round(final.recall_floor, 4),
            "page_coverage": round(final.page_coverage, 4),
            "mean_ms": round(final.mean_ms, 1),
            "p95_ms": round(final.p95_ms, 1),
        },
        "variants": {
            key: {"file": outcome.outputs.get(key, ""),
                  "recall": round(scores.get(key, t.score).recall, 4),
                  "page_coverage": round(scores.get(key, t.score).page_coverage, 4),
                  "mean_ms": round(scores.get(key, t.score).mean_ms, 1),
                  "p95_ms": round(scores.get(key, t.score).p95_ms, 1),
                  "knobs": T.describe(outcome.emit_knobs[key])}
            for key, t in result.variants.items()},
        "baseline": None if "baseline" not in scores else {
            "recall": round(scores["baseline"].recall, 4),
            "mean_ms": round(scores["baseline"].mean_ms, 1)},
        "holdout": None if outcome.holdout is None else {
            "pages": len(holdout),
            "recall": round(outcome.holdout.recall, 4),
            "page_coverage": round(outcome.holdout.page_coverage, 4),
            "mean_ms": round(outcome.holdout.mean_ms, 1)},
        "trials": [{
            "phase": t.phase, "name": t.name, "accepted": t.accepted, "note": t.note,
            "recall": round(t.score.recall, 4), "mean_ms": round(t.score.mean_ms, 1),
            "extra": t.score.extra, "knobs": T.describe(t.knobs)}
            for t in result.trials],
        "per_page": final.per_page,
    }


def _write(outcome: TuneOutcome, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    for key, knobs in outcome.emit_knobs.items():
        name = VARIANT_TEMPLATE_NAMES.get(key, key)
        path = os.path.join(out_dir, f"{name}.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(T.build(knobs, name), handle, indent=2, ensure_ascii=False)
        outcome.outputs[key] = os.path.basename(path)
        outcome.paths[key] = path
    # outputs only became known just now, so refresh the variant file names
    for key in outcome.summary.get("variants", {}):
        outcome.summary["variants"][key]["file"] = outcome.outputs.get(key, "")

    combined = os.path.join(out_dir, "autotuned-templates.json")
    with open(combined, "w", encoding="utf-8") as handle:
        json.dump(outcome.combined_template(), handle, indent=2, ensure_ascii=False)
    outcome.paths["combined"] = combined

    results = os.path.join(out_dir, "results.json")
    with open(results, "w", encoding="utf-8") as handle:
        json.dump(outcome.summary, handle, indent=2, ensure_ascii=False)
    outcome.paths["results"] = results

    if outcome.request.write_report:
        report = os.path.join(out_dir, "report.html")
        with open(report, "w", encoding="utf-8") as handle:
            handle.write(rp.render(outcome.result, outcome.request.images,
                                   outcome.outputs, outcome.scores, outcome.emit_knobs))
        outcome.paths["report"] = report
