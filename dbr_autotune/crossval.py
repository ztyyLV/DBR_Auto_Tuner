"""K-fold cross-validation of a tuning run.

A single tune-then-holdout split tells you one number from one arbitrary
partition. With a few dozen images that number moves a lot depending on which
images happened to land in the holdout, and it is easy to read a lucky split as
evidence the template generalises.

K-fold runs the whole pipeline K times, each time tuning on K-1 folds and
scoring on the fold left out, and reports the spread as well as the mean. A
large spread means the result depends on which images you tuned on - that is the
signal to collect more images, not to trust the best fold.

It costs K full tuning runs. Use ``--quick`` or ``--time-budget`` with it.
"""
from __future__ import annotations

import random
import statistics
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from . import dataset as ds, template as T
from .api import TuneRequest, evaluate_on, tune
from .config import resolve_license
from .repro import environment_fingerprint
from .scoring import Score


@dataclass
class FoldResult:
    index: int
    tuned_on: int
    validated_on: int
    knobs: Dict[str, Any]
    train: Score
    validation: Score

    def to_json(self) -> Dict[str, Any]:
        return {
            "fold": self.index,
            "tuned_on": self.tuned_on,
            "validated_on": self.validated_on,
            "train_recall": round(self.train.recall, 4),
            "train_page_coverage": round(self.train.page_coverage, 4),
            "validation_recall": round(self.validation.recall, 4),
            "validation_page_coverage": round(self.validation.page_coverage, 4),
            "validation_mean_ms": round(self.validation.mean_ms, 1),
            "knobs": T.describe(self.knobs),
        }


@dataclass
class CrossValidation:
    folds: List[FoldResult] = field(default_factory=list)
    elapsed_s: float = 0.0
    k: int = 0
    pages: int = 0
    seed: int = 0

    # -- aggregate ---------------------------------------------------------

    def _values(self, attribute: str) -> List[float]:
        return [getattr(f.validation, attribute) for f in self.folds]

    def mean(self, attribute: str = "page_coverage") -> float:
        values = self._values(attribute)
        return statistics.fmean(values) if values else 0.0

    def stdev(self, attribute: str = "page_coverage") -> float:
        values = self._values(attribute)
        return statistics.stdev(values) if len(values) > 1 else 0.0

    def spread(self, attribute: str = "page_coverage") -> tuple:
        values = self._values(attribute)
        return (min(values), max(values)) if values else (0.0, 0.0)

    def knob_stability(self) -> Dict[str, Dict[str, int]]:
        """How often each fold settled on the same value for each knob.

        A knob that every fold agrees on is a property of the problem. One that
        changes fold to fold was fitted to whichever images that fold happened to
        see, and should not be trusted.
        """
        tally: Dict[str, Dict[str, int]] = {}
        for fold in self.folds:
            for knob, value in T.describe(fold.knobs).items():
                rendered = repr(value)
                tally.setdefault(knob, {}).setdefault(rendered, 0)
                tally[knob][rendered] += 1
        return tally

    def consensus_knobs(self) -> Dict[str, Any]:
        """Knob values a strict majority of folds agreed on."""
        out: Dict[str, Any] = {}
        need = len(self.folds) // 2 + 1
        for fold in self.folds:
            for knob, value in T.describe(fold.knobs).items():
                if knob in out:
                    continue
                agreeing = sum(1 for f in self.folds
                               if T.describe(f.knobs).get(knob) == value)
                if agreeing >= need:
                    out[knob] = value
        return out

    def to_json(self) -> Dict[str, Any]:
        low, high = self.spread("page_coverage")
        return {
            "schema": "dbr-autotune-cv/1",
            "k": self.k,
            "pages": self.pages,
            "seed": self.seed,
            "elapsed_s": round(self.elapsed_s, 1),
            "environment": environment_fingerprint(),
            "validation": {
                "page_coverage_mean": round(self.mean("page_coverage"), 4),
                "page_coverage_stdev": round(self.stdev("page_coverage"), 4),
                "page_coverage_min": round(low, 4),
                "page_coverage_max": round(high, 4),
                "recall_mean": round(self.mean("recall"), 4),
                "recall_stdev": round(self.stdev("recall"), 4),
                "mean_ms": round(self.mean("mean_ms"), 1),
            },
            "knob_stability": self.knob_stability(),
            "consensus_knobs": self.consensus_knobs(),
            "folds": [f.to_json() for f in self.folds],
        }

    def summary_lines(self) -> List[str]:
        low, high = self.spread("page_coverage")
        lines = [
            f"  folds             {self.k} x {self.pages} pages (seed {self.seed})",
            f"  page coverage     {self.mean('page_coverage'):6.1%} "
            f"+/- {self.stdev('page_coverage'):.1%}   "
            f"(worst fold {low:.1%}, best {high:.1%})",
            f"  recall            {self.mean('recall'):6.1%} "
            f"+/- {self.stdev('recall'):.1%}",
            f"  mean decode       {self.mean('mean_ms'):8.1f} ms/page",
        ]
        unstable = [k for k, counts in self.knob_stability().items() if len(counts) > 1]
        if unstable:
            lines.append(f"  unstable knobs    {', '.join(sorted(unstable))}")
            lines.append("                    (chosen differently by different folds - "
                         "fitted to the sample, not the problem)")
        return lines


def _folds(data: ds.Dataset, k: int, seed: int) -> List[ds.Dataset]:
    """Split into K folds **by file**, never by page.

    Pages of one PDF or TIFF are usually near-duplicates of each other, and a
    multi-page file cannot be handed to two folds anyway, so splitting per page
    would put the same document on both sides and quietly inflate every
    validation score.
    """
    by_file: Dict[str, List[ds.Sample]] = {}
    for sample in data.samples:
        by_file.setdefault(sample.path, []).append(sample)

    paths = sorted(by_file)
    random.Random(seed).shuffle(paths)
    buckets: List[List[ds.Sample]] = [[] for _ in range(k)]
    # Round-robin the files, largest first, so page counts stay even when some
    # documents carry many more pages than others.
    for index, path in enumerate(sorted(paths, key=lambda p: -len(by_file[p]))):
        buckets[index % k].extend(by_file[path])
    return [ds.Dataset(sorted(b, key=lambda s: s.key)) for b in buckets]


def cross_validate(request: TuneRequest, k: int = 5,
                   log: Optional[Callable[[str], None]] = None,
                   progress: Optional[Callable[[Dict[str, Any]], None]] = None
                   ) -> CrossValidation:
    """Run the pipeline K times, each fold validated on pages it never saw."""
    log = log or (lambda _msg: None)
    emit = progress or (lambda _event: None)
    started = time.perf_counter()

    data = ds.discover(request.images, recursive=request.recursive, limit=request.limit)
    if len(data) < k * 2:
        raise ValueError(f"need at least {k * 2} pages for {k}-fold cross-validation, "
                         f"found {len(data)}")

    buckets = _folds(data, k, request.seed)
    outcome = CrossValidation(k=k, pages=len(data), seed=request.seed)
    licence = resolve_license(request.license)

    for index in range(k):
        validation = buckets[index]
        training = ds.Dataset(sorted(
            [s for j, bucket in enumerate(buckets) if j != index for s in bucket.samples],
            key=lambda s: s.key))
        log(f"\n===== fold {index + 1}/{k}: tune on {len(training)} pages, "
            f"validate on {len(validation)} =====")
        emit({"event": "fold_start", "fold": index + 1, "k": k,
              "tuned_on": len(training), "validated_on": len(validation)})

        # Tune on this fold's training pages only. No artefacts per fold - the
        # point is the score, not K sets of templates.
        fold_request = TuneRequest(**{**request.to_json(),
                                      "images": [s.path for s in training.samples],
                                      "out": None, "holdout": 0.0,
                                      "write_report": False})
        fold_outcome = tune(fold_request, log=log)
        best_knobs = fold_outcome.emit_knobs["max_recall"]

        score = evaluate_on(validation, best_knobs, licence,
                            fold_request.to_options(),
                            probe_knobs=[t.knobs for t in fold_outcome.result.trials
                                         if t.phase == "probe"])
        fold = FoldResult(index=index + 1, tuned_on=len(training),
                          validated_on=len(validation), knobs=best_knobs,
                          train=fold_outcome.final, validation=score)
        outcome.folds.append(fold)
        log(f"  fold {index + 1} validation: {score.summary()}")
        emit({"event": "fold_done", "fold": index + 1, "k": k, **fold.to_json()})

    outcome.elapsed_s = time.perf_counter() - started
    emit({"event": "cv_done", "summary": outcome.to_json()})
    return outcome
