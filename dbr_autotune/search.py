"""The tuning strategy.

Five phases, in order:

1. **Probe** - run a spread of seed configurations over every symbology to find
   out what is actually in the images and how hard they are.
2. **Truth** - fold every probe result into a pseudo ground-truth set.
3. **Narrow** - restrict the template to the symbologies that were actually
   seen. This is usually the single largest speed win and it cannot cost recall
   for codes of those formats.
4. **Ascend** - coordinate ascent: walk each knob's ladder, keep any level that
   improves the objective, repeat until a full sweep changes nothing.
5. **Trim** - having reached peak recall, switch stages back off one at a time
   and keep every removal that recall survives, then fit the timeout.

Anything the search decodes that was not in the truth set gets folded back in -
a configuration that reads a barcode nothing else could is a genuine find, not a
false positive - and every earlier candidate is re-scored so the comparison
stays on one denominator.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from . import space, template as T
from .engine import Engine, RunResult, TemplateRejected
from .scoring import Objective, Score, pareto_front, score as score_run
from .truth import GroundTruth, infer


@dataclass
class Trial:
    name: str
    phase: str
    knobs: Dict[str, Any]
    run: RunResult
    score: Score
    accepted: bool = False
    note: str = ""


@dataclass
class TuneOptions:
    speed_weight: float = 0.0
    recall_tolerance: float = 0.0
    max_trials: int = 200
    time_budget_s: float = 0.0        # 0 = unlimited
    rounds: int = 2
    grow_truth: bool = True
    min_agree_risky: int = 2
    min_weak_length: int = 4
    formats: Optional[List[str]] = None   # user override, skips discovery
    expected_count: Optional[int] = None
    skip_trim: bool = False


@dataclass
class TuneResult:
    truth: GroundTruth
    trials: List[Trial]
    best: Trial
    variants: Dict[str, Trial]
    profile: Dict[str, Any]
    baseline: Optional[Trial]
    front: List[Score]
    elapsed_s: float
    budget_hit: str = ""


class BudgetExhausted(RuntimeError):
    pass


class Tuner:
    def __init__(self, engine: Engine, options: TuneOptions,
                 log: Callable[[str], None] = print):
        self.engine = engine
        self.options = options
        self.log = log
        self.truth = GroundTruth(source="inferred")
        self.trials: List[Trial] = []
        self.objective = Objective(speed_weight=options.speed_weight,
                                   recall_tolerance=options.recall_tolerance)
        self._started = 0.0
        self._budget_hit = ""
        # Optional callback fired after every evaluated configuration. The web
        # UI uses it to stream progress; nothing in the search depends on it.
        self.on_trial: Callable[[Trial], None] | None = None

    # -- plumbing ----------------------------------------------------------

    def _check_budget(self) -> None:
        if self.options.max_trials and self.engine.trials >= self.options.max_trials:
            self._budget_hit = f"trial limit ({self.options.max_trials}) reached"
            raise BudgetExhausted(self._budget_hit)
        if self.options.time_budget_s:
            spent = time.perf_counter() - self._started
            if spent >= self.options.time_budget_s:
                self._budget_hit = f"time budget ({self.options.time_budget_s:.0f}s) reached"
                raise BudgetExhausted(self._budget_hit)

    def trial(self, knobs: Dict[str, Any], name: str, phase: str,
              absorb: bool = True) -> Trial:
        """Run one configuration, fold its finds into the truth set, score it."""
        self._check_budget()
        try:
            run = self.engine.run(knobs, name=name)
        except TemplateRejected as exc:
            self.log(f"    {name:<34} REJECTED {exc}")
            raise

        grew = 0
        if absorb and self.options.grow_truth:
            grew = self.truth.absorb(run, self.options.min_agree_risky,
                                     self.options.min_weak_length)
            if grew:
                self._rescore_all()

        result = score_run(run, self.truth)
        record = Trial(name=name, phase=phase, knobs=T.normalize(knobs), run=run,
                       score=result)
        self.trials.append(record)
        suffix = f"  (+{grew} new to truth)" if grew else ""
        self.log(f"    {name:<34} {result.summary()}{suffix}")
        if self.on_trial is not None:
            try:
                self.on_trial(record)
            except Exception:
                pass          # a broken listener must never abort the search
        return record

    def _rescore_all(self) -> None:
        for trial in self.trials:
            trial.score = score_run(trial.run, self.truth)

    def _best_of(self, trials: List[Trial]) -> Trial:
        best = trials[0]
        for candidate in trials[1:]:
            if self.objective.better(candidate.score, best.score):
                best = candidate
        return best

    # -- phase 1/2: probe and truth ---------------------------------------

    def probe(self, profile: Dict[str, Any]) -> Trial:
        formats = self.options.formats or space.DISCOVERY_FORMATS
        self.log(f"\n[1/5] Probing {len(self.engine.dataset)} pages with "
                 f"{len(space.seed_configs(formats))} seed configurations")
        probes: List[Trial] = []
        for name, knobs in space.seed_configs(formats):
            try:
                probes.append(self.trial(knobs, name, phase="probe"))
            except BudgetExhausted:
                break
            except TemplateRejected:
                continue
        if not probes:
            raise RuntimeError("no probe configuration ran successfully")

        self.log(f"\n[2/5] Ground truth inferred: {self.truth.total} codes across "
                 f"{self.truth.pages_with_codes}/{len(self.engine.dataset)} pages"
                 f"  formats={sorted(self.truth.formats()) or ['none']}")
        unconfirmed = self.truth.unconfirmed()
        if unconfirmed:
            self.log(f"      {len(unconfirmed)} unverified read(s) in weak-checksum "
                     f"symbologies were withheld from the truth set:")
            for key, entry in unconfirmed[:6]:
                self.log(f"        {entry.format:<26} {entry.text[:28]!r:<30} "
                         f"seen by {entry.agreements} config(s)")
            if len(unconfirmed) > 6:
                self.log(f"        ... and {len(unconfirmed) - 6} more")
        # Each probe was scored while the truth set was still filling up.
        self.log("      probe scores against the completed truth set:")
        for probe in probes:
            self.log(f"        {probe.name:<32} {probe.score.summary()}")
        return self._best_of(probes)

    # -- phase 3: narrow formats ------------------------------------------

    def narrow(self, start: Trial) -> Trial:
        self.log("\n[3/5] Narrowing symbologies")
        seen = sorted({fmt for fmt in self.truth.formats() if fmt.startswith("BF_")})
        if self.options.formats:
            self.log(f"      user-specified formats kept: {self.options.formats}")
            return start
        if not seen:
            self.log("      nothing decoded yet - keeping the full discovery set")
            return start
        if set(seen) == set(start.knobs["formats"]):
            return start
        candidate = T.with_knobs(start.knobs, formats=seen)
        try:
            trial = self.trial(candidate, f"narrow:{'+'.join(f[3:] for f in seen)[:26]}",
                               phase="narrow")
        except (BudgetExhausted, TemplateRejected):
            return start
        if trial.score.recall >= start.score.recall - 1e-9:
            trial.accepted = True
            trial.note = f"restricted to {', '.join(seen)}"
            speedup = start.score.mean_ms / max(trial.score.mean_ms, 1e-6)
            self.log(f"      accepted: {len(seen)} format(s), {speedup:.2f}x faster")
            return trial
        self.log("      rejected: narrowing cost recall, keeping the wide set")
        return start

    # -- phase 4: coordinate ascent ---------------------------------------

    def ascend(self, start: Trial, profile: Dict[str, Any]) -> Trial:
        self.log("\n[4/5] Coordinate ascent")
        best = start
        ladders = space.ladders(profile)
        for round_index in range(self.options.rounds):
            improved_any = False
            self.log(f"  -- round {round_index + 1}/{self.options.rounds} "
                     f"(incumbent recall {best.score.recall:.1%}, "
                     f"{best.score.mean_ms:.0f}ms)")
            for knob, levels in ladders:
                current = best.knobs[knob]
                candidates = [lv for lv in levels if lv != current]
                if not candidates:
                    continue
                round_best = best
                for level in candidates:
                    knobs = T.with_knobs(best.knobs, **{knob: level})
                    if T.fingerprint(knobs) == T.fingerprint(best.knobs):
                        continue
                    try:
                        trial = self.trial(knobs, f"{knob}={_short(level)}", phase="ascend")
                    except BudgetExhausted:
                        raise
                    except TemplateRejected:
                        continue
                    if self.objective.better(trial.score, round_best.score):
                        round_best = trial
                if round_best is not best:
                    round_best.accepted = True
                    round_best.note = f"{knob} -> {_short(round_best.knobs[knob])}"
                    self.log(f"      + {knob}: {_short(current)} -> "
                             f"{_short(round_best.knobs[knob])}   "
                             f"recall {best.score.recall:.1%} -> {round_best.score.recall:.1%}"
                             f", {best.score.mean_ms:.0f} -> {round_best.score.mean_ms:.0f}ms")
                    best = round_best
                    improved_any = True
            if not improved_any:
                self.log("      converged")
                break
        return best

    # -- phase 5: trim and fit --------------------------------------------

    def trim(self, start: Trial) -> Trial:
        self.log("\n[5/5] Trimming for speed at constant recall")
        best = start
        target_recall = start.score.recall
        for knob, off_value in space.TRIM_ORDER:
            if best.knobs[knob] == off_value:
                continue
            knobs = T.with_knobs(best.knobs, **{knob: off_value})
            try:
                trial = self.trial(knobs, f"trim:{knob}", phase="trim")
            except BudgetExhausted:
                break
            except TemplateRejected:
                continue
            if (trial.score.recall >= target_recall - 1e-9
                    and trial.score.extra <= best.score.extra
                    and trial.score.mean_ms < best.score.mean_ms):
                trial.accepted = True
                trial.note = f"{knob} removed, recall held at {target_recall:.1%}"
                self.log(f"      + dropped {knob}: "
                         f"{best.score.mean_ms:.0f} -> {trial.score.mean_ms:.0f}ms")
                best = trial

        # Expected-barcode count lets the SDK stop as soon as the page is done.
        counts = {len(page) for page in self.truth.pages.values() if page}
        if self.options.expected_count is not None:
            wanted = [self.options.expected_count]
        elif len(counts) == 1:
            wanted = [counts.pop()]
        elif counts:
            wanted = [max(counts)]
        else:
            wanted = []
        for count in wanted:
            if count == best.knobs["expected_count"]:
                continue
            knobs = T.with_knobs(best.knobs, expected_count=count)
            try:
                trial = self.trial(knobs, f"trim:expected_count={count}", phase="trim")
            except (BudgetExhausted, TemplateRejected):
                break
            if (trial.score.recall >= target_recall - 1e-9
                    and trial.score.mean_ms <= best.score.mean_ms):
                trial.accepted = True
                trial.note = f"ExpectedBarcodesCount={count}"
                self.log(f"      + ExpectedBarcodesCount={count}: "
                         f"{best.score.mean_ms:.0f} -> {trial.score.mean_ms:.0f}ms")
                best = trial
        return best

    def fit_timeout(self, best: Trial) -> Trial:
        """Set Timeout from the observed tail, with headroom."""
        tail = max(best.score.max_ms, best.score.p95_ms * 1.5)
        fitted = int(max(500, min(60000, round(tail * 2.0 / 100.0) * 100)))
        if fitted == best.knobs["timeout_ms"]:
            return best
        knobs = T.with_knobs(best.knobs, timeout_ms=fitted)
        try:
            trial = self.trial(knobs, f"timeout={fitted}ms", phase="trim")
        except (BudgetExhausted, TemplateRejected):
            return best
        if trial.score.recall >= best.score.recall - 1e-9:
            trial.accepted = True
            trial.note = f"Timeout fitted to {fitted}ms (2x observed worst page)"
            self.log(f"      + Timeout fitted to {fitted}ms")
            return trial
        return best

    # -- variants ----------------------------------------------------------

    # A rival has to be this much faster before it displaces an incumbent.
    # Search timings are measured with several decodes in flight, so small
    # differences are scheduling noise; picking the bare minimum of hundreds of
    # noisy measurements reliably picks the luckiest one, not the best one.
    _SPEED_MARGIN = 0.90

    def variants(self, best: Trial) -> Dict[str, Trial]:
        """Best-recall / balanced / fastest picks off the explored front.

        ``best`` - the configuration the search converged on, trimmed and with a
        fitted timeout - is the incumbent, not merely another candidate.
        """
        scored = [(t, t.score) for t in self.trials if t.score.expected]
        if not scored:
            return {"max_recall": best}
        top_recall = max(s.recall for _t, s in scored)

        def fastest_above(recall_floor: float, beat_ms: float) -> Optional[Trial]:
            pool = [(t, s) for t, s in scored
                    if s.recall >= recall_floor and s.mean_ms < beat_ms * self._SPEED_MARGIN]
            if not pool:
                return None
            return min(pool, key=lambda pair: pair[1].mean_ms)[0]

        out: Dict[str, Trial] = {"max_recall": best}
        # Only displace the converged configuration for a decisive speed win.
        challenger = fastest_above(top_recall - 1e-9, best.score.mean_ms)
        if challenger is not None:
            out["max_recall"] = challenger

        incumbent_ms = out["max_recall"].score.mean_ms
        balanced = fastest_above(top_recall - 0.02, incumbent_ms)
        if balanced is not None and balanced is not out["max_recall"]:
            out["balanced"] = balanced
            incumbent_ms = balanced.score.mean_ms

        fastest = fastest_above(top_recall * 0.90, incumbent_ms)
        if fastest is not None and not any(existing is fastest for existing in out.values()):
            out["fastest"] = fastest
        return out

    # -- driver ------------------------------------------------------------

    def tune(self, profile: Dict[str, Any]) -> TuneResult:
        self._started = time.perf_counter()
        baseline: Optional[Trial] = None
        try:
            best = self.probe(profile)
            baseline = next((t for t in self.trials if t.name == "seed:default"), best)
            best = self.narrow(best)
            best = self.ascend(best, profile)
            if not self.options.skip_trim:
                best = self.trim(best)
                best = self.fit_timeout(best)
        except BudgetExhausted as exc:
            self.log(f"\n  ! budget exhausted: {exc}")
            best = self._best_of(self.trials)
        best.accepted = True

        return TuneResult(
            truth=self.truth,
            trials=self.trials,
            best=best,
            variants=self.variants(best),
            profile=profile,
            baseline=baseline,
            front=pareto_front([t.score for t in self.trials]),
            elapsed_s=time.perf_counter() - self._started,
            budget_hit=self._budget_hit,
        )


def _short(value: Any) -> str:
    """Compact one-line rendering of a knob level for the log."""
    if value is None:
        return "off"
    if isinstance(value, bool):
        return "on" if value else "off"
    if isinstance(value, list):
        if not value:
            return "[]"
        trimmed = [v.split("_", 1)[1] if isinstance(v, str) and "_" in v else str(v)
                   for v in value]
        text = "+".join(trimmed)
        if len(text) <= 40:
            return text
        # Keep the leading mode so different long lists stay distinguishable.
        return f"{trimmed[0]}+{len(trimmed) - 1} more"
    if isinstance(value, dict):
        return ",".join(f"{k}={v}" for k, v in value.items())
    return str(value)
