"""Turning a run into comparable numbers, and comparing them.

Scores are pure functions of ``(RunResult, GroundTruth)``, so when the truth set
grows mid-search every cached run can be re-scored for free and all candidates
stay on one denominator.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from .engine import RunResult
from .truth import GroundTruth, match


@dataclass
class Score:
    name: str
    fingerprint: str
    found: int = 0               # ground-truth codes recovered
    expected: int = 0            # ground-truth codes present
    extra: int = 0               # detections not in the truth set
    pages_complete: int = 0      # pages where every expected code was read
    pages_partial: int = 0       # pages where at least one was read
    pages_expected: int = 0      # pages that contain at least one expected code
    pages_total: int = 0
    errors: int = 0
    timeouts: int = 0
    mean_ms: float = 0.0
    p95_ms: float = 0.0
    max_ms: float = 0.0
    total_ms: float = 0.0
    wall_ms: float = 0.0
    serial: bool = False
    per_page: Dict[str, dict] = field(default_factory=dict)

    @property
    def recall(self) -> float:
        """Recall over the truth set.

        With an inferred truth set the denominator only counts codes that *some*
        configuration decoded, so this number cannot see a barcode nothing ever
        read. Read it together with :attr:`pages_undetermined`.
        """
        return self.found / self.expected if self.expected else 0.0

    @property
    def pages_undetermined(self) -> int:
        """Pages where the truth set holds nothing.

        Either the page genuinely carries no barcode, or it carries one that no
        configuration managed to decode. Inference cannot tell the two apart -
        only a supplied ground truth can.
        """
        return self.pages_total - self.pages_expected

    @property
    def page_coverage(self) -> float:
        """Pages where at least one code was read, over *every* page.

        Unlike :attr:`recall` this denominator is the whole dataset, so it never
        flatters a template by quietly dropping the pages nothing could read.
        """
        return self.pages_partial / self.pages_total if self.pages_total else 0.0

    @property
    def recall_floor(self) -> float:
        """Recall if every undetermined page turned out to hold one code."""
        worst = self.expected + self.pages_undetermined
        return self.found / worst if worst else 0.0

    @property
    def page_rate(self) -> float:
        """Fraction of code-bearing pages where every expected code was read."""
        return self.pages_complete / self.pages_expected if self.pages_expected else 0.0

    def summary(self) -> str:
        undetermined = (f"  {self.pages_undetermined} undetermined"
                        if self.pages_undetermined else "")
        return (f"recall {self.recall:6.1%}  pages {self.pages_complete}/{self.pages_expected}"
                f"  mean {self.mean_ms:7.1f}ms  p95 {self.p95_ms:7.1f}ms"
                + (f"  +{self.extra} extra" if self.extra else "") + undetermined)


def score(run: RunResult, truth: GroundTruth) -> Score:
    result = Score(name=run.name, fingerprint=run.fingerprint,
                   wall_ms=run.wall_ms, serial=run.serial)
    times: List[float] = []
    for key, page in run.pages.items():
        expected = truth.pages.get(key, {})
        matched, extra = match(page.detections, expected)
        result.expected += len(expected)
        result.found += len(matched)
        result.extra += len(extra)
        result.pages_total += 1
        if expected:
            result.pages_expected += 1
            if len(matched) == len(expected):
                result.pages_complete += 1
            if matched:
                result.pages_partial += 1
        if page.error:
            result.errors += 1
            if "timeout" in page.error.lower():
                result.timeouts += 1
        times.append(page.ms)
        result.per_page[key] = {
            "label": page.sample.label,
            "ms": round(page.ms, 1),
            "expected": len(expected),
            "found": len(matched),
            "extra": len(extra),
            "texts": sorted(d.text for d in page.detections),
            "formats": sorted({d.format for d in page.detections}),
            "error": page.error,
        }
    if times:
        times.sort()
        result.total_ms = sum(times)
        result.mean_ms = result.total_ms / len(times)
        result.p95_ms = times[min(len(times) - 1, int(round(0.95 * (len(times) - 1))))]
        result.max_ms = times[-1]
    return result


# --- candidate comparison -------------------------------------------------

@dataclass
class Objective:
    """How recall and speed trade off against each other.

    ``speed_weight`` 0.0 is pure recall: speed only breaks ties between
    configurations that read the same number of codes. Above 0 the two are
    traded on one scale, where **each halving of decode time is worth
    ``speed_weight * 10`` recall points** - so at 1.0 a template that is twice
    as fast may give up ten points of recall and still win.

    ``recall_tolerance`` applies to the tie-breaking path only: how much recall
    a faster template may give up outright. 0.0 means never.
    """
    speed_weight: float = 0.0
    recall_tolerance: float = 0.0

    # Latency the penalty is measured from. Any constant works - it cancels
    # when two candidates are compared - so this is only about readability.
    reference_ms: float = 100.0

    def value(self, s: Score) -> float:
        if self.speed_weight <= 0:
            return s.recall
        import math
        doublings = math.log2(max(s.mean_ms, 1.0) / max(self.reference_ms, 1.0))
        return s.recall - self.speed_weight * 0.1 * doublings

    def better(self, candidate: Score, incumbent: Score) -> bool:
        """Is ``candidate`` a strict improvement over ``incumbent``?"""
        if self.speed_weight <= 0:
            if candidate.recall > incumbent.recall + 1e-9:
                return True
            if candidate.recall < incumbent.recall - self.recall_tolerance - 1e-9:
                return False
            # Equal (or within tolerance) recall: prefer fewer extras, then speed.
            if candidate.extra != incumbent.extra:
                return candidate.extra < incumbent.extra
            return candidate.mean_ms < incumbent.mean_ms * 0.98
        return self.value(candidate) > self.value(incumbent) + 1e-9


def pareto_front(scores: List[Score]) -> List[Score]:
    """Scores not dominated on both recall (higher) and mean_ms (lower)."""
    front: List[Score] = []
    for candidate in scores:
        dominated = any(
            other.recall >= candidate.recall and other.mean_ms <= candidate.mean_ms
            and (other.recall > candidate.recall or other.mean_ms < candidate.mean_ms)
            for other in scores if other is not candidate)
        if not dominated:
            front.append(candidate)
    front.sort(key=lambda s: (-s.recall, s.mean_ms))
    # Drop duplicates that share both coordinates.
    unique, seen = [], set()
    for s in front:
        key = (round(s.recall, 6), round(s.mean_ms, 1))
        if key not in seen:
            seen.add(key)
            unique.append(s)
    return unique
