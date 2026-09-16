"""Ground truth: either supplied by the user, or inferred from the images.

Without a labelled dataset the tuner builds a *pseudo* ground truth: the union
of everything the probe configurations decoded. Barcode symbologies carry error
correction, so a decode that succeeds is almost always correct - the notable
exceptions are the 1D symbologies with weak or optional checksums, which are
only trusted when several independent configurations agree.

The truth set is allowed to grow during the search: if a later trial reads a
barcode nothing had read before, that is a genuine find and it is added. All
scores are computed lazily against the final truth set so every configuration
is judged on the same denominator.
"""
from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Set

from .config import canonical_format, is_weak_checksum
from .dataset import Dataset
from .engine import Detection, RunResult


@dataclass
class TruthEntry:
    text: str
    format: str
    confidence: int = 0
    agreements: int = 0
    first_seen: str = ""          # name of the run that first decoded it

    @property
    def ident(self) -> tuple:
        return (self.format, self.text)


@dataclass
class GroundTruth:
    """Expected detections per page key."""
    pages: Dict[str, Dict[tuple, TruthEntry]] = field(default_factory=dict)
    source: str = "inferred"
    provisional: Dict[str, Dict[tuple, TruthEntry]] = field(default_factory=dict)

    # -- queries -----------------------------------------------------------

    def expected(self, page_key: str) -> Set[tuple]:
        return set(self.pages.get(page_key, {}))

    def expected_texts(self, page_key: str) -> Set[str]:
        return {text for _fmt, text in self.pages.get(page_key, {})}

    @property
    def total(self) -> int:
        return sum(len(v) for v in self.pages.values())

    @property
    def pages_with_codes(self) -> int:
        return sum(1 for v in self.pages.values() if v)

    def formats(self) -> Set[str]:
        return {fmt for page in self.pages.values() for fmt, _text in page}

    # -- construction ------------------------------------------------------

    def absorb(self, run: RunResult, min_agree_risky: int = 2,
               min_weak_length: int = 4) -> int:
        """Fold one run's detections in. Returns how many entries were added."""
        added = 0
        for key, page in run.pages.items():
            confirmed = self.pages.setdefault(key, {})
            pending = self.provisional.setdefault(key, {})
            for detection in page.detections:
                ident = detection.ident
                if ident in confirmed:
                    confirmed[ident].agreements += 1
                    confirmed[ident].confidence = max(confirmed[ident].confidence,
                                                      detection.confidence)
                    continue
                entry = pending.get(ident)
                if entry is None:
                    entry = TruthEntry(text=detection.text, format=detection.format,
                                       confidence=detection.confidence,
                                       first_seen=run.name)
                    pending[ident] = entry
                entry.agreements += 1
                entry.confidence = max(entry.confidence, detection.confidence)
                if _trustworthy(entry, min_agree_risky, min_weak_length):
                    confirmed[ident] = entry
                    pending.pop(ident, None)
                    added += 1
        return added

    def unconfirmed(self) -> List[tuple]:
        """Weak-checksum reads that were never trusted enough to count."""
        out = []
        for key, pending in self.provisional.items():
            for entry in pending.values():
                out.append((key, entry))
        return sorted(out, key=lambda pair: (-pair[1].agreements, pair[0]))

    def to_json(self) -> Dict:
        return {
            "source": self.source,
            "total": self.total,
            "pages": {
                key: [{"format": e.format, "text": e.text, "confidence": e.confidence,
                       "agreements": e.agreements, "first_seen": e.first_seen}
                      for e in page.values()]
                for key, page in sorted(self.pages.items())
            },
        }


def _trustworthy(entry: TruthEntry, min_agree_risky: int, min_weak_length: int) -> bool:
    """Should this detection be believed without a label to check it against?

    Barcodes with real error correction are believed on a single sighting. The
    weak-checksum 1D symbologies are where inference goes wrong: laser etching,
    fabric weave and print texture all produce short spurious reads, and several
    configurations tend to make the *same* mistake, so agreement alone is not
    enough - a very short payload in one of those symbologies is rejected too.
    """
    if not is_weak_checksum(entry.format):
        return True
    if len(entry.text) < min_weak_length:
        return False
    return entry.agreements >= min_agree_risky


def infer(runs: Iterable[RunResult], min_agree_risky: int = 2,
          min_weak_length: int = 4) -> GroundTruth:
    truth = GroundTruth(source="inferred")
    for run in runs:
        truth.absorb(run, min_agree_risky=min_agree_risky,
                     min_weak_length=min_weak_length)
    return truth


def load(path: str, dataset: Dataset) -> GroundTruth:
    """Load user-supplied ground truth from CSV or JSON.

    CSV: ``file,text[,format]`` - one row per expected barcode, repeat the file
    name for multiple codes. ``file`` may be a path or a bare file name.
    JSON: ``{"IMG_001.jpg": ["ABC", "DEF"], ...}`` or a list of the CSV rows.
    """
    by_name: Dict[str, List[str]] = {}
    for sample in dataset:
        by_name.setdefault(os.path.basename(sample.path), []).append(sample.key)
        by_name.setdefault(sample.path, []).append(sample.key)

    rows: List[tuple] = []
    ext = os.path.splitext(path)[1].lower()
    if ext == ".json":
        data = json.load(open(path, encoding="utf-8"))
        if isinstance(data, dict):
            for name, values in data.items():
                if isinstance(values, str):
                    values = [values]
                for value in values:
                    if isinstance(value, dict):
                        rows.append((name, value.get("text", ""), value.get("format", "")))
                    else:
                        rows.append((name, value, ""))
        else:
            for row in data:
                rows.append((row.get("file"), row.get("text", ""), row.get("format", "")))
    else:
        with open(path, newline="", encoding="utf-8-sig") as handle:
            reader = csv.reader(handle)
            header = next(reader, None)
            if header and header[0].strip().lower() not in ("file", "filename", "image", "path"):
                handle.seek(0)
                reader = csv.reader(handle)
            for row in reader:
                if not row or not row[0].strip():
                    continue
                rows.append((row[0].strip(), row[1].strip() if len(row) > 1 else "",
                             row[2].strip() if len(row) > 2 else ""))

    truth = GroundTruth(source=f"file:{os.path.basename(path)}")
    for key in (s.key for s in dataset):
        truth.pages.setdefault(key, {})
    unmatched: List[str] = []
    for name, text, fmt in rows:
        if not text:
            continue
        keys = by_name.get(name) or by_name.get(os.path.basename(str(name)))
        if not keys:
            unmatched.append(str(name))
            continue
        for key in keys:
            entry = TruthEntry(text=text, format=canonical_format(fmt) if fmt else "*",
                               agreements=1, first_seen="file")
            truth.pages.setdefault(key, {})[entry.ident] = entry
    if unmatched:
        truth.source += f" ({len(set(unmatched))} unmatched rows)"
    return truth


def match(detections: List[Detection], expected: Dict[tuple, TruthEntry]) -> tuple:
    """Return (matched idents, extra detections).

    A ground-truth format of ``*`` (user-supplied without a format column)
    matches on text alone.
    """
    wildcard = {text for fmt, text in expected if fmt == "*"}
    exact = set(expected)
    matched, extra = set(), []
    for detection in detections:
        if detection.ident in exact:
            matched.add(detection.ident)
        elif detection.text in wildcard:
            matched.add(("*", detection.text))
        else:
            extra.append(detection)
    return matched, extra
