"""Running a template over a dataset and collecting per-page results.

One :class:`Engine` owns a pool of thread-local ``CaptureVisionRouter``
instances - the SDK router is not safe to share across threads, but one router
per worker is. Results of identical templates are memoised by fingerprint so the
search never pays twice for the same configuration.
"""
from __future__ import annotations

import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List

from . import template as T
from .config import canonical_format
from .dataset import Dataset, Sample

_EC_OK = 0
_EC_JSON_KEY_WARNING = -10077


@dataclass
class Detection:
    text: str
    format: str
    confidence: int = 0
    module_size: int = 0
    is_dpm: bool = False
    is_mirrored: bool = False
    location: List[List[int]] = field(default_factory=list)

    @property
    def ident(self) -> tuple:
        """Identity used for matching against ground truth."""
        return (self.format, self.text)


@dataclass
class PageResult:
    sample: Sample
    detections: List[Detection] = field(default_factory=list)
    ms: float = 0.0
    error: str = ""

    @property
    def texts(self) -> set:
        return {d.text for d in self.detections}

    @property
    def idents(self) -> set:
        return {d.ident for d in self.detections}


@dataclass
class RunResult:
    name: str
    fingerprint: str
    pages: Dict[str, PageResult] = field(default_factory=dict)
    wall_ms: float = 0.0
    serial: bool = False

    @property
    def total_ms(self) -> float:
        return sum(p.ms for p in self.pages.values())

    @property
    def mean_ms(self) -> float:
        return self.total_ms / len(self.pages) if self.pages else 0.0

    def percentile_ms(self, q: float) -> float:
        if not self.pages:
            return 0.0
        values = sorted(p.ms for p in self.pages.values())
        idx = min(len(values) - 1, max(0, int(round(q * (len(values) - 1)))))
        return values[idx]

    @property
    def errors(self) -> List[str]:
        return [f"{p.sample.label}: {p.error}" for p in self.pages.values() if p.error]


class TemplateRejected(RuntimeError):
    """The SDK refused the generated template JSON."""


class Engine:
    """Executes templates against a dataset."""

    def __init__(self, dataset: Dataset, license_key: str, jobs: int = 1,
                 progress: Callable[[str], None] | None = None):
        self.dataset = dataset
        self.license_key = license_key
        self.jobs = max(1, jobs)
        self.progress = progress or (lambda _msg: None)
        self._cache: Dict[str, RunResult] = {}
        self._local = threading.local()
        self._trials = 0

        from dynamsoft_capture_vision_bundle import LicenseManager
        code, message = LicenseManager.init_license(license_key)
        if code not in (_EC_OK, _EC_JSON_KEY_WARNING):
            raise RuntimeError(f"license initialisation failed ({code}): {message}")

        # Group pages by file so multi-page documents are decoded in one call.
        self._by_file: Dict[str, List[Sample]] = {}
        for sample in dataset:
            self._by_file.setdefault(sample.path, []).append(sample)
        for samples in self._by_file.values():
            samples.sort(key=lambda s: s.page)

    # -- router pool -------------------------------------------------------

    def _router(self):
        router = getattr(self._local, "router", None)
        if router is None:
            from dynamsoft_capture_vision_bundle import CaptureVisionRouter
            router = CaptureVisionRouter()
            self._local.router = router
            self._local.loaded = None
        return router

    def _prepare(self, template_json: str, key: str):
        router = self._router()
        if getattr(self._local, "loaded", None) != key:
            code, message = router.init_settings(template_json)
            if code not in (_EC_OK, _EC_JSON_KEY_WARNING):
                raise TemplateRejected(f"({code}) {message}")
            self._local.loaded = key
        return router

    # -- execution ---------------------------------------------------------

    @property
    def trials(self) -> int:
        return self._trials

    def run(self, knobs: Dict[str, Any], name: str = "AutoTuned",
            serial: bool = False, use_cache: bool = True) -> RunResult:
        """Decode the whole dataset with ``knobs`` and return timings + results."""
        return self._execute(T.to_json(knobs, name), T.fingerprint(knobs),
                             name, serial, use_cache)

    def run_document(self, document: Dict[str, Any], name: str | None = None,
                     serial: bool = True, use_cache: bool = True) -> RunResult:
        """Decode with a template document that did not come from a knob vector.

        Used to score templates written by hand or produced by an earlier
        version of the tuner, which is what makes cross-checking possible.
        """
        import hashlib
        if name is None:
            name = document["CaptureVisionTemplates"][0]["Name"]
        payload = json.dumps(document, sort_keys=True, ensure_ascii=False)
        fingerprint = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]
        return self._execute(json.dumps(document, ensure_ascii=False), fingerprint,
                             name, serial, use_cache)

    def _execute(self, template_json: str, fingerprint: str, name: str,
                 serial: bool, use_cache: bool) -> RunResult:
        cache_key = fingerprint + ("|serial" if serial else "")
        if use_cache and cache_key in self._cache:
            return self._cache[cache_key]

        load_key = f"{fingerprint}|{name}"
        # Validate once up front so a bad template fails fast, not per worker.
        self._prepare(template_json, load_key)

        result = RunResult(name=name, fingerprint=fingerprint, serial=serial)
        started = time.perf_counter()
        paths = list(self._by_file)

        def work(path: str) -> List[PageResult]:
            return self._decode_file(path, template_json, load_key, name)

        if serial or self.jobs == 1:
            for path in paths:
                for page in work(path):
                    result.pages[page.sample.key] = page
        else:
            with ThreadPoolExecutor(max_workers=self.jobs) as pool:
                for pages in pool.map(work, paths):
                    for page in pages:
                        result.pages[page.sample.key] = page

        result.wall_ms = (time.perf_counter() - started) * 1000.0
        self._trials += 1
        if use_cache:
            self._cache[cache_key] = result
        return result

    def _decode_file(self, path: str, template_json: str, load_key: str,
                     name: str) -> List[PageResult]:
        samples = self._by_file[path]
        multipage = samples[0].multipage
        try:
            router = self._prepare(template_json, load_key)
        except TemplateRejected as exc:
            return [PageResult(s, error=str(exc)) for s in samples]

        started = time.perf_counter()
        try:
            if multipage:
                array = router.capture_multi_pages(path, name)
                raw = list(array.get_results()) if array else []
            else:
                raw = [router.capture(path, name)]
        except Exception as exc:                        # SDK / IO failure
            elapsed = (time.perf_counter() - started) * 1000.0
            share = elapsed / len(samples)
            return [PageResult(s, ms=share, error=f"{type(exc).__name__}: {exc}") for s in samples]
        elapsed = (time.perf_counter() - started) * 1000.0

        by_page: Dict[int, Any] = {}
        for index, captured in enumerate(raw):
            if captured is None:
                continue
            by_page[self._page_number(captured, index)] = captured

        share = elapsed / len(samples)
        pages: List[PageResult] = []
        for sample in samples:
            page = PageResult(sample, ms=share)
            captured = by_page.get(sample.page)
            if captured is not None:
                code = captured.get_error_code()
                if code not in (_EC_OK, _EC_JSON_KEY_WARNING):
                    page.error = f"({code}) {captured.get_error_string()}"
                page.detections = self._detections(captured)
            pages.append(page)
        return pages

    @staticmethod
    def _page_number(captured, fallback: int) -> int:
        try:
            tag = captured.get_original_image_tag()
            number = tag.get_page_number()
            if number is not None and number >= 0:
                return number
        except Exception:
            pass
        return fallback

    @staticmethod
    def _detections(captured) -> List[Detection]:
        out: List[Detection] = []
        items = captured.get_items() or []
        for item in items:
            try:
                text = item.get_text()
            except Exception:
                continue
            if text is None:
                continue
            detection = Detection(
                text=text,
                format=canonical_format(_safe(item, "get_format_string", "")))
            detection.confidence = _safe(item, "get_confidence", 0) or 0
            detection.module_size = _safe(item, "get_module_size", 0) or 0
            detection.is_dpm = bool(_safe(item, "is_dpm", False))
            detection.is_mirrored = bool(_safe(item, "is_mirrored", False))
            try:
                points = item.get_location().points
                detection.location = [
                    [int(p.x), int(p.y)] if hasattr(p, "x") else [int(p[0]), int(p[1])]
                    for p in points
                ]
            except Exception:
                pass
            out.append(detection)
        return out


def _safe(obj, method: str, default):
    try:
        return getattr(obj, method)()
    except Exception:
        return default


def image_stats(dataset: Dataset, sample_limit: int = 12) -> Dict[str, Any]:
    """Cheap profile of the dataset: resolution, colour, file size."""
    from PIL import Image

    widths, heights, modes, sizes = [], [], [], []
    seen = 0
    for sample in dataset:
        if seen >= sample_limit:
            break
        ext = os.path.splitext(sample.path)[1].lower()
        if ext == ".pdf":
            continue
        try:
            with Image.open(sample.path) as im:
                widths.append(im.width)
                heights.append(im.height)
                modes.append(im.mode)
            sizes.append(os.path.getsize(sample.path))
            seen += 1
        except Exception:
            continue
    if not widths:
        return {}
    return {
        "sampled": seen,
        "min_short_edge": min(min(w, h) for w, h in zip(widths, heights)),
        "median_short_edge": sorted(min(w, h) for w, h in zip(widths, heights))[len(widths) // 2],
        "max_long_edge": max(max(w, h) for w, h in zip(widths, heights)),
        "megapixels": round(sum(w * h for w, h in zip(widths, heights)) / len(widths) / 1e6, 2),
        "colour": sorted(set(modes)),
        "mean_file_kb": round(sum(sizes) / len(sizes) / 1024) if sizes else 0,
    }
