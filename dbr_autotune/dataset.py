"""Image discovery and the unit of evaluation.

A dataset is a list of :class:`Sample`. One sample is one *page*: a plain image
file is a single sample, while a PDF or multi-page TIFF expands into one sample
per page, so recall is measured per page rather than per file.
"""
from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from typing import Iterable, List

from .config import IMAGE_EXT, MULTI_PAGE_EXT


@dataclass(frozen=True)
class Sample:
    path: str
    page: int = 0            # 0-based; always 0 for single-page formats
    multipage: bool = False

    @property
    def key(self) -> str:
        return f"{self.path}#{self.page}" if self.multipage else self.path

    @property
    def label(self) -> str:
        base = os.path.basename(self.path)
        return f"{base} p{self.page + 1}" if self.multipage else base


@dataclass
class Dataset:
    samples: List[Sample] = field(default_factory=list)
    skipped: List[str] = field(default_factory=list)
    uncounted: List[str] = field(default_factory=list)   # page count unknown

    def __len__(self) -> int:
        return len(self.samples)

    def __iter__(self) -> Iterable[Sample]:
        return iter(self.samples)

    def split(self, holdout_fraction: float, seed: int = 0) -> tuple["Dataset", "Dataset"]:
        """Deterministic tune/holdout split, **by file**. Returns (tune, holdout).

        Whole files move together: the pages of one PDF or TIFF are usually near
        duplicates, so letting them straddle the split would leak the tuning set
        into the validation set.
        """
        by_file: Dict[str, List[Sample]] = {}
        for sample in self.samples:
            by_file.setdefault(sample.path, []).append(sample)

        target = len(self.samples) * holdout_fraction
        if target < 1 or len(by_file) < 2:
            return Dataset(list(self.samples)), Dataset([])

        paths = sorted(by_file)
        random.Random(seed).shuffle(paths)
        holdout: List[Sample] = []
        taken: set = set()
        for path in paths:
            if len(holdout) >= target:
                break
            # Never take the last file - the tuning set must stay non-empty.
            if len(by_file) - len(taken) <= 1:
                break
            holdout.extend(by_file[path])
            taken.add(path)
        tune = [s for s in self.samples if s.path not in taken]
        if not tune or not holdout:
            return Dataset(list(self.samples)), Dataset([])
        return (Dataset(sorted(tune, key=lambda s: s.key)),
                Dataset(sorted(holdout, key=lambda s: s.key)))


def _pdf_page_count(path: str) -> int | None:
    """Page count via whichever PDF library is installed, else None."""
    try:
        from pypdf import PdfReader
        return max(1, len(PdfReader(path).pages))
    except ImportError:
        pass
    except Exception:
        return None
    try:
        import fitz                                   # PyMuPDF
        with fitz.open(path) as doc:
            return max(1, doc.page_count)
    except ImportError:
        pass
    except Exception:
        return None
    try:
        import pikepdf
        with pikepdf.open(path) as doc:
            return max(1, len(doc.pages))
    except ImportError:
        return None
    except Exception:
        return None


def _page_count(path: str) -> tuple[int, bool]:
    """(pages, counted). ``counted`` is False when it had to be guessed as 1."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        pages = _pdf_page_count(path)
        return (1, False) if pages is None else (pages, True)
    try:
        from PIL import Image
        with Image.open(path) as im:
            return max(1, getattr(im, "n_frames", 1)), True
    except Exception:
        return 1, False


def discover(roots: Iterable[str], recursive: bool = True,
             expand_pages: bool = True, limit: int | None = None) -> Dataset:
    """Collect every supported image under ``roots``."""
    files: List[str] = []
    for root in roots:
        if os.path.isfile(root):
            files.append(root)
            continue
        if not os.path.isdir(root):
            raise FileNotFoundError(root)
        if recursive:
            for dirpath, _, names in os.walk(root):
                files.extend(os.path.join(dirpath, n) for n in names)
        else:
            files.extend(os.path.join(root, n) for n in os.listdir(root))

    dataset = Dataset()
    for path in sorted(set(files)):
        ext = os.path.splitext(path)[1].lower()
        if ext not in IMAGE_EXT:
            if os.path.isfile(path):
                dataset.skipped.append(path)
            continue
        if ext in MULTI_PAGE_EXT:
            if expand_pages:
                pages, counted = _page_count(path)
                if not counted:
                    dataset.uncounted.append(path)
            else:
                pages = 1
            for page in range(pages):
                dataset.samples.append(Sample(path, page, multipage=True))
        else:
            dataset.samples.append(Sample(path))

    if limit is not None and len(dataset.samples) > limit:
        step = len(dataset.samples) / limit
        picked = [dataset.samples[int(i * step)] for i in range(limit)]
        dataset.samples = picked
    return dataset
