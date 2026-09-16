"""dbr-autotune - derive a high read-rate, low-latency Dynamsoft Barcode Reader
11.x CaptureVision template from a folder of images.

The emitted template is plain DCV template JSON and is usable unchanged from the
C++, .NET, Java, Python and mobile editions of the SDK.

Programmatic use::

    from dbr_autotune import TuneRequest, tune

    outcome = tune(TuneRequest(images=["./photos"], out="./out"))
    print(outcome.final.page_coverage)
    template = outcome.templates()["AutoTuned_MaxRecall"]

Cross-validating instead of trusting one split::

    from dbr_autotune import TuneRequest, cross_validate

    cv = cross_validate(TuneRequest(images=["./photos"]), k=5)
    print(cv.mean("page_coverage"), "+/-", cv.stdev("page_coverage"))
"""

__version__ = "1.1.0"

__all__ = [
    "__version__",
    # core entry points
    "TuneRequest", "TuneOutcome", "tune", "evaluate", "cross_validate",
    # results
    "Score", "CrossValidation", "FoldResult",
    # building blocks, for anyone assembling their own pipeline
    "Dataset", "Sample", "discover", "Engine", "GroundTruth",
    "build_template", "template_json", "describe_knobs",
    "environment_fingerprint",
]


def __getattr__(name):
    # Imported lazily so `import autotune` stays cheap and does not require the
    # SDK to be installed just to read __version__.
    if name in ("TuneRequest", "TuneOutcome", "tune", "evaluate"):
        from . import api
        return getattr(api, name)
    if name in ("cross_validate", "CrossValidation", "FoldResult"):
        from . import crossval
        return getattr(crossval, name)
    if name == "Score":
        from .scoring import Score
        return Score
    if name in ("Dataset", "Sample", "discover"):
        from . import dataset
        return getattr(dataset, name)
    if name == "Engine":
        from .engine import Engine
        return Engine
    if name == "GroundTruth":
        from .truth import GroundTruth
        return GroundTruth
    if name in ("build_template", "template_json", "describe_knobs"):
        from . import template
        return {"build_template": template.build,
                "template_json": template.to_json,
                "describe_knobs": template.describe}[name]
    if name == "environment_fingerprint":
        from .repro import environment_fingerprint
        return environment_fingerprint
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
