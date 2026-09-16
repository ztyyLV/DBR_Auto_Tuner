"""Tests for the parts that do not need the SDK: template building, truth
inference, scoring and candidate comparison.

    python -m unittest discover -s tests -v
"""
from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dbr_autotune import template as T, truth as gt                       # noqa: E402
from dbr_autotune.config import canonical_format, is_weak_checksum        # noqa: E402
from dbr_autotune.dataset import Dataset, Sample                          # noqa: E402
from dbr_autotune.engine import Detection, PageResult, RunResult          # noqa: E402
from dbr_autotune.scoring import Objective, pareto_front, score           # noqa: E402


def make_run(name: str, per_page: dict, ms: float = 100.0) -> RunResult:
    """per_page: {page label: [(format, text) or (format, text, confidence)]}"""
    run = RunResult(name=name, fingerprint=name)
    for label, detections in per_page.items():
        sample = Sample(label)
        page = PageResult(sample, ms=ms)
        for item in detections:
            fmt, text = item[0], item[1]
            confidence = item[2] if len(item) > 2 else 90
            page.detections.append(Detection(text=text, format=fmt, confidence=confidence))
        run.pages[sample.key] = page
    return run


class TemplateBuilding(unittest.TestCase):
    def test_defaults_round_trip(self):
        doc = T.build(T.normalize({}), "X")
        self.assertEqual(doc["CaptureVisionTemplates"][0]["Name"], "X")
        task = doc["BarcodeReaderTaskSettingOptions"][0]
        self.assertEqual(task["BarcodeFormatIds"], ["BF_DEFAULT"])
        sections = [s["Section"] for s in task["SectionArray"]]
        self.assertEqual(sections, ["ST_REGION_PREDETECTION", "ST_BARCODE_LOCALIZATION",
                                    "ST_BARCODE_DECODING"])

    def test_unknown_knob_rejected(self):
        with self.assertRaises(KeyError):
            T.normalize({"not_a_knob": 1})
        with self.assertRaises(KeyError):
            T.with_knobs(T.normalize({}), nope=1)

    def test_fingerprint_ignores_name_but_not_knobs(self):
        a = T.normalize({"formats": ["BF_QR_CODE"]})
        b = T.normalize({"formats": ["BF_QR_CODE"]})
        c = T.normalize({"formats": ["BF_AZTEC"]})
        self.assertEqual(T.fingerprint(a), T.fingerprint(b))
        self.assertNotEqual(T.fingerprint(a), T.fingerprint(c))

    def test_disabled_stages_are_omitted(self):
        knobs = T.normalize({"region_predetect": False, "resist_deformation": False,
                             "complement_barcode": False, "scale_barcode_image": False})
        task = T.build(knobs)["BarcodeReaderTaskSettingOptions"][0]
        by_section = {s["Section"]: s for s in task["SectionArray"]}
        self.assertEqual(by_section["ST_REGION_PREDETECTION"]["StageArray"], [])
        decoding = [s["Stage"] for s in by_section["ST_BARCODE_DECODING"]["StageArray"]]
        self.assertEqual(decoding, ["SST_DECODE_BARCODES"])

    def test_binarization_covers_both_binary_surfaces(self):
        knobs = T.normalize({"binarization": {"block": 39, "fill": 0, "compensation": -10}})
        stages = {s["Stage"]: s for s in T.build(knobs)["ImageParameterOptions"][0]["ApplicableStages"]}
        self.assertIn("SST_BINARIZE_IMAGE", stages)
        self.assertIn("SST_BINARIZE_TEXTURE_REMOVED_GRAYSCALE", stages)
        mode = stages["SST_BINARIZE_IMAGE"]["BinarizationModes"][0]
        self.assertEqual((mode["BlockSizeX"], mode["BlockSizeY"]), (39, 39))
        self.assertEqual(mode["ThresholdCompensation"], -10)

    def test_format_spec_only_when_needed(self):
        self.assertNotIn("BarcodeFormatSpecificationOptions", T.build(T.normalize({})))
        doc = T.build(T.normalize({"mirror_mode": "MM_BOTH"}))
        self.assertIn("BarcodeFormatSpecificationOptions", doc)
        self.assertIn("BarcodeFormatSpecificationNameArray",
                      doc["BarcodeReaderTaskSettingOptions"][0])

    def test_multi_variant_names_are_unique(self):
        doc = T.build_multi({
            "A": T.normalize({"formats": ["BF_QR_CODE"], "mirror_mode": "MM_BOTH"}),
            "B": T.normalize({"formats": ["BF_AZTEC"], "mirror_mode": "MM_BOTH"}),
        })
        self.assertEqual([t["Name"] for t in doc["CaptureVisionTemplates"]], ["A", "B"])
        for section in ("TargetROIDefOptions", "BarcodeReaderTaskSettingOptions",
                        "ImageParameterOptions", "BarcodeFormatSpecificationOptions"):
            names = [o["Name"] for o in doc[section]]
            self.assertEqual(len(names), len(set(names)), f"duplicate names in {section}")
        # Each template must reference its own options, not the other variant's.
        roi_a = doc["CaptureVisionTemplates"][0]["ImageROIProcessingNameArray"][0]
        roi_b = doc["CaptureVisionTemplates"][1]["ImageROIProcessingNameArray"][0]
        self.assertNotEqual(roi_a, roi_b)

    def test_describe_reports_only_changes(self):
        knobs = T.normalize({"expected_count": 1, "dpm_modes": ["DPMCRM_GENERAL"]})
        self.assertEqual(set(T.describe(knobs)), {"expected_count", "dpm_modes"})


class FormatNaming(unittest.TestCase):
    def test_result_strings_gain_the_bf_prefix(self):
        self.assertEqual(canonical_format("DATAMATRIX"), "BF_DATAMATRIX")
        self.assertEqual(canonical_format("BF_DATAMATRIX"), "BF_DATAMATRIX")

    def test_weak_checksum_detection_survives_the_prefix_gap(self):
        self.assertTrue(is_weak_checksum("PHARMACODE_ONE_TRACK"))
        self.assertTrue(is_weak_checksum("BF_CODE_39"))
        self.assertFalse(is_weak_checksum("DATAMATRIX"))
        self.assertFalse(is_weak_checksum("QR_CODE"))


class TruthInference(unittest.TestCase):
    def test_error_corrected_read_is_believed_immediately(self):
        truth = gt.infer([make_run("r1", {"a.jpg": [("BF_DATAMATRIX", "PAYLOAD123")]})])
        self.assertEqual(truth.total, 1)

    def test_short_weak_checksum_read_is_withheld_even_when_corroborated(self):
        runs = [make_run(f"r{i}", {"a.jpg": [("BF_PHARMACODE_TWO_TRACK", "27")]})
                for i in range(4)]
        truth = gt.infer(runs)
        self.assertEqual(truth.total, 0)
        self.assertEqual(len(truth.unconfirmed()), 1)

    def test_long_weak_checksum_read_needs_agreement(self):
        one = gt.infer([make_run("r1", {"a.jpg": [("BF_CODE_39", "ABC12345")]})])
        self.assertEqual(one.total, 0)
        two = gt.infer([make_run("r1", {"a.jpg": [("BF_CODE_39", "ABC12345")]}),
                        make_run("r2", {"a.jpg": [("BF_CODE_39", "ABC12345")]})])
        self.assertEqual(two.total, 1)

    def test_truth_grows_across_runs(self):
        truth = gt.infer([make_run("r1", {"a.jpg": [("BF_QR_CODE", "one")]})])
        self.assertEqual(truth.total, 1)
        added = truth.absorb(make_run("r2", {"a.jpg": [("BF_QR_CODE", "one"),
                                                       ("BF_QR_CODE", "two")]}))
        self.assertEqual(added, 1)
        self.assertEqual(truth.total, 2)

    def test_formats_are_template_spellings(self):
        run = make_run("r1", {"a.jpg": [("BF_DATAMATRIX", "X" * 10)]})
        self.assertEqual(gt.infer([run]).formats(), {"BF_DATAMATRIX"})


class GroundTruthFile(unittest.TestCase):
    def test_csv_matches_on_basename_and_wildcard_format(self):
        import tempfile
        data = Dataset([Sample(os.path.join("some", "dir", "a.jpg"))])
        with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False,
                                         encoding="utf-8", newline="") as handle:
            handle.write("file,text,format\na.jpg,HELLO,\n")
            path = handle.name
        try:
            truth = gt.load(path, data)
        finally:
            os.unlink(path)
        self.assertEqual(truth.total, 1)
        key = data.samples[0].key
        detections = [Detection(text="HELLO", format="BF_DATAMATRIX")]
        matched, extra = gt.match(detections, truth.pages[key])
        self.assertEqual(len(matched), 1)      # wildcard format matches on text
        self.assertEqual(extra, [])


class Scoring(unittest.TestCase):
    def setUp(self):
        self.truth = gt.infer([make_run("probe", {
            "a.jpg": [("BF_QR_CODE", "one")],
            "b.jpg": [("BF_QR_CODE", "two"), ("BF_QR_CODE", "three")],
            "c.jpg": [],
        })])

    def test_recall_counts_codes_not_pages(self):
        s = score(make_run("x", {"a.jpg": [("BF_QR_CODE", "one")],
                                 "b.jpg": [("BF_QR_CODE", "two")],
                                 "c.jpg": []}), self.truth)
        self.assertEqual((s.found, s.expected), (2, 3))
        self.assertAlmostEqual(s.recall, 2 / 3)
        self.assertEqual(s.pages_complete, 1)     # only a.jpg is fully read
        self.assertEqual(s.pages_partial, 2)
        self.assertEqual(s.pages_expected, 2)     # c.jpg has no codes to find

    def test_unexpected_detection_counts_as_extra(self):
        s = score(make_run("x", {"a.jpg": [("BF_QR_CODE", "one"), ("BF_CODE_39", "9")],
                                 "b.jpg": [], "c.jpg": []}), self.truth)
        self.assertEqual(s.extra, 1)

    def test_timing_percentiles(self):
        run = RunResult(name="t", fingerprint="t")
        for index, ms in enumerate([10, 20, 30, 40, 1000]):
            sample = Sample(f"{index}.jpg")
            run.pages[sample.key] = PageResult(sample, ms=ms)
        s = score(run, gt.GroundTruth())
        self.assertEqual(s.max_ms, 1000)
        self.assertAlmostEqual(s.mean_ms, 220.0)


class Comparison(unittest.TestCase):
    def make(self, recall: float, ms: float, extra: int = 0):
        run = make_run(f"{recall}@{ms}", {"a.jpg": []}, ms=ms)
        s = score(run, gt.GroundTruth())
        s.found, s.expected, s.extra = int(recall * 100), 100, extra
        s.mean_ms = ms
        return s

    def test_recall_first_objective(self):
        o = Objective(speed_weight=0.0)
        self.assertTrue(o.better(self.make(0.9, 5000), self.make(0.8, 10)))
        self.assertFalse(o.better(self.make(0.8, 10), self.make(0.9, 5000)))

    def test_speed_breaks_ties_but_needs_a_real_margin(self):
        o = Objective(speed_weight=0.0)
        self.assertTrue(o.better(self.make(0.9, 500), self.make(0.9, 1000)))
        self.assertFalse(o.better(self.make(0.9, 995), self.make(0.9, 1000)))

    def test_fewer_extras_wins_at_equal_recall(self):
        o = Objective(speed_weight=0.0)
        self.assertTrue(o.better(self.make(0.9, 1000, extra=0),
                                 self.make(0.9, 100, extra=3)))

    def test_speed_weight_allows_trading_recall(self):
        fast = self.make(0.85, 100)
        slow = self.make(0.90, 3200)      # 5 doublings slower for 5 recall points
        self.assertFalse(Objective(speed_weight=0.0).better(fast, slow))
        self.assertTrue(Objective(speed_weight=1.0).better(fast, slow))

    def test_pareto_front_drops_dominated_points(self):
        front = pareto_front([self.make(0.9, 100), self.make(0.9, 500),
                              self.make(0.8, 50), self.make(0.5, 900)])
        self.assertEqual([(round(s.recall, 2), s.mean_ms) for s in front],
                         [(0.9, 100), (0.8, 50)])


class DatasetSplit(unittest.TestCase):
    def test_split_is_deterministic_and_disjoint(self):
        data = Dataset([Sample(f"{i}.jpg") for i in range(10)])
        tune_a, hold_a = data.split(0.3)
        tune_b, hold_b = data.split(0.3)
        self.assertEqual([s.key for s in hold_a], [s.key for s in hold_b])
        self.assertEqual(len(hold_a), 3)
        self.assertEqual(len(tune_a), 7)
        self.assertEqual(set(s.key for s in tune_a) & set(s.key for s in hold_a), set())

    def test_split_refuses_to_empty_either_side(self):
        data = Dataset([Sample("only.jpg")])
        tune, hold = data.split(0.5)
        self.assertEqual(len(tune), 1)
        self.assertEqual(len(hold), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)


class Reproducibility(unittest.TestCase):
    def test_request_survives_a_json_round_trip(self):
        from dbr_autotune.api import TuneRequest
        original = TuneRequest(images=["a", "b"], rounds=3, formats=["BF_QR_CODE"],
                               seed=7, speed_weight=0.5)
        restored = TuneRequest.from_json(original.to_json())
        self.assertEqual(restored.to_json(), original.to_json())

    def test_unknown_fields_are_ignored_not_fatal(self):
        from dbr_autotune.api import TuneRequest
        restored = TuneRequest.from_json({"images": ["x"], "from_a_future_version": 1})
        self.assertEqual(restored.images, ["x"])

    def test_fingerprint_has_the_fields_that_move_results(self):
        from dbr_autotune.repro import environment_fingerprint
        fingerprint = environment_fingerprint()
        for key in ("autotune", "dbr_bundle", "python", "platform", "cpu_count"):
            self.assertIn(key, fingerprint)

    def test_environment_drift_is_described(self):
        from dbr_autotune.repro import compare_environments, describe_drift
        diff = compare_environments({"dbr_bundle": "11.2", "python": "3.12.2"},
                                    {"dbr_bundle": "11.6", "python": "3.12.2"})
        self.assertEqual(set(diff), {"dbr_bundle"})
        self.assertIn("SDK version", describe_drift(diff))
        self.assertEqual(describe_drift({}), "environments match")


class SplitsAvoidLeakage(unittest.TestCase):
    """Pages of one document must never straddle a split."""

    def _mixed(self) -> Dataset:
        samples = [Sample(f"img{i}.jpg") for i in range(6)]
        samples += [Sample("doc.pdf", p, multipage=True) for p in range(4)]
        return Dataset(samples)

    def test_holdout_moves_whole_files(self):
        tune, hold = self._mixed().split(0.3)
        self.assertTrue(len(tune) and len(hold))
        self.assertEqual({s.path for s in tune} & {s.path for s in hold}, set())

    def test_folds_move_whole_files(self):
        from dbr_autotune.crossval import _folds
        folds = _folds(self._mixed(), 3, seed=0)
        self.assertEqual(sum(len(f) for f in folds), 10)
        owner = {}
        for index, fold in enumerate(folds):
            for sample in fold.samples:
                self.assertEqual(owner.setdefault(sample.path, index), index,
                                 f"{sample.path} landed in two folds")

    def test_folds_are_deterministic_for_a_seed(self):
        from dbr_autotune.crossval import _folds
        a = [[s.key for s in f.samples] for f in _folds(self._mixed(), 3, seed=42)]
        b = [[s.key for s in f.samples] for f in _folds(self._mixed(), 3, seed=42)]
        self.assertEqual(a, b)


class CrossValidationAggregates(unittest.TestCase):
    def _cv(self, coverages, knob_values):
        from dbr_autotune.crossval import CrossValidation, FoldResult
        cv = CrossValidation(k=len(coverages), pages=20, seed=0)
        for index, (coverage, knob) in enumerate(zip(coverages, knob_values), start=1):
            train = score(make_run("t", {"a.jpg": []}), gt.GroundTruth())
            validation = score(make_run("v", {"a.jpg": []}), gt.GroundTruth())
            validation.pages_partial, validation.pages_total = int(coverage * 10), 10
            cv.folds.append(FoldResult(index=index, tuned_on=16, validated_on=4,
                                       knobs=T.normalize({"expected_count": knob}),
                                       train=train, validation=validation))
        return cv

    def test_mean_and_spread(self):
        cv = self._cv([0.8, 0.9, 1.0], [1, 1, 1])
        self.assertAlmostEqual(cv.mean("page_coverage"), 0.9)
        self.assertEqual(cv.spread("page_coverage"), (0.8, 1.0))
        self.assertGreater(cv.stdev("page_coverage"), 0)

    def test_agreeing_folds_report_no_unstable_knobs(self):
        cv = self._cv([0.9, 0.9, 0.9], [1, 1, 1])
        unstable = [k for k, c in cv.knob_stability().items() if len(c) > 1]
        self.assertEqual(unstable, [])
        self.assertEqual(cv.consensus_knobs()["expected_count"], 1)

    def test_disagreeing_folds_are_flagged(self):
        cv = self._cv([0.9, 0.5, 0.9], [1, 2, 3])
        unstable = [k for k, c in cv.knob_stability().items() if len(c) > 1]
        self.assertIn("expected_count", unstable)
        self.assertNotIn("expected_count", cv.consensus_knobs())
        self.assertIn("unstable knobs", "\n".join(cv.summary_lines()))


class UndeterminedPages(unittest.TestCase):
    def test_pages_nothing_read_leave_the_recall_denominator(self):
        truth = gt.infer([make_run("probe", {"a.jpg": [("BF_QR_CODE", "one")],
                                             "b.jpg": []})])
        s = score(make_run("x", {"a.jpg": [("BF_QR_CODE", "one")], "b.jpg": []}), truth)
        self.assertEqual(s.recall, 1.0)              # flattering
        self.assertEqual(s.pages_undetermined, 1)
        self.assertEqual(s.page_coverage, 0.5)       # honest
        self.assertEqual(s.recall_floor, 0.5)        # worst case if b.jpg has a code
