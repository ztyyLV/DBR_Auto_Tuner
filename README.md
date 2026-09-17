# dbr-autotune

Point it at a folder of images and it produces a **Dynamsoft Barcode Reader 11.x
CaptureVision template** tuned for those images — high read rate first, then as
fast as it can be made without giving any of that read rate back.

The emitted file is ordinary DCV template JSON. It is not Python-specific: load
it from the **C++**, .NET, Java, Python or mobile edition of the SDK with
`InitSettingsFromFile` and select the template by name.

## Install

**Not on PyPI** — install it from this repository:

```bash
git clone https://github.com/ztyyLV/DBR_Auto_Tuner
cd DBR_Auto_Tuner
pip install -e ".[pdf]"
```

That registers a `dbr-autotune` command you can run from anywhere:

```bash
dbr-autotune ui
```

Prefer not to install anything? Everything also works straight out of the
clone, with `python -m` instead of the command name:

```bash
python -m dbr_autotune ui
python -m dbr_autotune ./images -o out
```

The two are interchangeable; the rest of this README writes `dbr-autotune` for
brevity.

### The SDK dependency

Dynamsoft ships the same Python API under two distribution names, versioned
independently:

| Package | Version line | Contents |
|---|---|---|
| `dynamsoft-barcode-reader-bundle` | 11.x | barcode only — what this installs |
| `dynamsoft-capture-vision-bundle` | 3.x | the full suite, also fine |

**Either one satisfies this tool** — it takes whichever it finds, so if you
already have the full bundle you will not be made to install a second copy of
the same native libraries. The version recorded in `results.json` says which
one ran.

---

## Three ways in

### Web UI — `dbr-autotune ui`

Opens a local page: browse to a folder, press Start, watch every configuration as
it is tried, read the result. No flags to learn.

The picker shows how many images sit in each subfolder, so you can see where your
pictures are without opening each one, and individual files can be selected as
well as whole folders. Every control carries a one-line explanation of what it
does and why you would change it, and the page ends with a glossary of every term
it uses — read rate, page coverage, undetermined, p95, DPM, ground truth, holdout
— written for someone who has never used a barcode SDK.

Loopback-only and token-guarded (see [Security](#security)).

### Command line

```bash
dbr-autotune ./images -o out              # tune once
dbr-autotune cv ./images -k 5             # 5-fold cross-validation
dbr-autotune evaluate tpl.json ./images   # score a template you already have
dbr-autotune replay out/results.json      # repeat a recorded run
dbr-autotune selfcheck                    # validate the search space against the SDK
```

### Library

```python
from dbr_autotune import TuneRequest, tune

outcome = tune(TuneRequest(images=["./photos"], out="./out", jobs=1))
print(outcome.final.page_coverage)                      # 0.75
template = outcome.templates()["AutoTuned_MaxRecall"]   # dict, ready for the SDK
```

`import dbr_autotune` costs about a millisecond and does **not** load the SDK —
the native libraries are imported only when a run actually starts, so importing
this in a project that may never tune anything is free.

[`examples/integrate.py`](examples/integrate.py) is a runnable walk-through of
the three things integrators normally want: tune a folder and get the template
as a dict, hand that dict straight to the SDK and decode with it, then re-check
a stored template later as a regression gate.

`tune()` takes `log=` for human lines and `progress=` for structured events —
that is the whole interface the web UI is built on, so anything it does, you can
do. See [`api.py`](dbr_autotune/api.py) for `evaluate()`, `evaluate_on()` and
`TuneOutcome`.

---

## What it does

| Phase | What happens |
|---|---|
| **1. Probe** | Runs seven seed configurations — speed-first, default, read-rate-first, DPM, DotCode, and two exhaustive ones — across every symbology, including the long-tail formats `BF_DEFAULT` leaves out. |
| **2. Ground truth** | Folds every probe result into a truth set. Barcodes carry error correction, so a successful decode is taken as correct; reads in weak-checksum 1D symbologies (Code 39, ITF, Pharmacode …) need corroboration. |
| **3. Narrow** | Restricts the template to the symbologies actually present. Usually the single biggest speed win, and it is accepted only if recall holds. |
| **4. Ascend** | Coordinate ascent over 13 knob ladders. Not a grid search: the full product is ~110 million combinations, about 12 years at 3.4 s per trial. Moving one knob at a time costs a *sum* instead of a *product* — about 114 trials, seven minutes. |
| **5. Trim** | At peak recall, switch stages back off one at a time and keep every removal recall survives; set `ExpectedBarcodesCount`; fit `Timeout` to twice the observed worst page. |

Anything a later trial decodes that the truth set did not have is folded back in
— a configuration that reads a barcode nothing else could is a genuine find, not
a false positive — and **every earlier candidate is re-scored**, so all
configurations are always compared on the same denominator.

Coordinate ascent is greedy: it can stop at a local optimum and it will miss
knobs that only help in combination. Three things blunt that — multiple sweeps
(`--rounds`), seven deliberately dissimilar starting points, and the reverse
search in the trim phase. None of them eliminate it.

### Worked example

16 phone photos of laser-etched DataMatrix codes on dark glass bottles. Both
stock presets read **nothing at all**:

```
========================================================================
  ground truth      14 codes on 14/16 pages (inferred)
  SDK default         0.0% recall     1205.1 ms/page
  auto-tuned         85.7% recall      736.3 ms/page   (p95 1019 ms)
  page coverage      75.0%          12/16 pages produced a read
  change            +85.7 recall pts, 1.64x faster
  evaluated         114 configurations in 443s
------------------------------------------------------------------------
  ! 2 page(s) were never decoded by ANY configuration, so the
    inferred truth set records them as carrying no barcode. If they do
    carry one, the real recall is at most 75.0%, not 85.7% -
    ...
      IMG_1013.JPG
      IMG_1021.JPG
========================================================================
```

What it found unprompted: DataMatrix only, DPM mode on, localization by
`LM_STATISTICS_MARKS`, the **inverted** grayscale only, a 39-pixel local
binarization block with negative threshold compensation, and the green colour
channel — green glass.

Both flagged pages do carry a DataMatrix, so **75% page coverage is the number
to quote here, not 85.7% recall** — which is exactly why the run says so rather
than leaving you to notice.

---

## Reading the numbers

Two figures, answering different questions:

| | Denominator | Answers |
|---|---|---|
| **recall** | codes in the truth set | of the codes we know are there, how many did we get |
| **page coverage** | every page in the dataset | how much of the input actually produced a read |

With an **inferred** truth set these diverge, and the gap is the thing to watch.
Inference can only record a code that some configuration decoded, so a page no
configuration ever cracked enters the truth set as *empty* — and then drops out
of the recall denominator entirely. Recall is measured over the pages that were
already solvable; page coverage is measured over everything you handed it.

Such pages are labelled **undetermined**, never "no code" — the tuner cannot
tell a blank page from one it failed on. The run prints them by name and the
report banners them, along with the worst-case recall if they all turn out to
carry a code.

A supplied `--ground-truth` removes the ambiguity: the denominator is then
yours, unread pages count against recall properly, and misreads become visible.

```csv
file,text,format
IMG_1013.JPG,A1B2C3D4,BF_DATAMATRIX
IMG_1014.JPG,E5F6G7H8,
```

The format column may be blank to match on text alone. JSON also works:
`{"IMG_1013.JPG": ["A1B2C3D4"]}`.

---

## Cross-validation

A single tune-then-holdout split reports one number from one arbitrary
partition. With a few dozen images that number moves a lot depending on which
images landed in the holdout, and a lucky split reads exactly like a template
that generalises.

```bash
dbr-autotune cv ./images -k 5
```

```
  folds             3 x 16 pages (seed 0)
  page coverage      68.9% +/- 10.2%   (worst fold 60.0%, best 80.0%)
  recall            100.0% +/- 0.0%
  mean decode          685.0 ms/page
  unstable knobs    localization_modes, timeout_ms
                    (chosen differently by different folds - fitted to the
                     sample, not the problem)
```

The **spread** is the point. ±10 points across folds means the result depends on
which images you tuned on — more images would help more than more tuning. Any
knob listed as unstable was fitted to whichever images that fold happened to
see; `consensus_knobs` in the JSON gives the values a majority agreed on.

Folds are split **by file, never by page**: pages of one PDF or TIFF are usually
near-duplicates, and letting them straddle a split would quietly inflate every
validation score. `--holdout` splits the same way.

It costs K full tuning runs — pair it with `--quick` or `--time-budget`.

---

## Reproducing someone else's result

`results.json` records the complete request plus an environment fingerprint, so
a run can be repeated rather than described:

```bash
dbr-autotune replay their-results.json --images ./my-copy-of-the-images
```

```
  ! environment differs: the SDK version differs, so read rates may differ too
      dbr_bundle   recorded '11.2.1000'  now '11.6.3000'

  recorded page coverage 75.0%  ->  this run 81.2%   (+6.2 pts)
```

Splits are seeded (`--seed`), so the same seed gives the same folds. Timings
depend on the machine and never reproduce exactly; read rates should.

To check a template against images it was not tuned on — someone else's data,
next month's batch, a new SDK version:

```bash
dbr-autotune evaluate AutoTuned_MaxRecall.json ./other-images
```

---

## Using the template from C++

```cpp
CCaptureVisionRouter router;
char msg[512] = {0};
router.InitSettingsFromFile("autotuned-templates.json", msg, sizeof(msg));

CCapturedResultArray* results =
    router.CaptureMultiPages("bottle.jpg", "AutoTuned_MaxRecall");
```

[`cpp_consumer/`](cpp_consumer/) is a complete CMake project that does exactly
this and prints read rate and timings, so the numbers can be confirmed natively
rather than through the Python bindings:

```bash
cd cpp_consumer
cmake -S . -B build -DDBR_SDK_ROOT=/path/to/barcode-reader-c-cpp-samples-main
cmake --build build --config Release
./build/Release/dbr_verify ../out/autotuned-templates.json ../images AutoTuned_MaxRecall
```

`DBR_SDK_ROOT` is any folder holding `Include/` and `Dist/`. On Windows the
build copies the DLLs and `Models/` next to the executable automatically.

No CMake? Visual Studio's compiler is enough, from a Developer Command Prompt:

```bat
set SDK=C:\path\to\barcode-reader-c-cpp-samples-main
set LIB64=%SDK%\Dist\Lib\Windows\x64
cl /nologo /EHsc /std:c++17 /O2 /Fe:dbr_verify.exe cpp_consumer\main.cpp /I"%SDK%\Include" ^
   /link "%LIB64%\DynamsoftCorex64.lib" "%LIB64%\DynamsoftLicensex64.lib" "%LIB64%\DynamsoftCaptureVisionRouterx64.lib"
copy "%LIB64%\*.dll" .
xcopy /E /I /Y "%SDK%\Dist\Models" Models
```

---

## Output

| File | Contents |
|---|---|
| `AutoTuned_MaxRecall.json` | Best read rate found, fastest configuration that achieves it |
| `AutoTuned_Balanced.json` | Within 2 points of peak recall, as fast as possible |
| `AutoTuned_Fastest.json` | Within 90% of peak recall, fastest overall |
| `autotuned-templates.json` | Every variant in one file — load once, pick by name at capture time |
| `results.json` | Request, environment, ground truth, every trial, per-page results |
| `report.html` | Standalone report: KPIs, recall/speed frontier, per-page table, search log |

`Balanced` and `Fastest` are written only when meaningfully faster than
`MaxRecall` (>10%). When the highest-recall template is also the quickest — which
happens often, because narrowing the symbology list cuts both — you get one file.

---

## Options

```
dbr-autotune tune IMAGES... [-o OUT]

dataset
  --limit N               evenly subsample to at most N pages
  --no-recursive          do not descend into subfolders
  --holdout 0.3           hold out 30% of the files and report on them separately
  --jobs N                parallel decode workers (default: half the cores)
  --seed 0                seed for the holdout / fold split

objective
  --speed-weight 0.0      0 = recall first, speed only breaks ties (default)
                          above 0, each halving of decode time is worth
                          SPEED_WEIGHT x 10 recall points
  --recall-tolerance 0.0  recall fraction that may be sacrificed for speed

budget
  --max-trials 200        maximum configurations to evaluate
  --time-budget 600       stop searching after this many seconds
  --rounds 2              coordinate-ascent sweeps
  --quick                 --rounds 1 --max-trials 60
  --skip-trim             stop after the ascent phase

priors  (each one removes work)
  --formats BF_DATAMATRIX BF_QR_CODE
  --expected-count 1
  --ground-truth labels.csv      file,text[,format] — or JSON
  --freeze-truth                 do not add later discoveries to the truth set
  --min-agree 2                  configurations that must agree before a
                                 weak-checksum 1D read is believed
  --min-weak-length 4            shortest payload believed in such a symbology

  --license KEY           or set DBR_LICENSE
```

---

## Timing

Search timings are taken with `--jobs` decodes in flight and are used only to
**rank** configurations, and only by a >10% margin — smaller gaps are scheduling
noise, and picking the bare minimum of hundreds of noisy measurements reliably
picks the luckiest one rather than the best one.

Everything user-facing comes from a final **single-threaded** pass over each
emitted template, so it is comparable to a production single-image call.

`Timeout` is then fitted to twice the slowest page of that pass — and the
template is **re-measured with it**, keeping the original if anything stopped
decoding. `Timeout` is not a pure wall-clock cap: the SDK budgets its decoding
effort against it, so a lower value can cost a page that had finished well
inside the old limit. Fitting it matters because an exhaustive probe may have
been allowed 30 seconds, and shipping that means one bad image can block a
caller for half a minute.

Multi-page documents are decoded with one `CaptureMultiPages` call per file, so
the per-page time for a PDF or TIFF is the file total divided evenly across its
pages. Totals and means are exact.

---

## When the SDK is upgraded

The tool keeps working. The template may not stay optimal.

Those are different things, and the difference matters. Measured across a real
upgrade, 11.2.1000 to 11.6.3000, on the same 16 images:

| template | tuned on | on 11.2 | on 11.6 |
|---|---|---|---|
| A | 11.2 | **14/16** | 13/16 |
| B | 11.6 | 13/16 | **14/16** |
| A, re-tuned on 11.6 | 11.6 | — | **14/16** |

Each template is best on the version it was tuned on and loses an image on the
other. Nothing broke — every template still loaded and ran — but a template
carried across an upgrade is no longer the best one available.

So there are two separate checks, and they answer different questions.

**Does the tuner still work?** `selfcheck` compiles every level of the search
space and hands each to the SDK. If a parameter was renamed or dropped it fails
loudly rather than the search quietly skipping that part of the space. Across
this upgrade all 108 levels were still accepted.

```bash
dbr-autotune selfcheck
```

**Is my template still good?** `evaluate` scores a stored template against
images. With thresholds it exits non-zero, so it belongs in CI next to the
upgrade.

```bash
dbr-autotune evaluate my-template.json ./regression-images --min-coverage 0.85
```

If it has slipped, re-run the tuner — that is the whole fix, and it restored
the lost image here.

`results.json` records which SDK produced it, so `replay` says plainly what
changed:

```
! environment differs: the SDK version differs, so read rates may differ too
    sdk_version  recorded '11.2.1000'  now '11.6.3000'
```

**Tune on the version you deploy on.** The Python bundle and the C++ SDK are
versioned and shipped separately, and it is easy to end up tuning against one
while production runs the other. That mismatch is exactly what costs the image
in the table above.

## Surviving an SDK crash

Decoding runs in **child processes, never in the parent** — not even for a
single-worker run.

This is not architectural neatness. Driving the SDK concurrently segfaults it:
the same configurations that kill a pool of four decode the whole set cleanly in
one process. A segfault takes its whole process down with no Python traceback
(exit 139), so before this the crash killed the tuner outright, and from the web
UI it took the server and the run with it.

Now a crash costs one worker. The parent sees a `BrokenProcessPool` and
**re-runs that configuration on a single worker**, which has not crashed in any
run. That matters for correctness as much as robustness: scoring a crashed
configuration zero would discard one that might be the best available. Only if
the single worker dies too is the configuration recorded as failed, and it is
remembered so the search never pays for it twice.

```
! a decode worker crashed; re-running this configuration on a single worker
  scale_image=type=ST_SCALE_UP,edge=1080  recall 83.3%  mean 3840.6ms
```

That configuration was then rejected on its merits — far too slow — rather than
lost to a crash.

The cost is one process spawn per worker plus a licence handshake each, paid
once per run, and inter-process transfer of results. Per-page timings are taken
inside the worker around the decode call, so they are unaffected.

## Security

The web UI browses the filesystem and starts long jobs, so it:

* binds to **127.0.0.1 only** — `serve()` refuses a routable address;
* requires a token minted at startup and carried in the URL, which stops another
  page in your browser from driving it (a web page can POST to localhost, but
  cannot read the token);
* rejects requests whose `Origin` is a real website.

That is enough for a tool on your own machine. It is **not** authentication —
do not expose it through a tunnel or reverse proxy without putting real auth in
front of it.

---

## Requirements

Python 3.10+. `pip install -e .` from the clone pulls in the SDK and Pillow;
the `[pdf]` extra adds PDF page counting — without it a PDF is scored as a
single page and the run says so.

Runs default to the public trial license from the DBR samples, which needs
network access. Set `DBR_LICENSE` or pass `--license` for your own.

Two checks worth running after an SDK upgrade or a code change:

```bash
dbr-autotune selfcheck    # compiles all 93 search-space levels against the SDK
python -m pytest          # logic tests; no SDK needed
```

`selfcheck` is the one that catches SDK drift — if Dynamsoft renames a
parameter, it fails loudly instead of the tuner silently skipping that part of
the search space.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). The short version: `template.py` is the
only module that knows the template schema, front ends hold no tuning logic, and
any new metric has to be explicit about what it divides by.

## License

MIT — see [LICENSE](LICENSE).
