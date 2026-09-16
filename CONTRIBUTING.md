# Contributing

## Getting set up

```bash
git clone https://github.com/ztyyLV/DBR_Auto_Tuner
cd DBR_Auto_Tuner
pip install -e ".[dev,pdf]"
```

The Dynamsoft SDK is a normal dependency and installs from PyPI. Runs use the
public trial license by default, which needs network access; set `DBR_LICENSE`
to use your own.

## Before opening a pull request

```bash
pytest                      # logic tests, no SDK needed
dbr-autotune selfcheck      # compiles every search-space level against the SDK
ruff check .
```

`selfcheck` is the one that catches SDK drift: it builds a template for every
level of every knob and hands each to `init_settings`. If Dynamsoft renames or
removes a parameter, this fails loudly instead of the tuner silently skipping
that part of the search space.

## Where things live

| Module | Responsibility |
|---|---|
| `template.py` | knob vector → DCV template JSON. **The only module that knows the schema.** |
| `space.py` | seed configurations and the per-knob candidate ladders |
| `engine.py` | running a template over a dataset, thread-local routers, result caching |
| `truth.py` | ground truth: supplied, or inferred with the weak-checksum guards |
| `scoring.py` | recall / page coverage / timing, and how two candidates are compared |
| `search.py` | the five-phase strategy |
| `crossval.py` | K-fold, split by file |
| `api.py` | the public entry points everything else is built on |
| `cli.py`, `server.py`, `web/` | front ends — they must not contain tuning logic |
| `report.py` | the standalone HTML report |

## House rules

**Adding a knob.** Add the field to `template.DEFAULTS`, emit it in
`template.build()`, add its ladder to `space.ladders()`, and run `selfcheck`.
Nothing else should need to change — if it does, the abstraction leaked.

**Front ends hold no logic.** If the CLI and the web UI could disagree about a
result, the logic is in the wrong place. Put it in `api.py`.

**Be honest about denominators.** The most valuable thing this tool does is not
overstate its own results. Inferred ground truth cannot see a barcode nothing
decoded, so pages nothing read are reported as *undetermined*, never as "no
barcode", and `page_coverage` (denominator: every page) is reported alongside
`recall` (denominator: the truth set). Any new metric must be equally explicit
about what it divides by.

**Search timings rank, they do not measure.** They are taken with several
decodes in flight. Anything user-facing comes from the single-threaded pass.

## Sharing results

`results.json` carries the full request plus an environment fingerprint, so
`dbr-autotune replay results.json` repeats a run and reports what changed. When
reporting a read-rate problem, attach `results.json` — it beats a description.

Image sets are the most useful contribution of all, but **check before sharing
images**: product photos often carry serial numbers, batch codes and customer
identifiers.
