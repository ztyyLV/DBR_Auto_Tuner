"""Self-contained HTML report for a tuning run."""
from __future__ import annotations

import html
import json
import os
from typing import Any, Dict, List

from . import template as T
from .scoring import Score
from .search import TuneResult

VARIANT_TITLE = "AutoTuned_MaxRecall"

_CSS = """
:root{--bg:#f7f7f5;--panel:#fff;--ink:#1c1c1a;--muted:#6b6b66;--line:#e3e3de;
      --good:#1f7a4d;--warn:#a8601b;--bad:#a52f2f;--accent:#2b5fd9;--code:#f2f2ef;}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){
      --bg:#17171a;--panel:#1f1f23;--ink:#e9e9e6;--muted:#9a9a94;--line:#33333a;
      --good:#5fc08c;--warn:#d9a05b;--bad:#e07a7a;--accent:#7aa2f7;--code:#26262c;}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
     font:14px/1.55 ui-sans-serif,-apple-system,"Segoe UI",Roboto,sans-serif}
.wrap{max-width:1180px;margin:0 auto;padding-block:32px;padding-left:20px;padding-right:20px}
h1{font-size:22px;margin:0 0 4px} h2{font-size:15px;margin:32px 0 10px;
   text-transform:uppercase;letter-spacing:.07em;color:var(--muted)}
.sub{color:var(--muted);margin:0 0 24px}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:16px}
.grid{display:grid;gap:12px}
.cards{grid-template-columns:repeat(auto-fit,minmax(215px,1fr))}
.kpi .n{font-size:26px;font-weight:650;letter-spacing:-.02em}
.kpi .l{color:var(--muted);font-size:12px;text-transform:uppercase;letter-spacing:.06em}
.kpi .d{font-size:12px;margin-top:6px}
.up{color:var(--good)} .down{color:var(--bad)} .flat{color:var(--muted)}
table{border-collapse:collapse;width:100%;font-size:13px}
th,td{text-align:left;padding:7px 10px;border-bottom:1px solid var(--line);vertical-align:top}
th{color:var(--muted);font-weight:600;font-size:11px;text-transform:uppercase;letter-spacing:.06em;
   position:sticky;top:0;background:var(--panel)}
td.num,th.num{text-align:right;font-variant-numeric:tabular-nums}
tr.acc td{background:color-mix(in srgb,var(--good) 9%,transparent)}
.scroll{overflow:auto;max-height:460px;border:1px solid var(--line);border-radius:10px;background:var(--panel)}
.tag{display:inline-block;padding:1px 7px;border-radius:99px;font-size:11px;
     border:1px solid var(--line);color:var(--muted);margin-right:4px}
code,pre{font-family:ui-monospace,"Cascadia Code",Consolas,monospace}
pre{background:var(--code);border:1px solid var(--line);border-radius:10px;padding:14px;
    overflow:auto;font-size:12px;max-height:520px;margin:0}
.mono{font-family:ui-monospace,Consolas,monospace;font-size:12px}
.trunc{max-width:340px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;display:block}
details{margin-top:10px} summary{cursor:pointer;color:var(--accent);font-size:13px}
.legend{color:var(--muted);font-size:12px;margin-top:8px}
.warn-panel{margin-top:12px;border-color:var(--warn);
   background:color-mix(in srgb,var(--warn) 9%,var(--panel))}
"""


def _fmt_ms(value: float) -> str:
    return f"{value:,.0f}" if value >= 10 else f"{value:.1f}"


def _delta(new: float, old: float, higher_is_better: bool, percent: bool) -> str:
    if percent:
        # A point difference needs no division, so a zero baseline is fine here -
        # and 0% to 90% is exactly the comparison worth showing.
        diff = (new - old) * 100
        good = diff > 0 if higher_is_better else diff < 0
        if abs(diff) < 0.05:
            return '<span class="flat">unchanged</span>'
        cls = "up" if good else "down"
        return f'<span class="{cls}">{diff:+.1f} pts vs SDK default</span>'
    if old == 0 or new == 0:
        return '<span class="flat">-</span>'
    ratio = old / new if new else 0
    if 0.97 < ratio < 1.03:
        return '<span class="flat">unchanged</span>'
    cls = "up" if ratio > 1 else "down"
    word = "faster" if ratio > 1 else "slower"
    shown = ratio if ratio > 1 else (1 / ratio if ratio else 0)
    return f'<span class="{cls}">{shown:.2f}x {word}</span>'


def _pareto_svg(front: List[Score], best: Score) -> str:
    points = [(s.mean_ms, s.recall, s.name) for s in front]
    if len(points) < 2:
        return ""
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x0, x1 = min(xs) * 0.9, max(xs) * 1.05
    y0, y1 = max(0.0, min(ys) - 0.05), min(1.0, max(ys) + 0.05)
    if x1 <= x0 or y1 <= y0:
        return ""
    w, h, pad = 640, 240, 44

    def px(v):
        return pad + (v - x0) / (x1 - x0) * (w - pad - 16)

    def py(v):
        return h - pad - (v - y0) / (y1 - y0) * (h - pad - 16)

    parts = [f'<svg viewBox="0 0 {w} {h}" style="max-width:100%;height:auto" '
             f'role="img" aria-label="recall versus decode time">']
    parts.append(f'<line x1="{pad}" y1="{h-pad}" x2="{w-16}" y2="{h-pad}" '
                 f'stroke="currentColor" opacity=".25"/>')
    parts.append(f'<line x1="{pad}" y1="16" x2="{pad}" y2="{h-pad}" '
                 f'stroke="currentColor" opacity=".25"/>')
    path = " ".join(f"{'M' if i == 0 else 'L'}{px(x):.1f},{py(y):.1f}"
                    for i, (x, y, _n) in enumerate(sorted(points)))
    parts.append(f'<path d="{path}" fill="none" stroke="currentColor" opacity=".35"/>')
    for x, y, name in points:
        is_best = name == best.name
        radius = 6 if is_best else 4
        colour = "var(--accent)" if is_best else "currentColor"
        parts.append(f'<circle cx="{px(x):.1f}" cy="{py(y):.1f}" r="{radius}" '
                     f'fill="{colour}" opacity="{1 if is_best else .55}">'
                     f'<title>{html.escape(name)}: {y:.1%} @ {_fmt_ms(x)}ms</title></circle>')
    parts.append(f'<text x="{pad}" y="{h-14}" font-size="11" fill="currentColor" '
                 f'opacity=".6">{_fmt_ms(x0)}ms</text>')
    parts.append(f'<text x="{w-16}" y="{h-14}" font-size="11" text-anchor="end" '
                 f'fill="currentColor" opacity=".6">{_fmt_ms(x1)}ms  (mean decode time)</text>')
    parts.append(f'<text x="6" y="24" font-size="11" fill="currentColor" '
                 f'opacity=".6">{y1:.0%} recall</text>')
    parts.append(f'<text x="6" y="{h-pad}" font-size="11" fill="currentColor" '
                 f'opacity=".6">{y0:.0%}</text>')
    parts.append("</svg>")
    return "".join(parts)


def render(result: TuneResult, dataset_roots: List[str], outputs: Dict[str, str],
           serial_scores: Dict[str, Score] | None = None,
           emit_knobs: Dict[str, dict] | None = None) -> str:
    best = result.best
    base = result.baseline
    serial_scores = serial_scores or {}
    # What was actually written to disk, which may carry a fitted Timeout.
    emit_knobs = emit_knobs or {key: t.knobs for key, t in result.variants.items()}
    e = html.escape

    kpis = []
    final_score = serial_scores.get("best", best.score)
    base_score = serial_scores.get("baseline", base.score if base else None)

    kpis.append(("Recall", f"{final_score.recall:.1%}",
                 f"{final_score.found}/{final_score.expected} codes  "
                 + (_delta(final_score.recall, base_score.recall, True, True) if base_score else "")))
    kpis.append(("Pages fully read", f"{final_score.pages_complete}/{final_score.pages_expected}",
                 f"{final_score.page_rate:.0%} of code-bearing pages"))
    kpis.append(("Mean decode", f"{_fmt_ms(final_score.mean_ms)} ms",
                 _delta(final_score.mean_ms, base_score.mean_ms, False, False) if base_score else ""))
    kpis.append(("p95 decode", f"{_fmt_ms(final_score.p95_ms)} ms",
                 f"worst page {_fmt_ms(final_score.max_ms)} ms"))
    kpis.append(("Page coverage", f"{final_score.page_coverage:.1%}",
                 f"{final_score.pages_partial}/{final_score.pages_total} pages produced a read"
                 + (f" &middot; {final_score.pages_undetermined} undetermined"
                    if final_score.pages_undetermined else "")))

    cards = "".join(
        f'<div class="panel kpi"><div class="l">{e(label)}</div>'
        f'<div class="n">{value}</div><div class="d">{detail}</div></div>'
        for label, value, detail in kpis)

    # --- variants ---------------------------------------------------------
    variant_rows = []
    for key, trial in result.variants.items():
        s = serial_scores.get(key, trial.score)
        variant_rows.append(
            f"<tr><td><b>{e(key)}</b><div class='mono' style='color:var(--muted)'>"
            f"{e(outputs.get(key, ''))}</div></td>"
            f"<td class='num'>{s.recall:.1%}</td>"
            f"<td class='num'>{s.pages_complete}/{s.pages_expected}</td>"
            f"<td class='num'>{_fmt_ms(s.mean_ms)}</td>"
            f"<td class='num'>{_fmt_ms(s.p95_ms)}</td>"
            f"<td>{_knob_tags(emit_knobs.get(key, trial.knobs))}</td></tr>")

    # --- per page ---------------------------------------------------------
    # Use the single-threaded pass, so the times here match the KPIs above
    # rather than the noisier timings taken during the parallel search.
    page_score = final_score if final_score.per_page else best.score
    page_source = ("measured single-threaded" if page_score.serial
                   else "measured during the parallel search")
    page_rows = []
    for key in sorted(page_score.per_page, key=lambda k: page_score.per_page[k]["label"]):
        info = page_score.per_page[key]
        expected, found = info["expected"], info["found"]
        if expected == 0:
            status, cls = "undetermined", "warn"
        elif found == expected:
            status, cls = "read", "up"
        elif found:
            status, cls = f"partial {found}/{expected}", "warn"
        else:
            status, cls = "MISSED", "down"
        texts = " | ".join(info["texts"]) or "-"
        page_rows.append(
            f"<tr><td>{e(info['label'])}</td>"
            f"<td class='{cls}'>{e(status)}</td>"
            f"<td class='num'>{info['ms']:,.0f}</td>"
            f"<td>{' '.join(f'<span class=tag>{e(f)}</span>' for f in info['formats'])}</td>"
            f"<td><span class='trunc mono' title='{e(texts)}'>{e(texts)}</span></td>"
            f"<td class='num'>{info['extra'] or ''}</td></tr>")

    # --- trials -----------------------------------------------------------
    trial_rows = []
    for index, trial in enumerate(result.trials, 1):
        s = trial.score
        trial_rows.append(
            f"<tr class='{'acc' if trial.accepted else ''}'>"
            f"<td class='num'>{index}</td><td>{e(trial.phase)}</td>"
            f"<td class='mono'>{e(trial.name)}</td>"
            f"<td class='num'>{s.recall:.1%}</td>"
            f"<td class='num'>{_fmt_ms(s.mean_ms)}</td>"
            f"<td class='num'>{s.extra or ''}</td>"
            f"<td>{e(trial.note)}</td></tr>")

    # --- truth ------------------------------------------------------------
    truth_rows = []
    for key, page in sorted(result.truth.pages.items()):
        for entry in page.values():
            truth_rows.append(
                f"<tr><td>{e(os.path.basename(key))}</td><td>{e(entry.format)}</td>"
                f"<td class='mono'><span class='trunc' title='{e(entry.text)}'>"
                f"{e(entry.text)}</span></td>"
                f"<td class='num'>{entry.agreements}</td>"
                f"<td class='num'>{entry.confidence}</td>"
                f"<td>{e(entry.first_seen)}</td></tr>")

    withheld_rows = "".join(
        f"<tr><td>{e(os.path.basename(key))}</td><td>{e(entry.format)}</td>"
        f"<td class='mono'>{e(entry.text)}</td>"
        f"<td class='num'>{entry.agreements}</td>"
        f"<td class='num'>{entry.confidence}</td></tr>"
        for key, entry in result.truth.unconfirmed())
    withheld_block = f"""
<h2>Withheld reads</h2>
<div class="scroll"><table>
<thead><tr><th>Page</th><th>Format</th><th>Text</th><th class="num">Configs agreeing</th>
<th class="num">Confidence</th></tr></thead><tbody>{withheld_rows}</tbody></table></div>
<p class="legend">Reads in symbologies without reliable error detection that were not
corroborated by enough configurations, or whose payload was too short to believe.
They are excluded from the truth set, so no template is credited or penalised for
them. Tighten or relax with <code>--min-agree</code> and
<code>--min-weak-length</code>, or supply <code>--ground-truth</code>.</p>
""" if withheld_rows else ""

    undetermined_labels = sorted(info["label"] for info in final_score.per_page.values()
                                 if info["expected"] == 0)
    undetermined_block = ""
    if undetermined_labels and result.truth.source.startswith("inferred"):
        undetermined_block = (
            '<div class="panel warn-panel">'
            f'<b>{len(undetermined_labels)} page(s) were never decoded by any '
            'configuration.</b> The inferred truth set therefore records them as '
            'carrying no barcode, and the recall above excludes them &mdash; its '
            'denominator only counts codes something managed to read. '
            f'<b>If these pages do carry a barcode, real recall is at most '
            f'{final_score.recall_floor:.1%}, not {final_score.recall:.1%}.</b> '
            'Check them by eye, then pass <code>--ground-truth</code> to score them '
            'properly.'
            f'<div class="mono" style="margin-top:8px">{e(", ".join(undetermined_labels))}</div>'
            '</div>')

    profile_bits = " &nbsp;·&nbsp; ".join(
        f"{e(str(k).replace('_', ' '))}: <b>{e(str(v))}</b>"
        for k, v in result.profile.items())

    selected_knobs = emit_knobs.get("max_recall", best.knobs)
    final_json = json.dumps(T.build(selected_knobs, "AutoTuned_MaxRecall"),
                            indent=2, ensure_ascii=False)
    svg = _pareto_svg(result.front, best.score)

    return f"""<title>DBR Auto-Tune Report</title>
<style>{_CSS}</style>
<div class="wrap">
<h1>Dynamsoft Barcode Reader &mdash; auto-tuned template</h1>
<p class="sub">{len(page_score.per_page)} pages from {e(', '.join(dataset_roots))}
 &nbsp;·&nbsp; {len(result.trials)} configurations evaluated in
 {result.elapsed_s:,.0f}s &nbsp;·&nbsp; ground truth: {e(result.truth.source)}
 {'&nbsp;·&nbsp; <b>budget: ' + e(result.budget_hit) + '</b>' if result.budget_hit else ''}</p>

<div class="grid cards">{cards}</div>
{undetermined_block}

<h2>Dataset profile</h2>
<div class="panel">{profile_bits or 'not profiled'}</div>

<h2>Emitted templates</h2>
<div class="scroll"><table>
<thead><tr><th>Variant</th><th class="num">Recall</th><th class="num">Pages</th>
<th class="num">Mean ms</th><th class="num">p95 ms</th><th>Tuned parameters</th></tr></thead>
<tbody>{''.join(variant_rows)}</tbody></table></div>
<p class="legend">Load the combined file once and pick a template by name at capture time.</p>

{f'<h2>Recall vs. speed &mdash; explored frontier</h2><div class="panel">{svg}'
 f'<p class="legend">Every point is a configuration that nothing else beat on both axes. '
 f'The highlighted point is the template that was selected.</p></div>' if svg else ''}

<h2>Per-page result &mdash; {e(VARIANT_TITLE)}</h2>
<div class="scroll"><table>
<thead><tr><th>Page</th><th>Status</th><th class="num">ms</th><th>Formats</th>
<th>Decoded</th><th class="num">Extra</th></tr></thead>
<tbody>{''.join(page_rows)}</tbody></table></div>
<p class="legend">Times {e(page_source)}. <b>undetermined</b> = nothing was ever decoded here, so whether the page carries a barcode at all is unknown.</p>

<h2>Search log</h2>
<div class="scroll"><table>
<thead><tr><th class="num">#</th><th>Phase</th><th>Configuration</th><th class="num">Recall</th>
<th class="num">Mean ms</th><th class="num">Extra</th><th>Outcome</th></tr></thead>
<tbody>{''.join(trial_rows)}</tbody></table></div>
<p class="legend">Highlighted rows were accepted as improvements.</p>

<h2>Ground truth ({result.truth.total} codes)</h2>
<div class="scroll"><table>
<thead><tr><th>Page</th><th>Format</th><th>Text</th><th class="num">Configs agreeing</th>
<th class="num">Confidence</th><th>First read by</th></tr></thead>
<tbody>{''.join(truth_rows) or '<tr><td colspan=6>nothing decoded</td></tr>'}</tbody>
</table></div>

{withheld_block}
<h2>Selected template</h2>
<details open><summary>{e(outputs.get('max_recall', 'template JSON'))}</summary>
<pre>{e(final_json)}</pre></details>
</div>
"""


def _knob_tags(knobs: Dict[str, Any]) -> str:
    tuned = T.describe(knobs)
    if not tuned:
        return "<span class='tag'>SDK defaults</span>"
    from .search import _short
    return " ".join(f"<span class='tag'>{html.escape(k)}={html.escape(_short(v))}</span>"
                    for k, v in sorted(tuned.items()))
