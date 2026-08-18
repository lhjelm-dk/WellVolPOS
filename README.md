# WellVolPOS

**The chance your prospect works is not the chance your well works.**

Every prospect carries a probability of success. It goes in the portfolio, the
partner presentation and the board paper. It is a property of the **prospect**.

Then someone picks a location — often late, often traded against rig schedule,
shallow hazard, fault proximity or partner preference — and the well that gets
drilled has a *different*, always *lower*, chance of success. Almost nobody
recomputes it. The dry hole is then scored against the prospect number, and the
post-mortem concludes the geology was optimistic when the geology may have been
right and the well 200 m too far down-dip.

WellVolPOS computes the second number, and keeps it apart from the first.

```
r_location = P(contact deeper than the well | hydrocarbons present)
P_well     = POS_prospect × r_location
```

`r_location` is the only quantity the well's position controls. `POS_prospect` is
the only quantity it does not. **They are never multiplied into one reported
number.**

---

## What that looks like on real data

The bundled prospect C, at five locations on one structure. The chance table is
unchanged throughout — this is the same prospect, assessed identically, drilled in
different places:

| entry (m TVDSS) | `r_location` | **`P_well`** | `Pc` commercial | proven if it works | left up-dip if dry |
|---|---|---|---|---|---|
| 1500 — crest | 100 % | **45.4 %** | 22.9 % | 33 MMboe | 3 MMboe |
| 1600 | 82.7 % | **37.5 %** | 22.9 % | 109 | 42 |
| 1650 | 33.0 % | **15.0 %** | 14.5 % | 170 | 83 |
| 1700 | 21.6 % | **9.8 %** | 9.7 % | 211 | 93 |
| 1750 | 9.4 % | **4.3 %** | 4.3 % | 251 | 107 |

`POS_prospect` is **45.4 % in every row**. The well's chance runs from 45 % to 4 %
across 250 m of the same closure.

And the sting: the tool's own commercial floor on this prospect is **1638 m** —
shallower than that, the well does not demonstrate a commercial volume even when it
works. **The well you must drill to prove commerciality is the one least likely to
find anything.**

---

## Three consequences worth naming

**Hazard avoidance is not free, and the invoice is never itemised.** The crest
usually sits against the master boundary fault. Everything that argues for standing
off it — damage zone, imaging, overpressure, fault-position uncertainty — argues for
stepping down-dip, which is exactly where `r_location` collapses. On prospect C a
150 m step-out costs **30 points of chance**. Nothing in the prospect POS moves, and
no AFE says "we just cut the chance of success by two thirds". It is often still the
right call; it should be a priced one.

**Presence and commerciality reward opposite locations.** `P_well` asks *did we see
hydrocarbons*. `Pc` asks *did we see a developable accumulation*. Optimise the first
and you drill high: a good discovery rate, a small proven volume, and a large attic
you have not tested. Optimise the second and you drill deeper: fewer discoveries, but
a discovery is a development. If the KPI is "discovery rate", the commercially
correct well looks like the worse well.

**Look-back calibration inherits the error.** Score outcomes against prospect POS
without correcting for location and your chance estimates will look systematically
optimistic. Lowering them "fixes" a number that was already right, and the mechanism
that caused the miss is still there. The correction is location discipline, not risk
deflation.

---

## What the tool answers

1. The chance **this well** finds hydrocarbons, as distinct from the chance the
   **prospect** contains them.
2. What a discovery would have **proven**, and what stays unproven below it.
3. If dry, how much sits **up-dip**, and how likely that is to be material.
4. Given a volume that must be proven, **where the well has to go**, and the cost
   in chance.
5. Which **risk elements** carry the location penalty.
6. Between which two depths the well is defensible at all — deep enough to prove a
   commercial volume, shallow enough not to strand one.

It does not replace GeoX; it re-cuts GeoX's output against a well. It does not build
the contact distribution. It does not do economics.

---

## Screenshots

<!-- Images live in docs/screenshots/. Name each one after the figure number the
     app shows, and say in the caption which prospect and which well it was taken
     at — a chance without its location is the thing this tool argues against. -->

*To be added.*

---

## Quick start

**New here? Follow [`GETTING_STARTED.md`](GETTING_STARTED.md)** — step by step, with
the Windows specifics.

```bash
python -m venv .venv && .venv/Scripts/Activate.ps1   # Windows
pip install -r requirements.txt
pytest                 # the specification; it prints its own count
streamlit run app.py   # opens on the bundled demo data
```

No data preparation: the app opens on a demo dataset.

---

## The six tabs

| | tab | the question |
|---|---|---|
| ① | Setup and input | what am I working with, is it sound, and what conventions am I using? |
| ② | Prospect | what is this prospect, before any well? |
| ③ | Where to drill | where should the well go? — every figure that sweeps depth |
| ④ | At this well | what do I get at the depth I chose? |
| ⑤ | Risk & report | attribution, the summary table, the export |
| ⑥ | Theory & guide | what any of it means |

Tab ② is deliberately **well-free**: if a well marker ever appears there, the tab has
lost its subject.

---

## Where to drill, not just what is optimal

Five criteria can all be maximised at the shallowest depth in the sweep, and on
ordinary data they are: above the shallowest sampled contact every success trial is a
discovery, so `r_location` is exactly 1 and every criterion that does not involve a
threshold is *indifferent* across that whole band. A list of optima is then five
copies of one number.

So tab ③ leads with a **window** rather than an optimum. Two quantities bound it, and
both are read as guarantees — the value stays on the right side of the threshold from
that depth all the way down, never a first crossing:

| | |
|---|---|
| **floor** | the shallowest depth from which the **proven mean** stays at or above MEFS. A second, conservative floor asks the same of the proven **P90**, so a poor discovery clears it too. |
| **ceiling** | the depth past which the **attic mean** reaches MEFS. Deeper, and a dry hole would have left a commercial volume untested up-dip. |

Either end can be absent, and that is reported rather than papered over. A window
with the floor deeper than the ceiling is possible too, and it is a finding about the
prospect — at that threshold no location both proves a commercial volume and avoids
stranding one.

The optima are listed under the window: best chance, most chance-weighted volume,
most chance-weighted **proven** volume, best commercial chance `Pc`, best
risk-adjusted volume under exponential utility, best odds subject to a commercial
confidence, and Haskett's uncertainty-reduction peak. Each carries the span of depths
within 2 % of its own best, because on a nearly flat curve a single depth is false
precision — and where a criterion cannot tell two depths apart, the shallower is
reported, since a shallower well never costs chance.

---

## Data

`data/` ships **prospect C**, the default, and **prospect A** in two export forms.

**Prospect C** is a real export with **every depth shifted by one constant**, so the
file names no location. That is a rigid translation, so it changes nothing the tool
says: column heights are differences, `A(z)` is the same curve moved, and
`r_location` counts contacts either side of a depth that moved with them.
`wellvolpos/io/anonymise.py` does it, and a test asserts the invariance rather than
assuming it. It anonymises *where*, not *how much*. It is also **success-case only**,
with no chance failures at all, so it exercises the risking branch no other file
reaches: POS comes from the chance table and the footer says so.

**Prospect A** is fictional, ships in a 7-column paste and the full 60-column form —
so both the everyday case and the duplicate-header trap stay exercised — and is what
the parity suite is locked to. One trap worth knowing: the two forms' `TrialNumber`
columns hold the same identifiers attached to *different rows*, so joining two GeoX
exports on `TrialNumber` silently scrambles them. Nothing here does;
`tests/test_adapters.py` asserts it.

`io/synthetic.py` generates two more files for branches one real prospect cannot
exercise: a **correlated area / net-pay** file, which makes the per-trial split's
uniform-pay guard speak, and a **success-case-only** file. Their closure is a cone, so
`A(z)` is known exactly — the one thing real data cannot offer, and what let a
sensitivity in the reservoir-thickness inversion be found.

To try it on your own data, choose **Upload your own…** in tab ①. Nothing is written
to disk — an upload is read as bytes and passed straight to the adapter.

---

## Conventions that are never implicit

Each of these changes the numbers, so each is an explicit setting rather than a
default buried in the code:

- **Risking convention** — are the trials already risked, or success-case only?
  Asked at import, with the detector's evidence shown.
- **Reference contour** for the location factor — crest/apex (Milkov 2021, the
  default) or P90 area (Rose).
- **Risk-element allocation** — none, equal cube root, or all-to-closure (Rose). All
  three give the *same* `P_well`; only the attribution differs.
- **Assessment minimum** — a minimum column height below the apex.
- **Engine** — reference grouping, or the per-trial proven / unproven-below-LKH
  decomposition. Both are shown; neither is labelled "correct".

---

## What comes out

Tab ⑤ builds four artefacts from **one** assembled bundle, so the workbook, the PDF
and the figures cannot disagree with each other or with the screen:

| Format | For |
|---|---|
| **XLSX** | The reviewer who wants to check the arithmetic. Values only — a formula in an exported workbook is a second implementation of the same calculation. |
| **PDF** | The well proposal. A stamped cover page, then every figure, one per page, in the order the app shows them. |
| **PNG / SVG zip** | Slides. The stamp and the case travel inside the archive, because a figure dropped into a deck is separated from its provenance immediately. |
| **JSON case** | Reopening the session. Settings only, never results — so a reloaded case cannot show numbers this build would not produce. |

Every artefact carries the same stamp: the POS in force **and where it came from**,
`r_location`, `P_well`, the well, the reference contour, the allocation scheme and the
threshold volume. A caption can be cropped out of a screenshot; a cover page cannot be
cropped out of a file.

**Two ways to draw the figures, and the numbers are identical either way.**
matplotlib is the default and needs nothing extra. Choosing *plotly* renders the
figures the app itself draws, through [kaleido](https://pypi.org/project/kaleido/) —
so a figure in the document is the figure that was on screen. kaleido drives a
headless browser and is a large download, so it is optional: without it the app
disables that option and says which package is missing.

---

## Two rules the tests enforce

**Depth is always on the y-axis, increasing downward.** Not a style preference: it
makes a plot spatially congruent with the subsurface, so a row of panels sharing one
axis reads straight across at constant depth beside a well log or a structural
section, and the attic sits literally above the well marker. `tests/test_axes.py`.

**The source workbook is the specification.** `tests/test_excel_parity.py` locks
fifteen values read from it, so the port cannot silently drift from the tool that is
already trusted. It was written before any other code.

---

## Layout

```
app.py                       Streamlit entry point — six tabs
wellvolpos/
  io/adapters/               trial-file readers; add a simulator by adding a file
  io/adapters/generic.py     the fallback reader; proposes a mapping, never assumes one
  io/units.py                unit validation: reject, never convert
  io/failure.py              chance-failure detector -> POS from the trials
  io/qc.py                   the report that gates the analysis tabs
  io/anonymise.py            shift a real export's depths so it can be published
  core/structure.py          A(z), recovered from the trials themselves
  core/groups.py             reference engine: whole-trial grouping (Schneider et al. 2023)
  core/classes.py            extension: proven / unproven-below-LKH / attic per trial
  core/summary.py            the headline, the drilling window, and the optima
  core/mefs.py               every volume read against the MEFS / MCFS line
  core/dependence.py         what the exit moves, and whether the spacing is a vertical well
  core/utility.py            certainty equivalent, and the commerciality hurdle
  core/chance.py             r_location, reference contours, risk allocation
  core/reservoir.py          reservoir thickness, back-calculated from pay
  core/rose.py               No Regrets, Pmcfs(well), Pc(well)
  core/sweep.py              both sweeps, and the inverse
  core/stats.py              bootstrap and Wilson intervals, sample-size diagnostics
  viz/theme.py               one palette, one styling entry point, the depth-axis rule
  viz/interactive.py         the plotly figures — what the app draws
  viz/figures.py             the matplotlib twins — what the export draws
  report/export.py           one bundle, four formats, two renderers
  report/guide.py            the theory & guide tab
data/                        the demo trial files
docs/screenshots/            images used by this README
tests/                       the specification
```

---

## References

- Schneider, M., Citron, G.P., Haryott, P. & Cook, D. (2023) *Drilling an exploration
  prospect downdip.* AAPG Bulletin 107(5): 743–759.
  [doi:10.1306/09232222051](https://doi.org/10.1306/09232222051) — open access.
- Milkov, A.V. (2021) *Reporting the expected exploration outcome.* J. Pet. Sci. Eng.
  204: 108754. [doi:10.1016/j.petrol.2021.108754](https://doi.org/10.1016/j.petrol.2021.108754)
- Haskett, W.J. (2003) *Optimal appraisal well location…* SPE 84241.
- Hood, K.C. (2024) *Hydrocarbon column heights, Parts 1–2.* Rose & Associates.

Full discussion in `docs/WellVolPOS_Design_Plan.md`; every source is listed in tab ⑥
with whether it was read directly or cited through another work.

## Licence

MIT.
