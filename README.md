# WellVolPOS

**Well probability of success and volume, from a stochastic prospect model.**

Takes one segment's Monte Carlo trial export (GeoX) and one proposed well
trajectory — a reservoir entry depth and a reservoir exit depth — and answers:

1. What is the chance **this well** finds hydrocarbons, as distinct from the
   chance the **prospect** contains them?
2. If it is a discovery, what will the well have **proven**, and what stays
   unproven below it?
3. If it is dry, how much sits **up-dip**, and how likely is that to be material?
4. Given a volume that must be proven, **where does the well have to go**, and
   what does that cost in chance?
5. Which **risk elements** carry the location penalty?

It does not replace GeoX — it re-cuts GeoX's output against a well. It does not
build the contact distribution; that is the [HCWC Builder](../HCWC%20builder)'s
job. It does not do economics.

---

## Quick start

```bash
pip install -r requirements.txt
pytest                 # the specification — should be 68 passed
streamlit run app.py   # opens on the bundled demo data
```

No setup, no data preparation: the app opens on a demo dataset.

---

## The one idea

Everything rests on separating two numbers that the original spreadsheet
multiplied together:

```
r_location = P(contact deeper than the well | hydrocarbons present)
P_well     = POS_prospect × r_location
```

`r_location` is the only quantity the well's position controls. `POS_prospect`
is the only quantity it does not. On the reference dataset:

```
POS_prospect = 0.7605      (read from the trials: 1 − 2395/10000 chance failures)
r_location   = 0.6017      (4576 of 7605 success trials have a deeper contact)
P_well       = 0.4576
```

The source workbook computed `1 − PERCENTRANK(all contacts, entry)`, which
already includes the failure trials, and then multiplied by a separately entered
POS. That is correct only when the entered POS is 1.0 — which it was, so the
shipped answer was right by accident. Any real chance table would have made it
roughly 40 % too low.

---

## Status

| Phase | Content | State |
|---|---|---|
| 0 | Repo, theme, GeoX adapter, failure detector, QC + risking panel, demo selector, parity suite | **done** |
| 1 | Reference grouping engine, figures A3/A4/A5, B3, depth sweep | next |
| 2 | `A(z)`, proven/possible classes, figures A1/A2/A6, B0/B1/B2, live section | |
| 3 | Chance table, reference contours, allocation schemes, B4/B5, threshold mapping | |
| 4 | Inverse tool, optimum finders, bootstrap bands | |
| 5 | Export, dark mode, case save/load, synthetic generators | |

The core calculation modules for phases 2 and 3 are already written and tested —
what is missing from those phases is the figures, not the arithmetic.

---

## Layout

```
app.py                       Streamlit entry point
wellvolpos/
  io/adapters/               trial-file readers; add a simulator by adding a file
  io/failure.py              chance-failure detector -> POS from the trials
  io/qc.py                   the report that gates the analysis tabs
  core/structure.py          A(z), recovered from the trials themselves
  core/groups.py             reference engine: whole-trial grouping (Schneider et al. 2023)
  core/classes.py            extension: proven / possible / attic per trial
  core/chance.py             r_location, reference contours, risk allocation
  core/threshold.py          minimum column height <-> area <-> volume percentile
  core/xlcompat.py           Excel-exact PERCENTILE / PERCENTRANK
  viz/theme.py               one palette, one styling entry point, the depth-axis rule
data/                        two demo trial files (fictional)
docs/                        the design plan and the figure sheets
tests/                       the specification
```

---

## Conventions that are never implicit

Each of these changes the numbers, so each is an explicit setting rather than a
default buried in the code:

- **Risking convention** — are the trials already risked, or success-case only?
  Asked at import, with the detector's evidence shown.
- **Reference contour** for the location factor — crest/apex (Milkov 2021, the
  default, and what the source workbook does) or P90 area (Rose). On this
  dataset the Rose convention is a flat 1.11× uplift at every depth plus a cap
  up-dip of the P90-area contour.
- **Risk-element allocation** — none (report `r` separately), equal cube-root
  (the workbook's), or all-to-trap (Rose). All three give the *same* `P_well`;
  only the attribution differs.
- **Assessment minimum** — a minimum column height below the apex, with the
  equivalent area and volume percentile displayed beside it.
- **Engine** — reference grouping, or the proven/possible decomposition. Both
  are shown; neither is labelled "correct".

## Two rules the tests enforce

**Depth is always on the y-axis, increasing downward.** Not a style preference: a
depth axis on y makes a plot spatially congruent with the subsurface, so a row of
panels sharing one axis can be read straight across at constant depth beside a
well log or a structural section, and the attic sits literally above the well
marker. `tests/test_axes.py`.

**The source workbook is the specification.** `tests/test_excel_parity.py` locks
fifteen values read from it, so the port cannot silently drift from the tool that
is already trusted. It was written before any other code.

---

## Data

`data/` contains two demo trial files. **The data are fictional** and safe to
publish. They are the same 10 000 realisations exported twice — once as the
7-column paste, once as the full 60-column GeoX export — which makes them a good
pair for exercising the importer.

One trap worth knowing: their `TrialNumber` columns hold the same identifiers
attached to *different rows*. Joining two GeoX exports on `TrialNumber` will
silently scramble them. Nothing here does; `tests/test_adapters.py` asserts it.

---

## References

- Schneider, M., Citron, G.P., Haryott, P. & Cook, D. (2023) *Drilling an
  exploration prospect downdip.* AAPG Bulletin 107(5): 743–759.
  [doi:10.1306/09232222051](https://doi.org/10.1306/09232222051) — open access.
- Milkov, A.V. (2021) *Reporting the expected exploration outcome.* J. Pet. Sci.
  Eng. 204: 108754. [doi:10.1016/j.petrol.2021.108754](https://doi.org/10.1016/j.petrol.2021.108754)
- Haskett, W.J. (2003) *Optimal appraisal well location…* SPE 84241.
- Hood, K.C. (2024) *Hydrocarbon column heights, Parts 1–2.* Rose & Associates.

Full discussion in `docs/WellVolPOS_Design_Plan.md`.

## Licence

MIT.
