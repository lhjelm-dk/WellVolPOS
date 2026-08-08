# WellVolPOS — Well POS and Volume

**Design plan · v4 · 8 August 2026 · supersedes v1–v3**
Prepared for Lars Hjelm.

This version closes every open item and removes the HCWC / vertical-risk material, which now lives in its own project at `..\HCWC builder\`. What remains is a clean, single-purpose specification.

---

## 1 · The tool

**WellVolPOS answers one question: what does *this well*, at *this depth*, do to the chance and the volume of *this prospect*?**

It takes a stochastic resource model of one prospect segment — a GeoX trial export — and one proposed well trajectory through it, defined by a reservoir **entry** depth and a reservoir **exit** depth. It returns:

1. The chance **this well** finds hydrocarbons, as distinct from the chance the **prospect** contains them.
2. If it is a discovery: what the well will have **proven**, and what stays unproven below it.
3. If it is dry: how much sits **up-dip**, and how likely that is to be material.
4. Given a volume that must be proven: **where the well has to go**, and what that costs in chance.
5. Which **risk elements** carry the location penalty.

Every answer is also produced as a curve across the whole structure, so the trade-off between chance and regret is visible rather than argued.

**Boundaries.** It does not replace GeoX — it re-cuts GeoX's output against a well. It does not build the contact distribution — that is the HCWC Builder's job (§9). It does not do economics.

---

## 2 · The verified foundation

Everything in this section was reproduced in numpy from your workbook and demo data. These are the numbers the test suite locks.

### 2.1 Excel parity

| Quantity | Value |
|---|---|
| Prospect mean / P50 / P10 (MMboe) | 10.3133 / 10.5754 / 19.4154 |
| Well-associated P99.5 / P90 / P50 / mean / P10 | 8.0696 / 10.9406 / 15.3987 / 16.5206 / 23.5232 |
| Tested by well, 3500–3550 m (n = 3 706) | 10.6037 / 14.5035 / 14.7753 / 19.4030 |
| Up-dip mean / P10 | 5.0764 / 11.3568 |
| Calc. well POS at 3500 m | 0.45759 |

### 2.2 The trials contain the chance failures — confirmed

| Test | Result |
|---|---|
| Zero-volume trials | **2 395 of 10 000 (23.95 %)** |
| Their HC-water contact | **exactly 3166.59 m in all 2 395** — one value |
| Their gross pay / area / HC-GRV | **exactly 0.0 in all 2 395** |
| Their crest, GRV, N/G, porosity, thickness | still fully sampled (2 274–2 387 distinct values) |
| Gap to the shallowest success contact (3360.1 m) | **193.5 m containing zero trials** |

The 193.5 m hole is what rules out a truncated contact distribution: a distribution clipped at a bound piles mass *at* the bound and also scatters just above it. GeoX is writing risked trials — chance failure zeroes the hydrocarbon quantities and stamps a placeholder contact above any possible crest.

**`POS_trials = 1 − 2395/10000 = 0.7605`**, readable straight from the file. *You have confirmed this against the GeoX project's chance factors.*

### 2.3 The decomposition — the spine of the whole app

```
r_location = P(contact deeper than entry | hydrocarbons present) = 4576/7605 = 0.6017
P_well     = POS_prospect × r_location = 0.7605 × 0.6017 = 0.4576
```

Two numbers, two meanings, never mixed. `r_location` is the only quantity the well's position controls; `POS_prospect` is the only quantity it does not.

**The defect this fixes.** `Results!E6` runs `PERCENTRANK` over **all** trials including the sentinels, so it already returns the fully risked well POS — and then multiplies by `E3` again. Your 0.4576 is correct only because `E3 = 1.0`. A chance table of 0.60 would have produced 0.2746, about 40 % too low.

Peer-reviewed backing: Milkov (2021) — "when the well is drilled downdip, the geological PoS for the segment interacts with the probabilistic distribution of petroleum-water contact, and the geological PoS for the well is the multiple of these two factors." His segment A2 falls from PoS 0.34 at crest to 0.07 at 2650 m. Same formula, same software.

### 2.4 The up-dip volume is diluted

`Results!M7` averages the 2 395 chance-failure zeros into the attic:

| | Value |
|---|---|
| E[up-dip \| well is dry] — includes chance failures | **5.08 MMboe** (your number) |
| E[up-dip \| well is dry **and** prospect charged] | **9.09 MMboe** |

Both are legitimate and answer different questions. The second is what gets quoted when someone argues for a sidetrack. Both are shown, labelled.

### 2.5 `Results!V15` solved

```
V15 = (P_well / POS_prospect)^(1/3) = r_location^(1/3)
```
0.458^(1/3) = **0.7708238778** against your cell's 0.7708238778159993. Because chance factors are multiplicative, the natural additive space is logarithmic — so your cube root is **an equal split of the location log-risk across three elements**, charge / trap / retention, with reservoir exempt. Internally consistent, and now named.

### 2.6 The structure is recoverable from the trials

`HC-bearing GRV = Productive area × Average gross pay` exactly (ratio 1.000000, zero SD), and productive area is a deterministic function of contact depth — isotonic fit **R² = 0.9999999987**. So the area–depth curve `A(z)` comes out of the trial file with no extra input, and every trial can be split at any structural level.

---

## 3 · Two published methods, both supported

### 3.1 Volume grouping — the reference engine

Schneider, Citron, Haryott & Cook (2023), step 2 of the workflow:

> *"If the sampled area is greater than or equal to the AAWL, then place that EUR value in the downdip group. This assumes a discovery at the well location, which includes the discovered updip volume."*

Whole trials are assigned to one of two groups by comparing sampled area against the area at the well location. **Your workbook already implements this correctly.** It is the reference engine, not a legacy mode.

The same paper then names, but does not compute, the finer split:

> *"the downdip EUR distribution represents the range of EUR from the prospect crest to the base of hydrocarbons in the well, plus the remaining EUR distribution from the base of the hydrocarbons in the well to the hydrocarbon water contact. Additional complexity — for example, incorporating a range of column heights — can be assessed using the techniques discussed in this paper."*

That is exactly the proven / upside decomposition in §4. It runs alongside the reference engine as an **extension**, and the app never labels either one "correct".

### 3.2 The location factor — reference contour

| | Rose (2017 / 2021 / 2023) | Milkov (2021) / your Excel |
|---|---|---|
| Formula | `P_well = Pg × (P_trap@well / 90 %)` | `P_well = POS × P(contact ≥ z_well)` |
| Reference | the **P90 productive area** — Rose (2001) assess trap chance as confidence the trap holds at least the P90 area, consistent with Pg being the chance of the P99 EUR | the **crest / apex** |
| Up-dip of the reference | `P_well = Pg`, flat | keeps rising toward POS |

On Prospect A the P90 area is 1.691 km² at 3448.8 m, and Rose's normalisation is a flat **1.11× (= 1/0.90)** uplift at every depth plus a cap up-dip of that contour:

| Well entry | Crest-referenced | Rose | ratio |
|---|---|---|---|
| 3450 m | 0.682 | 0.758 | 1.11× |
| 3500 m | 0.458 | 0.508 | 1.11× |
| 3550 m | 0.087 | 0.097 | 1.11× |

**Decision (your 10.3): default = crest / apex**, with **P90 area (Rose)** and a user percentile as options. Both curves are drawn on figure A3 so the choice is never invisible.

---

## 4 · Volume classes

Per trial *i*, with `A(z)` the recovered area–depth curve:

```
discovery_i = HC present AND contact_i > z_entry
LKH_i       = min(contact_i, z_exit)                        lowest known hydrocarbon in the well
proven_i    = res_i × min(1, A(LKH_i)/A(contact_i))         if discovery, else 0
possible_i  = res_i − proven_i                              below the reservoir exit, not proven
attic_i     = res_i                                         if HC present but contact ≤ z_entry
```

**Decision (your 10.1):** volume potentially below the reservoir exit point is **not proven**. It is reported as a separate class named **"Possible — below reservoir exit"**, so the language matches the reserves convention and nobody reads it as proven.

At entry 3500 / exit 3550:

| Class | n | P90 | P50 | mean | P10 |
|---|---|---|---|---|---|
| Prospect, all trials | 10 000 | 0.00 | 10.58 | 10.31 | 19.42 |
| Discovery case (Rose "downdip group") | 4 576 | 10.94 | 15.40 | 16.52 | 23.52 |
| **Proven at the well** — headline KPI | 4 576 | 10.94 | 15.31 | **15.76** | 21.32 |
| Possible — below reservoir exit | 4 576 | 0.00 | 0.00 | 0.76 | 2.64 |
| Attic \| dry hole (Rose "updip group", charged only) | 3 029 | 5.94 | 8.91 | 9.09 | 12.39 |

**Outcome tree:** 23.9 % chance failure · 30.3 % dry with an attic · 37.1 % discovery with the contact logged · 8.7 % discovery with HC down to exit · **P_well 45.8 %**.

---

## 5 · Chance and the risk-element split

### 5.1 The allocation formula

```
P_element_at_well = P_element × r_location ^ w_element        with  Σ w = 1
```

| Scheme | Weights (charge, trap, reservoir, retention) | Provenance |
|---|---|---|
| **None — report `r` separately** *(default)* | (0, 0, 0, 0) | Milkov 2021 |
| **Equal cube-root** | (⅓, ⅓, 0, ⅓) | your Excel |
| **All to trap** | (0, 1, 0, 0) | Rose, Eq. 1 |
| **Custom** | sliders, auto-normalised | — |

All schemes give **identical `P_well`**. Only the attribution differs, and the figure states that every time.

**Why "none" is the default.** The location factor comes from the volumetric Monte Carlo; the element table is a separate categorical assessment. Spreading one across the other presents a single number differently — it does not add information about charge or trap. Milkov's treatment, the only peer-reviewed one, keeps them apart. Your allocation ships as a first-class option because a partner or a licence round may want the risk table stated at the well.

**Reservoir is exempt, and that is right:** the contact distribution is a fill / spill / retention / charge statement, not a reservoir-presence statement.

**Decision (your 10.4):** the per-element *manual* location override is **not built**. Recorded here so the reasoning survives: reservoir chance can genuinely vary with location for facies reasons, and that variation is not in the contact distribution, so it can only ever be entered by hand. If it is wanted later it is a small addition — one column in the chance table and one extra step in the waterfall. A floor guard (warn when any revised element falls below 0.10) ships now regardless.

### 5.2 The minimum flowable threshold

**Decision (your 10.6): the primary control is a minimum column height in metres below the apex.** Hood's argument is that the assessment minimum "can be effectively linked to seal capacity if based on a minimum column height and not a minimum hydrocarbon volume."

You said the column-height and P99.5-volume definitions should be the same thing. Checked against Prospect A — **almost, but not exactly**, and the tool will show you the difference rather than assume it away:

- At a **fixed** contact depth, and therefore a fixed column height, the resource still spans about **3×** (2.8× over 3360–3380 m, 3.4× over 3450–3470 m), because area is fixed but gross pay and yield still vary.
- In practice the two are close: cutting at the P98 level they disagree on only about **1 %** of the excluded set. They diverge further if area and net pay are correlated.
- In your existing GeoX run **both are already applied upstream.** Extrapolating `A(z)` to zero puts the apex near 3218 m, so the shallowest sampled contact at 3360.1 m already sits roughly **142 m** of column above it. No realisation in the file is below the assessment minimum.

So the app: takes an **apex depth** as an input (not extrapolated — you know it from the map); the threshold control is **minimum column height (m)**; it displays the equivalent minimum area and equivalent volume percentile beside it so the mapping is visible; and it warns when the requested minimum is looser than the one already baked into the trials, in which case it changes nothing. P99.5 remains available as an alternative expression of the same control.

---

## 6 · Graphics

Two figure sheets, both drawn on the demo data. Attached as `sheetA_structure_outcomes.png` and `sheetB_sweep_risk.png`.

### 6.1 The depth rule

**Any axis carrying a depth — contact, entry, exit, LKH, apex, spill — goes on the y-axis, increasing downward, labelled m TVDSS. Panels in a row share the range.**

This is not a style preference. A depth axis on y makes the plot **spatially congruent with the subsurface**: higher on the page is shallower is up-dip. Consequences:

- A row of panels sharing one depth axis becomes a log-style array where the same height on the page always means the same depth — readable straight across, and directly comparable with a well log or a structural section.
- The well, contact, apex and spill appear **where they physically are**. The attic is literally above the well marker; "possible below exit" is literally below it. Geometry and semantics agree.
- No mental transposition against seismic or a Petrel section.
- It is the domain convention — Milkov's Fig. 5, Hood's column-height figures, and your own `image6.png` all do it.

Put depth on x and the picture is rotated 90° from the thing it describes, and every reader re-maps it every time.

### 6.2 Sheet A — structure, location and outcomes

*(A1–A3 share one inverted depth axis)*

| # | Figure | Job |
|---|---|---|
| A1 | Area–depth curve recovered from the trials, entry and exit marked | the structural spine |
| A2 | Outcome tree vs well location — chance failure / dry-with-attic / discovery-contact-seen / discovery-HC-to-exit | the clearest single picture of what moving the well does |
| A3 | Chance decomposition — `P_well` and `r_location` as separate curves, `POS_trials` as a rule | makes §2.3 impossible to misread; carries the reference-contour toggle |
| A4 | Resource vs contact depth — log-density hexbin + smoothed P90/P50/P10 | replaces the unreadable 10 000-point scatter |
| A5 | Exceedance curves at the chosen location — prospect / discovery / proven / attic, MEFS rule | the money chart; a live version of your `image6.png` |
| A6 | Overlap of the two live outcomes — attic-if-dry against proven-if-discovery | Schneider's "surprising overlap", on your prospect |

### 6.3 Sheet B — location sweep and risk

*(B0–B3 share one inverted depth axis)*

| # | Figure | Job |
|---|---|---|
| B0 | **Schematic section**, redrawn from `A(z)`, colour-keyed to attic / proven / possible-below, with the well | anchors the whole row to the structure |
| B1 | Volume split vs location — proven, attic, possible below exit | Schneider Fig. 7/11/12 equivalent |
| B2 | **Chance vs regret** — `P_well`, P(proven > MEFS \| discovery), P(attic > MEFS \| dry) | the most decision-relevant plot in the tool; the crossings are the argument |
| B3 | Uncertainty reduction vs location (Haskett 2003), optimum found automatically | your chart, with the maximum found rather than eyeballed |
| B4 | **Chance waterfall**, log scale — elements then the location factor | bar length *is* the risk each element contributes |
| B5 | **Allocation dumbbell** — three schemes against the prospect baseline | reservoir shows no arrow under any scheme |
| B6 | Inverse — volume-to-prove → required entry depth, with band and resulting `P_well` | your H38–H40 block as a curve |

Also offered, since it is where Rose read `P_trap@well`: the **area distribution on log-probit axes**.

### 6.4 Visual system

Blue = discovery / chance · orange = attic / up-dip / regret · yellow = proven · aqua = prospect totals. Fixed by meaning, never cycled, validated colourblind-safe in light and dark. Density = one blue hue light→dark, never a rainbow. **No dual y-axes.** Legend on every ≥2-series chart, plus direct labels where ≤4 series. MEFS always a dotted ink rule. Dark mode is a selected palette, not an inversion. Crosshair, tooltip and a "show table" toggle on every chart.

**Units (your 10.5): MMboe, m, km² only.** No field units, no separate oil/gas reporting. Unit strings are read from the export and displayed on every axis, and the importer rejects a file whose units do not match rather than silently converting.

---

## 7 · The application

Six tabs. The sidebar holds the live controls: entry depth (or up-dip area), exit depth, MEFS, minimum column height.

| Tab | Content |
|---|---|
| **① Data** | demo selector or upload · adapter detection · column mapping · unit confirmation |
| **② QC & Risking** | the QC report and the risking question. **Gates everything else** |
| **③ Prospect** | the un-cut model: exceedance curves, contact distribution, area–depth curve, resource-vs-depth |
| **④ Well location** | entry/exit sliders, KPI strip (**proven mean is the headline**), the volume classes, A5/A6, live section |
| **⑤ Location sweep** | B0–B3, B6, optimum finders |
| **⑥ Risk & report** | chance table, reference contour, allocation scheme, B4/B5, export |

**Settings that are explicit and never implicit:** risking convention · reference contour · allocation scheme · minimum column height · engine (reference grouping vs proven/possible decomposition).

**Persistent notes on tabs ④–⑥:**
> *Single HC-water contact only. A prospect with both a gas–oil and an oil–water contact, where a well may test one and not the other, is not represented.* (your 10.7 / Q6)
> *Vertical, depth-dependent risk is assumed to be already contained in the GeoX contact distribution. Building that distribution is the job of the HCWC Builder.*

### 7.1 The risking panel

On import, the failure detector runs and states plainly:

> *Detected 2 395 zero-volume trials (23.95 %) with a single sentinel contact at 3166.59 m and all HC quantities collapsed. This is chance-failure coding. POS implied by the trials = 0.7605.*
> ( ) Correct — trials are risked; use 0.7605 and lock the chance table to display-only
> ( ) No — trials are success-case only; apply my chance table on top
> ( ) The zeros are geometric, not chance failure; treat separately

The choice is echoed as a one-line provenance stamp on every page. There is no path through the app where the risking convention is implicit.

---

## 8 · Data import

A dedicated subsystem, because the GeoX export has real traps and every downstream number depends on the mapping.

| Trap | Detail | Handling |
|---|---|---|
| **Duplicate column headers** | Six names appear **twice** with **identical units strings** — the in-place and recoverable variants, prefix missing. Max difference in the pair: 133.5 × 10⁹ scf | positional disambiguation + value heuristic (in-place ≥ recoverable) + explicit user confirmation. Never match by name alone |
| **Two header rows** | Row 3 = quantity name prefixed with the segment name (`prospect A TEST.`), row 4 = units, data from row 5 | parse both, strip the prefix, keep units as metadata and display them on every axis |
| **Decimal convention** | your Instruction, Step Two: *"GeoX uses '.' as commas!"* | sniff separators, show the parsed first five rows for confirmation before committing |
| **Failure sentinels** | §2.2 | detector + the risking panel |
| **Trial count** | not fixed; your workbook hard-wires 10 000 in ~400 formulas | any n. Warn below 10 000; recommend ≥ 50 000. Rose used 20 000, Milkov 5 000 |

**Adapter interface** — each adapter declares `sniff` (confidence this adapter fits), `read` (→ canonical `TrialSet`), `validate` (→ QC report). One canonical object downstream:

| Field | Unit | Required | GeoX source |
|---|---|---|---|
| `trial` | — | yes | `TrialNumber` |
| `contact` | m TVDSS | **yes** | `HC water contact - result` |
| `resource` | MMboe | **yes** | `Recoverable.Accumulation size Total Resources` |
| `area` | km² | strongly | `Productive area` |
| `gross_pay` | m | optional | `Average gross pay` |
| `hc_grv`, `hc_pv` | 10⁶ m³ | optional | as named |
| `crest`, `spill`, `net_gross`, `porosity`, `thickness` | | optional | as named |

Ships with **`geox`** and **`generic_csv`** (interactive mapping, profile remembered per file signature). Documented, adapter-shaped stubs for RoseRA / MMRA, Petrel PPA, @RISK and Crystal Ball, so adding one later is a file rather than a refactor.

**QC report, shown before anything is computed:** n trials · units per column · % zero-volume and the failure verdict · contact-distribution spikes · does `HC GRV = area × gross pay`? · `A(z)` fit quality with residual SD · missing values · duplicate-name resolutions · trial-count adequacy. Each item pass / warn / fail; **a fail blocks the analysis tabs.** Downloadable as a text stamp that goes into the report export.

### 8.1 Test data

| File | Rows × cols | Purpose |
|---|---|---|
| `data/demo_prospectA_reduced.csv` | 10 000 × 7 | the 7-column paste — the everyday case |
| `data/demo_prospectA_full.csv` | 10 000 × 60 | the full GeoX export — exercises duplicate headers, unit rows, prefixes |

Extracted from your `Trials data` and `Sheet1`. The app opens on a demo selector — *Prospect A (reduced)* / *Prospect A (full export)* / *Upload your own* — so it runs with no setup.

> **Correction to v2 and v3.** Those versions said the two sheets were *two independent GeoX runs*, on the evidence that joining them on `TrialNumber` gives only 5.6 % agreement. Building the test suite disproved it: row for row the contacts, resources and areas are **identical**. It is one run exported twice. What differs is the `TrialNumber` column itself, which holds the same identifiers attached to different rows — so **`TrialNumber` is not a reliable key in this export**, and joining two GeoX exports on it will silently scramble them. Nothing in the codebase does; `tests/test_adapters.py` asserts both facts. The pair is still useful for exercising the importer (7 columns versus 60), but not for checking stability across runs.

Two synthetic generators to be added for cases the real data cannot exercise: a **correlated area–net-pay** case (your Q1 "both"), and a **success-case-only** file with no failure trials, to test the risking branch.

---

## 9 · Relationship to the HCWC Builder

```
..\HCWC builder\        →   GeoX   →   trial export   →   ..\WellVolPOS\
builds the contact                                        reads it back out
distribution                                              through r_location
```

Your two @RISK builder workbooks implement Hood's competing-limits method — separate distributions for charge-limited fill, closure, fault geometry, fault leakage, top and base seal capillary and continuity, and tilt-related spillage, combined by `MIN` over the active limits. All six of the depth-increasing mechanisms you described are already in there.

**Which is exactly why WellVolPOS must not model vertical risk.** Because those mechanisms live in the contact distribution, `r_location` **already carries them**. A separate "risk increases with depth" overlay inside this tool would count them twice — the same defect as §2.3, one level up.

That project, the papers, the reference workbooks and a concept sketch now live in `..\HCWC builder\`. **For WellVolPOS, the contact distribution in the GeoX trials is taken as correct**, and the app says so.

---

## 10 · Architecture

```
wellvolpos/
  app.py                              Streamlit entry, routing, sidebar state
  wellvolpos/
    io/adapters/{base,geox,generic}.py  TrialAdapter protocol, TrialSet
    io/qc.py                            QC report, GRV = A x h check, A(z) fit quality
    io/failure.py                       failure-case detector + POS_trials
    core/structure.py                   A(z) isotonic fit, z <-> A, apex handling
    core/groups.py                      REFERENCE engine — Rose/Excel trial assignment
    core/classes.py                     proven / possible decomposition
    core/chance.py                      r_location, reference contours, allocation schemes
    core/threshold.py                   minimum column height <-> area <-> volume percentile
    core/stats.py                       adaptive conditional percentiles, bootstrap CIs
    core/xlcompat.py                    PERCENTILE.INC/.EXC, PERCENTRANK.EXC — Excel-exact
    core/sweep.py                       location sweep, uncertainty reduction, inverse MEFS
    viz/{theme,figures,section}.py      shared with the HCWC Builder
    report/export.py                    XLSX / PNG / SVG / PDF / JSON case
  data/                                the two demo CSVs
  tests/
    test_excel_parity.py                the 15 verified numbers in §2.1
    test_failure_detect.py              2395 / 0.7605 / sentinel 3166.59, both demos
    test_groups.py                      attic|dry = 9.09, discovery mean = 16.52
    test_classes.py                     proven mean = 15.76 at 3500/3550
    test_chance.py                      r^(1/3) == 0.7708238778; Rose = 1.11 x crest-referenced
    test_threshold.py                   column height <-> volume mapping, and the ~3x spread
    test_adapters.py                    duplicate-header resolution, decimal sniffing
    test_axes.py                        every depth axis is y and inverted
```

Python 3.12 · Streamlit · pandas · numpy · scipy · scikit-learn (isotonic) · plotly · matplotlib · openpyxl · pytest. No database; session state plus JSON case files.

**Deployment (your Q7).** Local `streamlit run` now, built public-ready from day one: no absolute paths, no machine-specific config, pinned `requirements.txt`, a `Dockerfile`, a permissive licence, demo data shipped in-repo so a stranger can run it in one command. Uploads stay in memory and are never written to disk. The demo data is fictional and can go in a public repo unchanged — with a line in the README saying so.

**`test_excel_parity.py` is written first.** Your workbook is the specification; that test is what stops the port silently drifting from the tool your colleagues already trust.

---

## 11 · Build phases

| Phase | Content | Effort |
|---|---|---|
| **0 · Skeleton** | Repo, theme, GeoX adapter, QC + risking panel, demo selector, parity tests green | ~1.5 d |
| **1 · Reference engine** | Rose/Excel trial grouping · A3, A4, A5 · B3 · depth sweep. **Deliverable: your workbook, working, correct risking, no `#DIV/0!`, no manual axes** | ~2 d |
| **2 · Extension** | `A(z)` · proven/possible classes · A1, A2, A6 · B0, B1, B2 · live section | ~2–3 d |
| **3 · Chance & threshold** | Chance table · reference contours · allocation schemes · B4, B5 · minimum column height with its equivalences | ~1.5 d |
| **4 · Sweep & inverse** | B6 · optimum finders · bootstrap bands · sample-size diagnostics | ~1–1.5 d |
| **5 · Polish** | Export · dark mode · tooltips and tables · case save/load · synthetic generators · README and docs | ~1.5–2 d |

Phase 1 alone replaces the spreadsheet. Everything after is upside.

**Out of scope, recorded so the reasoning is not lost:**

- **Economics** — EMV, NPV, decision trees. Schneider et al. (2023) Eq. 3, Figs 13–15 and the NPV appendix give a complete recipe if you change your mind. Seam left at `core/economics.py`; roughly 2 days.
- **Per-element manual location override** — §5.1.
- **HCWC distribution building** — its own project, §9.
- Multi-segment aggregation · multiple contacts (GOC + OWC) · multi-prospect comparison · built-in Monte Carlo.

---

## 12 · Decisions of record

| # | Decision | Where |
|---|---|---|
| 1 | Volume below the reservoir exit is **not proven** — reported as "Possible — below reservoir exit". Headline KPI = proven mean, 15.76 MMboe | §4 |
| 2 | Trials contain chance failures; `POS_trials = 0.7605`. **Confirmed against the GeoX project** | §2.2 |
| 3 | Reference contour default = **crest / apex**, with P90 area (Rose) as an option | §3.2 |
| 4 | Per-element manual location override **not built**; reasoning recorded for later | §5.1 |
| 5 | Units: **MMboe, m, km² only** | §6.4 |
| 6 | Minimum flowable = **minimum column height** below apex, with the volume-percentile equivalence displayed | §5.2 |
| 7 | Engine: Rose/Excel trial grouping is the **reference**; proven/possible split is an **extension**, both shown | §3.1 |
| 8 | **Depth always on y, inverted**, shared across a row | §6.1 |
| 9 | Vertical/depth-dependent risk belongs in the HCWC Builder, **never** here | §9 |
| 10 | Input: GeoX trial export, one prospect segment per session | §8 |

---

## 13 · References

**Read in full**

- Schneider, M., Citron, G.P., Haryott, P. & Cook, D. (2023) *Drilling an exploration prospect downdip: quantifying the trade-offs between chance of success and associated resource potential.* AAPG Bulletin 107(5): 743–759. DOI 10.1306/09232222051. Gold Open Access, CC-BY. — **the definitive reference**
- Milkov, A.V. (2021) *Reporting the expected exploration outcome: when, why and how the probability of geological success and success-case volumes for the well differ from those for the prospect.* J. Pet. Sci. Eng. 204: 108754. DOI 10.1016/j.petrol.2021.108754.
- Schneider, M. & Cook, D.M. Jr. (2017) AAPG Search & Discovery #42102 (poster), and the Rose & Associates long-form version of May 2021 with Equations 1–3.
- Haskett, W.J. (2003) *Optimal Appraisal Well Location Through Efficient Uncertainty Reduction And Value Of Information Techniques.* SPE 84241, SPE ATCE Denver.
- Singh, V., Yemez, I., Izaguirre, E. & Racero, A. (2017) Am. J. Applied Sciences 14(2): 217–230. DOI 10.3844/ajassp.2017.217.230.

**Consulted online**

- [Rose & Associates — Pwell Implementation 2017–2021](https://www.roseassoc.com/pwell-implementation-from-2017-to-2021/) — vendor marketing, not peer-reviewed; MMRA 2018 → RoseRA 2021.
- [SLB — GeoX play and prospect assessment](https://www.slb.com/products-and-services/delivering-digital-at-scale/software/geox/geox-software-play-and-prospect-assessment) — no well-location or downdip analysis advertised; the gap this tool fills.
- [Longley, I. (27 Jan 2026) *Understanding the "Minimum Economic Field Size" concept*, GeoExpro](https://geoexpro.com/understanding-the-minimum-economic-field-size-concept-and-aggregating-targets/) — a MEFS cut raises the unrisked mean while lowering commercial chance; they do not balance out. Relevant because the app draws MEFS as a reference line to read probabilities against, and never applies it to the distributions.

Milkov's own literature survey notes there was no peer-reviewed treatment of well-versus-prospect POS besides his paper and the Schneider & Cook poster. My searching agrees: nothing newer than the 2023 Bulletin paper, and no open-source implementation of any of it.

**Filed in `Papers\`.** Column-height material has moved to `..\HCWC builder\Papers\`.

**Not obtained.** Milkov & Samis (2020, AAPG Bull. 104) and Samis & Milkov (2020), on the real-option value of untested up-dip volume after a dry hole. Both paywalled and not in the folder. If the attic / regret analysis becomes central, they are the next two to get.

**One filing note.** `O3_P156_SchneiderAPGCE2022ExtendedAbstract.pdf` is by **Fred** Schneider, not Mark Schneider of Rose, and concerns machine-learning prospect ranking (PROMETHEE + neural networks). Unrelated to Pwell; filed with a marker in the filename.
