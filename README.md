# WellVolPOS

**A prospect has a chance of holding hydrocarbons. A well has a chance of finding
them. They are not the same number, and only one of them is usually written down.**

---

## In plain terms

Before anything is drilled, a prospect is assessed. How likely is it that oil or gas
is there at all, and if it is, how much? That work produces a probability of success
and a range of volumes, and those numbers travel — into the portfolio, the partner
meeting, the investment case.

Then a well is planned, and someone has to choose exactly where it goes.

A well is a single point. The prospect is a whole structure. A well placed high on
that structure will find hydrocarbons in almost every case where they are present,
but it may only touch the top of the accumulation. A well placed lower down proves
much more if it works — and misses entirely in every case where the hydrocarbons did
not fill that deep. Same prospect, same assessment, different question being asked.

So the chance that *this well* succeeds is not the chance that *the prospect* works.
It is lower, always, and by an amount that depends entirely on where the well is put.

WellVolPOS computes that second number and reports it beside the first, without
mixing them together.

---

## The issue this is meant to help with

This is offered as something worth checking, not as a diagnosis. It may not apply
where you work.

A prospect can be assessed carefully and well. The chance factors can be defensible,
the volume distribution can be honest about its uncertainty, and the spatial
description can be sound. None of what follows is a criticism of that work.

The difficulty is one of sequence. The prospect probability is produced early,
because that is when it is needed. The well location is settled later, and often for
good reasons that have nothing to do with the subsurface case — rig schedule, a
shallow hazard, standing off a boundary fault, a partner's preference, a permit
boundary. By the time the location exists, the number that everyone has been quoting
was calculated without it.

**The well then tests a sub-population, not the prospect.** Every trial in the model
is a possible version of the prospect. A well at a given depth can only succeed on
the subset of those versions where the contact happens to lie below it. The rest are
not failures of the prospect — they are outcomes this particular well was never able
to see. What the tool computes is a conditional probability: the chance of success
*given that you drilled here*.

And there is a human consequence that is worth saying out loud. Deciding to drill
down-dip to prove commercial volumes lowers the chance of finding anything. If that
lower chance was never stated, the dry hole is measured against the prospect number,
and the person who assessed the prospect is the one who looks wrong. The geology may
have been right and the location may have been reasonable; what was missing was the
number that connects them.

---

## How it works

Start from a stochastic model of the prospect — a GeoX run, or any comparable
simulation — exported as its trials. Ten thousand of them, typically. Each trial is
one internally consistent version of the prospect: a hydrocarbon–water contact, a
productive area, a recoverable volume.

The **contact distribution across those trials is the important part.** It describes
how deep the hydrocarbons might reach, and so, indirectly, how much of the structure
is filled. If it was built properly it already carries the reasons a deep fill is
less likely than a shallow one — a larger closure is harder to charge, a longer
column tests the seal harder, and further down-dip there are more faults, more leak
points and more doubtful pinch-outs, with reservoir presence and quality varying
across the structure. All of that belongs in the contact distribution. This tool
reads that distribution; it does not add to it, and it does not re-risk it.

The rest is a count. Your well enters the reservoir at a chosen depth. A trial is a
success for your well when its contact lies deeper than that entry. The fraction of
success trials that qualify is the location factor:

```
r_location = P(contact deeper than the well | hydrocarbons present)
P_well     = POS_prospect × r_location
```

`r_location` is the only quantity the well's position controls. `POS_prospect` is the
only quantity it does not. **They are never multiplied into a single reported
number**, because a decision needs to know which part it can still change.

---

## What the tool answers

**How likely is this well to work?** Prospect chance and well chance side by side,
with the location factor between them, swept across every entry depth so the cost of
moving is visible rather than inferred.

**If it works, what will it have proven?** A discovery does not demonstrate the whole
accumulation — only what the well penetrated. The tool splits every trial into what
the well proves, what stays unproven below it, and what sits up-dip and untouched.

**If it is dry, what did we leave behind?** A dry hole in a charged prospect leaves an
attic. The tool reports how large that is likely to be, and how likely it is to be
material rather than a curiosity.

**How likely is a commercial result, not just a discovery?** Finding hydrocarbons and
finding a developable accumulation are different questions with different answers, and
they favour different locations. Both are computed.

**Where does the well have to go to prove a given volume?** Name the volume; the tool
returns the shallowest depth that demonstrates it, and what that costs in chance.

**Between which depths is the well defensible at all?** Deep enough to prove something
commercial, shallow enough not to strand something commercial up-dip. Those two bounds
are usually more useful than any single optimum.

**Which risk elements carry the penalty?** The location factor can be attributed back
to charge, closure, reservoir or retention under several conventions, all of which
give the same well chance and differ only in the attribution.

It does not replace the stochastic model; it re-cuts that model's output against a
well. It does not build the contact distribution. It does not do economics.

---

## Your own data

The app opens on bundled demo data, so there is nothing to prepare to try it. To use
your own, choose **Upload your own…** in tab ①. A GeoX export is recognised
automatically; anything else goes through a generic reader that proposes a column
mapping and asks you to confirm it, rather than guessing.

**Uploaded files are never written to disk.** The file is read into memory, passed
straight to the reader, and exists only for that browser session. There is no upload
folder, no cache, no temporary file — a check in the test suite keeps it that way.

Two things to be aware of, so the claim is not stronger than it is:

- **Run it locally and your data never leaves your machine.** That is the mode to use
  for anything confidential.
- **On a hosted deployment**, the bytes are held in the memory of whatever server the
  app runs on for the duration of the session. Nothing is stored, but it is somebody
  else's machine. Host it yourself if that matters.

---

## The app

Every image below is the bundled prospect C, at one well, at the shipped defaults.

### The answer, before the working

![Tab ④ headline and the outcome tree](docs/screenshots/4.1_outcome-tree-and-headline.png)

One sentence, then the four things that can actually happen. The blue slice is the
case worth dwelling on: hydrocarbons present, but sitting entirely up-dip of where we
drilled. It is recorded as a dry hole. It is not a failure of the prospect.

### What the location costs, and what it buys

![3.5 · volume split vs location](docs/screenshots/3.5_volume-split.png)

What a discovery would prove rises as the well goes deeper — and so does what a dry
hole would leave behind. Each is conditional on its own outcome, so neither can be
read against the other without saying which case you are in.

![3.6 · chance vs regret](docs/screenshots/3.6_chance-vs-regret.png)

The falling chance of success against the rising chance that a dry hole strands
something material. Where they cross is where being wrong starts to cost more than
being right is likely.

### Where to drill

![Candidate depths, with the floor and the ceiling](docs/screenshots/3_candidate-depths.png)

Two rows bound the answer; the others only optimise something. Several optima land on
the same depth, and that is real rather than a bug — high on the structure, every
charged version of the prospect is a success, so any criterion that ignores the
volume threshold cannot tell those depths apart.

![3.11 · chance-weighted resource](docs/screenshots/3.11_chance-weighted.png)

Chance times volume peaks somewhere in the middle. The two markers are different
questions — what the accumulation holds, against what this well would actually
establish.

![3.3 · uncertainty reduction](docs/screenshots/3.3_uncertainty-reduction.png)

An appraisal view: not what the well finds, but what it settles. It peaks deeper than
the chance-based optima and never refers to a threshold volume at all.

### The volumes, five ways

![Tab ④ volume classes, the section and the exceedance curves](docs/screenshots/4_volume-classes.png)

The volume classes nest inside one another, and the section shows why. On the
exceedance curves the two probabilities are drawn where the risked curves *begin*,
rather than as labels beside them — so the gap between them is the location penalty,
measured.

![4.8 · conceptual map view](docs/screenshots/4.8_map-view.png)

The same split in plan: what lies up-dip of the well, what lies between entry and
exit, and what lies below. The shape is illustrative; the areas and depths are not.

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

## A window, not an optimum

Several criteria can all be maximised at the shallowest depth in the sweep, and often
are. High on the structure every charged version of the prospect is a success, so any
criterion that does not involve a volume threshold is *indifferent* across that whole
band — and a list of optima becomes several copies of one depth.

So the tool leads with a **window**. Two quantities bound it, and both are read as
guarantees rather than as first crossings:

| | |
|---|---|
| **floor** | the shallowest depth from which a discovery still demonstrates a commercial volume. A second, more conservative floor asks the same of a *poor* discovery. |
| **ceiling** | the depth past which a dry hole would have left a commercial volume untested up-dip. |

Either end can be absent, and that is reported rather than papered over. The floor can
also come out deeper than the ceiling, which is a finding about the prospect rather
than an error: at that threshold, no location both proves a commercial volume and
avoids stranding one.

The optima are listed underneath — best chance, best chance-weighted volume, best
chance-weighted *proven* volume, best commercial chance, best risk-adjusted volume,
best odds subject to a commercial confidence, and the appraisal optimum. Each carries
the span of depths within a couple of per cent of its own best, because on a nearly
flat curve a single depth is false precision.

---

## Data

`data/` ships two prospects, both safe to publish.

**Prospect C** is the default. It is a real export with every depth shifted by one
constant, so the file names no location. That is a rigid translation and changes
nothing the tool says: column heights are differences, the area–depth curve is the
same curve moved, and the location factor counts contacts either side of a depth that
moved with them. A test asserts that invariance rather than assuming it. It
anonymises *where*, not *how much*.

**Prospect A** is fictional, and ships both as a seven-column paste and as a full
export — so the everyday case and the duplicate-header trap both stay exercised. It is
what the parity suite is locked to. One trap worth knowing: the two forms carry the
same trial identifiers attached to *different rows*, so joining two exports on trial
number silently scrambles them. Nothing here does, and a test says so.

`io/synthetic.py` generates further files for branches one real prospect cannot
exercise. Their closure is a cone, so the area–depth relationship is known exactly —
the one thing real data cannot offer, and what let a sensitivity in the
reservoir-thickness inversion be found.

---

## Conventions that are never implicit

Each of these changes the numbers, so each is an explicit setting rather than a
default buried in the code:

- **Risking convention** — are the trials already risked, or success-case only? Asked
  at import, with the evidence shown.
- **Reference contour** for the location factor — crest/apex, or P90 area.
- **Risk-element allocation** — none, equal cube root, or all to closure. All three
  give the *same* well chance; only the attribution differs.
- **Assessment minimum** — a minimum column height below the apex.
- **Engine** — whole-trial grouping, or the per-trial volume split. Both are shown;
  neither is labelled "correct".

---

## What comes out

Tab ⑤ builds four artefacts from **one** assembled result, so the workbook, the PDF
and the figures cannot disagree with each other or with the screen:

| Format | For |
|---|---|
| **XLSX** | The reviewer who wants to check the arithmetic. Values only — a formula in an exported workbook is a second implementation of the same calculation. |
| **PDF** | The well proposal. A stamped cover page, then every figure, one per page, in the order the app shows them. |
| **PNG / SVG zip** | Slides. The provenance travels inside the archive, because a figure dropped into a deck is separated from its caption immediately. |
| **JSON case** | Reopening the session. Settings only, never results — so a reloaded case cannot show numbers this build would not produce. |

Every artefact carries the same stamp: the prospect chance **and where it came from**,
the location factor, the well chance, the well, the reference contour, the allocation
scheme and the threshold volume.

Figures can be drawn two ways, and the numbers are identical either way. matplotlib is
the default and needs nothing extra; choosing plotly renders the figures the app
itself draws, via [kaleido](https://pypi.org/project/kaleido/), so a figure in the
document is the figure that was on screen. kaleido drives a headless browser and is a
large download, so it is optional — without it, that option disables itself and says
which package is missing.

---

## Layout

```
app.py                       Streamlit entry point — six tabs
wellvolpos/
  io/adapters/               trial-file readers; add a simulator by adding a file
  io/adapters/generic.py     the fallback reader; proposes a mapping, never assumes one
  io/units.py                unit validation: reject, never convert
  io/failure.py              chance-failure detector
  io/qc.py                   the report that gates the analysis tabs
  io/anonymise.py            shift a real export's depths so it can be published
  core/structure.py          the area–depth curve, recovered from the trials themselves
  core/groups.py             whole-trial grouping (Schneider et al. 2023)
  core/classes.py            the per-trial volume split
  core/summary.py            the headline, the drilling window, and the optima
  core/mefs.py               every volume read against the threshold
  core/dependence.py         what the exit moves, and whether the geometry is a vertical well
  core/utility.py            certainty equivalent, and the commerciality hurdle
  core/chance.py             the location factor, reference contours, risk allocation
  core/reservoir.py          reservoir thickness, back-calculated from pay
  core/rose.py               No Regrets, Pmcfs(well), Pc(well)
  core/sweep.py              both sweeps, and the inverse
  core/stats.py              bootstrap and Wilson intervals, sample-size diagnostics
  viz/theme.py               one palette, one styling entry point, the depth-axis rule
  viz/interactive.py         the plotly figures — what the app draws
  viz/figures.py             the matplotlib twins — what the export draws
  report/export.py           one result, four formats, two renderers
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

Every source is listed in tab ⑥ with whether it was read directly or cited through
another work.

## Licence

MIT.
