"""WellVolPOS — Streamlit entry point.

Phases 0–5. Six tabs, ordered so a reader moves **explore, then evaluate**:

    ① Setup and input     what am I working with, is it sound, and what
                          conventions am I using?
    ② Prospect            what is this prospect, before any well?
    ③ Where to drill      where should the well go? — every figure that sweeps depth
    ④ At this well        what do I get at the depth I chose?
    ⑤ Risk & report       attribution, the summary table, the export
    ⑥ Theory & guide      what any of it means

③ and ④ were the other way round until 2026-08-11, which put the answer for one
depth in front of the material that informs choosing it.

**This module is the orchestrator, not the app.** Each tab's body lives in
``wellvolpos/ui/tab*.py`` with one ``render(ctx)`` entry point; what stays here is
the genuinely global part — page config, the sidebar, loading the trials, and the
single resolution of which POS is in force. It was 1 629 lines in one file until
2026-08-11, and a change to tab ② meant scrolling past tab ①. The split moved no
numbers; ``tests/test_app_shell.py`` drives the real app and is the guard.

Two ordering facts are load-bearing and survive the split:

* **Tab ① is built in two halves, around the sidebar**, because the entry/exit
  sliders take their range from the chosen file's contact column. So the file must
  be picked before the sidebar exists, and the QC report is appended to the same
  tab afterwards.
* **Tab ② owns the eight chance widgets, but they are read here**, before it
  renders, because ``POS_prospect`` is needed first. The widgets own the keys, so
  a change reruns and the read at the top sees the new value — no lag, no second
  copy of the state. See ``wellvolpos/ui/context.py``.

Tab ① gates the rest: it carries the trial-data selector, the import summary, a
preview of the trials, the QC report, the risking convention and all eight chance
inputs — four at play level and four conditional on the play. **Tab ② is well-free
on purpose**: A1 and A4 are drawn there without the entry/exit rules and without
A1's shaded volume classes, because those need a well and the same figures appear
on ④ with them. Reference contour and allocation scheme are sidebar-level
conventions, per CLAUDE.md's "never implicit" rule.

Figures are the interactive (plotly) ones from ``wellvolpos.viz.interactive``;
the matplotlib set in ``wellvolpos.viz.figures`` is the export path and both
are styled from ``wellvolpos.viz.theme``. Each figure is a standalone plot
rather than a panel of a merged grid, so a row is made readable across by
handing every figure in it the same ``row_zlim`` and letting them share the
one panel height — that is what puts the depths level.

Run with:  streamlit run app.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from wellvolpos.core import (
    ELEMENTS,
    SCHEME_LABELS,
    SHIPPED_SCHEMES,
    AreaDepth,
    ReferenceContour,
    check_area_pay_correlation,
    group_trials,
    p_well,
    split_trials,
    thickness_from_pay,
)
from wellvolpos.io.adapters import (
    CANONICAL_FIELDS,
    GenericCsvAdapter,
    Source,
    propose,
    score_adapters,
)
from wellvolpos.io.adapters import signature as adapter_signature
from wellvolpos.report.case import Case
from wellvolpos.ui import Ctx
from wellvolpos.ui import tab2_prospect, tab3_where, tab4_well, tab5_report
from wellvolpos.ui import tab6_guide
from wellvolpos.ui.common import badge as _badge
# One copy of the chance-table defaults, in the module that owns the conventions.
# app.py had its own duplicate pair plus two help dicts nothing read -- left over
# from the tab split, and two copies of a default is a drift waiting to happen.
from wellvolpos.ui.conventions import CHANCE_DEFAULTS, PLAY_DEFAULTS
from wellvolpos.ui.loading import load as _load
from wellvolpos.core.rose import AT_WELL_WINDOW_M
from wellvolpos.ui.well import deviation_caption, read_well, well_editor
from wellvolpos.ui.tabstyle import apply_metric_size as _shrink_metrics
from wellvolpos.ui.tabstyle import inject as _inject_tab_style
from wellvolpos.viz import (
    AREA_SCALES,
)

DATA = Path(__file__).parent / "data"
# Prospect B first, so it is the default (Lars, 2026-08-10). It comes from the 2018
# macro workbook and is the better demo in almost every way: 43 clean columns with
# no duplicate names, a units row that the unit check can actually verify, and both
# `Reservoir thickness` and `Spill point depth`. It is also **success-case only**,
# with no chance failures at all, so opening on it exercises the risking branch that
# no real file previously reached — POS comes from the chance table and the footer
# says so. Prospect A stays: it is what the parity suite is locked to, and its
# duplicate-header trap is the one the reader most needs to keep passing.
#
# **Prospect A leads, and prospect B is not in the repository** (2026-08-15). A's
# provenance is confirmed fictional and safe to publish; B's is *unconfirmed*, and
# CLAUDE.md has treated it as the licensee's since it arrived. Publishing data whose
# origin nobody has established is not a risk worth taking for a demo file, so it stays
# in the working folder and out of git.
#
# Built by filtering on existence rather than listed flat: a clone has A only, this
# machine has both, and neither should need a different app.py.
# **The "(default)" marker is computed, not typed.** Which file leads depends on which
# ones exist, so a hardcoded label goes stale the moment the list is filtered -- and a
# label reading "(default)" on the second entry is worse than none.
_DEMO_ORDER = (
    # **Prospect C is prospect B, moved.** Every depth shifted by one constant so the
    # file names no location, and nothing else touched -- see io/anonymise.py. The
    # translation is rigid, so r_location, P_well and every volume are identical at a
    # well moved by the same offset; a test asserts that rather than trusting it.
    # C ships; B stays in the working folder.
    ("Prospect C — reduced (7 columns)", DATA / "demo_prospectC_reduced.csv"),
    ("Prospect B — reduced (7 columns)", DATA / "demo_prospectB_reduced.csv"),
    ("Prospect B — full export, 43 columns", DATA / "demo_prospectB_full.csv"),
    ("Prospect A — reduced (7 columns)", DATA / "demo_prospectA_reduced.csv"),
    ("Prospect A — full GeoX export (60 columns)", DATA / "demo_prospectA_full.csv"),
)
_present = [(label, path) for label, path in _DEMO_ORDER if path.exists()]
DEMOS = {
    (f"{label}, default" if i == 0 else label): path
    for i, (label, path) in enumerate(_present)
}

# Stable keys for the risking convention. The app branches on these; the text
# beside them is presentation only.
CONVENTION_KEYS = ("trials_risked", "success_case_only", "geometric")
CONVENTION_LABELS = {
    "trials_risked": (
        "Correct — trials are risked; use the implied POS and lock the chance table to display-only"
    ),
    "success_case_only": "No — trials are success-case only; apply my chance table on top",
    "geometric": (
        "The zeros are geometric (contact above crest), not chance failure; treat separately"
    ),
}
CONVENTION_PROVENANCE = {
    "trials_risked": "trials (chance table display-only)",
    "success_case_only": "chance table",
    "geometric": "chance table (geometric reading not yet implemented)",
}

st.set_page_config(page_title="WellVolPOS", layout="wide", page_icon="🛢")

#: C2 carries the nesting braces below its zero line, so it needs more room than a
#: panel in a row. Its own constant rather than a multiple of PANEL_HEIGHT, because
#: PANEL_HEIGHT means "the height a row of depth panels shares" and C2 is in a row
#: with nothing.
C2_HEIGHT = 620

#: C1 is a thumbnail above C2 -- there to be recognised, not read -- so it gets a
#: fraction of a panel's height rather than the shared one.
C1_THUMB_HEIGHT = 250


# Dark mode was built and then dropped, on Lars's instruction (2026-08-10). The
# app draws in the light palette only. ``viz/theme.py`` keeps its dark palette and
# every figure still takes a ``dark`` keyword -- they cost nothing, the
# colour-vision-deficiency test in ``tests/test_axes.py`` exercises both, and the
# export path can use them -- but nothing here selects it, and it should not be
# rebuilt without asking. What made it not worth having was that Streamlit's own
# chrome follows its theme setting while the figures had to be told separately, so
# the two could disagree; one palette cannot.


# ------------------------------------------------------------------ loading








# ------------------------------------------------------------------- tabs
# Declared before anything writes into them, because the trial-data selector now
# lives in tab ① and everything downstream depends on the file it picks. The
# sidebar keeps only the well geometry and the conventions -- the things a reader
# changes repeatedly while looking at a figure.
st.title("WellVolPOS")
st.caption("Well POS and volume, from a stochastic prospect model")
# Explore, then evaluate (Lars, 2026-08-11). The sweep tab and the well-result tab
# were the other way round, which put the answer for one depth before the material
# that informs choosing it. The order now follows the question a reader is asking:
# what do I have, what is this prospect, where should the well go, what do I get
# there, what is it worth, and what does any of it mean.
tabs = st.tabs(
    [
        "① Setup and input",
        "② Prospect",
        "③ Where to drill",
        "④ At this well",
        "⑤ Risk & report",
        "⑥ Theory & guide",
    ]
)
_inject_tab_style()
# Metric values one size down: the default is a headline size, right for one
# number and wrong for a strip of eight (Lars, 2026-08-18).
_shrink_metrics()

with tabs[0]:
    st.subheader("Trial data")
    choice = st.selectbox("Data set", list(DEMOS) + ["Upload your own…"])
    uploaded = None
    if choice == "Upload your own…":
        uploaded = st.file_uploader(
            "Trial export — GeoX, or any delimited file",
            type=["csv", "txt", "tsv", "dat", "xlsx", "xlsm"],
            help=(
                "Read in memory and never written to disk. A GeoX export is recognised "
                "automatically; anything else goes through the generic reader, which proposes "
                "a column mapping for you to confirm below."
            ),
        )

# Uploads are held as (name, bytes) and never written to disk — design plan §10,
# and it is the licensee's data. `Source` is what makes one code path serve both a
# demo file and an upload.
source = None
if uploaded is not None:
    source = Source.from_any(uploaded)
elif choice in DEMOS:
    source = Source.from_any(DEMOS[choice])

if source is None:
    # Nothing to configure yet: MEFS scales with the resource column and the
    # conventions need a loaded file. There is no sidebar to explain itself any more
    # (removed 2026-08-14), so the message goes in the tab where the controls will
    # appear -- which is also the tab the file is chosen in.
    with tabs[0]:
        st.info(
            "**Waiting for trial data.** The threshold and convention controls take their "
            "range from the loaded file, so they appear here once one is chosen. Well "
            "locations are set on **tab ③**, where they are compared."
        )
    with tabs[0]:
        st.info(
            "Choose a demo dataset above, or upload your own trial export. "
            "Nothing is written to disk."
        )
    st.stop()

# One mapping override per *header signature*, so correcting a mapping once means
# the next export with the same columns is already right (design plan §8's
# "profile remembered per file signature"). Session-scoped: this app does not
# write the user's mapping choices anywhere.
try:
    sig = adapter_signature(source)
except Exception:
    sig = source.name
overrides = dict(st.session_state.get(f"_map_{sig}", {}))

ts, qc = _load(source.name, source.data, tuple(sorted(overrides.items())))
zmin, zmax = float(ts.col("contact").min()), float(ts.col("contact").max())

with tabs[0]:
    scores = score_adapters(source)
    st.caption(
        "Reader: **"
        + ts.source
        + "** — "
        + ", ".join(f"{a.name.split()[0]} {s:.2f}" for s, a in scores)
        + " confidence. The generic reader is capped at 0.30 so it can never outrank an "
        "adapter that actually recognised the format."
    )
    generic = ts.source == GenericCsvAdapter().name
    if generic:
        with st.expander("Column mapping — confirm before relying on anything downstream",
                         expanded=bool(overrides) is False):
            st.caption(
                "This file is not a GeoX export, so the mapping below was **proposed** from the "
                "headers rather than known. A wrong mapping is the most damaging thing that can "
                "happen at import: every number downstream is then computed from the wrong "
                "column and nothing looks broken."
            )
            proposal = propose(source, mapping=overrides)
            cols = ["— none —"] + proposal.columns
            picked: dict[str, str] = {}
            grid = st.columns(3)
            for i, canon in enumerate(("contact", "resource", "area", "gross_pay",
                                       "hc_grv", "thickness")):
                cur = proposal.mapping.get(canon)
                with grid[i % 3]:
                    sel = st.selectbox(
                        f"{canon} ({CANONICAL_FIELDS[canon][0] or '—'})", cols,
                        index=cols.index(cur) if cur in cols else 0,
                        key=f"map_{sig}_{canon}",
                        help=proposal.why.get(canon, "not found in the headers"),
                    )
                if sel != "— none —":
                    picked[canon] = sel
            weak = proposal.needs_confirmation
            if weak:
                st.warning(
                    "Matched on a weak header match, so worth a look: "
                    + ", ".join(f"**{f}** ← `{proposal.mapping[f]}`" for f in weak)
                )
            if st.button("Use this mapping", key=f"apply_map_{sig}"):
                st.session_state[f"_map_{sig}"] = picked
                st.rerun()

# Case save/load sits here, beside the trial data, because both are *inputs* to
# the session and having two upload boxes in two different places invited the
# reasonable question of whether they did the same thing. They do not: a case
# carries settings, a trial file carries data.
#
# It has to run *before* the sidebar widgets it restores. A Streamlit widget reads
# session_state only when it is first created, so writing the values and then
# rerunning is the only order in which a loaded case actually reaches the
# controls; setting them afterwards silently does nothing.
with tabs[0]:
    st.divider()
    st.subheader("Case — the settings, not the data")
    st.caption(
        "A **case** is every choice that turns this trial file into an answer: the well, the "
        "threshold volume, the four conventions and the chance table. It carries **no results** — "
        "every number is recomputed on load, so a reopened case cannot show you figures this "
        "build would not produce. It is not another way to load trials."
    )
    cc1, cc2 = st.columns(2)
    with cc1:
        up = st.file_uploader("Load a case (.json)", type=["json"], key="_case_upload")
        if up is not None and st.button("Apply this case", key="_case_apply"):
            try:
                loaded = Case.from_json(up.getvalue())
            except ValueError as e:
                st.error(str(e))
            else:
                st.session_state.update({
                    "w_entry": float(np.clip(loaded.entry, zmin, zmax)),
                    "w_exit": float(np.clip(loaded.exit, zmin, zmax)),
                    "w_mefs": loaded.mefs,
                    "w_ref": ReferenceContour(loaded.reference),
                    "w_scheme": loaded.scheme,
                    "w_area_scale": loaded.area_scale,
                    "w_map_interval": loaded.map_interval,
                    "w_map_azimuth": int(round(loaded.map_azimuth_deg)),
                    **{f"w_play_{el}": v for el, v in loaded.play_elements.items()},
                    "risking_convention": loaded.risking_convention,
                    "_case_warnings": loaded.check_against(ts),
                    "_case_loaded": loaded.dataset or "a case file",
                })
                for el, v in loaded.chance_table.items():
                    st.session_state[f"w_chance_{el}"] = v
                st.rerun()
        if st.session_state.get("_case_loaded"):
            st.success(f"Settings restored from **{st.session_state['_case_loaded']}**.")
            for w in st.session_state.get("_case_warnings", []):
                st.warning(w)
    _case_save_slot = cc2

# **The well geometry sits in tab (1) with the other settings**, and is read here.
# The same arrangement the chance table uses, and it works for the same reason:
# `entry` and `exit_` are needed before any tab renders, a widget owns its key, and
# the next rerun sees the change with no second copy of the state.
#
# It was four candidate locations on tab (3) for one day (2026-08-13/14) and Lars
# removed it -- see CLAUDE.md. With one well the sliders are a *setting*, so they
# belong beside MEFS and the conventions rather than on the tab that sweeps depth.
well = read_well(ts, zmin, zmax)
entry, exit_ = well.entry, well.exit
# The at-the-well window, read here for the same reason the chance table is: it feeds
# a *swept curve* on tab ③ as well as the metric on tab ④, and the sweep runs before
# either tab renders. Seeded with setdefault so the widget on tab ④ still owns the key.
at_well_window = float(st.session_state.setdefault("w_atw_window", AT_WELL_WINDOW_M))
with tabs[0]:
    st.divider()
    st.subheader("Import")
    c1, c2, c3 = st.columns(3)
    c1.metric("Trials", f"{ts.n_trials:,}")
    c2.metric("Adapter", ts.source)
    c3.metric("Segment", ts.prospect or "—")
    st.markdown("**Column mapping** — confirm before relying on anything downstream.")
    st.dataframe(
        pd.DataFrame(
            [
                {"canonical field": k, "source column": v, "units": ts.units.get(k, "")}
                for k, v in ts.source_columns.items()
            ]
        ),
        hide_index=True,
        width="stretch",
    )
    if ts.notes:
        for n in ts.notes:
            st.info(n)
    st.caption("Units are fixed: MMboe, m, km².")

    st.divider()
    st.subheader("Trial preview")
    n_preview = st.number_input(
        "Rows to show", min_value=1, max_value=int(ts.n_trials), value=min(20, int(ts.n_trials)),
        step=10,
        help=(
            "The canonical columns after mapping, as the rest of the app sees them — not the raw "
            "file. Worth a look before trusting anything downstream: the failure trials are the "
            "ones with every hydrocarbon quantity at exactly zero."
        ),
    )
    st.dataframe(ts.frame.head(int(n_preview)), width="stretch")
    st.caption(
        f"First {int(n_preview):,} of {ts.n_trials:,} trials, in canonical form. "
        f"TrialNumber is shown but is **not** a reliable key in a GeoX export — never join on it."
    )

    # **The well goes above QC** (Lars, 2026-08-15, design review). It sat seventh of
    # eight blocks on this tab, below the file, the case, the mapping, the trial table,
    # the QC list and the zero-volume detector -- and it is the control a reader
    # touches every few seconds while the QC is read once. Its own `st.empty()`, for
    # the same reason the settings block has one: the widgets must be *created* before
    # the analysis tabs render, but they belong *here*, and `st.empty()` reserves the
    # position where it is declared.
    st.divider()
    _well_slot = st.empty()

    st.divider()
    st.subheader("Quality control")
    for c in qc.checks:
        st.markdown(f"{_badge(c.level)} **{c.name}** — {c.message}")

    st.divider()
    st.subheader("Zero-volume trials — what the detector found")
    # The *finding* stays here with the rest of QC; the *decision* moved to tab ②
    # (Lars, 2026-08-11). The split is the honest one: whether 2,395 trials have
    # every hydrocarbon quantity at exactly zero is a fact about this file, and
    # belongs beside the checks that establish it. How to read those zeros is a
    # judgement that sets POS_prospect, and POS_prospect is a property of the
    # prospect -- so it now sits on the prospect tab, next to the curve it risks.
    f = qc.failure
    if f and f.has_failures:
        st.markdown(f"**{f.summary()}**")
        with st.expander("Evidence"):
            for e in f.evidence:
                st.markdown(f"- {e}")
        st.info(
            "**How these zeros are read is set in tab ② (Prospect)**, together with the chance "
            "table, because that choice is what POS_prospect is and tab ② is where POS_prospect "
            "is drawn. The evidence above is what you read to make it."
        )
    else:
        st.markdown("No zero-volume trials — the export looks success-case only.")

    if qc.blocked:
        st.error("A check failed. The analysis tabs stay closed until it is resolved.")

if qc.blocked:
    st.stop()

with tabs[0]:
    # Declared at the foot of tab (1) and filled a few lines below. `st.empty()`
    # reserves its position where it is *declared*, so this is what puts the settings
    # after the import and QC material -- choose a file, check it, then set the
    # conventions. They have to be *created* before the grouping below, which needs
    # the reference contour, so declaration and filling are separated. Same device the
    # case-save button already uses.
    _settings_slot = st.empty()

# ---------------------------------------------------------------- the settings
# **No sidebar** (Lars, 2026-08-14). Everything that was in it is here, in tab ①,
# which is now "Setup and input": the file, the mapping, the QC, the threshold and
# the four conventions. One place to set a session up, and the six analysis tabs are
# then all output.
#
# They are still *created* before the tabs are rendered, because `mefs`, `ref`,
# `scheme` and `area_scale` are needed by every tab below -- so the widgets are
# written into a container declared inside tab ① earlier in the script. That is the
# same device the case-save button already uses, and it is what lets a control live
# on a tab while its value is available to the whole page.
with _well_slot.container():
    # **The well geometry is a setting**, and the one most often changed. It was in the
    # sidebar until 2026-08-13, then on tab (3) for a day as four candidate locations;
    # tab (3) sweeps *every* entry depth and argues about which to pick, so the pair
    # you settle on belongs with the settings rather than on the tab whose subject is
    # not having settled.
    st.subheader("The well")
    st.caption(
        "Where the well enters and leaves the reservoir. Tab ③ sweeps every entry "
        "depth and shows what each one buys; tab ④ is the write-up at the pair set "
        "here."
    )
    _deviation_slot = well_editor(well, zmin, zmax)

with _settings_slot.container():
    st.divider()
    st.subheader("Threshold and conventions")
    st.caption(
        "Never implicit — every one of these changes the numbers, and each is stamped "
        "in the footer on every tab."
    )
    _s1, _s2 = st.columns(2)
    with _s1:
        # MEFS scales with the prospect: 14 MMboe is a sensible threshold against
        # prospect A's 16 MMboe discovery mean and a rounding error against B's 121.
        _default_mefs = float(np.round(
            np.mean(ts.col("resource")[ts.col("resource") > 0]) * 0.85))
        mefs = st.number_input(
            "MEFS / MCFS (MMboe)", min_value=0.0, value=_default_mefs, step=0.5,
            key="w_mefs",
            help="Minimum economic field size, or Rose's minimum commercial field "
                 "size — the same threshold under two names, so one input serves "
                 "both. Where a house does separate them (economic at NPV = 0, "
                 "commercial adding strategic and contractual hurdles, so MCFS >= "
                 "MEFS), enter the one you are testing against. Drawn as a reference "
                 "line and never applied to the distributions.")
        ref = st.radio(
            "Reference contour for the location factor",
            [ReferenceContour.CREST, ReferenceContour.P90_AREA],
            format_func=lambda r: {"crest": "Crest / apex (Milkov 2021)",
                                   "p90_area": "P90 area (Rose)"}[r.value],
            key="w_ref",
        )
    with _s2:
        # Default to the equal-cube-root scheme, which is the one Lars's risk summary
        # is drawn against. Under "none" the location factor is reported separately
        # instead of being attributed to elements -- still available, but a poor
        # default for a table whose third column is meant to show the penalty.
        scheme = st.selectbox(
            "Risk-element allocation", SHIPPED_SCHEMES,
            format_func=lambda k: SCHEME_LABELS[k],
            index=list(SHIPPED_SCHEMES).index("equal_cube_root"), key="w_scheme",
        )
        # GeoX plots its area-depth curve against area squared, so that convention is
        # offered alongside ours. The transform is on the axis only -- every number the
        # tool computes stays in km2 (non-negotiable 4).
        area_scale = st.selectbox(
            "Area–depth x-axis", list(AREA_SCALES), index=0,
            help="area is this tool's default; area² is GeoX's convention; "
                 "√area straightens a cone.",
            key="w_area_scale",
        )
    st.caption(
        "Well locations are **not** here: they are set on tab ③, where candidates are "
        "compared against every swept figure."
    )

# The chance table and the risking convention live in **tab ②** now (Lars,
# 2026-08-11), moved out of tab ①. Both determine POS_prospect, and POS_prospect is
# a property of the prospect -- so the inputs sit on the prospect tab, beside the
# distribution they risk. That the risked A5 curve had gone missing on that very tab
# is the argument in miniature: the number and its consequence were two tabs apart.
#
# **Read here, created there.** Every number below comes out of session_state rather
# than off a widget, because the widgets are built further down the script than this
# and Streamlit runs top to bottom. That is not a workaround: it is what makes the
# ordering a *layout* decision instead of a computation one. The widgets own these
# keys, so changing one triggers a rerun and this read sees the new value at the top
# of it -- there is no lag and no second copy of the state.
#
# Charge, closure, reservoir and retention are judgements about the prospect: made
# before anyone picks a location and unchanged by picking one. The location factor is
# *computed* from the trials and the well's depth, so the summary that multiplies the
# two can only be assembled afterwards -- and that summary is in tab ⑤. Keeping the
# input and the summary apart is what stops the third column of that summary being
# read as something a person entered.
elements = {el: float(st.session_state.get(f"w_chance_{el}", CHANCE_DEFAULTS[el]))
            for el in ELEMENTS}
play_elements = {el: float(st.session_state.get(f"w_play_{el}", PLAY_DEFAULTS[el]))
                 for el in ELEMENTS}
play_chance = float(np.prod(list(play_elements.values())))

# The play chance multiplies the four elements: POS_prospect is the chance the
# play works *and* every element of this prospect does.
pos_from_table = float(play_chance) * float(np.prod(list(elements.values())))

# The convention is *seeded here*, before POS is read, not down in tab ② where the
# radio is built. Seeding it beside the widget left the first run after a file
# change reading the old default while the radio already showed the new one -- the
# footer said "from the chance table" under a radio saying the trials carry it.
# Widget defaults have to exist before the value is used, and the value is used
# here.
_f0 = qc.failure
if _f0 and _f0.has_failures:
    if "risking_convention" not in st.session_state:
        st.session_state["risking_convention"] = (
            CONVENTION_KEYS[0] if _f0.verdict == "chance_failure" else CONVENTION_KEYS[1]
        )
    risking_convention = st.session_state["risking_convention"]
else:
    # No zero-volume trials: there is nothing to interpret, so the convention is
    # *forced* rather than chosen -- and it is deliberately not written to session
    # state. Writing it there is how a value forced by one trial file survives into
    # the next one and silently overrides the choice that file deserves.
    risking_convention = "success_case_only"
# `is not None`, not truthiness: a file whose every trial failed gives
# pos_trials == 0.0, which is falsy, and would silently fall through to the
# chance table.
pos_trials = qc.failure.pos_trials if qc.failure else None
if risking_convention == "trials_risked" and pos_trials is not None:
    pos = pos_trials
    pos_source = "the trials (chance table is display-only)"
else:
    pos = pos_from_table
    pos_source = "the chance table"
pos_provenance = CONVENTION_PROVENANCE.get(risking_convention, "chance table")

groups = group_trials(ts, entry, exit_)
chance = p_well(ts, entry, pos, reference=ref)
has_area = ts.has("area")
if has_area:
    ad = AreaDepth.from_trials(ts.col("contact"), ts.col("area"))
    vc = split_trials(ts, ad, groups, entry, exit_)
# Filled here, drawn under the sliders in tab (1): the verdict needs A(z), which is
# only fitted at this point in the script.
deviation_caption(_deviation_slot, well,
                  thickness_from_pay(ts, ad).thickness if has_area else None)


# The correlation check warns rather than fails (decided 2026-08-10), because the
# assumption it tests belongs to the extension alone and blocking closed the
# reference engine with it. Warning in the QC list is not enough on its own: the
# reader who goes straight to tab ③ never sees that list, so the caveat is raised
# again wherever the split's own numbers are drawn.
_split_level, _split_message, _split_r = check_area_pay_correlation(ts)


def _split_caveat() -> None:
    if _split_level == "warn" and np.isfinite(_split_r) and abs(_split_r) >= 0.5:
        st.warning(f"**The proven/possible split is not defensible on this data.** {_split_message}")
    # The apportionment changed on 2026-08-11 and moved the headline numbers by
    # about 8 %, so it is stated wherever the split's own numbers are drawn rather
    # than only in the footer.
    if vc.apportionment == "area":
        st.info(
            "**Split apportioned by map area**, not on the wedge — this file carries no "
            "gross pay and no HC gross rock volume, so the reservoir thickness cannot be "
            "recovered and the wedge cannot be built. The area rule assumes uniform pay per "
            "unit area, which understates proven and overstates possible."
        )
    elif vc.n_thickness_assumed:
        st.caption(
            f"{vc.n_thickness_assumed:,} discovery trials could not resolve a reservoir "
            f"thickness from pay and were treated as **charged to base**, which is what the "
            f"thickness inversion flags them as."
        )

# --------------------------------------------------------------- the context
# Everything resolved once, then handed to each tab. See wellvolpos/ui/context.py
# for why `pos`, `vc` and the chance elements in particular must be *given* to a
# tab rather than re-derived inside one.
ctx = Ctx(
    ts=ts, dataset=choice, source=source, overrides=overrides, qc=qc,
    ad=ad if has_area else None, has_area=has_area,
    entry=entry, exit_=exit_,
    at_well_window=at_well_window,
    mefs=mefs, ref=ref, scheme=scheme, area_scale=area_scale,
    elements=elements, play_elements=play_elements, play_chance=play_chance,
    risking_convention=risking_convention, pos=pos, pos_source=pos_source,
    pos_from_table=pos_from_table, pos_trials=pos_trials,
    groups=groups, vc=vc if has_area else None, chance=chance,
    split_level=_split_level, split_message=_split_message, split_r=_split_r,
)

with tabs[1]:
    tab2_prospect.render(ctx)

with tabs[2]:
    tab3_where.render(ctx)

with tabs[3]:
    tab4_well.render(ctx)

with tabs[4]:
    tab5_report.render(ctx)

with tabs[5]:
    tab6_guide.render(ctx)

# Last, because a case is the state of every widget and the chance table on tab ②
# is only created above. It is *drawn* into the slot beside the case loader in
# tab ①, so save and load sit together where a reader looks for them.
tab5_report.render_case_save(ctx, _case_save_slot)


st.divider()
# Outside every tab container, so this one line is the provenance stamp the
# design plan (§7.1) requires on every page: there must be no path through the
# app where the risking convention is implicit.
st.caption(
    f"Single HC-water contact only — a prospect with both a gas–oil and an oil–water contact, "
    f"where a well may test one and not the other, is not represented. "
    f"Vertical (depth-dependent) risk is assumed already contained in the contact distribution; "
    f"building that distribution is the HCWC Builder's job. "
    f"Risking: POS {pos:.4f} from the {pos_provenance} "
    f"(play {play_chance:.2f}). "
    f"Reference contour: {ref.value}. Allocation: {scheme}. "
    f"Split apportioned on the {vc.apportionment}."
)
