"""WellVolPOS — Streamlit entry point.

Phases 0–5. Six tabs, all live. Tab ① carries the trial-data selector, the import
summary, a preview of the trials themselves, the QC report and the risking
question, and it gates the rest; the risking convention chosen there and the
chance table entered in tab ⑤ together determine POS_prospect (see the "Entered
here" comment below for why the chance-table widgets sit before the computation
that uses them). Reference contour and allocation scheme are sidebar-level
conventions, per CLAUDE.md's "never implicit" rule. Tab ⑤ ends in the export,
which builds every artefact from one assembled bundle.

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
    SUMMARY_COLUMNS,
    risk_summary,
    chance_from_counts,
    MIN_SUPPORT,
    SCHEME_LABELS,
    SHIPPED_SCHEMES,
    AreaDepth,
    ReferenceContour,
    allocate,
    check_area_pay_correlation,
    REPORT_PERCENTILES,
    class_percentiles,
    class_summary,
    expected_volume,
    describe_support,
    group_summary,
    group_trials,
    invert_volume_target,
    p_well,
    run_sweep,
    run_volume_sweep,
    split_trials,
    volume_target_curve,
)
from wellvolpos.io.adapters import (
    CANONICAL_FIELDS,
    GenericCsvAdapter,
    Source,
    propose,
    read_trials,
    score_adapters,
)
from wellvolpos.io.adapters import signature as adapter_signature
from wellvolpos.report import export as export_mod
from wellvolpos.report.case import Case, fingerprint
from wellvolpos.report.guide import render as render_guide
from wellvolpos.io.qc import run_qc
from wellvolpos.viz import (
    AREA_SCALES,
    PANEL_HEIGHT,
    pfig_a1_area_depth,
    pfig_a2_outcome_tree,
    pfig_a3_chance_decomposition,
    pfig_a4_resource_vs_depth,
    pfig_a5_exceedance,
    pfig_a8_contact_distribution,
    pfig_a6_overlap,
    pfig_b0_section,
    pfig_b1_volume_split,
    pfig_b2_chance_vs_regret,
    pfig_b3_uncertainty_reduction,
    pfig_b4_chance_waterfall,
    pfig_b5_allocation_dumbbell,
    pfig_b6_inverse,
    pfig_b7_frontier,
    pfig_b8_commercial_chance,
    pfig_b9_chance_weighted,
    pfig_c1_section,
    pfig_c2_exceedance,
    pfig_map_view,
    row_zlim,
    suggest_grid,
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
DEMOS = {
    "Prospect B — full export, 43 columns (default)": DATA / "demo_prospectB_full.csv",
    "Prospect B — reduced (7 columns)": DATA / "demo_prospectB_reduced.csv",
    "Prospect A — reduced (7 columns)": DATA / "demo_prospectA_reduced.csv",
    "Prospect A — full GeoX export (60 columns)": DATA / "demo_prospectA_full.csv",
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

#: Chance-table defaults (Lars, 2026-08-11). They were 1.0 each, which made
#: POS_prospect 1.0 and hid the whole conditional/unconditional distinction on the
#: default demo -- every risked curve coincided with its conditional twin. These
#: multiply to 0.432, so the app now opens with a POS worth reasoning about.
CHANCE_DEFAULTS = {"charge": 0.90, "trap": 1.00, "reservoir": 0.60, "retention": 0.80}
CHANCE_HELP = {
    "charge": "Chance that hydrocarbons were generated and migrated into the trap.",
    "trap": "Chance that a valid trap geometry and seal exist.",
    "reservoir": "Chance of effective reservoir presence and quality. **Exempt from the "
                 "location penalty**: a well that misses the column still saw the rock.",
    "retention": "Chance the accumulation was retained rather than lost after charge.",
}

#: C2 carries the nesting braces below its zero line, so it needs more room than a
#: panel in a row. Its own constant rather than a multiple of PANEL_HEIGHT, because
#: PANEL_HEIGHT means "the height a row of depth panels shares" and C2 is in a row
#: with nothing.
C2_HEIGHT = 620


# Dark mode was built and then dropped, on Lars's instruction (2026-08-10). The
# app draws in the light palette only. ``viz/theme.py`` keeps its dark palette and
# every figure still takes a ``dark`` keyword -- they cost nothing, the
# colour-vision-deficiency test in ``tests/test_axes.py`` exercises both, and the
# export path can use them -- but nothing here selects it, and it should not be
# rebuilt without asking. What made it not worth having was that Streamlit's own
# chrome follows its theme setting while the figures had to be told separately, so
# the two could disagree; one palette cannot.


# ------------------------------------------------------------------ loading
@st.cache_data(show_spinner=False)
def _load(name: str, data: bytes, mapping_items: tuple = ()):
    """Read and QC one trial file, keyed on its *bytes*.

    Bytes rather than a path, because an upload is never written to disk (design
    plan §10) and there is no path to key on. ``st.cache_data`` hashes them
    happily, and at these sizes -- the largest demo export is under 3 MB -- the
    cost of holding them is nothing next to re-parsing 10 000 rows on every
    slider drag.

    ``mapping_items`` is a sorted tuple rather than a dict so it is hashable: it
    carries a manual column mapping for the generic reader, and a different
    mapping is a different import.
    """
    src = Source(name=name, data=data)
    adapter = GenericCsvAdapter(mapping=dict(mapping_items)) if mapping_items else None
    ts = read_trials(src, adapter=adapter)
    return ts, run_qc(ts)


@st.cache_data(show_spinner=False)
def _volume_sweep(name: str, data: bytes, mapping_items: tuple,
                  pos: float, gap: float, mefs: float, reference: str):
    """The proven/possible sweep, cached on the settings that determine it.

    The most expensive computation on the page -- it re-splits every trial at
    every one of sixty depths and bootstraps each step -- and B6's slider does
    not change any of its inputs. Keyed on the file's bytes plus the scalars
    rather than on the TrialSet, because a dataclass holding a DataFrame is not
    hashable -- and ``_load`` is itself cached, so re-reading here is free.
    """
    ts_, _ = _load(name, data, mapping_items)
    ad_ = AreaDepth.from_trials(ts_.col("contact"), ts_.col("area"))
    return run_volume_sweep(
        ts_, ad_, pos, z_gap=gap, mefs=mefs,
        reference=ReferenceContour(reference), n_boot=400,
    )


def _badge(level: str) -> str:
    return {"pass": "✅", "warn": "⚠️", "fail": "⛔"}[level]


def _chart(fig, key: str, height: int = PANEL_HEIGHT):
    """Render a project figure with the two settings a row's alignment needs.

    ``height`` is pinned rather than left at Streamlit's default of ``"content"``:
    on ``"content"`` each chart is sized from its own contents, so panels in a
    row end up different heights and a shared depth range still does not put a
    given depth on the same pixel row.

    ``theme=None`` keeps ``wellvolpos.viz.theme`` authoritative. Streamlit's own
    plotly theme otherwise restyles fonts, title and template on top of ours,
    which is exactly the drift between the two backends that CLAUDE.md's
    "both driven from viz/theme.py" rule exists to prevent.

    ``height`` defaults to the shared panel height, which is what keeps a row of
    depth panels aligned. C1 overrides it: it is a *stacked composite*, not a panel
    in a row, and at 470 px both of its halves were squashed to the point where the
    braces collided with the axis.
    """
    return st.plotly_chart(fig, width="stretch", height=height, theme=None, key=key)


# ------------------------------------------------------------------- tabs
# Declared before anything writes into them, because the trial-data selector now
# lives in tab ① and everything downstream depends on the file it picks. The
# sidebar keeps only the well geometry and the conventions -- the things a reader
# changes repeatedly while looking at a figure.
st.title("WellVolPOS")
st.caption("Well POS and volume, from a stochastic prospect model")
tabs = st.tabs(
    [
        "① Input data, QC and Risk",
        "② Prospect",
        "③ Well location",
        "④ Location sweep",
        "⑤ Risk & report",
        "⑥ Theory & guide",
    ]
)

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
    # The sidebar's controls cannot exist yet — the depth sliders take their range
    # from the contact column — but a sidebar that *vanishes* reads as a crash, so
    # it says why instead of disappearing.
    st.sidebar.subheader("Well")
    st.sidebar.info(
        "Waiting for trial data.\n\nThe well and convention controls take their range from the "
        "contact column, so they appear once a file is loaded. Pick a demo or upload a file in "
        "**tab ①**."
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
                    "w_play": loaded.play_chance,
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

st.sidebar.subheader("Well")
# Defaults derived from the data, not hardcoded. They used to be 3500/3550 m -- the
# reference well of prospect A's workbook -- which put prospect B's entry at its
# deepest contact of 2400 m and then collapsed the exit slider's range to a single
# point. The parity suite still pins 3500/3550; those live in tests/conftest.py,
# where they belong, because they are a property of that workbook and not of the UI.
_succ = ts.col("contact")[ts.col("resource") > 0]
_default_entry = float(np.round(np.median(_succ) / 5.0) * 5.0) if _succ.size else zmin
_default_entry = float(np.clip(_default_entry, zmin, zmax))
entry = st.sidebar.slider("Reservoir entry depth (m TVDSS)", zmin, zmax,
                          _default_entry, 5.0, key="w_entry")
# ``max_value`` cannot equal ``min_value``, and an exit at or below the deepest
# contact is meaningful -- it says the well passes through the whole reservoir -- so
# the range is widened rather than clamped.
exit_ = st.sidebar.slider("Reservoir exit depth (m TVDSS)", entry, max(zmax, entry + 5.0),
                          min(entry + 50.0, max(zmax, entry + 5.0)), 5.0, key="w_exit")
# MEFS scales with the prospect: 14 MMboe is a sensible threshold against prospect
# A's 16 MMboe discovery mean and a rounding error against prospect B's 121.
_default_mefs = float(np.round(np.mean(ts.col("resource")[ts.col("resource") > 0]) * 0.85))
mefs = st.sidebar.number_input("MEFS (MMboe)", min_value=0.0, value=_default_mefs, step=0.5,
                               key="w_mefs",
                               help="Minimum economic field size — Rose's MCFS under another "
                                    "name. Drawn as a reference line and never applied to the "
                                    "distributions.")

st.sidebar.divider()
st.sidebar.subheader("Conventions")
st.sidebar.caption("Never implicit — every one of these changes the numbers.")
ref = st.sidebar.radio(
    "Reference contour for the location factor",
    [ReferenceContour.CREST, ReferenceContour.P90_AREA],
    format_func=lambda r: {"crest": "Crest / apex (Milkov 2021)", "p90_area": "P90 area (Rose)"}[r.value],
    key="w_ref",
)
# Default to the equal-cube-root scheme, which is the one the source workbook uses
# and the one Lars's risk summary is drawn against. Under "none" the location factor
# is reported separately instead of being attributed to elements — still available,
# but a poor default for a table whose third column is meant to show the penalty.
scheme = st.sidebar.selectbox(
    "Risk-element allocation", SHIPPED_SCHEMES, format_func=lambda k: SCHEME_LABELS[k],
    index=list(SHIPPED_SCHEMES).index("equal_cube_root"), key="w_scheme",
)
# GeoX plots its area-depth curve against area squared, so that convention is
# offered alongside ours. The transform is on the axis only — every number the
# tool computes stays in km² (non-negotiable 4).
area_scale = st.sidebar.selectbox(
    "Area–depth x-axis", list(AREA_SCALES), index=0,
    help="area is this tool's default; area² is GeoX's convention; √area straightens a cone.",
    key="w_area_scale",
)

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
    st.subheader("The trials themselves")
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

    st.divider()
    st.subheader("Quality control")
    for c in qc.checks:
        st.markdown(f"{_badge(c.level)} **{c.name}** — {c.message}")

    st.divider()
    st.subheader("Risking convention")
    f = qc.failure
    if f and f.has_failures:
        st.markdown(f"**{f.summary()}**")
        with st.expander("Evidence"):
            for e in f.evidence:
                st.markdown(f"- {e}")
        default = 0 if f.verdict == "chance_failure" else 1
        # Branch on a stable key, never on the label text: the label is user
        # copy and rewording it must not be able to change which POS the whole
        # app uses.
        conv = st.radio(
            "Is that right?",
            CONVENTION_KEYS,
            index=default,
            format_func=lambda k: CONVENTION_LABELS[k],
        )
        st.session_state["risking_convention"] = conv
        if conv == "geometric":
            st.info(
                "**To be implemented in a later update.** The principle is in Lowry, Suttill & "
                "Taylor (2005), *Advances in risking exploration prospects*, APPEA Journal 45(1) "
                "179–188, [doi:10.1071/AJ04012](https://doi.org/10.1071/AJ04012): the geological "
                "chance factors and the *geometric* chance that a given location is within the "
                "accumulation are separate things, and conflating them double-counts. Until it is "
                "built this option behaves as *success-case only*, and the caveat below says why "
                "that is not the same answer."
            )
            st.warning(
                "Not yet implemented. Reading the zeros as geometric means they are *charged* "
                "trials with no trapped column above the crest, so they belong in the "
                "denominator of r_location — which conditions on `resource > 0` and therefore "
                "currently drops them. Until that is built, this option behaves like "
                "'success-case only' and r_location is not what this reading requires."
            )
    else:
        st.markdown("No zero-volume trials — the export looks success-case only.")
        st.session_state["risking_convention"] = "success_case_only"

    if qc.blocked:
        st.error("A check failed. The analysis tabs stay closed until it is resolved.")

if qc.blocked:
    st.stop()

# The chance table is an **input**, so it lives in tab ① with the data and the
# risking convention (Lars, 2026-08-11). Charge, trap, reservoir and retention are
# judgements about the prospect: made before anyone picks a location, and unchanged
# by picking one. The location factor is *computed* from the trials and the well's
# depth, so the summary that multiplies the two can only be assembled afterwards --
# and that summary is in tab ⑤. Keeping the input and the summary apart is what
# stops the third column of that summary being read as something a person entered.
with tabs[0]:
    st.divider()
    st.subheader("Chance table — the geological risk elements")
    st.caption(
        "**Inputs, not results.** Per-element chance of success for the prospect, multiplied "
        "together to give POS_prospect. Nothing here depends on where the well goes: the "
        "location factor is computed from the trials, and the two are brought together in the "
        "risk summary in tab ⑤. If the risking convention above says the trials already carry "
        "the chance, this table drives the attribution figures only."
    )
    ec = st.columns(4)
    elements = {
        el: ec[i].number_input(
            el.capitalize(), 0.01, 1.0, CHANCE_DEFAULTS[el], 0.01, key=f"w_chance_{el}",
            help=CHANCE_HELP[el],
        )
        for i, el in enumerate(ELEMENTS)
    }
    # The play chance is a *fifth* input, one level above the four elements: the
    # chance the play works at all. Added because the risk summary shows a Play
    # column and a Play chance result line, and a column that can only ever read
    # 100 % is a column that teaches nothing. It multiplies POS_prospect, so it
    # multiplies P_well too.
    pc1, pc2 = st.columns([1, 3])
    play_chance = pc1.number_input(
        "Play chance", 0.01, 1.0, 1.00, 0.01, key="w_play",
        help="Chance the play works at all — that the petroleum system is present and "
             "functioning at this level. One level above the four elements below, which are "
             "conditional on it. Leave at 1.00 to assess the segment on its own.",
    )
    pc2.caption(
        f"**POS_prospect = play × charge × trap × reservoir × retention = "
        f"{play_chance:.2f} × {float(np.prod(list(elements.values()))):.4f} = "
        f"{play_chance * float(np.prod(list(elements.values()))):.4f}**\n\n"
        f"Everything here is conditional on the play: the four elements are read as "
        f"*probability of the prospect given the play*, which is what the middle column of "
        f"the risk summary in tab ⑤ is called."
    )

# The play chance multiplies the four elements: POS_prospect is the chance the
# play works *and* every element of this prospect does.
pos_from_table = float(play_chance) * float(np.prod(list(elements.values())))
risking_convention = st.session_state.get("risking_convention", "success_case_only")
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


# The correlation check warns rather than fails (decided 2026-08-10), because the
# assumption it tests belongs to the extension alone and blocking closed the
# reference engine with it. Warning in the QC list is not enough on its own: the
# reader who goes straight to tab ③ never sees that list, so the caveat is raised
# again wherever the split's own numbers are drawn.
_split_level, _split_message, _split_r = check_area_pay_correlation(ts)


def _split_caveat() -> None:
    if _split_level == "warn" and np.isfinite(_split_r) and abs(_split_r) >= 0.5:
        st.warning(f"**The proven/possible split is not defensible on this data.** {_split_message}")

with tabs[1]:
    st.subheader("Prospect — the un-cut model")
    res_all = ts.col("resource")
    s = group_summary(ts, groups)["prospect"]
    # P99 and P1 in the petroleum orientation: P99 is exceeded 99 % of the time,
    # so it is the *low* end. On this file P99 is 0.00 because 23.95 % of trials
    # are chance failures at exactly zero — which is the honest answer and worth
    # seeing next to the mean.
    p99 = float(np.percentile(res_all, 1.0))
    p1 = float(np.percentile(res_all, 99.0))
    c = st.columns(6)
    for col, label, value in zip(
        c, ("P99", "P90", "P50", "Mean", "P10", "P1"),
        (p99, s["p90"], s["p50"], s["mean"], s["p10"], p1),
    ):
        col.metric(label, f"{value:.2f}")
    st.caption(
        "MMboe, petroleum orientation — P99 is the low end, exceeded 99 % of the time. "
        "P99 and P90 are 0.00 here because 23.95 % of trials are chance failures at exactly zero: "
        "the low end of the *prospect* distribution is dominated by the cases with no hydrocarbons "
        "at all."
    )

    if not has_area:
        st.warning(
            "No productive-area column in this export — the map view, A1, A4 and A5 need it "
            "and are skipped."
        )
    else:
        st.divider()
        # One depth range for the row, so A1 and A4 can be read straight across
        # at constant depth (non-negotiable 2). The map view is in plan view and
        # A5 has no depth axis, so neither joins the alignment.
        succ_contact = ts.col("contact")[res_all > 0.0]
        zrow_prospect = row_zlim(
            (ad.shallowest, ad.deepest),
            (float(succ_contact.min()), float(succ_contact.max())),
            pad_frac=0.02,
        )
        # A4's two renderings of one dataset. It was briefly two figures — A4's blue
        # log-density and a separate A7 grid — which showed the same trials twice
        # under two numbers; merged on Lars's instruction (2026-08-11).
        _auto_r, _auto_z = suggest_grid(res_all[res_all > 0.0], succ_contact)
        ac1, ac2, ac3 = st.columns([2, 1, 1])
        a4_render = ac1.radio(
            "A4 rendering", ["grid", "hexbin"], horizontal=True, key="w_a4_render",
            format_func=lambda k: {"grid": "Trial-count grid (workbook, inferno)",
                                   "hexbin": "Log-density hexbin"}[k],
        )
        a4_nx = ac2.number_input("Resource bins", 10, 120, _auto_r, 2, key="w_grid_res",
                                 disabled=a4_render != "grid")
        a4_ny = ac3.number_input("Depth bins", 10, 120, _auto_z, 2, key="w_grid_z",
                                 disabled=a4_render != "grid")
        st.caption(
            f"Grid default **{_auto_r} × {_auto_z}** from the Freedman–Diaconis rule on each "
            f"axis — bin width 2·IQR/n^⅓, which adapts to spread *and* sample size and survives "
            f"the long right tail a resource distribution always has. The workbook's own "
            f"`resource grid` sheet is a fixed 100 × 100, which on 10 000 trials leaves most "
            f"cells empty. Counts are on a **log** scale either way: the modal cell holds two "
            f"orders of magnitude more trials than the tails, and the tails are where a location "
            f"question lives."
        )

        c1, c2 = st.columns(2)
        with c1:
            _chart(pfig_a1_area_depth(
                    ad, ts=ts, current_entry=entry, current_exit=exit_, zlim=zrow_prospect,
                ), key="a1")
        with c2:
            _chart(pfig_a4_resource_vs_depth(
                    ts, current_entry=entry, current_exit=exit_, mefs=mefs,
                    render=a4_render, n_resource=int(a4_nx), n_depth=int(a4_ny),
                    zlim=zrow_prospect, show_depth_labels=False,
                ), key="a4")
        st.caption(
            f"A1 and A4 share one depth range ({zrow_prospect[0]:.0f}–{zrow_prospect[1]:.0f} m TVDSS) "
            f"so the row reads straight across. Both draw the mean thick and the P90/P50/P10 family "
            f"thin and grey — the mean is the number that gets quoted, and on a skewed distribution "
            f"it is not the P50. A4 uses success trials only: the chance-failure zeros belong to "
            f"POS, not to the shape of the resource distribution."
        )

        st.divider()
        _chart(pfig_a5_exceedance(ts, groups, vc, mefs=mefs), key="a5")
        st.caption(
            "A5 — the exceedance curves at the current entry/exit, and the figure the whole tool "
            "builds towards: the well-associated (discovery-case) distribution against the "
            "prospect's own. No depth axis, so it sits below the row rather than in it."
        )

        # The numbers behind A5, in both readings (Lars, 2026-08-11). One row per
        # case and statistic, long-form rather than a wide grid, so that every cell
        # is labelled and nothing has to be inferred from a column header.
        st.markdown("**The numbers behind A5**")
        _p_updip = max(chance.pos_prospect - chance.p_well, 0.0)
        _a5_cases = [
            ("Prospect (all trials)", res_all[res_all > 0], chance.pos_prospect),
            ("Discovery case", vc.discovery_total[groups.discovery], chance.p_well),
            ("Proven at well", vc.proven[groups.discovery], chance.p_well),
            ("Attic | dry hole", vc.attic[groups.dry_with_attic], _p_updip),
        ]
        _rows = []
        for _name, _values, _ch in _a5_cases:
            _s = class_percentiles(_values, _ch)
            for _stat in ("P99", "P90", "P50", "Pmean", "P10", "P1"):
                if _stat == "Pmean":
                    _vol, _cond = _s["mean"], _s["mean_at"] / 100.0
                else:
                    _q = int(_stat[1:])
                    _vol, _cond = _s[f"p{_q}"], _q / 100.0
                _rows.append({
                    "case": _name,
                    "statistic": _stat,
                    "volume (MMboe)": _vol,
                    "probability — unrisked": _cond,
                    "probability — risked": _cond * _ch,
                })
        st.dataframe(
            pd.DataFrame(_rows), hide_index=True, width="stretch", height=330,
            column_config={
                "volume (MMboe)": st.column_config.NumberColumn(format="%.2f"),
                "probability — unrisked": st.column_config.NumberColumn(format="percent"),
                "probability — risked": st.column_config.NumberColumn(format="percent"),
            },
        )
        st.caption(
            "**Unrisked** is the conditional reading: the chance of exceeding that volume "
            "*given the case happens*, which is why P90 reads 90 % and P50 reads 50 % — that is "
            "the definition of a percentile, and it is the distribution the industry quotes. "
            "**Risked** is the unconditional one: the same volume, multiplied by the chance the "
            "case happens at all. Solid curves in A5 are the unrisked reading, dashed are the "
            "risked one.\n\n"
            "**Pmean is not a percentile.** It is the arithmetic mean, and its unrisked "
            "probability is wherever it happens to fall on the curve — above P50 on a "
            "right-skewed distribution. That is why it gets its own row rather than sitting "
            "between P50 and P10 as if it were one of them."
        )

        st.divider()
        _chart(pfig_a8_contact_distribution(ts, current_entry=entry), key="a8")
        st.caption(
            "**A8** — the contact distribution recovered from the trials, and `P(contact deeper "
            "than this depth)` over it. Read a depth off the y-axis and the line gives the "
            "fraction of success trials whose contact lies below it, which **is** `r_location` at "
            "that entry, crest-referenced. So A8 is A3's raw material shown as a distribution "
            "instead of as a chance curve, and the two agree at every depth by construction. "
            "This distribution is what the HCWC Builder produces and GeoX consumes; every "
            "location result in this tool ultimately rests on its shape."
        )

        st.divider()
        st.subheader("Conceptual map view")
        # The apex is derived from A(z), not offered as an input: this figure is
        # conceptual, and a second apex control here could disagree with the one
        # the column-height mapping uses in tab ⑤.
        map_apex = float(ad.apex_estimate())
        m1, m2 = st.columns([1, 3])
        with m1:
            map_interval = st.number_input(
                "Contour interval (m)", min_value=5.0, max_value=500.0, value=50.0, step=5.0,
                help="Contours land on multiples of this, so they read like a depth map.",
                key="w_map_interval",
            )
            map_azimuth = st.slider(
                "Well azimuth on the map (°)", 0, 359, 35,
                help=(
                    "Arbitrary. Only the well's radius carries meaning — it puts the well on the "
                    "contour of its own entry depth. A(z) records enclosed area per depth and "
                    "nothing about the closure's shape."
                ),
                key="w_map_azimuth",
            )
            st.metric("Apex (derived)", f"{map_apex:.0f} m")
            st.caption("From A(z)'s shallow tail, extrapolated to zero area.")
        with m2:
            _chart(pfig_map_view(
                    ad, apex=map_apex, z_entry=entry, z_exit=exit_,
                    interval=map_interval, well_azimuth_deg=float(map_azimuth),
                ), key="mapview")
        st.caption(
            f"Concentric contours whose *areas* come from A(z), apex at the centre, deepest "
            f"sampled contact ({ad.deepest:.0f} m) as the outer ring. The shaded area inside the "
            f"entry contour is what a dry hole would leave up-dip. **Every contour is dashed; the "
            f"one solid ring is the well's entry depth**, so line style says only 'is this the "
            f"well?'. Contours shallower than the shallowest sampled contact "
            f"({ad.shallowest:.0f} m) are drawn faint — the trials never reached the crest, so "
            f"their area is a taper to the apex, not a model output. "
            f"**The shape is a cartoon**: circles of the right area, in the wrong outline."
        )

with tabs[2]:
    st.subheader("Well location")
    _split_caveat()

    if has_area:
        cs = class_summary(vc, groups)
        gs = group_summary(ts, groups)
        # The two numbers a well proposal is actually made on, side by side with
        # the two the prospect is described by — because they are what this tool
        # exists to distinguish. A discovery-case mean is not comparable with a
        # prospect mean, and P_well is not comparable with POS.
        k = st.columns(4)
        k[0].metric("Proven mean — headline KPI", f"{cs['proven']['mean']:.2f}",
                    help="MMboe. What this well would establish between entry and exit.")
        k[1].metric("Well associated mean", f"{gs['discovery']['mean']:.2f}",
                    help="MMboe. The whole accumulation given a discovery — Rose's 'Downdip'.")
        k[2].metric("P well", f"{chance.p_well:.1%}",
                    help="POS prospect × r location. The chance THIS well finds hydrocarbons.")
        k[3].metric("Attic mean | dry & charged", f"{cs['attic_dry_hole']['mean']:.2f}",
                    help="MMboe left up-dip if the well is dry but the prospect is charged.")

        # Expected volumes: mean × chance. Additive across prospects, and the
        # only figures here that are — but they describe no outcome that can
        # occur, so they sit below the success-case means, never instead of them.
        e = st.columns(3)
        e[0].metric("Expected prospect volume",
                    f"{expected_volume(gs['prospect']['mean'], chance.pos_prospect):.2f}")
        e[1].metric("Expected well associated",
                    f"{expected_volume(gs['discovery']['mean'], chance.p_well):.2f}")
        e[2].metric("Expected proven",
                    f"{expected_volume(cs['proven']['mean'], chance.p_well):.2f}")
        st.caption(
            "MMboe. **Expected** volumes are mean × chance — the source workbook's "
            "\"'Risked' Pmean\" column. They are what a portfolio adds up, and they describe no "
            "outcome that can happen: this well either finds something near "
            f"{gs['discovery']['mean']:.1f} or it finds nothing. Quote them beside the chance and "
            "the size, never instead of them."
        )

    c = st.columns(4)
    c[0].metric("POS prospect", f"{chance.pos_prospect:.4f}")
    c[1].metric("r location", f"{chance.r_location:.4f}")
    c[2].metric("P well", f"{chance.p_well:.4f}")
    c[3].metric("Trials", f"{ts.n_trials:,}")
    # risked_shares, not shares(): shares() is what the trial file's own zero
    # count implies (POS_trials), which only equals the entered POS_prospect
    # when the risking convention is "Correct". Showing raw shares() next to
    # a P_well metric drawn from an entered chance table would silently print
    # two different POS figures on the same tab.
    sh = groups.risked_shares(chance.pos_prospect, chance.p_well)
    st.markdown(
        f"**Outcome tree** — chance failure {sh['chance_failure']:.1%} · "
        f"dry with attic {sh['dry_with_attic']:.1%} · "
        f"discovery with contact logged {sh['contact_seen']:.1%} · "
        f"discovery with HC to exit {sh['hc_to_exit']:.1%}"
    )

    if not has_area:
        st.warning(
            "No productive-area column in this export — the proven/possible split, A6 and the "
            "live section need it and are skipped."
        )
    else:
        st.divider()
        st.markdown("### Volume classes")
        st.caption(
            "Five ways of measuring the same prospect, each answering a different question. "
            "They **nest**: up-dip ⊂ tested ⊂ well associated ⊂ prospect."
        )
        class_defs = [
            ("Prospect resource potential",
             "The whole un-cut model, crest to spill. What the prospect holds if it works — "
             "**with no well in it**. This is the number that gets quoted in a portfolio, and it "
             "is not what any one borehole can find."),
            ("Well associated volume",
             "The accumulation given that **this well** finds hydrocarbons — the contact is deeper "
             "than the reservoir entry, so the well is in the column. Rose calls it *Downdip*. "
             "**This is the volume a well proposal is made on.**"),
            ("Resource tested by the well",
             "The part of that accumulation between the reservoir **entry** and the well's lowest "
             "known hydrocarbon, which is the shallower of the contact and the reservoir **exit**. "
             "What a discovery would have *proven*. The headline KPI."),
            ("Possible — below the reservoir exit",
             "The rest of the well-associated volume: what lies **below the depth at which the "
             "well left the reservoir**. The well was still in hydrocarbons when it exited, so it "
             "never established how far down they go. Well associated, but **not proven** — that "
             "distinction is the whole reason this class exists instead of being folded into the "
             "discovery case."),
            ("Up-dip / attic volume",
             "The accumulation in the trials where hydrocarbons are present but sit **entirely "
             "above** the well: the well is dry, the prospect is not. This is what a dry hole "
             "leaves behind, and the number quoted when somebody argues for a sidetrack."),
        ]
        with st.expander("What each class means — worth reading once", expanded=False):
            for name, text in class_defs:
                st.markdown(f"**{name}** — {text}")
            st.caption(
                "Colours are fixed by concept across every figure in the app; the key is in tab ⑥."
            )

        # The chance that each class *occurs at all*. The prospect's is
        # POS_prospect; the three well-associated classes share P_well; the up-dip
        # case is dry but charged, so POS_prospect - P_well. Taken from `chance`,
        # never from the trial file's own zero count -- see
        # core.classes.risked_exceedance for the three times that went wrong.
        res_all = ts.col("resource")
        p_updip = max(chance.pos_prospect - chance.p_well, 0.0)
        rows = [
            ("Prospect resource potential", res_all[res_all > 0], chance.pos_prospect),
            ("Well associated volume", vc.discovery_total[groups.discovery], chance.p_well),
            ("Resource tested by the well", vc.proven[groups.discovery], chance.p_well),
            ("Possible — below the reservoir exit", vc.possible[groups.discovery], chance.p_well),
            ("Up-dip / attic volume", vc.attic[groups.dry_with_attic], p_updip),
        ]
        stats = [(name, class_percentiles(values, ch)) for name, values, ch in rows]
        pcols = [f"P{q}" for q in REPORT_PERCENTILES]

        # One table, not two (Lars, 2026-08-11). The percentiles are the
        # **conditional (success case)** ones, which is where the industry defines
        # them -- "P90 is 90 % probability of exceeding the P90 estimated value"
        # (Milkov 2021) -- and the added chance column is what turns them into
        # unconditional probabilities: multiply. A second table of those products
        # said nothing the multiplication does not.
        #
        # The chance column is `n / N` from the trial counts, as asked, and is
        # labelled that way because it is the *file's* chance. Under the
        # "trials are risked" convention it equals the app's chance; under
        # "success-case only, chance table on top" it does not, and the note below
        # the table says so rather than letting the two be confused.
        st.markdown(
            "**Volumes (MMboe), conditional on the case occurring** — with the chance of being "
            "on each distribution at all"
        )
        st.dataframe(
            pd.DataFrame([
                {"class": name,
                 "n": int(s_["n"]),
                 "chance of the case (n / N)": chance_from_counts(int(s_["n"]), ts.n_trials),
                 **{f"P{q}": s_[f"p{q}"] for q in REPORT_PERCENTILES},
                 "Mean": s_["mean"]}
                for name, s_ in stats
            ])[["class", "n", "chance of the case (n / N)",
                "P99", "P90", "P50", "Mean", "P10", "P1"]],
            hide_index=True, width="stretch",
            column_config={
                "chance of the case (n / N)": st.column_config.NumberColumn(format="percent"),
                **{c: st.column_config.NumberColumn(format="%.2f")
                   for c in [f"P{q}" for q in REPORT_PERCENTILES] + ["Mean"]},
            },
        )
        _counts_chance = chance_from_counts(int(stats[1][1]["n"]), ts.n_trials)
        st.caption(
            "**These percentiles are conditional — they assume the case happens.** P90 is exceeded "
            "90 % of the time *given* the case, P50 half the time, and so on; that is what the "
            "industry means by \"the P50\", and Schneider et al. (2023) determine this success-case "
            "distribution **before** any chance is applied. To get the *unconditional* (risked) "
            "probability of any volume in the table, multiply its percentile by the chance column: "
            f"the well-associated P50 of {stats[1][1]['p50']:.1f} MMboe is exceeded "
            f"{_counts_chance * 0.5:.1%} of the time, not 50 %. **The volumes do not change between "
            "the two readings — only the probability attached to them.**\n\n"
            "**P99 is the low case** and P1 the high case. And the **mean is not a percentile**: on "
            f"these right-skewed distributions it sits at P{stats[1][1]['mean_at']:.0f} of the "
            "well-associated case rather than at P50, so \"mean\" and \"middle\" are not "
            "interchangeable words here."
        )
        if abs(_counts_chance - chance.p_well) > 5e-4:
            st.info(
                f"The chance column counts trials: {int(stats[1][1]['n']):,} of {ts.n_trials:,} "
                f"are well associated, so **{_counts_chance:.1%}** — the *file's* chance. The app is "
                f"using **P_well = {chance.p_well:.1%}**, because the risking convention in tab ① "
                f"says the chance comes from the chance table rather than from the trials. C1's "
                f"dashed curves and every figure use P_well; this column is the raw count."
            )
        st.divider()
        # Two figures, one above the other (Lars, 2026-08-11). They were one stacked
        # composite; split, each renders at its own natural height and either can be
        # exported and placed on its own.
        _chart(pfig_c1_section(
                ad, ts, z_entry=entry, z_exit=exit_, area_scale=area_scale,
            ), key="c1")
        _chart(pfig_c2_exceedance(
                ts, groups, vc, pos_prospect=chance.pos_prospect,
                p_well=chance.p_well, mefs=mefs,
            ), key="c2", height=C2_HEIGHT)
        st.caption(
            "**C1 — the concepts, in one picture.** Above: where each volume sits in the "
            "structure. Below: the same volumes as exceedance curves, **two per concept in one "
            "colour**.\n\n"
            "The **solid** curve is *conditional* — the success case, given that case happens. It "
            "starts at 100 % and it is where the percentiles live: that is what anyone means by "
            "\"the P50\". The **dashed** curve is *unconditional* (risked): the same volumes with "
            "the chance folded in, so it starts at the chance instead. Here the prospect's dashed "
            f"curve begins at {chance.pos_prospect:.0%} and the well-associated one at "
            f"{chance.p_well:.0%} — and **the vertical gap between those two starts is the location "
            "penalty**, the chance the prospect holds something this well would miss.\n\n"
            "**Markers on both curves**, at P90 / P50 / mean / P10, each labelled with its "
            "volume — values to the right of the conditional markers and to the left of the "
            "unconditional ones so the pair does not overwrite itself. The volume is *the same "
            "number* on both curves; only the height differs, which is the whole lesson. The "
            "braces below show the nesting — up-dip inside tested inside well associated inside "
            "prospect — and the axis carries no negative labels: that space is for the braces, "
            "not for probabilities."
        )

        st.divider()
        c1, c2 = st.columns(2)
        with c1:
            _chart(pfig_a6_overlap(vc, groups, mefs=mefs), key="a6")
        with c2:
            # A6 has no depth axis, so this row has only one depth-carrying
            # panel and nothing to align it against; the section keeps its own
            # full A(z) range.
            _chart(pfig_b0_section(ad, z_entry=entry, z_exit=exit_, title="Live section"), key="live")
        st.caption(
            "A6 — Schneider et al.'s 'surprising overlap' between what a dry hole leaves in the "
            "attic and what a discovery proves. Live section — the closure shape from A(z), "
            "colour-keyed to what the well now standing at entry/exit would prove."
        )

@st.fragment
def _location_sweep_tab():
    st.subheader("Location sweep")
    _split_caveat()
    # The sweeps carry the well's *own* entry-to-exit spacing, so a swept
    # location is the same well moved up or down the structure. Left at a
    # default gap, B1's proven curve would disagree with the headline KPI in
    # tab ③ for the very well the user has the sliders set to.
    gap = exit_ - entry
    with st.spinner("Sweeping well location…"):
        sweep = run_sweep(ts, pos, reference=ref, z_gap=gap)
    # Every panel in both rows below carries depth on y, so all of them get one
    # range: the sweep's own grid unioned with A(z)'s extent. This is the row
    # that non-negotiable 2 was written for — six panels, read straight across
    # at constant depth beside a log or a section.
    zrow_sweep = row_zlim(
        (float(sweep.z.min()), float(sweep.z.max())),
        (ad.shallowest, ad.deepest) if has_area else None,
        pad_frac=0.02,
    )
    c1, c2, c3 = st.columns(3)
    with c1:
        _chart(pfig_a2_outcome_tree(sweep, current_z=entry, zlim=zrow_sweep), key="a2")
    with c2:
        _chart(pfig_a3_chance_decomposition(
                sweep, pos_prospect=pos, pos_trials=pos_trials, current_z=entry,
                zlim=zrow_sweep, show_depth_labels=False,
            ), key="a3")
    with c3:
        _chart(pfig_b3_uncertainty_reduction(
                sweep, current_z=entry, zlim=zrow_sweep, show_depth_labels=False
            ), key="b3")
    st.caption(
        f"Haskett (2003) optimum: {sweep.reduction_optimum:.0f}% expected uncertainty reduction "
        f"at entry {sweep.z_optimum:.1f} m TVDSS. A2's exit is a hypothetical entry + "
        f"{sweep.z_gap:.0f} m, swept alongside entry — it does not affect r_location or P_well. "
        f"All panels share {zrow_sweep[0]:.0f}–{zrow_sweep[1]:.0f} m TVDSS."
    )

    if not has_area:
        st.warning("No productive-area column in this export — B0, B1 and B2 need it and are skipped.")
        return

    st.divider()
    with st.spinner("Sweeping the volume split…"):
        vsweep = _volume_sweep(source.name, source.data,
                               tuple(sorted(overrides.items())), pos, gap, mefs, ref.value)
    d1, d2, d3 = st.columns(3)
    with d1:
        _chart(pfig_b0_section(ad, z_entry=entry, z_exit=exit_, zlim=zrow_sweep), key="b0")
    with d2:
        _chart(pfig_b1_volume_split(
                vsweep, current_z=entry, zlim=zrow_sweep, show_depth_labels=False
            ), key="b1")
    with d3:
        _chart(pfig_b2_chance_vs_regret(
                vsweep, current_z=entry, zlim=zrow_sweep, show_depth_labels=False
            ), key="b2")
    # Both conditional groups, because they thin at opposite ends: the discovery
    # group fails down-dip, the dry-with-attic group up-dip where almost nothing
    # is dry. Reporting only the first left the missing top of B1's orange curve
    # unexplained.
    sup_disc = describe_support(vsweep.n_discovery, vsweep.z, name="discovery")
    sup_dry = describe_support(vsweep.n_dry, vsweep.z, name="dry-with-attic")
    st.caption(
        f"B1/B2 sweep entry with a fixed {vsweep.z_gap:.0f} m entry-to-exit spacing, on the same "
        f"depth range as the row above. B2's dotted rule marks where those two particular curves "
        f"meet — it is not a risked comparison, since P_well is unconditional and the regret "
        f"curve is conditional on a dry *and* charged outcome. {sup_disc.message()} "
        f"{sup_dry.message()}"
    )

    # ---- B7 and B8, both from the 2018 macro workbook (Lars, 2026-08-11)
    st.divider()
    st.subheader("The trade-off, and where it is commercial")
    tb1, tb2 = st.columns(2)
    with tb1:
        _chart(pfig_b7_frontier(vsweep, current_z=entry), key="b7")
    with tb2:
        _chart(pfig_b8_commercial_chance(vsweep, current_z=entry, zlim=zrow_sweep), key="b8")
    _chart(pfig_b9_chance_weighted(vsweep, current_z=entry, zlim=zrow_sweep), key="b9")
    st.caption(
        "**B9 — the targeting tool.** `P_well × mean volume`, swept: a falling curve times a "
        "rising one, so it peaks somewhere in between and that depth maximises the expectation. "
        "It is drawn for the proven volume and for the whole well-associated volume, which peak "
        "in different places — the gap between those two stars is the exit depth's doing.\n\n"
        "**An expected value describes no outcome that can happen.** The well finds something "
        "near the success-case mean or it finds nothing; it never finds the chance-weighted "
        "number. Use B9 to *rank* locations and B1 or B7 to say how big the prize is."
    )
    st.caption(
        "**B7** is the workbook's *Well POS vs. Well to be tested Mean Resource*, and it is the "
        "most direct statement of what this tool is about: moving the well down-dip **buys volume "
        "with chance**. Read it as an efficient frontier — up and to the right is better and "
        "unavailable — with the depth labels giving the rate of exchange in metres. Neither axis is "
        "a depth, so this figure is exempt from the depth rule.\n\n"
        "**B8** puts the workbook's two MEFS charts on one pair of axes, because the difference "
        "between them *is* the content. `Pmcfs(well)` **rises** down-dip — a deeper well finds a "
        "bigger accumulation — and is **conditional** on a discovery. `P_well` **falls** down-dip. "
        "Their product `Pc(well) = P_well × Pmcfs(well)` is **unconditional**: the chance of a "
        "commercial discovery, full stop. A rising curve times a falling one usually peaks in "
        "between, and that starred peak is where the well goes on commercial grounds — `Pc(well)` "
        "being the number Rose says to carry into an EMV."
    )

    _inverse_section(vsweep, ts)


def _current_case() -> Case:
    """The settings on screen, as a :class:`Case`.

    Reads the widget values out of ``session_state`` by key rather than closing
    over the local variables, because three of them (minimum column height, the
    two map controls) only exist when the export carries a productive-area
    column. A missing widget must give the field's default, not a NameError at
    the moment someone clicks *Export*.
    """
    ss = st.session_state
    return Case(
        entry=entry, exit=exit_, mefs=mefs,
        risking_convention=risking_convention,
        reference=ref.value, scheme=scheme,
        chance_table=dict(elements), play_chance=float(play_chance),
        area_scale=area_scale,
        map_interval=float(ss.get("w_map_interval", 50.0)),
        map_azimuth_deg=float(ss.get("w_map_azimuth", 35.0)),
        dataset=str(choice) if choice in DEMOS else source.name,
        n_trials=ts.n_trials, fingerprint=fingerprint(ts),
    )


@st.fragment
def _export_section():
    """Export, in its own fragment.

    The four formats are built **on request**, not on every rerun. A PDF of
    sixteen figures takes seconds to draw, and ``st.download_button`` wants its
    bytes up front -- so building eagerly would mean every slider drag redrawing
    a document nobody asked for. One button assembles the bundle once and the
    downloads then come out of it.

    Everything is derived from one :class:`~wellvolpos.report.export.Bundle`, so
    the workbook, the PDF and the figures cannot disagree with each other or with
    the screen.
    """
    st.subheader("Export")
    case = _current_case()
    st.caption(
        "Every artefact is stamped with the POS in force and where it came from, the reference "
        "contour and the allocation scheme. A caption can be cropped out of a screenshot; a "
        "cover page and a **Case** sheet cannot be cropped out of a file."
    )

    st.caption(
        "Saving and reloading the **case** — the settings on their own, with no results — lives "
        "beside the trial data in **tab ①**, so the two things you can load into a session are "
        "in one place."
    )

    st.divider()
    if st.button("Build the report", key="build_export", type="primary"):
        with st.spinner("Drawing every figure and assembling the workbook…"):
            bundle = export_mod.assemble(
                ts, case, pos=pos, pos_source=pos_source, qc=qc,
            )
            st.session_state["_export"] = {
                "stamp": bundle.stamp,
                "xlsx": export_mod.workbook_bytes(bundle),
                "pdf": export_mod.pdf_bytes(bundle),
                "png": export_mod.figures_zip(bundle, "png"),
                "svg": export_mod.figures_zip(bundle, "svg"),
                "tables": {k: v for k, v in export_mod.tables(bundle).items()},
                "warnings": bundle.warnings,
            }

    payload = st.session_state.get("_export")
    if payload is None:
        st.info("Press **Build the report** to assemble the workbook, the PDF and the figures.")
        return

    st.code(payload["stamp"], language=None)
    for w in payload["warnings"]:
        st.warning(w)
    d1, d2, d3, d4 = st.columns(4)
    d1.download_button(
        "⬇ Workbook (.xlsx)", payload["xlsx"], file_name="wellvolpos_report.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key="dl_xlsx",
    )
    d2.download_button(
        "⬇ Report (.pdf)", payload["pdf"], file_name="wellvolpos_report.pdf",
        mime="application/pdf", key="dl_pdf",
    )
    d3.download_button(
        "⬇ Figures (PNG .zip)", payload["png"], file_name="wellvolpos_figures_png.zip",
        mime="application/zip", key="dl_png",
    )
    d4.download_button(
        "⬇ Figures (SVG .zip)", payload["svg"], file_name="wellvolpos_figures_svg.zip",
        mime="application/zip", key="dl_svg",
    )
    st.caption(
        "The figures in every artefact are the matplotlib set, not screenshots of the "
        "interactive ones — both are driven from `viz/theme.py`, so they carry the same colours "
        "and the same depth rule. Each figure archive also contains the stamp and the case, "
        "because a figure dropped into a slide gets separated from its provenance immediately."
    )

    with st.expander("What is in the workbook"):
        for name, frame in payload["tables"].items():
            st.markdown(f"**{name}** — {len(frame):,} rows")
            st.dataframe(frame, width="stretch", hide_index=True)


@st.fragment
def _inverse_section(vsweep, ts):
    """B6, in its own fragment.

    The volume-to-prove slider must not re-run either sweep: at n=60 with a
    bootstrap that is the most expensive thing on the page, and CLAUDE.md's rule
    is that dragging a slider does not recompute everything. The sweeps are
    cached besides, so this is belt and braces.
    """
    st.divider()
    st.subheader("Inverse — given a volume to prove, where must the well go?")
    # The *supported* range: offering a target the tool would refuse to draw in
    # B1/B2 would be inviting a requirement it will not stand behind.
    targets, _, _ = volume_target_curve(vsweep, n=2)
    if targets.size == 0:
        st.warning("No proven-volume curve to invert on this sweep.")
        return
    lo_t, hi_t = float(targets[0]), float(targets[-1])
    default_t = float(np.clip(15.76, lo_t, hi_t))
    target = st.slider(
        "Volume to prove — mean proven (MMboe)",
        lo_t, hi_t, default_t, max((hi_t - lo_t) / 100.0, 0.01),
        help=(
            "The mean proven volume the well must establish. B6 returns the shallowest entry "
            "depth from which the proven mean stays at or above it all the way down — a "
            "guarantee rather than a first touch, because the sampled curve dips where the "
            "discovery group is small. The range covers only well-supported volumes."
        ),
    )
    inv = invert_volume_target(vsweep, target, ts=ts)
    st.markdown(f"**{inv.message()}**")
    if inv.achievable and inv.n_discovery_at is not None and inv.n_discovery_at < MIN_SUPPORT:
        st.warning(
            f"That depth rests on only {inv.n_discovery_at:,} discovery trials, below the "
            f"{MIN_SUPPORT}-trial floor — treat the requirement as indicative, not surveyed."
        )
    _chart(pfig_b6_inverse(vsweep, target=target, ts=ts), key="b6")
    st.caption(
        "The workbook's H38–H40 block as a curve. Marker colour is P_well at that depth — the "
        "cost side of the trade — because a second y-axis is not allowed and the trade is the "
        "point. The shaded band is the bootstrap interval on the proven mean, inverted through "
        "the same curve, so it widens down-dip where the discovery group thins. The level is "
        "nominal: a percentile bootstrap under-covers on small skewed samples.\n\n"
        "**The four thin grey lines are the workbook's BB–BE block**, added 2026-08-11: for each "
        "target volume, the P99 / P90 / P50 / P10 **contact depth among the trials that actually "
        "hold that much**. It answers a different question from the curve, and the two are easy "
        "to conflate. The curve gives one depth per target — how deep the well must go for the "
        "*mean proven volume* to reach it, a guarantee. These lines give the *spread* of contacts "
        "consistent with the volume, read straight off the trials. So the grey band is geological "
        "range and the shaded band is sampling error; neither should be read as the other.\n\n"
        "Rose's Figure 4 is why the spread is worth drawing rather than averaged: *“The EUR of "
        "9.4 MMBO is associated with productive areas from 200 to 1500 acres.”* The workbook's "
        "own `BA` column averages those contacts into a single number, and an average over that "
        "range is not a required depth."
    )


with tabs[3]:
    _location_sweep_tab()

with tabs[4]:
    st.caption(f"Effective POS prospect: **{pos:.4f}**, from {pos_source}.")
    if risking_convention == "trials_risked" and abs(pos_from_table - pos) > 1e-9:
        st.info(
            f"The table above multiplies to {pos_from_table:.4f}, but the trials imply "
            f"{pos:.4f} and the convention says the trials are authoritative. B4 therefore "
            f"carries a named reconciliation step; it is not a rounding error."
        )
    # ---- the risk summary: the entered table times the computed location factor
    st.subheader("Risk summary — the chance table, at this well")
    st.caption(
        "**Why this is here and the chance table is in tab ①.** Charge, trap, reservoir and "
        "retention are *inputs*: judgements about the prospect, made before anyone picks a "
        "location and unchanged by picking one. The location factor `r_location` is *computed*, "
        "from the trial file and this well's entry depth. Only once both exist can they be "
        "multiplied — so the summary comes last, and the third column below is a **result**, not "
        "something anyone typed."
    )
    _summary = risk_summary(elements, chance.r_location, scheme=scheme,
                            play_chance=play_chance)
    st.dataframe(
        pd.DataFrame(_summary.as_records()),
        hide_index=True, width="stretch",
        column_config={
            c: st.column_config.NumberColumn(format="percent") for c in SUMMARY_COLUMNS
        } | {
            "Carries the location penalty": st.column_config.CheckboxColumn(
                help="Reservoir is exempt under every shipped scheme — a well that misses the "
                     "column still saw the rock, so its reservoir risk is unchanged by where "
                     "it was drilled."
            )
        },
    )
    rc1, rc2 = st.columns([2, 3])
    with rc1:
        st.dataframe(
            pd.DataFrame(_summary.result_records()),
            hide_index=True, width="stretch",
            column_config={"value": st.column_config.NumberColumn(format="percent")},
        )
        st.metric("HC probability correction factor", f"{_summary.correction_factor:.3f}",
                  help="What the location costs each element that carries it. Under the "
                       "equal-cube-root scheme this is r^(1/3), because three of the four "
                       "elements share the penalty and reservoir is exempt — which is why it is "
                       "a cube root and not a fourth root.")
    with rc2:
        st.markdown(
            f"**Read it across.** Each element starts at its entered chance "
            f"(*{SUMMARY_COLUMNS[1]}*) and is reduced by the correction factor "
            f"**{_summary.correction_factor:.3f}** where it carries the location penalty. "
            f"Multiply the third column and you get **P_well = {_summary.well_pos:.4f}** — the same "
            f"number as `POS_prospect × r_location` = {chance.pos_prospect:.4f} × "
            f"{chance.r_location:.4f}, by construction rather than by coincidence.\n\n"
            f"**Allocation is a convention, not a fact.** All three shipped schemes give the same "
            f"P_well; only the split across elements differs, which is what B5 below shows. This "
            f"table uses **{SCHEME_LABELS.get(scheme, scheme)}**, set in the sidebar.\n\n"
            f"The *Play* column is 1.00 throughout: this tool assesses **one prospect segment** "
            f"from one trial file (decision 10) and models no play level above it."
        )
    for w in _summary.warnings:
        st.warning(w)
    st.divider()

    c1, c2 = st.columns(2)
    with c1:
        _chart(pfig_b4_chance_waterfall(elements, chance.r_location, pos, scheme=scheme), key="b4")
    with c2:
        _chart(pfig_b5_allocation_dumbbell(elements, chance.r_location, pos_prospect=pos), key="b5")
    st.caption(
        "B4 decomposes the POS in use through the location factor at the current entry, under "
        "the sidebar's allocation scheme; hatched steps are location, solid are geological "
        "chance, and the total is P_well by construction. B5 shows all three shipped schemes "
        "side by side — every scheme gives the same P_well (the dotted rule); only the "
        "attribution across elements differs, and reservoir is exempt under all of them."
    )


    st.divider()
    _export_section()

# The case download is written into the slot beside the case *loader* in tab ①, not
# here, so save and load sit together. It has to happen this late because a case is
# the state of every widget, and the last of them (the chance table) is only
# created above. Writing into a container declared earlier is exactly what
# Streamlit containers are for.
with _case_save_slot:
    st.download_button(
        "⬇ Save this case (.json)", _current_case().to_json(),
        file_name="wellvolpos_case.json", mime="application/json", key="dl_case",
    )
    st.caption(
        "Records the trial file it was saved against and fingerprints it, so reopening it on "
        "different trials says so rather than quietly answering a different question."
    )

with tabs[5]:
    render_guide(
        ts=ts, ad=ad if has_area else None, groups=groups,
        vc=vc if has_area else None, chance=chance, mefs=mefs,
        entry=entry, exit_=exit_, pos_source=pos_source,
    )

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
    f"Reference contour: {ref.value}. Allocation: {scheme}."
)
