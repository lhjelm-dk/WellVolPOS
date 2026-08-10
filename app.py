"""WellVolPOS — Streamlit entry point.

Phases 0–4. Five tabs; everything is live except export. Tab ① carries the
trial-data selector, the import summary, a preview of the trials themselves, the
QC report and the risking question, and it gates the rest; the risking
convention chosen there and the chance table entered in tab ⑤ together
determine POS_prospect (see the "Entered
here" comment below for why the chance-table widgets sit before the
computation that uses them). Reference contour and allocation scheme are
sidebar-level conventions, per CLAUDE.md's "never implicit" rule.

Figures are the interactive (plotly) ones from ``wellvolpos.viz.interactive``;
the matplotlib set in ``wellvolpos.viz.figures`` is the export path and both
are styled from ``wellvolpos.viz.theme``. Each figure is a standalone plot
rather than a panel of a merged grid, so a row is made readable across by
handing every figure in it the same ``row_zlim`` and letting them share the
one panel height — that is what puts the depths level.

Run with:  streamlit run app.py
"""

from __future__ import annotations

from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from wellvolpos.core import (
    ELEMENTS,
    MIN_SUPPORT,
    SCHEME_LABELS,
    SHIPPED_SCHEMES,
    AreaDepth,
    ReferenceContour,
    allocate,
    apply_min_column_height,
    class_summary,
    compare_definitions,
    expected_volume,
    describe_support,
    group_summary,
    group_trials,
    invert_volume_target,
    p_well,
    run_sweep,
    run_volume_sweep,
    split_trials,
    spread_at_fixed_column,
    volume_percentile_threshold,
    volume_target_curve,
)
from wellvolpos.io.adapters import read_trials
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
    pfig_a6_overlap,
    pfig_b0_section,
    pfig_b1_volume_split,
    pfig_b2_chance_vs_regret,
    pfig_b3_uncertainty_reduction,
    pfig_b4_chance_waterfall,
    pfig_b5_allocation_dumbbell,
    pfig_b6_inverse,
    pfig_concepts,
    pfig_map_view,
    row_zlim,
)

DATA = Path(__file__).parent / "data"
DEMOS = {
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


# -------------------------------------------------------------- dark mode
def _dark() -> bool:
    """True when Streamlit is rendering the page dark.

    Read from the running context rather than offered as a toggle. A toggle can
    disagree with the page it sits on, and a dark figure on a light page is worse
    than no dark mode at all. Change it the usual way — the ☰ menu, *Settings*,
    *Theme* — and the figures follow.

    Dark is a *selected palette* in ``viz/theme.py``, not an inversion of the
    light one (design plan §7.2): the hues are the same volume-concept hues, with
    the lightnesses re-tuned until every pair that can share a figure still
    survives simulated colour-vision deficiency.
    """
    try:
        return str(getattr(st.context.theme, "type", "light")).lower() == "dark"
    except Exception:                                  # no context, e.g. under pytest
        return False


DARK = _dark()
if DARK:
    # Bound once, here, instead of adding ``dark=DARK`` to eighteen call sites
    # where the nineteenth would eventually be forgotten — and a figure drawn in
    # the wrong palette is a figure whose colours no longer mean what the key
    # says they mean (non-negotiable 3).
    _ns = globals()
    for _name in [n for n in list(_ns) if n.startswith("pfig_")]:
        _ns[_name] = partial(_ns[_name], dark=True)


# ------------------------------------------------------------------ loading
@st.cache_data(show_spinner=False)
def _load(path: str):
    ts = read_trials(path)
    return ts, run_qc(ts)


@st.cache_data(show_spinner=False)
def _volume_sweep(path: str, pos: float, gap: float, mefs: float, reference: str):
    """The proven/possible sweep, cached on the settings that determine it.

    The most expensive computation on the page -- it re-splits every trial at
    every one of sixty depths and bootstraps each step -- and B6's slider does
    not change any of its inputs. Keyed on ``path`` plus the scalars rather than
    on the TrialSet, because a dataclass holding a DataFrame is not hashable and
    the file path already identifies the trials.
    """
    ts_, _ = _load(path)
    ad_ = AreaDepth.from_trials(ts_.col("contact"), ts_.col("area"))
    return run_volume_sweep(
        ts_, ad_, pos, z_gap=gap, mefs=mefs,
        reference=ReferenceContour(reference), n_boot=400,
    )


def _badge(level: str) -> str:
    return {"pass": "✅", "warn": "⚠️", "fail": "⛔"}[level]


def _chart(fig, key: str):
    """Render a project figure with the two settings a row's alignment needs.

    ``height`` is pinned rather than left at Streamlit's default of ``"content"``:
    on ``"content"`` each chart is sized from its own contents, so panels in a
    row end up different heights and a shared depth range still does not put a
    given depth on the same pixel row.

    ``theme=None`` keeps ``wellvolpos.viz.theme`` authoritative. Streamlit's own
    plotly theme otherwise restyles fonts, title and template on top of ours,
    which is exactly the drift between the two backends that CLAUDE.md's
    "both driven from viz/theme.py" rule exists to prevent.
    """
    return st.plotly_chart(fig, width="stretch", height=PANEL_HEIGHT, theme=None, key=key)


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
        uploaded = st.file_uploader("GeoX trial export", type=["csv", "txt", "tsv", "xlsx"])

path = None
if uploaded is not None:
    tmp = Path(st.session_state.setdefault("_tmpdir", ".streamlit_uploads"))
    tmp.mkdir(exist_ok=True)
    path = tmp / uploaded.name
    path.write_bytes(uploaded.getbuffer())
elif choice in DEMOS:
    path = DEMOS[choice]

if path is None:
    with tabs[0]:
        st.info("Choose a demo dataset or upload a GeoX trial export to begin.")
    st.stop()

ts, qc = _load(str(path))

# ------------------------------------------------------------------ sidebar
zmin, zmax = float(ts.col("contact").min()), float(ts.col("contact").max())

# Case load runs *before* the widgets it restores, because a Streamlit widget
# reads session_state only when it is first created. Writing the values then
# rerunning is the only order in which a loaded case actually reaches the
# controls; setting them afterwards silently does nothing.
with st.sidebar.expander("Case", expanded=False):
    st.caption(
        "A case is the settings, never the results — every number is recomputed on load, "
        "so a reopened case cannot show you figures this build would not produce."
    )
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
                "w_min_col": loaded.min_column_height,
                "w_map_interval": loaded.map_interval,
                "w_map_azimuth": int(round(loaded.map_azimuth_deg)),
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

st.sidebar.subheader("Well")
entry = st.sidebar.slider("Reservoir entry depth (m TVDSS)", zmin, zmax,
                          min(max(3500.0, zmin), zmax), 5.0, key="w_entry")
exit_ = st.sidebar.slider("Reservoir exit depth (m TVDSS)", entry, zmax,
                          min(entry + 50.0, zmax), 5.0, key="w_exit")
mefs = st.sidebar.number_input("MEFS (MMboe)", min_value=0.0, value=14.0, step=0.5, key="w_mefs")

st.sidebar.divider()
st.sidebar.subheader("Conventions")
st.sidebar.caption("Never implicit — every one of these changes the numbers.")
ref = st.sidebar.radio(
    "Reference contour for the location factor",
    [ReferenceContour.CREST, ReferenceContour.P90_AREA],
    format_func=lambda r: {"crest": "Crest / apex (Milkov 2021)", "p90_area": "P90 area (Rose)"}[r.value],
    key="w_ref",
)
scheme = st.sidebar.selectbox(
    "Risk-element allocation", SHIPPED_SCHEMES, format_func=lambda k: SCHEME_LABELS[k],
    key="w_scheme",
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

# Entered here (tab ⑤) even though it is used immediately below, because
# Streamlit executes every tab's body on every rerun regardless of which is
# visually active -- entering the same `with tabs[4]:` block twice just
# appends to that tab in order, so the input widgets can live before the
# quantities they feed and still render at the top of the tab that owns them.
with tabs[4]:
    st.subheader("Chance table")
    st.caption(
        "Per-element chance of success. Multiplied together they define the prospect's "
        "POS, unless the risking convention (tab ①) says the trials already carry it — "
        "in which case this table is for the attribution figures below only."
    )
    ec = st.columns(4)
    elements = {
        el: ec[i].number_input(el.capitalize(), 0.01, 1.0, 1.0, 0.01, key=f"w_chance_{el}")
        for i, el in enumerate(ELEMENTS)
    }

pos_from_table = float(np.prod(list(elements.values())))
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
        c1, c2 = st.columns(2)
        with c1:
            _chart(pfig_a1_area_depth(
                    ad, ts=ts, current_entry=entry, current_exit=exit_, zlim=zrow_prospect,
                ), key="a1")
        with c2:
            _chart(pfig_a4_resource_vs_depth(
                    ts, current_entry=entry, current_exit=exit_, mefs=mefs,
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
            f"entry contour is what a dry hole would leave up-dip. Contours shallower than the "
            f"shallowest sampled contact ({ad.shallowest:.0f} m) are dotted — the trials never "
            f"reached the crest, so their area is a taper to the apex, not a model output. "
            f"**The shape is a cartoon**: circles of the right area, in the wrong outline."
        )

with tabs[2]:
    st.subheader("Well location")

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
        st.markdown("**Volume classes** (MMboe) — at the current entry/exit")
        st.dataframe(
            pd.DataFrame(
                [
                    {"class": "Discovery", **cs["discovery"]},
                    {"class": "Proven at well", **cs["proven"]},
                    {"class": "Possible — below reservoir exit", **cs["possible"]},
                    {"class": "Attic | dry hole", **cs["attic_dry_hole"]},
                ]
            )[["class", "n", "p90", "p50", "mean", "p10"]],
            hide_index=True,
            width="stretch",
        )
        st.divider()
        _chart(pfig_concepts(
                ad, ts, groups, vc, z_entry=entry, z_exit=exit_,
                pos_prospect=chance.pos_prospect, p_well=chance.p_well, mefs=mefs,
                area_scale=area_scale,
            ), key="concepts")
        st.caption(
            "**The concepts, in one picture.** Left: where each volume sits in the structure. "
            "Right: the same four volumes as *risked* exceedance curves — zeros included for the "
            "outcomes that do not happen, so each curve starts at its own chance rather than at "
            "100 %. That is why the prospect curve begins at "
            f"{chance.pos_prospect:.0%} and the well-associated curve at {chance.p_well:.0%}: the "
            "vertical gap between those two starts **is** the location penalty. The braces show "
            "the nesting — up-dip inside tested inside well associated inside prospect."
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
        vsweep = _volume_sweep(str(path), pos, gap, mefs, ref.value)
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
        min_column_height=float(ss.get("w_min_col", 0.0)),
        chance_table=dict(elements),
        area_scale=area_scale, dark=DARK,
        map_interval=float(ss.get("w_map_interval", 50.0)),
        map_azimuth_deg=float(ss.get("w_map_azimuth", 35.0)),
        dataset=str(choice), n_trials=ts.n_trials, fingerprint=fingerprint(ts),
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

    # The case JSON needs no computation, so it is always available -- and it is
    # the artefact that makes a session reproducible, which makes it the one
    # worth never putting behind a button.
    st.download_button(
        "⬇ Case settings (.json)", case.to_json(),
        file_name="wellvolpos_case.json", mime="application/json", key="dl_case",
    )
    st.caption(
        "Settings only, never results: reopening a case recomputes every number, so it cannot "
        "show you figures this build would not produce. It records which trial file it was saved "
        "against and says so if reopened on different trials."
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
        "nominal: a percentile bootstrap under-covers on small skewed samples."
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
    # allocate()'s floor warnings are the design plan's §5.1 guard. Raised once
    # here rather than inside each figure, because a warning drawn on a chart is
    # a warning that can be cropped out of a screenshot.
    _, alloc_warnings = allocate(elements, chance.r_location, scheme)
    for w in alloc_warnings:
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
    st.subheader("Minimum column height")
    st.caption(
        "**A mapping, not a filter.** This states what a minimum column height means in "
        "contact depth, area and volume terms; it does **not** exclude trials from any "
        "figure or KPI. Filtering would first need a decision on whether a sub-minimum "
        "trial becomes a chance failure (lowering POS) or simply leaves the population "
        "(renormalising it) — two different answers, so it is not assumed here."
    )
    if not has_area:
        st.warning(
            "No productive-area column in this export — the minimum-column-height mapping "
            "needs it and is skipped."
        )
    else:
        # The apex is always derived from the trials, never entered — Lars's
        # instruction, and it overrides design plan §5.2 / decision 6, which
        # wanted a mapped apex here. One apex per session, from one source, so
        # the map view and this mapping cannot disagree.
        #
        # The honest consequence: it is an *extrapolation* of A(z)'s shallow tail
        # to zero area, because the trials do not contain the apex. The
        # `crest` column in a full GeoX export looks like it should supply it and
        # does not — on the reference file 60 % of success trials have their
        # "crest" deeper than their own contact, which is impossible, so that
        # column is not per-row trustworthy. Minimum column height is measured
        # from this apex and inherits its error.
        tc1, tc2 = st.columns(2)
        apex = float(ad.apex_estimate())
        tc1.metric("Apex (derived from A(z))", f"{apex:.1f} m TVDSS")
        min_col = tc2.number_input("Minimum column height (m)", min_value=0.0, value=0.0, step=5.0,
                                   key="w_min_col")
        st.caption(
            f"Apex **derived from the trials**, not entered: A(z)'s shallow tail extrapolated to "
            f"zero area gives {apex:.1f} m TVDSS, against a shallowest sampled contact of "
            f"{ad.shallowest:.1f} m. The trials do not contain the crest, so this is an "
            f"extrapolation and the column heights below inherit its error."
        )

        tm = apply_min_column_height(ts, ad, apex, min_col)
        st.markdown(tm.message)
        # Shown whether or not the threshold binds: per the design plan the
        # mapping is the deliverable, and threshold.py's own docstring says
        # non-binding is the normal case on this data.
        m1, m2, m3 = st.columns(3)
        m1.metric("Min admissible contact", f"{tm.min_contact_depth:.1f} m TVDSS")
        if tm.min_area is None:
            m2.metric("Equivalent area", "—")
        elif tm.min_contact_depth < ad.shallowest:
            # AreaDepth.area_at is np.interp, which clips rather than
            # extrapolating, so above the shallowest sampled contact it keeps
            # returning that contact's area instead of tending to zero.
            m2.metric("Equivalent area", "above sampled range")
        else:
            m2.metric("Equivalent area", f"{tm.min_area:.3f} km²")
        if tm.binds and tm.equivalent_percentile is not None:
            m3.metric("Exceeded by", f"{tm.equivalent_percentile:.1%} of success trials")
        else:
            m3.metric("Trials excluded", f"{tm.n_excluded:,} ({tm.frac_excluded:.2%})")

        with st.expander("How far is a column-height cut from a volume cut, here?"):
            cmp = compare_definitions(ts, apex, min_col if min_col > 0 else 175.0)
            spread = spread_at_fixed_column(ts, apex, min_col if min_col > 0 else 175.0)
            if cmp.get("comparable"):
                st.markdown(
                    f"- Cutting by depth keeps {cmp['n_kept_by_depth']:,} success trials; the "
                    f"volume cut that keeps the same number sits at "
                    f"{cmp['volume_threshold']:.3f} MMboe.\n"
                    f"- They disagree on **{cmp['disagreement_frac']:.2%}** of that set — close, "
                    f"but not the same operation.\n"
                    + (
                        f"- At a fixed column height the resource still spans "
                        f"**{spread['ratio']:.1f}×** ({spread['n']:,} trials), because area is "
                        f"pinned while gross pay and yield are not."
                        if np.isfinite(spread.get("ratio", float("nan"))) else ""
                    )
                )
            else:
                st.markdown("Not comparable at this apex and column height.")
            st.caption(
                "P99.5 volume floor, the source workbook's alternative expression of the same "
                f"control: {volume_percentile_threshold(ts, 0.995):.3f} MMboe."
            )

    st.divider()
    _export_section()

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
    f"Risking: POS {pos:.4f} from the {pos_provenance}. "
    f"Reference contour: {ref.value}. Allocation: {scheme}."
)
