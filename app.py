"""WellVolPOS — Streamlit entry point.

Phase 0–3 scope: all six tabs are live except export. Data and QC & Risking
gate everything else; the risking convention chosen in tab ② and the chance
table entered in tab ⑥ together determine POS_prospect (see the "Entered
here" comment below for why the chance-table widgets sit before the
computation that uses them). Reference contour and allocation scheme are
sidebar-level conventions, per CLAUDE.md's "never implicit" rule.

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
    allocate,
    apply_min_column_height,
    class_summary,
    compare_definitions,
    group_summary,
    group_trials,
    p_well,
    run_sweep,
    run_volume_sweep,
    split_trials,
    spread_at_fixed_column,
    volume_percentile_threshold,
)
from wellvolpos.io.adapters import read_trials
from wellvolpos.io.qc import run_qc
from wellvolpos.viz import (
    fig_a1_area_depth,
    fig_a2_outcome_tree,
    fig_a3_chance_decomposition,
    fig_a4_resource_vs_depth,
    fig_a5_exceedance,
    fig_a6_overlap,
    fig_b0_section,
    fig_b1_volume_split,
    fig_b2_chance_vs_regret,
    fig_b3_uncertainty_reduction,
    fig_b4_chance_waterfall,
    fig_b5_allocation_dumbbell,
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


# ------------------------------------------------------------------ loading
@st.cache_data(show_spinner=False)
def _load(path: str):
    ts = read_trials(path)
    return ts, run_qc(ts)


def _badge(level: str) -> str:
    return {"pass": "✅", "warn": "⚠️", "fail": "⛔"}[level]


# ------------------------------------------------------------------ sidebar
st.sidebar.title("WellVolPOS")
st.sidebar.caption("Well POS and volume, from a stochastic prospect model")

choice = st.sidebar.selectbox("Trial data", list(DEMOS) + ["Upload your own…"])
uploaded = None
if choice == "Upload your own…":
    uploaded = st.sidebar.file_uploader("GeoX trial export", type=["csv", "txt", "tsv", "xlsx"])

path = None
if uploaded is not None:
    tmp = Path(st.session_state.setdefault("_tmpdir", ".streamlit_uploads"))
    tmp.mkdir(exist_ok=True)
    path = tmp / uploaded.name
    path.write_bytes(uploaded.getbuffer())
elif choice in DEMOS:
    path = DEMOS[choice]

if path is None:
    st.info("Choose a demo dataset or upload a GeoX trial export to begin.")
    st.stop()

ts, qc = _load(str(path))

st.sidebar.divider()
st.sidebar.subheader("Well")
zmin, zmax = float(ts.col("contact").min()), float(ts.col("contact").max())
entry = st.sidebar.slider("Reservoir entry depth (m TVDSS)", zmin, zmax, min(max(3500.0, zmin), zmax), 5.0)
exit_ = st.sidebar.slider("Reservoir exit depth (m TVDSS)", entry, zmax, min(entry + 50.0, zmax), 5.0)
mefs = st.sidebar.number_input("MEFS (MMboe)", min_value=0.0, value=14.0, step=0.5)

st.sidebar.divider()
st.sidebar.subheader("Conventions")
st.sidebar.caption("Never implicit — every one of these changes the numbers.")
ref = st.sidebar.radio(
    "Reference contour for the location factor",
    [ReferenceContour.CREST, ReferenceContour.P90_AREA],
    format_func=lambda r: {"crest": "Crest / apex (Milkov 2021)", "p90_area": "P90 area (Rose)"}[r.value],
)
scheme = st.sidebar.selectbox(
    "Risk-element allocation", SHIPPED_SCHEMES, format_func=lambda k: SCHEME_LABELS[k]
)

# ------------------------------------------------------------------- tabs
tabs = st.tabs(
    ["① Data", "② QC & Risking", "③ Prospect", "④ Well location", "⑤ Location sweep", "⑥ Risk & report"]
)

with tabs[0]:
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
        use_container_width=True,
    )
    if ts.notes:
        for n in ts.notes:
            st.info(n)
    st.caption("Units are fixed: MMboe, m, km².")

with tabs[1]:
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

# Entered here (tab ⑥) even though it is used immediately below, because
# Streamlit executes every tab's body on every rerun regardless of which is
# visually active -- entering the same `with tabs[5]:` block twice just
# appends to that tab in order, so the input widgets can live before the
# quantities they feed and still render at the top of the tab that owns them.
with tabs[5]:
    st.subheader("Chance table")
    st.caption(
        "Per-element chance of success. Multiplied together they define the prospect's "
        "POS, unless the risking convention (tab ②) says the trials already carry it — "
        "in which case this table is for the attribution figures below only."
    )
    ec = st.columns(4)
    elements = {el: ec[i].number_input(el.capitalize(), 0.01, 1.0, 1.0, 0.01) for i, el in enumerate(ELEMENTS)}

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

with tabs[2]:
    st.subheader("Prospect — the un-cut model")
    s = group_summary(ts, groups)["prospect"]
    c = st.columns(4)
    c[0].metric("P90", f"{s['p90']:.2f}")
    c[1].metric("P50", f"{s['p50']:.2f}")
    c[2].metric("Mean", f"{s['mean']:.2f}")
    c[3].metric("P10", f"{s['p10']:.2f}")
    st.caption("MMboe.")

    if not has_area:
        st.warning("No productive-area column in this export — A1, A4 and A5 need it and are skipped.")
    else:
        st.divider()
        c1, c2, c3 = st.columns(3)
        with c1:
            fig_a1, _ = fig_a1_area_depth(ad, current_entry=entry, current_exit=exit_)
            st.pyplot(fig_a1, clear_figure=True)
        with c2:
            fig_a4, _ = fig_a4_resource_vs_depth(ts, current_entry=entry, mefs=mefs)
            st.pyplot(fig_a4, clear_figure=True)
        with c3:
            fig_a5, _ = fig_a5_exceedance(ts, groups, vc, mefs=mefs)
            st.pyplot(fig_a5, clear_figure=True)
        st.caption(
            "A1 — the area–depth curve recovered from the trials. A4 uses success trials only — the "
            "chance-failure zeros belong to POS, not to the shape of the resource distribution. "
            "A5 is evaluated at the current entry/exit."
        )

with tabs[3]:
    st.subheader("Well location")
    if has_area:
        cs = class_summary(vc, groups)
        st.metric("Proven mean — headline KPI", f"{cs['proven']['mean']:.2f} MMboe")
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
            use_container_width=True,
        )
        st.divider()
        c1, c2 = st.columns(2)
        with c1:
            fig_a6, _ = fig_a6_overlap(vc, groups, mefs=mefs)
            st.pyplot(fig_a6, clear_figure=True)
        with c2:
            fig_live, _ = fig_b0_section(ad, z_entry=entry, z_exit=exit_, title="Live section")
            st.pyplot(fig_live, clear_figure=True)
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
    # tab ④ for the very well the user has the sliders set to.
    gap = exit_ - entry
    with st.spinner("Sweeping well location…"):
        sweep = run_sweep(ts, pos, reference=ref, z_gap=gap)
    c1, c2, c3 = st.columns(3)
    with c1:
        fig_a2, _ = fig_a2_outcome_tree(sweep, current_z=entry)
        st.pyplot(fig_a2, clear_figure=True)
    with c2:
        fig_a3, _ = fig_a3_chance_decomposition(
            sweep, pos_prospect=pos, pos_trials=pos_trials, current_z=entry
        )
        st.pyplot(fig_a3, clear_figure=True)
    with c3:
        fig_b3, _ = fig_b3_uncertainty_reduction(sweep, current_z=entry)
        st.pyplot(fig_b3, clear_figure=True)
    st.caption(
        f"Haskett (2003) optimum: {sweep.reduction_optimum:.0f}% expected uncertainty reduction "
        f"at entry {sweep.z_optimum:.1f} m TVDSS. A2's exit is a hypothetical entry + "
        f"{sweep.z_gap:.0f} m, swept alongside entry — it does not affect r_location or P_well."
    )

    if not has_area:
        st.warning("No productive-area column in this export — B0, B1 and B2 need it and are skipped.")
        return

    st.divider()
    with st.spinner("Sweeping the volume split…"):
        vsweep = run_volume_sweep(ts, ad, pos, z_gap=gap, mefs=mefs, reference=ref)
    d1, d2, d3 = st.columns(3)
    with d1:
        fig_b0, _ = fig_b0_section(ad, z_entry=entry, z_exit=exit_)
        st.pyplot(fig_b0, clear_figure=True)
    with d2:
        fig_b1, _ = fig_b1_volume_split(vsweep, current_z=entry)
        st.pyplot(fig_b1, clear_figure=True)
    with d3:
        fig_b2, _ = fig_b2_chance_vs_regret(vsweep, current_z=entry)
        st.pyplot(fig_b2, clear_figure=True)
    st.caption(
        f"B1/B2 sweep entry with a fixed {vsweep.z_gap:.0f} m entry-to-exit spacing. "
        f"B6 (inverse: volume-to-prove → required entry) lands in phase 4."
    )


with tabs[4]:
    _location_sweep_tab()

with tabs[5]:
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
        fig_b4, _ = fig_b4_chance_waterfall(elements, chance.r_location, pos, scheme=scheme)
        st.pyplot(fig_b4, clear_figure=True)
    with c2:
        fig_b5, _ = fig_b5_allocation_dumbbell(elements, chance.r_location, pos_prospect=pos)
        st.pyplot(fig_b5, clear_figure=True)
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
        tc1, tc2 = st.columns(2)
        apex_default = float(ad.apex_estimate())
        apex = tc1.number_input(
            "Apex depth (m TVDSS)", value=apex_default, step=1.0,
            help=(
                "A mapped value is preferred. The default is a linear extrapolation of A(z)'s "
                "shallow tail to zero, offered only as a starting point — see "
                "AreaDepth.apex_estimate."
            ),
        )
        min_col = tc2.number_input("Minimum column height (m)", min_value=0.0, value=0.0, step=5.0)
        apex_is_default = abs(apex - apex_default) < 1e-9
        st.caption(
            f"Apex: **{'extrapolated from A(z)' if apex_is_default else 'entered'}** "
            f"({apex:.1f} m TVDSS)."
            + (
                "  Extrapolating the shallow tail has unbounded error where the trials do not "
                "reach the crest — prefer the mapped apex."
                if apex_is_default else ""
            )
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
    st.info("Export (XLSX / PNG / SVG / PDF / JSON) lands in phase 5.")

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
