"""WellVolPOS — Streamlit entry point.

Phase 0–2 scope: Data and QC & Risking gate everything else, and the Prospect,
Well location and Location sweep tabs are live (A1–A6, B0–B3 plus the live
section). Risk & report is still a stub, so the settings that must never be
implicit are visible from the first run even where their figures are pending.

Run with:  streamlit run app.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from wellvolpos.core import (
    AreaDepth,
    ReferenceContour,
    class_summary,
    group_summary,
    group_trials,
    p_well,
    run_sweep,
    run_volume_sweep,
    split_trials,
)
from wellvolpos.core.chance import SCHEME_LABELS
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
)

DATA = Path(__file__).parent / "data"
DEMOS = {
    "Prospect A — reduced (7 columns)": DATA / "demo_prospectA_reduced.csv",
    "Prospect A — full GeoX export (60 columns)": DATA / "demo_prospectA_full.csv",
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
    "Risk-element allocation", list(SCHEME_LABELS)[:3], format_func=lambda k: SCHEME_LABELS[k]
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
        conv = st.radio(
            "Is that right?",
            [
                "Correct — trials are risked; use the implied POS and lock the chance table to display-only",
                "No — trials are success-case only; apply my chance table on top",
                "The zeros are geometric (contact above crest), not chance failure; treat separately",
            ],
            index=default,
        )
        st.session_state["risking_convention"] = conv
    else:
        st.markdown("No zero-volume trials — the export looks success-case only.")
        st.session_state["risking_convention"] = "success_case_only"

    if qc.blocked:
        st.error("A check failed. The analysis tabs stay closed until it is resolved.")

if qc.blocked:
    st.stop()

pos = qc.failure.pos_trials if (qc.failure and qc.failure.pos_trials) else 1.0
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
            st.pyplot(fig_a1)
        with c2:
            fig_a4, _ = fig_a4_resource_vs_depth(ts, current_entry=entry, mefs=mefs)
            st.pyplot(fig_a4)
        with c3:
            fig_a5, _ = fig_a5_exceedance(ts, groups, vc, mefs=mefs)
            st.pyplot(fig_a5)
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
    sh = groups.shares()
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
            st.pyplot(fig_a6)
        with c2:
            fig_live, _ = fig_b0_section(ad, z_entry=entry, z_exit=exit_, title="Live section")
            st.pyplot(fig_live)
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
        st.pyplot(fig_a2)
    with c2:
        fig_a3, _ = fig_a3_chance_decomposition(sweep, pos_trials=pos, current_z=entry)
        st.pyplot(fig_a3)
    with c3:
        fig_b3, _ = fig_b3_uncertainty_reduction(sweep, current_z=entry)
        st.pyplot(fig_b3)
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
        st.pyplot(fig_b0)
    with d2:
        fig_b1, _ = fig_b1_volume_split(vsweep, current_z=entry)
        st.pyplot(fig_b1)
    with d3:
        fig_b2, _ = fig_b2_chance_vs_regret(vsweep, current_z=entry)
        st.pyplot(fig_b2)
    st.caption(
        f"B1/B2 sweep entry with a fixed {vsweep.z_gap:.0f} m entry-to-exit spacing. "
        f"B6 (inverse: volume-to-prove → required entry) lands in phase 4."
    )


with tabs[4]:
    _location_sweep_tab()

with tabs[5]:
    st.info("Phase 3 — chance table, allocation schemes, B4/B5, and export.")

st.divider()
st.caption(
    f"Single HC-water contact only — a prospect with both a gas–oil and an oil–water contact, "
    f"where a well may test one and not the other, is not represented. "
    f"Vertical (depth-dependent) risk is assumed already contained in the contact distribution; "
    f"building that distribution is the HCWC Builder's job. "
    f"Reference contour: {ref.value}. Allocation: {scheme}."
)
