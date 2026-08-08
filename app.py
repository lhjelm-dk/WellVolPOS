"""WellVolPOS — Streamlit entry point.

Phase 0/1 scope: the Data and QC & Risking tabs are live and gate everything
else. The analysis tabs are present but stubbed, so the navigation and the
settings that must never be implicit are visible from the first run.

Run with:  streamlit run app.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from wellvolpos.core import ReferenceContour, group_summary, group_trials, p_well
from wellvolpos.core.chance import SCHEME_LABELS
from wellvolpos.io.adapters import read_trials
from wellvolpos.io.qc import run_qc

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

with tabs[2]:
    st.subheader("Prospect — the un-cut model")
    s = group_summary(ts, groups)["prospect"]
    c = st.columns(4)
    c[0].metric("P90", f"{s['p90']:.2f}")
    c[1].metric("P50", f"{s['p50']:.2f}")
    c[2].metric("Mean", f"{s['mean']:.2f}")
    c[3].metric("P10", f"{s['p10']:.2f}")
    st.caption("MMboe. Figures A1, A4 land here in phase 1.")

with tabs[3]:
    st.subheader("Well location")
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
    st.caption("Proven mean becomes the headline KPI in phase 2. Figures A5, A6 and the live section land there.")

with tabs[4]:
    st.info("Phase 4 — location sweep: B0–B3, B6.")

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
