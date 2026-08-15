"""Tab ⑤ — attribution, the risk summary, and the export.

The summary can only be assembled once ``r_location`` exists, which needs a
well. Keeping it away from the chance table on tab ② is what stops its
third column being read as something a person typed."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from ..core import (
    SCHEME_LABELS,
    risk_summary,
)
from ..viz import (
    pfig_b4_chance_waterfall,
    pfig_b5_allocation_dumbbell,
)
from ..core import SUMMARY_COLUMNS
from ..report import export as export_mod
from ..report.case import Case, fingerprint
from .common import chart as _chart, split_caveat
from .context import Ctx
from .numbering import ref as fig_ref


def _current_case(ctx: Ctx) -> Case:
    """The settings on screen, as a :class:`Case`.

    Reads the widget values out of ``session_state`` by key rather than closing
    over the local variables, because three of them (minimum column height, the
    two map controls) only exist when the export carries a productive-area
    column. A missing widget must give the field's default, not a NameError at
    the moment someone clicks *Export*.
    """
    ss = st.session_state
    return Case(
        entry=ctx.entry, exit=ctx.exit_, mefs=ctx.mefs,
        risking_convention=ctx.risking_convention,
        reference=ctx.ref.value, scheme=ctx.scheme,
        chance_table=dict(ctx.elements), play_elements=dict(ctx.play_elements),
        area_scale=ctx.area_scale,
        map_interval=float(ss.get("w_map_interval", 50.0)),
        map_azimuth_deg=float(ss.get("w_map_azimuth", 35.0)),
        dataset=ctx.dataset,
        n_trials=ctx.ts.n_trials, fingerprint=fingerprint(ctx.ts),
    )


def render_case_save(ctx: Ctx, slot) -> None:
    """Draw the case download into a container tab ① declared earlier.

    Save and load sit together in tab ①, but the button can only be built *here*:
    a case is the state of every widget, and the last of them -- the chance table on
    tab ② -- is created after tab ① has already rendered. Writing into a container
    declared earlier is exactly what Streamlit containers are for.
    """
    with slot:
        st.download_button(
            "⬇ Save this case (.json)", _current_case(ctx).to_json(),
            file_name="wellvolpos_case.json", mime="application/json", key="dl_case",
        )
        st.caption(
            "Records the trial file it was saved against and fingerprints it, so reopening it "
            "on different trials says so rather than quietly answering a different question."
        )


@st.fragment
def _export_section(ctx: Ctx):
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
    case = _current_case(ctx)
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
                ctx.ts, case, pos=ctx.pos, pos_source=ctx.pos_source, qc=ctx.qc,
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


def render(ctx: Ctx) -> None:
    ts, ad, has_area = ctx.ts, ctx.ad, ctx.has_area
    groups, vc, chance = ctx.groups, ctx.vc, ctx.chance
    entry, exit_, mefs = ctx.entry, ctx.exit_, ctx.mefs
    ref, scheme, area_scale = ctx.ref, ctx.scheme, ctx.area_scale
    pos, pos_source, pos_from_table = ctx.pos, ctx.pos_source, ctx.pos_from_table
    pos_trials, risking_convention = ctx.pos_trials, ctx.risking_convention
    elements, play_elements, play_chance = ctx.elements, ctx.play_elements, ctx.play_chance
    qc, gap = ctx.qc, ctx.gap
    source, overrides = ctx.source, ctx.overrides

    def _split_caveat() -> None:
        split_caveat(ctx)

    st.caption(f"Effective POS prospect: **{pos:.4f}**, from {pos_source}.")
    if risking_convention == "trials_risked" and abs(pos_from_table - pos) > 1e-9:
        st.info(
            f"The table above multiplies to {pos_from_table:.4f}, but the trials imply "
            f"{pos:.4f} and the convention says the trials are authoritative. {fig_ref('{b4}')} "
            f"therefore "
            f"carries a named reconciliation step; it is not a rounding error."
        )
    # ---- the risk summary: the entered table times the computed location factor
    st.subheader("Risk summary — the chance table, at this well")
    st.caption(
        "**Why this is here and the chance table is in tab ②.** Charge, closure, reservoir and "
        "retention are *inputs*: judgements about the prospect, made before anyone picks a "
        "location and unchanged by picking one. The location factor `r_location` is *computed*, "
        "from the trial file and this well's entry depth. Only once both exist can they be "
        "multiplied — so the summary comes last, and the third column below is a **result**, not "
        "something anyone typed."
    )
    _summary = risk_summary(elements, chance.r_location, scheme=scheme,
                            play_elements=play_elements)
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
            f"P_well; only the split across elements differs, which is what {fig_ref('{b5}')} below "
            f"shows. This "
            f"table uses **{SCHEME_LABELS.get(scheme, scheme)}**, set in the sidebar.\n\n"
            f"The *Play* column is 1.00 throughout: this tool assesses **one prospect segment** "
            f"from one trial file (decision 10) and models no play level above it."
        )
    for w in _summary.warnings:
        st.warning(w)
    st.divider()

    # **Stacked, not side by side** (Lars, 2026-08-15). 5.1 is a waterfall of six or
    # seven steps and 5.2 a dumbbell across three schemes; at half width the step
    # labels on one and the scheme names on the other both wrap. They are also read
    # in sequence rather than compared -- 5.1 decomposes the chance under the scheme
    # in use, 5.2 asks what a different scheme would have done -- so the second is a
    # follow-on, not a companion.
    _chart(pfig_b4_chance_waterfall(elements, chance.r_location, pos, scheme=scheme), key="b4")
    _chart(pfig_b5_allocation_dumbbell(elements, chance.r_location, pos_prospect=pos), key="b5")
    st.caption(
        f"{fig_ref('{b4}')} decomposes the POS in use through the location factor at the current entry, under "
        "the sidebar's allocation scheme; hatched steps are location, solid are geological "
        f"chance, and the total is P_well by construction. {fig_ref('{b5}')} shows all three "
        f"shipped schemes "
        "side by side — every scheme gives the same P_well (the dotted rule); only the "
        "attribution across elements differs, and reservoir is exempt under all of them."
    )


    st.divider()
    _export_section(ctx)
