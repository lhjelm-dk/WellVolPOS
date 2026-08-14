"""Tab ③ — where should the well go? Every figure that sweeps depth.

Comes *before* tab ④ deliberately: this is the comparative material a choice is
made from, and tab ④ is the answer for one depth. They were the other way round
until 2026-08-11, which put the answer in front of the material that informs it.

Behind ``st.fragment`` because the volume sweep re-splits every trial at every
depth and bootstraps each step -- CLAUDE.md's rule is that dragging a slider must
not recompute everything. The sweeps are cached besides, so it is belt and braces.

**Not built, and deliberately** (offered and declined, 2026-08-11): "use this
depth" buttons that would push B9's expectation peak, B8's commercial peak or B6's
required depth into the sidebar slider. The depth stays something a person types,
so this tab informs rather than decides.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from ..core import (
    chance_table,
    compare_wells,
    risked_table,
    volume_table,
    BAND_MODES,
    BAND_MODE_LABELS,
    DEFAULT_N_BANDS,
    MIN_SUPPORT,
    banded_percentiles,
    describe_support,
    TARGET_STATISTIC_LABELS,
    TARGET_STATISTICS,
    invert_volume_target,
    run_sweep,
    volume_target_curve,
)
from ..viz import (
    add_well_markers,
    add_well_points,
    PROBABILITY_SCALES,
    VOLUME_SCALES,
    TALL_PANEL_HEIGHT,
    level_row,
    pfig_a2_outcome_tree,
    pfig_a3_chance_decomposition,
    pfig_b1_volume_split,
    pfig_b13_below_exit,
    pfig_b2_chance_vs_regret,
    pfig_b3_uncertainty_reduction,
    pfig_b6_inverse,
    pfig_b7_frontier,
    pfig_b8_commercial_chance,
    pfig_b9_chance_weighted,
    pfig_b11_pos_sensitivity,
    pfig_b12_banded_percentiles,
    row_zlim,
)
from .common import chart as _chart, split_caveat
from .context import Ctx
from .wells import well_editor
from .numbering import ref as fig_ref
from .loading import volume_sweep as _volume_sweep


@st.fragment
def _inverse_section(vsweep, ts, mefs, wells=(), selected_well=None):
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
    # **Which proven-volume statistic the target refers to** (Lars, 2026-08-11, who
    # asked whether P50/P10/P90 could be used instead of the mean). They answer
    # materially different questions -- P90 asks the well to prove the target even in
    # a poor discovery and therefore demands a deeper location, P10 only asks that a
    # good one would -- so this is an explicit setting, never a default buried in
    # code (non-negotiable 5), and the figure titles and axis labels carry it.
    stat = st.radio(
        "Target refers to", list(TARGET_STATISTICS), horizontal=True, key="w_b6_stat",
        format_func=lambda k: TARGET_STATISTIC_LABELS[k],
        help=(
            "Which statistic of the proven-volume distribution the target is measured "
            "against. mean is additive across prospects, which is why portfolios use it, "
            "but on a skewed distribution it sits above the median. P90 is the low case "
            "and demands the deepest well; P10 is the high case and the shallowest."
        ),
    )
    targets, _, _ = volume_target_curve(vsweep, n=2, statistic=stat)
    if targets.size == 0:
        st.warning("No proven-volume curve to invert on this sweep.")
        return
    lo_t, hi_t = float(targets[0]), float(targets[-1])
    # Default to **MEFS**, because that is the volume a well actually has to prove
    # and it is already on screen in the sidebar. The default used to be a hardcoded
    # 15.76 — prospect A's proven mean — which on prospect B clipped straight to the
    # shallow end of the supported range, so the worked reading under B6 came out as
    # "43.2 % against 43.2 %" and taught nothing (Lars, 2026-08-11). Where MEFS falls
    # outside the supported range the midpoint is used instead of clipping to an
    # endpoint, for the same reason: an endpoint gives a degenerate answer.
    default_t = float(mefs) if lo_t <= float(mefs) <= hi_t else 0.5 * (lo_t + hi_t)
    target = st.slider(
        f"Volume to prove — {TARGET_STATISTIC_LABELS[stat]} (MMboe)",
        lo_t, hi_t, default_t, max((hi_t - lo_t) / 100.0, 0.01),
        help=(
            "The proven volume the well must establish, measured by the statistic chosen "
            f"above. {fig_ref('{b6}')} returns the shallowest entry depth from which that statistic stays "
            "at or above it all the way down — a guarantee rather than a first touch, "
            "because the sampled curve dips where the discovery group is small. The range "
            "covers only well-supported volumes."
        ),
    )
    inv = invert_volume_target(vsweep, target, ts=ts, statistic=stat)
    st.markdown(f"**{inv.message()}**")
    if inv.achievable and inv.n_discovery_at is not None and inv.n_discovery_at < MIN_SUPPORT:
        st.warning(
            f"That depth rests on only {inv.n_discovery_at:,} discovery trials, below the "
            f"{MIN_SUPPORT}-trial floor — treat the requirement as indicative, not surveyed."
        )
    # Two panels, one depth axis. The spread was briefly its own figure (B10);
    # merged back on 2026-08-11 so the requirement and the range around it are one
    # glance, with each panel keeping its own honest x-axis.
    # A quarter taller (Lars, 2026-08-12): the leader lines, the bootstrap band and a
    # five-deep contact family all share one pair of axes, and at row height they
    # crowd each other.
    _f_b6 = pfig_b6_inverse(vsweep, target=target, ts=ts, mefs=mefs,
                            statistic=stat, height=TALL_PANEL_HEIGHT)
    add_well_markers(_f_b6, wells, selected=selected_well)
    _chart(_f_b6, key="b6")
    # The worked sentence first, in the app's live numbers, because "how do I read
    # this" is the question B6 kept failing to answer (Lars, 2026-08-11).
    _worked = ""
    if inv.achievable:
        _crest_p = float(vsweep.p_well[0]) if vsweep.p_well.size else float("nan")
        _worked = (
            f"**Reading it:** start at your target on the bottom axis — **{target:.0f} MMboe** — "
            f"go up to the curve, then across to the depth. It says **enter at "
            f"{inv.z_required:.0f} m or deeper**, and that costs you: P_well there is "
            f"**{inv.p_well_at:.1%}**, against **{_crest_p:.1%}** at the shallowest location "
            f"the sweep covers.\n\n"
        )
    st.caption(
        _worked
        + "**The proven-volume relation, read backwards.** Marker colour is P_well at that depth — the "
        "cost side of the trade — because a second y-axis is not allowed and the trade is the "
        "point. The shaded band is the bootstrap interval on the proven mean, inverted through "
        "the same curve, so it widens down-dip where the discovery group thins.\n\n"
        "**Read the band as a width, not as a guarantee.** It is a percentile bootstrap, "
        "which under-covers on small, skewed samples — and the discovery group is both, at "
        "the deep end. So the true coverage is *below* the level it is drawn at, by an amount "
        "this app does not know. What it is good for is comparing one part of the curve with "
        "another: where the band is wide, the estimate is thin.\n\n"
        "**It answers a guarantee, not a first touch**, which is why the axis says *or deeper*. "
        "The depth returned is the shallowest one from which the proven mean stays at or above "
        "your target all the way down. A sampled proven-mean curve wobbles wherever the discovery "
        "group is thin, and inverting its first crossing gives depths that deeper locations "
        "contradict — no basis for a well proposal.\n\n"
        "**The grey family is the range around that requirement**: for each volume, the "
        "P99 / P90 / P50 / P10 / P1 **hydrocarbon–water contact among the trials that actually "
        "hold at least that much**. The shading spans **P90–P10 only** — the body of the "
        "distribution — with P99 and P1 left as thin lines outside it, because a fill reads as "
        "*equally likely everywhere inside it* and filling to the extremes claimed that of the "
        "whole range. It is wide even so, and that is the content — Rose's Figure 4, "
        "*“The EUR of 9.4 MMBO is associated with productive areas from 200 to 1500 acres.”* The "
        "conventional shortcut is to average those contacts into one number and call it the "
        "required depth; an average over a range that wide is not a requirement.\n\n"
        "**The two families do not measure the same thing, which is why both axis titles name "
        "both readings.** For the coloured curve, x is a *target mean proven volume over the "
        "discovery group* and y is a *required entry depth*. For the grey lines, x is the *total "
        "resource held by one trial* and y is a *sampled contact* — 33.9–277.7 against "
        "2.2–482.1 MMboe here. So **where they cross means nothing**: that is not one family "
        "passing through another, it is two questions sharing borrowed axes. Read each against "
        "its own definition and they are both useful; read one off the other and they are not."
    )


@st.fragment
def _location_sweep_tab(ctx: Ctx):
    ts, ad, has_area = ctx.ts, ctx.ad, ctx.has_area
    entry, exit_, mefs = ctx.entry, ctx.exit_, ctx.mefs
    ref, pos = ctx.ref, ctx.pos
    pos_trials, gap = ctx.pos_trials, ctx.gap
    source, overrides = ctx.source, ctx.overrides
    wells, selected_well = ctx.wells, ctx.selected_well

    def _split_caveat() -> None:
        split_caveat(ctx)

    # The well geometry lives here now, not in the sidebar (Lars, 2026-08-13): this
    # tab is where locations are compared, and a location decision compares two or
    # three of them rather than nudging one. The widgets own the keys; app.py reads
    # them at the top of the script, before any tab renders.
    st.subheader("Well options")
    st.caption(
        "**Well A is the default and always exists.** Add B, C and D to put candidate "
        "locations side by side — they appear as labelled rules on every swept figure "
        "below, and as rows in the comparison on tab ④. Exactly one is carried into "
        "tab ④'s own figures, which are about a single well; choose it there."
    )
    well_editor(wells, float(ts.col("contact").min()), float(ts.col("contact").max()))
    if len(wells) > 1:
        st.caption(
            "**" + " · ".join(w.describe() for w in wells) + "** — "
            f"Well **{selected_well}** is the one tab ④ is currently about."
        )

    st.divider()
    st.subheader("Location sweep")
    _split_caveat()
    # The sweeps carry the well's *own* entry-to-exit spacing, so a swept
    # location is the same well moved up or down the structure. Left at a
    # default gap, B1's proven curve would disagree with the headline KPI in
    # tab ③ for the very well the user has the sliders set to.
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
    # Built first, levelled, then drawn. The bottom margin is sized to each figure's
    # own legend now, so three panels with three different series counts would
    # otherwise get three different plot areas -- and a shared depth range does not
    # put a depth on the same pixel row unless the plot areas match too. level_row
    # takes the largest margin and height across the row; see theme.level_row.
    f_a2 = pfig_a2_outcome_tree(sweep, current_z=entry, zlim=zrow_sweep)
    f_a3 = pfig_a3_chance_decomposition(
        sweep, pos_prospect=pos, pos_trials=pos_trials, current_z=entry, zlim=zrow_sweep)
    f_b3 = pfig_b3_uncertainty_reduction(sweep, current_z=entry, zlim=zrow_sweep,
                                         height=TALL_PANEL_HEIGHT)
    # **The sensitivity fan** (Lars, 2026-08-12), from the workbook's charts 8 and 22.
    # It sits beside 3.2 because it is the same quantity with the chance table swept
    # instead of fixed: 3.2 answers "what is P_well here", this answers "how much of
    # that is the chance table rather than the geometry".
    f_b11 = pfig_b11_pos_sensitivity(sweep, pos_prospect=pos, current_z=entry,
                                     zlim=zrow_sweep)
    # b11 is NOT levelled with them: it is drawn full width below the row, not as a
    # fourth panel in it, so borrowing the row's height only padded it.
    # 3.3 is **not** in the row (Lars, 2026-08-12). It carries two curves and an
    # argument about which population they are over, and a third of the width was not
    # enough for either. It is drawn full width directly under the row instead, which
    # is also where the reader meets it after 3.2.
    # Every candidate as a labelled rule, on every figure whose y-axis is a depth.
    # Applied here rather than threaded through fourteen figure signatures and their
    # twins; 3.8 and 3.12 are skipped for the same reason they are exempt from the
    # depth rule, neither having depth on an axis.
    for _f in (f_a2, f_a3, f_b3, f_b11):
        add_well_markers(_f, wells, selected=selected_well)

    level_row(f_a2, f_a3)
    c1, c2 = st.columns(2)
    with c1:
        _chart(f_a2, key="a2", height=int(f_a2.layout.height))
    with c2:
        _chart(f_a3, key="a3", height=int(f_a3.layout.height))
    _chart(f_b3, key="b3")
    # A warning box rather than a line of caption (Lars, 2026-08-13). It is the one
    # thing about this figure that gets misread, and a caption below eleven other
    # figures is not where a caveat survives contact with a reader in a hurry.
    st.warning(
        "**3.3 measures information, not value.** The peak is the depth whose result "
        "you can least predict — it values learning *dry* exactly as much as learning "
        "*large*, and it contains no dry-hole cost, no development case and no "
        f"discount rate. It is **not** a recommended location. For that, read "
        f"{fig_ref('{b9}')} (chance-weighted volume) and {fig_ref('{b8}')} "
        f"(commercial chance)."
    )
    st.caption(
        "**3.3 — how much would a well here *tell* you?** The expected shrinkage of the "
        "prospect's P10–P90 range once you know which side of the well the contact fell on. "
        "The trials split in two — deeper than the entry, or not — which is Haskett's "
        "*discrete learning*: one bit, no partial outcome, and the whole population splits "
        "on it. The inter-percentile range stands in for variance because that is what "
        "Haskett himself recommends — a range is read straight off a distribution "
        "everyone already has in front of them, and it moves with variance, so nothing "
        "is lost by using the more legible of the two."
        "\n\n"
        "**It is his measure, not his setting.** Haskett (2003) is about **appraisal** — a "
        "discovery exists and you are siting well *two*. He is explicit: *“Appraisal "
        "activities must be distinguished from exploration and development activities… In "
        "order to appraise there first needs to be a successful exploration effort.”* Here "
        "there is no discovery yet, and one branch of the split is a dry hole. The machinery "
        "carries over; the conclusion does not."
        "\n\n"
        "**So read the peak as the most *informative* depth, not the best one.** The measure "
        "values learning *dry* exactly as much as learning *large* — it has no dry-hole cost, "
        f"no development case and no discount rate in it. For where the well is *worth* "
        f"drilling see {fig_ref('{b9}')} and {fig_ref('{b8}')}."
        "\n\n"
        "**The two curves are two populations, and the gap between them is the point.** The "
        "solid curve is over the **success cases**, which is the same conditioning "
        "`r_location` uses: a chance failure is a property of the prospect, not of where the "
        "well goes. The dotted grey curve is over **every trial**. Where a file carries "
        "chance failures the parent's P90 is zero, which inflates the range and makes the "
        "grey curve mostly report *“we learned it was not a chance failure”* — something a "
        "well at any depth tells you equally. On the reference data that moved the apparent "
        "optimum 92 m up-dip and nearly doubled the percentage. Where a file has no chance "
        "failures the two curves coincide exactly, which is worth seeing as well."
        "\n\n"
        "**The dashed curve is the same measure on a P20–P80 range** instead of P10–P90. "
        "Haskett's choice of range is a convention rather than a result, so this is how "
        "much of the answer rests on it — and on both demo prospects the optimum moves "
        "only a few metres, which is the reassuring answer. If the two ever peak in "
        "different places, the tails are doing the work and the recommendation is fragile."
    )
    _chart(f_b11, key="b11", height=int(f_b11.layout.height))
    st.caption(
        "**3.4 — how much of your answer is the chance table?** Every thin grey curve is "
        "`P_well` against depth for a different `POS_prospect`, a decile at a time; the heavy "
        "one is the POS actually in force. Deciles rather than every percentile: neighbours "
        "would differ by a hundredth and the fan would read as a smear.\n\n"
        "**They are all the same shape, scaled vertically**, and that is the content rather "
        "than a shortcoming of the drawing. `P_well = POS_prospect × r_location(z)`, and only "
        "the second factor moves with depth — so revising the chance table and moving the well "
        "are *independent* levers. A reader who has seen this fan cannot believe that drilling "
        "deeper fixes a poor chance table, which is exactly the confusion the whole "
        "`POS × r` separation exists to prevent."
    )
    st.caption(
        f"Haskett (2003) optimum: {sweep.reduction_optimum:.0f}% expected uncertainty reduction "
        f"at entry {sweep.z_optimum:.1f} m TVDSS. {fig_ref('{a2}')}'s exit is a hypothetical entry + "
        f"{sweep.z_gap:.0f} m, swept alongside entry — it does not affect r_location or P_well. "
        f"All panels share {zrow_sweep[0]:.0f}–{zrow_sweep[1]:.0f} m TVDSS."
    )

    if not has_area:
        st.warning(fig_ref("No productive-area column in this export — {b1} and {b2} "
                           "need it and are skipped."))
        return

    st.divider()
    with st.spinner("Sweeping the volume split…"):
        vsweep = _volume_sweep(source.name, source.data,
                               tuple(sorted(overrides.items())), pos, gap, mefs, ref.value)
    # **The schematic section is not drawn here** (Lars, 2026-08-14). 4.3 draws the
    # same section at the chosen well, from the same A(z), and two copies of one
    # figure on two tabs is one more place for them to disagree. The two curves that
    # remain are what this row was for: what the well proves, and what it risks.
    f_b1 = pfig_b1_volume_split(vsweep, current_z=entry, zlim=zrow_sweep)
    # The below-exit volume on its own axes (Lars, 2026-08-14): four volumes and four
    # percentile ladders on one figure was unreadable, and this one is conditional on a
    # different event from the other three anyway.
    f_b13 = pfig_b13_below_exit(vsweep, current_z=entry, zlim=zrow_sweep)
    f_b2 = pfig_b2_chance_vs_regret(vsweep, current_z=entry, zlim=zrow_sweep)
    for _f in (f_b1, f_b13, f_b2):
        add_well_markers(_f, wells, selected=selected_well)
    level_row(f_b1, f_b13, f_b2)
    d1, d2, d3 = st.columns(3)
    with d1:
        _chart(f_b1, key="b1", height=int(f_b1.layout.height))
    with d2:
        _chart(f_b13, key="b13", height=int(f_b13.layout.height))
    with d3:
        _chart(f_b2, key="b2", height=int(f_b2.layout.height))
    # Both conditional groups, because they thin at opposite ends: the discovery
    # group fails down-dip, the dry-with-attic group up-dip where almost nothing
    # is dry. Reporting only the first left the missing top of B1's orange curve
    # unexplained.
    # The two possible-below-exit readings, in the app's own live numbers. Lars asked
    # what the curve meant and the honest answer is that one curve could not say it:
    # the unconditional mean is diluted by the discoveries that leave nothing below
    # the exit, and the conditional one is the prize without the chance of getting it.
    import numpy as _np

    _i_here = int(_np.argmin(_np.abs(_np.asarray(vsweep.z, dtype=float) - float(entry))))
    _p_any = (float(vsweep.p_well_exits_in_hc[_i_here])
              if vsweep.p_well_exits_in_hc is not None else float("nan"))
    _m_all = float(vsweep.possible_mean[_i_here])
    _m_any = (float(vsweep.possible_mean_if_any[_i_here])
              if vsweep.possible_mean_if_any is not None else float("nan"))
    if _np.isfinite(_p_any) and _np.isfinite(_m_any):
        st.info(
            f"**The two possible-below-exit curves, at Well {selected_well}.** *If* the well "
            f"leaves the reservoir still in hydrocarbons, the untested volume below the exit "
            f"averages **{_m_any:,.2f} MMboe** — that is the dotted curve, and it is the "
            f"additional potential you are asking about. The chance of that happening at all "
            f"is **{_p_any:.1%}**, and the dashed curve is the product: "
            f"**{_p_any:.1%} × {_m_any:,.2f} = {_m_all:,.2f} MMboe** averaged over *every* "
            f"discovery, including the ones whose contact falls inside the penetrated interval "
            f"and leave nothing below it.\n\n"
            f"**Each is wrong on its own.** The dashed one is the *additive* member of the "
            f"volume classes — proven + possible = well associated, exactly — so it is what "
            f"makes the split a decomposition. The dotted one is the size of the upside and "
            f"cannot be added to proven; quoted alone it overstates the prize exactly the way "
            f"a success-case volume quoted without POS does. This is `POS × r` one level down."
        )

    sup_disc = describe_support(vsweep.n_discovery, vsweep.z, name="discovery")
    sup_dry = describe_support(vsweep.n_dry, vsweep.z, name="dry-with-attic")
    st.caption(
        f"{fig_ref('{b1}')}/{fig_ref('{b2}')} sweep entry with a fixed {vsweep.z_gap:.0f} m "
        f"entry-to-exit spacing, on the same "
        f"depth range as the row above. {fig_ref('{b2}')}'s dotted rule marks where those two "
        f"particular curves "
        f"meet — it is not a risked comparison, since P_well is unconditional and the regret "
        f"curve is conditional on a dry *and* charged outcome. {sup_disc.message()} "
        f"{sup_dry.message()}"
    )

    # **P(up-dip <= MEFS) at this well** (Lars, 2026-08-12) -- the workbook's
    # `P(Updip vol <= MCFS)@well`. Stated as a number rather than drawn as a curve,
    # and deliberately: it is the exact complement of the attic curve already on 3.6,
    # so a fifth line would be that curve mirrored about 50 % -- more ink, no more
    # information. As a number it says the useful thing in the useful direction.
    if vsweep.p_attic_exceeds_mefs is not None:
        _i = int(np.argmin(np.abs(vsweep.z - entry)))
        _p_regret = vsweep.p_attic_exceeds_mefs[_i]
        if np.isfinite(_p_regret):
            st.success(
                f"**If this well is dry but the prospect is charged, there is a "
                f"{1.0 - _p_regret:.1%} chance the volume left up-dip is *below* MEFS "
                f"({vsweep.mefs:.1f} MMboe)** — that is, a "
                f"{1.0 - _p_regret:.1%} chance being dry here costs you nothing material. "
                f"The complement, {_p_regret:.1%}, is the attic curve on {fig_ref('{b2}')}; this is the same "
                f"number said the other way round, which is the way a decision is usually put."
            )

    # ---- B7 and B8, both from the 2018 macro workbook (Lars, 2026-08-11)
    st.divider()
    st.subheader("The trade-off, and where it is commercial")
    tb1, = st.columns(1)
    with tb1:
        b7_scale = st.radio(
            f"{fig_ref('{b7}')} chance axis", ["log", "linear"], horizontal=True,
            key="w_b7_scale",
            format_func=lambda k: {"linear": "Linear", "log": "Log"}[k],
            help=("The chance axis, not the volume one (Lars, 2026-08-12). Linear shows "
                  "the absolute rate of exchange — how many MMboe a point of chance "
                  "buys. Log runs 1 % to 110 % and spreads the low-chance end out, "
                  "which is where the deep locations sit and where a linear axis "
                  "compresses the whole trade into the bottom centimetre."),
        )
        # 3.8's axes are volume and chance, so a candidate is a **point** on the
        # frontier rather than a rule -- see viz.add_well_points.
        _f_b7 = pfig_b7_frontier(vsweep, current_z=entry, chance_scale=b7_scale)
        add_well_points(_f_b7, vsweep, wells, selected=selected_well)
        _chart(_f_b7, key="b7")
    # 3.9 is full width (Lars, 2026-08-14). It carries three curves and a starred
    # interior maximum, and half a row was not enough to see where that peak sits.
    _f_b8 = pfig_b8_commercial_chance(vsweep, current_z=entry, zlim=zrow_sweep,
                                      height=TALL_PANEL_HEIGHT)
    add_well_markers(_f_b8, wells, selected=selected_well)
    _chart(_f_b8, key="b8")
    # A quarter taller (Lars, 2026-08-14): two starred peaks and a grey percentile
    # family share one pair of axes, and at row height the peaks are hard to place.
    _f_b9 = pfig_b9_chance_weighted(vsweep, current_z=entry, zlim=zrow_sweep,
                                    height=TALL_PANEL_HEIGHT)
    add_well_markers(_f_b9, wells, selected=selected_well)
    _chart(_f_b9, key="b9")
    st.caption(
        f"**{fig_ref('{b9}')} — the targeting tool.** `P_well × mean volume`, swept: a falling curve times a "
        "rising one, so it peaks somewhere in between and that depth maximises the expectation. "
        "It is drawn for the proven volume and for the whole well-associated volume, which peak "
        "in different places — the gap between those two stars is the exit depth's doing.\n\n"
        "**An expected value describes no outcome that can happen.** The well finds something "
        "near the success-case mean or it finds nothing; it never finds the chance-weighted "
        f"number. Use {fig_ref('{b9}')} to *rank* locations and {fig_ref('{b1}')} or "
        f"{fig_ref('{b7}')} to say how big the prize is."
    )
    st.caption(
        f"**{fig_ref('{b7}')}** is the "
        "most direct statement of what this tool is about: moving the well down-dip **buys volume "
        "with chance**. Read it as an efficient frontier — up and to the right is better and "
        "unavailable — with the depth labels giving the rate of exchange in metres. Neither axis is "
        "a depth, so this figure is exempt from the depth rule.\n\n"
        f"**{fig_ref('{b8}')}** puts the conditional and the unconditional MEFS probability on one "
        f"pair of axes, because the difference "
        "between them *is* the content. `Pmcfs(well)` **rises** down-dip — a deeper well finds a "
        "bigger accumulation — and is **conditional** on a discovery. `P_well` **falls** down-dip. "
        "Their product `Pc(well) = P_well × Pmcfs(well)` is **unconditional**: the chance of a "
        "commercial discovery, full stop. A rising curve times a falling one usually peaks in "
        "between, and that starred peak is where the well goes on commercial grounds — `Pc(well)` "
        "being the number Rose says to carry into an EMV."
    )

    # ------------------------------------------------------------ the comparison
    # Moved to tab 3 on 2026-08-14: that tab is the bench -- define candidates,
    # sweep them, compare them -- and this one is the write-up of the chosen well.
    if len(wells) > 1:
        st.divider()
        st.subheader("Compare the candidates")
        _rows = compare_wells(
            ts, ad if has_area else None,
            [(w.label, w.entry, w.exit) for w in wells],
            # `pos` here, not `chance.pos_prospect`: this tab unpacks the POS
            # directly and has no `chance` object. Same number, different name.
            pos_prospect=pos, reference=ref,
        )

        st.markdown("**Chance** — what each location does to the odds")
        st.dataframe(
            pd.DataFrame(chance_table(_rows)), hide_index=True, width="stretch",
            column_config={
                "Entry (m)": st.column_config.NumberColumn(format="%.0f"),
                "Exit (m)": st.column_config.NumberColumn(format="%.0f"),
                "POS prospect": st.column_config.NumberColumn(format="percent"),
                "r location": st.column_config.NumberColumn(format="percent"),
                "P well": st.column_config.NumberColumn(format="percent"),
                "Discovery trials": st.column_config.NumberColumn(format="%d"),
            },
        )

        _concepts = ["proven", "well_associated", "possible", "attic"] if has_area \
            else ["well_associated", "attic"]
        _vol = [r for c in _concepts for r in volume_table(_rows, c)]
        st.markdown("**Volumes, MMboe — success case, conditional on the outcome each "
                    "concept belongs to**")
        st.dataframe(
            pd.DataFrame(_vol), hide_index=True, width="stretch",
            column_config={c: st.column_config.NumberColumn(format="%.2f")
                           for c in ("P90", "P50", "Mean", "P10")},
        )

        st.markdown("**Risked volumes, MMboe** — mean × chance")
        st.dataframe(
            pd.DataFrame(risked_table(_rows)), hide_index=True, width="stretch",
            column_config={
                "P well": st.column_config.NumberColumn(format="percent"),
                "Expected proven": st.column_config.NumberColumn(format="%.2f"),
                "Expected well associated": st.column_config.NumberColumn(format="%.2f"),
            },
        )
        _best_p = max(_rows, key=lambda r: r.p_well)
        _best_v = max(_rows, key=lambda r: r.proven.get("mean", float("-inf"))
                      if r.proven else float("-inf"))
        _best_e = max(_rows, key=lambda r: r.expected_proven)
        st.caption(
            f"**Three different winners is the normal outcome, and the point.** Well "
            f"**{_best_p.label}** has the best chance ({_best_p.p_well:.1%}), well "
            f"**{_best_v.label}** the largest proven volume if it works, and well "
            f"**{_best_e.label}** the largest chance-weighted volume. The last is the "
            f"one a portfolio adds up; none of them is an economic answer, because "
            f"none of them knows what a well costs.\n\n"
            f"**Percentiles are conditional and risked figures are kept apart.** A "
            f"risked *percentile* is not reported at all: risking scales the "
            f"probability attached to a volume, never the volume, so the P50 volume "
            f"does not change — its exceedance probability does."
        )


    _inverse_section(vsweep, ts, mefs, wells, selected_well)



@st.fragment
def _band_section(ctx: Ctx):
    """B12, in its own fragment.

    Its controls only change how the same trials are cut, so re-running either sweep
    for them would be pure waste -- and the banding itself is a couple of
    ``np.percentile`` calls, so this redraws instantly.
    """
    st.divider()
    st.subheader("Resource by contact-depth band")
    ts, groups, vc = ctx.ts, ctx.groups, ctx.vc
    entry, exit_, mefs = ctx.entry, ctx.exit_, ctx.mefs
    selected_well = ctx.selected_well
    split_caveat(ctx)

    cb1, cb2, cb3 = st.columns([1.4, 1, 1])
    with cb1:
        # An explicit setting, not a default buried in code (non-negotiable 5): the
        # two modes answer the same question about different populations, and which
        # one is on screen changes the depth interval in every legend entry.
        # Equal depth interval is the default (Lars, 2026-08-12) -- the intervals are
        # then the same thing a structural section is contoured on, so a band can be
        # pointed at on a map. Equal count buys uniform statistical support instead.
        mode = st.radio(
            "Band the contacts by", list(BAND_MODES), horizontal=True, key="w_b12_mode",
            index=list(BAND_MODES).index("equal_width"),
            format_func=lambda k: BAND_MODE_LABELS[k],
            help=(
                "Equal depth interval is easier to read against a structural section, "
                "but the contact distribution is not uniform, so the shallow and deep "
                "bands come out thin and the percentile ladder is gated by the thinnest "
                "of them. Equal trial count gives every band the same number of trials, "
                "so the ladder is uniformly supported and the intervals vary instead."
            ),
        )
    with cb2:
        if mode == "equal_count":
            n_bands = st.number_input("Bands", 3, 10, DEFAULT_N_BANDS, 1, key="w_b12_n")
            interval = None
        else:
            n_bands = DEFAULT_N_BANDS
            interval = st.number_input("Interval (m)", 10.0, 400.0, 50.0, 10.0,
                                       key="w_b12_int")
    with cb3:
        show_proven = st.checkbox("Proven at this well", True, key="w_b12_proven")
        # Off by default (Lars, 2026-08-12): with two families drawn the diamonds are
        # the third thing on a crowded figure, and the mean is in the table anyway.
        show_mean = st.checkbox("Mean markers", False, key="w_b12_mean")

    # Both axes are switchable, and the four combinations are not cosmetic variants
    # of one another -- see the figure's docstring. Explicit settings, and the axis
    # titles carry whichever is on, because a curve that looks straight means
    # something different in each.
    cs1, cs2 = st.columns(2)
    prob_scale = cs1.radio(
        "Probability axis", list(PROBABILITY_SCALES), horizontal=True,
        key="w_b12_pscale",
        format_func=lambda k: {"probit": "Probit", "linear": "Linear 0-100 %"}[k],
        help=("Probit stretches the tails so that a lognormal plots as a straight "
              "line — which turns the shape of a distribution into something you can "
              "check against a ruler. Linear puts the probability back on its own "
              "even scale, which is the honest one for reading a chance off the axis."),
    )
    vol_scale = cs2.radio(
        "Resource axis", list(VOLUME_SCALES), horizontal=True, key="w_b12_vscale",
        format_func=lambda k: {"log": "Log", "linear": "Linear"}[k],
        help=("Log compares the bands by *proportion* — equal spacing means an equal "
              "ratio — and keeps the shallow bands readable when the deep ones are an "
              "order of magnitude larger. Linear compares them in absolute MMboe."),
    )

    try:
        bp = banded_percentiles(
            ts, groups, vc, z_entry=entry, z_exit=exit_, mode=mode,
            n_bands=int(n_bands), interval_m=interval,
        )
    except ValueError as exc:
        st.warning(str(exc))
        return

    # 3.12 takes no candidate rules and that is not an omission. Its axes are
    # resource and probability, so a well is not a line on it -- and more than that,
    # **the banding itself is anchored on the selected well's entry**, because a band
    # that straddles the entry mixes dry trials with discoveries. So each candidate
    # produces a *different figure*, not another curve on this one, and the title says
    # which well is on screen.
    _chart(pfig_b12_banded_percentiles(
        bp, mefs=mefs, show_proven=show_proven, show_mean=show_mean,
        probability_scale=prob_scale, volume_scale=vol_scale,
        well_label=selected_well), key="b12")

    dropped = ""
    if bp.n_bands_dropped:
        dropped = (f" {bp.n_bands_dropped} band(s) held fewer than {MIN_SUPPORT} trials "
                   "and are not drawn.")
    peel = _peel_note(bp) if show_proven else ""
    ladder = ", ".join("P" + str(q) for q in bp.percentiles)
    st.caption(
        f"**{fig_ref('{b12}')} — the prospect cut by where the contact lands.** Schneider's "
        "Figure 9 with the parameterisation changed: he draws one distribution per "
        "*productive-area increment*, this draws one per **contact-depth interval**. Area is a "
        "deterministic function of contact depth here, so the two band the same trials — but a "
        "depth is what a well chooses, and an area is not."
        "\n\n"
        "**Solid is the whole resource in the band; dotted is the part this well would prove.** "
        f"The well entry — **{entry:.0f} m** — is always a band boundary, so no band mixes dry "
        "trials with discoveries and the bands above the entry have no dotted curve at all: "
        f"nothing is proven there.{peel}"
        "\n\n"
        "**Blues for the total, mauve for the proven part**, each running light to dark with "
        "increasing depth — so the *concept* is in the hue and the *depth ordering* is in the "
        "lightness. Both families were blue at first, which made a solid and its own dotted twin "
        "read as one curve."
        "\n\n"
        "**On a probit probability axis a lognormal is a straight line.** A family that is "
        "straight and parallel says the bands differ by a scale factor and nothing else; "
        "curvature says the shape itself changes with depth. Switch the axis to linear to read a "
        "chance off it directly, and the resource axis between log (compare *proportions*) and "
        "linear (compare absolute MMboe). Percentiles are exceedance throughout: P99 is a small "
        "volume, P1 a large one, and the optional open diamond is each band's **mean at its own "
        "exceedance probability**, because a mean is not a percentile."
        "\n\n"
        f"Ladder drawn: {ladder} — a percentile is only reported where at least two trials fall "
        "beyond it, gated once on the thinnest series so that every band reports the same points."
        f"{dropped} The MEFS rule is a reference line: each band's crossing of it *is* "
        "`P(resource > MEFS | contact in that band)`, read straight off the probability axis. It "
        "never truncates a distribution."
    )


def _peel_note(bp) -> str:
    """One sentence naming what a deeper contact costs in *proof*.

    The figure's decision content, put in numbers rather than left to the eye: the
    total keeps growing down-dip and the proven part does not, so the ratio at the
    P50 falls. Compared between the shallowest and deepest bands that have a proven
    curve at all.
    """
    withp = [b for b in bp.bands if b.proven is not None and b.proven.size]
    if len(withp) < 2 or 50 not in bp.percentiles:
        return ""
    i = bp.percentiles.index(50)
    shallow, deep = withp[0], withp[-1]
    f_shallow = float(shallow.proven[i]) / float(shallow.total[i])
    f_deep = float(deep.proven[i]) / float(deep.total[i])
    return (f" At the P50 this well proves **{f_shallow:.0%}** of the {shallow.label} "
            f"band's resource but only **{f_deep:.0%}** of the {deep.label} band's — the "
            "widening gap between the two families is what a deeper contact costs in "
            "*proof*, on top of what it already cost in chance.")


def render(ctx: Ctx) -> None:
    _location_sweep_tab(ctx)
    _band_section(ctx)
