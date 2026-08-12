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

import streamlit as st

from ..core import (
    MIN_SUPPORT,
    describe_support,
    TARGET_STATISTIC_LABELS,
    TARGET_STATISTICS,
    invert_volume_target,
    run_sweep,
    volume_target_curve,
)
from ..viz import (
    level_row,
    pfig_a2_outcome_tree,
    pfig_a3_chance_decomposition,
    pfig_b0_section,
    pfig_b1_volume_split,
    pfig_b2_chance_vs_regret,
    pfig_b3_uncertainty_reduction,
    pfig_b6_inverse,
    pfig_b7_frontier,
    pfig_b8_commercial_chance,
    pfig_b9_chance_weighted,
    row_zlim,
)
from .common import chart as _chart, split_caveat
from .context import Ctx
from .loading import volume_sweep as _volume_sweep


@st.fragment
def _inverse_section(vsweep, ts, mefs):
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
            "above. B6 returns the shallowest entry depth from which that statistic stays "
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
    _chart(pfig_b6_inverse(vsweep, target=target, ts=ts, mefs=mefs,
                           statistic=stat), key="b6")
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
        + "The workbook's H38–H40 block as a curve. Marker colour is P_well at that depth — the "
        "cost side of the trade — because a second y-axis is not allowed and the trade is the "
        "point. The shaded band is the bootstrap interval on the proven mean, inverted through "
        "the same curve, so it widens down-dip where the discovery group thins. The level is "
        "nominal: a percentile bootstrap under-covers on small skewed samples.\n\n"
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
        "workbook's own `BA` column averages those contacts into one number and calls it a "
        "required depth; an average over that range is not one.\n\n"
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

    def _split_caveat() -> None:
        split_caveat(ctx)

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
    f_b3 = pfig_b3_uncertainty_reduction(sweep, current_z=entry, zlim=zrow_sweep)
    level_row(f_a2, f_a3, f_b3)
    c1, c2, c3 = st.columns(3)
    with c1:
        _chart(f_a2, key="a2", height=int(f_a2.layout.height))
    with c2:
        _chart(f_a3, key="a3", height=int(f_a3.layout.height))
    with c3:
        _chart(f_b3, key="b3", height=int(f_b3.layout.height))
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
    f_b0 = pfig_b0_section(ad, z_entry=entry, z_exit=exit_, zlim=zrow_sweep)
    f_b1 = pfig_b1_volume_split(vsweep, current_z=entry, zlim=zrow_sweep)
    f_b2 = pfig_b2_chance_vs_regret(vsweep, current_z=entry, zlim=zrow_sweep)
    level_row(f_b0, f_b1, f_b2)
    d1, d2, d3 = st.columns(3)
    with d1:
        _chart(f_b0, key="b0", height=int(f_b0.layout.height))
    with d2:
        _chart(f_b1, key="b1", height=int(f_b1.layout.height))
    with d3:
        _chart(f_b2, key="b2", height=int(f_b2.layout.height))
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
        b7_scale = st.radio(
            "3.7 volume axis", ["linear", "log"], horizontal=True, key="w_b7_scale",
            format_func=lambda k: {"linear": "Linear", "log": "Log"}[k],
            help=("Linear shows the absolute rate of exchange — how many MMboe a point "
                  "of chance buys. Log shows the proportional one, which is the readable "
                  "choice when volume spans an order of magnitude across the swept range "
                  "and the shallow end is crushed into the axis."),
        )
        _chart(pfig_b7_frontier(vsweep, current_z=entry, volume_scale=b7_scale), key="b7")
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

    _inverse_section(vsweep, ts, mefs)


def render(ctx: Ctx) -> None:
    _location_sweep_tab(ctx)
