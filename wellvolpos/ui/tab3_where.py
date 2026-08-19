"""Tab ③ — where should the well go? Every figure that sweeps depth.

Comes *before* tab ④ deliberately: this is the comparative material a choice is
made from, and tab ④ is the answer for one depth. They were the other way round
until 2026-08-11, which put the answer in front of the material that informs it.

Behind ``st.fragment`` because the volume sweep re-splits every trial at every
depth and bootstraps each step -- the rule here is that dragging a slider must
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
    DEFAULT_CONFIDENCE,
    hurdle_curve,
    DEFAULT_RISK_FRACTION,
    candidate_depths,
    drilling_window,
    ce_curve,
    constrained_best,
    thickness_from_pay,
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
    pfig_b14_hurdle_cost,
    row_zlim,
)
from .common import (chart as _chart, figure_note, split_caveat,
                     well_readout)
from .context import Ctx
from .numbering import ref as fig_ref
from .loading import volume_sweep as _volume_sweep


@st.fragment
def _inverse_section(vsweep, ts, mefs):
    """B6, in its own fragment.

    The volume-to-prove slider must not re-run either sweep: at n=60 with a
    bootstrap that is the most expensive thing on the page, and the rule
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
    # **The mean only** (Lars, 2026-08-18). The four statistics answer materially
    # different questions and the figure said which -- but in practice the mean is
    # the one that gets used: it is what the source workbook computes and the only
    # one additive across prospects, which is what a portfolio needs. The other
    # three remain in core.sweep for anyone who wants them, and the guide still
    # explains the difference.
    stat = "mean"
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
    _chart(_f_b6, key="b6")
    # The worked sentence first, in the app's live numbers, because "how do I read
    # this" is the question B6 kept failing to answer (Lars, 2026-08-11).
    _worked = ""
    _worked_short = ""
    if inv.achievable:
        _worked_short = (
            f"To prove **{target:,.0f} MMboe** the well must enter at "
            f"**{inv.z_required:,.0f} m or deeper**, and P_well there is "
            f"**{inv.p_well_at:.1%}**."
        )
        _crest_p = float(vsweep.p_well[0]) if vsweep.p_well.size else float("nan")
        _worked = (
            f"**Reading it:** start at your target on the bottom axis — **{target:.0f} MMboe** — "
            f"go up to the curve, then across to the depth. It says **enter at "
            f"{inv.z_required:.0f} m or deeper**, and that costs you: P_well there is "
            f"**{inv.p_well_at:.1%}**, against **{_crest_p:.1%}** at the shallowest location "
            f"the sweep covers.\n\n"
        )
    figure_note(
        (_worked_short or "Name a volume to prove; read off the shallowest entry that guarantees it."),
        detail=_worked
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
        "its own definition and they are both useful; read one off the other and they are not.",
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

    # **Three headings, and a way in** (Lars, 2026-08-15, design review). Twelve
    # figures with no grouping is a wall, and nothing said which of them decide the
    # question. The headings name the question each group answers rather than the
    # figures they contain, so a reader picks the group before the figure.
    st.divider()
    well_readout(entry, exit_,
                 note="every curve below is swept at this spacing")
    st.markdown(
        "**A well tests a sub-population, not the prospect.** Every trial in the "
        "model is one possible version of it, and this well can only encounter "
        "hydrocarbons in the versions whose contact lies below the entry depth. "
        "Moving the well changes which versions those are — and nothing else on "
        "this tab."
    )
    _mv1, _mv2 = st.columns(2)
    _mv1.markdown(
        "**Move it shallower**\n\n"
        "- higher well POS\n"
        "- less volume demonstrated if it works\n"
        "- less volume left untested if it is dry"
    )
    _mv2.markdown(
        "**Move it deeper**\n\n"
        "- lower well POS\n"
        "- more volume demonstrated if it works\n"
        "- more volume left untested if it is dry"
    )
    st.caption("That is the whole trade. Every figure below is a way of pricing it.")

    st.info(
        "**If you only read three:** "
        + fig_ref("{a3}") + " for what the location costs in chance, "
        + fig_ref("{b7}") + " for the trade it buys, and "
        + fig_ref("{b6}") + " to invert it — name a volume, get a depth. "
        "The rest is the working."
    )

    st.subheader("What changes as the well moves")
    st.caption("Chance, outcome shares and how much a well here would tell you. "
               "No volumes yet — these are properties of the location alone.")
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
    # **A log share axis** (Lars, 2026-08-15). On a linear axis an outcome worth 2 %
    # of trials is a sliver beside the 60 % band next to it, and how fast the small
    # shares grow down-dip is most of what this figure is for.
    #
    # It stacks on both scales. It briefly did not, on the reasoning that stacking is
    # addition and a log scale does not preserve addition -- true, and beside the
    # point, since a band spans the *interval* between two cumulative boundaries and
    # an interval is well defined on any axis. See the figure's own note.
    _a2_scale = st.radio(
        fig_ref("{a2} share axis"), ["linear", "log"], horizontal=True, key="w_a2_scale",
        help="Both stack the four outcomes to 100 %. Log runs 1 % to 110 %, which is "
             "the only way to read the small shares — at the cost that a band's "
             "width on screen is then no longer its share, so read the boundaries.",
    )
    f_a2 = pfig_a2_outcome_tree(sweep, current_z=entry, current_exit=exit_, zlim=zrow_sweep,
                                share_scale=_a2_scale)
    f_a3 = pfig_a3_chance_decomposition(
        sweep, pos_prospect=pos, pos_trials=pos_trials, current_z=entry, current_exit=exit_, zlim=zrow_sweep)
    f_b3 = pfig_b3_uncertainty_reduction(sweep, current_z=entry, current_exit=exit_, zlim=zrow_sweep,
                                         height=TALL_PANEL_HEIGHT)
    # **The sensitivity fan** (Lars, 2026-08-12), from the workbook's charts 8 and 22.
    # It sits beside 3.2 because it is the same quantity with the chance table swept
    # instead of fixed: 3.2 answers "what is P_well here", this answers "how much of
    # that is the chance table rather than the geometry".
    f_b11 = pfig_b11_pos_sensitivity(sweep, pos_prospect=pos, current_z=entry, current_exit=exit_,
                                     zlim=zrow_sweep)
    # b11 is NOT levelled with them: it is drawn full width below the row, not as a
    # fourth panel in it, so borrowing the row's height only padded it.
    # 3.3 is **not** in the row (Lars, 2026-08-12). It carries two curves and an
    # argument about which population they are over, and a third of the width was not
    # enough for either. It is drawn full width directly under the row instead, which
    # is also where the reader meets it after 3.2.
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
        fig_ref("**{b3} measures information, not value.**") + " The peak is the depth whose result "
        "you can least predict — it values learning *dry* exactly as much as learning "
        "*large*, and it contains no dry-hole cost, no development case and no "
        f"discount rate. It is **not** a recommended location. For that, read "
        f"{fig_ref('{b9}')} (chance-weighted volume) and {fig_ref('{b8}')} "
        f"(commercial chance)."
    )
    figure_note(
        f"How much a well here narrows the answer. The best-informed depth is "
        f"**{sweep.z[int(np.nanargmax(sweep.uncertainty_reduction))]:,.0f} m**, and it "
        f"is rarely the best depth to drill.",
        detail=fig_ref("**{b3} — how much would a well here *tell* you?**") + " The expected shrinkage of the "
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
        "different places, the tails are doing the work and the recommendation is fragile.",
    )
    _chart(f_b11, key="b11", height=int(f_b11.layout.height))
    figure_note(
        "Every curve is the same shape scaled by a different POS — so revising the chance table and moving the well are independent levers.",
        detail=fig_ref("**{b11} — how much of your answer is the chance table?**") + " Every thin grey curve is "
        "`P_well` against depth for a different `POS_prospect`, a decile at a time; the heavy "
        "one is the POS actually in force. Deciles rather than every percentile: neighbours "
        "would differ by a hundredth and the fan would read as a smear.\n\n"
        "**They are all the same shape, scaled vertically**, and that is the content rather "
        "than a shortcoming of the drawing. `P_well = POS_prospect × r_location(z)`, and only "
        "the second factor moves with depth — so revising the chance table and moving the well "
        "are *independent* levers. A reader who has seen this fan cannot believe that drilling "
        "deeper fixes a poor chance table, which is exactly the confusion the whole "
        "`POS × r` separation exists to prevent.",
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
    st.subheader("What the well would prove")
    st.caption("The volume split, swept: what a well at each depth establishes, what "
               "it leaves unproven below, and what it forfeits up-dip if it is dry.")
    with st.spinner("Sweeping the volume split…"):
        vsweep = _volume_sweep(source.name, source.data,
                               tuple(sorted(overrides.items())), pos, gap, mefs, ref.value,
                               ctx.at_well_window)
    # **The schematic section is not drawn here** (Lars, 2026-08-14). 4.3 draws the
    # same section at the chosen well, from the same A(z), and two copies of one
    # figure on two tabs is one more place for them to disagree. The two curves that
    # remain are what this row was for: what the well proves, and what it risks.
    # **One per line** (Lars, 2026-08-14). Three depth panels in a row left each of
    # them a third of the width, and all three carry percentile families now.
    f_b2 = pfig_b2_chance_vs_regret(vsweep, current_z=entry, current_exit=exit_, zlim=zrow_sweep,
                                    height=TALL_PANEL_HEIGHT)
    f_b1 = pfig_b1_volume_split(vsweep, current_z=entry, current_exit=exit_, zlim=zrow_sweep,
                                height=TALL_PANEL_HEIGHT)
    f_b13 = pfig_b13_below_exit(vsweep, current_z=entry, current_exit=exit_, zlim=zrow_sweep,
                                height=TALL_PANEL_HEIGHT)
    _chart(f_b1, key="b1")
    sup_disc = describe_support(vsweep.n_discovery, vsweep.z, name="discovery")
    sup_dry = describe_support(vsweep.n_dry, vsweep.z, name="dry-with-attic")
    figure_note(
        f"What this well would prove, what it would leave up-dip, and the volume right at it — each conditional on its own outcome.",
        detail=f"**{fig_ref('{b1}')} — what the well proves, what it leaves, and the seam between "
        f"them.** Three volumes swept against entry depth at a fixed {vsweep.z_gap:.0f} m "
        f"entry-to-exit spacing, each with a bold mean and a dotted P90 / P50 / P10 in its "
        f"own colour."
        "\n\n"
        f"**At the well** is the boundary case, and the one worth dwelling on: the mean "
        f"**total** resource of the trials whose *contact* lands within "
        f"±{vsweep.at_well_window:g} m of the reservoir entry — neither a discovery nor a dry "
        f"hole, but the accumulation you get if the contact turns out to be exactly at your "
        f"well. It runs between the attic and the proven curves at every depth because that "
        f"is literally what it is, and it sits much closer to the attic than most people "
        f"expect. The window is set on tab ④, beside the metric of the same name."
        "\n\n"
        f"Both conditional groups thin at opposite ends — the discovery group fails down-dip, "
        f"the dry-with-attic group up-dip where almost nothing is dry. {sup_disc.message()} "
        f"{sup_dry.message()}",
    )

    st.divider()
    _chart(f_b2, key="b2")
    figure_note(
        f"Chance falls down-dip while the up-dip volume you would regret grows. Where they cross is stated below.",
        detail=f"**{fig_ref('{b2}')} — chance against regret.** `P_well` falls down-dip while the "
        f"chance a dry hole leaves something material up-dip rises, and the dotted rule marks "
        f"where those two particular curves meet. **It is not a risked comparison**: `P_well` "
        f"is unconditional and the regret curve is conditional on a dry *and* charged outcome, "
        f"so the crossing is where two different scales happen to cross rather than a "
        f"break-even."
        "\n\n"
        f"`P(unproven below LKH > MEFS | discovery)` is conditional on a **discovery**, not on "
        f"the well exiting in hydrocarbons — so a discovery that leaves nothing below LKH "
        f"correctly counts as failing the test. That is what makes it comparable with the "
        f"proven curve beside it, which is conditional on a discovery too. The other reading "
        f"follows by division: `P(> MEFS | HC to exit) = P(> MEFS | discovery) ÷ "
        f"P(HC to exit | discovery)`.",
    )

    st.divider()
    _chart(f_b13, key="b13")
    # The two unproven-below-LKH readings, in the app's own live numbers. Lars asked
    # what the curve meant and the honest answer is that one curve could not say it:
    # the unconditional mean is diluted by the discoveries that leave nothing below
    # the exit, and the conditional one is the prize without the chance of getting it.
    import numpy as _np

    _i_here = int(_np.argmin(_np.abs(_np.asarray(vsweep.z, dtype=float) - float(entry))))
    _p_any = (float(vsweep.p_well_exits_in_hc[_i_here])
              if vsweep.p_well_exits_in_hc is not None else float("nan"))
    _m_all = float(vsweep.below_lkh_mean[_i_here])
    _m_any = (float(vsweep.below_lkh_mean_if_any[_i_here])
              if vsweep.below_lkh_mean_if_any is not None else float("nan"))
    if _np.isfinite(_p_any) and _np.isfinite(_m_any):
        st.info(
            f"**The two unproven-below-LKH curves.** *If* the well "
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
    st.caption(
        f"**{fig_ref('{b13}')} moves with the exit, and almost nothing else does.** A well "
        f"that penetrates further proves more of the column and leaves less unproven, so "
        f"this curve and the proven one on {fig_ref('{b1}')} are the only two that change "
        f"when you move the exit slider without moving the entry. The chance, the attic and "
        f"the volume at the well are fixed by the **entry** alone — the discovery group is "
        f"defined by it. Every curve here is swept at the {exit_ - entry:,.0f} m spacing "
        f"set in tab ①."
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
    st.subheader("Where the optimum sits")
    st.caption("Three different optima — most volume for the chance, most "
               "chance-weighted volume, best commercial chance — and the inverse that "
               "turns a volume you must prove into a depth.")
    # ------------------------------------------------------ candidate depths
    # **The three optima, named together** (Lars, 2026-08-15, design review). The
    # tool already finds each of them, on three different figures, and a reader
    # comparing them had to remember two while looking at the third.
    #
    # It names depths; it does not select one. "Use this depth" buttons were offered
    # and declined on 2026-08-11 so the tab informs rather than decides, and that
    # still holds -- there is no control here, only a table and the figure each row
    # came from.
    # **Two criteria the expectation cannot express** (Lars, 2026-08-16).
    #
    # The hurdle answers the way a mandate is usually written -- "the best odds I can
    # get, subject to being confident it is commercial if it works" -- which is a
    # *constraint*, not an objective, and behaves differently: chance falls down-dip
    # while conditional commerciality rises, so the feasible set is a half-line and its
    # shallow end is the answer.
    #
    # The risk tolerance replaces the risk-neutral expectation with a certainty
    # equivalent. Its unit is MMboe, which is a real limitation stated in
    # core/utility.py: risk tolerance is properly monetary and no well cost appears
    # anywhere in this tool.
    # **The two controls get an explanation, not just a tooltip** (Lars, 2026-08-18:
    # *"I need some explanation to what the certainty equivalent means. And what is
    # rho?"*). A help icon is the right place for a definition a reader half-knows and
    # the wrong place for one they have never met -- and exponential utility is not
    # part of the exploration-geoscience vocabulary the rest of this app is written in.
    with st.expander("What these two controls mean"):
        st.markdown(
            "**The commercial confidence** is a *mandate*: how sure you insist on "
            "being that a discovery here would clear MEFS. It is not something the "
            "trials know — it is a statement about what your house will accept. "
            "Tighten it and the well is pushed down-dip, which costs chance.\n\n"
            "**The certainty equivalent** answers a different question: *what "
            "guaranteed volume would I swap this gamble for?*\n\n"
            "A well at some depth is a gamble — `P_well` of finding a distribution "
            "of volumes, and nothing otherwise. Its **expected value** is chance × "
            "mean, and an expectation is *risk-neutral*: it values a 10 % chance of "
            "500 MMboe exactly like a 50 % chance of 100. Most parties do not. The "
            "certainty equivalent is the certain volume you would accept **instead "
            "of** the gamble, and for anyone risk-averse it is smaller than the "
            "expectation. The gap between the two curves on the targeting figure is "
            "that difference, and it widens down-dip because that is where the "
            "low-chance, high-volume tail lives.\n\n"
            "**ρ (rho) is the risk tolerance** — the one number that says *how* "
            "risk-averse. It is the scale over which you stop caring about upside:\n\n"
            "- **ρ small** — very averse. A big prize far out in the tail adds almost "
            "nothing, so the certainty equivalent sits near the low end of the "
            "distribution.\n"
            "- **ρ large** — nearly risk-neutral. The certainty equivalent converges "
            "on the plain expectation, and this figure stops saying anything new.\n\n"
            "The formula is Cozzolino's, standard in petroleum decision analysis:\n\n"
            "`CE = −ρ · ln( p · E[e^(−V/ρ)] + (1 − p) )`\n\n"
            "**One honest limitation.** ρ is properly a *monetary* tolerance — the "
            "loss a company can absorb — and this tool carries no well cost, so ρ is "
            "expressed in MMboe and has to be calibrated by feel. It still ranks "
            "locations correctly when the well cost is the same at every depth, "
            "which is the case a single prospect is usually in. Do **not** carry a "
            "certainty equivalent in MMboe into an economic model as if it were an "
            "expected value."
        )

    _u1, _u2 = st.columns(2)
    with _u1:
        # **Integer per cent, not a fraction.** `format="%.0f%%"` on a 0.50-0.99 float
        # renders every value as 0 % or 1 %, because it formats the raw number -- the
        # slider worked and the panel below it read 71 % while the control said 1 %.
        # A label that contradicts the number it sets is worse than no label.
        _conf = st.slider(
            "Commercial confidence to insist on", 50, 99,
            int(round(DEFAULT_CONFIDENCE * 100)), 1, format="%d%%",
            key="w_confidence",
            help="P(a discovery clears MEFS). The panel reports the shallowest depth "
                 "from which it stays at or above this all the way down, and the best "
                 "P_well available under that constraint. This is a **mandate**, not "
                 "something the trials know: the default of 50 % asks only that the "
                 "median discovery clears MEFS, which usually binds weakly, so any "
                 "chance you give up above it is a choice you made.",
        )
    _succ = ts.col("resource")[ts.col("resource") > 0]
    _mean_succ = float(_succ.mean()) if _succ.size else 1.0
    with _u2:
        _rho_frac = st.slider(
            "Risk tolerance, as a multiple of the success-case mean", 0.10, 4.0,
            float(DEFAULT_RISK_FRACTION), 0.05, key="w_rho",
            help=f"Exponential utility. Smaller is more risk-averse; large is "
                 f"risk-neutral, where the certainty equivalent converges on the "
                 f"expectation. One mean here is {_mean_succ:,.0f} MMboe.",
        )
    _rho = max(_mean_succ * float(_rho_frac), 1e-6)
    with _u2:
        # **The multiple is unitless and the decision is not** (Lars, 2026-08-18).
        # A tolerance of "1.0x the success-case mean" is only calibratable once it is
        # stated in the units the prospect is measured in, so the product is printed
        # under the control that sets it rather than buried in a help tooltip.
        st.caption(
            f"= **{_rho:,.1f} MMboe** risk tolerance · success-case mean "
            f"{_mean_succ:,.1f} MMboe"
        )
    _constrained = (constrained_best(vsweep, confidence=float(_conf) / 100.0)
                    if vsweep.p_discovery_exceeds_mefs is not None else None)
    _ce = ce_curve(ts, vsweep, rho=_rho)

    _cands = candidate_depths(vsweep, sweep=sweep, constrained=_constrained,
                              risk_adjusted=_ce)
    if _cands:
        # **The window comes first, and the optima under it** (Lars, 2026-08-18:
        # *"the entry depth on this table does not give any guidance to an ideal depth
        # to go for"*). Every row used to be an *optimum* of some criterion, and an
        # optimum says what is best on one axis, never that a depth is wrong. Two rows
        # now do say so -- the shallowest that proves MEFS, and the depth past which a
        # dry hole leaves MEFS up-dip -- and between them they are a range you can
        # defend rather than five numbers to choose between.
        _floor, _ceiling = drilling_window(_cands)
        if _floor is not None or _ceiling is not None:
            if _floor is not None and _ceiling is not None and _floor < _ceiling:
                st.success(
                    f"**Drill between {_floor:,.0f} m and {_ceiling:,.0f} m TVDSS.** "
                    f"Shallower than {_floor:,.0f} m the well does not demonstrate "
                    f"{mefs:,.1f} MMboe even when it works; deeper than "
                    f"{_ceiling:,.0f} m a dry hole would have left that much "
                    f"un-tested up-dip. Every optimum below is a preference *within* "
                    f"this range — or, if it sits outside it, a warning."
                )
            elif _floor is not None and _ceiling is not None:
                st.warning(
                    f"**No depth satisfies both bounds.** The shallowest that proves "
                    f"{mefs:,.1f} MMboe is {_floor:,.0f} m, but a dry hole already "
                    f"leaves that much up-dip below {_ceiling:,.0f} m. That is a "
                    f"finding about the prospect, not an error: at this threshold "
                    f"there is no location that both demonstrates a commercial volume "
                    f"and cannot strand one. Read the two rows below and decide which "
                    f"risk you would rather carry."
                )
            elif _floor is not None:
                st.success(
                    f"**Drill at {_floor:,.0f} m TVDSS or deeper.** Shallower than "
                    f"that the well does not demonstrate {mefs:,.1f} MMboe even when "
                    f"it works. Nothing bounds it from below: the attic mean never "
                    f"reaches {mefs:,.1f} MMboe at any depth, so no location strands "
                    f"a commercial volume up-dip."
                )
            else:
                st.warning(
                    f"**Go no deeper than {_ceiling:,.0f} m TVDSS**, past which a dry "
                    f"hole leaves {mefs:,.1f} MMboe up-dip. No depth in the sweep "
                    f"proves that much, so there is no floor — this prospect cannot "
                    f"be demonstrated commercial by one well at any location."
                )

        _KIND = {"optimum": "Optimum", "floor": "⬇ At least this deep",
                 "ceiling": "⬆ No deeper than"}
        st.markdown("**Candidate depths** — what bounds the answer, and what optimises it")
        st.dataframe(
            pd.DataFrame([{
                "": _KIND.get(c.kind, c.kind),
                "Criterion": c.label,
                "What it maximises": c.maximises,
                "Entry (m TVDSS)": f"{c.depth:,.0f} m",
                "Value there": c.value,
                # Within 2 % of the best, so the value in the column before it is not
                # the value here -- only indistinguishable from it.
                "Equally good over": c.describe_plateau(),
                "Figure": fig_ref("{" + c.figure + "}"),
            } for c in sorted(_cands, key=lambda c: ({"floor": 0, "ceiling": 1}.get(c.kind, 2),
                                                     c.depth))]),
            hide_index=True, width="stretch",
        )
        # The longer reasoning, one click away -- it is what distinguishes a criterion
        # from the others rather than what it computes, so it does not belong in a
        # column a reader scans.
        with st.expander("Why each criterion is here, and what it is not"):
            for c in _cands:
                if c.note:
                    st.markdown(f"**{c.label}** — {c.note}")
        # **A range, where the maximum is weak.** Reporting one depth was false
        # precision: prospect B's commercial optimum moved 2064 -> 2115 m between two
        # grid resolutions at an identical Pc of 21.9 %, because the curve is exactly
        # flat above the shallowest contact -- there every success trial is a
        # discovery, so r_location is 1 and nothing changes until the entry passes it.
        # The span is every depth within 2 % of the best.
        if _constrained is not None:
            st.caption(_constrained.message())
            # **The hurdle swept, not the depth.** The row above is one point on this
            # curve; the curve earns a figure because Pc *falls* as the hurdle
            # tightens, which reads as a contradiction in a table and is obvious here.
            _hurdle = hurdle_curve(vsweep)
            if _hurdle.feasible.any():
                _chart(pfig_b14_hurdle_cost(_hurdle, current=float(_conf) / 100.0),
                       key="b14")
                _i = int(np.nanargmax(_hurdle.pc))
                figure_note(
                    f"Insisting on more confidence costs chance faster than it buys "
                    f"commerciality — so **Pc peaks at a "
                    f"{_hurdle.confidence[_i]:.0%} hurdle** and falls either side.",
                    detail=(
                        "**A constraint is not an objective.** Both curves fall to the "
                        "right: demanding more confidence pushes the well deeper, and "
                        "deeper costs `P_well` faster than it buys conditional "
                        "commerciality. The product — `Pc`, the number Rose says to "
                        "carry into an EMV — therefore falls as the hurdle tightens."
                        "\n\n"
                        "That is not an argument against setting a hurdle. It is an "
                        "argument for knowing what one costs, which is what the "
                        "vertical distance between the two curves shows at any x. "
                        "Labels on the upper curve are the entry depth each hurdle "
                        "requires, so the figure names the well as well as the price."
                        "\n\n"
                        "**Both axes are probabilities** and neither is a depth, so "
                        "this figure is exempt from the depth rule for the same "
                        "reason " + fig_ref("{b7}") + " is."
                    ),
                )
        # **When several rows land on one depth, say why** (Lars, 2026-08-18: *"the
        # entry depths are all the same — is this correct? I think not"*). They are,
        # and it is: above the shallowest sampled contact every success trial is a
        # discovery, so r_location is exactly 1 and every criterion that does not
        # involve MEFS is indifferent across that whole band. Without the sentence the
        # table looks like four rows computing one number, which is what it looked
        # like.
        _shared = [c for c in _cands if abs(c.depth - _cands[0].depth) < 1e-6]
        if len(_shared) > 2:
            _shallowest = min(float(np.min(vsweep.z)), _cands[0].depth)
            st.info(
                f"**{len(_shared)} of these land on the same depth, and that is a "
                f"property of the prospect rather than a bug.** Above the shallowest "
                f"sampled contact every success trial is a discovery, so `r_location` "
                f"is exactly 1, `P_well` is flat at POS_prospect, and the volume given "
                f"a discovery is the whole prospect distribution. Every criterion that "
                f"does not involve MEFS is therefore **indifferent** over that band — "
                f"the *Equally good over* column shows how wide it is. Where a "
                f"criterion cannot tell two depths apart the shallower one is reported, "
                f"because a shallower well never costs chance."
            )

        _flat = [c for c in _cands if c.is_flat]
        _widest = max(_cands, key=lambda c: (c.plateau[1] - c.plateau[0])
                      if c.plateau else 0.0)
        st.caption(
            ("**The depth is where the value is; the last column is where it "
             "barely matters.** " if _flat else
             "**Three measures, three depths.** ")
            + (f"Each *Value there* is the number at that entry depth. *Equally good "
               f"over* is every depth within 2 % of it, which on this prospect is a "
               f"wide band — {_widest.label.lower()} is flat over "
               f"{_widest.plateau[1] - _widest.plateau[0]:,.0f} m, so treating the "
               f"depth beside it as exact would be false precision. "
               if _flat else "")
            + "Best chance is the shallowest supported depth by construction — "
            "P_well only falls down-dip — so read that row as the top of the sweep "
            "rather than as advice.\n\n"
            "**The rest optimise different things, and one of them is not an optimum "
            "at all.** Chance-weighted volume is what a portfolio adds up and is "
            "*risk-neutral*; the risk-adjusted row is the same distribution under "
            "exponential utility, and can never sit deeper. Commercial chance is "
            "Rose's Pc. The confidence row is a **constraint**, not a criterion — the "
            "best odds available once the hurdle is met — which is why it can sit "
            "deeper than every optimum above it and still be the right answer to the "
            "question it was asked.\n\n"
            "Nothing here changes the well; the depth is set at the top of this tab."
        )
        st.divider()

    tb1, = st.columns(1)
    with tb1:
        b7_scale = st.radio(
            # **Label and help hidden** (Lars, 2026-08-18). The radio sits
            # directly above the figure it controls, so the words repeated the
            # figure's own number and the help icon invited a click for something
            # "Log / Linear" already says. The reasoning stays here, where it is
            # useful, rather than behind a tooltip nobody opens.
            f"{fig_ref('{b7}')} chance axis", ["log", "linear"], horizontal=True,
            label_visibility="collapsed",
            key="w_b7_scale",
            format_func=lambda k: {"linear": "Linear", "log": "Log"}[k],
        )
        _f_b7 = pfig_b7_frontier(vsweep, current_z=entry, chance_scale=b7_scale)
        _chart(_f_b7, key="b7")
    figure_note(
        "The trade-off frontier: moving down-dip buys volume with chance. Up and to "
        "the right is better, and unavailable.",
        detail=f"**{fig_ref('{b7}')}** is the most direct statement of what this tool "
        "is about: moving the well down-dip **buys volume with chance**. Read it as an "
        "efficient frontier — up and to the right is better and unavailable — with the "
        "depth labels giving the rate of exchange in metres.\n\n"
        "The bold curve is the well-associated mean with its P90 / P50 / P10 thin "
        "beside it, so the frontier reads as a range rather than a line: a frontier "
        "through means alone says what an *average* discovery buys and nothing about "
        "whether a poor one clears the bar.\n\n"
        "Neither axis is a depth, so this figure is exempt from the depth rule.",
    )

    # 3.10 is full width (Lars, 2026-08-14). It carries three curves and a starred
    # interior maximum, and half a row was not enough to see where that peak sits.
    _f_b8 = pfig_b8_commercial_chance(vsweep, current_z=entry, current_exit=exit_,
                                      zlim=zrow_sweep, height=TALL_PANEL_HEIGHT)
    _chart(_f_b8, key="b8")
    figure_note(
        "A rising conditional times a falling chance, so the commercial optimum sits "
        "somewhere in between — and the band marks how far it can move for nothing.",
        detail=f"**{fig_ref('{b8}')}** puts the conditional and the unconditional MEFS "
        "probability on one pair of axes, because the difference between them *is* the "
        "content. `Pmcfs(well)` **rises** down-dip — a deeper well finds a bigger "
        "accumulation — and is **conditional** on a discovery. `P_well` **falls** "
        "down-dip. Their product `Pc(well) = P_well × Pmcfs(well)` is "
        "**unconditional**: the chance of a commercial discovery, full stop, and the "
        "number Rose says to carry into an EMV.\n\n"
        "**Why the shallow end is flat.** Above the shallowest sampled contact every "
        "success trial is a discovery, so `r_location` is 1 and `P_well` cannot fall; "
        "the trials that drop out as the well deepens are the shallow-contact ones, "
        "which are also the small accumulations. Removing a trial that was never going "
        "to be commercial takes it out of `P_well` and out of `Pmcfs`'s denominator "
        "together, so their product does not move. `Pc` is exactly the share of trials "
        "that are **both** a discovery here **and** above MEFS — it only changes once "
        "the well starts excluding trials that *were* commercial.\n\n"
        "The light band is every depth within 2 % of the best `Pc`. It is a **plateau**, "
        "not an uncertainty: the curve genuinely barely moves across it, so a single "
        "starred depth would be false precision.",
    )

    # A quarter taller (Lars, 2026-08-14): two starred peaks and a grey percentile
    # family share one pair of axes, and at row height the peaks are hard to place.
    _f_b9 = pfig_b9_chance_weighted(vsweep, current_z=entry, current_exit=exit_,
                                    ce=_ce, zlim=zrow_sweep, height=TALL_PANEL_HEIGHT)
    _chart(_f_b9, key="b9")
    figure_note(
        "P_well times mean volume, and beside it the same distribution risk-adjusted. "
        "Where the two part company is what risk aversion costs a deep location.",
        detail=f"**{fig_ref('{b9}')} — the targeting tool.** `P_well × mean volume`, "
        "swept: a falling curve times a rising one, so it peaks somewhere in between "
        "and that depth maximises the expectation. It is drawn for the proven volume "
        "and for the whole well-associated volume, which peak in different places — "
        "the gap between those two stars is the exit depth's doing.\n\n"
        "**An expected value describes no outcome that can happen.** The well finds "
        "something near the success-case mean or it finds nothing; it never finds the "
        f"chance-weighted number. Use {fig_ref('{b9}')} to *rank* locations and "
        f"{fig_ref('{b1}')} or {fig_ref('{b7}')} to say how big the prize is.\n\n"
        "**The dashed green curve is the certainty equivalent** — the same distribution "
        "under exponential utility at the risk tolerance set above. An expectation is "
        "risk-neutral: it values a 10 % chance of 500 MMboe exactly like a 50 % chance "
        "of 100. A risk-averse party does not, and the gap between the two curves is "
        "that difference, widening down-dip because that is where the low-chance, "
        "high-volume tail lives.",
    )

    _inverse_section(vsweep, ts, mefs)



@st.fragment
def _band_section(ctx: Ctx):
    """B12, in its own fragment.

    Its controls only change how the same trials are cut, so re-running either sweep
    for them would be pure waste -- and the banding itself is a couple of
    ``np.percentile`` calls, so this redraws instantly.
    """
    st.divider()
    ts, groups, vc = ctx.ts, ctx.groups, ctx.vc
    entry, exit_, mefs = ctx.entry, ctx.exit_, ctx.mefs
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

    # **The banding is anchored on the well's entry**, because a band that straddles
    # it mixes dry trials with discoveries -- see core/bands.py. So moving the entry
    # produces a different figure rather than another curve on this one.
    _chart(pfig_b12_banded_percentiles(
        bp, mefs=mefs, show_proven=show_proven, show_mean=show_mean,
        probability_scale=prob_scale, volume_scale=vol_scale,
        well_label=None), key="b12")

    dropped = ""
    if bp.n_bands_dropped:
        dropped = (f" {bp.n_bands_dropped} band(s) held fewer than {MIN_SUPPORT} trials "
                   "and are not drawn.")
    peel = _peel_note(bp) if show_proven else ""
    ladder = ", ".join("P" + str(q) for q in bp.percentiles)
    figure_note(
        _band_headline(bp, entry) if show_proven else
        "The prospect split by contact-depth band, on log-probit axes — a lognormal "
        "is a straight line here.",
        detail=f"**{fig_ref('{b12}')} — the prospect cut by where the contact lands.** Schneider's "
        "Figure 9 with the parameterisation changed: he draws one distribution per "
        "*productive-area increment*, this draws one per **contact-depth interval**. Area is a "
        "deterministic function of contact depth here, so the two band the same trials — but a "
        "depth is what a well chooses, and an area is not."
        "\n\n"
        "**Solid is the part this well would prove; dotted is the whole resource in the "
        "band.** The emphasis is on the well, because that is the only half of the "
        "figure a location decision can move. "
        f"The well entry — **{entry:.0f} m** — is always a band boundary, so no band mixes dry "
        "trials with discoveries, and a band lying entirely up-dip of it has **no solid "
        "curve at all**: the well never enters it, so it proves none of it. Those bands "
        f"are named *above the well* in the legend.{peel}"
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
        "never truncates a distribution.",
    )


def _band_headline(bp, entry: float) -> str:
    """The tier-1 line, carrying this prospect's own numbers.

    A headline that only names the line styles teaches nothing about the prospect on
    screen -- and the styles are in the subtitle already. What the figure is *for* is
    the share of each band the well would prove, so that is what the one visible line
    says.
    """
    withp = [b for b in bp.bands if not b.above_the_well and b.proven is not None]
    above = [b for b in bp.bands if b.above_the_well]
    if not withp or 50 not in bp.percentiles:
        return ("The prospect split by contact-depth band, on log-probit axes — a "
                "lognormal is a straight line here.")
    deep = withp[-1]
    share = deep.proven_share(50, bp.percentiles)
    lead = (f"At the P50 this well proves **{share:.0%}** of the deepest band "
            f"({deep.label})") if share == share else ""
    tail = (f", and none of the {len(above)} band(s) that lie entirely above "
            f"{entry:,.0f} m." if above else ".")
    return (lead + tail + " Solid is what the well proves, dotted the whole band.")


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
    # **3.12 behind an expander** (Lars, 2026-08-15). It is the specialist view on this
    # tab -- log-probit axes, a banding scheme with its own two controls -- and it is
    # also the most expensive figure here. A reader choosing a depth does not need it
    # open; a reader arguing about the shape of the distribution does, and one click is
    # the right price for that.
    with st.expander("Resource by contact-depth band — the prospect cut by where the "
                     "contact lands", expanded=False):
        _band_section(ctx)
