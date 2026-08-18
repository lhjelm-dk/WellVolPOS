"""Tab ④ — what this well gets at the depth chosen on tab ③.

The KPI strip, both engines' summaries, the class table, the live section,
A6's overlap, the C1/C2 concepts pair and the conceptual map view. The map
belongs here rather than on ② because it draws the entry contour and the
three areas a *well* divides the closure into."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from ..core import (
    thickness_from_pay,
    rose_partition,
    boundary_ties,
    at_the_well_volume,
    outcome_overlap,
    REPORT_PERCENTILES,
    chance_from_counts,
    class_percentiles,
    class_summary,
    expected_volume,
    group_summary,
)
from ..viz import (
    OVERLAP_OPACITY,
    pfig_a6_overlap,
    pfig_b0_section,
    pfig_c1_section,
    pfig_c2_exceedance,
    pfig_c3_mefs_bars,
    pfig_c5_partitions,
    pfig_c6_outcome_tree,
    pfig_map_view,
)
from ..core import MEFS_RUNGS, c2_crossings, headline as _headline, mefs_readout
from ..core.rose import AT_WELL_WINDOW_M, commercial_chance
from ..viz.theme import reference_label
from .common import (C2_HEIGHT, chart as _chart, figure_note, kpi_ladder,
                     split_caveat, track_deltas, well_readout)
from .context import Ctx
from .numbering import ref as fig_ref


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

    # ------------------------------------------------------- the headline
    # **The answer, before the working** (Lars, 2026-08-15, design review). The tool
    # computed P_well, Pc, the proven mean and the attic mean and never put them in
    # one place, so a reader had to assemble the conclusion from six figures.
    #
    # `core.summary` does the assembly. This block only formats -- a tab that
    # computes is a tab that can disagree with the figure under it.
    _cs_head = (class_summary(vc, groups, mefs=mefs, resource=ts.col('resource'))
                if has_area else None)
    _cc_head = (commercial_chance(ts, groups, vc.proven, chance.p_well, mefs)
                if has_area else None)
    _head = _headline(entry=entry, exit_=exit_, chance=chance, groups=groups,
                      class_stats=_cs_head, commercial=_cc_head,
                      mefs=mefs if has_area else None)

    st.subheader("The answer at this well")
    st.markdown(_head.sentence())
    well_readout(entry, exit_, show_depths=False)

    # **What the last move did** (Lars, 2026-08-15). The tool's subject is
    # sensitivity to one depth and it re-rendered everything with no sign of what
    # moved. The comparison is dropped whenever the file, the chance table or the
    # threshold changes -- see ui.common.track_deltas.
    _d = track_deltas(
        "headline",
        f"{ctx.dataset}|{chance.pos_prospect:.6f}|{mefs:.4f}|{ref.value}",
        (entry, exit_),
        {k: v for k, v in (("p_well", _head.p_well * 100.0),
                           ("pc", (_head.pc_well or float("nan")) * 100.0),
                           ("proven", _head.proven_mean or float("nan")),
                           ("attic", _head.attic_mean or float("nan")))},
        fmt="{:+.1f}",
    )

    # **P_well is not in this strip.** It is bold in the sentence above and it is the
    # product of the decomposition a few lines below, where its delta is shown. Three
    # copies on one tab is what the "P_well appears once" note already fixed.
    _h = st.columns(4)
    _h[0].metric("Pc — commercial",
                 "—" if _head.pc_well is None else f"{_head.pc_well:.1%}",
                 delta=_d.get("pc"),
                 help="Rose's commercial chance: the chance of finding more than "
                      "MEFS. Always at or below P well.")
    _h[1].metric("Proven mean",
                 "—" if _head.proven_mean is None else f"{_head.proven_mean:,.1f}",
                 delta=_d.get("proven"),
                 help="MMboe. Conditional on a discovery — the headline volume this "
                      "well would establish.")
    _h[2].metric("Attic mean",
                 "—" if _head.attic_mean is None else f"{_head.attic_mean:,.1f}",
                 delta=_d.get("attic"),
                 help="MMboe. Conditional on a CHARGED DRY HOLE — what is left "
                      "up-dip if this well misses. A different outcome from the "
                      "three beside it.")
    _h[3].metric("Reservoir penetrated", f"{_head.gap:,.0f} m",
                 help="Entry to exit. Tab ① holds the sliders; tab ③ sweeps every "
                      "entry depth and shows what each one buys.")
    st.caption(
        ("**Deltas are against the well before the last move**, in points of chance "
         "and MMboe. They clear when the trial file, the chance table or MEFS "
         "changes — a difference measured across those is not a sensitivity.\n\n"
         if _d else "")
        + "**Two chances and two volumes, and they are not on one footing.** P well and "
        "Pc are unconditional — they already carry the chance of the outcome. The two "
        "volumes are success-case means, each conditional on a *different* event: "
        "proven on a discovery, attic on a charged dry hole. Multiplying a volume by "
        "a chance from this row gives an expectation, not a resource."
    )

    # **The headline sentence as a picture.** Every share comes from
    # Groups.risked_shares -- an outcome tree that counts trial masks reports
    # POS_trials under a P_well label, which is what A2 did and what B4 did the
    # arithmetic equivalent of.
    _chart(pfig_c6_outcome_tree(
        groups, pos_prospect=chance.pos_prospect, p_well=chance.p_well,
        pc_well=None if _cc_head is None else _cc_head.pc_well), key="c6")

    st.divider()
    _split_caveat()

    # --------------------------------------------------------------- the chance
    # **P_well appears once**, and here, as the multiplication it is: POS x r. It used
    # to be printed twice -- 20.3 % among the volumes and 0.2031 in the decomposition
    # four rows down -- which leaves a reader working out whether the two agree.
    # Reading the row left to right *is* the argument the whole tool rests on.
    st.markdown("##### The chance this well finds hydrocarbons")
    ch = st.columns(4)
    ch[0].metric("POS prospect", f"{chance.pos_prospect:.1%}",
                 help="The chance the PROSPECT holds hydrocarbons. From the chance table "
                      "in tab ②, or from the trials, depending on the risking convention.")
    ch[1].metric("× r location", f"{chance.r_location:.1%}",
                 help="P(contact deeper than the well | hydrocarbons present). The only "
                      "quantity the well's position controls.")
    ch[2].metric("= P well", f"{chance.p_well:.1%}", delta=_d.get("p_well"),
                 help="The chance THIS well finds hydrocarbons. Never quote it as the "
                      "prospect's chance, and never quote the prospect's as this.")
    ch[3].metric("Reference contour", reference_label(ref).replace("-referenced", ""),
                 help="What r location is measured against. An explicit setting, because "
                      "it changes the number.")
    st.caption(
        f"`P_well = POS_prospect × r_location` — **{chance.pos_prospect:.4f} × "
        f"{chance.r_location:.4f} = {chance.p_well:.4f}**. Two numbers with two meanings, "
        "never multiplied into one for reporting: only the second moves when the well moves, "
        "so a poor chance table cannot be fixed by drilling deeper."
    )

    # ------------------------------------------------------- the commercial chance
    # **P_well and Pc are different questions and the app said only the first here**
    # (Lars, 2026-08-14). P_well is the chance of *seeing hydrocarbons*; Pc is the
    # chance of seeing enough of them to be worth developing. A well can be very
    # likely to find something and unlikely to find a field.
    #
    # Rose gives Pc(well) as the number to carry into an EMV calculation. It is a
    # *chance*, not an economic value, which is why it is in scope while economics is
    # not. Drawn as the same left-to-right multiplication as the row above, because
    # that is the form that stops the two factors being confused for each other.
    if has_area:
        # `ctx.vc` -- the split computed once in app.py and passed down. Re-splitting
        # here would let this block and the volumes below disagree about one well.
        _cc = commercial_chance(ts, groups, vc.proven, chance.p_well, mefs)
        st.markdown("##### The chance this well finds a *commercial* accumulation")
        cc_cols = st.columns(4)
        cc_cols[0].metric("P well", f"{chance.p_well:.1%}",
                          help="The chance of finding hydrocarbons at all — carried "
                               "down from the row above.")
        cc_cols[1].metric("× Pmcfs(well)", f"{_cc.p_mcfs_downdip:.1%}",
                          help="Given a discovery, the chance the WELL-ASSOCIATED "
                               "volume exceeds MEFS. Conditional on the discovery, so "
                               "it is not a chance of anything on its own.")
        # **Pc carries its sampling interval** (Lars, 2026-08-14, asking whether
        # uncertainty in exceeding MCFS should be incorporated). It is the number an
        # EMV takes, so it is the one worth knowing the precision of.
        _pc_lo, _pc_hi = _cc.pc_interval(0.90)
        cc_cols[2].metric("= Pc(well)", f"{_cc.pc_well:.1%}",
                          help="Rose's commercial chance at this location: the "
                               "unconditional chance of a commercial discovery. This "
                               "is the number an EMV calculation takes.")
        cc_cols[2].caption(f"90 % CI {_pc_lo:.1%} – {_pc_hi:.1%}")
        cc_cols[3].metric("MEFS / MCFS", f"{mefs:,.1f} MMboe",
                          help="Minimum economic (this app) or commercial (Rose) "
                               "field size — the same threshold under two names. Set "
                               "in tab ①, drawn as a line, never applied to the "
                               "distributions.")
        figure_note(
            f"P_well is the chance of *seeing* hydrocarbons; Pc the chance of seeing a "
        f"**developable** accumulation. A location can score well on one and poorly on the other.",
            detail=f"`Pc(well) = P_well × Pmcfs(well)` — **{chance.p_well:.4f} × "
            f"{_cc.p_mcfs_downdip:.4f} = {_cc.pc_well:.4f}**. Read the two apart: "
            f"**P_well {chance.p_well:.1%}** is the chance of *seeing hydrocarbons*, "
            f"**Pc {_cc.pc_well:.1%}** the chance of seeing a *developable* accumulation. "
            f"A location can score well on the first and poorly on the second.\n\n"
            f"**Rose conditions on the whole well-associated volume; "
            f"{fig_ref('{b2}')} conditions on the proven split.** On the proven volume "
            f"alone the conditional chance is {_cc.p_mcfs_proven:.1%}, which would give "
            f"Pc = {chance.p_well * _cc.p_mcfs_proven:.1%}. Both are legitimate and they "
            f"answer different questions — what the accumulation holds, against what "
            f"this well would establish. Neither may be quoted as the other.\n\n"
            f"**The interval under Pc is sampling error, and only that.** `Pc` reduces "
            f"to one binomial proportion — the identity is exact, because the discovery "
            f"count cancels:\n\n"
            f"`Pc = P_well × Pmcfs = POS × (n_disc / n_succ) × (n_comm / n_disc) = "
            f"POS × (n_comm / n_succ)`\n\n"
            f"so here **{_cc.n_commercial:,} of {_cc.n_success:,}** success trials are "
            f"both a discovery at this depth and above MEFS, and one Wilson interval on "
            f"that share — scaled by POS — is the interval on Pc. Wilson rather than the "
            f"textbook normal interval because the normal one runs outside 0–100 % "
            f"exactly where this is read hardest: at the deep end, where the discovery "
            f"group is small and the share is near zero.\n\n"
            f"**POS_prospect is deliberately outside it.** It comes from the chance "
            f"table, which is a judgement rather than a sample, so it has no sampling "
            f"error to report — its uncertainty is what {fig_ref('{b11}')} draws. And no "
            f"interval computed from these trials can reach the larger question of "
            f"whether the input distributions are right.",
        )

    if has_area:
        cs = class_summary(vc, groups, mefs=mefs, resource=ts.col('resource'))
        gs = group_summary(ts, groups)
        # Same thickness and apex the split above used, so Rose's cut and ours cannot
        # disagree about the geometry.
        _thickness = thickness_from_pay(ts, ad).thickness
        _apex = float(ad.apex_estimate())
        # ------------------------------------------------------ the success case
        # Every number in this block is **conditional** on its own outcome and none of
        # them is risked. Keeping them in one block, away from the expected volumes
        # below, is the layout doing the work that this codebase's recurring bug --
        # an unrisked number under a risked label -- otherwise requires prose to do.
        st.markdown("##### Volumes if it works, in MMboe — success case, unrisked")
        # The window is a **setting**, not a constant (Lars, 2026-08-12). No trial
        # lands exactly on the entry, so "the volume when the contact is at the well"
        # needs a tolerance, and the answer moves with it: wider takes in contacts
        # that are not at the well, narrower runs out of trials. +/-2 m is the default
        # because it is what the original calculation used, but it is a tuning
        # constant with a result attached and it belongs on screen.
        _atw_win = st.number_input(
            "At-the-well window, ± m", 0.5, 25.0, step=0.5,
            key="w_atw_window",
            help=fig_ref(
                "How close a trial's contact must be to the reservoir entry to count "
                "as landing *on* the well. Widen it for more trials and a blurrier "
                "answer; narrow it until the count gets too small to mean anything. "
                "**It also sets the at-the-well curve on {b1}**, which used to keep the "
                "2 m default no matter what was typed here."),
        )
        # The well-associated volume in the **same shape as tab ②'s prospect row**,
        # so the two can be read one above the other. That comparison -- what the
        # prospect holds against what this well would find -- is the one the tool
        # exists to make, and it was previously two different layouts two tabs apart.
        st.markdown("**Well associated | discovery** — the accumulation this well "
                    "would find, in the same shape as the prospect row on tab ②")
        kpi_ladder(
            chance_label="P well", chance=chance.p_well,
            values=cs["discovery"],
            chance_help="The chance THIS well finds hydrocarbons. Unconditional; the "
                        "six beside it are conditional on it happening.",
            value_help="MMboe, success case — conditional on a discovery.",
        )
        st.divider()

        _atw, _atw_n = at_the_well_volume(ts, entry, window_m=float(_atw_win))
        k = st.columns(4)
        k[0].metric("Proven mean — headline KPI", f"{cs['proven']['mean']:.2f}",
                    help="What this well would establish between entry and exit.")
        k[1].metric("Well associated mean", f"{gs['discovery']['mean']:.2f}",
                    help="The whole accumulation given a discovery: crest to contact, "
                         "which is proven + possible. NOT Rose's 'downdip' volume — his "
                         "partitions the closure at the well, so his downdip is only the "
                         "part BELOW it. See tab ⑥.")
        k[2].metric("Attic mean — if dry but charged", f"{cs['attic_dry_hole']['mean']:.2f}",
                    help="What is left up-dip if the well is dry but the prospect is charged.")
        if _atw_n:
            k[3].metric("At the well — contact on entry", f"{_atw:.2f}",
                        help=f"Mean of the {_atw_n:,} trials whose contact lands within "
                             f"±{_atw_win:g} m of the reservoir entry. Neither a discovery "
                             f"nor a dry hole — the boundary case.")

        if _atw_n:
            _span = gs["discovery"]["mean"] - cs["attic_dry_hole"]["mean"]
            _frac = ((_atw - cs["attic_dry_hole"]["mean"]) / _span) if _span else float("nan")
            figure_note(
                f"The accumulation if the contact lands *on* the well — neither a discovery "
            f"nor a dry hole, from {_atw_n:,} trials within ±{_atw_win:g} m.",
                detail=f"**The boundary case.** The accumulation you get if the hydrocarbon–water "
                f"contact lands *on* the well: **{_atw:.2f} MMboe**, which sits **{_frac:.0%}** "
                f"of the way from the attic mean ({cs['attic_dry_hole']['mean']:.2f}) to the "
                f"discovery mean ({gs['discovery']['mean']:.2f}) — closer to the dry case than "
                f"most people expect. This is Rose's *“No Regrets”* volume in probabilistic "
                f"form: his is a single deterministic product, this is the mean of the trials "
                f"that actually landed there, so it carries the model's own correlations. He is "
                f"candid that the deterministic version *“is an oversimplification”*, because "
                f"*“there remains a chance the updip volume will exceed MCFS”* — which is what "
                f"{fig_ref('{b2}')}'s regret curve answers.",
            )

        # --------------------------------------------------- every volume vs the line
        # **A percentile has no probability of exceeding a threshold** (Lars asked for
        # one, 2026-08-14). P90 is a fixed volume: it clears MEFS or it does not, and
        # that is 0 or 1. What has a probability is the *concept*, and that probability
        # is the exceedance percentile the line sits at -- one number, not four.
        #
        # So the table gives both halves of the reading: the ladder with a clears-or-not
        # mark, which says between which percentiles the threshold falls and is what a
        # reader scans for, and the exact chance beside it. core/mefs.py owns the
        # arithmetic and asserts the two cannot contradict each other.
        _mr = mefs_readout(vc, groups, cs, mefs)
        # **The commercial class, with its own n and chance** (Lars, 2026-08-15).
        # It is the accumulation *given* it clears MEFS, so its chance is Pc and its
        # mean sits above the well-associated mean -- which is Longley's point, not a
        # contradiction: a threshold raises the surviving mean while lowering the
        # chance of surviving. Both halves are in the row, so neither reads alone.
        _comm = cs.get("commercial")
        if _comm and _comm["n"] > 0:
            st.markdown("##### The commercial accumulation — given it clears "
                        f"MEFS / MCFS, {mefs:,.1f} MMboe")
            st.dataframe(
                pd.DataFrame([{
                    "Volume": "Commercial accumulation",
                    "Chance (Pc)": f"{_cc.pc_well:.1%}",
                    "Trials": f"{int(_comm['n']):,} of {int(cs['discovery']['n']):,} "
                              f"discoveries",
                    "P90": f"{_comm['p90']:,.2f}", "P50": f"{_comm['p50']:,.2f}",
                    "Pmean": f"{_comm['mean']:,.2f}", "P10": f"{_comm['p10']:,.2f}",
                }]), hide_index=True, width="stretch",
            )
            figure_note(
                f"The distribution Pc belongs to. Its mean sits **above** the "
                f"well-associated mean because cutting at a threshold raises what is left.",
                detail=f"**The distribution behind Pc.** Everything else on this tab is "
                f"conditional on an event Pc does not describe — this is the one it "
                f"does. Its mean is **{_comm['mean']:,.1f} MMboe** against the "
                f"well-associated **{cs['discovery']['mean']:,.1f}**, and that gap is "
                f"the threshold at work rather than an inconsistency: cutting at MEFS "
                f"raises the mean of what is left while lowering the chance of getting "
                f"it (Longley 2026). **The four classes above are not truncated** — "
                f"this is an additional class conditional on a different event, which "
                f"is why the app can show it without applying the cut anywhere else.",
            )
            st.divider()

        st.markdown(f"##### Every volume against the MEFS / MCFS line, {mefs:,.1f} MMboe")
        _rows = []
        for _c in _mr.concepts:
            _row = {"Volume": _c.label, "Conditional on": _c.condition}
            for _r in MEFS_RUNGS:
                _v = _c.volumes[_r]
                _name = "Pmean" if _r == "mean" else _r.upper()
                _row[_name] = f"{_v:,.2f} {'✓' if _c.clears(_r) else '✗'}"
            _row["P(> MEFS)"] = f"{_c.p_exceeds:.1%}"
            _row["Trials"] = f"{_c.n:,}"
            _rows.append(_row)
        st.dataframe(pd.DataFrame(_rows), hide_index=True, width="stretch")
        _wa = _mr.by_key("discovery")
        figure_note(
            f"✓ clears MEFS, ✗ does not. The ticks bracket the answer; the last column "
            f"is the answer.",
            detail=f"**✓ clears the line, ✗ does not.** The ticks bracket the answer and the "
            f"last column is the answer: on the well-associated volume the threshold "
            f"falls {_wa.bracket()}, and the chance of clearing it is "
            f"**{_wa.p_exceeds:.1%}** — which is the same {mefs:,.1f} MMboe read as a "
            f"percentile rather than as a volume.\n\n"
            f"**Each row is conditional on a different event**, named in its own column, "
            f"so the four probabilities are not comparable and must not be summed. "
            f"*Pmean is not a percentile* — it sits between P50 and P10 because the "
            f"distribution is right-skewed, not because it is a rung.\n\n"
            f"**The line is never applied to the distributions.** Per Longley (2026) a "
            f"volume cut-off raises the unrisked mean while lowering commercial chance, "
            f"and the two do not cancel — so MEFS is read against, never used to filter.",
        )

        st.divider()
        # ----------------------------------------------- the same closure, Rose's way
        # Added 2026-08-13. Their partition is at the *well*, ours at the *penetrated
        # interval*, and both are useful -- but mixing the vocabularies silently is how
        # a number gets quoted under the wrong name, which had already happened once
        # here. Shown side by side so the two cuts can be compared rather than
        # confused.
        _rp = rose_partition(ts, ad, entry, thickness=_thickness, apex=_apex)
        if _rp.n_discovery:
            st.markdown("##### The same closure cut Rose's way — split at the well, not at "
                        "the penetrated interval")
            rc = st.columns(3)
            rc[0].metric("Rose updip — crest to well", f"{_rp.updip_mean:.2f}",
                         help="What a dry hole leaves behind. His deterministic "
                              "'No Regrets' volume is this, taken at the means.")
            rc[1].metric("Rose downdip — well to contact", f"{_rp.downdip_mean:.2f}",
                         help="The extra volume the well opens up by being deeper. NOT "
                              "the whole accumulation.")
            rc[2].metric("Their sum", f"{_rp.total_mean:.2f}",
                         help="Equals the well-associated mean above, by construction.")
            # **The contact must be a discovery's.** Passing the median *successful*
            # contact drew this for a dry hole -- 2203.3 m against a 2205 m entry on
            # prospect B -- so nothing sat below either cut and only the upper half of
            # each panel appeared. Conditioning on discovery is also the right reading:
            # the figure is about how a *discovery* gets partitioned.
            _disc_c = ts.col("contact")[(ts.col("resource") > 0)
                                        & (ts.col("contact") > entry)]
            if _disc_c.size:
                _chart(pfig_c5_partitions(
                    ad, z_entry=entry, z_exit=exit_,
                    z_contact=float(np.median(_disc_c)),
                    area_scale=area_scale), key="c5")
            figure_note(
                "Rose cuts at the well, this app at the interval the well penetrates. The "
                "violet band is the slice they disagree about"
                + (f", drawn at the median discovery contact, "
                   f"{float(np.median(_disc_c)):,.0f} m." if _disc_c.size else "."),
                detail=f"**Two cuts of one closure.** Rose splits at the well; this app splits at "
                f"the interval the well actually penetrates, because a well proves what it "
                f"drills through. So his updip ({_rp.updip_mean:.2f}) is our proven "
                f"({cs['proven']['mean']:.2f}) *minus* the entry-to-exit slice, and his "
                f"downdip ({_rp.downdip_mean:.2f}) is our unproven-below-LKH volume "
                f"({cs['below_lkh_of_discovery']['mean']:.2f}) *plus* that same slice — "
                f"{cs['proven']['mean'] - _rp.updip_mean:.2f} MMboe here. Both partitions sum "
                f"to the well-associated volume; neither is the well-associated volume.",
            )

        # ------------------------------------------------------ risked, and separate
        st.markdown("##### Expected volumes, in MMboe — risked, mean × chance")
        e = st.columns(3)
        # **prospect_success, not prospect.** The plain `prospect` mean spans every
        # trial including the chance failures, so it is already unconditional and
        # multiplying it by POS_prospect risks it twice -- 7.84 where the answer is
        # 10.31 on the reference data. Invisible on a file with no zero-volume trials,
        # which is why it survived.
        e[0].metric("Expected prospect volume",
                    f"{expected_volume(gs['prospect_success']['mean'], chance.pos_prospect):.2f}",
                    help=f"Success-case prospect mean "
                         f"({gs['prospect_success']['mean']:.2f}) × POS prospect "
                         f"({chance.pos_prospect:.1%}).")
        e[1].metric("Expected well associated",
                    f"{expected_volume(gs['discovery']['mean'], chance.p_well):.2f}",
                    help=f"Discovery mean × P well ({chance.p_well:.1%}).")
        e[2].metric("Expected proven",
                    f"{expected_volume(cs['proven']['mean'], chance.p_well):.2f}",
                    help=f"Proven mean × P well ({chance.p_well:.1%}).")
        figure_note(
            "Each success-case mean above, times the chance of the outcome it belongs "
            "to. An expectation, not a volume anyone finds.",
            detail="A **risked mean**: each success-case mean above, times the chance of the outcome "
            "it belongs to. These are the only volumes here that are *additive across "
            "prospects*, which is why a portfolio uses them — and they describe no outcome that "
            f"can happen: this well either finds something near {gs['discovery']['mean']:.1f} "
            "MMboe or it finds nothing. Quote them beside the chance and the size, never "
            "instead of them.",
        )

    st.divider()
    st.divider()
    sh = groups.risked_shares(chance.pos_prospect, chance.p_well)
    # The knife edge, stated (Lars, 2026-08-12). A discovery is `contact > z_entry`
    # *strictly*, so a contact exactly on the entry is a dry hole -- correct, since
    # that is zero column at the well, but invisible. An invisible tie is where a
    # boundary rule goes wrong quietly: one prospect-B trial sitting exactly on
    # 2205.0 m was enough to invert a band's percentiles in 3.12.
    _ties, _tie_frac = boundary_ties(ts, entry)
    if _ties:
        figure_note(
            f"{_ties:,} success trials ({_tie_frac:.1%}) sit within ±0.5 m of the entry — the boundary is a real population, not a rounding artefact.",
            detail=f"**On the knife edge:** {_ties:,} success trials ({_tie_frac:.1%}) have their "
            f"contact within ±0.5 m of the reservoir entry. A discovery here is *contact "
            f"deeper than the entry*, strictly — a contact exactly on it means zero column "
            f"at the well and counts as dry. Move the entry a metre and these trials change "
            f"sides, which is worth knowing before reading a small difference as a signal.",
        )
    st.markdown(
        f"**Outcome tree**, over {ts.n_trials:,} trials — chance failure "
        f"{sh['chance_failure']:.1%} · dry with attic {sh['dry_with_attic']:.1%} · "
        f"discovery with contact logged {sh['contact_seen']:.1%} · "
        f"discovery with HC to exit {sh['hc_to_exit']:.1%}"
    )

    if not has_area:
        st.warning(fig_ref(
            "No productive-area column in this export — the proven/possible split, {a6} and "
            "the live section need it and are skipped."
        ))
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
            ("Unproven below LKH",
             "What lies **below LKH** — the well's lowest known hydrocarbon, which is the "
             "shallower of the contact and the reservoir exit. The well was still in "
             "hydrocarbons when it left the reservoir, so its **presence is confirmed and only "
             "its extent is not**. That is why it is *unproven* rather than *possible*: "
             "\"possible\" is a PRMS reserves class meaning low confidence of recovery, and "
             "this volume is the opposite — certain to exist, unknown in size. Defined in "
             "tab ⑥."),
            ("Up-dip / attic volume",
             "The accumulation in the trials where hydrocarbons are present but sit **entirely "
             "above** the well: the well is dry, the prospect is not. This is what a dry hole "
             "leaves behind, and the number quoted when somebody argues for a sidetrack."),
            ("Commercial accumulation",
             "The well-associated volume among the discoveries that **clear MEFS**. Its chance "
             "is Rose's `Pc(well)`, so this is the row whose distribution belongs to the number "
             "an EMV takes. It is an *additional* class, not a cut of the ones above — nothing "
             "here truncates the other distributions, because a volume cut-off raises the "
             "unrisked mean while lowering commercial chance and the two do not cancel. Its "
             "mean sitting above the well-associated mean is that effect made visible."),
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
        # The chance the well leaves the reservoir *in* hydrocarbons, given a
        # discovery. P_well times this is the chance of the possible class occurring
        # at all, and it is necessarily below P_well -- which is what Lars predicted
        # and what the old row denied by reusing P_well unchanged.
        _n_disc = int(np.asarray(groups.discovery).sum())
        _n_below = int(np.asarray(groups.hc_to_exit).sum())
        _p_below_exit = (_n_below / _n_disc) if _n_disc else 0.0
        rows = [
            ("Prospect resource potential", res_all[res_all > 0], chance.pos_prospect),
            ("Well associated volume", vc.discovery_total[groups.discovery], chance.p_well),
            ("Resource tested by the well", vc.proven[groups.discovery], chance.p_well),
            # **Conditional on there being anything below the exit**, which is the
            # event this row's name describes -- so its n and its chance are smaller
            # than the discovery group's, and its percentiles are percentiles of the
            # thing it is named after rather than of a population 41 % of which
            # contributes a zero.
            ("Unproven below LKH",
             vc.below_lkh[groups.hc_to_exit],
             chance.p_well * _p_below_exit),
            ("Up-dip / attic volume", vc.attic[groups.dry_with_attic], p_updip),
        ]
        # **The commercial accumulation, in the table it belongs to** (Lars,
        # 2026-08-18). It is the well-associated volume among the discoveries that
        # clear MEFS, and its chance is Rose's `Pc(well)` -- so it is the one row
        # whose distribution goes with the number an EMV calculation takes. Every
        # other row is conditional on an event Pc does not describe.
        #
        # **Not a MEFS cut of the rows above it.** Nothing there is truncated: a
        # volume cut-off raises the unrisked mean while lowering commercial chance
        # (Longley 2026), so applying one would put a reader's economics into
        # everyone's volumes. This is an *additional* class, conditional on a
        # different event, and its mean sitting above the well-associated mean is
        # exactly that effect made visible rather than hidden.
        _res_disc = np.asarray(res_all, dtype=float)
        _commercial = np.asarray(groups.discovery, dtype=bool) & (_res_disc > float(mefs))
        if _commercial.any():
            _p_mcfs = float(_commercial.sum()) / max(_n_disc, 1)
            rows.append(("Commercial accumulation (clears MEFS)",
                         _res_disc[_commercial], chance.p_well * _p_mcfs))
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
        figure_note(
            "Conditional percentiles: each assumes its own case happens. Risking scales "
            "the probability, never the volume.",
            detail="**These percentiles are conditional — they assume the case happens.** P90 is exceeded "
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
            "interchangeable words here.",
        )
        if abs(_counts_chance - chance.p_well) > 5e-4:
            st.info(
                f"The chance column counts trials: {int(stats[1][1]['n']):,} of {ts.n_trials:,} "
                f"are well associated, so **{_counts_chance:.1%}** — the *file's* chance. The app is "
                f"using **P_well = {chance.p_well:.1%}**, because the risking convention in tab ② "
                f"says the chance comes from the chance table rather than from the trials. "
                f"{fig_ref('{c1}')}'s dashed curves and every figure use P_well; this column is "
                f"the raw count."
            )
        st.divider()
        # Two figures, one above the other (Lars, 2026-08-11). They were one stacked
        # composite; split, each renders at its own natural height and either can be
        # exported and placed on its own.
        # C1 is fully labelled again (Lars, 2026-08-11). It ran for a while as a
        # small unlabelled thumbnail on the argument that A1 higher up this tab
        # carried the readable version -- but C1's job here is to be read *with* C2
        # directly below it, and the pair only makes its argument if both halves
        # carry a scale: C2 shows what each volume is worth, C1 has to show where
        # it sits, and "above the well at 2205 m" is not sayable without a depth
        # axis. So it takes a full panel height like everything else.
        _chart(pfig_c1_section(
                ad, ts, z_entry=entry, z_exit=exit_, area_scale=area_scale,
            ), key="c1")
        # **The two readings toggle separately** (Lars, 2026-08-15). Plotly's legend
        # groups by concept, so clicking hid both of a concept's curves; there was no
        # way to see every risked curve on its own. Checkboxes rather than legend
        # state, so the exported figure honours the same choice.
        _r1, _r2, _r3 = st.columns([1, 1, 2])
        _show_cond = _r1.checkbox("Unrisked (conditional)", value=True,
                                  key="w_c2_conditional")
        _show_uncond = _r2.checkbox("Risked (unconditional)", value=True,
                                    key="w_c2_unconditional")
        _chart(pfig_c2_exceedance(
                ts, groups, vc, pos_prospect=chance.pos_prospect,
                p_well=chance.p_well, mefs=mefs, pc_well=_cc.pc_well,
                show_conditional=_show_cond, show_unconditional=_show_uncond,
            ), key="c2", height=C2_HEIGHT)
        figure_note(
            fig_ref("**{c1}** shows where each volume sits in the structure; **{c2}** shows the same volumes as exceedance curves, two per concept in one colour."),
            detail=fig_ref(
                "**{c1} and {c2} — the concepts, twice.** {c1} shows where each volume "
                "sits in the structure; {c2} shows the same volumes as exceedance "
                "curves, **two per concept in one colour**. Read them as a pair: the "
                "first says *where*, the second says *how much and how likely*.\n\n")
            +
            "The **solid** curve is *conditional* — the success case, given that case happens. It "
            "starts at 100 % and it is where the percentiles live: that is what anyone means by "
            "\"the P50\". The **dashed** curve is *unconditional* (risked): the same volumes with "
            "the chance folded in, so it starts at the chance instead. Here the prospect's dashed "
            f"curve begins at {chance.pos_prospect:.0%} and the well-associated one at "
            f"{chance.p_well:.0%} — and **the vertical gap between those two starts is the location "
            "penalty**, the chance the prospect holds something this well would miss.\n\n"
            "**Markers on both curves at P90 / P50 / mean / P10, but values on the solid ones "
            "only** (Lars, 2026-08-12). Risking scales the *probability* and never the volume, so "
            "the number beside a dashed marker was always the number beside its solid twin — text "
            "without information, doubled on the busiest figure here. The dashed markers stay, "
            "because *where* they sit is the point: the same P50 volume at a lower height is the "
            "location penalty, made visible.\n\n"
            "The braces below show the nesting — up-dip inside tested inside well associated "
            "inside prospect — and the axis carries no negative labels: that space is for the "
            "braces, not for probabilities.\n\n"
            f"**The ringed markers on the MEFS line are the eight crossings** — filled on the "
            f"solid curves, open on the dashed. Each exceedance curve *is* a probability "
            f"curve, so the chance of clearing MEFS is where the curve meets the line rather "
            f"than a separate series. They are not labelled here because several land within "
            f"a percentage point of each other and eight labels on one vertical line "
            f"overlap — {fig_ref('{c3}')} gives them an axis of their own, and the table "
            f"under it gives the numbers.",
        )

        # ---------------------------------------------- the eight crossings, drawn
        # 4.2 marks them on its curves and cannot label them -- three of the four
        # conditional crossings land within half a point of each other on the demo
        # data, so eight labels on one vertical line overlap. Here they get an axis
        # of their own, from the same c2_cases definition, so the bars cannot
        # describe curves the figure above them did not draw.
        _chart(pfig_c3_mefs_bars(
            ts, groups, vc, pos_prospect=chance.pos_prospect,
            p_well=chance.p_well, mefs=mefs), key="c3")

        # ------------------------------------------- the eight crossings, tabulated
        # Lars, 2026-08-14: *"can I get a probability curve in 4.2 for exceedance MEFS,
        # risked and unrisked."* Same numbers the figure marks -- both come from
        # core.mefs.c2_cases, so the table cannot describe curves the figure did not
        # draw.
        _cx = c2_crossings(ts, groups, vc, chance.pos_prospect, chance.p_well, mefs)
        st.markdown(f"###### Chance of exceeding MEFS / MCFS, {mefs:,.1f} MMboe")
        st.dataframe(
            pd.DataFrame([{
                "Volume": c.name,
                "Unrisked — conditional": f"{c.conditional:.1%}",
                "× chance of the case": f"{c.chance:.1%}",
                "Risked — unconditional": f"{c.risked:.1%}",
                "Trials": f"{c.n:,}",
            } for c in _cx]),
            hide_index=True, width="stretch",
        )
        figure_note(
            "Unrisked is the solid curve at the MEFS line, risked the dashed one. The "
            "middle column is what separates them.",
            detail="**Unrisked is the solid curve's height at the line, risked is the dashed "
            "one's** — and the middle column is exactly what separates them, because "
            "risking scales the *probability* and never the volume. So the risked "
            "column is the product of the two beside it, not a second pass over "
            "different trials.\n\n"
            "**The up-dip row is risked by its own chance** — dry but charged, "
            f"`POS − P_well` = {max(chance.pos_prospect - chance.p_well, 0.0):.1%}, "
            "not P_well. It is the volume you leave behind if this well is dry, so "
            "the event it is conditional on is the one where the well fails.\n\n"
            "The well-associated unrisked figure is Rose's `Pmcfs(well)` from the "
            "chance block at the top of this tab; its risked twin is `Pc(well)`.",
        )

        st.divider()
        # **Separate figures, stacked** (Lars, 2026-08-12). They were side by side in
        # two columns, which halved both and -- worse -- left the section with no
        # index of its own, so the only way to refer to it was "the one next to A6".
        # It is 4.3 now and A6 is 4.4.
        _chart(pfig_b0_section(ad, z_entry=entry, z_exit=exit_, title="Live section"), key="live")
        st.caption(fig_ref(
            "**{live} — the live section**, drawn from A(z) at the well you have chosen "
            "and shaded in the same three colours the volume classes use everywhere "
            "else. Width is proportional to √(enclosed area) — a circular-closure "
            "proxy, so the *shape* is illustrative and the x-axis claims no unit. The "
            "**depths on y are the real quantity**."
            "\n\n"
            # **Why this is not a second copy of {c1}** (review, 2026-08-18). Both draw
            # the three volume classes on the same depth axis, which is exactly the
            # duplication that removed the schematic section from tab ③ -- but these
            # two differ in the thing that matters: one has a *quantity* on x and the
            # other a shape. Nothing said so, and a reader meeting two sections six
            # figures apart is owed the difference rather than left to find it.
            "**Not a second copy of {c1}.** That one puts real productive area on x "
            "and carries the base-reservoir uncertainty, so it is the one to measure "
            "against; this one is shaped like a section, so it is the one to look at."
        ))

        st.divider()
        a6_norm = st.radio(
            f"{fig_ref('{a6}')} histogram scaling", ["density", "peak"], horizontal=True, key="w_a6_norm",
            format_func=lambda k: {"density": "Density (area = 1 each)",
                                   "peak": "Normalised to each own peak"}[k],
            help=("Density is the honest default -- each class integrates to 1, so the areas "
                  "are comparable. Normalising to the peak instead makes the *shapes* "
                  "comparable when one class is far narrower than another, at the cost of the "
                  "vertical axis no longer meaning anything absolute."),
        )
        a6_curves = st.checkbox(
            f"Overlay the exceedance curves from {fig_ref('{c2}')} (conditional only)", value=False,
            key="w_a6_curves",
            help=("The same four classes as cumulative curves, on a second x-axis in per cent. "
                  "Conditional only: a risked curve beside an unrisked histogram would be two "
                  "readings on one figure, which is the mistake "
                  + fig_ref("{c2}") + " exists to keep apart."),
        )
        # **The fill is a control, not a constant** (Lars, 2026-08-15). This figure is
        # about where the classes overlap, and five series at the old 0.45 hid each
        # other -- but the right value depends on how many are on and how far apart
        # they sit, so there is no one number to tune it to.
        a6_opacity = st.slider(
            "Histogram transparency", min_value=0.05, max_value=1.0,
            value=OVERLAP_OPACITY, step=0.05, key="w_a6_opacity",
            help="Lower is more transparent. Turn it down to see the classes that sit "
                 "behind the others; turn it up to read one of them on its own.",
        )
        _chart(pfig_a6_overlap(vc, groups, ts=ts, mefs=mefs, opacity=a6_opacity,
                               normalise=a6_norm, show_exceedance=a6_curves), key="a6")
        # **The overlap, as a number** (Lars, 2026-08-12). The figure has always shown
        # that it is larger than anyone expects and never said how large. Schneider
        # et al. quote 68 % for their own example (their Figure 16); these are the same
        # family of statements for the trials loaded here.
        _ov = outcome_overlap(vc, groups)
        o = st.columns(3)
        o[0].metric("P(a dry hole beats a discovery)",
                    f"{_ov['p_attic_beats_proven']:.1%}",
                    help="Draw one dry outcome and one discovery independently: this is the "
                         "chance the volume left up-dip is larger than the volume the well "
                         "would have proved.")
        o[1].metric("Proven ≤ the best possible attic",
                    f"{_ov['proven_below_max_attic']:.0%}",
                    help="Schneider's framing of the overlap: the share of discoveries whose "
                         "proven volume is no larger than the largest attic in the set.")
        o[2].metric("Attic ≥ the smallest proven", f"{_ov['attic_above_min_proven']:.0%}",
                    help="The mirror statement, from the attic's side.")
        st.caption(
            f"**Schneider's “surprising overlap”, quantified.** The headline is the first "
            f"number: on these trials there is a **{_ov['p_attic_beats_proven']:.1%}** chance "
            f"that a dry hole leaves behind more than a discovery would have proved. Small, "
            f"and not negligible — and it is the number that turns the shape above into an "
            f"argument. The other two are the same overlap read from each side."
        )
        st.caption(
            f"{fig_ref('{a6}')} — Schneider et al.'s 'surprising overlap' between what a dry hole leaves in the "
            "attic and what a discovery proves. Live section — the closure shape from A(z), "
            "colour-keyed to what the well now standing at entry/exit would prove."
        )
        st.divider()
        st.subheader("Conceptual map view")
        # The apex is derived from A(z), not offered as an input: this figure is
        # conceptual, and a second apex control here could disagree with any other.
        # It lives with the well results rather than on the prospect tab because it
        # draws the *entry contour* and the three areas the well divides the closure
        # into -- all of which need a well (Lars, 2026-08-11).
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
        figure_note(
            "Contours whose areas come from A(z), apex at the centre, and the three "
                "areas the well divides the closure into.",
            detail=f"Concentric contours whose *areas* come from A(z), apex at the centre, deepest "
            f"sampled contact ({ad.deepest:.0f} m) as the outer ring. The shaded area inside the "
            f"entry contour is what a dry hole would leave up-dip. **Every contour is dashed; the "
            f"one solid ring is the well's entry depth**, so line style says only 'is this the "
            f"well?'. Contours shallower than the shallowest sampled contact "
            f"({ad.shallowest:.0f} m) are drawn faint — the trials never reached the crest, so "
            f"their area is a taper to the apex, not a model output. "
            f"**The shape is a cartoon**: circles of the right area, in the wrong outline.",
        )
