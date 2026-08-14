"""Tab ④ — what this well gets at the depth chosen in the sidebar.

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
    pfig_a6_overlap,
    pfig_b0_section,
    pfig_c1_section,
    pfig_c2_exceedance,
    pfig_map_view,
)
from ..core.rose import AT_WELL_WINDOW_M
from ..viz.theme import reference_label
from .common import C2_HEIGHT, chart as _chart, split_caveat
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
    wells, selected_well = ctx.wells, ctx.selected_well

    def _split_caveat() -> None:
        split_caveat(ctx)

    # --------------------------------------------------------- which well is this?
    # Exactly one candidate is carried onto this tab. It answers "what do I get at
    # the depth I chose", which is a question about one well -- letting four through
    # would turn every figure here into a comparison and lose what the tab is for.
    # The comparison itself is the table below.
    st.subheader("The selected well")
    _sel_l, _sel_r = st.columns([1, 2])
    with _sel_l:
        if len(wells) > 1:
            st.selectbox(
                "Well carried onto this tab", [w.label for w in wells],
                key="w_selected_well",
                format_func=lambda k: f"Well {k}",
                help="Candidates are defined and compared on tab ③. **Everything on "
                     "this tab is about this one well** — no other candidate appears "
                     "here, by design.",
            )
        else:
            st.metric("Well", f"Well {selected_well}")
    with _sel_r:
        st.metric(f"Well {selected_well} — reservoir entry to exit",
                  f"{entry:,.0f} – {exit_:,.0f} m TVDSS",
                  help=f"{exit_ - entry:,.0f} m of reservoir penetrated.")
    st.caption(
        "**This tab is one well.** Tab ③ is the bench — define candidates, sweep them, "
        "compare them; this is the write-up of the one you chose. Every number, table "
        f"and figure below is Well {selected_well}'s."
    )

    st.divider()
    st.subheader(f"At Well {selected_well} — {entry:,.0f}–{exit_:,.0f} m")
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
    ch[2].metric("= P well", f"{chance.p_well:.1%}",
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

    if has_area:
        cs = class_summary(vc, groups)
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
            help="How close a trial's contact must be to the reservoir entry to count "
                 "as landing *on* the well. Widen it for more trials and a blurrier "
                 "answer; narrow it until the count gets too small to mean anything. "
                 "**It also sets the at-the-well curve on 3.5**, which used to keep the "
                 "2 m default no matter what was typed here.",
        )
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
            st.caption(
                f"**The boundary case.** The accumulation you get if the hydrocarbon–water "
                f"contact lands *on* the well: **{_atw:.2f} MMboe**, which sits **{_frac:.0%}** "
                f"of the way from the attic mean ({cs['attic_dry_hole']['mean']:.2f}) to the "
                f"discovery mean ({gs['discovery']['mean']:.2f}) — closer to the dry case than "
                f"most people expect. This is Rose's *“No Regrets”* volume in probabilistic "
                f"form: his is a single deterministic product, this is the mean of the trials "
                f"that actually landed there, so it carries the model's own correlations. He is "
                f"candid that the deterministic version *“is an oversimplification”*, because "
                f"*“there remains a chance the updip volume will exceed MCFS”* — which is what "
                f"{fig_ref('{b2}')}'s regret curve answers."
            )

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
            st.caption(
                f"**Two cuts of one closure.** Rose splits at the well; this app splits at "
                f"the interval the well actually penetrates, because a well proves what it "
                f"drills through. So his updip ({_rp.updip_mean:.2f}) is our proven "
                f"({cs['proven']['mean']:.2f}) *minus* the entry-to-exit slice, and his "
                f"downdip ({_rp.downdip_mean:.2f}) is our unproven-below-LKH volume "
                f"({cs['below_lkh_of_discovery']['mean']:.2f}) *plus* that same slice — "
                f"{cs['proven']['mean'] - _rp.updip_mean:.2f} MMboe here. Both partitions sum "
                f"to the well-associated volume; neither is the well-associated volume."
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
        st.caption(
            "A **risked mean**: each success-case mean above, times the chance of the outcome "
            "it belongs to. These are the only volumes here that are *additive across "
            "prospects*, which is why a portfolio uses them — and they describe no outcome that "
            f"can happen: this well either finds something near {gs['discovery']['mean']:.1f} "
            "MMboe or it finds nothing. Quote them beside the chance and the size, never "
            "instead of them."
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
        st.caption(
            f"**On the knife edge:** {_ties:,} success trials ({_tie_frac:.1%}) have their "
            f"contact within ±0.5 m of the reservoir entry. A discovery here is *contact "
            f"deeper than the entry*, strictly — a contact exactly on it means zero column "
            f"at the well and counts as dry. Move the entry a metre and these trials change "
            f"sides, which is worth knowing before reading a small difference as a signal."
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
        _chart(pfig_c2_exceedance(
                ts, groups, vc, pos_prospect=chance.pos_prospect,
                p_well=chance.p_well, mefs=mefs,
            ), key="c2", height=C2_HEIGHT)
        st.caption(
            "**4.1 and 4.2 — the concepts, twice.** 4.1 shows where each volume sits in the "
            "structure; 4.2 shows the same volumes as exceedance curves, **two per concept in "
            "one colour**. Read them as a pair: the first says *where*, the second says *how "
            "much and how likely*.\n\n"
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
            "braces, not for probabilities."
        )

        st.divider()
        # **Separate figures, stacked** (Lars, 2026-08-12). They were side by side in
        # two columns, which halved both and -- worse -- left the section with no
        # index of its own, so the only way to refer to it was "the one next to A6".
        # It is 4.3 now and A6 is 4.4.
        _chart(pfig_b0_section(ad, z_entry=entry, z_exit=exit_, title="Live section"), key="live")
        st.caption(
            "**4.3 — the live section**, drawn from A(z) at the well you have chosen, and shaded "
            "in the same three colours the volume classes use everywhere else. Width is "
            "proportional to √(enclosed area) — a circular-closure proxy, so the *shape* is "
            "illustrative and the x-axis claims no unit. The **depths on y are the real "
            "quantity**."
        )

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
                  "readings on one figure, which is the mistake 4.2 exists to keep apart."),
        )
        _chart(pfig_a6_overlap(vc, groups, ts=ts, mefs=mefs,
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
