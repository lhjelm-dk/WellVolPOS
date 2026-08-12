"""Tab ④ — what this well gets at the depth chosen in the sidebar.

The KPI strip, both engines' summaries, the class table, the live section,
A6's overlap, the C1/C2 concepts pair and the conceptual map view. The map
belongs here rather than on ② because it draws the entry contour and the
three areas a *well* divides the closure into."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from ..core import (
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

    def _split_caveat() -> None:
        split_caveat(ctx)

    st.subheader("At this well")
    _split_caveat()

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
        # **The volume when the contact lands on the well** (Lars, 2026-08-12) --
        # the workbook's Results!G8, and Rose's "No Regrets" volume in probabilistic
        # form. It belongs beside the attic and discovery means because it is the seam
        # between them, and the surprise is how much closer it sits to the attic.
        _atw, _atw_n = at_the_well_volume(ts, entry)
        if _atw_n:
            w = st.columns(2)
            w[0].metric("At the well — contact exactly at entry", f"{_atw:.2f}",
                        help=f"MMboe, mean of the {_atw_n:,} trials whose contact lands "
                             f"within ±2 m of the reservoir entry.")
            _span = gs["discovery"]["mean"] - cs["attic_dry_hole"]["mean"]
            _frac = ((_atw - cs["attic_dry_hole"]["mean"]) / _span) if _span else float("nan")
            w[1].metric("...where that sits between dry and discovery", f"{_frac:.0%}",
                        help="0 % would be the attic mean, 100 % the discovery mean.")
            st.caption(
                f"**The boundary case.** Not a discovery and not a dry hole, but the "
                f"accumulation you get if the hydrocarbon–water contact lands *on* the well: "
                f"**{_atw:.2f} MMboe**, between the attic mean of "
                f"{cs['attic_dry_hole']['mean']:.2f} and the discovery mean of "
                f"{gs['discovery']['mean']:.2f}. This is Rose's *“No Regrets”* volume in "
                f"probabilistic form — his is a single deterministic product, this is the mean "
                f"of the trials that actually landed there, so it carries the model's own "
                f"correlations. He is candid that the deterministic version *“is an "
                f"oversimplification”*, because *“there remains a chance the updip volume will "
                f"exceed MCFS”* — which is what {fig_ref('{b2}')}'s regret curve answers."
            )

        e = st.columns(3)
        e[0].metric("Expected prospect volume",
                    f"{expected_volume(gs['prospect']['mean'], chance.pos_prospect):.2f}")
        e[1].metric("Expected well associated",
                    f"{expected_volume(gs['discovery']['mean'], chance.p_well):.2f}")
        e[2].metric("Expected proven",
                    f"{expected_volume(cs['proven']['mean'], chance.p_well):.2f}")
        st.caption(
            "MMboe. **Expected** volumes are mean × chance — a *risked mean* — "
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
            ("Possible — below the reservoir exit",
             "The rest of the well-associated volume: what lies **below the depth at which the "
             "well left the reservoir**. The well was still in hydrocarbons when it exited, so it "
             "never established how far down they go. Well associated, but **not proven** — that "
             "distinction is the whole reason this class exists instead of being folded into the "
             "discovery case."),
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
        rows = [
            ("Prospect resource potential", res_all[res_all > 0], chance.pos_prospect),
            ("Well associated volume", vc.discovery_total[groups.discovery], chance.p_well),
            ("Resource tested by the well", vc.proven[groups.discovery], chance.p_well),
            ("Possible — below the reservoir exit", vc.possible[groups.discovery], chance.p_well),
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
