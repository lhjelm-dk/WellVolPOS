"""Tab ⑥ — Theory, definitions, references and guidelines.

Not in the design plan, which specifies six tabs and mentions only "README and
docs" in phase 5. Added on Lars's request, and written around ``pfig_concepts``
so the definitions are read next to the picture that shows them.

Everything quantitative here is computed live from the loaded trials rather than
typed in, so the worked examples cannot drift from what the other tabs report.
"""

from __future__ import annotations

import numpy as np
import streamlit as st

from ..ui.numbering import guide_table, ref

from wellvolpos.core import commercial_chance, expected_volume, no_regrets, thickness_from_pay
from wellvolpos.viz import pfig_colour_key


def render(*, ts, ad, groups, vc, chance, mefs, entry, exit_, pos_source):
    """Draw the guide. Live numbers throughout, from the loaded trial set."""
    st.subheader("Theory, definitions and references")
    st.caption(
        "Written to be read beside the concepts figure on the Well location tab. "
        "Every number below is computed from the trials currently loaded, so it agrees "
        "with the rest of the app by construction rather than by proofreading."
    )

    # ------------------------------------------------------------- the one idea
    st.markdown("### The one idea everything rests on")
    st.latex(r"r_{location} = P(\text{contact deeper than the well} \mid \text{HC present})")
    st.latex(r"P_{well} = POS_{prospect} \times r_{location}")
    st.markdown(
        f"""
Two numbers, two meanings, **never multiplied into one**. `r_location` is the only
quantity the well's position controls; `POS_prospect` is the only one it does not.
Right now: **{chance.pos_prospect:.4f} × {chance.r_location:.4f} = {chance.p_well:.4f}**,
with POS taken from {pos_source}.

**Why it matters.** You drill a well, not a prospect. The prospect's POS and its full
resource range describe something no single borehole can capture. Quoting them against a
well proposal overstates both the chance and the volume — which is the argument this whole
tool is built to make.

The source workbook computed `1 − PERCENTRANK(all contacts, entry)`, which *already*
contains the chance failures, and then multiplied by a separately entered POS. That is
right only when the entered POS is 1.0, which it happened to be. A chance table of 0.60
would have produced a well POS about 40 % too low. Reading the workbook showed the same
conflation in **206 cells**, not one.
        """
    )

    # ----------------------------------------------------------- the volumes
    st.divider()
    st.markdown("### The volumes, and the colour each one always has")
    st.plotly_chart(pfig_colour_key(), width="stretch", theme=None, key="colourkey")
    st.info(
        "**They nest**: minimum ⊂ up-dip ⊂ tested ⊂ well associated ⊂ prospect. And **a chance "
        "takes the colour of the volume it is the chance of** — P_well is olive like the "
        "well-associated case, POS_prospect navy — so on an exceedance plot the two POS values "
        "read against the two distributions they risk."
    )

    # ------------------------------------------------ pay vs reservoir thickness
    st.divider()
    st.markdown("### Pay thickness is not reservoir thickness")
    try:
        tfp = thickness_from_pay(ts, ad)
        tsum = tfp.summary()
        recovered = (
            f"On these trials the inversion recovers **P50 {tsum['p50']:.1f} m** "
            f"(P90–P10 {tsum['p90']:.1f}–{tsum['p10']:.1f} m) from "
            f"{tfp.n_resolved:,} realisations."
        )
    except ValueError:
        recovered = "This export lacks the columns needed to invert the wedge."
    st.markdown(
        """
**Reservoir thickness** is a property of the *rock* — top reservoir to base reservoir,
present throughout the closure whether or not anything is charged. **Pay** is a property
of the *accumulation* — top reservoir down to the contact, capped by the base.

With a flat contact and a dipping layer the charged interval is a **wedge**: full thickness
at the crest, thinning down-dip, zero where the top surface meets the contact. So an
area-averaged gross pay is always **less** than the reservoir thickness *and varies with the
contact depth*. It is not a rock property, which is why a base-reservoir surface cannot be
drawn by shifting A(z) down by pay.

It can be recovered, though. The hydrocarbon-bearing gross rock volume is
        """
    )
    st.latex(r"GRV(z_c, T) = \int_{z_c - T}^{z_c} A(z)\, dz")
    st.markdown(
        f"""
which increases monotonically in `T`, so the thickness is **uniquely recoverable per
trial** — no fitting. {recovered} It reproduces GeoX's own thickness column to a mean
difference of 0.01 m at r = 0.9998, which validates the geometry and settles that the
column is a genuine gross rock thickness. Thickness is always **true vertical**, so the
base is `top + T` with no dip correction.
        """
    )

    # --------------------------------------------------------- the thresholds
    st.divider()
    st.markdown("### Two different minimum volumes")
    st.warning(
        "**MCFS / MEFS** — minimum commercial (Rose) or economic (this app's sidebar) field "
        "size. The same threshold under two names: the smallest discovery worth **developing**."
        "\n\n**The assessment minimum** — a minimum column height below the apex. The smallest "
        "accumulation worth **carrying in the assessment at all**. A different quantity, and "
        "the one Lars's reference figure labels *Min. volume*."
    )
    st.markdown(
        """
The app currently gives both the same colour, which is a known simplification.

**MEFS is drawn as a reference line and never applied to the distributions.** That is a
decision, not an oversight: per Longley (2026), a MEFS cut *raises* the unrisked mean while
*lowering* commercial chance, and the two do not cancel. Truncating the distributions would
bake one reader's economics into everyone's volumes. Read probabilities against the line
instead.

The minimum column height is likewise a **mapping, not a filter** — it reports the
equivalent contact depth, area and volume percentile and excludes nothing. Filtering would
first need a decision on whether a sub-minimum trial becomes a chance failure (lowering POS)
or simply leaves the population (renormalising it). Those give different answers, so neither
is assumed.
        """
    )

    # ------------------------------------------------------- Rose's quantities
    st.divider()
    st.markdown("### Rose's named quantities, at this location")
    try:
        nr = no_regrets(ts, ad, entry)
        st.markdown(nr.message())
        st.caption(
            "The poster calls No Regrets *“useful to the decision-maker, however, it is an "
            "oversimplification”*, because *“for most downdip well locations, there remains a "
            "chance the updip volume will exceed MCFS”*. That chance is exactly what "
            + ref("{b2}") + " draws "
            "on the Location sweep tab — so this tool shows Rose's number **and** the thing he "
            "says it understates."
        )
    except (ValueError, KeyError):
        st.info("This export lacks the pay or volume columns needed for the No Regrets volume.")

    if mefs:
        cc = commercial_chance(ts, groups, vc.proven, chance.p_well, mefs)
        st.markdown(cc.message())
        st.caption(
            "Rose's `Pmcfs(well)` conditions on the **whole** well-associated volume; our "
            + ref("{b2}") + " "
            "curve conditions on the **proven** volume between entry and exit. Both are "
            "legitimate and they are different numbers, so both are shown rather than one being "
            "quoted as the other. `Pc(well)` is the chance the poster says to use for EMV — a "
            "chance, not an economic value, so it is in scope even though economics is not."
        )

    # ---------------------------------------------------- expected vs success
    st.divider()
    st.markdown("### Expected volume is not a volume anyone finds")
    from wellvolpos.core import group_summary

    gs = group_summary(ts, groups)
    ev = expected_volume(gs["discovery"]["mean"], chance.p_well)
    st.markdown(
        f"""
The workbook's *&ldquo;Risked&rdquo; Pmean* column, and this app's **Expected** metrics, are
mean × chance: **{gs['discovery']['mean']:.2f} × {chance.p_well:.1%} = {ev:.2f} MMboe**.

That is the only volume figure here that is **additive across prospects** — two success-case
means cannot be added, because each is conditional on its own outcome. It is also a number
that describes no outcome that can occur: this well either finds something near
{gs['discovery']['mean']:.1f} MMboe or it finds nothing. Quote it beside the chance and the
size, never instead of them.
        """
    )

    # --------------------------------------------------------- how to read it
    st.divider()
    st.markdown("### How to read each figure")
    st.caption(
        "Figures are numbered **by tab**: 2.1 is the first figure on tab ②, 3.10 the tenth "
        "on tab ③. The old letter codes are kept in brackets because the code, the exported "
        "file names and this project's notes still argue about \"B6\" by letter — the letter "
        "is the figure's identity, the number is where it sits today."
    )
    st.markdown(guide_table())


    # ------------------------------------------- Haskett, and what 3.3 measures here
    st.divider()
    st.markdown("### Uncertainty reduction (3.3) — and what it is measuring *here*")
    _res = np.asarray(ts.col("resource"), dtype=float)
    _disc = groups.discovery
    _p = float(_disc.mean())
    _hi, _lo = _res[_disc], _res[~_disc]
    _parent_var = float(_res.var())
    _within = _p * float(_hi.var()) + (1.0 - _p) * float(_lo.var())
    _between = _parent_var - _within
    _spread = lambda v: float(np.percentile(v, 90) - np.percentile(v, 10)) if v.size else 0.0
    _attic = vc.attic[groups.dry_with_attic] if vc is not None else np.array([])
    _proven = vc.proven[_disc] if vc is not None else np.array([])
    st.markdown(
        f"""
Haskett (2003) is about **appraisal** wells: you have a discovery, and you are choosing
where to drill next so that the *next* well tells you as much as possible about a field you
already know exists. That is not our situation. This app places an **exploration** well, and
3.3 is not asking how to appraise a discovery.

**What transfers is Haskett's mechanism, not his setting.** He separates two kinds of
learning, and names the one we use:

> *“Discrete learning results from the assessment of mutually exclusive, distinct events or
> discovery possibilities. Reservoir extent is the prime example. The productive reservoir
> section either is, or is not present at a particular appraisal location.”*

An exploration well at a chosen reservoir entry depth is **exactly** a discrete learning
event, and the cleanest possible one: the hydrocarbon–water contact is either deeper than
the entry or it is not. There is no partial outcome and no sampling — one bit of
information, and the whole trial population splits on it.

**So what 3.3 measures is the collapse between the two futures this well creates**, not
uncertainty about a known accumulation:

| | the distribution you are left with | mean | P10–P90 spread |
|---|---|---|---|
| **before the well** | the whole prospect, un-cut | {float(_res.mean()):.2f} | {_spread(_res):.2f} |
| **if it finds the column** | the resource this well would have **proven** | {float(_proven.mean()) if _proven.size else float("nan"):.2f} | {_spread(_proven):.2f} |
| **if it is dry but charged** | the resource left **up-dip**, in the attic | {float(_attic.mean()) if _attic.size else float("nan"):.2f} | {_spread(_attic):.2f} |

Each outcome is narrower than the prospect it came from, and *that narrowing, weighted by
how likely each outcome is*, is what the curve plots. The location that maximises it is the
one whose result you can least predict in advance — which is exactly the location that
teaches you most, whichever way it goes. A well certain to succeed and a well certain to
fail both score zero: neither tells you anything you did not already know.

**Read it as information, never as value.** Haskett's full method multiplies uncertainty
reduction by what the information is worth and subtracts what it costs. 3.3 stops at the
information: it has no dry-hole cost, no development case and no discount rate in it, so a
peak on this curve is *not* an economic optimum. For the money side of the same trade see
3.8, and for the volume side 3.7.

##### A correction worth making to the paper

Haskett writes that *“the risked variance of the two child distributions will sum to equal
the variance of the parent. Both the mean and the variance are additive.”* The **mean** is —
that is the law of total expectation, and on these trials it holds to floating point. The
**variance is not.** The law of total variance has two terms:
"""
    )
    st.latex(r"\operatorname{Var}(X) = \underbrace{E[\operatorname{Var}(X \mid G)]}_{\text{within outcomes}}"
             r" + \underbrace{\operatorname{Var}(E[X \mid G])}_{\text{between outcomes}}")
    st.markdown(
        f"""
Probability-weighting the two child variances gives only the **first** term. On the trials
loaded now that is {_within:,.0f} against a parent variance of {_parent_var:,.0f} — so
**{100.0 * _between / _parent_var:.0f} % of the variance is unaccounted for** by the
additive rule.

That missing term is not an error in the method; it *is* the answer. `Var(E[X | G])` — how
far apart the two outcome means sit — is precisely the uncertainty the well removes, and it
is what 3.3 is built from. So the paper's arithmetic claim is loose while its conclusion
stands, and it is worth knowing which is which before quoting either.
"""
    )

    # -------------------------------------------------------------- guidelines
    st.divider()
    st.markdown("### Guidelines")
    st.markdown(
        """
1. **Set the risking convention first.** Everything downstream depends on whether the trials
   already carry the geological risk. It is stamped in the footer on every tab so it is never
   implicit.
2. **Quote `P_well` and the well-associated volume** for a well decision; show the prospect
   figures as the contrast, not the headline.
3. **Say which "proven" you mean.** The workbook's *PROVEN mean at well* is 14.78 MMboe —
   the mean total resource of trials whose contact falls between entry and exit. This app's
   proven mean is 16.04 MMboe — the per-trial **wedge** split across all discovery trials
   (15.76 before the apportionment moved from map area to the wedge). Both are here; the word
   collides.
4. **Distrust the deep end of any swept curve.** Conditional groups thin down-dip; steps
   resting on fewer than 30 trials are left undrawn rather than shown as firmly as the rest.
5. **Treat the apex as an extrapolation.** It comes from A(z)'s shallow tail, because the
   trials do not contain the crest — and the `crest` column cannot supply it (60 % of success
   trials there have a "crest" deeper than their own contact, which is impossible).
6. **Never join a GeoX export on `TrialNumber`.** The identifiers can sit on different rows
   than their own data.
        """
    )

    # -------------------------------------------------------------- references
    st.divider()
    st.markdown("### References")
    st.caption(
        "Every entry below was checked against the document itself — the PDFs are in "
        "`Papers/`, and the two online sources were opened and read. Where a work is cited "
        "*through* another rather than read directly, it says so."
    )
    st.markdown(
        """
**Read in full — the PDFs are in `Papers/`**

| Work | What it contributes here |
|---|---|
| **Schneider, M., Citron, G.P., Haryott, P. & Cook, D. (2023)** *Drilling an exploration prospect downdip: quantifying the trade-offs between chance of success and associated resource potential.* AAPG Bulletin **107**(5) 743–759. [doi:10.1306/09232222051](https://doi.org/10.1306/09232222051) · [open access](https://pubs.geoscienceworld.org/aapg/aapgbull/article/107/5/743/622239/Drilling-an-exploration-prospect-downdip) | The definitive reference. Whole-trial up-dip/down-dip grouping — this app's **reference engine**. Names the finer proven/possible split as *“additional complexity”* without computing it; `core/classes.py` is that complexity, implemented. Also the source of the convention that the EUR distribution is the **success case** and is determined *before* the chance, and that Pg is the chance of exceeding the **P99** EUR. |
| **Milkov, A.V. (2021)** *Reporting the expected exploration outcome: when, why and how the probability of geological success and success-case volumes for the well differ from those for the prospect.* J. Pet. Sci. Eng. **204** 108754. [doi:10.1016/j.petrol.2021.108754](https://doi.org/10.1016/j.petrol.2021.108754) | The crest/apex reference convention, and the peer-reviewed statement that well POS is prospect POS times the contact distribution. His segment A2 falls from 0.34 at the crest to 0.07 down-dip. Also the definition this app uses for percentiles: *“P90 is defined as 90 % probability of exceeding the P90 estimated value.”* |
| **Schneider, M. & Cook, D.M. Jr. (2017)** *Drilling a Downdip Location: Effect on Updip and Downdip Resource Estimates and Commercial Chance.* AAPG Search & Discovery **#42102**, posted 3 July 2017 — with the Rose & Associates long-form version, Houston, May 2021. [search & discovery](https://www.searchanddiscovery.com/documents/2017/42102schneider/ndx_schneider.pdf) | Equation 1 (`Pwell = Pg × Ptrap@well / Ptrap`, cap included), the P90-area reference contour, the *“No Regrets”* volume, `Pmcfs(well)` and `Pc(well)`. Both documents are in `Papers/`; the 2021 file is a Rose & Associates client report authored by Schneider and Cook, not by Rose. |
| **Haskett, W.J. (2003)** *Optimal Appraisal Well Location Through Efficient Uncertainty Reduction and Value of Information Techniques.* SPE **84241**, SPE ATCE, Denver, 5–8 October 2003. [doi:10.2118/84241-MS](https://doi.org/10.2118/84241-MS) | Appraisal placement as value of information — 3.3's uncertainty-reduction curve, with the optimum found by argmax rather than by eye. |
| **Singh, V., Yemez, I., Izaguirre, E. & Racero, A. (2017)** *Optimal Subsurface Appraisal: A Key Link to the Success of Development Projects.* Am. J. Applied Sciences **14**(2) 217–230. [doi:10.3844/ajassp.2017.217.230](https://doi.org/10.3844/ajassp.2017.217.230) · open access | Appraisal-value framing around the same trade-off: what a well is worth is what it resolves, not what it finds. |
| **Hood, K.C. (2019)** *Column Height.* Risk Coordinators' Workshop, 14 November 2019 (ExxonMobil Upstream Integrated Solutions) — and **Hood, K.C. (2024)** *Hydrocarbon Column Heights, Parts 1 & 2*, Rose & Associates blog. Both in `Papers/HCWC/` | The assessment minimum belongs to a minimum **column height**, linked to seal capacity, not to a minimum volume — decision 6, and why this app maps a column height rather than filtering on a volume. Two separate documents; the 2019 workshop deck and the 2024 blog pair are often conflated. |

**Consulted online, and re-read to check these claims**

| Source | What it contributes |
|---|---|
| [**Longley, I. (27 January 2026)** *Understanding the “Minimum Economic Field Size” concept and aggregating targets*, GeoExpro](https://geoexpro.com/understanding-the-minimum-economic-field-size-concept-and-aggregating-targets/) | *“Applying a volume cut-off will always lead to the Mean Unrisked volume to increase in comparison to the non-truncated case”* while *“the chance of ‘commercial’ success (Pc) comes down.”* The two do not cancel — which is exactly why this app draws MEFS as a **reference line to read probabilities against** and never truncates a distribution with it. |
| [**Rose & Associates** — Pwell implementation, 2017–2021](https://www.roseassoc.com/pwell-implementation-from-2017-to-2021/) | Vendor documentation for the MMRA (2018) → RoseRA (2021) lineage. Not peer-reviewed, and cited only for that history. |

**Cited through another work, not read here**

| Work | Route |
|---|---|
| **Rose, P.R. (2001)** *Risk Analysis and Management of Petroleum Exploration Ventures.* AAPG Methods in Exploration **12** | Reached via Schneider et al. (2023), who cite it for Pg being the chance of a discovery *“equal to or exceeding the P99 EUR”* — the anchor this app's unconditional curves are checked against. The book itself is not in `Papers/`. |
| **Capen (1976)**; **Otis & Schneidermann (1997)** | Cited by Schneider et al. (2023) as the established basis for unbiased assessment of the success-case distribution and its chance. Listed so the lineage is visible, not because they were consulted. |

**Deliberately not obtained.** Milkov & Samis (2020, AAPG Bulletin 104) and Samis &
Milkov (2020), on the real-option value of untested up-dip volume after a dry hole.
Both paywalled. If the attic / regret analysis becomes central they are the next two
to get — and {b2}'s regret curve is the place they would land.

---

**Two provenance notes.** Prospect A's demo data is fictional and safe to publish.
Prospect B is extracted from the 2018 macro workbook and its publication status has
**not** been confirmed — treat it as the licensee's until it has. The source workbook
remains the specification either way: `tests/test_excel_parity.py` locks fifteen of its
values and was written before any other code.
        """
    )
