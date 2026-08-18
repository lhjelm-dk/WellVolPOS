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

from ..core.chance import ELEMENTS
from ..ui.common import element_chip
from ..ui.numbering import guide_table, ref

from wellvolpos.core import commercial_chance, expected_volume, no_regrets, thickness_from_pay
from wellvolpos.viz import pfig_colour_key


def render(*, ts, ad, groups, vc, chance, mefs, entry, exit_, pos_source):
    """Draw the guide. Live numbers throughout, from the loaded trial set."""
    st.subheader("Theory, definitions and references")
    st.caption(
        "Every number below is computed from the trials currently loaded, so it agrees "
        "with the rest of the app by construction rather than by proofreading."
    )

    # ------------------------------------------------ what this tool assumes
    # **First, and assuming competence** (Lars, 2026-08-15, design review). The tab
    # used to open by defining percentiles and exceedance to a reader who has known
    # both since university, while the things specific to *this* tool -- the ones that
    # would change how they read a number -- were spread through long passages further
    # down. Inverted: the local assumptions lead, each with its number attached, and
    # the vocabulary that half the industry genuinely disputes stays. The rest went.
    st.markdown("### What this tool assumes")
    st.caption(
        "Six things worth knowing before quoting anything out of this app. None is "
        "hidden elsewhere in the tab; this is where they live."
    )
    _apex = float(ad.apex_estimate()) if ad is not None else float("nan")
    _shallow = float(ts.col("contact")[ts.col("resource") > 0].min())
    st.markdown(
        f"""
1. **The split apportions on the wedge, not by map area.** The charged interval stands at
   full reservoir thickness up-dip and pinches to zero at the contact, so volume sits
   further up-dip than a per-area rule allows. The old `A(lkh)/A(contact)` rule understated
   proven and overstated what is left below by about six points of the accumulation. The
   figure two sections down is the geometry; `apportionment="area"` restores the old rule.

2. **The apex is extrapolated, never mapped.** A(z)'s shallow tail is run out to zero area:
   **{_apex:,.0f} m** here, against a shallowest *sampled* contact of **{_shallow:,.0f} m**.
   The trials do not contain the crest, so every column-height statement inherits that
   error. The export's `crest` column cannot rescue it — 60 % of success trials there carry
   a "crest" deeper than their own contact.

3. **LKH here is modelled, not logged.** It is `min(contact, exit)` per trial — the same
   concept as a wellsite lowest-known-hydrocarbon, but an output of the model rather than an
   observation. Everything called *unproven below LKH* is measured from it.

4. **The exit moves exactly two volume classes.** A discovery is `contact > entry`, so every
   population — and therefore `r_location` and `P_well` — is fixed by the entry alone.
   Moving the exit shifts the boundary between proven and unproven below LKH and nothing
   else. If you move it and the chance does not budge, that is the model working.

5. **MEFS is a line to read against and is never applied to the distributions.** A volume
   cut-off raises the unrisked mean while lowering commercial chance, and the two do not
   cancel (Longley 2026), so truncating would bake one reader's economics into everyone's
   volumes. The one place a threshold does touch a distribution is the *commercial* class,
   which is an additional class conditional on clearing MEFS — not a cut of the others.

6. **Yield and net-to-gross are uniform inside the charged interval.** The wedge fixes the
   pay geometry and not this. It is checked on import — area against net pay — and the
   check warns rather than blocks, because it disqualifies the extension and leaves the
   reference engine untouched.
        """
    )

    st.divider()

    # -------------------------------------------------------------- guidelines
    # First on the tab (Lars, 2026-08-12). These and the figure table were at the
    # bottom, behind five sections of theory, so the two most operational parts of
    # the tab were the last things a reader found.
    st.markdown("### Guidelines — six things to get right")
    st.markdown(
        """
1. **Set the risking convention first.** Everything downstream depends on whether the trials
   already carry the geological risk. It is stamped in the footer on every tab so it is never
   implicit.
2. **Quote `P_well` and the well-associated volume** for a well decision; show the prospect
   figures as the contrast, not the headline.
3. **Say which "proven" you mean.** The word carries two different numbers and both are
   in this app. One is the mean *total* resource of the trials whose contact happens to fall
   between entry and exit — 14.78 MMboe on prospect A, reported as *tested by the well*. The
   other is the per-trial **wedge** split, averaged over every discovery trial — 16.04 MMboe,
   the headline proven mean. Neither is wrong; quoting one under the other's name is.
4. **Distrust the deep end of any swept curve.** Conditional groups thin down-dip; steps
   resting on fewer than 30 trials are left undrawn rather than shown as firmly as the rest.
5. **Treat the apex as an extrapolation.** It comes from A(z)'s shallow tail, because the
   trials do not contain the crest — and the `crest` column cannot supply it (60 % of success
   trials there have a "crest" deeper than their own contact, which is impossible).
6. **Never join a GeoX export on `TrialNumber`.** The identifiers can sit on different rows
   than their own data.
        """
    )


    # ------------------------------------------------------- scope and disclaimer
    st.divider()
    st.markdown("### Scope, assumptions and disclaimer")
    st.markdown(
        """
**What this tool does.** It re-cuts one prospect segment's Monte Carlo trial export
against one proposed well trajectory, and answers five questions: the chance *this
well* finds hydrocarbons as distinct from the chance the prospect contains them; what
a discovery would have proven and what stays unproven below it; how much sits up-dip
if the well is dry; where the well must go to prove a given volume, and what that
costs in chance; and which risk elements carry the location penalty.

**What it does not do.** It does not replace the volumetric model — it re-cuts that
model's own output, and it cannot be better than the input. It does not build the
hydrocarbon–water contact distribution, which is the single most important input to
every location result here. It does not do economics: MEFS appears as a reference
line to read probabilities against and is never applied to a distribution.
        """
    )
    st.warning(
        "**Single hydrocarbon–water contact only.** A prospect with both a gas–oil and an "
        "oil–water contact, where a well may test one and not the other, is **not "
        "represented**. Neither is vertical (depth-dependent) risk: it is assumed to be "
        "carried already by the contact distribution, because `r_location` is derived from "
        "that distribution and modelling it again here would count it twice."
    )
    st.markdown(
        """
**Assumptions that are load-bearing, stated once.**

* **Uniform yield and net-to-gross inside the charged interval.** The wedge geometry
  fixes the *shape* of the charged interval but not the properties within it. The
  proven/possible split rests on this; the reference grouping does not.
* **The apex is an extrapolation** of the area–depth curve's shallow tail to zero
  area. The trials do not contain the crest, so minimum column height inherits that
  error.
* **The area–depth curve is recovered from the trials**, not mapped. It fits them
  extremely closely, but it is a fit.
* **Percentiles are exceedance throughout** — a P90 is a *small* volume, exceeded 90 %
  of the time. Half the industry writes it the other way round.

Where an assumption is not met on the data actually loaded, the QC report on tab ①
says so, and a caveat is repeated on every tab whose numbers depend on it.
        """
    )
    st.error(
        "**No warranty.** This is a decision-support tool, not an authority. Every number it "
        "produces is a consequence of the trial file, the well depths and the conventions you "
        "chose, and any of those can be wrong. Check results against your own model before "
        "they inform a commitment. The author accepts no responsibility or liability for any "
        "decision, loss or damage arising from its use, or from any error in its calculations "
        "or in the data supplied to it."
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

**The mistake this separation exists to prevent.** It is tempting to read
`P(contact deeper than the entry)` off *all* the trials and multiply that by an entered
POS. But the trials already contain the chance failures — GeoX writes them in as
zero-volume rows — so that percentile *is already risked*, and multiplying it again
counts the geological chance twice. The error hides completely while the entered POS is
1.0 and appears the moment it is not: at a chance table of 0.60 the well POS comes out
about 40 % too low. `r_location` is therefore measured over the **success cases only**,
and the chance is applied once, separately.
        """
    )

    # ----------------------------------------------------------- the volumes
    st.divider()
    st.markdown("### Colour associations")
    st.caption(
        "Colour is assigned by **what a thing is**, never cycled — so the same concept is the "
        "same colour on every figure and in every export, and a colour can be learned once."
    )
    st.markdown("**The volumes**")
    st.plotly_chart(pfig_colour_key(), width="stretch", theme=None, key="colourkey")
    st.info(
        "**They nest**: minimum ⊂ up-dip ⊂ tested ⊂ well associated ⊂ prospect. And **a chance "
        "takes the colour of the volume it is the chance of** — P_well is olive like the "
        "well-associated case, POS_prospect navy — so on an exceedance plot the two POS values "
        "read against the two distributions they risk."
    )
    st.markdown("**The risk elements** — a separate family, used on the chance table and on "
                "5.1 and 5.2")
    _ec = st.columns(len(ELEMENTS))
    for _i, _el in enumerate(ELEMENTS):
        _ec[_i].markdown(element_chip(_el), unsafe_allow_html=True)
    st.caption(
        "A different question from the volumes, so a different family: these say which *chance "
        "element* a bar belongs to. The chips are pale because a name is written on them; the "
        "lines and bars use the saturated versions of the same four hues, which is what keeps "
        "them apart under colour-vision deficiency."
    )
    st.markdown("**Line style carries the reading, not the quantity**")
    st.markdown(
        """
| style | meaning |
|---|---|
| **solid** | conditional — the success case, starting at 100 %. Percentiles live here. |
| **dashed** | unconditional (risked) — the same volumes, starting at the chance. |
| **dotted** | a second *volume concept* in the same family, as on 3.12 where dotted is what the well proves. Never risking — that is what dashed means. |
| **thin grey** | a spread read off the trials, drawn behind the answer it qualifies. |
| **shaded band** | sampling error on an estimate, which is a different kind of thing from a geological spread. Both appear on 3.11 and the legend says which is which. |
        """
    )

    # --------------------------------------------------------- how to read it
    st.divider()
    st.markdown("### How to read each figure")
    st.caption(
        "Figures are numbered **by tab**: 2.1 is the first figure on tab ②, 3.10 the tenth "
        "on tab ③, so a number tells you where to find it. The exported figure files keep a "
        "short letter code in their names instead, because a file dropped into a deck outlives "
        "the tab it came from."
    )
    st.markdown(guide_table())


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
        """
    )
    # **The wedge, drawn** (Lars, 2026-08-15, design review). This geometry is what
    # core/reservoir.py inverts and what the whole proven / unproven split rests on,
    # and until now the reader was asked to accept it in prose. One home, here: the
    # figure is a schematic, and tab ④'s split caption points at it rather than
    # carrying a second copy that could drift.
    if ad is not None:
        # `thickness_from_pay` is imported at module level. Re-importing it here
        # would make it a *local* name for the whole function and break the earlier
        # use above -- which is exactly what happened first time.
        import numpy as _np

        from ..ui.common import chart as _chart
        from ..viz import pfig_c4_wedge

        _t = thickness_from_pay(ts, ad).thickness
        _t = _t[_np.isfinite(_t) & (_t > 0)]
        _succ = ts.col("contact")[ts.col("resource") > 0]
        if _t.size and _succ.size:
            # **The area-averaged pay comes from the trials, not from the
            # schematic** (Lars, 2026-08-18). GeoX's gross-pay column *is* the
            # area-averaged charged thickness -- that is what the wedge inversion
            # validated against, to 0.01 m -- so the figure's headline number is a
            # measurement. Averaging the schematic instead would be a *distance*
            # average over an arbitrary horizontal extent, which is fine for the
            # shape and wrong for the number.
            _pay = _np.asarray(ts.col("gross_pay"), dtype=float) if ts.has("gross_pay") else None
            if _pay is not None:
                _pay = _pay[(_np.asarray(ts.col("resource"), dtype=float) > 0)
                            & _np.isfinite(_pay) & (_pay > 0)]
            _chart(
                pfig_c4_wedge(
                    thickness=float(_np.percentile(_t, 50)),
                    z_contact=float(_np.median(_succ)),
                    z_entry=entry, z_exit=exit_,
                    apex=float(ad.apex_estimate()),
                    mean_pay=(float(_pay.mean()) if _pay is not None and _pay.size
                              else None),
                ),
                key="c4",
            )
            st.caption(
                "**Schematic — the dip and the width are drawn for legibility.** What "
                "is real is the relationship, and the two numbers taken from these "
                "trials: the recovered thickness and a median contact. The gap "
                "between the two bars on the right is the whole point — pay averaged "
                "over the charged area is less than the thickness, because the taper "
                "is part of the average."
            )
    st.markdown(
        """
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
    st.divider()
    st.markdown("### “Unproven below LKH” — and why not “possible”")
    st.markdown(
        """
**LKH** is the **lowest known hydrocarbon**: the deepest point at which a well has
actually seen them. In this tool it is computed per trial as the shallower of the
hydrocarbon–water contact and the reservoir **exit** — so if the contact is inside the
penetrated interval the well reached it and LKH *is* the contact, and if the well left
the reservoir still in hydrocarbons then LKH is the exit and the accumulation carries
on below.

**Unproven below LKH** is that continuation: the part of the well-associated volume
that lies under the deepest point the well confirmed. It exists only in the second
case, which is why it is reported *conditional on the well leaving the reservoir in
hydrocarbons* — averaging it over every discovery mixes in the cases that have none,
and those contribute an exact zero.
        """
    )
    st.info(
        "**Its presence is confirmed; only its extent is not.** If the well is in "
        "hydrocarbons at the exit, there *is* more below — Milkov (2021) puts it as a "
        "successful downdip well *“confirming the presence of a large petroleum "
        "accumulation extending”* further. What nobody knows is how far down it goes, "
        "because the well never reached the contact."
    )
    st.markdown(
        """
**Why the name changed** (2026-08-14). It used to be called *“possible — below the
reservoir exit”*, and **“possible” is wrong in a specific and damaging way**: it is a
PRMS reserves class (Proved / Probable / **Possible**, 1P / 2P / 3P) meaning *low
confidence of commercial recovery* — the far tail. This volume is the opposite kind of
thing. A reserves-literate reader imports precisely the wrong expectation.

The alternatives were weighed against this project's own reference literature:

| candidate | verdict |
|---|---|
| **Unproven below LKH** | **Chosen.** *Unproven* says which kind of uncertainty it is; *LKH* is the standard term and names the exact boundary. Singh et al. (2017) use *Lowest Known Gas* in the identical sense — a well that encountered LKG with the structure *“assumed to be filled to the spill point”*. |
| *Untested below exit* | **Taken.** Milkov (2021) uses *“untested”* for the volume left **updip** of a dry hole — our attic. Reusing it a metre away would collide. |
| *Downdip volume* | **Taken, differently.** Schneider & Cook partition at the *well*; ours partitions at the *penetrated interval*. See the comparison on tab ④. |
| *Un-bottomed volume* | Real usage — a well that does not reach the contact has not *bottomed* the reservoir — but less widely recognised. |

The internal name changed with the label, so nothing in the code still says
*possible*. The **colour is unchanged**: the tan that has always meant this volume
still does, on every figure and in the exported artefacts.
        """
    )

    st.divider()
    st.markdown("### Choosing a location: five criteria, one frontier")
    st.markdown(
        """
There is no location that is optimal *simpliciter*. 3.8 is the honest object — the
trade-off frontier — and every single-number criterion is a different scalarisation of
it. Disagreement between them is information, not noise.

| Criterion | Maximises | Whose question |
|---|---|---|
| **Chance** | `P_well` | Always the shallowest supported depth, so it is a boundary rather than a recommendation. |
| **Expectation** | `P_well × mean volume` | The portfolio's. Additive across prospects, and **risk-neutral**. |
| **Commercial chance** | `Pc = P_well × P(>MEFS \| discovery)` | Rose's, and the number an EMV calculation takes. |
| **Certainty equivalent** | risk-adjusted volume | A risk-averse party's. Never deeper than the expectation peak. |
| **Information** | uncertainty reduction (3.3) | Haskett's, and it is about *appraisal* — see the caveat there. |

**A hurdle is not a criterion.** "90 % confident it is commercial" is a *constraint*:
the feasible set is a half-line, and the answer is the best odds inside it. Three
readings, and only two are attainable:

- **Unconditional `Pc ≥ 0.90`** — impossible unless `POS_prospect ≥ 0.90`, since
  `Pc ≤ P_well ≤ POS`. If a mandate is written this way, say so.
- **`P(commercial | discovery) ≥ 0.90`** — the slider on tab ③.
- **P90 proven ≥ MEFS**, i.e. even a poor discovery clears the bar — the inverse on
  tab ③ with the statistic set to P90 and the target set to MEFS. The two usually land
  within a few metres of each other, because both are asking the low tail to clear.

**Risk tolerance here is in MMboe, and that is a real limitation.** It is properly a
monetary quantity — about the loss a balance sheet can absorb — and no well cost appears
anywhere in this tool. It ranks locations correctly under a fixed well cost, which is
the case a single prospect is usually in. Do not carry a certainty equivalent in MMboe
into an economic model as though it were an expected value.
        """
    )

    st.divider()
    st.markdown("### Two sequential colour ramps, and what each one means")
    st.markdown(
        "**Blues counts trials. Inferno shows chance.** 2.2's grid and any density "
        "shading run light-to-dark in one blue hue; 3.11's markers run inferno. They "
        "encode different *kinds* of thing and nothing said so before now. The concept "
        "palette is separate from both again: it names *which volume*, never how much "
        "or how likely."
    )

    st.divider()
    st.markdown("### Two different minimum volumes")
    st.warning(
        "**MCFS / MEFS** — minimum commercial (Rose) or economic (this app's input) field "
        "size. As used here, the same threshold under two names: the smallest discovery "
        "worth **developing**, and one input serves both."
        "\n\n**The assessment minimum** — a minimum column height below the apex. The smallest "
        "accumulation worth **carrying in the assessment at all**. A different quantity, and "
        "the one Lars's reference figure labels *Min. volume*."
    )
    st.markdown(
        """
**Where the two names do come apart.** Some houses separate them: *economic* is the volume
at which the project breaks even, NPV = 0; *commercial* adds strategic, contractual and
above-ground hurdles on top, so MCFS ≥ MEFS. This app carries **one** threshold and one
line — if your house separates them, enter whichever you are testing against and the label
follows it.
        """
    )

    st.divider()
    st.markdown("### Reading a volume against the line")
    st.info(
        "**A percentile does not have a probability of exceeding the threshold.** P90 is a "
        "fixed volume: it clears MEFS or it does not, and that is 0 or 1 — so a column of "
        "four probabilities against P90 / P50 / Pmean / P10 would be four restatements of "
        "one fact."
        "\n\n**The quantity that does have a probability is the concept**, and it already "
        "has a name: `P(volume > MEFS)` is the *exceedance probability at MEFS* — which is "
        "the percentile the line itself sits at. One number per volume concept."
    )
    st.markdown(
        """
So tab ④ gives both halves, and they are different kinds of reading:

* the **ladder** — P90 / P50 / Pmean / P10 with a ✓ or ✗ against the line. This says
  *between which percentiles* the threshold falls, which is what a reader scans for.
  Pmean is marked as not a percentile; it sits between P50 and P10 because the
  distribution is right-skewed.
* **`P(> MEFS)`** — the exact chance, **conditional on that concept's own outcome**. The
  attic's is conditional on a charged dry hole, the unproven volume's on the well leaving
  the reservoir in hydrocarbons. They are not on one footing and must never be summed.

The two agree by construction — if the P50 clears the line then the chance is at least a
half — and a test asserts it, because a ladder that contradicts its own probability is
invisible on screen.

**On the well-associated volume, `P(> MEFS)` is exactly Rose's `Pmcfs(well)`**, so the
ladder and the commercial-chance strip above it are the same arithmetic read two ways.
        """
    )
    st.markdown(
        """
**Only one of them is ever drawn.** MEFS is a line on five figures and carries the palette's red. The assessment minimum is a *mapping* — it reports the contact depth, area and volume percentile a minimum column height corresponds to, and filters nothing — so it has no colour to collide with. If it is ever plotted it should take a line style rather than an eighth hue: the palette is already at the limit of what stays separable under colour-vision deficiency.

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
A **risked** or **expected** mean — this app's **Expected** metrics — is
mean × chance: **{gs['discovery']['mean']:.2f} × {chance.p_well:.1%} = {ev:.2f} MMboe**.

That is the only volume figure here that is **additive across prospects** — two success-case
means cannot be added, because each is conditional on its own outcome. It is also a number
that describes no outcome that can occur: this well either finds something near
{gs['discovery']['mean']:.1f} MMboe or it finds nothing. Quote it beside the chance and the
size, never instead of them.
        """
    )

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

    # -------------------------------------------------------------- references
    st.divider()
    st.markdown("### References")
    st.caption(
        "Every entry below was checked against the document itself — the PDFs are in "
        "`_local/Papers/`, and the two online sources were opened and read. Where a work is cited "
        "*through* another rather than read directly, it says so."
    )
    st.markdown(
        # Wrapped so the {b2} placeholder in the closing paragraph resolves. It was
        # left literal, and the reader was shown "{b2}'s regret curve".
        ref(
            """
**Read in full — the PDFs are in `_local/Papers/`**

| Work | What it contributes here |
|---|---|
| **Schneider, M., Citron, G.P., Haryott, P. & Cook, D. (2023)** *Drilling an exploration prospect downdip: quantifying the trade-offs between chance of success and associated resource potential.* AAPG Bulletin **107**(5) 743–759. [doi:10.1306/09232222051](https://doi.org/10.1306/09232222051) · [open access](https://pubs.geoscienceworld.org/aapg/aapgbull/article/107/5/743/622239/Drilling-an-exploration-prospect-downdip) | The definitive reference. Whole-trial up-dip/down-dip grouping — this app's **reference engine**. Names the finer proven/possible split as *“additional complexity”* without computing it; `core/classes.py` is that complexity, implemented. Also the source of the convention that the EUR distribution is the **success case** and is determined *before* the chance, and that Pg is the chance of exceeding the **P99** EUR. |
| **Milkov, A.V. (2021)** *Reporting the expected exploration outcome: when, why and how the probability of geological success and success-case volumes for the well differ from those for the prospect.* J. Pet. Sci. Eng. **204** 108754. [doi:10.1016/j.petrol.2021.108754](https://doi.org/10.1016/j.petrol.2021.108754) | The crest/apex reference convention, and the peer-reviewed statement that well POS is prospect POS times the contact distribution. His segment A2 falls from 0.34 at the crest to 0.07 down-dip. Also the definition this app uses for percentiles: *“P90 is defined as 90 % probability of exceeding the P90 estimated value.”* |
| **Schneider, M. & Cook, D.M. Jr. (2017)** *Drilling a Downdip Location: Effect on Updip and Downdip Resource Estimates and Commercial Chance.* AAPG Search & Discovery **#42102**, posted 3 July 2017 — with the Rose & Associates long-form version, Houston, May 2021. [search & discovery](https://www.searchanddiscovery.com/documents/2017/42102schneider/ndx_schneider.pdf) | Equation 1 (`Pwell = Pg × Ptrap@well / Ptrap`, cap included), the P90-area reference contour, the *“No Regrets”* volume, `Pmcfs(well)` and `Pc(well)`. Both documents are in `_local/Papers/`; the 2021 file is a Rose & Associates client report authored by Schneider and Cook, not by Rose. |
| **Haskett, W.J. (2003)** *Optimal Appraisal Well Location Through Efficient Uncertainty Reduction and Value of Information Techniques.* SPE **84241**, SPE ATCE, Denver, 5–8 October 2003. [doi:10.2118/84241-MS](https://doi.org/10.2118/84241-MS) | Appraisal placement as value of information — 3.3's uncertainty-reduction curve, with the optimum found by argmax rather than by eye. |
| **Singh, V., Yemez, I., Izaguirre, E. & Racero, A. (2017)** *Optimal Subsurface Appraisal: A Key Link to the Success of Development Projects.* Am. J. Applied Sciences **14**(2) 217–230. [doi:10.3844/ajassp.2017.217.230](https://doi.org/10.3844/ajassp.2017.217.230) · open access | Appraisal-value framing around the same trade-off: what a well is worth is what it resolves, not what it finds. |
| **Hood, K.C. (2019)** *Column Height.* Risk Coordinators' Workshop, 14 November 2019 (ExxonMobil Upstream Integrated Solutions) — and **Hood, K.C. (2024)** *Hydrocarbon Column Heights, Parts 1 & 2*, Rose & Associates blog. Both in `_local/Papers/HCWC/` | The assessment minimum belongs to a minimum **column height**, linked to seal capacity, not to a minimum volume — decision 6, and why this app maps a column height rather than filtering on a volume. Two separate documents; the 2019 workshop deck and the 2024 blog pair are often conflated. |

**Consulted online, and re-read to check these claims**

| Source | What it contributes |
|---|---|
| [**Longley, I. (27 January 2026)** *Understanding the “Minimum Economic Field Size” concept and aggregating targets*, GeoExpro](https://geoexpro.com/understanding-the-minimum-economic-field-size-concept-and-aggregating-targets/) | *“Applying a volume cut-off will always lead to the Mean Unrisked volume to increase in comparison to the non-truncated case”* while *“the chance of ‘commercial’ success (Pc) comes down.”* The two do not cancel — which is exactly why this app draws MEFS as a **reference line to read probabilities against** and never truncates a distribution with it. |
| [**Rose & Associates** — Pwell implementation, 2017–2021](https://www.roseassoc.com/pwell-implementation-from-2017-to-2021/) | Vendor documentation for the MMRA (2018) → RoseRA (2021) lineage. Not peer-reviewed, and cited only for that history. |

**Cited through another work, not read here**

| Work | Route |
|---|---|
| **Rose, P.R. (2001)** *Risk Analysis and Management of Petroleum Exploration Ventures.* AAPG Methods in Exploration **12** | Reached via Schneider et al. (2023), who cite it for Pg being the chance of a discovery *“equal to or exceeding the P99 EUR”* — the anchor this app's unconditional curves are checked against. The book itself is not in `_local/Papers/`. |
| **Capen (1976)**; **Otis & Schneidermann (1997)** | Cited by Schneider et al. (2023) as the established basis for unbiased assessment of the success-case distribution and its chance. Listed so the lineage is visible, not because they were consulted. |

**Deliberately not obtained.** Milkov & Samis (2020, AAPG Bulletin 104) and Samis &
Milkov (2020), on the real-option value of untested up-dip volume after a dry hole.
Both paywalled. If the attic / regret analysis becomes central they are the next two
to get — and {b2}'s regret curve is the place they would land.

---

**Two provenance notes.** Prospect A's demo data is fictional and safe to publish.
Prospect B comes from an internal study and its publication status has **not** been
confirmed — treat it as the licensee's until it has. Either way the numbers are held to a
fixed specification: a parity suite locks fifteen reference values and was written before
any other code in this tool.
        """
        )
    )
