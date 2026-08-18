"""Chance: the location factor, and how it is attributed to risk elements.

The whole tool rests on one decomposition::

    r_location = P(contact deeper than the well | hydrocarbons present)
    P_well     = POS_prospect x r_location

Two numbers, two meanings, never mixed. ``r_location`` is the only quantity the
well's position controls; ``POS_prospect`` is the only quantity it does not.

The source workbook computed ``1 - PERCENTRANK(all contacts, entry)``, which
already includes the chance-failure trials, and then multiplied by a separately
entered POS. That is right only when the entered POS is 1.0.

Two published conventions differ in where the location factor is referenced:

* **Crest / apex** (Milkov 2021, and the source workbook): ``P_well = POS`` only
  at the very top of the structure.
* **P90 area** (Rose; Schneider et al. 2023 Eq. 1): closure chance is conventionally
  assessed as the confidence the closure holds at least the P90 area, to stay
  consistent with POS being the chance of the P99 EUR. So ``P_well = POS`` for
  any well at or up-dip of that contour, and the factor is normalised by 0.90.

On the reference dataset the Rose convention is a flat 1.11x (= 1/0.90) uplift
at every depth plus a cap up-dip of the P90-area contour. That is a real
difference, not rounding, so it is an explicit setting rather than a default
buried in the code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from ..io.adapters.base import TrialSet

ELEMENTS = ("charge", "trap", "reservoir", "retention")

#: Element key -> the name shown to a reader. **Closure**, not "trap" (Lars,
#: 2026-08-12): what the element assesses is whether a mapped closure with a seal
#: exists, and "trap" reads as the trapping *mechanism* or as a verb.
#:
#: The keys stay ``trap`` on purpose. This project's rule is that behaviour branches
#: on stable keys and never on label text -- rewording user copy must not be able to
#: change which number the app uses -- and a case saved before the rename still has
#: to load. So the wording lives here and nowhere else.
ELEMENT_LABELS = {
    "charge": "Charge",
    "trap": "Closure",
    "reservoir": "Reservoir",
    "retention": "Retention",
}


class ReferenceContour(str, Enum):
    CREST = "crest"        # Milkov 2021 / the source workbook -- default
    P90_AREA = "p90_area"  # Rose / Schneider et al. 2023 Eq. 1
    CUSTOM = "custom"


@dataclass
class ChanceResult:
    pos_prospect: float
    r_location: float
    p_well: float
    reference: ReferenceContour
    reference_percentile: float | None = None
    reference_depth: float | None = None


def r_location(
    ts: TrialSet,
    z_entry: float,
    *,
    reference: ReferenceContour = ReferenceContour.CREST,
    reference_percentile: float = 0.90,
) -> tuple[float, float | None]:
    """The location factor, conditional on hydrocarbons being present.

    Returns ``(r, reference_depth)``. Conditioning on success is what keeps the
    factor free of geological risk -- the failure trials belong to POS, not here.
    """
    res = ts.col("resource")
    contact = ts.col("contact")
    succ = res > 0.0
    if succ.sum() == 0:
        return float("nan"), None
    raw = float((contact[succ] > z_entry).mean())

    if reference is ReferenceContour.CREST:
        return raw, None

    # Rose: normalise by the percentile at which closure chance was assessed, and
    # hold the factor at 1.0 for any location up-dip of that contour.
    ref_depth = float(np.percentile(contact[succ], (1.0 - reference_percentile) * 100.0))
    return float(min(1.0, raw / reference_percentile)), ref_depth


def p_well(
    ts: TrialSet,
    z_entry: float,
    pos_prospect: float,
    *,
    reference: ReferenceContour = ReferenceContour.CREST,
    reference_percentile: float = 0.90,
) -> ChanceResult:
    r, ref_depth = r_location(
        ts, z_entry, reference=reference, reference_percentile=reference_percentile
    )
    return ChanceResult(
        pos_prospect=pos_prospect,
        r_location=r,
        p_well=pos_prospect * r,
        reference=reference,
        reference_percentile=reference_percentile if reference is not ReferenceContour.CREST else None,
        reference_depth=ref_depth,
    )


# --------------------------------------------------------------- allocation
SCHEMES: dict[str, dict[str, float]] = {
    # name -> weights per element, summing to 1 (or all zero for "none")
    "none": {"charge": 0.0, "trap": 0.0, "reservoir": 0.0, "retention": 0.0},
    "equal_cube_root": {"charge": 1 / 3, "trap": 1 / 3, "reservoir": 0.0, "retention": 1 / 3},
    "all_to_trap": {"charge": 0.0, "trap": 1.0, "reservoir": 0.0, "retention": 0.0},
}

SCHEME_LABELS = {
    "none": "None — report r separately (Milkov 2021)",
    "equal_cube_root": "Equal cube-root — charge, closure, retention",
    "all_to_trap": "All to closure (trap, Rose Eq. 1)",  # 'trap' kept in the
    # label on purpose (Lars, 2026-08-12): Rose's Eq. 1 assigns the whole location
    # penalty to what he calls the *trap* element, and a reader who knows the
    # equation by that word has to be able to find it here.
    "custom": "Custom weights",
}

# The schemes the app offers, in the order it offers them. Named rather than
# sliced off SCHEME_LABELS positionally, so adding "custom" (or any other
# label) cannot silently reorder or change what the UI presents.
SHIPPED_SCHEMES = ("none", "equal_cube_root", "all_to_trap")


def normalised_weights(
    weights: dict[str, float] | str = "none",
) -> tuple[dict[str, float], list[str]]:
    """Resolve a scheme name or a weight dict to weights summing to 1, or all zero.

    Split out of :func:`allocate` because the waterfall needs the *same*
    normalised weights in order to work out how much of the location factor
    the elements have already absorbed. Reading the raw ``SCHEMES`` table
    instead would double-count it for any weight set that does not already
    sum to 1.
    """
    if isinstance(weights, str):
        if weights not in SCHEMES:
            raise KeyError(f"unknown scheme {weights!r}; choose from {sorted(SCHEMES)} or pass weights")
        w = dict(SCHEMES[weights])
    else:
        w = dict(weights)

    total = sum(w.values())
    notes: list[str] = []
    if total > 0 and abs(total - 1.0) > 1e-9:
        w = {k: v / total for k, v in w.items()}
        notes.append(f"Weights summed to {total:.3f}; normalised to 1.")
    return w, notes


def allocate(
    elements: dict[str, float],
    r: float,
    weights: dict[str, float] | str = "none",
    *,
    floor: float = 0.10,
) -> tuple[dict[str, float], list[str]]:
    """Attribute the location factor across the geological chance elements.

    ``P_element_at_well = P_element * r ** w_element`` with the weights summing
    to 1. Every scheme returns the *same* ``P_well`` -- only the attribution
    differs, which is why the figures say so explicitly. Spreading a single
    number across four elements presents it differently; it does not add
    information about charge or closure.

    Returns ``(revised_elements, warnings)``.
    """
    w, warnings = normalised_weights(weights)

    revised = {}
    for el in ELEMENTS:
        base = float(elements.get(el, 1.0))
        revised[el] = base * (r ** w.get(el, 0.0)) if r > 0 else 0.0
        if revised[el] < floor and base >= floor:
            warnings.append(
                f"{ELEMENT_LABELS[el]} falls from {base:.2f} to {revised[el]:.2f}, below the {floor:.2f} "
                f"floor. An allocation is a presentation of one number, not a re-assessment — "
                f"check this is still geologically sayable."
            )
    return revised, warnings


def expected_volume(mean: float, chance: float) -> float:
    """A success-case mean multiplied by the chance of getting it.

    The source workbook's column O, "'Risked' Pmean" (`Results!O4:O8`), and the
    only volume figure in this tool that is **additive across prospects** -- two
    success-case means cannot be added, because each is conditional on its own
    outcome, whereas two expected volumes can.

    It is deliberately *not* the headline anywhere. An expected volume of 7.6
    MMboe describes no outcome that can actually occur: the well either finds
    something near 16.5 or it finds nothing. Quoting it alone hides both the
    chance and the size, which is why the app shows it beside them rather than
    instead of them. It is what a portfolio adds up, not what a well finds.
    """
    return float(mean) * float(chance)


def cube_root_factor(r: float) -> float:
    """The source workbook's ``Results!V15``.

    Equals ``r ** (1/3)``. Because chance factors are multiplicative, their
    natural additive space is logarithmic -- so a cube root is an equal split of
    the location log-risk across three elements (charge, closure, retention), with
    reservoir exempt. The exemption is right: the contact distribution is a
    fill / spill / retention / charge statement, not a reservoir-presence one.
    """
    return float(r ** (1.0 / 3.0))


# ---------------------------------------------------------------- waterfall
# (label, factor, role) -- role is "chance", "location" or "reconcile"
WaterfallStep = tuple[str, float, str]


def step_element(label: str) -> str | None:
    """The element key a waterfall step belongs to, or ``None``.

    ``"Closure"`` and ``"Closure · r^0.33"`` both map to ``"trap"``; the location
    residual and the POS reconciliation map to ``None``, because they belong to no
    single element.

    Lives here rather than in the figures for the same reason the arithmetic does: a
    figure that recovered the element by string-matching its own label would get a
    different answer the moment the wording changed, and the wording is
    :data:`ELEMENT_LABELS`' business.
    """
    head = label.split(" · ")[0].strip()
    for key, name in ELEMENT_LABELS.items():
        if head == name:
            return key
    return None


def waterfall_steps(
    elements: dict[str, float],
    r: float,
    pos_prospect: float,
    weights: dict[str, float] | str = "none",
) -> list[WaterfallStep]:
    """The multiplicative steps from 1.0 down to ``P_well``, for B4.

    The arithmetic lives here rather than in the figure so that the product of
    the returned factors is **exactly** ``pos_prospect * r`` by construction,
    and so it can be tested without a plotting library. A figure that merely
    computed its own running product could -- and did -- end up drawing a total
    that disagreed with the ``P_well`` shown elsewhere in the app.

    Three kinds of step:

    ``chance``
        One per geological element, at its entered value.
    ``location``
        The location factor, or the share of it a scheme has pushed onto an
        element. Split out rather than folded silently into the element,
        because "which elements carry the location penalty" is one of the
        questions the tool exists to answer.
    ``reconcile``
        Present only when the entered chance table does not multiply to the
        POS actually in use -- which is the normal case when the trials carry
        the risking and the table is display-only. Naming it is the point: the
        alternative is a waterfall that quietly totals something other than
        ``P_well``.
    """
    w, _ = normalised_weights(weights)
    steps: list[WaterfallStep] = []

    prod_elements = 1.0
    for el in ELEMENTS:
        base = float(elements.get(el, 1.0))
        prod_elements *= base
        steps.append((ELEMENT_LABELS[el], base, "chance"))
        wi = float(w.get(el, 0.0))
        if wi > 0.0:
            steps.append((f"{ELEMENT_LABELS[el]} · r^{wi:.2f}", float(r ** wi), "location"))

    if prod_elements > 0.0 and abs(prod_elements - pos_prospect) > 1e-12:
        steps.append(("POS reconciliation", pos_prospect / prod_elements, "reconcile"))

    residual = 1.0 - sum(w.values())
    if residual > 1e-9:
        label = "Location (r)" if residual >= 1.0 - 1e-9 else f"Location (r^{residual:.2f})"
        steps.append((label, float(r ** residual), "location"))
    return steps


# ---------------------------------------------------------- the risk summary
#: The three columns of the risk summary, in the order a reader works through
#: them. The names are Lars's, from the sheet this reproduces.
SUMMARY_COLUMNS = ("Probability (Play)", "Probability (Prospect given Play)",
                   "Probability (at well location)")


@dataclass
class RiskSummary:
    """The chance table, the location factor, and the one multiplied through the
    other -- as a table rather than a figure.

    Reproduces the summary block Lars keeps in the workbook, and it exists because
    **the two halves come from different places and at different times**:

    * ``charge``, ``trap`` (shown as *Closure*), ``reservoir`` and ``retention``
      are **inputs** --
      judgements about the prospect, made before anyone picks a location and
      unchanged by picking one. They belong with the data and the risking
      convention, which is why the chance table lives in tab ② beside the
      distributions it risks.
    * ``r_location`` is **computed**, from the trial file and the well's entry
      depth. It exists only once there is a well.

    So a summary that multiplies the first by the second can only be assembled
    after both, which is why it sits at the end, in tab ⑤. Putting the input and
    the summary in one place would invite reading the third column as something a
    person entered.

    ``correction_factor`` is what the location costs each element that carries it:
    ``r^(1/3)`` under the shipped equal-cube-root scheme, because three of the four
    elements share the penalty and **reservoir is exempt** -- a well that misses
    the column still saw the rock, so its reservoir risk is unchanged by where it
    was drilled. That exemption is why the factor is a cube root and not a fourth
    root, and it is the same arithmetic :func:`allocate` and B4 already use.
    """

    rows: list[dict[str, float | str]]
    play_chance: float
    conditional_prospect_chance: float
    prospect_pos: float
    well_pos: float
    correction_factor: float
    scheme: str
    warnings: list[str] = field(default_factory=list)
    #: The four play-level chances, when the caller risked the play element by
    #: element rather than as one number. Empty when a scalar was given.
    play_elements: dict[str, float] = field(default_factory=dict)

    def as_records(self) -> list[dict[str, object]]:
        """The element rows, ready for a dataframe. Private fields dropped."""
        return [{k: v for k, v in r.items() if not k.startswith("_")}
                for r in self.rows]

    def element_keys(self) -> tuple[str | None, ...]:
        """The stable element key per row, ``None`` for the location-factor row.

        Parallel to :meth:`as_records`, so a caller can colour row *i* without
        matching the label in it.
        """
        return tuple(r.get("_key") for r in self.rows)

    def result_records(self) -> list[dict[str, object]]:
        """The four result lines under the table, in the order they are read."""
        return [
            {"result": "Play chance", "value": self.play_chance},
            {"result": "Cond. prospect chance", "value": self.conditional_prospect_chance},
            {"result": "Final prospect POS", "value": self.prospect_pos},
            {"result": "Well location POS", "value": self.well_pos},
        ]


def risk_summary(
    elements: dict[str, float], r: float, *, scheme: str | dict[str, float] = "equal_cube_root",
    play_chance: float | None = None, play_elements: dict[str, float] | None = None,
) -> RiskSummary:
    """Build the risk summary table.

    The third column is :func:`allocate`'s output, so this function introduces no
    arithmetic of its own -- which is deliberate. The product of the third column
    is ``POS_prospect x r_location = P_well`` by construction, and
    ``test_the_summary_multiplies_to_p_well`` checks it against
    :func:`wellvolpos.core.chance.p_well` rather than against this table's own
    numbers: a figure that totals its own steps can agree with itself and still
    disagree with `p_well`, which is how B4 shipped a wrong total.

    **The play is risked element by element** (Lars, 2026-08-11), not as a single
    number: ``play_elements`` carries a chance for each of charge, closure, reservoir
    and retention *at the play level*, and ``elements`` carries the same four
    **conditional on the play working**. Eight inputs, two levels, and the first
    column of the table stops being a constant.

    That split is the standard one and it is worth stating why it is not cosmetic:
    "is there a working petroleum system here at all" and "does *this* closure have
    a seal" are different questions with different evidence, and a single number for
    the play cannot be argued about element by element the way a column can. The
    arithmetic is unaffected -- ``POS_prospect`` is the product of all eight either
    way -- but the conversation is not.

    ``play_chance`` remains accepted as a scalar shortcut for callers that have only
    one number; passing both is an error, since they would disagree.
    """
    if play_chance is not None and play_elements is not None:
        raise ValueError(
            "pass play_elements or play_chance, not both -- two statements of the "
            "play chance can disagree, and there would be no way to say which won"
        )
    if play_elements is None:
        play_elements = {}
        play_scalar = 1.0 if play_chance is None else float(play_chance)
    else:
        play_elements = {k: float(v) for k, v in play_elements.items()}
        play_scalar = float(np.prod(list(play_elements.values()))) if play_elements else 1.0

    at_well, warnings = allocate(elements, r, scheme)
    conditional = float(np.prod([float(v) for v in elements.values()])) if elements else 1.0
    prospect_pos = play_scalar * conditional
    rows: list[dict[str, float | str]] = []
    for name in elements:
        given_play = float(elements[name])
        # Per-element play chance where one was given, else the scalar spread over
        # no elements at all -- which is why the fallback is the whole play chance
        # on the first row would be wrong. It goes on every row as a constant, the
        # way it always did, and the column header says "(Play)" not "(this
        # element's play)".
        at_play = play_elements.get(name, play_scalar if not play_elements else 1.0)
        rows.append({
            # **The stable key travels beside the label**, prefixed so
            # :meth:`as_records` can drop it. A caller that wants the element's
            # colour must not have to match the displayed wording -- that is the
            # rule the whole ``trap``/"Closure" rename rests on.
            "_key": name,
            "Chance element": ELEMENT_LABELS.get(name, name.capitalize()),
            SUMMARY_COLUMNS[0]: float(at_play),
            SUMMARY_COLUMNS[1]: given_play,
            SUMMARY_COLUMNS[2]: float(at_well[name]),
            # Named so the exemption is visible in the table rather than only in
            # prose: reservoir carries no location penalty under any shipped scheme.
            "Carries the location penalty": not np.isclose(at_well[name], given_play),
        })

    # ``P_well`` is ``play x POS_prospect x r_location`` by definition, and it is
    # computed that way here rather than as the product of the third column.
    #
    # Those two agree under a scheme that *allocates* the penalty to elements, but
    # the "none" scheme deliberately does not -- it reports ``r`` separately -- so
    # the third column then equals the second and multiplying it out gives
    # POS_prospect. Reporting that as the well POS would be this codebase's
    # recurring mistake for the fifth time: an unrisked number under a risked label.
    # Under "none" the location factor gets its own row instead, so the column still
    # multiplies to the number at the bottom of the table.
    well_pos = prospect_pos * float(r)
    allocated = float(np.prod([float(v) for v in at_well.values()])) * play_scalar
    if not np.isclose(allocated, well_pos):
        rows.append({
            "Chance element": "Location factor r",
            SUMMARY_COLUMNS[0]: play_scalar if not play_elements else 1.0,
            SUMMARY_COLUMNS[1]: 1.0,
            SUMMARY_COLUMNS[2]: float(r),
            "Carries the location penalty": True,
        })
    factor = _correction_factor(elements, at_well)
    return RiskSummary(
        rows=rows, play_chance=play_scalar, play_elements=play_elements,
        conditional_prospect_chance=conditional, prospect_pos=prospect_pos,
        well_pos=well_pos, correction_factor=factor,
        scheme=scheme if isinstance(scheme, str) else "custom weights",
        warnings=list(warnings),
    )


def _correction_factor(elements: dict[str, float], at_well: dict[str, float]) -> float:
    """The per-element multiplier the location applies, where it applies at all.

    Reported as one number because under the shipped schemes it *is* one number --
    ``r^(1/3)`` shared by the three elements that carry it. Where a custom weighting
    gives each element a different factor there is no single value to report, so
    this returns NaN rather than an average that would look like one.
    """
    ratios = [
        float(at_well[k]) / float(v)
        for k, v in elements.items()
        if float(v) > 0 and not np.isclose(at_well[k], float(v))
    ]
    if not ratios:
        return 1.0
    return float(ratios[0]) if np.allclose(ratios, ratios[0]) else float("nan")
