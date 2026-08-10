"""Reference-engine and extension figures: A1-A6, B0-B5.

Every figure here is built on :func:`wellvolpos.viz.theme.new_figure` so
palette and rcParams can never drift between figures, and every axis that
carries a depth goes through :func:`wellvolpos.viz.theme.depth_axis`. A5, A6,
B4 and B5 are the figures in this set with no depth on either axis (all four
are about chance or the resource distribution, not where either sits
structurally), so they are the ones that do not call it -- see
``tests/test_axes.py`` and ``tests/test_figures.py``.

Colour is assigned by meaning throughout, per ``theme.ROLES``: the two chance
curves in A3 and the single curve in B3 both use the discovery/chance blue
(distinguished by line style, not colour, since both are chances); A4's
percentile trend and A1's area-depth curve both use the prospect aqua because
they characterise the whole un-cut model, not any one outcome; A5's and B1's
series map onto the canonical roles directly; A2's stacked outcome tree and
B0's schematic section colour-key the same four outcomes; B2 layers the
chance/regret curves onto the same roles as A5/B1; B4's waterfall and B5's
"at the well" markers both use the chance/discovery blue, since every bar and
every revised point in those two figures is a chance, not an outcome.
"""

from __future__ import annotations

import numpy as np

from ..core.chance import ELEMENTS, SCHEME_LABELS, SHIPPED_SCHEMES, allocate
from ..core.chance import waterfall_steps as chance_waterfall_steps
from ..core.classes import VolumeClasses
from ..core.groups import Groups
from ..core.structure import AreaDepth
from ..core.sweep import Sweep, VolumeSweep
from ..io.adapters.base import TrialSet
from .theme import SEQUENTIAL_CMAP, colour, depth_axis, new_figure, palette

__all__ = [
    "fig_a1_area_depth",
    "fig_a2_outcome_tree",
    "fig_a3_chance_decomposition",
    "fig_a4_resource_vs_depth",
    "fig_a5_exceedance",
    "fig_a6_overlap",
    "fig_b0_section",
    "fig_b1_volume_split",
    "fig_b2_chance_vs_regret",
    "fig_b3_uncertainty_reduction",
    "fig_b4_chance_waterfall",
    "fig_b5_allocation_dumbbell",
]


def _depth_percentile_trend(contact: np.ndarray, resource: np.ndarray, n_bins: int = 40):
    """Bin by contact depth and return (z, p90, p50, p10) trend lines.

    Equal-count bins along sorted contact depth, so each bin has comparable
    statistical weight even where trials are unevenly sampled across the
    structure. Percentiles follow the petroleum convention used everywhere
    else in the app: P90 is the low value, P10 the high one.
    """
    order = np.argsort(contact)
    c, r = contact[order], resource[order]
    z, p90, p50, p10 = [], [], [], []
    for idx in np.array_split(np.arange(c.size), min(n_bins, c.size)):
        if idx.size == 0:
            continue
        z.append(float(c[idx].mean()))
        p90.append(float(np.percentile(r[idx], 10.0)))
        p50.append(float(np.percentile(r[idx], 50.0)))
        p10.append(float(np.percentile(r[idx], 90.0)))
    return np.array(z), np.array(p90), np.array(p50), np.array(p10)


def _exceedance(values: np.ndarray):
    """Sorted values and their exceedance probability P(X >= value), in %."""
    v = np.sort(np.asarray(values, dtype=float))
    n = v.size
    if n == 0:
        return v, np.array([])
    return v, 100.0 * (n - np.arange(n)) / n


def fig_a1_area_depth(
    ad: AreaDepth, *, current_entry: float | None = None, current_exit: float | None = None,
    dark: bool = False,
):
    """The area-depth curve recovered from the trials, entry/exit marked.

    The structural spine of the whole tool -- A(z) is what turns a well's
    depth into its position on the structure, and every figure that splits a
    trial at the well rests on this curve. Uses the prospect aqua because it
    characterises the whole closure, not any one outcome; the well itself
    gets its own dedicated colour so it reads as the thing being placed
    against the curve, not a feature of it.
    """
    fig, ax = new_figure(figsize=(5, 5.5), dark=dark)
    p = palette(dark)

    ax.plot(ad.a, ad.z, color=colour("prospect", dark), lw=2.0)
    if current_entry is not None:
        ax.axhline(current_entry, color=p["well"], ls="--", lw=1.4, label="Entry")
    if current_exit is not None and current_exit != current_entry:
        ax.axhline(current_exit, color=p["well"], ls=":", lw=1.4, label="Exit")

    depth_axis(ax, zlim=(ad.shallowest, ad.deepest))
    ax.set_xlim(left=0)
    ax.set_xlabel("Enclosed area (km²)")
    ax.set_title("A1 · Area–depth curve")
    if current_entry is not None:
        ax.legend(loc="lower right", fontsize=7.5)
    fig.tight_layout()
    return fig, ax


def fig_a2_outcome_tree(sweep: Sweep, *, current_z: float | None = None, dark: bool = False):
    """The outcome tree vs entry depth, as a stacked area chart.

    The four outcomes -- chance failure, dry-with-attic, discovery with the
    contact seen, discovery with hydrocarbons continuing past the exit --
    partition every trial regardless of location, so the bands always sum to
    100 %. Chance failure does not vary with location (it is decided before
    the well is placed) and gets the neutral muted grey reserved for
    everything outside the four canonical roles; the other three map onto
    attic / discovery / possible directly, matching how the same outcomes are
    coloured everywhere else in the tool.
    """
    fig, ax = new_figure(figsize=(5, 5.5), dark=dark)
    p = palette(dark)

    cum0 = np.full_like(sweep.z, sweep.share_chance_failure * 100.0)
    cum1 = cum0 + sweep.share_dry_with_attic * 100.0
    cum2 = cum1 + sweep.share_contact_seen * 100.0
    cum3 = cum2 + sweep.share_hc_to_exit * 100.0

    ax.fill_betweenx(sweep.z, 0, cum0, color=p["muted"], label="Chance failure")
    ax.fill_betweenx(sweep.z, cum0, cum1, color=colour("attic", dark), label="Dry, with attic")
    ax.fill_betweenx(sweep.z, cum1, cum2, color=colour("discovery", dark), label="Discovery, contact seen")
    ax.fill_betweenx(sweep.z, cum2, cum3, color=colour("possible", dark), label="Discovery, HC to exit")

    if current_z is not None and sweep.z.min() <= current_z <= sweep.z.max():
        ax.axhline(current_z, color=p["text"], ls="--", lw=1.0)

    depth_axis(ax, zlim=(float(sweep.z.min()), float(sweep.z.max())))
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of trials (%)")
    ax.set_title(f"A2 · Outcome tree vs location (exit = entry + {sweep.z_gap:.0f} m)")
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=7.5)
    fig.tight_layout()
    return fig, ax


def fig_a3_chance_decomposition(
    sweep: Sweep, *, pos_prospect: float | None = None, pos_trials: float | None = None,
    current_z: float | None = None, dark: bool = False,
):
    """P_well and r_location vs entry depth, POS as a rule.

    Both curves are the same quantity's family -- a chance -- so both use
    the discovery/chance blue; solid for P_well, dashed for r_location. This
    is the figure that makes the decomposition in CLAUDE.md's "one idea
    everything rests on" impossible to misread: the two lines never touch
    except at the crest, and only P_well answers "will this well work".

    ``pos_prospect`` is the POS actually in use -- the one the curves are built
    from. ``pos_trials`` is the trial file's own implied POS, and is drawn as a
    second rule only when the two differ, which happens as soon as a chance
    table is entered instead of the trials being taken as already risked. The
    two used to share one argument labelled "POS_trials", which meant that
    entering a chance table drew a rule at the entered value under the file's
    label.
    """
    fig, ax = new_figure(figsize=(6, 5), dark=dark)
    p = palette(dark)
    c = colour("p_well", dark)

    ax.plot(sweep.p_well * 100.0, sweep.z, color=c, lw=2.0, label=r"$P_{well}$ = POS $\times$ r")
    ax.plot(sweep.r_location * 100.0, sweep.z, color=c, lw=1.4, ls="--",
            label="r = P(contact deeper | HC present)")

    def _pos_rule(value: float, label: str, ls: str) -> None:
        ax.axvline(value * 100.0, color=p["muted"], ls=ls, lw=1.0)
        ax.text(value * 100.0, sweep.z.min(), f" {label} {value:.3f}",
                rotation=90, va="top", ha="right", fontsize=7, color=p["text_secondary"])

    if pos_prospect is not None:
        _pos_rule(pos_prospect, "POS$_{prospect}$", ":")
    if pos_trials is not None and (
        pos_prospect is None or abs(pos_trials - pos_prospect) > 1e-9
    ):
        _pos_rule(pos_trials, "POS$_{trials}$", "-.")

    if current_z is not None and sweep.z.min() <= current_z <= sweep.z.max():
        p_here = float(np.interp(current_z, sweep.z, sweep.p_well))
        r_here = float(np.interp(current_z, sweep.z, sweep.r_location))
        ax.axhline(current_z, color=p["text_secondary"], ls="--", lw=1.0)
        ax.plot([p_here * 100.0], [current_z], "o", color=p["text"], zorder=5)
        ax.annotate(
            f"$P_{{well}}$ = {p_here:.1%}\nr = {r_here:.1%}",
            (p_here * 100.0, current_z), xytext=(8, -8), textcoords="offset points", fontsize=8,
            color=p["text"],
        )

    ax.set_xlim(0, 100)
    depth_axis(ax, zlim=(float(sweep.z.min()), float(sweep.z.max())))
    ax.set_xlabel("Probability (%)")
    ax.set_title("A3 · Chance decomposition vs location")
    ax.legend(loc="upper right", fontsize=7.5)
    fig.tight_layout()
    return fig, ax


def fig_a4_resource_vs_depth(
    ts: TrialSet, *, current_entry: float | None = None, mefs: float | None = None,
    n_bins: int = 40, zlim: tuple[float, float] | None = None, dark: bool = False,
):
    """Log-density hexbin of resource vs contact depth, with smoothed P90/P50/P10.

    Replaces the unreadable full-trial scatter. Success trials only -- the
    chance-failure zeros belong to POS, not to the shape of the resource
    distribution conditional on a contact actually being sampled.
    """
    res, contact = ts.col("resource"), ts.col("contact")
    succ = res > 0.0
    fig, ax = new_figure(figsize=(6, 5), dark=dark)
    p = palette(dark)

    ax.hexbin(res[succ], contact[succ], gridsize=45, cmap=SEQUENTIAL_CMAP, mincnt=1, bins="log")

    z, p90, p50, p10 = _depth_percentile_trend(contact[succ], res[succ], n_bins=n_bins)
    c = colour("prospect", dark)
    ax.plot(p50, z, color=c, lw=1.8, label="P50")
    ax.plot(p90, z, color=c, lw=1.1, ls=":", label="P90")
    ax.plot(p10, z, color=c, lw=1.1, ls=":", label="P10")

    if current_entry is not None:
        ax.axhline(current_entry, color=p["text_secondary"], ls="--", lw=1.0)
    if mefs is not None:
        ax.axvline(mefs, color=p["muted"], ls=":", lw=1.0)

    # zlim so a row can be given one shared depth range; without it A4
    # autoscales to its own hexbin extent and stops lining up with A1/A5.
    depth_axis(ax, ylabel="HC-water contact (m TVDSS)", zlim=zlim)
    ax.set_xlim(left=0)
    ax.set_xlabel("Recoverable resource (MMboe)")
    ax.set_title("A4 · Resource vs contact depth")
    ax.legend(loc="lower right", fontsize=7.5)
    fig.tight_layout()
    return fig, ax


def fig_a5_exceedance(
    ts: TrialSet, groups: Groups, vc: VolumeClasses, *, mefs: float | None = None, dark: bool = False,
):
    """Exceedance curves for prospect / discovery / proven / attic, at the chosen location.

    The money chart. No depth on either axis -- see the module docstring --
    so the four canonical colour roles map directly onto the four series.
    """
    res = ts.col("resource")
    fig, ax = new_figure(figsize=(6, 5), dark=dark)
    p = palette(dark)

    series = [
        ("Prospect (all trials)", res, "prospect"),
        ("Discovery case", res[groups.discovery], "discovery"),
        ("Proven at well", vc.proven[groups.discovery], "proven"),
        ("Attic | dry hole", res[groups.dry_with_attic], "attic"),
    ]
    for label, values, role in series:
        v, pct = _exceedance(values)
        if v.size == 0:
            continue
        ax.plot(v, pct, color=colour(role, dark), lw=1.8, label=label)

    if mefs is not None:
        ax.axvline(mefs, color=p["muted"], ls=":", lw=1.0)
        ax.text(mefs, 101, "MEFS", ha="center", va="bottom", fontsize=7.5, color=p["text_secondary"])

    ax.set_xlim(left=0)
    ax.set_ylim(0, 105)
    ax.set_xlabel("Recoverable resource (MMboe)")
    ax.set_ylabel("Probability of exceedance (%)")
    ax.set_title("A5 · Exceedance curves at the chosen location")
    ax.grid(True, lw=0.6, alpha=0.7)
    ax.legend(loc="upper right", fontsize=7.5)
    fig.tight_layout()
    return fig, ax


def fig_a6_overlap(
    vc: VolumeClasses, groups: Groups, *, mefs: float | None = None, dark: bool = False, bins: int = 40,
):
    """Schneider et al.'s "surprising overlap" -- proven vs attic, on this prospect.

    Overlapping density histograms of proven-at-the-well (discovery trials)
    against attic (dry-hole trials): a dry hole's up-dip volume and a
    discovery's proven volume draw from distributions that overlap far more
    than intuition expects. No depth on either axis -- like A5, this is about
    the resource distribution itself, not where it sits structurally.
    """
    proven = vc.proven[groups.discovery]
    attic = vc.attic[groups.dry_with_attic]
    fig, ax = new_figure(figsize=(6, 5), dark=dark)
    p = palette(dark)

    hi = max(float(proven.max()) if proven.size else 0.0, float(attic.max()) if attic.size else 0.0, 1.0)
    edges = np.linspace(0.0, hi, bins + 1)
    if attic.size:
        ax.hist(attic, bins=edges, color=colour("attic", dark), alpha=0.55, density=True, label="Attic | dry hole")
    if proven.size:
        ax.hist(proven, bins=edges, color=colour("proven", dark), alpha=0.55, density=True, label="Proven | discovery")

    if mefs is not None:
        ax.axvline(mefs, color=p["muted"], ls=":", lw=1.0)
        ax.text(mefs, ax.get_ylim()[1], "MEFS", ha="center", va="bottom", fontsize=7.5, color=p["text_secondary"])

    ax.set_xlim(left=0)
    ax.set_xlabel("Recoverable resource (MMboe)")
    ax.set_ylabel("Density")
    ax.set_title("A6 · Attic vs proven — the overlap")
    ax.legend(loc="upper right", fontsize=7.5)
    fig.tight_layout()
    return fig, ax


def fig_b0_section(
    ad: AreaDepth, *, z_entry: float, z_exit: float, dark: bool = False, title: str = "B0 · Schematic section",
):
    """A schematic cross-section built from A(z), colour-keyed to the well's outcomes.

    Not a plot of any one trial -- a cartoon of the geometry the decision is
    made against. Width is proportional to the square root of the recovered
    enclosed area (a circular-closure proxy; the shape is illustrative, not a
    map), mirrored about the well. Above the entry is attic-if-dry; the
    entry-to-exit interval is what the well proves; below the exit is
    possible-but-unproven. Reused, unmodified, as the "live section" in the
    Well location tab.
    """
    fig, ax = new_figure(figsize=(4, 6), dark=dark)
    p = palette(dark)

    halfwidth = np.sqrt(np.maximum(ad.a, 0.0))
    z = ad.z

    def band(lo: float, hi: float, role: str, label: str) -> None:
        m = (z >= lo) & (z <= hi)
        if m.sum() < 2:
            return
        ax.fill_betweenx(z[m], -halfwidth[m], halfwidth[m], color=colour(role, dark), alpha=0.55, lw=0)
        # Direct in-axes labels rather than a legend: four series or fewer, and
        # the whole point of a section is that each band sits where it belongs,
        # so naming it in place is what makes the colour key readable.
        ax.text(
            0.0, 0.5 * (max(lo, ad.shallowest) + min(hi, ad.deepest)), label,
            ha="center", va="center", fontsize=7.5, color=p["text"], zorder=6,
        )

    band(ad.shallowest, z_entry, "attic", "attic if dry")
    band(z_entry, z_exit, "proven", "proven")
    band(z_exit, ad.deepest, "possible", "possible\nbelow exit")

    ax.plot(halfwidth, z, color=p["text_secondary"], lw=1.0)
    ax.plot(-halfwidth, z, color=p["text_secondary"], lw=1.0)
    ax.plot([0, 0], [z_entry, z_exit], color=p["well"], lw=4.5, solid_capstyle="butt", label="Well", zorder=5)

    depth_axis(ax, zlim=(ad.shallowest, ad.deepest))
    # No unit on the axis: the plotted half-width is sqrt(area), which is
    # neither a radius nor a diameter of anything, so calling it km would be a
    # unit claim the number does not support. The shape is a cartoon; the depths
    # on y are the real quantity.
    ax.set_xlabel("Schematic width (∝ √area) — not to scale")
    ax.set_xticks([])
    ax.set_title(title)
    fig.tight_layout()
    return fig, ax


def fig_b1_volume_split(vsweep: VolumeSweep, *, current_z: float | None = None, dark: bool = False):
    """Mean proven / possible / attic volume vs entry depth.

    The Schneider Fig. 7/11/12 equivalent: as the well moves down-dip, mean
    proven volume grows and mean attic shrinks, with possible-below-exit as
    the band the well leaves untested at any given location.
    """
    fig, ax = new_figure(figsize=(6, 5), dark=dark)
    p = palette(dark)

    ax.plot(vsweep.proven_mean, vsweep.z, color=colour("proven", dark), lw=2.0, label="Proven | discovery")
    ax.plot(vsweep.possible_mean, vsweep.z, color=colour("possible", dark), lw=1.6, ls="--",
            label="Possible below exit | discovery")
    ax.plot(vsweep.attic_mean, vsweep.z, color=colour("attic", dark), lw=2.0, label="Attic | dry hole")

    if current_z is not None and vsweep.z.min() <= current_z <= vsweep.z.max():
        ax.axhline(current_z, color=p["text_secondary"], ls="--", lw=1.0)

    depth_axis(ax, zlim=(float(vsweep.z.min()), float(vsweep.z.max())))
    ax.set_xlim(left=0)
    ax.set_xlabel("Mean resource (MMboe)")
    ax.set_title(f"B1 · Volume split vs location (exit = entry + {vsweep.z_gap:.0f} m)")
    ax.legend(loc="upper right", fontsize=7.5)
    fig.tight_layout()
    return fig, ax


def fig_b2_chance_vs_regret(vsweep: VolumeSweep, *, current_z: float | None = None, dark: bool = False):
    """Chance vs regret vs entry depth -- the most decision-relevant plot in the tool.

    ``P_well``, the chance a discovery proves more than MEFS, and the chance a
    dry hole would have left more than MEFS in the attic: the crossings between
    these three curves are the argument for where to place the well. Requires
    ``vsweep`` to have been run with a MEFS threshold.

    Both conditional curves are stated with their conditioning spelled out,
    because the regret curve conditions on the well being dry **and the
    prospect charged**, not merely on the well being dry. The chance failures
    are indistinguishable from a dry hole at the bore, and folding them in
    roughly halves the number here. That is the same distinction the design
    plan draws for the up-dip mean (5.08 vs 9.09 MMboe): both are legitimate
    and they answer different questions, so neither may be labelled ambiguously.
    """
    if vsweep.mefs is None or vsweep.p_proven_exceeds_mefs is None or vsweep.p_attic_exceeds_mefs is None:
        raise ValueError("fig_b2_chance_vs_regret needs a VolumeSweep run with a mefs threshold")

    fig, ax = new_figure(figsize=(6, 5), dark=dark)
    p = palette(dark)

    ax.plot(vsweep.p_well * 100.0, vsweep.z, color=colour("p_well", dark), lw=2.0, label=r"$P_{well}$")
    ax.plot(vsweep.p_proven_exceeds_mefs * 100.0, vsweep.z, color=colour("proven", dark), lw=1.8,
            label="P(proven > MEFS | discovery)")
    ax.plot(vsweep.p_attic_exceeds_mefs * 100.0, vsweep.z, color=colour("attic", dark), lw=1.8,
            label="P(attic > MEFS | dry & charged)")

    if current_z is not None and vsweep.z.min() <= current_z <= vsweep.z.max():
        ax.axhline(current_z, color=p["text_secondary"], ls="--", lw=1.0)

    ax.set_xlim(0, 100)
    depth_axis(ax, zlim=(float(vsweep.z.min()), float(vsweep.z.max())))
    ax.set_xlabel("Probability (%)")
    ax.set_title(f"B2 · Chance vs regret (MEFS {vsweep.mefs:.1f} MMboe)")
    ax.legend(loc="upper right", fontsize=7.5)
    fig.tight_layout()
    return fig, ax


def fig_b3_uncertainty_reduction(sweep: Sweep, *, current_z: float | None = None, dark: bool = False):
    """Haskett (2003) uncertainty-reduction curve vs entry depth, optimum marked.

    ``sweep.z_optimum`` is found by argmax over the sweep grid rather than
    eyeballed -- see :func:`wellvolpos.core.sweep.run_sweep`.
    """
    fig, ax = new_figure(figsize=(6, 5), dark=dark)
    p = palette(dark)
    c = colour("p_well", dark)

    ax.fill_betweenx(sweep.z, 0, sweep.uncertainty_reduction, color=c, alpha=0.15)
    ax.plot(sweep.uncertainty_reduction, sweep.z, color=c, lw=2.0)
    ax.plot([sweep.reduction_optimum], [sweep.z_optimum], "o", color=p["text"], zorder=5)
    ax.annotate(
        f"max {sweep.reduction_optimum:.0f}% @ {sweep.z_optimum:.0f} m",
        (sweep.reduction_optimum, sweep.z_optimum), xytext=(-8, 8),
        textcoords="offset points", ha="right", fontsize=8, color=p["text"],
    )
    if current_z is not None and sweep.z.min() <= current_z <= sweep.z.max():
        ax.axhline(current_z, color=p["text_secondary"], ls="--", lw=1.0)

    depth_axis(ax, zlim=(float(sweep.z.min()), float(sweep.z.max())))
    top = float(np.nanmax(sweep.uncertainty_reduction)) if np.isfinite(sweep.uncertainty_reduction).any() else 5.0
    ax.set_xlim(0, max(5.0, top * 1.15))
    ax.set_xlabel("Expected uncertainty reduction (%)")
    ax.set_title("B3 · Uncertainty reduction vs location (Haskett 2003)")
    fig.tight_layout()
    return fig, ax


def fig_b4_chance_waterfall(
    elements: dict[str, float],
    r: float,
    pos_prospect: float,
    *,
    scheme: str | dict[str, float] = "none",
    dark: bool = False,
):
    """The chance elements then the location factor, as a log-scale waterfall.

    Chance factors are multiplicative, so the natural picture is a running
    product on a log axis -- bar length *is* the risk each step contributes.
    The steps come from :func:`wellvolpos.core.chance.waterfall_steps`, whose
    factors multiply to ``pos_prospect * r`` exactly, so this figure cannot
    total something other than the ``P_well`` shown elsewhere in the app.

    Colour follows meaning, and the location steps are the interesting case.
    They stay the chance/discovery blue, because ``r`` *is* a chance and A3
    already draws it in blue -- giving it a second colour here would break the
    one mapping a reader is meant to learn once. They are instead separated by
    **hatching**, so that under an allocating scheme you can still see how much
    of each element's bar is geological chance and how much is the location
    penalty it has been made to carry. Under the default "none" scheme the
    location factor is one hatched bar of its own, which is exactly Milkov's
    "report r separately". The reconciliation step, when present, is neither a
    chance nor a location term and takes the neutral muted grey.
    """
    steps = chance_waterfall_steps(elements, r, pos_prospect, scheme)
    labels = [s[0] for s in steps]
    values = [s[1] for s in steps]
    roles = [s[2] for s in steps]

    fig, ax = new_figure(figsize=(6.5, 5), dark=dark)
    p = palette(dark)
    c = colour("p_well", dark)

    cum = 1.0
    tops, bottoms = [], []
    for v in values:
        bottoms.append(cum)
        cum *= v
        tops.append(cum)
    total = cum

    # A well at or below the deepest sampled contact has r = 0, so the running
    # product reaches exactly zero and there is no bottom to a log axis. Say so
    # rather than drawing a misleading stub.
    if total <= 0.0:
        ax.text(0.5, 0.5, "r = 0 at this depth\nP_well = 0", transform=ax.transAxes,
                ha="center", va="center", fontsize=9, color=p["text"])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title("B4 · Chance waterfall")
        fig.tight_layout()
        return fig, ax

    x = np.arange(len(labels))
    for xi, (b, t, role) in enumerate(zip(bottoms, tops, roles)):
        if role == "reconcile":
            face, hatch = p["muted"], None
        elif role == "location":
            face, hatch = c, "///"
        else:
            face, hatch = c, None
        ax.bar(xi, abs(b - t), bottom=min(b, t), color=face, width=0.6,
               hatch=hatch, edgecolor=p["surface"] if hatch else "none", linewidth=0.0)
    for xi, (b, v) in enumerate(zip(bottoms, values)):
        ax.text(xi, b, f"×{v:.3f}", ha="center", va="bottom", fontsize=7.5, color=p["text_secondary"])

    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=7.5)
    ax.axhline(total, color=p["muted"], ls=":", lw=1.0)
    ax.text(-0.45, total, f"$P_{{well}}$ = {total:.4f}", ha="left", va="bottom",
            fontsize=8, color=p["text"])
    if all(abs(float(elements.get(e, 1.0)) - 1.0) < 1e-12 for e in ELEMENTS):
        ax.text(
            0.5, 0.02, "Chance table is at 1.0 throughout, so the element steps have no height.",
            transform=ax.transAxes, ha="center", va="bottom", fontsize=7,
            color=p["text_secondary"],
        )
    ax.set_ylabel("Cumulative chance (log scale)")
    label = SCHEME_LABELS.get(scheme, "custom weights") if isinstance(scheme, str) else "custom weights"
    ax.set_title(f"B4 · Chance waterfall ({label})")
    fig.tight_layout()
    return fig, ax


def fig_b5_allocation_dumbbell(
    elements: dict[str, float], r: float, *, pos_prospect: float | None = None, dark: bool = False,
):
    """Three schemes against the prospect baseline, one row per chance element.

    Every scheme gives the same ``P_well`` (:func:`wellvolpos.core.chance.allocate`
    proves it, and ``tests/test_chance.py`` locks it) -- only the attribution
    across elements differs, which is exactly what three panels sharing one
    x-axis are for. When ``pos_prospect`` is given, that shared ``P_well`` is
    drawn as a rule on all three panels, so the claim is visible in the figure
    rather than only asserted in its caption.

    Reservoir is exempt under every shipped scheme, so its baseline and "at the
    well" markers coincide and no arrow is visible; that falls out of the
    weights rather than being drawn specially. The "none" panel shows no
    movement at all, which is the whole content of Milkov's position -- so it is
    annotated with where the location factor went, rather than left looking
    like a panel that failed to draw.
    """
    schemes = list(SHIPPED_SCHEMES)
    fig, axes = new_figure(1, len(schemes), figsize=(3.4 * len(schemes), 4.2), dark=dark, sharey=True)
    axes = np.atleast_1d(axes)
    p = palette(dark)
    c = colour("p_well", dark)
    y = np.arange(len(ELEMENTS))

    for i, (ax, scheme) in enumerate(zip(axes, schemes)):
        revised, _ = allocate(elements, r, scheme)
        base = [float(elements.get(e, 1.0)) for e in ELEMENTS]
        rev = [revised[e] for e in ELEMENTS]
        for yi, (b, rv) in enumerate(zip(base, rev)):
            ax.plot([b, rv], [yi, yi], color=c, lw=1.5, zorder=1)
        ax.scatter(base, y, s=28, facecolor=p["surface"], edgecolor=p["muted"], zorder=2,
                   label="Baseline")
        ax.scatter(rev, y, s=28, color=c, zorder=3, label="At the well")
        if pos_prospect is not None:
            ax.axvline(pos_prospect * r, color=p["muted"], ls=":", lw=1.0)
            if i == 0:
                ax.text(pos_prospect * r, len(ELEMENTS) - 0.5, r" $P_{well}$",
                        fontsize=7, color=p["text_secondary"], va="top")
        if scheme == "none":
            ax.text(0.5, -0.6, f"r = {r:.3f} reported separately", ha="center", va="center",
                    fontsize=7, color=p["text_secondary"])
        ax.set_xlim(0, 1.02)
        ax.set_ylim(-0.9, len(ELEMENTS) - 0.4)
        ax.set_xlabel("Chance")
        ax.set_title(SCHEME_LABELS.get(scheme, scheme), fontsize=8)
        ax.grid(True, axis="x", lw=0.6, alpha=0.6)

    axes[0].set_yticks(y)
    axes[0].set_yticklabels([e.capitalize() for e in ELEMENTS])
    axes[0].legend(loc="lower right", fontsize=7)
    fig.suptitle("B5 · Allocation dumbbell", fontsize=9.5, fontweight="bold", color=p["text"])
    fig.tight_layout()
    return fig, axes
