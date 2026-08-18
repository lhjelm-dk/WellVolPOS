"""Reference-engine and extension figures: A1-A6, B0-B5.

Every figure here is built on :func:`wellvolpos.viz.theme.new_figure` so
palette and rcParams can never drift between figures, and every axis that
carries a depth goes through :func:`wellvolpos.viz.theme.depth_axis`. A5, A6,
B4 and B5 are the figures in this set with no depth on either axis (all four
are about chance or the resource distribution, not where either sits
structurally), so they are the ones that do not call it -- see
``tests/test_axes.py`` and ``tests/test_figures.py``.

Colour is assigned by the **volume concept**, per ``theme.ROLES`` -- prospect
navy, well-associated olive, tested-by-well mauve, possible-below-exit pale
khaki, up-dip light blue, a threshold volume red. A1's and A4's mean curves take
the prospect navy because they characterise the whole un-cut model; A5's and
B1's series map onto the concepts directly; A2 and B0 colour-key the same
outcomes; and a *chance* takes the colour of the volume it belongs to, so A3's
and B2's ``P_well`` and B4's whole waterfall are olive -- the chance of the
well-associated case -- while ``POS_prospect`` is navy. A3's two curves are
separated by line style rather than colour, since both are chances of the same
family.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import NullFormatter

from ..core.chance import (
    ELEMENT_LABELS,
    ELEMENTS,
    SCHEME_LABELS,
    SHIPPED_SCHEMES,
    allocate,
    step_element,
)
from ..core.chance import waterfall_steps as chance_waterfall_steps
from ..core.mefs import c2_cases, c2_crossings
from ..core.summary import plateau_span
from ..core.classes import (
    READING_LABELS,
    VolumeClasses,
    conditional_exceedance,
    risked_exceedance,
)
from ..core.bands import BAND_MODE_LABELS, BandedPercentiles
from ..core.groups import Groups
from ..core.reservoir import thickness_from_pay
from ..core.stats import MIN_SUPPORT, thin
from ..core.structure import AreaDepth
from ..core.sweep import (
    entry_depth_percentiles,
    TARGET_STATISTIC_LABELS,
    Sweep,
    VolumeSweep,
    find_crossing,
    invert_volume_target,
    volume_target_band,
    volume_target_curve,
)
from ..io.adapters.base import TrialSet
from .theme import (
    FAN_POS_LEVELS,
    VOLUME_SCALES,
    concept_shades,
    element_colour,
    depth_shades,
    log_tick_text,
    log_ticks,
    probability_axis,
    probability_coords,
    AREA_SCALES,
    SEQUENTIAL_CMAP,
    VALUE_CMAP,
    colour,
    depth_axis,
    new_figure,
    palette,
    reference_label,
    OVERLAP_OPACITY,
    OVERLAP_BINS,
)

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
    "fig_b6_inverse",
    "fig_b7_frontier",
    "fig_b8_commercial_chance",
    "fig_b9_chance_weighted",
    "fig_b11_pos_sensitivity",
    "fig_a8_contact_distribution",
    "fig_a9_prospect_density",
    "exceedance_marks",
    "_depth_percentiles",
    "fig_colour_key",
    "fig_c1_section",
    "fig_c2_exceedance",
    "fig_c3_mefs_bars",
    "fig_map_view",
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


def _depth_percentiles(contact: np.ndarray, values: np.ndarray, n_bins: int = 40,
                       percentiles=(99, 90, 50, 10, 1)):
    """Equal-count depth bins, returning ``(z, {percentile: values, "mean": ...})``.

    The general form of :func:`_depth_band`, which returns a fixed P90/P50/mean/P10
    tuple. A4 needs P99 and P1 as well (Lars, 2026-08-11), and a dict keeps the
    caller from having to remember a five-tuple's order -- which is the kind of
    thing that silently swaps P90 and P10 and inverts a figure.

    Petroleum convention: **P99 is the low value**, exceeded 99 % of the time, so it
    is the 1st percentile of the values.
    """
    order = np.argsort(contact)
    c, v = np.asarray(contact)[order], np.asarray(values)[order]
    z: list[float] = []
    out: dict = {q: [] for q in percentiles}
    out["mean"] = []
    for idx in np.array_split(np.arange(c.size), min(n_bins, max(c.size, 1))):
        if idx.size == 0:
            continue
        z.append(float(c[idx].mean()))
        for q in percentiles:
            out[q].append(float(np.percentile(v[idx], 100 - q)))
        out["mean"].append(float(v[idx].mean()))
    return np.array(z), {k: np.array(val) for k, val in out.items()}


def _depth_band(contact: np.ndarray, values: np.ndarray, n_bins: int = 40):
    """Equal-count depth bins, returning (z, p90, p50, mean, p10) of ``values``.

    The percentile family A1 and A4 both draw. Petroleum convention: P90 is the
    low value, P10 the high one. The mean is carried alongside because it is the
    number that gets quoted, and it is not the P50 -- on a skewed resource
    distribution they can differ by a lot, and seeing both on one panel is the
    point.
    """
    order = np.argsort(contact)
    c, v = np.asarray(contact)[order], np.asarray(values)[order]
    z, p90, p50, mean, p10 = [], [], [], [], []
    for idx in np.array_split(np.arange(c.size), min(n_bins, max(c.size, 1))):
        if idx.size == 0:
            continue
        z.append(float(c[idx].mean()))
        p90.append(float(np.percentile(v[idx], 10.0)))
        p50.append(float(np.percentile(v[idx], 50.0)))
        mean.append(float(v[idx].mean()))
        p10.append(float(np.percentile(v[idx], 90.0)))
    return (np.array(z), np.array(p90), np.array(p50), np.array(mean), np.array(p10))


def area_spread_is_material(ad: AreaDepth) -> tuple[bool, float]:
    """Is there real area uncertainty *at a fixed depth*, worth reading off A1?

    Returns ``(material, relative_residual)``, the residual scatter about the
    fitted A(z) as a fraction of the mean area.

    Measured from the isotonic residual, deliberately, and **not** from
    percentiles within depth bins. A binned P90-to-P10 spread mostly reflects how
    much *depth* each bin spans, so on the reference file it comes out at 20 % of
    the mean and looks like substantial area uncertainty -- while the fit says
    area is a deterministic function of contact depth to nine decimal places
    (R2 = 0.9999999987, residual SD 0.00004 km2). Reporting the binned figure
    would invent an uncertainty the model does not contain, which is the sort of
    thing this project exists to avoid.
    """
    mean_area = float(np.mean(ad.a)) if ad.a.size else 0.0
    if mean_area <= 0 or not np.isfinite(ad.resid_sd):
        return False, 0.0
    rel = float(ad.resid_sd) / mean_area
    return rel > 0.01, rel


def _exceedance(values: np.ndarray):
    """Sorted values and their exceedance probability P(X >= value), in %."""
    v = np.sort(np.asarray(values, dtype=float))
    n = v.size
    if n == 0:
        return v, np.array([])
    return v, 100.0 * (n - np.arange(n)) / n


def exceedance_marks(values, chance: float = 1.0) -> list[tuple[str, float, float]]:
    """(label, volume, exceedance %) for P90, P50, mean and P10.

    ``chance`` scales the *heights* so the markers land on the curve they belong
    to. That argument is the whole point of this signature: without it the markers
    were computed conditionally and drawn on an unconditional curve, so on the
    concepts figure they floated well above the line -- which is what Lars spotted.
    The volumes never move; only the heights do.

    The percentiles themselves are always taken on the **conditional (success
    case)** distribution, because that is where the industry defines them: "P90 is
    defined as 90% probability of exceeding the P90 estimated value" (Milkov 2021).
    So P90 is the *low* case and P10 the high one, which is the opposite of what
    the numbers look like and the reason these markers are worth drawing at all.

    The **mean** is not a percentile. It is placed by asking the curve where it
    falls, and on a right-skewed resource distribution that is well above the P50 --
    near P40 on the demo data. Seeing that gap is the point: the mean is the number
    that gets quoted and it is not the middle.

    Returns an empty list for an empty distribution, so no caller needs to guard.
    """
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return []
    stats = [
        ("P90", float(np.percentile(v, 10.0))),
        ("P50", float(np.percentile(v, 50.0))),
        ("Mean", float(np.mean(v))),
        ("P10", float(np.percentile(v, 90.0))),
    ]
    out = []
    for label, value in stats:
        if not np.isfinite(value):
            continue
        # Read off the empirical curve rather than assuming 90/50/10: with ties or
        # a truncated distribution the nominal percentile and the plotted height
        # are not the same number, and the marker has to sit *on* the line.
        pct = 100.0 * float(chance) * float(np.mean(v >= value))
        out.append((label, value, pct))
    return out


def _mark_exceedance_mpl(ax, values, role: str, dark: bool, *, chance: float = 1.0,
                         show_text: bool = True, size: float = 5.0, ha: str = "left"):
    """The export-path twin of ``interactive._mark_exceedance``.

    Labelled with the volume rather than the percentile name, for the same reason:
    the percentile is already the axis. The mean gets a square because it is not a
    percentile, and on a right-skewed distribution it sits visibly above the P50.
    """
    marks = exceedance_marks(values, chance)
    if not marks:
        return
    p = palette(dark)
    for label, value, pct in marks:
        ax.plot([value], [pct], marker="s" if label == "Mean" else "D",
                ms=size + (0.5 if label == "Mean" else 0.0),
                mfc=colour(role, dark), mec=p["surface"], mew=0.8, ls="none", zorder=5)
        if show_text:
            label = f" {value:,.1f}" if ha == "left" else f"{value:,.1f} "
            ax.annotate(label, (value, pct), fontsize=6.5,
                        color=p["text_secondary"], va="center", ha=ha, zorder=5)


def fig_a1_area_depth(
    ad: AreaDepth, *, ts: TrialSet | None = None, current_entry: float | None = None,
    current_exit: float | None = None, n_bins: int = 40, area_scale: str = "area",
    show_reservoir: bool = True, show_classes: bool = True, dark: bool = False,
):
    """The area-depth curve recovered from the trials, entry/exit marked.

    The structural spine of the whole tool -- A(z) is what turns a well's
    depth into its position on the structure, and every figure that splits a
    trial at the well rests on this curve. Uses the prospect aqua because it
    characterises the whole closure, not any one outcome; the well itself
    gets its own dedicated colour so it reads as the thing being placed
    against the curve, not a feature of it.
    """
    fig, ax = new_figure(figsize=(6.5, 5.5), dark=dark)
    p = palette(dark)

    # **The area family, which the export had been missing entirely** (found by
    # audit, 2026-08-11). Given ``ts`` the plotly original draws P90 / P50 / mean /
    # P10 of the area in each depth bin -- the thing Lars asked for -- and the export
    # drew a bare A(z) instead. So the figure that goes into a well proposal was not
    # the figure on screen, and nothing said so.
    #
    # Thin grey for the percentiles, prospect colour and weight for the mean: the
    # mean is the number that gets quoted, and on a skewed distribution it is not
    # the P50.
    subtitle = ""
    with_area = ts is not None and ts.has("area")
    if with_area:
        contact, area = ts.col("contact"), ts.col("area")
        ok = area > 0
        zb, a90, a50, amean, a10 = _depth_band(contact[ok], area[ok], n_bins=n_bins)
        material, rel_resid = area_spread_is_material(ad)
        for values, name in ((a90, "P90"), (a50, "P50"), (a10, "P10")):
            ax.plot(values, zb, color=p["muted"], lw=1.0, label=name)
        ax.plot(amean, zb, color=colour("prospect", dark), lw=2.5, label="Mean area")
        subtitle = (
            f"area scatter about A(z) is {rel_resid:.1%} of the mean — real area uncertainty"
            if material else
            "area is a deterministic function of contact depth here, so the P90–P10 "
            "spread shown is the depth range within each bin, not area uncertainty"
        )
    else:
        ax.plot(ad.a, ad.z, color=colour("prospect", dark), lw=2.5, label="A(z)")

    if current_entry is not None:
        ax.axhline(current_entry, color=p["well"], ls="--", lw=1.4, label="well entry")
    if current_exit is not None and current_exit != current_entry:
        ax.axhline(current_exit, color=p["well"], ls=":", lw=1.4, label="well exit")

    # C1's reservoir section, merged in (Lars, 2026-08-11): A1 and C1 drew the same
    # A(z), and the only thing C1 added was the base reservoir and the three shaded
    # classes. The base reservoir carries its own P90/P50/mean/P10, because the
    # thickness recovered from pay is a distribution.
    stats = None
    if show_reservoir and ts is not None:
        stats = _reservoir_section_mpl(
            ax, ad, ts, area_scale=area_scale,
            z_entry=current_entry if current_entry is not None else ad.shallowest,
            z_exit=current_exit if current_exit is not None else ad.deepest,
            dark=dark, show_classes=show_classes,
        )

    depth_axis(ax, zlim=(ad.shallowest, ad.deepest))
    ax.set_xlim(left=0)
    ax.set_xlabel(AREA_SCALES.get(area_scale, AREA_SCALES["area"])[0])
    # The same two-line title the plotly original carries. The caveat about what the
    # P90-P10 spread means on a deterministic A(z) is the part that must not be lost
    # in export: without it a reader takes the grey band for area uncertainty.
    note = ""
    if stats is not None:
        note = (f"base reservoir = top + thickness from pay: P90 {stats['p90']:.0f} · "
                f"P50 {stats['p50']:.0f} · mean {stats['mean']:.0f} · P10 {stats['p10']:.0f} m")
    lines = [f"A1 · Area–depth curve and reservoir (isotonic R² = {ad.r2:.6f})"]
    lines += [t for t in (subtitle, note) if t]
    ax.set_title("\n".join(lines), fontsize=9)
    if ax.get_legend_handles_labels()[1]:
        ax.legend(loc="lower right", fontsize=6.5, ncol=2)
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
    ax.fill_betweenx(sweep.z, cum1, cum2, color=colour("tested", dark), label="Discovery, contact seen")
    ax.fill_betweenx(sweep.z, cum2, cum3, color=colour("below_lkh", dark), label="Discovery, HC to exit")

    if current_z is not None and sweep.z.min() <= current_z <= sweep.z.max():
        ax.axhline(current_z, color=p["text"], ls="--", lw=1.0)

    depth_axis(ax, zlim=(float(sweep.z.min()), float(sweep.z.max())))
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of trials (%)")
    ax.set_title(f"A2 · Outcome tree vs location "
                 f"(exit = entry + {sweep.z_gap:.0f} m, from the well input)")
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
    is the figure that makes the decomposition in the "one idea
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

    ax.plot(sweep.p_well * 100.0, sweep.z, color=c, lw=2.0, label="P_well = POS × r")
    ax.plot(sweep.r_location * 100.0, sweep.z, color=c, lw=1.4, ls="--",
            label="r = P(contact deeper | HC)")

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
    ax.set_title(f"A3 · Chance decomposition vs location ({reference_label(sweep.reference)})")
    ax.legend(loc="upper right", fontsize=7.5)
    fig.tight_layout()
    return fig, ax


def fig_a4_resource_vs_depth(
    ts: TrialSet, *, current_entry: float | None = None, current_exit: float | None = None,
    mefs: float | None = None, render: str = "grid", n_bins: int = 40,
    n_resource: int | None = None, n_depth: int | None = None,
    zlim: tuple[float, float] | None = None, dark: bool = False,
):
    """A4 for the export path. Twin of ``pfig_a4_resource_vs_depth``.

    Two renderings of one dataset: ``"grid"`` counts trials per cell in inferno, the
    workbook's rendering with a Freedman-Diaconis default; ``"hexbin"`` is the
    original blue log-density. Over either, the conditional percentile family
    P99/P90/P50/P10/P1 and the mean.

    Success trials only -- the chance-failure zeros belong to POS, not to the shape
    of the resource distribution conditional on a contact being sampled.
    """
    from matplotlib.colors import LogNorm

    from .interactive import suggest_grid

    if render not in ("grid", "hexbin"):
        raise ValueError(f"unknown render {render!r}; expected 'grid' or 'hexbin'")
    res, contact = ts.col("resource"), ts.col("contact")
    succ = res > 0.0
    x, y = res[succ], contact[succ]
    fig, ax = new_figure(figsize=(6.4, 5.2), dark=dark)
    p = palette(dark)

    if render == "grid":
        auto_r, auto_z = suggest_grid(x, y)
        nx, ny = int(n_resource or auto_r), int(n_depth or auto_z)
        counts, xe, ye = np.histogram2d(x, y, bins=(nx, ny))
        shown = np.ma.masked_where(counts.T <= 0, counts.T)
        mesh = ax.pcolormesh(xe, ye, shown, cmap="inferno",
                             norm=LogNorm(vmin=1, vmax=max(2.0, float(counts.max()))))
        fig.colorbar(mesh, ax=ax, label="trials per cell (log)")
        label = f"{nx} × {ny} grid"
    else:
        ax.hexbin(x, y, gridsize=45, cmap=SEQUENTIAL_CMAP, mincnt=1, bins="log")
        label = "log-density"

    z, band = _depth_percentiles(y, x, n_bins=n_bins)
    c = colour("prospect", dark)
    for q, ls, lw in ((99, ":", 0.9), (90, "--", 1.0), (50, "-", 1.8),
                      (10, "--", 1.0), (1, ":", 0.9)):
        ax.plot(band[q], z, color=c, lw=lw, ls=ls, label=f"P{q}")
    ax.plot(band["mean"], z, color=c, lw=2.2, label="Mean")
    if current_exit is not None:
        ax.axhline(current_exit, color=p["well"], ls=":", lw=1.2)

    if current_entry is not None:
        ax.axhline(current_entry, color=p["text_secondary"], ls="--", lw=1.0)
    if mefs is not None:
        ax.axvline(mefs, color=p["muted"], ls=":", lw=1.0)

    # zlim so a row can be given one shared depth range; without it A4
    # autoscales to its own hexbin extent and stops lining up with A1/A5.
    depth_axis(ax, ylabel="HC-water contact (m TVDSS)", zlim=zlim)
    ax.set_xlim(left=0)
    ax.set_xlabel("Recoverable resource (MMboe)")
    ax.set_title(f"A4 · Resource vs contact depth ({label})")
    ax.legend(loc="lower right", fontsize=6.5, ncol=2)
    fig.tight_layout()
    return fig, ax


def fig_a5_exceedance(
    ts: TrialSet, groups: Groups, vc: VolumeClasses, *, mefs: float | None = None,
    pos_prospect: float | None = None, p_well: float | None = None, dark: bool = False,
):
    """Exceedance curves for prospect / discovery / proven / attic, at the chosen location.

    The money chart. No depth on either axis -- see the module docstring --
    so the four canonical colour roles map directly onto the four series.
    """
    res = ts.col("resource")
    fig, ax = new_figure(figsize=(6, 5), dark=dark)
    p = palette(dark)

    # Both readings, like the plotly twin: solid conditional, dashed unconditional,
    # each series risked by *its own* chance.
    # Prospect only, like the plotly twin: the other three series live in C2 and in
    # tab 3's table, and three places for one set of numbers is three places to
    # disagree. Their populations were verified identical before removing them here.
    series = [("Prospect recoverable resource", res[res > 0], pos_prospect, "prospect")]
    for label, values, chance_of, role in series:
        readings = [("conditional", 1.0)]
        if chance_of is not None:
            readings.append(("unconditional", float(chance_of)))
        for reading, chance_used in readings:
            v, pct = risked_exceedance(values, chance_used)
            if v.size == 0:
                continue
            ax.plot(v, pct, color=colour(role, dark),
                    lw=2.2 if reading == "conditional" else 1.6,
                    ls="-" if reading == "conditional" else "--",
                    label=label if reading == "conditional" else f"{label} — risked")
            _mark_exceedance_mpl(ax, values, role, dark, chance=chance_used, show_text=True)
            # Where the curve crosses MEFS -- see the plotly twin. The threshold line
            # says where the bar is; this says what clearing it costs.
            if mefs is not None:
                y_at = float(np.interp(float(mefs), v, pct))
                ax.plot([float(mefs)], [y_at], marker="x", ms=7, mew=2.0,
                        color=colour(role, dark), zorder=5)
                ax.annotate(f"{y_at:.1f}% > MEFS", (float(mefs), y_at),
                            textcoords="offset points", xytext=(6, 0), va="center",
                            fontsize=7.5, color=colour(role, dark))

    if mefs is not None:
        ax.axvline(mefs, color=p["muted"], ls=":", lw=1.0)
        ax.text(mefs, 101, "MEFS", ha="center", va="bottom", fontsize=7.5, color=p["text_secondary"])

    ax.set_xlim(left=0)
    ax.set_ylim(0, 105)
    ax.set_xlabel("Recoverable resource (MMboe)")
    ax.set_ylabel("Probability of exceedance (%)")
    ax.set_title("A5 · Prospect resource — solid conditional, dashed risked")
    ax.grid(True, lw=0.6, alpha=0.7)
    ax.legend(loc="upper right", fontsize=7.5)
    fig.tight_layout()
    return fig, ax


def fig_a6_overlap(
    vc: VolumeClasses, groups: Groups, *, ts: TrialSet | None = None,
    mefs: float | None = None, dark: bool = False, bins: int = OVERLAP_BINS,
    normalise: str = "density", show_exceedance: bool = False,
    opacity: float = OVERLAP_OPACITY,
):
    """A6 for the export path. Twin of ``pfig_a6_overlap``.

    All four classes as densities, so group size does not distort the comparison.
    Attic and proven are Schneider's pair; the two larger distributions they are
    carved out of sit behind them. Opacity is lower with four series than with two --
    the whole content of this figure is what shows through what.
    """
    p = palette(dark)
    fig, ax = new_figure(figsize=(6, 4.6), dark=dark)
    series = [
        ("Prospect resource potential",
         ts.col("resource")[ts.col("resource") > 0] if ts is not None else np.array([]),
         "prospect"),
        ("Well associated | discovery", vc.discovery_total[groups.discovery], "well_associated"),
        ("Attic | dry hole", vc.attic[groups.dry_with_attic], "attic"),
        ("Proven | discovery", vc.proven[groups.discovery], "proven"),
    ]
    # The commercial class -- see the plotly twin for why it belongs on this figure.
    if mefs is not None and ts is not None:
        _res = np.asarray(ts.col("resource"), dtype=float)
        _comm = _res[np.asarray(groups.discovery, dtype=bool) & (_res > float(mefs))]
        if _comm.size:
            series.append(("Commercial | clears MEFS", _comm, "commercial"))
    hi = max([float(v.max()) for _n, v, _r in series if v.size] + [1.0])
    edges = np.linspace(0.0, hi, bins + 1)
    if normalise not in ("density", "peak"):
        raise ValueError(f"unknown normalise {normalise!r}; expected 'density' or 'peak'")
    centres = 0.5 * (edges[:-1] + edges[1:])
    for name, values, role in series:
        if not values.size:
            continue
        counts, _ = np.histogram(values, bins=edges, density=True)
        if normalise == "peak":
            peak = float(counts.max())
            counts = counts / peak if peak > 0 else counts
        # Explicit bars, matching the plotly twin: histnorm has no peak option
        # there, so re-binning separately would make the two disagree about edges.
        ax.bar(centres, counts, width=float(edges[1] - edges[0]),
               color=colour(role, dark), alpha=float(opacity),
               label=f"{name} (n={values.size:,})")

    if show_exceedance:
        ax2 = ax.twinx()
        for name, values, role in series:
            if not values.size:
                continue
            v, pct = conditional_exceedance(values)
            ax2.plot(v, pct, color=colour(role, dark), lw=2.0,
                     label=f"{name} — P(exceed)")
        ax2.set_ylim(0, 105)
        ax2.set_ylabel("P(exceeding) — conditional (%)")

    if mefs is not None:
        ax.axvline(mefs, color=p["muted"], ls=":", lw=1.0)
    ax.set_xlabel("Recoverable resource (MMboe)")
    ax.set_ylabel("Density (area = 1 per class)" if normalise == "density"
                  else "Scaled to each class's own peak (not a density)")
    ax.set_title("A6 · Where the four volume classes overlap")
    ax.legend(loc="upper right", fontsize=6.5)
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
    band(z_exit, ad.deepest, "below_lkh", "unproven\nbelow LKH")

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


def fig_b1_volume_split(
    vsweep: VolumeSweep, *, current_z: float | None = None,
    min_support: int = MIN_SUPPORT, zlim: tuple[float, float] | None = None,
    dark: bool = False,
):
    """B1 for the export path. Twin of ``pfig_b1_volume_split``.

    Proven, attic and the at-the-well volume, each with a bold mean and a thin dotted
    P90 / P50 / P10 ladder. The volume below the reservoir exit is
    :func:`fig_b13_below_exit`, because it is conditional on a different event.
    """
    fig, ax = new_figure(figsize=(6, 5), dark=dark)
    p = palette(dark)

    families = [
        (vsweep.proven_mean, vsweep.proven_p90, vsweep.proven_p50, vsweep.proven_p10,
         vsweep.n_discovery, "Proven | discovery", "proven", "-"),
        (vsweep.attic_mean, vsweep.attic_p90, vsweep.attic_p50, vsweep.attic_p10,
         vsweep.n_dry, "Attic | dry hole", "attic", "-"),
    ]
    if vsweep.at_well_mean is not None:
        families.append(
            (vsweep.at_well_mean, vsweep.at_well_p90, vsweep.at_well_p50,
             vsweep.at_well_p10, vsweep.at_well_n,
             f"At the well (contact within ±{vsweep.at_well_window:g} m)", None, "-"))

    # **A faint P90-P10 wash behind each concept family** (Lars, 2026-08-18). Three
    # bold means with nine dotted lines around them is a lot of line, and the eye has
    # to assemble each distribution from its parts before it can compare any two. The
    # wash gives it the body of the distribution first.
    #
    # **P90 to P10 only, and deliberately very light.** A fill reads as *equally likely
    # anywhere inside it*, which is false -- so it is faint enough to be a backdrop
    # rather than a claim, the same argument that keeps B6's contact fill off the P99
    # and P1 extremes. Drawn before the lines, so every mean stays on top of it. The
    # at-the-well series has no role colour and gets no wash: it is a *seam* between
    # two classes rather than a class, so a filled body would overstate it.
    for mean, p90, p50, p10, counts, label, role, style in families:
        if p90 is None or p10 is None or not role:
            continue
        lo = thin(p90, counts, min_support)
        hi = thin(p10, counts, min_support)
        band = np.isfinite(lo) & np.isfinite(hi)
        if band.sum() < 2:
            continue
        _base, _, _cond = label.partition(" |")
        ax.fill_betweenx(vsweep.z[band], lo[band], hi[band],
                         color=colour(role, dark), alpha=0.10, lw=0,
                         label=(f"{_base} P90–P10 |{_cond}" if _cond
                                else f"{_base} P90–P10"))

    for mean, p90, p50, p10, counts, label, role, style in families:
        col = colour(role, dark) if role else p["muted"]
        ax.plot(thin(mean, counts, min_support), vsweep.z, color=col,
                lw=2.0 if role else 1.6, ls=style, label=label)
        for tag, arr in (("P90", p90), ("P50", p50), ("P10", p10)):
            if arr is None:
                continue
            # The full label, conditioning suffix included, so the twin-agreement
            # guard can compare series names between the backends.
            _base, _, _cond = label.partition(" |")
            ax.plot(thin(arr, counts, min_support), vsweep.z, color=col, lw=0.8,
                    ls=":",
                    label=(f"{_base} {tag} |{_cond}" if _cond else f"{_base} {tag}"))

    if vsweep.mefs is not None:
        ax.axvline(vsweep.mefs, color=colour("minimum", dark), lw=1.0, ls=":")
    if current_z is not None:
        ax.axhline(current_z, color=p["well"], lw=1.0, ls="--")

    ax.set_xlim(left=0)
    ax.set_xlabel("Mean volume (MMboe)")
    depth_axis(ax, zlim=zlim or (float(vsweep.z.min()), float(vsweep.z.max())))
    ax.set_title("B1 · Volume split vs location — bold mean, dotted P90/P50/P10, "
                 "wash P90–P10")
    ax.grid(True, lw=0.6, alpha=0.7)
    ax.legend(loc="lower right", fontsize=6)
    fig.tight_layout()
    return fig, ax


def fig_b13_below_exit(
    vsweep: VolumeSweep, *, current_z: float | None = None,
    min_support: int = MIN_SUPPORT, zlim: tuple[float, float] | None = None,
    dark: bool = False,
):
    """B13 for the export path. Twin of ``pfig_b13_below_exit``.

    Conditional on the well leaving the reservoir still in hydrocarbons, which is why
    it is not on B1: its curves are not on the same footing as proven's or the attic's.
    """
    fig, ax = new_figure(figsize=(6, 5), dark=dark)
    p = palette(dark)
    col = colour("below_lkh", dark)

    if vsweep.below_lkh_mean_if_any is None:
        ax.text(0.5, 0.5, "No unproven volume below LKH on this sweep", ha="center",
                va="center", transform=ax.transAxes, fontsize=9, color=p["text"])
        fig.tight_layout()
        return fig, ax

    ax.plot(thin(vsweep.below_lkh_mean_if_any, vsweep.n_discovery, min_support),
            vsweep.z, color=col, lw=2.0, label="Mean | HC seen to the exit")
    for tag, arr in (("P90", vsweep.below_lkh_p90_if_any),
                     ("P50", vsweep.below_lkh_p50_if_any),
                     ("P10", vsweep.below_lkh_p10_if_any)):
        if arr is not None:
            ax.plot(thin(arr, vsweep.n_discovery, min_support), vsweep.z, color=col,
                    lw=0.8, ls=":", label=f"{tag} | HC seen to the exit")
    if vsweep.mefs is not None:
        ax.axvline(vsweep.mefs, color=colour("minimum", dark), lw=1.0, ls=":")
    if current_z is not None:
        ax.axhline(current_z, color=p["well"], lw=1.0, ls="--")

    ax.set_xlim(left=0)
    ax.set_xlabel("Unproven volume below LKH (MMboe)")
    depth_axis(ax, zlim=zlim or (float(vsweep.z.min()), float(vsweep.z.max())))
    ax.set_title("B13 · Unproven below LKH — conditional on HC to the exit")
    ax.grid(True, lw=0.6, alpha=0.7)
    ax.legend(loc="lower right", fontsize=7)
    fig.tight_layout()
    return fig, ax


def fig_b2_chance_vs_regret(
    vsweep: VolumeSweep, *, current_z: float | None = None, min_support: int = MIN_SUPPORT,
    dark: bool = False,
):
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

    # P_well is unconditional -- it is a chance over all trials -- so it is
    # never thinned. The two conditional curves are.
    p_proven = thin(vsweep.p_proven_exceeds_mefs, vsweep.n_discovery, min_support)
    p_attic = thin(vsweep.p_attic_exceeds_mefs, vsweep.n_dry, min_support)

    ax.plot(vsweep.p_well * 100.0, vsweep.z, color=colour("p_well", dark), lw=2.0, label="P_well")
    ax.plot(p_proven * 100.0, vsweep.z, color=colour("proven", dark), lw=1.8,
            label="P(proven > MEFS | discovery)")
    ax.plot(p_attic * 100.0, vsweep.z, color=colour("attic", dark), lw=1.8,
            label="P(attic > MEFS | dry & charged)")
    # The mirror image of the proven curve: deepening the well moves volume from
    # possible into proven. See the plotly twin for why the geometric reading
    # (p_well_exits_in_hc) is deliberately not drawn beside these.
    if vsweep.p_below_lkh_exceeds_mefs is not None:
        p_possible = thin(vsweep.p_below_lkh_exceeds_mefs, vsweep.n_discovery, min_support)
        ax.plot(p_possible * 100.0, vsweep.z, color=colour("below_lkh", dark), lw=1.8,
                label="P(unproven below LKH > MEFS | discovery)")

    # Labelled for exactly what it is. It is tempting to call this "where
    # chance stops outweighing regret", and wrong: P_well is unconditional
    # while the regret curve is conditional on the well being dry *and* the
    # prospect charged, so the two are not on one scale. A properly risked
    # comparison would multiply the regret by P(dry & charged) and crosses
    # some 7 m deeper on the reference data. Naming the curves that meet is
    # the honest version of the same annotation.
    crossing = find_crossing(vsweep.z, vsweep.p_well, p_attic)
    if crossing is not None:
        ax.axhline(crossing, color=p["text"], ls=":", lw=1.2)
        ax.annotate(
            f"$P_{{well}}$ = P(attic > MEFS | dry & charged) at {crossing:.0f} m",
            (99, crossing), xytext=(0, 4),
            textcoords="offset points", ha="right", fontsize=7, color=p["text"],
        )
    if current_z is not None and vsweep.z.min() <= current_z <= vsweep.z.max():
        ax.axhline(current_z, color=p["text_secondary"], ls="--", lw=1.0)

    ax.set_xlim(0, 100)
    depth_axis(ax, zlim=(float(vsweep.z.min()), float(vsweep.z.max())))
    ax.set_xlabel("Probability (%)")
    ax.set_title(
        f"B2 · Chance vs regret (MEFS {vsweep.mefs:.1f} MMboe, "
        f"{reference_label(vsweep.reference)})"
    )
    ax.legend(loc="upper right", fontsize=7.5)
    fig.tight_layout()
    return fig, ax


def fig_b6_inverse(
    vsweep: VolumeSweep, *, target: float | None = None, n_targets: int = 40,
    ts: TrialSet | None = None, mefs: float | None = None, statistic: str = "mean",
    zlim: tuple[float, float] | None = None, dark: bool = False,
):
    """B6 -- the inverse, on one pair of axes.

    The export twin of :func:`wellvolpos.viz.interactive.pfig_b6_inverse`; that
    docstring carries the argument for why both axis titles name *two* quantities.
    Returns ``(fig, ax)``.
    """
    targets, z_req, p_at = volume_target_curve(vsweep, n=n_targets, ts=ts,
                                               statistic=statistic)
    stat_label = TARGET_STATISTIC_LABELS[statistic]
    fig, ax = new_figure(figsize=(7.0, 5.0), dark=dark)
    p = palette(dark)

    if targets.size == 0 or not np.isfinite(z_req).any():
        ax.text(0.5, 0.5, "No proven-volume curve to invert", transform=ax.transAxes,
                ha="center", va="center", fontsize=9, color=p["text"])
        ax.set_title("B6 \u00b7 Inverse \u2014 volume to prove")
        fig.tight_layout()
        return fig, ax

    ok = np.isfinite(z_req)
    if vsweep.alpha is not None:
        z_lo, z_hi = volume_target_band(vsweep, targets)
        band = np.isfinite(z_lo) & np.isfinite(z_hi)
        if band.any():
            ax.fill_between(
                targets[band], z_lo[band], z_hi[band],
                color=colour("p_well", dark), alpha=0.15, lw=0,
                label=(f"nominal {100 * (1 - vsweep.alpha):.0f}% CI on the {stat_label}"
                       " \u2014 sampling error"),
            )

    # The contact spread first, so the coloured markers sit on top of the grey.
    spread_depths = []
    if ts is not None:
        res_all = np.asarray(ts.col("resource"), dtype=float)
        res_all = res_all[res_all > 0]
        held = np.linspace(float(np.percentile(res_all, 5)),
                           float(np.percentile(res_all, 95)), int(n_targets))
        band_pct = entry_depth_percentiles(ts, held)
        # P90 to P10, not P99 to P1: the fill is the body of the distribution and
        # the extremes stay as thin lines outside it (Lars, 2026-08-11).
        lo, hi = band_pct[90], band_pct[10]
        inner = np.isfinite(lo) & np.isfinite(hi)
        if inner.any():
            ax.fill_between(held[inner], lo[inner], hi[inner], color=p["muted"],
                            alpha=0.14, lw=0,
                            label="P90\u2013P10 contact spread \u2014 geological range")
        for q, lw, ls in ((99, 0.9, ":"), (90, 1.2, "--"), (50, 1.8, "-"),
                          (10, 1.2, "--"), (1, 0.9, ":")):
            depths = band_pct[q]
            good = np.isfinite(depths)
            if good.sum() >= 2:
                spread_depths.append(depths[good])
                ax.plot(held[good], depths[good], color=p["muted"], lw=lw, ls=ls,
                        label=f"P{q} contact — of trials holding this volume")
        # The mean contact of the qualifying trials -- grey like the percentiles
        # because it shares their axes, dash-dot and named "mean" because it is not
        # one of them. See the plotly twin.
        mean_contact = band_pct.get("mean")
        if mean_contact is not None:
            good = np.isfinite(mean_contact)
            if good.sum() >= 2:
                spread_depths.append(mean_contact[good])
                ax.plot(held[good], mean_contact[good], color=p["muted"], lw=1.6,
                        ls="-.", label="Pmean contact — of trials holding this volume")
        if mefs is not None:
            ax.axvline(float(mefs), color=p["muted"], ls=":", lw=1.0)

    # Inferno for the markers and the well's violet for the line joining them --
    # see the plotly twin. The requirement and the contact family were two greys.
    sc = ax.scatter(targets[ok], z_req[ok], c=p_at[ok] * 100.0, cmap=VALUE_CMAP.lower(),
                    vmin=0, vmax=100, s=26, zorder=4, edgecolor=p["surface"], linewidth=0.4)
    ax.plot(targets[ok], z_req[ok], color=colour("well", dark), lw=1.8, zorder=3,
            label=f"Required entry \u2014 for a target {stat_label} volume")
    cb = fig.colorbar(sc, ax=ax, pad=0.02)
    cb.set_label(r"$P_{well}$ at that depth (%)", fontsize=8)

    # The other three statistics, thin and violet, marker-free -- see the plotly twin
    # for why markers are withheld here.
    for other in ("p90", "p50", "p10"):
        if other == statistic:
            continue
        try:
            o_targets, o_z, _ = volume_target_curve(vsweep, n=n_targets, ts=ts,
                                                    statistic=other)
        except ValueError:
            continue
        good = np.isfinite(o_z)
        if good.sum() >= 2:
            ax.plot(o_targets[good], o_z[good], color=colour("well", dark), lw=0.9,
                    ls="--" if other == "p50" else ":",
                    label=f"Required entry — {TARGET_STATISTIC_LABELS[other]}")

    if target is not None:
        res = invert_volume_target(vsweep, float(target), ts=ts)
        if res.achievable:
            # A right-angle leader, like the plotly twin: up the target volume,
            # across to the depth, so the figure shows how to read itself.
            ax.plot([target, target, float(np.nanmin(targets))],
                    [float(np.nanmax(z_req)), res.z_required, res.z_required],
                    color=p["text"], lw=1.1, ls=":", zorder=5)
            ax.annotate(f"want {target:.0f} MMboe", (target, float(np.nanmax(z_req))),
                        xytext=(-4, 2), textcoords="offset points", ha="right",
                        fontsize=8, color=p["text"])
            ax.plot([target], [res.z_required], "o", color=p["text"], zorder=6)
            ax.annotate(
                f"enter at {res.z_required:.0f} m\n$P_{{well}}$ {res.p_well_at:.1%}",
                (target, res.z_required), xytext=(6, 6), textcoords="offset points",
                fontsize=8, color=p["text"],
            )

    all_z = [z_req[ok], *spread_depths]
    all_z = np.concatenate([a for a in all_z if a.size])
    # Both readings named on each axis: one pair of axes carrying two definitions
    # of volume and two kinds of depth is only honest if the axis says so.
    depth_axis(ax, ylabel=("Depth (m TVDSS) \u2014 required entry, or deeper (curve) / "
                           "contact (grey)"),
               zlim=zlim or (float(all_z.min()), float(all_z.max())))
    ax.set_xlim(left=0)
    ax.set_xlabel(f"Volume (MMboe) \u2014 target {stat_label} (curve) / held by one trial (grey)")
    ax.set_title(f"B6 \u00b7 Inverse \u2014 how deep must the well go to prove a {stat_label} volume?")
    ax.grid(True, lw=0.6, alpha=0.7)
    if ax.get_legend_handles_labels()[1]:
        ax.legend(loc="lower right", fontsize=6.5)
    fig.tight_layout()
    return fig, ax


def fig_b3_uncertainty_reduction(sweep: Sweep, *, current_z: float | None = None,
                                 show_all_trials: bool = True,
                                 show_ranges: bool = True, dark: bool = False):
    """B3 for the export path. Twin of ``pfig_b3_uncertainty_reduction``.

    See that docstring: Haskett's *measure* on Haskett's own P10-P90 proxy, but not
    Haskett's setting -- his paper is about appraisal after a discovery, and the peak
    here is the most *informative* depth rather than the best one.

    Two curves, and the gap between them is the argument: solid over the **success
    cases**, which is the conditioning ``r_location`` already uses, and dotted grey
    over every trial. ``sweep.z_optimum`` is found by argmax over the sweep grid
    rather than eyeballed.
    """
    fig, ax = new_figure(figsize=(6, 5), dark=dark)
    p = palette(dark)
    c = colour("p_well", dark)

    if show_all_trials and getattr(sweep, "uncertainty_reduction_all", None) is not None:
        ax.plot(sweep.uncertainty_reduction_all, sweep.z, color=p["muted"], lw=1.3,
                ls=":", label="over every trial — includes the chance failures")
    ax.fill_betweenx(sweep.z, 0, sweep.uncertainty_reduction, color=c, alpha=0.15)
    ax.plot(sweep.uncertainty_reduction, sweep.z, color=c, lw=2.0,
            label="over the success cases")
    _ranges = getattr(sweep, "uncertainty_reduction_ranges", None) or {}
    if show_ranges and _ranges:
        _shades = concept_shades("tested", max(len(_ranges), 1), dark)
        for _shade, (_pair, _curve) in zip(_shades, sorted(_ranges.items(), reverse=True)):
            _lo, _hi = _pair
            ax.plot(_curve, sweep.z, color=_shade, lw=1.1, ls="--",
                    label=f"success cases, P{100 - _hi:.0f}–P{100 - _lo:.0f} range")
    ax.plot([sweep.reduction_optimum], [sweep.z_optimum], "o", color=p["text"], zorder=5)
    ax.annotate(
        f"max {sweep.reduction_optimum:.0f}% @ {sweep.z_optimum:.0f} m",
        (sweep.reduction_optimum, sweep.z_optimum), xytext=(-8, 8),
        textcoords="offset points", ha="right", fontsize=8, color=p["text"],
    )
    if current_z is not None and sweep.z.min() <= current_z <= sweep.z.max():
        ax.axhline(current_z, color=p["text_secondary"], ls="--", lw=1.0)

    depth_axis(ax, zlim=(float(sweep.z.min()), float(sweep.z.max())))
    _tops = [sweep.uncertainty_reduction]
    if getattr(sweep, "uncertainty_reduction_all", None) is not None:
        _tops.append(sweep.uncertainty_reduction_all)
    _tops.extend(_ranges.values())
    _all = np.concatenate([np.asarray(t, dtype=float) for t in _tops])
    top = float(np.nanmax(_all)) if np.isfinite(_all).any() else 5.0
    ax.set_xlim(0, max(5.0, top * 1.15))
    ax.set_xlabel("Expected reduction in the prospect's inter-percentile range (%)")
    handles, _ = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc="lower right", fontsize=7)
    ax.set_title("B3 · How much a well here would tell you — expected reduction "
                 "in the prospect's P10–P90 range")
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
        # Colour is the element, hatching is the location share -- see the plotly
        # twin. Steps that belong to no element stay neutral, which is also what
        # tells them apart from the ones that do.
        el = step_element(labels[xi])
        if role == "reconcile":
            face, hatch = p["muted"], None
        elif el is not None:
            face = element_colour(el, dark)
            hatch = "///" if role == "location" else None
        else:
            face, hatch = c, "///" if role == "location" else None
        ax.bar(xi, abs(b - t), bottom=min(b, t), color=face, width=0.6,
               hatch=hatch, edgecolor=p["surface"] if hatch else "none", linewidth=0.0)
    for xi, (b, v) in enumerate(zip(bottoms, values)):
        ax.text(xi, b, f"×{v:.3f}", ha="center", va="bottom", fontsize=7.5, color=p["text_secondary"])

    # Ceiling pinned at 1.2, like the plotly twin: a chance cannot exceed 1, and
    # autoscaling gave a different top on every chance table so two waterfalls could
    # not be compared by eye. The headroom is for the "x1.000" labels.
    ax.set_yscale("log")
    _floor = float(np.nanmin([v for v in tops if v > 0] or [0.1]))
    ax.set_ylim(max(_floor * 0.6, 1e-4), 1.2)
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
    axes[0].set_yticklabels([ELEMENT_LABELS[e] for e in ELEMENTS])
    # The element colour on the tick labels, so 5.1 and 5.2 can be read against each
    # other and against the chance-table inputs without counting rows.
    for tick, el in zip(axes[0].get_yticklabels(), ELEMENTS):
        tick.set_color(element_colour(el, dark))
        tick.set_fontweight("bold")
    axes[0].legend(loc="lower right", fontsize=7)
    fig.suptitle("B5 · Allocation dumbbell", fontsize=9.5, fontweight="bold", color=p["text"])
    fig.tight_layout()
    return fig, axes




def fig_b11_pos_sensitivity(
    sweep: Sweep, *, pos_prospect: float, current_z: float | None = None,
    levels: tuple[float, ...] = FAN_POS_LEVELS,
    zlim: tuple[float, float] | None = None, dark: bool = False,
):
    """B11 for the export path. Twin of ``pfig_b11_pos_sensitivity``.

    See that docstring: every curve is the same shape scaled vertically, because
    ``P_well = POS_prospect x r_location(z)`` and only the second factor moves with
    depth. That is the content, not a limitation.
    """
    p = palette(dark)
    fig, ax = new_figure(figsize=(6, 5), dark=dark)
    r = np.asarray(sweep.r_location, dtype=float)

    for level in levels:
        if abs(level - float(pos_prospect)) < 5e-3:
            continue
        ax.plot(r * float(level) * 100.0, sweep.z, color=p["muted"], lw=0.9)
        if np.isfinite(r[0]):
            ax.annotate(f"{level:.0%}", (float(r[0] * level * 100.0), float(sweep.z[0])),
                        xytext=(0, -8), textcoords="offset points", ha="center",
                        fontsize=7, color=p["text_secondary"])

    ax.plot(r * float(pos_prospect) * 100.0, sweep.z, color=colour("p_well", dark),
            lw=2.6, label=f"POS in force: {pos_prospect:.0%}")
    if current_z is not None:
        ax.axhline(current_z, color=p["well"], ls="--", lw=1.0)

    ax.set_xlim(0, 100)
    ax.set_xlabel("P_well  (%)")
    depth_axis(ax, zlim=zlim or (float(sweep.z.min()), float(sweep.z.max())))
    ax.set_title("B11 · P_well sensitivity to POS_prospect "
                 f"({reference_label(sweep.reference)})")
    ax.grid(True, lw=0.6, alpha=0.7)
    ax.legend(loc="lower right", fontsize=7.5)
    fig.tight_layout()
    return fig, ax


def fig_b12_banded_percentiles(
    bp: BandedPercentiles, *, mefs: float | None = None, show_proven: bool = True,
    show_mean: bool = False, well_label: str | None = None,
    probability_scale: str = "probit",
    volume_scale: str = "log", dark: bool = False,
):
    """B12 for the export path. Twin of ``pfig_b12_banded_percentiles``.

    See that docstring for the argument: solid is the whole resource in a
    contact-depth band and dotted is what this well would prove in it, each family
    carries its own colour ramp ordered by depth, and straightness on log-probit axes
    is lognormality. Depth is the family rather than an axis, so this figure is exempt
    from non-negotiable 2.
    """
    if volume_scale not in VOLUME_SCALES:
        raise ValueError(
            f"unknown volume scale {volume_scale!r}; expected one of {VOLUME_SCALES}"
        )
    p = palette(dark)
    fig, ax = new_figure(figsize=(6.6, 6.8), dark=dark)
    n_bands = len(bp.bands)
    total_shades = depth_shades(n_bands, dark)
    proven_shades = concept_shades("tested", n_bands, dark)
    y = probability_coords(np.asarray(bp.percentiles, dtype=float), probability_scale)

    for band, shade, pshade in zip(bp.bands, total_shades, proven_shades):
        # Dotted and thinner: the band total is context, the proven part is the
        # subject -- see the plotly twin.
        ax.plot(band.total, y, color=shade, lw=1.4, ls=":", marker="o", ms=3.5,
                label=f"{band.label}  (n {band.n})")
        if show_mean and np.isfinite(band.total_mean_p):
            ax.plot([band.total_mean],
                    [float(probability_coords(band.total_mean_p, probability_scale))],
                    color=shade, marker="D", mfc="none", ms=7, mew=1.4, ls="none")
        if show_proven and band.proven is not None:
            ax.plot(band.proven, y, color=pshade, lw=2.4, marker="o", ms=3.5,
                    mfc=p["surface"])
            if show_mean and np.isfinite(band.proven_mean_p):
                ax.plot([band.proven_mean],
                        [float(probability_coords(band.proven_mean_p, probability_scale))],
                        color=pshade, marker="D", mfc="none", ms=6, mew=1.3, ls="none")

    # The style key is in the subtitle, not the legend -- see the plotly twin: as
    # legend entries these read as extra bands and sent the reader hunting for a line
    # that was never a separate series.
    handles, labels = ax.get_legend_handles_labels()
    key = []
    if mefs:
        ax.axvline(float(mefs), color=colour("mefs", dark), ls="--", lw=1.0)

    log = volume_scale == "log"
    if log:
        ax.set_xscale("log")
        # Plain numbers, 1-2-5 per decade, shared with the plotly half from
        # theme.log_ticks: matplotlib's default "4 x 10^0" is a physicist's notation
        # for an axis carrying MMboe, and the two backends must not label it
        # differently.
        ticks = log_ticks(*bp.volume_range)
        ax.set_xticks(ticks)
        ax.set_xticklabels(log_tick_text(ticks))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xlabel("Resource (MMboe) · log scale")
    else:
        ax.set_xscale("linear")
        ax.set_xlim(left=0.0)
        ax.set_xlabel("Resource (MMboe) · linear scale")
    if mefs:
        ax.annotate(f"MEFS {mefs:g}", (float(mefs), ax.get_ylim()[1]),
                    xytext=(3, -3), textcoords="offset points", fontsize=7,
                    va="top", color=colour("mefs", dark))
    probability_axis(ax, probability_scale)
    ax.set_title("B12 · Resource by contact-depth band "
                 f"({BAND_MODE_LABELS[bp.mode]}, "
                 + (f"Well {well_label}: " if well_label else "well ")
                 + f"{bp.z_entry:.0f}-{bp.z_exit:.0f} m)")
    _sub = ("solid = the whole resource in the band · "
            "dotted = the part this well would prove · colour = depth, light to dark")
    ax.annotate(_sub, (0.5, 1.005), xycoords="axes fraction", ha="center", va="bottom",
                fontsize=7, color=p["text_secondary"])
    ax.grid(True, which="both", lw=0.6, alpha=0.7)
    ax.legend(handles=handles + key, loc="lower left", fontsize=6.5)
    fig.tight_layout()
    return fig, ax


def fig_b7_frontier(
    vsweep: VolumeSweep, *, current_z: float | None = None, min_support: int = MIN_SUPPORT,
    label_every: int = 4, chance_scale: str = "linear", dark: bool = False,
):
    """B7, for the export path. Twin of ``pfig_b7_frontier``.

    Chance against volume, parametric in depth: the trade-off the whole tool is
    about. Neither axis carries a depth, so depth appears as labels along the curve
    and this figure joins A5, A6, B4 and B5 in the depth-rule exemption.

    ``chance_scale`` puts the log option on the **chance** axis, running 1-110 %,
    where it belongs: a linear chance axis spends its height on the shallow end and
    compresses every deep location into the bottom centimetre.
    """
    p = palette(dark)
    fig, ax = new_figure(figsize=(6.4, 5.4), dark=dark)
    n_disc = vsweep.n_discovery
    pw = thin(vsweep.p_well, n_disc, min_support) * 100.0
    proven = thin(vsweep.proven_mean, n_disc, min_support)
    assoc = (thin(vsweep.discovery_mean, n_disc, min_support)
             if vsweep.discovery_mean is not None else None)

    # The ladder, thin and dotted in the same hue -- see the plotly twin for why a
    # mean-only frontier understates what the argument needs.
    for stat, name, ls in (("discovery_p90", "P90", ":"),
                           ("discovery_p50", "P50", "--"),
                           ("discovery_p10", "P10", ":")):
        values = getattr(vsweep, stat, None)
        if values is None:
            continue
        ax.plot(thin(values, n_disc, min_support), pw,
                color=colour("well_associated", dark), lw=0.9, ls=ls,
                label=f"Well associated {name}")
    if assoc is not None:
        ax.plot(assoc, pw, color=colour("well_associated", dark), lw=2.4,
                label="Well associated mean")
    ax.plot(proven, pw, color=colour("tested", dark), lw=1.6, ls="--", label="Proven mean")

    base = assoc if assoc is not None else proven
    for i in range(0, vsweep.z.size, max(1, label_every)):
        if np.isfinite(base[i]) and np.isfinite(pw[i]):
            ax.annotate(f"{vsweep.z[i]:.0f}", (base[i], pw[i]), xytext=(4, 4),
                        textcoords="offset points", fontsize=6.5, color=p["text_secondary"])

    if current_z is not None:
        hx = float(np.interp(current_z, vsweep.z, np.nan_to_num(base, nan=0.0)))
        hy = float(np.interp(current_z, vsweep.z, np.nan_to_num(pw, nan=0.0)))
        ax.plot([hx], [hy], marker="o", ms=10, mfc="none", mec=p["well"], mew=2.5, zorder=5)
        ax.annotate(f"  this well, {current_z:.0f} m", (hx, hy), fontsize=8,
                    color=p["well"], va="center")
    if vsweep.mefs is not None:
        ax.axvline(vsweep.mefs, color=colour("minimum", dark), ls=":", lw=1.0)

    if chance_scale not in ("linear", "log"):
        raise ValueError(f"unknown chance_scale {chance_scale!r}; expected 'linear' or 'log'")
    ax.set_xlim(left=0)
    ax.set_xlabel("Mean resource (MMboe)")
    if chance_scale == "log":
        # 1 % is the floor rather than zero, because a log axis has no zero and a
        # location whose P_well rounds to nothing has nothing to contribute to a
        # trade-off curve. Plain percent labels, not exponents.
        ax.set_yscale("log")
        ax.set_ylim(1.0, 110.0)
        ax.set_yticks([1, 2, 5, 10, 20, 50, 100])
        ax.set_yticklabels(["1", "2", "5", "10", "20", "50", "100"])
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_ylabel(r"$P_{well}$  (%, log 1–110)")
    else:
        ax.set_ylim(0, 110)
        ax.set_ylabel(r"$P_{well}$  (%)")
    ax.set_title(f"B7 · Chance against volume ({reference_label(vsweep.reference)})")
    ax.grid(True, lw=0.6, alpha=0.7)
    ax.legend(loc="lower left", fontsize=7.5)
    fig.tight_layout()
    return fig, ax


def fig_b8_commercial_chance(
    vsweep: VolumeSweep, *, current_z: float | None = None,
    zlim: tuple[float, float] | None = None, min_support: int = MIN_SUPPORT,
    dark: bool = False,
):
    """B8, for the export path. Twin of ``pfig_b8_commercial_chance``.

    ``Pc(well) = P_well x Pmcfs(well)``: a falling curve times a rising one, so the
    product usually has an interior maximum and that maximum is where the well goes
    on commercial grounds. Conditional solid, unconditional dashed.
    """
    p = palette(dark)
    z = vsweep.z
    fig, ax = new_figure(figsize=(5.6, 5.5), dark=dark)
    if vsweep.p_discovery_exceeds_mefs is None:
        ax.set_title("B8 · Commercial chance — needs a MEFS")
        depth_axis(ax, zlim=zlim or (float(z.min()), float(z.max())))
        fig.tight_layout()
        return fig, ax

    pw = thin(vsweep.p_well, vsweep.n_discovery, min_support) * 100.0
    pmcfs = thin(vsweep.p_discovery_exceeds_mefs, vsweep.n_discovery, min_support) * 100.0
    pc = pw * pmcfs / 100.0

    ax.plot(pmcfs, z, color=colour("tested", dark), lw=1.9,
            label="Pmcfs(well) — conditional on a discovery")
    ax.plot(pw, z, color=colour("well_associated", dark), lw=1.9, label="P_well — chance of a discovery")
    ax.plot(pc, z, color=colour("minimum", dark), lw=2.4, ls="--",
            label="Pc(well) — commercial chance, unconditional")

    if np.any(np.isfinite(pc)):
        best = int(np.nanargmax(pc))
        # The plateau band -- see the plotly twin. Named, because a shaded region with
        # no legend entry is a reader guessing, and the wrong guess on a trade-off
        # figure is that it means uncertainty.
        _span = plateau_span(pc, z, best)
        if _span is not None and _span[1] - _span[0] > 1.0:
            ax.axhspan(_span[0], _span[1], color=colour("minimum", dark), alpha=0.10,
                       lw=0, zorder=0)
            ax.plot([float(np.nanmax(pc))] * 2, list(_span),
                    color=colour("minimum", dark), lw=5, alpha=0.35,
                    label=f"Within 2 % of the best Pc — {_span[0]:,.0f}–{_span[1]:,.0f} m")
        ax.plot([pc[best]], [z[best]], marker="*", ms=12, color=colour("minimum", dark), zorder=5)
        ax.annotate(f"  best {pc[best]:.1f}% at {z[best]:.0f} m", (pc[best], z[best]),
                    fontsize=7.5, color=colour("minimum", dark), va="center")
    if current_z is not None:
        ax.axhline(current_z, color=p["text"], ls="--", lw=1.0)

    depth_axis(ax, zlim=zlim or (float(z.min()), float(z.max())))
    ax.set_xlim(0, 105)
    ax.set_xlabel("Probability (%)")
    ax.set_title(f"B8 · Commercial chance vs location (MEFS {vsweep.mefs:.1f} MMboe)")
    ax.legend(loc="lower right", fontsize=7.5)
    fig.tight_layout()
    return fig, ax


def fig_a8_contact_distribution(
    ts: TrialSet, *, n_bins: int = 40, current_entry: float | None = None,
    zlim: tuple[float, float] | None = None, dark: bool = False,
):
    """A8 for the export path. Twin of ``pfig_a8_contact_distribution``.

    The contact distribution as a horizontal histogram with ``P(deeper than z)``
    over it. Two x-axes, which is allowed here for the reason given in the plotly
    twin: the depth axis still means exactly one thing.
    """
    p = palette(dark)
    res = np.asarray(ts.col("resource"), dtype=float)
    contact = np.asarray(ts.col("contact"), dtype=float)
    contact = contact[(res > 0) & np.isfinite(contact)]
    lo, hi = float(contact.min()), float(contact.max())

    fig, ax = new_figure(figsize=(5.6, 5.6), dark=dark)
    counts, edges = np.histogram(contact, bins=int(n_bins), range=(lo, hi))
    centres = 0.5 * (edges[:-1] + edges[1:])

    top = ax.twiny()
    top.barh(centres, counts, height=(edges[1] - edges[0]) * 0.92,
             color=colour("prospect", dark), alpha=0.55,
             edgecolor=colour("prospect", dark), linewidth=0.4)
    top.set_xlabel("trials per bin", fontsize=8)
    top.set_xlim(left=0)

    ordered = np.sort(contact)
    deeper = 100.0 * (ordered.size - np.arange(ordered.size)) / ordered.size
    ax.plot(deeper, ordered, color=colour("well_associated", dark), lw=2.2,
            label="P(contact deeper than this)", zorder=5)
    if current_entry is not None:
        ax.axhline(current_entry, color=p["well"], ls="--", lw=1.4, zorder=6)

    depth_axis(ax, zlim=zlim or (lo, hi))
    top.set_ylim(ax.get_ylim())
    ax.set_xlim(0, 105)
    ax.set_xlabel("P(contact deeper than this depth)  (%)")
    ax.set_title(f"A8 · Contact distribution and P(deeper) — {contact.size:,} trials")
    ax.legend(loc="lower left", fontsize=7.5)
    fig.tight_layout()
    return fig, ax


def fig_b9_chance_weighted(
    vsweep: VolumeSweep, *, ce=None, current_z: float | None = None,
    zlim: tuple[float, float] | None = None, min_support: int = MIN_SUPPORT,
    dark: bool = False,
):
    """B9 for the export path. Twin of ``pfig_b9_chance_weighted``.

    ``P_well x mean volume`` against depth: a falling curve times a rising one, so
    the product peaks somewhere in between and that depth maximises the expectation.
    An expected value describes no outcome that can happen -- it ranks locations, it
    does not forecast a volume.
    """
    p = palette(dark)
    z = vsweep.z
    fig, ax = new_figure(figsize=(5.8, 5.5), dark=dark)
    pw = thin(vsweep.p_well, vsweep.n_discovery, min_support)
    series = [("Proven MEAN × P_well",
               thin(vsweep.proven_mean, vsweep.n_discovery, min_support), "tested")]
    if vsweep.discovery_mean is not None:
        series.append(("Well associated MEAN × P_well",
                       thin(vsweep.discovery_mean, vsweep.n_discovery, min_support),
                       "well_associated"))

    if vsweep.proven_p90 is not None and vsweep.proven_p10 is not None:
        lo = pw * thin(vsweep.proven_p90, vsweep.n_discovery, min_support)
        hi = pw * thin(vsweep.proven_p10, vsweep.n_discovery, min_support)
        band = np.isfinite(lo) & np.isfinite(hi)
        if band.any():
            ax.fill_betweenx(z[band], lo[band], hi[band], color=colour("tested", dark),
                             alpha=0.20, lw=0, label="Proven P90–P10 × P_well")

    # P99 and P1 as thin grey lines outside the fill, like the plotly twin: on a
    # right-skewed distribution P1 runs far above P10, and filling out to it would
    # swamp the mean lines this figure is about.
    for stat, label, ls in (("proven_p99", "P99", ":"), ("proven_p90", "P90", "--"),
                            ("proven_p10", "P10", "--"), ("proven_p1", "P1", ":")):
        values = getattr(vsweep, stat, None)
        if values is None:
            continue
        weighted = pw * thin(values, vsweep.n_discovery, min_support)
        if np.isfinite(weighted).sum() >= 2:
            ax.plot(weighted, z, color=p["muted"], lw=0.9, ls=ls,
                    label=f"Proven {label} × P_well")

    if ce is not None and np.any(np.isfinite(ce.ce)):
        ax.plot(ce.ce, ce.z, color=colour("commercial", dark), lw=2.0, ls="--",
                label=f"Certainty equivalent (rho {ce.rho:,.0f})")

    for name, mean, role in series:
        weighted = pw * mean
        ax.plot(weighted, z, color=colour(role, dark), lw=2.2, label=name)
        if np.any(np.isfinite(weighted)):
            i = int(np.nanargmax(weighted))
            _sp = plateau_span(weighted, z, i)
            if _sp is not None and _sp[1] - _sp[0] > 1.0:
                ax.plot([float(np.nanmax(weighted))] * 2, list(_sp),
                        color=colour(role, dark), lw=5, alpha=0.30,
                        label=f"{name} — within 2 % of best")
            ax.plot([weighted[i]], [z[i]], marker="*", ms=12,
                    color=colour(role, dark), zorder=5)
            ax.annotate(f"  {weighted[i]:.1f} at {z[i]:.0f} m", (weighted[i], z[i]),
                        fontsize=7.5, color=colour(role, dark), va="center")
    if current_z is not None:
        ax.axhline(current_z, color=p["text"], ls="--", lw=1.0)

    depth_axis(ax, zlim=zlim or (float(z.min()), float(z.max())))
    ax.set_xlim(left=0)
    # Plain text, not mathtext: the plotly twin cannot render LaTeX, so writing it
    # here guaranteed the two labels differed and the guard could never compare
    # them. Legibility is worth less than the export saying what the screen says.
    ax.set_xlabel("P_well × mean volume  (MMboe, expected)")
    ax.set_title("B9 · Chance-weighted resource vs location")
    ax.legend(loc="lower right", fontsize=7.5)
    fig.tight_layout()
    return fig, ax

def fig_a9_prospect_density(
    ts: TrialSet, *, mefs: float | None = None, bins: int = 40, dark: bool = False,
):
    """A9 for the export path. Twin of ``pfig_a9_prospect_density``.

    One distribution rather than A6's four: this figure is about the *shape* of the
    prospect's resource, not about overlap. The mean is drawn thicker than the P50
    because on a right-skewed distribution they differ and the mean is the one that
    gets quoted.
    """
    p = palette(dark)
    res = np.asarray(ts.col("resource"), dtype=float)
    values = res[res > 0]
    fig, ax = new_figure(figsize=(6, 4.4), dark=dark)
    if not values.size:
        ax.set_title("A9 · Prospect resource — no successful trials")
        return fig, ax

    ax.hist(values, bins=bins, density=True, color=colour("prospect", dark), alpha=0.55)
    stats = {
        "P90": float(np.percentile(values, 10.0)),
        "P50": float(np.percentile(values, 50.0)),
        "Mean": float(np.mean(values)),
        "P10": float(np.percentile(values, 90.0)),
    }
    for name, value in stats.items():
        ax.axvline(value, color=colour("prospect", dark) if name == "Mean" else p["muted"],
                   lw=1.8 if name == "Mean" else 1.0,
                   ls="-" if name == "Mean" else "--")
        ax.annotate(f" {name} {value:,.1f}", (value, ax.get_ylim()[1]), fontsize=6.5,
                    color=p["text_secondary"], va="top", rotation=90)
    if mefs is not None:
        ax.axvline(mefs, color=colour("minimum", dark), ls=":", lw=1.0)

    ax.set_xlim(left=0)
    ax.set_xlabel("Recoverable resource (MMboe)")
    ax.set_ylabel("Density")
    ax.set_title("A9 · Prospect resource distribution (success case)")
    fig.tight_layout()
    return fig, ax


# ------------------------------------------------- the export-path twins
# Three figures were built on the interactive path first, because they were
# designed by looking at them. They are drawn again here so the export cannot
# ship a document missing the figures that carry the argument. Kept in this
# order: the colour key that explains the palette, the map that shows the split
# in plan, the concepts figure that shows it in section and in distribution.


def fig_colour_key(dark: bool = False):
    """The volume-concept colour key. The export path's twin of
    :func:`wellvolpos.viz.interactive.pfig_colour_key`.

    Reads ``interactive.CONCEPT_KEY`` rather than restating it, so the two keys
    cannot list different concepts -- which for a colour key would be worse than
    having only one.
    """
    from .interactive import CONCEPT_KEY

    p = palette(dark)
    n = len(CONCEPT_KEY)
    fig, ax = new_figure(figsize=(9, 0.42 * n + 0.4), dark=dark)
    for i, (role, label, meaning) in enumerate(CONCEPT_KEY):
        y = n - i
        ax.add_patch(
            plt.Rectangle((0.0, y - 0.3), 0.035, 0.6, facecolor=colour(role, dark), lw=0)
        )
        ax.text(0.05, y, f"{label} — {meaning}", va="center", ha="left", fontsize=8.5,
                color=p["text"])
    ax.set_xlim(0, 1)
    ax.set_ylim(0.4, n + 0.6)
    ax.axis("off")
    fig.tight_layout()
    return fig, ax


def fig_map_view(
    ad: AreaDepth, *, apex: float, z_entry: float, z_exit: float | None = None,
    interval: float = 50.0, well_azimuth_deg: float = 35.0, dark: bool = False,
):
    """Conceptual plan view of the closure. Twin of ``pfig_map_view``.

    A cartoon, not a map: each ring is the circle enclosing the area A(z) holds
    at that depth, so the areas and their spacing are faithful while the outline
    is not, and the well's map position is arbitrary -- only its radius means
    anything. Contours sit on round multiples of ``interval`` so they stay put
    when the apex estimate is nudged.

    No depth on either axis (both are map kilometres), so this figure does not
    call :func:`depth_axis` -- it is in the exempt set with A5, A6, B4 and B5.
    Equal aspect instead, because an area that reads as twice another must be
    twice another.
    """
    p = palette(dark)
    contours = ad.contour_radii(apex, interval=interval, z_max=ad.deepest)
    theta = np.linspace(0.0, 2.0 * np.pi, 181)
    fig, ax = new_figure(figsize=(6.2, 6.2), dark=dark)

    r_entry = ad.radius_at(z_entry, apex)
    r_exit = ad.radius_at(z_exit, apex) if z_exit is not None else r_entry
    r_base = float(contours.radii.max()) if contours.radii.size else r_exit

    # The three areas, widest first so each draws over the one outside it. Same
    # split B0 draws in section, so the two figures colour-key identically.
    a_attic = np.pi * r_entry ** 2
    a_proven = max(np.pi * (r_exit ** 2 - r_entry ** 2), 0.0)
    a_possible = max(np.pi * (r_base ** 2 - r_exit ** 2), 0.0)
    for r_out, r_in, role, label in (
        (r_base, r_exit, "below_lkh", f"Unproven below LKH ({a_possible:.2f} km²)"),
        (r_exit, r_entry, "tested", f"Potentially proven — entry to exit ({a_proven:.2f} km²)"),
        (r_entry, 0.0, "attic", f"Potential attic — up-dip of entry ({a_attic:.2f} km²)"),
    ):
        if r_out <= r_in + 1e-12:
            continue
        ax.fill(r_out * np.cos(theta), r_out * np.sin(theta),
                color=colour(role, dark), alpha=0.35, lw=0, label=label, zorder=1)
        if r_in > 0.0:
            # Punch the hole with the surface colour rather than drawing a path
            # with a hole: simpler, and the ring beneath is redrawn on top anyway.
            ax.fill(r_in * np.cos(theta), r_in * np.sin(theta),
                    color=p["surface"], lw=0, zorder=1.1)

    # Deepest ring first, so shallow ones draw on top.
    rings = sorted(
        zip(contours.depths, contours.radii, contours.extrapolated, contours.at_data_limit),
        key=lambda t: -t[0],
    )
    # Dashed contours with a small depth label, and the entry contour below as the
    # one solid ring -- so line style says "is this the well?" and nothing else.
    # Extrapolated rings above the shallowest sampled contact are marked by opacity
    # instead, which reads as less certain without competing with the entry.
    label_i = 0
    for zz, rr, is_extrap, is_limit in rings:
        ax.plot(
            rr * np.cos(theta), rr * np.sin(theta), zorder=2,
            color=colour("attic" if zz <= z_entry else "prospect", dark),
            lw=2.0 if is_limit else 0.9, ls="--",
            alpha=0.45 if is_extrap else 1.0,
        )
        if rr > 0:
            from .interactive import _LABEL_AZIMUTHS

            ang_lab = np.deg2rad(_LABEL_AZIMUTHS[label_i % len(_LABEL_AZIMUTHS)])
            ax.annotate(f"{zz:.0f}", (rr * np.cos(ang_lab), rr * np.sin(ang_lab)),
                        ha="center", va="center", fontsize=6.5, color=p["text_secondary"],
                        alpha=0.5 if is_extrap else 0.95, zorder=3,
                        bbox=dict(boxstyle="square,pad=0.12", fc=p["surface"], ec="none"))
            label_i += 1
    ax.plot(r_entry * np.cos(theta), r_entry * np.sin(theta), zorder=3.5,
            color=colour("attic", dark), lw=2.6, ls="-")
    ax.annotate(f"{z_entry:.0f} m — well entry", (0.0, r_entry), xytext=(0, 6),
                textcoords="offset points", ha="center", fontsize=8,
                color=colour("attic", dark), zorder=4)

    ang = np.deg2rad(well_azimuth_deg)
    xw, yw = r_entry * np.cos(ang), r_entry * np.sin(ang)
    ax.plot([xw], [yw], marker="o", ms=9, mfc="none", mec=p["well"], mew=2.5, zorder=4)
    ax.plot([xw], [yw], marker=".", ms=4, color=p["well"], zorder=4)
    ax.annotate("  WELL", (xw, yw), va="center", ha="left", fontsize=9,
                color=p["well"], zorder=4)
    ax.plot([0.0], [0.0], marker="x", ms=8, mew=2.0, color=p["text"], zorder=4)
    ax.annotate(f"  apex {apex:.0f} m", (0.0, 0.0), va="center", ha="left",
                fontsize=8, color=p["text_secondary"], zorder=4)

    lim = r_base * 1.12 if r_base > 0 else 1.0
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xlabel("km east of apex (equivalent-circle radius — shape is illustrative)")
    ax.set_ylabel("km north of apex")
    ax.set_title(
        f"Conceptual map view — contours on {interval:.0f} m multiples "
        f"(deepest sampled contact {contours.depths[-1]:.0f} m)"
    )
    ax.legend(loc="lower left", fontsize=7.5, framealpha=0.8)
    fig.tight_layout()
    return fig, ax


def _reservoir_section_mpl(ax, ad, ts, *, z_entry, z_exit, dark, area_scale="area",
                           show_classes=True):
    """The concepts figure's left panel, in matplotlib.

    See :func:`wellvolpos.viz.interactive._reservoir_section` for why x is area
    and not a lateral distance, and why the base reservoir comes from inverting
    the pay rather than from reading a thickness column.
    """
    p = palette(dark)
    _, transform = AREA_SCALES.get(area_scale, AREA_SCALES["area"])
    a, top = transform(ad.a), ad.z

    tfp = thickness_from_pay(ts, ad)
    stats = tfp.summary()
    thickness = stats["p50"] if tfp.n_resolved else None

    ax.plot(a, top, color=p["text"], lw=2.0, label="Top reservoir")
    if thickness is not None:
        # **Four base curves, matching the plotly original**: P90, P50, mean, P10 of
        # the thickness recovered from pay. The export drew only three, and the one
        # it dropped was the *mean* -- the number that gets quoted. It also left
        # P90/P10 unlabelled, so an exported A1 had a legend claiming a single
        # "Base reservoir" surface where the screen showed a sampled range.
        for key, name, lw, ls in (("p90", "Base P90", 0.9, ":"),
                                  ("p50", "Base P50", 1.5, "--"),
                                  ("mean", "Base mean", 1.2, "-"),
                                  ("p10", "Base P10", 0.9, ":")):
            ax.plot(a, top + stats[key],
                    color=p["text"] if key == "p50" else p["muted"],
                    lw=lw, ls=ls, label=name)
        base = top + thickness

        for lo, hi, role, label in () if not show_classes else (
            (-np.inf, z_entry, "up_dip", "up-dip"),
            (z_entry, z_exit, "tested", "tested"),
            (z_exit, np.inf, "below_lkh", "unproven below LKH"),
        ):
            upper = np.clip(top, lo, hi)
            lower = np.clip(base, lo, hi)
            m = lower > upper + 1e-9
            if m.sum() < 2:
                continue
            ax.fill(np.concatenate([a[m], a[m][::-1]]),
                    np.concatenate([upper[m], lower[m][::-1]]),
                    color=colour(role, dark), alpha=0.55, lw=0)
            mid = int(np.flatnonzero(m)[m.sum() // 2])
            ax.text(a[mid], 0.5 * (upper[mid] + lower[mid]), label, fontsize=7.5,
                    ha="center", va="center", color=p["text"])

    # **No well drawn here.** The plotly counterpart leaves it to the caller, and
    # both callers already draw one -- so this helper was adding a third marker to
    # A1 and a second to C1. The well is a property of the well, not of the
    # reservoir band.
    return stats if thickness is not None else None


def fig_c1_section(
    ad: AreaDepth, ts: TrialSet, *, z_entry: float, z_exit: float,
    area_scale: str = "area", dark: bool = False,
):
    """C1 for the export path -- the structure above C2's curves.

    Fully labelled, like the plotly twin (Lars, 2026-08-11). It spent a while as an
    unlabelled thumbnail on the argument that A1 carried the readable version; in
    use that failed, because a structural panel with no depth axis cannot show that
    the up-dip volume sits *above* the well at a particular depth, which is the one
    thing C1 exists to show.
    """
    p = palette(dark)
    label, _ = AREA_SCALES.get(area_scale, AREA_SCALES["area"])
    fig, ax = new_figure(figsize=(6.0, 4.0), dark=dark)
    _reservoir_section_mpl(ax, ad, ts, z_entry=z_entry, z_exit=z_exit,
                           dark=dark, area_scale=area_scale)
    for depth, ls, name in ((z_entry, "--", "well entry"), (z_exit, ":", "well exit")):
        ax.axhline(depth, color=p["well"], lw=1.1, ls=ls)
        ax.text(0.99, depth, name, transform=ax.get_yaxis_transform(),
                ha="right", va="bottom", fontsize=7.5, color=p["well"])

    # The well itself, as a vertical line -- see ``interactive._well_track`` for why
    # x = A(z_entry) is the only honest anchor on an area axis. The rules above give
    # the depths; this gives the borehole, which is what the eye looks for on a
    # section.
    _, xt = AREA_SCALES.get(area_scale, AREA_SCALES["area"])
    x_well = float(np.asarray(xt(np.asarray([ad.area_at(z_entry)], dtype=float)))[0])
    ax.plot([x_well, x_well], [ad.shallowest, z_entry],
            color=p["well"], lw=1.1, ls=":")
    ax.plot([x_well, x_well], [z_entry, z_exit],
            color=p["well"], lw=4.0, solid_capstyle="butt", label="the well")
    ax.set_xlim(left=0)
    ax.set_xlabel(label)
    depth_axis(ax, zlim=(ad.shallowest, ad.deepest))
    ax.set_title("C1 · the structure, and the volumes a well at this depth divides it into",
                 fontsize=9)
    ax.grid(True, lw=0.6, alpha=0.7)
    fig.tight_layout()
    return fig, ax


def fig_c2_exceedance(
    ts: TrialSet, groups: Groups, vc: VolumeClasses, *,
    pos_prospect: float, p_well: float, mefs: float | None = None,
    pc_well: float | None = None, dark: bool = False,
):
    """C2 for the export path. Twin of ``pfig_c2_exceedance``.

    Takes ``pc_well`` so the exported figure carries the commercial class as well.
    The per-reading toggles are deliberately *not* mirrored: an exported figure
    showing one reading without saying which would be the risked/unrisked confusion
    this pair exists to prevent, so the export always draws both.

    Two curves per concept in one colour: **solid** conditional (success case),
    starting at 100 % and carrying the percentiles, and **dashed** unconditional
    (risked), starting at the chance of that case. The vertical gap between the
    prospect's and the well's dashed starts is the location penalty.
    """
    res = ts.col("resource")
    fig, ax_exc = new_figure(figsize=(8.6, 6.2), dark=dark)

    disc, dry = groups.discovery, groups.dry_with_attic
    # One definition, shared with the plotly twin and with the caption that
    # quotes these curves' MEFS crossings -- see core/mefs.c2_cases.
    cases = c2_cases(ts, groups, vc, pos_prospect, p_well,
                     mefs=mefs, pc_well=pc_well)
    spans: dict[str, tuple[float, float, str]] = {}
    for name, values, chance_of, role in cases:
        for reading, chance_used in (("conditional", 1.0), ("unconditional", chance_of)):
            v, pct = risked_exceedance(values, chance_used)
            if v.size == 0:
                continue
            ax_exc.plot(v, pct, color=colour(role, dark),
                        lw=2.2 if reading == "conditional" else 1.6,
                        ls="-" if reading == "conditional" else "--",
                        label=f"{name} — "
                              f"{READING_LABELS[reading].split(' (')[0].lower()}")
            # Markers on both families, values on the conditional one only -- see the
            # plotly twin: the volumes are identical between the two readings, so the
            # second copy of each number was text without information.
            _mark_exceedance_mpl(ax_exc, values, role, dark, chance=chance_used,
                                 show_text=reading == "conditional", size=4.0,
                                 ha="left")
            # Where this curve crosses MEFS -- see the plotly twin. Filled =
            # conditional, open = risked; no text, because eight labels along one
            # vertical line overlap on this figure.
            if mefs is not None and v.size:
                y_at = float(np.interp(float(mefs), v, pct))
                ax_exc.plot([float(mefs)], [y_at], marker="o", ms=6,
                            mfc=colour(role, dark) if reading == "conditional" else "none",
                            mec=colour(role, dark), mew=1.6, zorder=6)
        vals = np.sort(np.asarray(values, dtype=float))
        positive = vals[np.isfinite(vals) & (vals > 0)]
        if positive.size:
            spans[name] = (float(positive.min()), float(positive.max()), role)

    for value, label, role in (
        (pos_prospect, "Asso. Final Prospect POS", "prospect"),
        (p_well, "Asso. Well POS", "well_associated"),
    ):
        ax_exc.axhline(value * 100.0, color=colour(role, dark), lw=0.9, ls=":")
        ax_exc.annotate(f"{label} {value:.0%}", (1.0, value * 100.0),
                        xycoords=("axes fraction", "data"), xytext=(-3, 3),
                        textcoords="offset points", ha="right", fontsize=7.5,
                        color=colour(role, dark))
    if mefs is not None:
        ax_exc.axvline(mefs, color=colour("minimum", dark), lw=1.1, ls=":")

    # The nesting braces, below the 0 % line and widest at the bottom.
    # **Commercial leads** (Lars, 2026-08-15: "there are 4 bars but the commercial is
    # missing"). It was populated in ``spans`` and then dropped by this hardcoded list.
    #
    # It sits first because its left end is the most informative thing about it: the
    # bar starts at MEFS, not at zero, which is the whole difference between it and the
    # well-associated bar under it. It is *not* part of the containment chain the other
    # four form -- min ⊂ up-dip ⊂ tested ⊂ well associated ⊂ prospect -- because it is a
    # threshold-conditioned subset rather than a spatial one.
    order = ["Commercial accumulation", "Up-dip volume", "Resource tested by well",
             "Well associated resource potential", "Prospect resource potential"]
    step, base = 7.5, -9.0
    for i, name in enumerate(order):
        if name not in spans:
            continue
        lo, hi, role = spans[name]
        y = base - i * step
        c = colour(role, dark)
        ax_exc.plot([lo, hi], [y, y], color=c, lw=2.2)
        for xx in (lo, hi):
            ax_exc.plot([xx, xx], [y - 1.8, y + 1.8], color=c, lw=2.2)
        ax_exc.annotate(f"  {name}", (hi, y), fontsize=7.5, color=c, va="center")

    ax_exc.set_xlim(left=0)
    ax_exc.set_ylim(base - len(order) * step - 3.0, 107.0)
    # Ticks pinned to 0-100: the space below carries the braces, and a negative
    # probability label is meaningless.
    ax_exc.set_yticks(list(range(0, 101, 20)))
    ax_exc.set_xlabel("Recoverable resource (MMboe)")
    ax_exc.set_ylabel("Probability of exceedance (%)")
    ax_exc.set_title(
        "C2 · The same volumes as exceedance curves — solid conditional, dashed unconditional",
        fontsize=9.5,
    )
    # A legend, matching the interactive half (Lars, 2026-08-12). The braces name the
    # four concepts, but they label *ranges* below the zero line rather than curves,
    # so eight curves in four colours and two styles had no key at all.
    handles, labels = ax_exc.get_legend_handles_labels()
    if handles:
        ax_exc.legend(loc="upper right", fontsize=6.5, framealpha=0.9)
    fig.tight_layout()
    return fig, ax_exc


def fig_c3_mefs_bars(
    ts: TrialSet, groups: Groups, vc: VolumeClasses, *,
    pos_prospect: float, p_well: float, mefs: float, dark: bool = False,
):
    """C3, for the export path. Twin of ``pfig_c3_mefs_bars``."""
    p = palette(dark)
    crossings = tuple(reversed(
        c2_crossings(ts, groups, vc, pos_prospect, p_well, mefs)))
    fig, ax = new_figure(figsize=(7.6, 3.6), dark=dark)
    y = np.arange(len(crossings), dtype=float)
    h = 0.36
    for offset, reading, values, hatch in (
        (+h / 2, "unrisked", [c.conditional for c in crossings], None),
        (-h / 2, "risked", [c.risked for c in crossings], "///"),
    ):
        for i, (c, v) in enumerate(zip(crossings, values)):
            ax.barh(y[i] + offset, v * 100.0, height=h,
                    color=colour(c.role, dark),
                    alpha=0.85 if reading == "unrisked" else 0.35,
                    edgecolor=colour(c.role, dark), linewidth=1.0, hatch=hatch)
            ax.text(v * 100.0 + 1.2, y[i] + offset, f"{v:.1%}", va="center",
                    fontsize=7.5, color=p["text"])
    ax.set_yticks(y)
    ax.set_yticklabels([c.short for c in crossings], fontsize=8)
    ax.set_xlim(0, 108)
    ax.set_xlabel("Probability of exceeding the threshold (%)")
    ax.set_title(f"C3 · Chance of clearing MEFS / MCFS, {mefs:,.1f} MMboe")
    # The key lives in the subtitle, as on the plotly twin: with a colour per row a
    # legend swatch can only show one of them.
    ax.text(0.0, 1.02, "solid = unrisked (given the case happens) · "
                       "hatched = risked (the case happening AND clearing)",
            transform=ax.transAxes, fontsize=7.5, color=p["text_secondary"])
    fig.tight_layout()
    return fig, ax


def fig_c4_wedge(
    *, thickness: float, z_contact: float, z_entry: float | None = None,
    z_exit: float | None = None, apex: float | None = None, dark: bool = False,
):
    """C4, for the export path. Twin of ``pfig_c4_wedge``.

    The geometry the proven / unproven split rests on, in one picture: a layer of
    constant true vertical thickness, a flat contact, and therefore a charged interval
    that stands at full thickness up-dip and pinches to zero where the top surface
    meets the contact. See the plotly twin for why it is schematic.
    """
    p = palette(dark)
    T = float(thickness)
    zc = float(z_contact)
    top_crest = float(apex) if apex is not None else zc - 3.0 * T
    x = np.linspace(0.0, 1.0, 400)
    z_top = top_crest + (zc - top_crest) * 1.35 * x
    z_base = z_top + T
    z_hc_base = np.minimum(z_base, zc)
    charged = np.clip(z_hc_base - z_top, 0.0, None)
    live = charged > 0

    fig, ax = new_figure(figsize=(7.2, 4.4), dark=dark)
    ax.fill_between(x, z_top, z_base, color=colour("muted", dark), alpha=0.18,
                    lw=1.0, edgecolor=p["muted"], label="Reservoir layer")
    ax.fill_between(x[live], z_top[live], z_hc_base[live],
                    color=colour("well_associated", dark), alpha=0.55, lw=1.4,
                    edgecolor=colour("well_associated", dark),
                    label="Charged interval — the wedge")
    ax.axhline(zc, color=colour("prospect", dark), ls="--", lw=1.2)
    ax.text(0.01, zc, "contact", va="bottom", fontsize=8,
            color=colour("prospect", dark))

    mean_pay = float(charged[live].mean()) if live.any() else 0.0
    x_bar = float(x[live].max()) if live.any() else 1.0
    ax.plot([x_bar * 1.02] * 2, [zc, zc - T], color=colour("muted", dark), lw=4,
            label=f"Reservoir thickness T = {T:,.0f} m")
    ax.plot([x_bar * 1.02] * 2, [zc, zc - mean_pay], color=colour("tested", dark),
            lw=4, ls=":", label=f"Area-averaged pay = {mean_pay:,.0f} m")

    if z_entry is not None:
        i = int(np.argmin(np.abs(z_top - float(z_entry))))
        bottom = float(z_exit) if z_exit is not None else float(z_entry) + T
        ax.plot([x[i], x[i]], [top_crest, bottom], color=p["well"], lw=2.5,
                label="The well")

    ax.set_xlim(-0.02, 1.12)
    ax.set_xticks([])
    ax.set_xlabel("Distance down dip (schematic)")
    depth_axis(ax, (float(min(z_top.min(), zc - T)) - 0.1 * T,
                    float(max(z_base.max(), zc)) + 0.1 * T))
    ax.set_title("C4 · The wedge — why area-averaged pay is less than the "
                 f"reservoir thickness ({T:,.0f} m)")
    ax.legend(fontsize=7.5, loc="lower left", frameon=False)
    fig.tight_layout()
    return fig, ax
def fig_c6_outcome_tree(
    groups, *, pos_prospect: float, p_well: float, pc_well: float | None = None,
    volumes: dict | None = None, dark: bool = False,
):
    """C6, for the export path. Twin of ``pfig_c6_outcome_tree``.

    Shares from ``Groups.risked_shares`` and nowhere else -- see the plotly twin for
    why that matters on this figure in particular.
    """
    p = palette(dark)
    s = groups.risked_shares(pos_prospect, p_well)
    disc = float(p_well)
    commercial = float(pc_well) if pc_well is not None else None
    leaves = [
        ("Chance failure — no hydrocarbons anywhere", s["chance_failure"], "muted"),
        ("Dry hole, hydrocarbons up-dip", s["dry_with_attic"], "up_dip"),
    ]
    if commercial is None:
        leaves.append(("Discovery", disc, "well_associated"))
    else:
        leaves.append(("Discovery, below MEFS", max(disc - commercial, 0.0), "tested"))
        leaves.append(("Discovery, commercial", commercial, "commercial"))

    fig, ax = new_figure(figsize=(8.4, 2.8), dark=dark)
    left = 0.0
    for value, role, label in ((1.0 - pos_prospect, "muted", f"no HC {1 - pos_prospect:.1%}"),
                               (pos_prospect, "prospect", f"HC present {pos_prospect:.1%}")):
        ax.barh("Prospect", value, left=left, color=colour(role, dark), alpha=0.5,
                edgecolor=colour(role, dark), lw=1)
        if value > 0.06:
            ax.text(left + value / 2, 1, label, ha="center", va="center", fontsize=7.5,
                    color=p["text"])
        left += value

    left = 0.0
    for name, value, role in leaves:
        ax.barh("This well", value, left=left, color=colour(role, dark), alpha=0.75,
                edgecolor=colour(role, dark), lw=1, label=name)
        if value > 0.06:
            ax.text(left + value / 2, 0, f"{value:.1%}", ha="center", va="center",
                    fontsize=7.5, color=p["text"])
        left += value

    ax.set_xlim(0, 1)
    ax.set_xlabel("Share of outcomes")
    ax.xaxis.set_major_formatter(lambda v, _pos: f"{v:.0%}")
    ax.set_title("C6 · What happens if this well is drilled")
    ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.28),
              ncol=2, frameon=False)
    fig.tight_layout()
    return fig, ax



def fig_c5_partitions(
    ad, *, z_entry: float, z_exit: float, z_contact: float,
    area_scale: str = "area", dark: bool = False,
):
    """C5, for the export path. Twin of ``pfig_c5_partitions``.

    The same closure cut twice: Rose at the well, this app at the penetrated interval.
    The violet band is the entry-to-exit slice the two disagree about.
    """
    # **The contact must lie below the cut, or there is nothing to partition.** This is
    # what made the first version look broken (Lars, 2026-08-15): it was called with the
    # median *successful* contact, 2203.3 m on prospect B, against an entry of 2205 --
    # so the figure was drawn for a **dry hole**, the below-the-cut region was empty,
    # and only the upper half of each panel appeared. Callers pass the median contact
    # among *discoveries*, which is deeper than the entry by construction.
    if float(z_contact) <= float(z_entry):
        raise ValueError(
            f"z_contact {z_contact:,.1f} is at or above z_entry {z_entry:,.1f}: that is "
            f"a dry hole, and neither partition has a below-the-cut part to draw"
        )
    p = palette(dark)
    label, transform = AREA_SCALES.get(area_scale, AREA_SCALES["area"])
    lkh = min(float(z_contact), float(z_exit))

    fig, axes = new_figure(nrows=1, ncols=2, figsize=(9.0, 4.6), dark=dark,
                           sharey=True)
    z = np.linspace(ad.shallowest, float(z_contact), 260)
    half = transform(np.asarray([ad.area_at(v) for v in z], dtype=float)) / 2.0

    panels = [
        ("Rose — cut at the well", float(z_entry),
         "Rose updip", "up_dip", "Rose downdip", "below_lkh"),
        ("This app — cut at the penetrated interval", lkh,
         "Proven", "proven", "Unproven below LKH", "below_lkh"),
    ]
    for ax, (title, cut, up_name, up_role, lo_name, lo_role) in zip(axes, panels):
        for name, role, lo_z, hi_z in ((up_name, up_role, z.min(), cut),
                                       (lo_name, lo_role, cut, float(z_contact))):
            m = (z >= lo_z) & (z <= hi_z)
            if m.sum() < 2:
                continue
            ax.fill_betweenx(z[m], -half[m], half[m], color=colour(role, dark),
                             alpha=0.55, lw=1.0, edgecolor=colour(role, dark),
                             label=name)
        ax.axhline(cut, color=p["well"], lw=2)
        if z_exit > z_entry:
            ax.axhspan(float(z_entry), min(float(z_exit), float(z_contact)),
                       color=p["well"], alpha=0.16, lw=0)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel(label)
        # ylabel is depth_axis's *first* argument; the second panel drops it, which
        # is the documented convention for a row sharing one depth range.
        depth_axis(ax, None if ax is axes[1] else "Depth (m TVDSS)",
                   (ad.shallowest, float(z_contact)))
        ax.legend(fontsize=7, loc="lower right", frameon=False)

    fig.suptitle("C5 · Two cuts of one closure — the difference is the "
                 f"{z_exit - z_entry:,.0f} m the well penetrates", fontsize=11)
    fig.tight_layout()
    return fig, axes[0]




def fig_b14_hurdle_cost(hurdle, *, current: float | None = None, label_every: int = 7,
                        dark: bool = False):
    """B14, for the export path. Twin of ``pfig_b14_hurdle_cost``.

    Sweeps the requirement rather than the depth. See the plotly twin for why the
    falling ``Pc`` curve is the point.
    """
    p = palette(dark)
    ok = hurdle.feasible
    x = np.asarray(hurdle.confidence, dtype=float) * 100.0

    fig, ax = new_figure(figsize=(7.2, 4.4), dark=dark)
    for values, name, role, ls, lw in (
        (hurdle.p_well, "P_well available under the hurdle", "well_associated", "-", 2.2),
        (hurdle.pc, "Pc — commercial chance there", "minimum", "--", 2.2),
    ):
        v = np.asarray(values, dtype=float) * 100.0
        ax.plot(x[ok], v[ok], color=colour(role, dark), lw=lw, ls=ls, label=name)

    pwv = np.asarray(hurdle.p_well, dtype=float) * 100.0
    for i in range(0, x.size, max(1, label_every)):
        if ok[i]:
            ax.annotate(f"{hurdle.depth[i]:,.0f}", (x[i], pwv[i]),
                        textcoords="offset points", xytext=(0, 6), ha="center",
                        fontsize=6.5, color=p["text_secondary"])

    if current is not None:
        ax.axvline(float(current) * 100.0, color=p["well"], ls=":", lw=1.2)

    ax.set_ylim(bottom=0)
    ax.set_xlabel("Confidence insisted on — P(discovery clears MEFS) (%)")
    ax.set_ylabel("Probability (%)")
    ax.set_title("B14 · What the commerciality hurdle costs")
    ax.legend(fontsize=7.5, frameon=False, loc="upper right")
    fig.tight_layout()
    return fig, ax
