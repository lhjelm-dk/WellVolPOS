"""Phase 1 figures: the reference-engine story, A3/A4/A5/B3.

Every figure here is built on :func:`wellvolpos.viz.theme.new_figure` so
palette and rcParams can never drift between figures, and every axis that
carries a depth goes through :func:`wellvolpos.viz.theme.depth_axis`. A5 is
the one figure in this set with no depth on either axis (it is volume vs
exceedance probability), so it is the one figure here that does not call it
-- see ``tests/test_axes.py`` and ``tests/test_figures.py``.

Colour is assigned by meaning throughout, per ``theme.ROLES``: the two chance
curves in A3 and the single curve in B3 both use the discovery/chance blue
(distinguished by line style, not colour, since both are chances); A4's
percentile trend uses the prospect aqua because it characterises the whole
un-cut model, not any one outcome; A5's four series map onto the four
canonical roles directly.
"""

from __future__ import annotations

import numpy as np

from ..core.classes import VolumeClasses
from ..core.groups import Groups
from ..core.sweep import Sweep
from ..io.adapters.base import TrialSet
from .theme import SEQUENTIAL_CMAP, colour, depth_axis, new_figure, palette

__all__ = [
    "fig_a3_chance_decomposition",
    "fig_a4_resource_vs_depth",
    "fig_a5_exceedance",
    "fig_b3_uncertainty_reduction",
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


def fig_a3_chance_decomposition(
    sweep: Sweep, *, pos_trials: float | None = None, current_z: float | None = None, dark: bool = False,
):
    """P_well and r_location vs entry depth, POS_trials as a rule.

    Both curves are the same quantity's family -- a chance -- so both use
    the discovery/chance blue; solid for P_well, dashed for r_location. This
    is the figure that makes the decomposition in CLAUDE.md's "one idea
    everything rests on" impossible to misread: the two lines never touch
    except at the crest, and only P_well answers "will this well work".
    """
    fig, ax = new_figure(figsize=(6, 5), dark=dark)
    p = palette(dark)
    c = colour("p_well", dark)

    ax.plot(sweep.p_well * 100.0, sweep.z, color=c, lw=2.0, label=r"$P_{well}$ = POS $\times$ r")
    ax.plot(sweep.r_location * 100.0, sweep.z, color=c, lw=1.4, ls="--",
            label="r = P(contact deeper | HC present)")

    if pos_trials is not None:
        ax.axvline(pos_trials * 100.0, color=p["muted"], ls=":", lw=1.0)
        ax.text(pos_trials * 100.0, sweep.z.min(), f" POS$_{{trials}}$ {pos_trials:.3f}",
                rotation=90, va="top", ha="right", fontsize=7, color=p["text_secondary"])

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
    n_bins: int = 40, dark: bool = False,
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

    depth_axis(ax, ylabel="HC-water contact (m TVDSS)")
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
