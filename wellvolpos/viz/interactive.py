"""Interactive (plotly) versions of A1-A6 and B0-B5.

CLAUDE.md: *"Use plotly for the interactive figures and matplotlib for the
export path, both driven from ``viz/theme.py`` so they cannot drift apart."*
This module is the interactive half; :mod:`wellvolpos.viz.figures` is the
export half. Neither makes styling choices of its own -- palette, panel height
and the depth rule all come from :mod:`wellvolpos.viz.theme`.

**Figures stay individual.** Each function returns one standalone
``plotly.graph_objects.Figure`` rather than a panel of a merged subplot grid,
so any figure can be dropped anywhere in the app, exported alone, or read on
its own. A row is made readable across instead by giving every figure in it the
same ``zlim`` and the same height: pass the row's shared depth range to each
call and the depths line up, which is what non-negotiable 2 asks for. The only
figure with internal panels is B5, whose three schemes side by side *are* the
figure.

Hover is where the interactive path earns its keep: the whole point of A5 and
B2 is reading a probability off a curve at a volume you care about, which on a
static image means holding a ruler to the screen. Every trace therefore carries
an explicit ``hovertemplate`` in domain units.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..core.chance import ELEMENTS, SCHEME_LABELS, SHIPPED_SCHEMES, allocate
from ..core.chance import waterfall_steps as chance_waterfall_steps
from ..core.classes import VolumeClasses
from ..core.groups import Groups
from ..core.stats import MIN_SUPPORT, thin
from ..core.structure import AreaDepth
from ..core.sweep import (
    Sweep,
    VolumeSweep,
    find_crossing,
    invert_volume_target,
    volume_target_band,
    volume_target_curve,
)
from ..io.adapters.base import TrialSet
from .figures import _depth_band, _exceedance, area_spread_is_material
from .theme import (
    PANEL_HEIGHT,
    SEQUENTIAL_CMAP,
    apply_plotly,
    colour,
    depth_axis_plotly,
    palette,
    rgba,
)

__all__ = [
    "pfig_concepts",
    "pfig_map_view",
    "pfig_a1_area_depth",
    "pfig_a2_outcome_tree",
    "pfig_a3_chance_decomposition",
    "pfig_a4_resource_vs_depth",
    "pfig_a5_exceedance",
    "pfig_a6_overlap",
    "pfig_b0_section",
    "pfig_b1_volume_split",
    "pfig_b2_chance_vs_regret",
    "pfig_b3_uncertainty_reduction",
    "pfig_b4_chance_waterfall",
    "pfig_b5_allocation_dumbbell",
    "pfig_b6_inverse",
    "row_zlim",
]

DEPTH_HOVER = "%{y:.0f} m TVDSS"


def row_zlim(*ranges: tuple[float, float] | None, pad_frac: float = 0.0) -> tuple[float, float]:
    """The shallowest-to-deepest envelope of everything in a row.

    Call once per row and hand the result to every figure in that row. Doing it
    here rather than in each figure is deliberate: a figure cannot know what it
    is sitting next to, so a row can only be made level by the caller that
    lays it out.
    """
    los = [float(min(r)) for r in ranges if r is not None]
    his = [float(max(r)) for r in ranges if r is not None]
    if not los:
        raise ValueError("row_zlim needs at least one range")
    lo, hi = min(los), max(his)
    pad = pad_frac * (hi - lo)
    return lo - pad, hi + pad


def _hline(fig, y: float, colour_: str, dash: str = "dash", label: str | None = None):
    # The annotation kwargs are passed *only* when there is a label. Passing
    # annotation_text=None alongside annotation_position makes plotly create an
    # annotation anyway and fill it with its placeholder, "new text" -- which
    # then appears on every unlabelled entry/exit rule in the app.
    kw = dict(
        annotation_text=label, annotation_position="top left", annotation_font_size=10
    ) if label else {}
    fig.add_hline(y=y, line=dict(color=colour_, width=1.2, dash=dash), **kw)
    return fig


def _vline(fig, x: float, colour_: str, dash: str = "dot", label: str | None = None):
    kw = dict(
        annotation_text=label, annotation_position="top", annotation_font_size=10
    ) if label else {}
    fig.add_vline(x=x, line=dict(color=colour_, width=1.0, dash=dash), **kw)
    return fig


# ------------------------------------------------------------------- A1
def pfig_a1_area_depth(
    ad: AreaDepth, *, ts: TrialSet | None = None,
    current_entry: float | None = None, current_exit: float | None = None,
    n_bins: int = 40, zlim: tuple[float, float] | None = None,
    show_depth_labels: bool = True, dark: bool = False, height: int | None = PANEL_HEIGHT,
):
    """A1 -- the area-depth curve recovered from the trials, entry/exit marked.

    Pass ``ts`` to show the *area uncertainty* at each depth as well: P90 / P50 /
    P10 of the sampled area within equal-count depth bins, thin and grey, against
    the mean in the prospect colour. On a GeoX run where productive area is a
    deterministic function of contact depth -- the reference file fits at isotonic
    R² = 0.9999999987 -- the three grey curves land on the mean, and the subtitle
    says so rather than leaving a reader to assume the spread is real but small.
    """
    p = palette(dark)
    fig = go.Figure()

    # bool(), because TrialSet.has returns numpy's bool and plotly's validators
    # reject np.True_ for showlegend.
    with_area = bool(ts is not None and ts.has("area"))
    subtitle = ""
    if with_area:
        contact, area = ts.col("contact"), ts.col("area")
        ok = area > 0
        zb, a90, a50, amean, a10 = _depth_band(contact[ok], area[ok], n_bins=n_bins)
        material, rel_resid = area_spread_is_material(ad)
        # Thin grey for the percentile family, per Lars: the mean is the number
        # that gets quoted, so it keeps the colour and the weight.
        for values, name in ((a90, "P90"), (a50, "P50"), (a10, "P10")):
            fig.add_scatter(
                x=values, y=zb, mode="lines", name=name,
                line=dict(color=p["muted"], width=1),
                hovertemplate=name + " %{x:.3f} km² in this depth bin<extra></extra>",
            )
        fig.add_scatter(
            x=amean, y=zb, mode="lines", name="Mean area",
            line=dict(color=colour("prospect", dark), width=2.5),
            hovertemplate="mean %{x:.3f} km² at " + DEPTH_HOVER + "<extra></extra>",
        )
        subtitle = (
            "<br><sub>"
            + (
                f"area scatter about A(z) is {rel_resid:.1%} of the mean — real area uncertainty"
                if material else
                "area is a deterministic function of contact depth here, so the P90–P10 "
                "spread shown is the depth range within each bin, not area uncertainty"
            )
            + "</sub>"
        )
    else:
        fig.add_scatter(
            x=ad.a, y=ad.z, mode="lines", name="A(z)",
            line=dict(color=colour("prospect", dark), width=2.5),
            hovertemplate="%{x:.3f} km² at " + DEPTH_HOVER + "<extra></extra>",
        )

    if current_entry is not None:
        _hline(fig, current_entry, p["well"], "dash", "well entry")
    if current_exit is not None and current_exit != current_entry:
        _hline(fig, current_exit, p["well"], "dot", "well exit")

    fig.update_layout(
        title=f"A1 · Area–depth curve (isotonic R² = {ad.r2:.6f}){subtitle}",
        xaxis_title="Productive area (km²)",
        showlegend=with_area,
    )
    fig.update_xaxes(rangemode="tozero")
    apply_plotly(fig, dark, height)
    depth_axis_plotly(fig, zlim or (ad.shallowest, ad.deepest), show_ticklabels=show_depth_labels)
    return fig


# --------------------------------------------------------------- map view
def pfig_map_view(
    ad: AreaDepth, *, apex: float, z_entry: float, z_exit: float | None = None,
    interval: float = 50.0, well_azimuth_deg: float = 35.0,
    dark: bool = False, height: int | None = PANEL_HEIGHT,
):
    """A conceptual map of the closure: apex at the centre, contours from A(z).

    **A cartoon, not a map.** Each contour is the circle enclosing the area A(z)
    holds at that depth, so the *areas* and the contour spacing are faithful
    while the shape is not -- A(z) says nothing about the outline of the closure
    or which way it is elongated. The well's map position is likewise arbitrary:
    only its *radius* means anything, and it means the well sits on the contour
    of its own reservoir entry depth.

    What the picture is for is seeing at a glance how much of the closure lies
    inside the entry contour -- the part a dry hole would leave up-dip -- against
    how much lies outside it.

    Contours sit on **round absolute depths** -- multiples of ``interval``, so a
    25 m interval gives 3225, 3250, 3275 and a 100 m one gives 3300, 3400, 3500 --
    rather than being stepped off the apex. The apex is usually an estimate, and
    contours referenced to it would shift every time it was nudged; round depths
    stay put and can be read against a depth map on the same datum. The outermost
    ring is the deepest sampled contact, which is not a round number and is
    labelled as the data's base rather than as a contour.

    Contours shallower than the shallowest sampled contact are dotted: the trials
    never reached the crest, so their area is a taper to the apex rather than
    anything the model states (see :meth:`AreaDepth.area_at_tapered`).
    """
    p = palette(dark)
    contours = ad.contour_radii(apex, interval=interval, z_max=ad.deepest)
    theta = np.linspace(0.0, 2.0 * np.pi, 181)
    fig = go.Figure()

    # Deepest first, so the shallow rings draw on top of it.
    rings = sorted(
        zip(contours.depths, contours.radii, contours.extrapolated, contours.at_data_limit),
        key=lambda t: -t[0],
    )
    for zz, rr, is_extrap, is_limit in rings:
        inside_well = zz <= z_entry
        fig.add_scatter(
            x=rr * np.cos(theta), y=rr * np.sin(theta), mode="lines",
            line=dict(
                color=colour("attic", dark) if inside_well else colour("prospect", dark),
                width=2.5 if is_limit else 1.2,
                dash="dot" if is_extrap else "solid",
            ),
            name=f"{zz:.0f} m", showlegend=False,
            hovertemplate=(
                f"{zz:.1f} m TVDSS<br>{np.pi * rr * rr:.3f} km² enclosed"
                + ("<br>deepest sampled contact" if is_limit else "")
                + ("<br>extrapolated above sampled range" if is_extrap else "")
                + "<extra></extra>"
            ),
        )

    # The three areas the well divides the closure into, in plan view: the same
    # split B0 draws in section, so the two figures colour-key identically.
    r_entry = ad.radius_at(z_entry, apex)
    r_exit = ad.radius_at(z_exit, apex) if z_exit is not None else r_entry
    r_base = float(contours.radii.max()) if contours.radii.size else r_exit

    def annulus(r_in: float, r_out: float, role: str, name: str, hover: str) -> None:
        """A ring, as one closed path: out around the rim, back around the hole."""
        if r_out <= r_in + 1e-12:
            return
        xs = np.concatenate([r_out * np.cos(theta), r_in * np.cos(theta[::-1])])
        ys = np.concatenate([r_out * np.sin(theta), r_in * np.sin(theta[::-1])])
        fig.add_scatter(
            x=xs, y=ys, mode="lines", fill="toself", fillcolor=rgba(role, 0.35, dark),
            line=dict(width=0), name=name,
            hovertemplate=hover + "<extra></extra>",
        )

    a_attic = np.pi * r_entry ** 2
    a_proven = max(np.pi * (r_exit ** 2 - r_entry ** 2), 0.0)
    a_possible = max(np.pi * (r_base ** 2 - r_exit ** 2), 0.0)

    # Inside the entry contour: nothing the well touches, so it is the attic a
    # dry hole leaves behind.
    fig.add_scatter(
        x=r_entry * np.cos(theta), y=r_entry * np.sin(theta), mode="lines",
        fill="toself", fillcolor=rgba("attic", 0.35, dark),
        line=dict(color=colour("attic", dark), width=2.5),
        name=f"Potential attic — up-dip of entry ({a_attic:.2f} km²)",
        hovertemplate=f"potential attic<br>{a_attic:.3f} km² up-dip of the {z_entry:.0f} m entry<extra></extra>",
    )
    # Without an exit depth there is no tested band, and everything outside the
    # entry contour is untested rather than "below exit".
    exit_label = f"{z_exit:.0f} m" if z_exit is not None else "the entry"
    annulus(
        r_entry, r_exit, "tested",
        f"Potentially proven — entry to exit ({a_proven:.2f} km²)",
        f"potentially proven<br>{a_proven:.3f} km² between {z_entry:.0f} m and {exit_label}",
    )
    annulus(
        r_exit, r_base, "possible",
        f"Possible — below exit ({a_possible:.2f} km²)",
        f"possible below exit<br>{a_possible:.3f} km² below {exit_label}",
    )

    ang = np.deg2rad(well_azimuth_deg)
    fig.add_scatter(
        x=[r_entry * np.cos(ang)], y=[r_entry * np.sin(ang)], mode="markers+text",
        marker=dict(symbol="circle-open-dot", size=14,
                    line=dict(color=p["well"], width=3), color=p["well"]),
        text=["  WELL"], textposition="middle right",
        textfont=dict(size=11, color=p["well"]), name="Well",
        hovertemplate=f"well on the {z_entry:.0f} m contour<br>map position arbitrary<extra></extra>",
    )
    fig.add_scatter(
        x=[0.0], y=[0.0], mode="markers+text",
        marker=dict(symbol="x-thin", size=11, line=dict(color=p["text"], width=2.5)),
        text=[f"  apex {apex:.0f} m"], textposition="middle right",
        textfont=dict(size=9, color=p["text_secondary"]), showlegend=False,
        hovertemplate=f"apex {apex:.0f} m TVDSS (derived from A(z))<extra></extra>",
    )

    lim = r_base * 1.12 if r_base > 0 else 1.0
    fig.update_layout(
        title=(
            f"Conceptual map view — contours on {interval:.0f} m multiples "
            f"(deepest sampled contact {contours.depths[-1]:.0f} m)"
        ),
        xaxis_title="km east of apex (equivalent-circle radius — shape is illustrative)",
        legend=dict(font=dict(size=9), yanchor="bottom", y=0.01, xanchor="left", x=0.01),
    )
    fig.update_xaxes(range=[-lim, lim], constrain="domain")
    # Equal aspect, so a contour enclosing twice the area looks twice the area.
    # scaleanchor is the only way to hold that through a resize.
    fig.update_yaxes(range=[-lim, lim], scaleanchor="x", scaleratio=1,
                     title="km north of apex", showticklabels=True)
    apply_plotly(fig, dark, height)
    # Legend inside the axes and given a background, like A2: this figure's
    # fills reach the corners, and apply_plotly owns placement.
    fig.update_layout(legend=dict(
        bgcolor="rgba(252,252,251,0.80)" if not dark else "rgba(26,26,25,0.80)",
        yanchor="bottom", y=0.01, xanchor="left", x=0.01, font=dict(size=9),
    ))
    return fig


# ------------------------------------------------------------------- A2
def pfig_a2_outcome_tree(
    sweep: Sweep, *, current_z: float | None = None, zlim: tuple[float, float] | None = None,
    show_depth_labels: bool = True, dark: bool = False, height: int | None = PANEL_HEIGHT,
):
    """A2 -- the four outcomes vs entry depth, as stacked bands summing to 100 %.

    The shares come from :class:`wellvolpos.core.sweep.Sweep`, already risked
    onto the entered POS, so this figure cannot disagree with A3 beside it.
    """
    p = palette(dark)
    z = sweep.z
    cum0 = np.full_like(z, sweep.share_chance_failure * 100.0)
    cum1 = cum0 + sweep.share_dry_with_attic * 100.0
    cum2 = cum1 + sweep.share_contact_seen * 100.0
    cum3 = cum2 + sweep.share_hc_to_exit * 100.0

    bands = [
        (np.zeros_like(z), cum0, "Chance failure", p["muted"]),
        (cum0, cum1, "Dry, with attic", colour("attic", dark)),
        (cum1, cum2, "Discovery, contact seen", colour("tested", dark)),
        (cum2, cum3, "Discovery, HC to exit", colour("possible", dark)),
    ]
    fig = go.Figure()
    for lower, upper, name, col in bands:
        # Closed polygon rather than fill='tonextx': explicit, and immune to
        # trace ordering.
        fig.add_scatter(
            x=np.concatenate([lower, upper[::-1]]),
            y=np.concatenate([z, z[::-1]]),
            fill="toself", fillcolor=col, mode="lines",
            line=dict(width=0), name=name, hoverinfo="skip",
        )
    # An invisible trace carrying the real numbers, so hovering reads out the
    # four shares at one depth instead of a polygon vertex.
    fig.add_scatter(
        x=cum3, y=z, mode="lines", line=dict(width=0), showlegend=False,
        customdata=np.column_stack([
            np.full_like(z, sweep.share_chance_failure) * 100.0,
            sweep.share_dry_with_attic * 100.0,
            sweep.share_contact_seen * 100.0,
            sweep.share_hc_to_exit * 100.0,
        ]),
        hovertemplate=(
            DEPTH_HOVER
            + "<br>chance failure %{customdata[0]:.1f}%"
            + "<br>dry with attic %{customdata[1]:.1f}%"
            + "<br>contact seen %{customdata[2]:.1f}%"
            + "<br>HC to exit %{customdata[3]:.1f}%<extra></extra>"
        ),
    )
    if current_z is not None:
        _hline(fig, current_z, p["text"], "dash")

    fig.update_layout(
        title=f"A2 · Outcome tree vs location (exit = entry + {sweep.z_gap:.0f} m)",
        xaxis_title="Share of trials (%)",
    )
    fig.update_xaxes(range=[0, 100])
    apply_plotly(fig, dark, height)
    # After apply_plotly, which owns legend *placement* -- keeping it inside the
    # axes for every panel is what stops one figure's legend expanding its
    # margins and knocking a row out of alignment. Only the background is
    # overridden here: this is the one figure whose bands fill the whole plot,
    # so the house-style transparent legend would sit unreadably on top of them.
    fig.update_layout(legend=dict(bgcolor="rgba(252,252,251,0.78)" if not dark else "rgba(26,26,25,0.78)",
                                  font=dict(size=9)))
    depth_axis_plotly(fig, zlim or (float(z.min()), float(z.max())), show_ticklabels=show_depth_labels)
    return fig


# ------------------------------------------------------------------- A3
def pfig_a3_chance_decomposition(
    sweep: Sweep, *, pos_prospect: float | None = None, pos_trials: float | None = None,
    current_z: float | None = None, zlim: tuple[float, float] | None = None,
    show_depth_labels: bool = True, dark: bool = False, height: int | None = PANEL_HEIGHT,
):
    """A3 -- P_well and r_location vs entry depth, POS as a rule.

    Both curves are chances, so both take the chance blue and are separated by
    line style, not colour.
    """
    p = palette(dark)
    c = colour("p_well", dark)
    fig = go.Figure()
    fig.add_scatter(
        x=sweep.p_well * 100.0, y=sweep.z, mode="lines", name="P<sub>well</sub> = POS × r",
        line=dict(color=c, width=3),
        hovertemplate="P<sub>well</sub> %{x:.1f}% at " + DEPTH_HOVER + "<extra></extra>",
    )
    fig.add_scatter(
        x=sweep.r_location * 100.0, y=sweep.z, mode="lines", name="r = P(contact deeper | HC)",
        line=dict(color=c, width=2, dash="dash"),
        hovertemplate="r %{x:.1f}% at " + DEPTH_HOVER + "<extra></extra>",
    )
    if pos_prospect is not None:
        _vline(fig, pos_prospect * 100.0, p["muted"], "dot", f"POS {pos_prospect:.3f}")
    if pos_trials is not None and (pos_prospect is None or abs(pos_trials - pos_prospect) > 1e-9):
        _vline(fig, pos_trials * 100.0, p["muted"], "dashdot", f"POS trials {pos_trials:.3f}")
    if current_z is not None:
        _hline(fig, current_z, p["text_secondary"], "dash")

    fig.update_layout(title="A3 · Chance decomposition vs location", xaxis_title="Probability (%)")
    fig.update_xaxes(range=[0, 100])
    apply_plotly(fig, dark, height)
    depth_axis_plotly(fig, zlim or (float(sweep.z.min()), float(sweep.z.max())),
                      show_ticklabels=show_depth_labels)
    return fig


# ------------------------------------------------------------------- A4
def pfig_a4_resource_vs_depth(
    ts: TrialSet, *, current_entry: float | None = None, current_exit: float | None = None,
    mefs: float | None = None,
    n_bins: int = 40, gridsize: int = 60, zlim: tuple[float, float] | None = None,
    show_depth_labels: bool = True, dark: bool = False, height: int | None = PANEL_HEIGHT,
):
    """A4 -- log-density heatmap of resource vs contact depth, with P90/P50/P10.

    Success trials only: the chance-failure zeros belong to POS, not to the
    shape of the resource distribution.
    """
    res, contact = ts.col("resource"), ts.col("contact")
    succ = res > 0.0
    x, y = res[succ], contact[succ]
    p = palette(dark)

    counts, xedges, yedges = np.histogram2d(x, y, bins=gridsize)
    with np.errstate(divide="ignore"):
        dens = np.log10(counts.T)
    dens[~np.isfinite(dens)] = np.nan

    fig = go.Figure()
    fig.add_heatmap(
        x=0.5 * (xedges[:-1] + xedges[1:]), y=0.5 * (yedges[:-1] + yedges[1:]),
        z=dens, colorscale="Blues", showscale=True,
        colorbar=dict(title=dict(text="log₁₀ n", side="right"), thickness=12, len=0.6),
        hovertemplate="%{x:.1f} MMboe at " + DEPTH_HOVER + "<br>log₁₀ n %{z:.2f}<extra></extra>",
    )
    # Same convention as A1: the mean keeps the prospect colour and the weight,
    # because it is the number that gets quoted; the percentile family is thin
    # and grey. Note the mean is *not* the P50 on a skewed resource
    # distribution, which is half the reason for showing both.
    zb, p90, p50, pmean, p10 = _depth_band(y, x, n_bins=n_bins)
    for values, name in ((p90, "P90"), (p50, "P50"), (p10, "P10")):
        fig.add_scatter(
            x=values, y=zb, mode="lines", name=name,
            line=dict(color=p["muted"], width=1),
            hovertemplate=name + " %{x:.2f} MMboe at " + DEPTH_HOVER + "<extra></extra>",
        )
    fig.add_scatter(
        x=pmean, y=zb, mode="lines", name="Mean",
        line=dict(color=colour("prospect", dark), width=2.5),
        hovertemplate="mean %{x:.2f} MMboe at " + DEPTH_HOVER + "<extra></extra>",
    )
    # Named, and both of them: an unlabelled rule at the entry depth was
    # indistinguishable from the exit, and the exit was not drawn at all.
    if current_entry is not None:
        _hline(fig, current_entry, p["well"], "dash", "well entry")
    if current_exit is not None and current_exit != current_entry:
        _hline(fig, current_exit, p["well"], "dot", "well exit")
    if mefs is not None:
        _vline(fig, mefs, p["muted"], "dot", "MEFS")

    fig.update_layout(title="A4 · Resource vs contact depth", xaxis_title="Recoverable resource (MMboe)")
    fig.update_xaxes(rangemode="tozero")
    apply_plotly(fig, dark, height)
    depth_axis_plotly(fig, zlim or (float(y.min()), float(y.max())),
                      title="HC-water contact (m TVDSS)", show_ticklabels=show_depth_labels)
    return fig


# ------------------------------------------------------------------- A5
def pfig_a5_exceedance(
    ts: TrialSet, groups: Groups, vc: VolumeClasses, *, mefs: float | None = None,
    dark: bool = False, height: int | None = PANEL_HEIGHT,
):
    """A5 -- exceedance curves at the chosen location. No depth on either axis.

    The money chart, and the one that most wants a cursor: hover reads the
    probability of exceeding any volume directly off each curve.
    """
    res = ts.col("resource")
    p = palette(dark)
    fig = go.Figure()
    series = [
        ("Prospect (all trials)", res, "prospect"),
        ("Discovery case", res[groups.discovery], "discovery"),
        ("Proven at well", vc.proven[groups.discovery], "proven"),
        ("Attic | dry hole", res[groups.dry_with_attic], "attic"),
    ]
    for name, values, role in series:
        v, pct = _exceedance(values)
        if v.size == 0:
            continue
        fig.add_scatter(
            x=v, y=pct, mode="lines", name=name, line=dict(color=colour(role, dark), width=2.5),
            hovertemplate=name + "<br>%{y:.1f}% chance of exceeding %{x:.2f} MMboe<extra></extra>",
        )
    if mefs is not None:
        _vline(fig, mefs, p["muted"], "dot", "MEFS")

    fig.update_layout(
        title="A5 · Exceedance curves at the chosen location",
        xaxis_title="Recoverable resource (MMboe)",
        yaxis_title="Probability of exceedance (%)",
    )
    fig.update_xaxes(rangemode="tozero")
    fig.update_yaxes(range=[0, 105])
    apply_plotly(fig, dark, height)
    return fig


# ------------------------------------------------------------------- A6
def pfig_a6_overlap(
    vc: VolumeClasses, groups: Groups, *, mefs: float | None = None, bins: int = 40,
    dark: bool = False, height: int | None = PANEL_HEIGHT,
):
    """A6 -- Schneider et al.'s "surprising overlap", proven against attic.

    Densities, not counts: the two groups have different n (4 576 against
    3 029 here) and the figure is about the shape overlap.
    """
    proven = vc.proven[groups.discovery]
    attic = vc.attic[groups.dry_with_attic]
    p = palette(dark)
    hi = max(float(proven.max()) if proven.size else 0.0,
             float(attic.max()) if attic.size else 0.0, 1.0)
    size = hi / bins

    fig = go.Figure()
    if attic.size:
        fig.add_histogram(
            x=attic, name=f"Attic | dry hole (n={attic.size:,})", histnorm="probability density",
            marker_color=colour("attic", dark), opacity=0.6,
            xbins=dict(start=0.0, end=hi, size=size),
            hovertemplate="attic %{x:.1f} MMboe<br>density %{y:.4f}<extra></extra>",
        )
    if proven.size:
        fig.add_histogram(
            x=proven, name=f"Proven | discovery (n={proven.size:,})", histnorm="probability density",
            marker_color=colour("proven", dark), opacity=0.6,
            xbins=dict(start=0.0, end=hi, size=size),
            hovertemplate="proven %{x:.1f} MMboe<br>density %{y:.4f}<extra></extra>",
        )
    if mefs is not None:
        _vline(fig, mefs, p["muted"], "dot", "MEFS")

    fig.update_layout(
        title="A6 · Attic vs proven — the overlap", barmode="overlay",
        xaxis_title="Recoverable resource (MMboe)", yaxis_title="Density",
    )
    apply_plotly(fig, dark, height)
    return fig


# ------------------------------------------------------------------- B0
def pfig_b0_section(
    ad: AreaDepth, *, z_entry: float, z_exit: float, zlim: tuple[float, float] | None = None,
    show_depth_labels: bool = True, title: str = "B0 · Schematic section",
    dark: bool = False, height: int | None = PANEL_HEIGHT,
):
    """B0 -- a schematic section from A(z), colour-keyed to the well's outcomes.

    Width is proportional to sqrt(enclosed area) -- a circular-closure proxy, so
    the shape is illustrative and the axis claims no unit. The depths on y are
    the real quantity.
    """
    p = palette(dark)
    halfwidth = np.sqrt(np.maximum(ad.a, 0.0))
    z = ad.z
    fig = go.Figure()

    def band(lo: float, hi: float, role: str, name: str) -> None:
        m = (z >= lo) & (z <= hi)
        if m.sum() < 2:
            return
        zz, hw = z[m], halfwidth[m]
        fig.add_scatter(
            x=np.concatenate([-hw, hw[::-1]]), y=np.concatenate([zz, zz[::-1]]),
            fill="toself", fillcolor=colour(role, dark), mode="lines", line=dict(width=0),
            name=name, opacity=0.6, hoverinfo="skip",
        )
        fig.add_annotation(
            x=0, y=0.5 * (max(lo, ad.shallowest) + min(hi, ad.deepest)), text=name,
            showarrow=False, font=dict(size=10, color=p["text"]),
        )

    band(ad.shallowest, z_entry, "attic", "attic if dry")
    band(z_entry, z_exit, "proven", "proven")
    band(z_exit, ad.deepest, "possible", "possible below exit")

    for sign in (1, -1):
        fig.add_scatter(x=sign * halfwidth, y=z, mode="lines",
                        line=dict(color=p["text_secondary"], width=1), showlegend=False,
                        hoverinfo="skip")
    fig.add_scatter(
        x=[0, 0], y=[z_entry, z_exit], mode="lines", name="Well",
        line=dict(color=p["well"], width=6),
        hovertemplate="well " + DEPTH_HOVER + "<extra></extra>",
    )

    fig.update_layout(
        title=title, xaxis_title="Schematic width (∝ √area) — not to scale", showlegend=False,
    )
    fig.update_xaxes(showticklabels=False)
    apply_plotly(fig, dark, height)
    depth_axis_plotly(fig, zlim or (ad.shallowest, ad.deepest), show_ticklabels=show_depth_labels)
    return fig


# ------------------------------------------------------------------- B1
def pfig_b1_volume_split(
    vsweep: VolumeSweep, *, current_z: float | None = None, zlim: tuple[float, float] | None = None,
    show_depth_labels: bool = True, min_support: int = MIN_SUPPORT,
    dark: bool = False, height: int | None = PANEL_HEIGHT,
):
    """B1 -- mean proven / possible / attic volume vs entry depth.

    Steps resting on fewer than ``min_support`` trials are left undrawn: the
    discovery group collapses down-dip (8 of 10 000 trials at 3677 m on the
    reference data) and drawing a mean of eight as boldly as a mean of four
    thousand invites the wrong conclusion.
    """
    p = palette(dark)
    fig = go.Figure()
    for values, name, role, dash, width in (
        (thin(vsweep.proven_mean, vsweep.n_discovery, min_support),
         "Proven | discovery", "proven", "solid", 3),
        (thin(vsweep.possible_mean, vsweep.n_discovery, min_support),
         "Possible below exit | discovery", "possible", "dash", 2),
        (thin(vsweep.attic_mean, vsweep.n_dry, min_support),
         "Attic | dry hole", "attic", "solid", 3),
    ):
        fig.add_scatter(
            x=values, y=vsweep.z, mode="lines", name=name,
            line=dict(color=colour(role, dark), width=width, dash=dash),
            hovertemplate=name + "<br>%{x:.2f} MMboe at " + DEPTH_HOVER + "<extra></extra>",
        )
    if current_z is not None:
        _hline(fig, current_z, p["text_secondary"], "dash")

    fig.update_layout(
        title=f"B1 · Volume split vs location (exit = entry + {vsweep.z_gap:.0f} m)",
        xaxis_title="Mean resource (MMboe)",
    )
    fig.update_xaxes(rangemode="tozero")
    apply_plotly(fig, dark, height)
    depth_axis_plotly(fig, zlim or (float(vsweep.z.min()), float(vsweep.z.max())),
                      show_ticklabels=show_depth_labels)
    return fig


# ------------------------------------------------------------------- B2
def pfig_b2_chance_vs_regret(
    vsweep: VolumeSweep, *, current_z: float | None = None, zlim: tuple[float, float] | None = None,
    show_depth_labels: bool = True, min_support: int = MIN_SUPPORT,
    dark: bool = False, height: int | None = PANEL_HEIGHT,
):
    """B2 -- chance against regret vs entry depth; the crossings are the argument.

    The regret curve conditions on the well being dry **and** the prospect
    charged, which is stated in its name: folding the chance failures in
    roughly halves it, and both readings are legitimate answers to different
    questions.
    """
    if vsweep.mefs is None or vsweep.p_proven_exceeds_mefs is None or vsweep.p_attic_exceeds_mefs is None:
        raise ValueError("pfig_b2_chance_vs_regret needs a VolumeSweep run with a mefs threshold")
    p = palette(dark)
    fig = go.Figure()
    # P_well is unconditional, so it is never thinned; the two conditional
    # curves are.
    p_proven = thin(vsweep.p_proven_exceeds_mefs, vsweep.n_discovery, min_support)
    p_attic = thin(vsweep.p_attic_exceeds_mefs, vsweep.n_dry, min_support)
    for values, name, role, width in (
        (vsweep.p_well, "P<sub>well</sub>", "p_well", 3),
        (p_proven, "P(proven > MEFS | discovery)", "proven", 2.5),
        (p_attic, "P(attic > MEFS | dry & charged)", "attic", 2.5),
    ):
        fig.add_scatter(
            x=np.asarray(values) * 100.0, y=vsweep.z, mode="lines", name=name,
            line=dict(color=colour(role, dark), width=width),
            hovertemplate=name + "<br>%{x:.1f}% at " + DEPTH_HOVER + "<extra></extra>",
        )
    # Named for the curves that actually meet -- see the matplotlib twin: these
    # two are not on one scale, so "chance = regret" would be a claim the
    # figure does not support.
    crossing = find_crossing(vsweep.z, vsweep.p_well, p_attic)
    if crossing is not None:
        _hline(fig, crossing, p["text"], "dot",
               f"P<sub>well</sub> = P(attic > MEFS | dry & charged) at {crossing:.0f} m")
    if current_z is not None:
        _hline(fig, current_z, p["text_secondary"], "dash")

    fig.update_layout(
        title=f"B2 · Chance vs regret (MEFS {vsweep.mefs:.1f} MMboe)",
        xaxis_title="Probability (%)",
    )
    fig.update_xaxes(range=[0, 100])
    apply_plotly(fig, dark, height)
    depth_axis_plotly(fig, zlim or (float(vsweep.z.min()), float(vsweep.z.max())),
                      show_ticklabels=show_depth_labels)
    return fig


# ------------------------------------------------------------------- B3
def pfig_b3_uncertainty_reduction(
    sweep: Sweep, *, current_z: float | None = None, zlim: tuple[float, float] | None = None,
    show_depth_labels: bool = True, dark: bool = False, height: int | None = PANEL_HEIGHT,
):
    """B3 -- Haskett (2003) uncertainty reduction vs entry depth, optimum marked.

    The optimum is ``sweep.z_optimum``, found by argmax over the swept grid
    rather than eyeballed.
    """
    p = palette(dark)
    c = colour("p_well", dark)
    fig = go.Figure()
    fig.add_scatter(
        x=sweep.uncertainty_reduction, y=sweep.z, mode="lines", name="Reduction",
        line=dict(color=c, width=3), fill="tozerox",
        fillcolor=rgba("p_well", 0.15, dark),
        hovertemplate="%{x:.1f}% reduction at " + DEPTH_HOVER + "<extra></extra>",
    )
    fig.add_scatter(
        x=[sweep.reduction_optimum], y=[sweep.z_optimum], mode="markers+text",
        marker=dict(color=p["text"], size=9),
        text=[f" max {sweep.reduction_optimum:.0f}% @ {sweep.z_optimum:.0f} m"],
        textposition="middle right", textfont=dict(size=10, color=p["text"]),
        showlegend=False,
        hovertemplate="optimum %{x:.1f}% at " + DEPTH_HOVER + "<extra></extra>",
    )
    if current_z is not None:
        _hline(fig, current_z, p["text_secondary"], "dash")

    top = float(np.nanmax(sweep.uncertainty_reduction)) if np.isfinite(sweep.uncertainty_reduction).any() else 5.0
    fig.update_layout(
        title="B3 · Uncertainty reduction vs location (Haskett 2003)",
        xaxis_title="Expected uncertainty reduction (%)", showlegend=False,
    )
    fig.update_xaxes(range=[0, max(5.0, top * 1.25)])
    apply_plotly(fig, dark, height)
    depth_axis_plotly(fig, zlim or (float(sweep.z.min()), float(sweep.z.max())),
                      show_ticklabels=show_depth_labels)
    return fig


# ------------------------------------------------------------------- B4
def pfig_b4_chance_waterfall(
    elements: dict[str, float], r: float, pos_prospect: float, *,
    scheme: str | dict[str, float] = "none", dark: bool = False,
    height: int | None = PANEL_HEIGHT,
):
    """B4 -- the chance elements then the location factor, on a log scale.

    Steps come from :func:`wellvolpos.core.chance.waterfall_steps`, whose
    factors multiply to ``pos_prospect * r`` exactly, so the total cannot
    disagree with the ``P_well`` shown elsewhere. Location steps keep the
    chance blue -- ``r`` is a chance, and A3 already draws it blue -- and are
    separated by hatching instead, so an allocating scheme still shows how much
    of each element's bar is the location penalty.
    """
    p = palette(dark)
    c = colour("p_well", dark)
    steps = chance_waterfall_steps(elements, r, pos_prospect, scheme)
    labels = [s[0] for s in steps]
    values = [s[1] for s in steps]
    roles = [s[2] for s in steps]

    cum, bottoms, tops = 1.0, [], []
    for v in values:
        bottoms.append(cum)
        cum *= v
        tops.append(cum)
    total = cum

    fig = go.Figure()
    if total <= 0.0:
        fig.add_annotation(x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False,
                           text="r = 0 at this depth<br>P<sub>well</sub> = 0",
                           font=dict(size=13, color=p["text"]))
        fig.update_layout(title="B4 · Chance waterfall")
        fig.update_xaxes(showticklabels=False)
        fig.update_yaxes(showticklabels=False)
        apply_plotly(fig, dark, height)
        return fig

    for label, v, role, b, t in zip(labels, values, roles, bottoms, tops):
        face = p["muted"] if role == "reconcile" else c
        fig.add_bar(
            x=[label], y=[abs(b - t)], base=[min(b, t)], name=label, showlegend=False,
            marker=dict(
                color=face,
                pattern=dict(shape="/" if role == "location" else "",
                             fgcolor=p["surface"], size=4),
            ),
            text=[f"×{v:.3f}"], textposition="outside", textfont=dict(size=10),
            hovertemplate=f"{label}<br>×{v:.4f}<br>running {t:.4f}<extra></extra>",
        )
    fig.add_hline(
        y=total, line=dict(color=p["muted"], width=1, dash="dot"),
        annotation_text=f"P_well = {total:.4f}", annotation_position="top left",
        annotation_font_size=11,
    )
    label = SCHEME_LABELS.get(scheme, "custom weights") if isinstance(scheme, str) else "custom weights"
    fig.update_layout(
        title=f"B4 · Chance waterfall ({label})",
        yaxis_title="Cumulative chance (log scale)", xaxis_title=None, bargap=0.35,
    )
    fig.update_yaxes(type="log")
    fig.update_xaxes(tickangle=-25)
    apply_plotly(fig, dark, height)
    return fig


# ------------------------------------------------------------------- B5
def pfig_b5_allocation_dumbbell(
    elements: dict[str, float], r: float, *, pos_prospect: float | None = None,
    dark: bool = False, height: int | None = PANEL_HEIGHT,
):
    """B5 -- the shipped schemes against the prospect baseline, one row per element.

    The one figure here with internal panels, because three schemes side by
    side *are* the figure. They share one x-axis so they are comparable, every
    scheme lands on the same ``P_well`` (drawn as a rule when ``pos_prospect``
    is given), and reservoir never moves because every shipped scheme gives it
    zero weight.
    """
    p = palette(dark)
    c = colour("p_well", dark)
    schemes = list(SHIPPED_SCHEMES)
    fig = make_subplots(
        rows=1, cols=len(schemes), shared_yaxes=True, horizontal_spacing=0.04,
        subplot_titles=[SCHEME_LABELS.get(s, s) for s in schemes],
    )
    names = [e.capitalize() for e in ELEMENTS]

    for i, scheme in enumerate(schemes, start=1):
        revised, _ = allocate(elements, r, scheme)
        base = [float(elements.get(e, 1.0)) for e in ELEMENTS]
        rev = [revised[e] for e in ELEMENTS]
        for name, b, rv in zip(names, base, rev):
            fig.add_scatter(x=[b, rv], y=[name, name], mode="lines",
                            line=dict(color=c, width=2), showlegend=False,
                            hoverinfo="skip", row=1, col=i)
        fig.add_scatter(
            x=base, y=names, mode="markers", name="Baseline", showlegend=i == 1,
            marker=dict(size=9, color=p["surface"], line=dict(color=p["muted"], width=1.5)),
            hovertemplate="baseline %{y} %{x:.3f}<extra></extra>", row=1, col=i,
        )
        fig.add_scatter(
            x=rev, y=names, mode="markers", name="At the well", showlegend=i == 1,
            marker=dict(size=9, color=c),
            hovertemplate="at the well %{y} %{x:.3f}<extra></extra>", row=1, col=i,
        )
        if pos_prospect is not None:
            fig.add_vline(x=pos_prospect * r, line=dict(color=p["muted"], width=1, dash="dot"),
                          row=1, col=i)
        if scheme == "none":
            fig.add_annotation(
                x=0.5, y=-0.5, text=f"r = {r:.3f} reported separately", showarrow=False,
                font=dict(size=10, color=p["text_secondary"]), row=1, col=i,
            )
        fig.update_xaxes(range=[0, 1.03], title_text="Chance", row=1, col=i)

    fig.update_layout(title="B5 · Allocation dumbbell")
    fig.update_annotations(font_size=10)
    apply_plotly(fig, dark, height)
    return fig


# ------------------------------------------------------------------- B6
def pfig_b6_inverse(
    vsweep: VolumeSweep, *, target: float | None = None, n_targets: int = 40,
    ts: TrialSet | None = None, zlim: tuple[float, float] | None = None,
    show_depth_labels: bool = True, dark: bool = False, height: int | None = PANEL_HEIGHT,
):
    """B6 -- the inverse: volume to prove against the entry depth it demands.

    The workbook's H38-H40 block as a curve, and the fourth question the tool
    exists to answer. Depth on y and inverted like every other depth axis, so
    demanding more volume moves the answer visibly *down* the structure.

    ``P_well`` is the colour of the curve rather than a second y-axis -- dual
    y-axes are forbidden, and the trade is the point: the marker darkens where
    the requirement is cheap in chance and pales where it is expensive. Hover
    gives all three numbers at once, which is what this figure is for.
    """
    p = palette(dark)
    targets, z_req, p_at = volume_target_curve(vsweep, n=n_targets, ts=ts)
    fig = go.Figure()

    if targets.size == 0 or not np.isfinite(z_req).any():
        fig.add_annotation(x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False,
                           text="No proven-volume curve to invert",
                           font=dict(size=13, color=p["text"]))
        fig.update_layout(title="B6 · Inverse — volume to prove")
        apply_plotly(fig, dark, height)
        return fig

    if vsweep.alpha is not None:
        z_lo, z_hi = volume_target_band(vsweep, targets)
        band = np.isfinite(z_lo) & np.isfinite(z_hi)
        if band.any():
            level = 100 * (1 - vsweep.alpha)
            fig.add_scatter(
                x=np.concatenate([targets[band], targets[band][::-1]]),
                y=np.concatenate([z_lo[band], z_hi[band][::-1]]),
                fill="toself", fillcolor=rgba("p_well", 0.15, dark), mode="lines",
                line=dict(width=0), name=f"nominal {level:.0f}% band", hoverinfo="skip",
            )

    ok = np.isfinite(z_req)
    fig.add_scatter(
        x=targets[ok], y=z_req[ok], mode="lines+markers",
        line=dict(color=p["text_secondary"], width=1.2),
        marker=dict(
            size=9, color=p_at[ok] * 100.0, colorscale=SEQUENTIAL_CMAP, cmin=0, cmax=100,
            colorbar=dict(title=dict(text="P<sub>well</sub> (%)", side="right"),
                          thickness=12, len=0.6),
        ),
        name="Required entry",
        customdata=p_at[ok] * 100.0,
        hovertemplate=(
            "to prove %{x:.2f} MMboe<br>enter at " + DEPTH_HOVER
            + "<br>P<sub>well</sub> %{customdata:.1f}%<extra></extra>"
        ),
    )

    if target is not None:
        res = invert_volume_target(vsweep, float(target), ts=ts)
        if res.achievable:
            _vline(fig, float(target), p["muted"], "dot", f"{target:.1f} MMboe")
            fig.add_scatter(
                x=[target], y=[res.z_required], mode="markers+text",
                marker=dict(size=11, color=p["text"], symbol="circle-open", line=dict(width=2.5)),
                text=[f" {res.z_required:.0f} m · P<sub>well</sub> {res.p_well_at:.1%}"],
                textposition="middle right", textfont=dict(size=10, color=p["text"]),
                showlegend=False, hoverinfo="skip",
            )

    fig.update_layout(
        title="B6 · Inverse — where the well must go",
        xaxis_title="Volume to prove — mean proven (MMboe)",
    )
    apply_plotly(fig, dark, height)
    depth_axis_plotly(
        fig, zlim or (float(np.nanmin(z_req)), float(np.nanmax(z_req))),
        title="Required entry depth (m TVDSS)", show_ticklabels=show_depth_labels,
    )
    return fig


# ----------------------------------------------------- the concepts figure
def pfig_concepts(
    ad: AreaDepth, ts: TrialSet, groups: Groups, vc: VolumeClasses, *,
    z_entry: float, z_exit: float,
    pos_prospect: float, p_well: float, mefs: float | None = None,
    dark: bool = False, height: int = 640,
):
    """The teaching figure: the same volumes in section and in distribution.

    A structural section on the left, the matching exceedance curves on the
    right, one colour per concept across both, and braces under the curves
    showing how each range nests inside the next:

        up-dip ⊂ tested by well ⊂ well associated ⊂ prospect

    This is the one figure in the project that is deliberately a *composite*
    rather than a standalone panel, because the pairing is the content. The
    section makes the geometry obvious and the exceedance curves make the
    consequence obvious, and it is seeing them together, in one colour scheme,
    that makes the point land.

    **The exceedance curves are risked**, and that is the whole trick. Plotting
    the *risked* distribution -- zeros standing in for the outcomes that do not
    occur -- makes each curve start at its own chance rather than at 100 %: the
    prospect curve begins at ``pos_prospect``, the well-associated curve at
    ``p_well``. So the two POS values are not annotations bolted on, they are
    where the curves physically start, and the vertical gap between those two
    starts *is* the location penalty. Which is the argument the tool exists to
    make: you drill a well, not a prospect.
    """
    p = palette(dark)
    res = ts.col("resource")
    fig = make_subplots(
        rows=1, cols=2, column_widths=[0.32, 0.68], horizontal_spacing=0.11,
        subplot_titles=("Section through the well", "The same volumes, as exceedance curves"),
    )

    # ---------------------------------------------------------------- section
    halfwidth = np.sqrt(np.maximum(ad.a, 0.0))
    z = ad.z

    def band(lo: float, hi: float, role: str, label: str) -> None:
        m = (z >= lo) & (z <= hi)
        if m.sum() < 2:
            return
        fig.add_scatter(
            x=np.concatenate([-halfwidth[m], halfwidth[m][::-1]]),
            y=np.concatenate([z[m], z[m][::-1]]),
            fill="toself", fillcolor=rgba(role, 0.55, dark), mode="lines",
            line=dict(width=0), showlegend=False, hoverinfo="skip", row=1, col=1,
        )
        fig.add_annotation(
            x=0, y=0.5 * (max(lo, ad.shallowest) + min(hi, ad.deepest)), text=label,
            showarrow=False, font=dict(size=9, color=p["text"]), row=1, col=1,
        )

    band(ad.shallowest, z_entry, "up_dip", "up-dip")
    band(z_entry, z_exit, "tested", "tested")
    band(z_exit, ad.deepest, "possible", "possible")
    for sign in (1, -1):
        fig.add_scatter(x=sign * halfwidth, y=z, mode="lines", showlegend=False,
                        line=dict(color=p["text_secondary"], width=1), hoverinfo="skip",
                        row=1, col=1)
    fig.add_scatter(
        x=[0, 0], y=[z_entry, z_exit], mode="lines", name="Well",
        line=dict(color=p["well"], width=6), showlegend=False,
        hovertemplate="well " + DEPTH_HOVER + "<extra></extra>", row=1, col=1,
    )
    fig.add_annotation(x=0, y=z_entry, text="WELL", showarrow=False, yshift=16,
                       font=dict(size=11, color=p["well"]), row=1, col=1)

    # ------------------------------------------------- risked exceedance curves
    risked = [
        ("Prospect resource potential", res, "prospect"),
        ("Well associated resource potential",
         np.where(groups.discovery, res, 0.0), "well_associated"),
        ("Resource tested by well", np.where(groups.discovery, vc.proven, 0.0), "tested"),
        ("Up-dip volume", np.where(groups.dry_with_attic, res, 0.0), "up_dip"),
    ]
    spans: dict[str, tuple[float, float, str]] = {}
    for name, values, role in risked:
        v, pct = _exceedance(values)
        fig.add_scatter(
            x=v, y=pct, mode="lines", name=name,
            line=dict(color=colour(role, dark), width=2.6),
            hovertemplate=name + "<br>%{y:.1f}% chance of exceeding %{x:.2f} MMboe<extra></extra>",
            row=1, col=2,
        )
        positive = v[v > 0]
        if positive.size:
            spans[name] = (float(positive.min()), float(positive.max()), role)

    # The two POS values, drawn where the curves actually start.
    for value, label, role in (
        (pos_prospect, "Asso. Final Prospect POS", "prospect"),
        (p_well, "Asso. Well POS", "well_associated"),
    ):
        fig.add_hline(
            y=value * 100.0, line=dict(color=colour(role, dark), width=1, dash="dot"),
            annotation_text=f"{label} {value:.0%}", annotation_position="top right",
            annotation_font=dict(size=10, color=colour(role, dark)), row=1, col=2,
        )
    if mefs is not None:
        fig.add_vline(
            x=mefs, line=dict(color=colour("minimum", dark), width=1.2, dash="dot"),
            annotation_text="min. volume", annotation_position="top",
            annotation_font=dict(size=10, color=colour("minimum", dark)), row=1, col=2,
        )

    # ------------------------------------------------------- the nesting braces
    # Below the 0 % line, widest at the bottom, so the containment reads at a
    # glance: each range sits inside the one under it.
    order = ["Up-dip volume", "Resource tested by well",
             "Well associated resource potential", "Prospect resource potential"]
    step, base = 7.5, -9.0
    for i, name in enumerate(order):
        if name not in spans:
            continue
        lo, hi, role = spans[name]
        y = base - i * step
        col_ = colour(role, dark)
        fig.add_scatter(x=[lo, hi], y=[y, y], mode="lines", showlegend=False,
                        hoverinfo="skip", line=dict(color=col_, width=2.5), row=1, col=2)
        for xx in (lo, hi):
            fig.add_scatter(x=[xx, xx], y=[y - 1.8, y + 1.8], mode="lines", showlegend=False,
                            hoverinfo="skip", line=dict(color=col_, width=2.5), row=1, col=2)
        fig.add_annotation(x=hi, y=y, text=f"  {name}", showarrow=False, xanchor="left",
                           font=dict(size=9, color=col_), row=1, col=2)

    fig.update_layout(
        title="Concepts — the same volumes in section and in distribution",
        showlegend=False,
    )
    fig.update_xaxes(title_text="Schematic width (∝ √area)", showticklabels=False, row=1, col=1)
    fig.update_xaxes(title_text="Recoverable resource (MMboe)", rangemode="tozero", row=1, col=2)
    fig.update_yaxes(title_text="Probability of exceedance (%)",
                     range=[base - len(order) * step - 3.0, 107.0], row=1, col=2)
    apply_plotly(fig, dark, height)
    depth_axis_plotly(fig, (ad.shallowest, ad.deepest), row=1, col=1)
    return fig
