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
from ..core.classes import (
    READING_DASH,
    READING_LABELS,
    VolumeClasses,
    conditional_exceedance,
    risked_exceedance,
)
from ..core.groups import Groups
from ..core.reservoir import thickness_from_pay
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
from .figures import (
    _depth_band,
    _exceedance,
    area_spread_is_material,
    exceedance_marks,
)
from .theme import (
    PANEL_HEIGHT,
    SEQUENTIAL_CMAP,
    apply_plotly,
    colour,
    depth_axis_plotly,
    palette,
    reference_label,
    rgba,
)

__all__ = [
    "pfig_colour_key",
    "CONCEPT_KEY",
    "pfig_c1_section",
    "pfig_c2_exceedance",
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
    "pfig_b7_frontier",
    "pfig_b8_commercial_chance",
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
    show_depth_labels: bool = True, area_scale: str = "area",
    dark: bool = False, height: int | None = PANEL_HEIGHT,
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
    axis_label, transform = AREA_SCALES.get(area_scale, AREA_SCALES["area"])
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
                x=transform(values), y=zb, mode="lines", name=name,
                line=dict(color=p["muted"], width=1),
                hovertemplate=name + " in this depth bin<extra></extra>",
            )
        fig.add_scatter(
            x=transform(amean), y=zb, mode="lines", name="Mean area",
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
            x=transform(ad.a), y=ad.z, mode="lines", name="A(z)",
            line=dict(color=colour("prospect", dark), width=2.5),
            hovertemplate="%{x:.3f} km² at " + DEPTH_HOVER + "<extra></extra>",
        )

    if current_entry is not None:
        _hline(fig, current_entry, p["well"], "dash", "well entry")
    if current_exit is not None and current_exit != current_entry:
        _hline(fig, current_exit, p["well"], "dot", "well exit")

    fig.update_layout(
        title=f"A1 · Area–depth curve (isotonic R² = {ad.r2:.6f}){subtitle}",
        xaxis_title=axis_label,
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
    # **Dashed contours, one solid line for the well's entry** (Lars, 2026-08-10).
    # Line style now carries one meaning only -- "is this the well?" -- instead of
    # doubling as the extrapolation flag, which was the confusing part: some rings
    # were solid and some dashed for a reason a reader had to know to see. The
    # extrapolated rings above the shallowest sampled contact are still marked, but
    # by *opacity* now, which reads as "less certain" without competing with the
    # entry contour.
    label_i = 0
    for zz, rr, is_extrap, is_limit in rings:
        inside_well = zz <= z_entry
        fig.add_scatter(
            x=rr * np.cos(theta), y=rr * np.sin(theta), mode="lines",
            line=dict(
                color=colour("attic", dark) if inside_well else colour("prospect", dark),
                width=2.2 if is_limit else 1.1,
                dash="dash",
            ),
            opacity=0.45 if is_extrap else 1.0,
            name=f"{zz:.0f} m", showlegend=False,
            hovertemplate=(
                f"{zz:.1f} m TVDSS<br>{np.pi * rr * rr:.3f} km² enclosed"
                + ("<br>deepest sampled contact" if is_limit else "")
                + ("<br>extrapolated above sampled range" if is_extrap else "")
                + "<extra></extra>"
            ),
        )
        # A small depth label on every ring, so the map can be read like a depth
        # map instead of by hovering. Placed on the contour itself rather than in a
        # legend: a legend of fifteen depths is a lookup table, and the point of
        # round contour values is that they can be read in place.
        #
        # Stepped around the circle rather than stacked at due north. Contour radii
        # are not evenly spaced -- A(z) is nearly quadratic, so the shallow rings
        # crowd together -- and on prospect B seven labels on one radial line
        # overlapped into an unreadable column.
        if rr > 0:
            ang_lab = np.deg2rad(_LABEL_AZIMUTHS[label_i % len(_LABEL_AZIMUTHS)])
            fig.add_annotation(
                x=rr * np.cos(ang_lab), y=rr * np.sin(ang_lab), text=f"{zz:.0f}",
                showarrow=False, font=dict(size=7.5, color=p["text_secondary"]),
                bgcolor=p["surface"], borderpad=1,
                opacity=0.55 if is_extrap else 0.95,
            )
            label_i += 1

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
    # The entry contour is the **only solid ring** on the map, and it is the one
    # that matters: everything inside it is what a dry hole leaves up-dip. Every
    # other contour is dashed, so this reads without needing the legend.
    fig.add_scatter(
        x=r_entry * np.cos(theta), y=r_entry * np.sin(theta), mode="lines",
        fill="toself", fillcolor=rgba("attic", 0.35, dark),
        line=dict(color=colour("attic", dark), width=3.0, dash="solid"),
        name=f"Potential attic — up-dip of entry ({a_attic:.2f} km²)",
        hovertemplate=f"potential attic<br>{a_attic:.3f} km² up-dip of the {z_entry:.0f} m entry<extra></extra>",
    )
    fig.add_annotation(
        x=0.0, y=r_entry, text=f"<b>{z_entry:.0f} m — well entry</b>", showarrow=False,
        yshift=9, font=dict(size=9, color=colour("attic", dark)),
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

    # Roles rather than colours, so the fills can be taken translucent from one
    # place. Translucent at Lars's request (2026-08-10): at full opacity the four
    # bands read as flat blocks of paint, and the boundaries between them -- which
    # are the only thing that moves with depth -- get lost against them. The alpha
    # also lets the current-depth rule show *through* the bands instead of being
    # drawn over them.
    bands = [
        (np.zeros_like(z), cum0, "Chance failure", "muted"),
        (cum0, cum1, "Dry, with attic", "attic"),
        (cum1, cum2, "Discovery, contact seen", "tested"),
        (cum2, cum3, "Discovery, HC to exit", "possible"),
    ]
    fig = go.Figure()
    for lower, upper, name, role in bands:
        # Closed polygon rather than fill='tonextx': explicit, and immune to
        # trace ordering.
        fig.add_scatter(
            x=np.concatenate([lower, upper[::-1]]),
            y=np.concatenate([z, z[::-1]]),
            fill="toself", fillcolor=rgba(role, 0.55, dark), mode="lines",
            line=dict(color=colour(role, dark), width=1.0), name=name, hoverinfo="skip",
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

    fig.update_layout(
        title=f"A3 · Chance decomposition vs location ({reference_label(sweep.reference)})",
        xaxis_title="Probability (%)",
    )
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
def _mark_exceedance(fig, values, role: str, dark: bool, *, chance: float = 1.0,
                     row=None, col=None, show_text: bool = True, size: int = 7):
    """Put P90 / P50 / mean / P10 markers on an exceedance curve, labelled by value.

    Labelled with the **volume**, not the percentile, because the percentile is
    already the axis: a reader looking at a curve wants to know "what is the P50",
    and printing "P50" beside it says nothing they cannot see. The statistic's name
    goes in the hover, the number goes on the page.

    Diamonds rather than dots, in the curve's own colour with a pale outline, so
    four markers on four overlapping curves stay separable. The mean is drawn as a
    square: it is not a percentile, and on a right-skewed distribution it sits
    visibly above the P50, which is the thing worth noticing.
    """
    marks = exceedance_marks(values, chance)
    if not marks:
        return
    p = palette(dark)
    for label, value, pct in marks:
        fig.add_scatter(
            x=[value], y=[pct], mode="markers+text" if show_text else "markers",
            marker=dict(
                symbol="square" if label == "Mean" else "diamond",
                size=size + (1 if label == "Mean" else 0),
                color=colour(role, dark),
                line=dict(color=p["surface"], width=1.2),
            ),
            text=[f" {value:,.1f}"] if show_text else None,
            textposition="middle right",
            textfont=dict(size=8, color=p["text_secondary"]),
            showlegend=False,
            hovertemplate=f"{label} = {value:,.2f} MMboe at {pct:.1f}%<extra></extra>",
            **({} if row is None else dict(row=row, col=col)),
        )


def pfig_a5_exceedance(
    ts: TrialSet, groups: Groups, vc: VolumeClasses, *, mefs: float | None = None,
    pos_prospect: float | None = None, p_well: float | None = None,
    dark: bool = False, height: int | None = PANEL_HEIGHT,
):
    """A5 -- exceedance curves at the chosen location. No depth on either axis.

    The money chart, and the one that most wants a cursor: hover reads the
    probability of exceeding any volume directly off each curve.
    """
    res = ts.col("resource")
    p = palette(dark)
    fig = go.Figure()
    # **Both readings**, on Lars's instruction (2026-08-11): solid conditional and
    # dashed unconditional, the same convention as C2 and B8. Each series is
    # conditional on a *different* event, so each unconditional twin uses its own
    # chance -- which is the thing worth seeing, because those four chances are not
    # the same number and the conditional curves hide that completely.
    #
    # The chances arrive as arguments and are never taken from the trial file's own
    # zero count. Passing them is what lets A5 agree with tab ③'s table.
    p_updip = (max(pos_prospect - p_well, 0.0)
               if (pos_prospect is not None and p_well is not None) else None)
    series = [
        ("Prospect (all trials)", res[res > 0], pos_prospect, "prospect"),
        ("Discovery case", res[groups.discovery], p_well, "discovery"),
        ("Proven at well", vc.proven[groups.discovery], p_well, "proven"),
        ("Attic | dry hole", res[groups.dry_with_attic], p_updip, "attic"),
    ]
    for name, values, chance_of, role in series:
        readings = [("conditional", 1.0)]
        if chance_of is not None:
            readings.append(("unconditional", float(chance_of)))
        for reading, chance_used in readings:
            v, pct = risked_exceedance(values, chance_used)
            if v.size == 0:
                continue
            fig.add_scatter(
                x=v, y=pct, mode="lines",
                name=name if reading == "conditional" else f"{name} — risked",
                legendgroup=name,
                line=dict(color=colour(role, dark),
                          width=2.5 if reading == "conditional" else 1.8,
                          dash=READING_DASH[reading]),
                hovertemplate=(
                    f"{name} — {READING_LABELS[reading]}"
                    "<br>%{y:.1f}% chance of exceeding %{x:.2f} MMboe<extra></extra>"
                ),
            )
            # Labelled values on the conditional markers only; the risked twins get
            # markers without text, or eight numbers per series would collide.
            _mark_exceedance(fig, values, role, dark, chance=chance_used,
                             show_text=reading == "conditional")
    if mefs is not None:
        _vline(fig, mefs, p["muted"], "dot", "MEFS")

    fig.update_layout(
        title="A5 · Exceedance curves — solid conditional, dashed unconditional (risked)",
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
        title=(
            f"B2 · Chance vs regret (MEFS {vsweep.mefs:.1f} MMboe, "
            f"{reference_label(vsweep.reference)})"
        ),
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
    # Top of the axis pinned at 1.2 (Lars, 2026-08-11). A chance cannot exceed 1,
    # so leaving plotly to autoscale gave a different ceiling on every chance table
    # and made two waterfalls impossible to compare by eye. The headroom above 1.0
    # is for the "x1.000" labels on the elements that cost nothing. Log axis, so the
    # range is in decades.
    top = np.log10(1.2)
    floor = float(np.nanmin([v for v in tops if v > 0] or [0.1]))
    fig.update_yaxes(type="log", range=[np.log10(max(floor * 0.6, 1e-4)), top])
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


#: Azimuths, in degrees, that successive contour labels are placed at. Stepped so
#: that neighbouring rings -- which crowd together where A(z) steepens -- do not
#: stack their labels into one unreadable column.
_LABEL_AZIMUTHS = (90.0, 55.0, 125.0, 20.0, 160.0, 70.0, 110.0, 40.0, 140.0)

AREA_SCALES = {
    "area": ("Productive area (km²)", lambda a: a),
    "area²": ("Productive area² (km⁴)", lambda a: a ** 2),
    "√area": ("√ productive area (km)", lambda a: np.sqrt(np.maximum(a, 0.0))),
}
"""x-axis transforms for the area-depth panels.

GeoX plots its area-depth curve against area **squared**, so that convention is
offered rather than only ours. `sqrt(area)` is included too because it is the one
that straightens a conical closure, which makes departures from a simple cone easy
to see. The transform touches the axis only -- every number the tool computes is
in km2 regardless (non-negotiable 4).
"""


def _reservoir_section(fig, ad, ts, *, z_entry, z_exit, dark, area_scale="area",
                       row=None, col=None):
    """The left panel: an area-depth section with top and base reservoir.

    x is **area**, not a lateral distance, and that is the point. A(z) records
    how much area each depth encloses and carries no lateral geometry
    whatsoever, so a physical cross-section cannot be drawn from it honestly --
    an earlier version of this panel used sqrt(area) as a pretend width and
    looked, correctly, like nothing in the subsurface. In area-depth space every
    line is a real quantity:

    * **Top reservoir** is A(z) itself.
    * **Base reservoir** is the same curve shifted down by the reservoir
      thickness, so the vertical gap between them *is* the thickness and the
      shaded wedges are gross rock volume per depth interval.

    Reservoir thickness is sampled in the trials (25-65 m on the reference
    file), so one base curve means one statistic -- the mean. The panel is
    therefore the mean-thickness case and the caller says so. Where the export
    carries no thickness column the base cannot be drawn at all, and the panel
    degrades to the top curve rather than inventing one.

    Returns the thickness used, or None when there was none to use.
    """
    p = palette(dark)
    # ``row``/``col`` are omitted entirely when None: C1 is a standalone figure now,
    # and plotly raises _grid_ref when a subplot coordinate is handed to a figure
    # that has no grid. One dict, spread into every call, so there is one place
    # where that decision lives.
    at = {} if row is None else dict(row=row, col=col)
    _, transform = AREA_SCALES.get(area_scale, AREA_SCALES["area"])
    a, top = transform(ad.a), ad.z

    # Reservoir thickness is *back-calculated from pay*, per trial, by inverting
    # the wedge -- see core.reservoir. It is not read from any thickness column,
    # for two reasons. It works on the 7-column paste, which has no such column
    # but does have area and pay. And where a column does exist the inversion
    # reproduces it to a mean difference of 0.01 m at r = 0.9998, so reading it
    # would add a dependency and buy nothing.
    tfp = thickness_from_pay(ts, ad)
    stats = tfp.summary()
    thickness = stats["p50"] if tfp.n_resolved else None

    fig.add_scatter(
        x=a, y=top, mode="lines", name="Top reservoir", showlegend=False,
        line=dict(color=p["text"], width=2),
        hovertemplate="top reservoir<br>%{x:.2f} km² at " + DEPTH_HOVER + "<extra></extra>",
        **at,
    )
    fig.add_annotation(x=a[-1], y=top[-1], text=" Top reservoir", showarrow=False,
                       xanchor="left", font=dict(size=9, color=p["text"]), **at)

    if thickness is not None:
        base = top + thickness
        # The P90 and P10 bases first, thin and grey, so the single P50 line is
        # visibly one case out of a sampled range rather than a fixed surface.
        for stat, label in (("p90", "P90"), ("p10", "P10")):
            fig.add_scatter(
                x=a, y=top + stats[stat], mode="lines", showlegend=False,
                line=dict(color=p["muted"], width=1, dash="dot"),
                hovertemplate=f"base reservoir, {label} thickness "
                              f"({stats[stat]:.0f} m)<extra></extra>",
                **at,
            )
        fig.add_scatter(
            x=a, y=base, mode="lines", name="Base reservoir", showlegend=False,
            line=dict(color=p["text"], width=1.6, dash="dash"),
            hovertemplate="base reservoir<br>%{x:.2f} km² at " + DEPTH_HOVER + "<extra></extra>",
            **at,
        )
        fig.add_annotation(x=a[-1], y=base[-1], text=" Base reservoir", showarrow=False,
                           xanchor="left", font=dict(size=9, color=p["text"]), **at)

        # The reservoir band, cut by the two well depths. At each area the band
        # runs top..top+thickness; clipping that interval to each depth window
        # gives the three volumes, and where the clipped interval is empty the
        # region simply does not exist at that area.
        for lo, hi, role, label in (
            (-np.inf, z_entry, "up_dip", "up-dip"),
            (z_entry, z_exit, "tested", "tested"),
            (z_exit, np.inf, "possible", "possible"),
        ):
            upper = np.clip(top, lo, hi)
            lower = np.clip(base, lo, hi)
            m = lower > upper + 1e-9
            if m.sum() < 2:
                continue
            fig.add_scatter(
                x=np.concatenate([a[m], a[m][::-1]]),
                y=np.concatenate([upper[m], lower[m][::-1]]),
                fill="toself", fillcolor=rgba(role, 0.55, dark), mode="lines",
                line=dict(width=0), showlegend=False, hoverinfo="skip", row=row, col=col,
            )
            mid = int(np.flatnonzero(m)[m.sum() // 2])
            fig.add_annotation(
                x=a[mid], y=0.5 * (upper[mid] + lower[mid]), text=label, showarrow=False,
                font=dict(size=9, color=p["text"]), row=row, col=col,
            )

    # The well, at the area its entry depth encloses. Only that area carries
    # meaning here; the line is drawn full height so the crossings read clearly.
    a_entry = float(transform(np.asarray(ad.area_at(z_entry))))
    fig.add_scatter(
        x=[a_entry, a_entry], y=[float(top.min()), float((top + (thickness or 0.0)).max())],
        mode="lines", name="Well", showlegend=False,
        line=dict(color=p["well"], width=2.5),
        hovertemplate=f"well at {a_entry:.2f} km²<extra></extra>", row=row, col=col,
    )
    fig.add_annotation(x=a_entry, y=float(top.min()), text="Well", showarrow=False, yshift=12,
                       font=dict(size=11, color=p["well"]), **at)
    for depth, label in ((z_entry, "Reservoir entry"), (z_exit, "Reservoir exit")):
        if depth is None:
            continue
        fig.add_annotation(
            x=a_entry, y=depth, text=f"{label} ", showarrow=True, arrowhead=0, arrowwidth=1,
            arrowcolor=p["text_secondary"], ax=38, ay=0, xanchor="right",
            font=dict(size=9, color=p["text_secondary"]), **at,
        )
    return thickness



# ------------------------------------------------------------------- B7
def pfig_b7_frontier(
    vsweep: VolumeSweep, *, current_z: float | None = None, min_support: int = MIN_SUPPORT,
    label_every: int = 4, dark: bool = False, height: int | None = PANEL_HEIGHT,
):
    """B7 -- the trade-off frontier: chance against volume, parametric in depth.

    From the 2018 macro workbook, where it is titled *"Well POS vs. Well to be
    tested Mean Resource"* with ``Well asso. Mean resource`` on x and ``Well POS``
    on y. It is the one chart in that workbook the app had no equivalent of, and it
    is the most direct statement of the thing this whole tool is about: **moving the
    well down-dip buys volume with chance.**

    Neither axis is a depth, so depth appears as *labels along the curve* -- every
    ``label_every``-th sweep step -- and the figure is exempt from the depth rule
    for the same reason A5, A6, B4 and B5 are.

    Read it like an efficient frontier. Up and to the right is better and
    unavailable; the curve is what the structure actually offers. A location is
    dominated if another sits above and to the right of it, and on a monotone
    closure none do -- which is the point: there is no free lunch, only a rate of
    exchange, and this figure is where you read that rate.

    ``x`` is the **well-associated** mean (Rose's *Downdip*, the whole accumulation
    given a discovery) because that is what the workbook plots and what a well
    proposal is written against. The proven mean is drawn beside it as a lighter
    line, since the two answer "what would I find" and "what would I have proven".
    """
    p = palette(dark)
    fig = go.Figure()
    n_disc = vsweep.n_discovery
    pw = thin(vsweep.p_well, n_disc, min_support) * 100.0
    assoc = thin(vsweep.discovery_mean, n_disc, min_support) \
        if vsweep.discovery_mean is not None else None
    proven = thin(vsweep.proven_mean, n_disc, min_support)

    if assoc is not None:
        fig.add_scatter(
            x=assoc, y=pw, mode="lines", name="Well associated mean",
            line=dict(color=colour("well_associated", dark), width=2.8),
            customdata=vsweep.z,
            hovertemplate=("entry %{customdata:.0f} m TVDSS<br>well associated mean "
                           "%{x:.1f} MMboe<br>P_well %{y:.1f}%<extra></extra>"),
        )
    fig.add_scatter(
        x=proven, y=pw, mode="lines", name="Proven mean",
        line=dict(color=colour("tested", dark), width=1.8, dash="dash"),
        customdata=vsweep.z,
        hovertemplate=("entry %{customdata:.0f} m TVDSS<br>proven mean %{x:.1f} MMboe"
                       "<br>P_well %{y:.1f}%<extra></extra>"),
    )

    # Depth labels along the frontier: without them the curve is a shape with no
    # handle on it, and "where do I drill" is answered in metres.
    base = assoc if assoc is not None else proven
    for i in range(0, vsweep.z.size, max(1, label_every)):
        if not (np.isfinite(base[i]) and np.isfinite(pw[i])):
            continue
        fig.add_annotation(
            x=float(base[i]), y=float(pw[i]), text=f"{vsweep.z[i]:.0f}", showarrow=False,
            xshift=6, yshift=7, font=dict(size=8, color=p["text_secondary"]),
        )

    if current_z is not None:
        here_x = float(np.interp(current_z, vsweep.z, np.nan_to_num(base, nan=0.0)))
        here_y = float(np.interp(current_z, vsweep.z, np.nan_to_num(pw, nan=0.0)))
        fig.add_scatter(
            x=[here_x], y=[here_y], mode="markers+text",
            marker=dict(symbol="circle-open-dot", size=15,
                        line=dict(color=p["well"], width=3), color=p["well"]),
            text=[f"  this well, {current_z:.0f} m"], textposition="middle right",
            textfont=dict(size=10, color=p["well"]), name="This well",
            hovertemplate=(f"this well at {current_z:.0f} m<br>{here_x:.1f} MMboe at "
                           f"{here_y:.1f}%<extra></extra>"),
        )
    if vsweep.mefs is not None:
        _vline(fig, vsweep.mefs, colour("minimum", dark), "dot", "MEFS")

    fig.update_layout(
        title=(f"B7 · Chance against volume — the location trade-off "
               f"({reference_label(vsweep.reference)})"),
        xaxis_title="Mean resource (MMboe)",
        yaxis_title="P_well  (%)",
    )
    fig.update_xaxes(rangemode="tozero")
    fig.update_yaxes(range=[0, 105])
    apply_plotly(fig, dark, height)
    return fig


# ------------------------------------------------------------------- B8
def pfig_b8_commercial_chance(
    vsweep: VolumeSweep, *, current_z: float | None = None,
    zlim: tuple[float, float] | None = None, show_depth_labels: bool = True,
    min_support: int = MIN_SUPPORT, dark: bool = False, height: int | None = PANEL_HEIGHT,
):
    """B8 -- commercial chance against depth: Rose's ``Pc(well)``, swept.

    The 2018 workbook draws this as two charts, *"Cond. Prob. of exceeding MEFS vs.
    depth"* and *"Prob. of exceeding MEFS vs. depth"* -- its columns ``AL`` and
    ``AM = AL x Well POS``. They belong on one pair of axes, because the whole
    content is the difference between them:

    * ``Pmcfs(well)`` -- **conditional**: given a discovery, the chance it clears
      MEFS. It *rises* down-dip, because a deeper well finds a bigger accumulation.
    * ``P_well`` -- the chance of a discovery at all. It *falls* down-dip.
    * ``Pc(well) = P_well x Pmcfs(well)`` -- **unconditional**: the chance of a
      commercial discovery, full stop. The product of a rising and a falling curve,
      so it usually has an interior maximum, and that maximum is the answer to
      "where should the well go" on commercial grounds.

    Rose gives ``Pc(well)`` as the number to carry into an EMV calculation. It is a
    *chance*, not a value, which is why it is inside this tool's scope while
    economics is not.

    Conditional solid, unconditional dashed, as everywhere else in the app.
    """
    p = palette(dark)
    z = vsweep.z
    fig = go.Figure()
    if vsweep.p_discovery_exceeds_mefs is None:
        fig.update_layout(title="B8 · Commercial chance — needs a MEFS")
        apply_plotly(fig, dark, height)
        depth_axis_plotly(fig, zlim or (float(z.min()), float(z.max())),
                          show_ticklabels=show_depth_labels)
        return fig

    pw = thin(vsweep.p_well, vsweep.n_discovery, min_support) * 100.0
    pmcfs = thin(vsweep.p_discovery_exceeds_mefs, vsweep.n_discovery, min_support) * 100.0
    pc = pw * pmcfs / 100.0

    for values, name, role, dash, width in (
        (pmcfs, "Pmcfs(well) — conditional on a discovery", "tested", "solid", 2.2),
        (pw, "P_well — chance of a discovery", "well_associated", "solid", 2.2),
        (pc, "Pc(well) — commercial chance, unconditional", "minimum", "dash", 2.8),
    ):
        fig.add_scatter(
            x=values, y=z, mode="lines", name=name,
            line=dict(color=colour(role, dark), width=width, dash=dash),
            hovertemplate=name + "<br>%{x:.1f}% at " + DEPTH_HOVER + "<extra></extra>",
        )

    # The interior maximum of Pc, which is the decision this figure supports.
    if np.any(np.isfinite(pc)):
        best = int(np.nanargmax(pc))
        fig.add_scatter(
            x=[pc[best]], y=[z[best]], mode="markers+text",
            marker=dict(symbol="star", size=13, color=colour("minimum", dark)),
            text=[f"  best Pc {pc[best]:.1f}% at {z[best]:.0f} m"],
            textposition="middle right",
            textfont=dict(size=9, color=colour("minimum", dark)), showlegend=False,
            hovertemplate=f"maximum commercial chance<br>{pc[best]:.1f}% at "
                          f"{z[best]:.0f} m TVDSS<extra></extra>",
        )
    if current_z is not None:
        _hline(fig, current_z, p["text"], "dash")

    fig.update_layout(
        title=f"B8 · Commercial chance vs location (MEFS {vsweep.mefs:.1f} MMboe)",
        xaxis_title="Probability (%)",
    )
    fig.update_xaxes(range=[0, 105])
    apply_plotly(fig, dark, height)
    depth_axis_plotly(fig, zlim or (float(z.min()), float(z.max())),
                      show_ticklabels=show_depth_labels)
    return fig

# ----------------------------------------------------- the concepts figure
def pfig_c1_section(
    ad: AreaDepth, ts: TrialSet, *, z_entry: float, z_exit: float,
    area_scale: str = "area", dark: bool = False, height: int | None = PANEL_HEIGHT,
):
    """C1 -- where each volume sits in the structure.

    Split out of the old composite (Lars, 2026-08-11). It was one figure of two
    stacked panels; two figures render at their own natural heights, can be exported
    and dropped into a deck separately, and neither has to compromise for the other.
    C2 is the matching exceedance figure and they are read together.

    x is **area**, not a lateral distance, and that is the point -- see
    :func:`_reservoir_section` for why a physical cross-section cannot honestly be
    drawn from A(z).
    """
    fig = go.Figure()
    _reservoir_section(fig, ad, ts, z_entry=z_entry, z_exit=z_exit, dark=dark,
                       area_scale=area_scale)
    tfp = thickness_from_pay(ts, ad)
    ss = tfp.summary()
    note = (
        f"   ·   base reservoir = top + thickness back-calculated from pay: "
        f"P50 {ss['p50']:.0f} m, P90–P10 {ss['p90']:.0f}–{ss['p10']:.0f} m (dotted)"
        if tfp.n_resolved else
        "   ·   no reservoir thickness recoverable from pay, so no base reservoir is drawn"
    )
    fig.update_layout(title="C1 · Where each volume sits in the structure", showlegend=False)
    fig.update_xaxes(title_text=AREA_SCALES.get(area_scale, AREA_SCALES["area"])[0] + note,
                     title_font=dict(size=9), rangemode="tozero")
    apply_plotly(fig, dark, height)
    depth_axis_plotly(fig, (ad.shallowest, ad.deepest))
    return fig


def pfig_c2_exceedance(
    ts: TrialSet, groups: Groups, vc: VolumeClasses, *,
    pos_prospect: float, p_well: float, mefs: float | None = None,
    dark: bool = False, height: int | None = PANEL_HEIGHT,
):
    """C2 -- the same volumes as exceedance curves, both readings drawn.

    Read with C1: that figure shows where each volume sits in the structure, this
    one shows what it is worth and how likely it is. They were a single stacked
    composite until 2026-08-11, when Lars asked for them split -- two figures render
    at their own heights, export separately, and neither compromises for the other.

    **Two curves per concept, in one colour, style carrying the reading:**

    * **solid = conditional (success case)** -- given that case happens. Starts at
      100 %, and this is where the percentiles live: "P90 is 90 % probability of
      exceeding the P90 estimated value" (Milkov 2021). Schneider et al. (2023)
      determine this distribution *before* any chance is applied.
    * **dashed = unconditional (risked)** -- the same volumes with the chance of the
      case folded in, so it starts at the chance.

    The two POS values are therefore *where the dashed curves start*, not
    annotations bolted on, and the vertical gap between the prospect's and the
    well's is the location penalty. Which is the argument the tool exists to make:
    you drill a well, not a prospect.

    Built through :func:`wellvolpos.core.classes.risked_exceedance`, so a dashed
    curve cannot start anywhere but at its chance -- see that docstring for the four
    times an unrisked number was drawn under a risked label.
    """
    p = palette(dark)
    res = ts.col("resource")
    fig = go.Figure()
    # ------------------------------- conditional and unconditional, both drawn
    # Two curves per volume concept, in one colour, distinguished by line style:
    #
    #   solid  = conditional (success case)  -- starts at 100 %, and this is where
    #            the percentiles live: "P90 is 90 % probability of exceeding the P90
    #            estimated value" (Milkov 2021). It is what anybody means by "the
    #            P50", and Schneider et al. (2023) determine it *before* the chance.
    #   dashed = unconditional (risked)      -- the same volumes with the chance of
    #            the case folded in, so it starts at the chance.
    #
    # Both, because they answer different questions and a reader kept having to
    # guess which was on screen. Built through ``core.classes.risked_exceedance``,
    # so the dashed curve starts at ``chance`` by construction rather than by being
    # zero-padded with the trial file's own masks -- see that docstring for the four
    # times this went wrong.
    disc, dry = groups.discovery, groups.dry_with_attic
    cases = [
        ("Prospect resource potential", res[res > 0], pos_prospect, "prospect"),
        ("Well associated resource potential", res[disc], p_well, "well_associated"),
        ("Resource tested by well", vc.proven[disc], p_well, "tested"),
        # The up-dip case needs its *own* chance: dry but charged, which is
        # POS_prospect - P_well, not P_well.
        ("Up-dip volume", res[dry], max(pos_prospect - p_well, 0.0), "up_dip"),
    ]
    spans: dict[str, tuple[float, float, str]] = {}
    for name, values, chance_of, role in cases:
        for reading, chance_used in (("conditional", 1.0), ("unconditional", chance_of)):
            v, pct = risked_exceedance(values, chance_used)
            if v.size == 0:
                continue
            fig.add_scatter(
                x=v, y=pct, mode="lines",
                name=f"{name} — {READING_LABELS[reading].split(' (')[0].lower()}",
                legendgroup=name,
                line=dict(color=colour(role, dark), width=2.6 if reading == "conditional" else 1.9,
                          dash=READING_DASH[reading]),
                hovertemplate=(
                    f"{name}<br>{READING_LABELS[reading]}"
                    "<br>%{y:.1f}% chance of exceeding %{x:.2f} MMboe<extra></extra>"
                ),
            )
            # Markers on the *unconditional* curve only. Eight per concept would be
            # noise, and the unconditional is the one a decision reads; the
            # conditional percentiles are in the table in tab ③ and in the hover.
            if reading == "unconditional":
                _mark_exceedance(fig, values, role, dark, chance=chance_used,
                                 show_text=False, size=6)
        positive = np.sort(np.asarray(values, dtype=float))
        positive = positive[np.isfinite(positive) & (positive > 0)]
        if positive.size:
            spans[name] = (float(positive.min()), float(positive.max()), role)

    # The two POS values, drawn where the unconditional curves actually start.
    for value, label, role in (
        (pos_prospect, "Asso. Final Prospect POS", "prospect"),
        (p_well, "Asso. Well POS", "well_associated"),
    ):
        fig.add_hline(
            y=value * 100.0, line=dict(color=colour(role, dark), width=1, dash="dot"),
            annotation_text=f"{label} {value:.0%}", annotation_position="top right",
            annotation_font=dict(size=10, color=colour(role, dark)),
        )
    if mefs is not None:
        # Annotated at the *bottom* of the panel: at the top it landed on the
        # subplot title once the two panels were stacked.
        fig.add_vline(
            x=mefs, line=dict(color=colour("minimum", dark), width=1.2, dash="dot"),
            annotation_text="min. volume", annotation_position="bottom right",
            annotation_font=dict(size=10, color=colour("minimum", dark)),
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
                        hoverinfo="skip", line=dict(color=col_, width=2.5))
        for xx in (lo, hi):
            fig.add_scatter(x=[xx, xx], y=[y - 1.8, y + 1.8], mode="lines", showlegend=False,
                            hoverinfo="skip", line=dict(color=col_, width=2.5))
        fig.add_annotation(x=hi, y=y, text=f"  {name}", showarrow=False, xanchor="left",
                           font=dict(size=9, color=col_))

    fig.update_layout(
        title="C2 · The same volumes as exceedance curves — solid conditional, dashed unconditional",
        showlegend=False,
    )
    fig.update_xaxes(title_text="Recoverable resource (MMboe)", rangemode="tozero")
    # The braces live below zero, so the axis reaches there -- but a negative
    # *probability* label is meaningless and was being read as one. Ticks are pinned
    # to 0-100 and the space below simply carries the braces (Lars, 2026-08-11).
    ticks = list(range(0, 101, 20))
    fig.update_yaxes(
        title_text="Probability of exceedance (%)",
        range=[base - len(order) * step - 3.0, 107.0],
        tickmode="array", tickvals=ticks, ticktext=[str(t) for t in ticks],
    )
    apply_plotly(fig, dark, height)
    return fig

# --------------------------------------------------------------- colour key
# (role, label, what it means) in nesting order, narrowest first, so the key
# itself teaches the containment.
CONCEPT_KEY = (
    ("minimum", "Minimum volume",
     "a threshold — MCFS/MEFS, or the assessment minimum (they differ)"),
    ("up_dip", "Up-dip / attic volume",
     "what a dry hole leaves behind — Rose's “Updip”"),
    ("tested", "Resource tested by the well",
     "between reservoir entry and exit — what a discovery proves"),
    ("possible", "Possible, below the exit",
     "well associated but never tested"),
    ("well_associated", "Well associated volume",
     "the accumulation given a discovery — Rose's “Downdip”. What a well proposal uses"),
    ("prospect", "Prospect resource potential",
     "the whole un-cut model, crest to spill"),
    ("well", "The well", "entry, exit and position"),
)


def pfig_colour_key(dark: bool = False, height: int = 300):
    """The volume-concept colour key, as a figure rather than styled HTML.

    Drawn rather than written because Streamlit strips inline ``style``
    attributes out of markdown, so an HTML swatch renders as a label with no
    colour beside it -- which is the one thing a colour key must not do. As a
    figure it also picks up the palette from :mod:`wellvolpos.viz.theme` like
    every other panel, so it cannot drift from the figures it explains.

    Ordered by nesting, narrowest at the top, so the key carries the containment
    as well as the mapping.
    """
    p = palette(dark)
    fig = go.Figure()
    n = len(CONCEPT_KEY)
    for i, (role, label, meaning) in enumerate(CONCEPT_KEY):
        y = n - i
        fig.add_shape(
            type="rect", x0=0.0, x1=0.045, y0=y - 0.3, y1=y + 0.3,
            fillcolor=colour(role, dark), line=dict(width=0), layer="above",
        )
        fig.add_annotation(
            x=0.06, y=y, text=f"<b>{label}</b> — {meaning}", showarrow=False,
            xanchor="left", font=dict(size=11, color=p["text"]),
        )
    # An invisible trace so the axes exist and the shapes have somewhere to sit.
    fig.add_scatter(x=[0, 1], y=[0.2, n + 0.8], mode="markers",
                    marker=dict(opacity=0), showlegend=False, hoverinfo="skip")
    fig.update_layout(title=None, showlegend=False,
                      margin=dict(l=6, r=6, t=6, b=6), height=height)
    fig.update_xaxes(visible=False, range=[0, 1], fixedrange=True)
    fig.update_yaxes(visible=False, range=[0.2, n + 0.8], fixedrange=True)
    apply_plotly(fig, dark, height)
    fig.update_layout(margin=dict(l=6, r=6, t=6, b=6))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return fig
