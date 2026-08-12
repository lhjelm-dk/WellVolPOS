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
    conditional_exceedance,
    READING_DASH,
    READING_LABELS,
    VolumeClasses,
    risked_exceedance,
)
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
from .figures import (
    _depth_band,
    _depth_percentiles,
    area_spread_is_material,
    exceedance_marks,
)
from .theme import (
    AREA_SCALES,
    PANEL_HEIGHT,
    VALUE_CMAP,
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
    "pfig_b9_chance_weighted",
    "pfig_a8_contact_distribution",
    "pfig_a9_prospect_density",
    "suggest_grid",
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

def _reservoir_band(fig, ad, ts, *, z_entry, z_exit, dark, transform, labels=True,
                    show_classes=True):
    """Top and base reservoir with the three volume classes shaded between them.

    Returns a one-line note about the thickness for the caller's subtitle, or "".

    The base reservoir is drawn **four times** -- P90, P50, mean and P10 of the
    thickness recovered from pay -- because that thickness is a distribution and a
    single base line implied a surface the trials do not support. The P50 keeps the
    weight; the rest are thin, the same convention A1's area family already uses.

    The shaded wedges are clipped to the well's depth windows, so up-dip, tested and
    possible-below-exit appear where they physically are. Extracted from what used
    to be C1 so A1 and C1 cannot draw it differently.
    """
    p = palette(dark)
    tfp = thickness_from_pay(ts, ad)
    stats = tfp.summary()
    if not tfp.n_resolved:
        return ("<br><sub>no reservoir thickness recoverable from pay, so no base "
                "reservoir is drawn</sub>")

    a, top = transform(ad.a), ad.z
    fig.add_scatter(
        x=a, y=top, mode="lines", name="Top reservoir", showlegend=labels,
        line=dict(color=p["text"], width=2),
        hovertemplate="top reservoir at " + DEPTH_HOVER + "<extra></extra>",
    )
    for key, name, width, dash in (("p90", "Base P90", 1.0, "dot"),
                                   ("p50", "Base P50", 1.8, "dash"),
                                   ("mean", "Base mean", 1.4, "solid"),
                                   ("p10", "Base P10", 1.0, "dot")):
        fig.add_scatter(
            x=a, y=top + stats[key], mode="lines", name=name, showlegend=labels,
            line=dict(color=p["muted"] if key != "p50" else p["text"],
                      width=width, dash=dash),
            hovertemplate=f"base reservoir, {name.split()[1]} thickness "
                          f"({stats[key]:.0f} m)<extra></extra>",
        )

    # The three shaded classes are a *well* result -- they need an entry and an exit
    # to exist at all -- so tab ② draws the band without them (Lars, 2026-08-11).
    # The reservoir itself is a property of the prospect and stays.
    if not show_classes:
        return (f"<br><sub>base reservoir = top + thickness from pay: P90 {stats['p90']:.0f} · "
                f"P50 {stats['p50']:.0f} · mean {stats['mean']:.0f} · "
                f"P10 {stats['p10']:.0f} m</sub>")

    base = top + stats["p50"]
    for lo, hi, role, label in ((-np.inf, z_entry, "up_dip", "up-dip"),
                                (z_entry, z_exit, "tested", "tested"),
                                (z_exit, np.inf, "possible", "possible")):
        upper, lower = np.clip(top, lo, hi), np.clip(base, lo, hi)
        m = lower > upper + 1e-9
        if m.sum() < 2:
            continue
        fig.add_scatter(
            x=np.concatenate([a[m], a[m][::-1]]),
            y=np.concatenate([upper[m], lower[m][::-1]]),
            fill="toself", fillcolor=rgba(role, 0.55, dark), mode="lines",
            line=dict(width=0), showlegend=False, hoverinfo="skip",
        )
        if labels:
            mid = int(np.flatnonzero(m)[m.sum() // 2])
            fig.add_annotation(x=a[mid], y=0.5 * (upper[mid] + lower[mid]), text=label,
                               showarrow=False, font=dict(size=9, color=p["text"]))
    return (f"<br><sub>base reservoir = top + thickness from pay: P90 {stats['p90']:.0f} · "
            f"P50 {stats['p50']:.0f} · mean {stats['mean']:.0f} · P10 {stats['p10']:.0f} m</sub>")


def pfig_a1_area_depth(
    ad: AreaDepth, *, ts: TrialSet | None = None, current_entry: float | None = None,
    current_exit: float | None = None, n_bins: int = 40,
    zlim: tuple[float, float] | None = None, show_depth_labels: bool = True,
    area_scale: str = "area", show_reservoir: bool = True, show_classes: bool = True,
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

    # **C1's reservoir section, merged in** (Lars, 2026-08-11). A1 and C1 drew the
    # same A(z) on the same axes; the only thing C1 added was the base reservoir and
    # the three shaded volume classes, and two figures of one curve is how a reader
    # comes to think they are two curves. C1 survives as a small unlabelled
    # thumbnail beside C2, for recognition.
    #
    # The base reservoir now carries its own **P90 / P50 / mean / P10**, because the
    # thickness recovered from pay is a distribution and drawing one base line
    # implied a surface the data does not support.
    reservoir_note = ""
    if show_reservoir and ts is not None:
        reservoir_note = _reservoir_band(
            fig, ad, ts,
            z_entry=current_entry if current_entry is not None else ad.shallowest,
            z_exit=current_exit if current_exit is not None else ad.deepest,
            dark=dark, transform=transform, show_classes=show_classes,
        )

    if current_entry is not None:
        _hline(fig, current_entry, p["well"], "dash", "well entry")
    if current_exit is not None and current_exit != current_entry:
        _hline(fig, current_exit, p["well"], "dot", "well exit")

    fig.update_layout(
        title=f"A1 · Area–depth curve and reservoir (isotonic R² = {ad.r2:.6f})"
              f"{subtitle}{reservoir_note}",
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
    mefs: float | None = None, render: str = "grid",
    n_bins: int = 40, n_resource: int | None = None, n_depth: int | None = None,
    gridsize: int = 60, zlim: tuple[float, float] | None = None,
    show_depth_labels: bool = True, dark: bool = False, height: int | None = PANEL_HEIGHT,
):
    """A4 -- how the trials fall in resource-by-depth space, with the percentiles.

    **One figure with two renderings** (Lars, 2026-08-11). It briefly existed as two
    -- A4's blue log-density and a separate A7 grid taken from the workbook's
    ``resource grid`` sheet -- which showed the same trials twice under two numbers.
    Merged, because two panels of one dataset is how a reader comes to believe they
    are looking at two facts.

    ``render="grid"``
        The workbook's rendering: **counts per cell**, in inferno, with the grid
        size selectable and defaulting to Freedman-Diaconis per axis. Cells rather
        than the workbook's contours, because a count in a cell is discrete with
        hard zeros outside the sampled envelope and contour interpolation invents
        values no trial supports. Inferno is perceptually uniform and monotonic in
        lightness, so darker is unambiguously fewer whether the reader sees colour
        or a greyscale print.
    ``render="hexbin"``
        The original blue log-density. Hexagons tile without the axis-aligned
        artefacts a rectangular grid shows on a diagonal trend, which is what this
        cloud is.

    Both are on a **log** count scale: the modal cell holds two orders of magnitude
    more trials than the tails, and the tails are where a location question lives.

    Over either, the conditional percentile family **P99 / P90 / P50 / P10 / P1**
    and the mean. The mean keeps the prospect colour and the weight because it is
    the number that gets quoted; the percentiles are thin and grey. On a skewed
    resource distribution the mean is **not** the P50, and showing both is half the
    reason this figure exists.

    Success trials only: the chance-failure zeros belong to POS, not to the shape of
    the resource distribution.
    """
    if render not in ("grid", "hexbin"):
        raise ValueError(f"unknown render {render!r}; expected 'grid' or 'hexbin'")
    res, contact = ts.col("resource"), ts.col("contact")
    succ = res > 0.0
    x, y = res[succ], contact[succ]
    p = palette(dark)

    if render == "grid":
        auto_r, auto_z = suggest_grid(x, y)
        nx, ny = int(n_resource or auto_r), int(n_depth or auto_z)
        colourscale, label = VALUE_CMAP, f"{nx} × {ny} grid"
    else:
        nx = ny = int(gridsize)
        colourscale, label = "Blues", f"{gridsize} × {gridsize} log-density"

    counts, xedges, yedges = np.histogram2d(x, y, bins=(nx, ny))
    counts = counts.T
    shown = np.where(counts > 0, counts, np.nan)
    with np.errstate(divide="ignore"):
        dens = np.log10(shown)

    fig = go.Figure()
    fig.add_heatmap(
        x=0.5 * (xedges[:-1] + xedges[1:]), y=0.5 * (yedges[:-1] + yedges[1:]),
        z=dens, colorscale=colourscale, showscale=True, customdata=shown,
        # Inside the axes, for the reason given on B6's colourbar: the depth-row
        # rule fixes the margins with autoexpand off, so plotly's default position
        # outside the plot area on the right is clipped away and the scale silently
        # disappears. A4 is a *count* grid -- without a scale the colour says only
        # "more or less", which is not what a trial count is for. Bottom-right is the
        # empty corner here: deep contacts hold large volumes, so the mass runs
        # top-left to bottom-right and the cells past it are empty.
        # Position is set by theme.apply_plotly, which owns the reserved band below
        # the axis and has to divide it between this and the legend -- they were
        # placed independently and landed on top of each other on A4.
        colorbar=dict(title=dict(text="trials (log₁₀)", side="top"), x=0.5),
        hovertemplate=("%{customdata:.0f} trials<br>%{x:.1f} MMboe at "
                       + DEPTH_HOVER + "<extra></extra>"),
    )

    zb, band = _depth_percentiles(y, x, n_bins=n_bins)
    for q, dash in ((99, "dot"), (90, "dash"), (50, "solid"), (10, "dash"), (1, "dot")):
        fig.add_scatter(
            x=band[q], y=zb, mode="lines", name=f"P{q}",
            line=dict(color=p["muted"], width=1, dash=dash),
            hovertemplate=f"P{q} " + "%{x:.2f} MMboe at " + DEPTH_HOVER + "<extra></extra>",
        )
    fig.add_scatter(
        x=band["mean"], y=zb, mode="lines", name="Mean",
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

    fig.update_layout(
        title=f"A4 · Resource vs contact depth ({label}, {x.size:,} success trials)",
        xaxis_title="Recoverable resource (MMboe)",
    )
    fig.update_xaxes(rangemode="tozero")
    apply_plotly(fig, dark, height)
    depth_axis_plotly(fig, zlim or (float(y.min()), float(y.max())),
                      title="HC-water contact (m TVDSS)", show_ticklabels=show_depth_labels)
    return fig


# ------------------------------------------------------------------- A5
def _mark_exceedance(fig, values, role: str, dark: bool, *, chance: float = 1.0,
                     row=None, col=None, show_text: bool = True, size: int = 7,
                     textposition: str = "middle right", prefix: str = " "):
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
            text=[f"{prefix}{value:,.1f}"] if show_text else None,
            textposition=textposition,
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
    # **The prospect only** (Lars, 2026-08-11). A5 sits on the *Prospect* tab, whose
    # subject is the un-cut model, and the other three series were saying again what
    # C2 draws and what tab ③'s table tabulates -- three places for one set of
    # numbers is three places to disagree. Verified identical before removing them:
    # A5's populations and the tab ③ table's were the same trials to the last
    # decimal, so nothing is lost by keeping them in one place.
    #
    # Both readings, as everywhere: solid conditional from 100 %, dashed
    # unconditional from POS_prospect.
    series = [("Prospect recoverable resource", res[res > 0], pos_prospect, "prospect")]
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
                          width=2.8 if reading == "conditional" else 2.0,
                          dash=READING_DASH[reading]),
                hovertemplate=(
                    f"{name} — {READING_LABELS[reading]}"
                    "<br>%{y:.1f}% chance of exceeding %{x:.2f} MMboe<extra></extra>"
                ),
            )
            _mark_exceedance(fig, values, role, dark, chance=chance_used, show_text=True)
    if mefs is not None:
        _vline(fig, mefs, p["muted"], "dot", "MEFS")

    fig.update_layout(
        title="A5 · Prospect resource — solid conditional, dashed unconditional (risked)",
        xaxis_title="Recoverable resource (MMboe)",
        yaxis_title="Probability of exceedance (%)",
    )
    fig.update_xaxes(rangemode="tozero")
    fig.update_yaxes(range=[0, 105])
    apply_plotly(fig, dark, height)
    return fig


# ------------------------------------------------------------------- A6
def pfig_a6_overlap(
    vc: VolumeClasses, groups: Groups, *, ts: TrialSet | None = None,
    mefs: float | None = None, bins: int = 40, normalise: str = "density",
    show_exceedance: bool = False,
    dark: bool = False, height: int | None = PANEL_HEIGHT,
):
    """A6 -- Schneider et al.'s "surprising overlap", now against all four classes.

    Densities, not counts: the groups have different n and the figure is about the
    *shape* overlap, which counts would distort by group size.

    Attic and proven are the pair Schneider names -- what a dry hole leaves against
    what a discovery proves -- and the surprise is how far they overlap. Well
    associated and prospect are drawn behind them (Lars, 2026-08-11) so the pair is
    seen in the context of the two larger distributions they are carved out of.

    Opacity is lower with four series than it was with two: at 0.6 the fourth
    histogram hid the first, and the whole content of this figure is what shows
    through what.

    ``normalise`` (Lars, 2026-08-12) chooses what the bars are comparable *in*:

    ``"density"``
        Each class integrates to 1. The honest default -- areas are comparable, so
        "most of the attic mass sits below the proven P50" is a statement the figure
        supports. A narrow class then towers over a broad one, which is correct and
        sometimes unhelpful.
    ``"peak"``
        Each class scaled to its own maximum. Makes the *shapes* comparable when one
        class is far narrower than another -- which is the usual case here, since
        proven is carved out of well associated -- at the cost of the y-axis no longer
        meaning anything absolute. Said out loud in the axis title, because a
        density axis and a peak-scaled axis look identical and mean different things.

    ``show_exceedance`` overlays the same four classes as **conditional** cumulative
    curves on a second x-axis in per cent. Conditional only, deliberately: a risked
    curve beside an unrisked histogram is two readings on one figure, which is the
    confusion C2 exists to keep apart.
    """
    if normalise not in ("density", "peak"):
        raise ValueError(f"unknown normalise {normalise!r}; expected 'density' or 'peak'")
    p = palette(dark)
    series = [
        ("Prospect resource potential", ts.col("resource")[ts.col("resource") > 0]
         if ts is not None else np.array([]), "prospect"),
        ("Well associated | discovery", vc.discovery_total[groups.discovery], "well_associated"),
        ("Attic | dry hole", vc.attic[groups.dry_with_attic], "attic"),
        ("Proven | discovery", vc.proven[groups.discovery], "proven"),
    ]
    hi = max([float(v.max()) for _n, v, _r in series if v.size] + [1.0])
    size = hi / bins

    edges = np.linspace(0.0, hi, int(bins) + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])

    fig = go.Figure()
    for name, values, role in series:
        if not values.size:
            continue
        counts, _ = np.histogram(values, bins=edges, density=True)
        if normalise == "peak":
            peak = float(counts.max())
            counts = counts / peak if peak > 0 else counts
        # Explicit bars rather than add_histogram, because plotly's histnorm has no
        # peak option and re-binning client-side would make the two modes disagree
        # about where a bar edge is.
        fig.add_bar(
            x=centres, y=counts, name=f"{name} (n={values.size:,})",
            marker=dict(color=colour(role, dark)), opacity=0.45,
            width=float(edges[1] - edges[0]),
            hovertemplate=name + " %{x:.1f} MMboe<br>%{y:.4f}<extra></extra>",
        )

    if show_exceedance:
        for name, values, role in series:
            if not values.size:
                continue
            v, pct = conditional_exceedance(values)
            fig.add_scatter(
                x=v, y=pct, mode="lines", name=f"{name} — P(exceed)", xaxis="x", yaxis="y2",
                line=dict(color=colour(role, dark), width=2.4),
                hovertemplate=(name + "<br>%{y:.1f}% chance of exceeding %{x:.2f} MMboe"
                               "<extra></extra>"),
            )

    if mefs is not None:
        _vline(fig, mefs, p["muted"], "dot", "MEFS")

    y_title = ("Density (area = 1 per class)" if normalise == "density"
               else "Scaled to each class's own peak (not a density)")
    fig.update_layout(
        title="A6 · Where the four volume classes overlap", barmode="overlay",
        xaxis_title="Recoverable resource (MMboe)", yaxis_title=y_title,
    )
    if show_exceedance:
        # A second **y** axis, which the no-dual-axis rule does allow here for the
        # same reason A8 gets a second x: the rule is about a *depth* axis meaning
        # one thing, and neither axis here carries a depth. Both families are read
        # against the same x, which is the volume.
        fig.update_layout(yaxis2=dict(
            title="P(exceeding) — conditional (%)", overlaying="y", side="right",
            range=[0, 105], showgrid=False,
        ))
    apply_plotly(fig, dark, height)
    return fig



# ------------------------------------------------------------------- A9
def pfig_a9_prospect_density(
    ts: TrialSet, *, mefs: float | None = None, bins: int = 40,
    dark: bool = False, height: int | None = PANEL_HEIGHT,
):
    """A9 -- the prospect's resource distribution, on its own.

    A6 puts four classes against each other and is about the *overlap*; this is the
    same rendering with one distribution and is about the **shape** -- where the mass
    sits, how long the tail is, and how far the mean sits from the mode. It belongs
    on the prospect tab because it is the only volume figure there that needs no
    well at all (Lars, 2026-08-11).

    The percentile family is drawn as rules rather than left to the eye, and the
    **mean is drawn thicker than the P50** because on a right-skewed resource
    distribution they are different numbers and the mean is the one that gets
    quoted. Seeing the gap is most of the point of the figure.

    Success trials only: the chance-failure zeros belong to POS, not to the shape.
    """
    p = palette(dark)
    res = np.asarray(ts.col("resource"), dtype=float)
    values = res[res > 0]
    fig = go.Figure()
    if not values.size:
        fig.update_layout(title="A9 · Prospect resource — no successful trials")
        apply_plotly(fig, dark, height)
        return fig

    fig.add_histogram(
        x=values, histnorm="probability density", name=f"Prospect (n={values.size:,})",
        marker=dict(color=rgba("prospect", 0.55, dark),
                    line=dict(color=colour("prospect", dark), width=0.4)),
        xbins=dict(start=0.0, end=float(values.max()), size=float(values.max()) / bins),
        hovertemplate="%{x:.1f} MMboe<br>density %{y:.5f}<extra></extra>",
    )
    stats = {
        "P90": float(np.percentile(values, 10.0)),
        "P50": float(np.percentile(values, 50.0)),
        "Mean": float(np.mean(values)),
        "P10": float(np.percentile(values, 90.0)),
    }
    for name, value in stats.items():
        _vline(fig, value, colour("prospect", dark) if name == "Mean" else p["muted"],
               "solid" if name == "Mean" else "dash", f"{name} {value:,.1f}")
    if mefs is not None:
        _vline(fig, mefs, colour("minimum", dark), "dot", "MEFS")

    fig.update_layout(
        title="A9 · Prospect resource distribution (success case)",
        xaxis_title="Recoverable resource (MMboe)", yaxis_title="Density",
        showlegend=False,
    )
    fig.update_xaxes(rangemode="tozero")
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
    # The **spread** around the proven mean, from the workbook's own "Proven 90 /
    # P50 / P10" curves (Lars, 2026-08-11). Thin and in the proven colour, so the
    # mean stays the read and the band is context: a mean without its range is the
    # number people quote and then argue about.
    #
    # Conditional percentiles -- the success case, given a discovery -- which is
    # where percentiles are defined. B9 is where they get weighted by chance.
    for values, name, dash in (
        (vsweep.proven_p90, "Proven P90 | discovery", "dot"),
        (vsweep.proven_p50, "Proven P50 | discovery", "dash"),
        (vsweep.proven_p10, "Proven P10 | discovery", "dot"),
    ):
        if values is None:
            continue
        fig.add_scatter(
            x=thin(values, vsweep.n_discovery, min_support), y=vsweep.z, mode="lines",
            name=name, line=dict(color=colour("proven", dark), width=1.0, dash=dash),
            hovertemplate=name + "<br>%{x:.2f} MMboe at " + DEPTH_HOVER + "<extra></extra>",
        )

    if current_z is not None:
        _hline(fig, current_z, p["text_secondary"], "dash")

    fig.update_layout(
        title=f"B1 · Volume split vs location (exit = entry + {vsweep.z_gap:.0f} m)",
        xaxis_title="Resource (MMboe) — thick lines are means, thin are proven P90/P50/P10",
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

    Four curves, and **only ``P_well`` is unconditional** -- the other three each
    condition on a different event, so none of them may be read against it as if on
    one scale. That is why the crossing marked on this figure names the two curves
    that meet rather than claiming "chance equals regret".

    The proven and possible curves are mirror images: deepening the well moves
    volume from the second into the first, and the pair is the clearest statement
    here of what a deeper exit actually buys. Their sum is not 1 -- a single trial
    can have both halves above MEFS.

    ``VolumeSweep`` also carries ``p_well_exits_in_hc``, the *geometric* reading:
    given a discovery, the chance the well leaves the reservoir still in
    hydrocarbons at all. It is not drawn here because it is not on the same footing
    as the others -- it has no threshold in it, and it is non-monotone in depth,
    since the discovery group it conditions on shrinks as the entry deepens.
    """
    if vsweep.mefs is None or vsweep.p_proven_exceeds_mefs is None or vsweep.p_attic_exceeds_mefs is None:
        raise ValueError("pfig_b2_chance_vs_regret needs a VolumeSweep run with a mefs threshold")
    p = palette(dark)
    fig = go.Figure()
    # P_well is unconditional, so it is never thinned; the two conditional
    # curves are.
    p_proven = thin(vsweep.p_proven_exceeds_mefs, vsweep.n_discovery, min_support)
    p_attic = thin(vsweep.p_attic_exceeds_mefs, vsweep.n_dry, min_support)
    series = [
        (vsweep.p_well, "P<sub>well</sub>", "p_well", 3),
        (p_proven, "P(proven > MEFS | discovery)", "proven", 2.5),
        (p_attic, "P(attic > MEFS | dry & charged)", "attic", 2.5),
    ]
    # **P(possible below exit > MEFS | discovery)** (Lars, 2026-08-12 asked whether
    # the possible-below-exit probability could be shown against depth). Same
    # conditioning and same threshold as the proven curve, so the two are directly
    # comparable and their sum is not 1: a trial can have both above MEFS.
    #
    # It falls as the well goes deeper, because a deeper exit leaves less below it --
    # which is the mirror image of the proven curve rising, and the pair is the
    # clearest statement on this figure of what deepening the well actually buys.
    if vsweep.p_possible_exceeds_mefs is not None:
        series.append((
            thin(vsweep.p_possible_exceeds_mefs, vsweep.n_discovery, min_support),
            "P(possible below exit > MEFS | discovery)", "possible", 2.5,
        ))
    for values, name, role, width in series:
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
    ts: TrialSet | None = None, mefs: float | None = None, statistic: str = "mean",
    zlim: tuple[float, float] | None = None,
    show_depth_labels: bool = True, dark: bool = False, height: int | None = PANEL_HEIGHT,
):
    """B6 -- the inverse: volume to prove against the entry depth it demands.

    **One question, one answer.** *"I must prove this much to sanction. How deep does
    the well have to enter, and what does that cost me in chance?"* x is the volume
    you want, y is the shallowest entry depth that delivers it, and the marker colour
    is ``P_well`` there. Read left to right: wanting more pushes you down-dip, and
    the markers pale as chance drains away.

    The workbook's H38-H40 block as a curve, and the fourth question the tool exists
    to answer. Depth on y and inverted like every other depth axis, so demanding more
    volume moves the answer visibly *down* the structure.

    ``P_well`` is the colour of the curve rather than a second y-axis -- dual y-axes
    are forbidden, and the trade is the point: the marker darkens where the
    requirement is cheap in chance and pales where it is expensive. Hover gives all
    three numbers at once, which is what this figure is for.

    **It answers a guarantee, not a first touch**, and the y-axis title says so.
    :func:`wellvolpos.core.sweep._required_depth` takes a running minimum from the
    deep end, so the depth returned is the shallowest one from which the proven mean
    stays at or above the target *all the way down*. A sampled proven-mean curve
    wobbles wherever the discovery group is thin; inverting its first crossing
    returns depths that deeper locations contradict, which is no basis for a well
    proposal.

    **The contact spread is drawn on the same axes** (Lars, 2026-08-11, who asked for
    it side by side, then on one graph): for each volume, the P99 / P90 / P50 / P10 /
    P1 hydrocarbon-water contact among the trials that actually hold at least that
    much. The requirement gives a depth; this shows how wide the honest answer around
    it is. Rose's Figure 4 is the point of it -- *"The EUR of 9.4 MMBO is associated
    with productive areas from 200 to 1500 acres"* -- against the workbook's ``BA``
    column, which averages those contacts into one number and calls it a required
    depth.

    **The two families do not measure the same thing, and both axes say so.** x is a
    *target mean proven volume, over the discovery group* for the curve and a *total
    resource held by one trial* for the grey lines -- 33.9-277.7 against 2.2-482.1
    MMboe on prospect B. y is a *required entry depth* against a *sampled contact*.
    They therefore **cross**, and a crossing here means nothing: it is not one family
    passing through another, it is two different questions plotted on borrowed axes.

    That is why the axis titles name both readings and the two families are separated
    by weight and hue -- the requirement carries the ``P_well`` colour scale, the
    spread is neutral muted grey drawn underneath it. Unlabelled, this is precisely
    the figure Lars reported as unreadable, so the labels are the mechanism rather
    than the decoration. If a later change shortens them to one quantity, the figure
    goes back to claiming a comparison it cannot support.
    """
    p = palette(dark)
    targets, z_req, p_at = volume_target_curve(vsweep, n=n_targets, ts=ts,
                                               statistic=statistic)
    stat_label = TARGET_STATISTIC_LABELS[statistic]
    fig = go.Figure()

    if targets.size == 0 or not np.isfinite(z_req).any():
        fig.add_annotation(x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False,
                           text="No proven-volume curve to invert",
                           font=dict(size=13, color=p["text"]))
        fig.update_layout(title="B6 \u00b7 Inverse \u2014 volume to prove")
        apply_plotly(fig, dark, height)
        return fig

    # -------------------------------------------------------- the requirement
    if vsweep.alpha is not None:
        z_lo, z_hi = volume_target_band(vsweep, targets)
        band = np.isfinite(z_lo) & np.isfinite(z_hi)
        if band.any():
            level = 100 * (1 - vsweep.alpha)
            fig.add_scatter(
                x=np.concatenate([targets[band], targets[band][::-1]]),
                y=np.concatenate([z_lo[band], z_hi[band][::-1]]),
                fill="toself", fillcolor=rgba("p_well", 0.15, dark), mode="lines",
                line=dict(width=0), name=f"nominal {level:.0f}% CI on the {stat_label} \u2014 sampling error",
                hoverinfo="skip",
            )

    # --------------------------------------------------- the contact spread
    # Drawn *before* the requirement so the coloured markers sit on top of the
    # grey. In an overlay the two families separate by weight and hue rather than
    # by position, so which is drawn last is not a detail.
    spread_depths: list[np.ndarray] = []
    if ts is not None:
        res_all = np.asarray(ts.col("resource"), dtype=float)
        res_all = res_all[res_all > 0]
        held = np.linspace(float(np.percentile(res_all, 5)),
                           float(np.percentile(res_all, 95)), int(n_targets))
        band_pct = entry_depth_percentiles(ts, held)
        # **P90 to P10**, not P99 to P1 (Lars, 2026-08-11). The shaded region is
        # the body of the contact distribution; P99 and P1 stay as thin lines
        # outside it. Filling to the extremes made the grey swamp the figure and
        # implied the whole range was equally likely, which is what a fill reads as.
        lo, hi = band_pct[90], band_pct[10]
        inner = np.isfinite(lo) & np.isfinite(hi)
        if inner.any():
            fig.add_scatter(
                x=np.concatenate([held[inner], held[inner][::-1]]),
                y=np.concatenate([lo[inner], hi[inner][::-1]]),
                fill="toself", fillcolor=rgba("muted", 0.14, dark), mode="lines",
                line=dict(width=0), name="P90\u2013P10 contact spread \u2014 geological range",
                hoverinfo="skip",
            )
        for q, width, dash in ((99, 1.0, "dot"), (90, 1.4, "dash"), (50, 2.2, "solid"),
                               (10, 1.4, "dash"), (1, 1.0, "dot")):
            depths = band_pct[q]
            good = np.isfinite(depths)
            if good.sum() < 2:
                continue
            spread_depths.append(depths[good])
            fig.add_scatter(
                x=held[good], y=depths[good], mode="lines",
                line=dict(color=p["muted"], width=width, dash=dash),
                name=f"P{q} contact \u2014 of trials holding this volume",
                hovertemplate=(f"P{q} of the contacts among trials holding "
                               "%{x:.1f} MMboe or more<br>" + DEPTH_HOVER
                               + "<extra></extra>"),
            )
        # The **mean** contact of the qualifying trials (Lars, 2026-08-12). Grey like
        # the percentiles because it belongs to the same family and is read against
        # the same axes -- but dash-dot, and named "mean", because it is not one of
        # them: on a skewed set of contacts it does not sit at the P50, and this is
        # the number the workbook's BA column quotes on its own.
        mean_contact = band_pct.get("mean")
        if mean_contact is not None:
            good = np.isfinite(mean_contact)
            if good.sum() >= 2:
                spread_depths.append(mean_contact[good])
                fig.add_scatter(
                    x=held[good], y=mean_contact[good], mode="lines",
                    line=dict(color=p["muted"], width=1.8, dash="dashdot"),
                    name="Pmean contact \u2014 of trials holding this volume",
                    hovertemplate=("mean contact among trials holding %{x:.1f} MMboe "
                                   "or more<br>" + DEPTH_HOVER + "<extra></extra>"),
                )
        if mefs is not None:
            _vline(fig, float(mefs), p["muted"], "dot", "MEFS")

    ok = np.isfinite(z_req)
    fig.add_scatter(
        x=targets[ok], y=z_req[ok], mode="lines+markers",
        # **The well's own violet**, not a second grey (Lars, 2026-08-11). The
        # requirement line and the contact lines were both greys, so the two
        # families separated only by weight -- and on a figure whose whole risk is
        # being read as one family, that was not enough. Violet is the palette's
        # ``well`` role and this line is literally where the well must go.
        line=dict(color=colour("well", dark), width=2.0),
        marker=dict(
            # **Inferno**, not the single-hue blue ramp (Lars, 2026-08-11: the
            # points were hard to tell apart). Blue light-to-dark is the project's
            # sequential default and it is genuinely hard to read as a *value* at
            # 9 px; inferno is still perceptually uniform and still not a rainbow,
            # so it keeps the spirit of the rule while being legible. It is the same
            # scale A4's trial-count grid uses, on Lars's earlier instruction.
            size=10, color=p_at[ok] * 100.0, colorscale=VALUE_CMAP,
            cmin=0, cmax=100, line=dict(width=0.6, color=p["surface"]),
            # **Inside the axes.** The depth-row rule pins margin.r at 25 with
            # autoexpand off, so a colourbar in plotly's default position -- outside
            # the plot area on the right -- is simply clipped away, which is what had
            # been happening here unnoticed. Since P_well *is* the cost side of the
            # trade this figure exists to show, an invisible scale makes the colour
            # decorative. Horizontal in the bottom-left corner, which the curve never
            # reaches: the requirement runs top-left to bottom-right.
            # Position set by theme.apply_plotly -- see the note on A4's colourbar.
            colorbar=dict(title=dict(text="P<sub>well</sub> (%)", side="top"), x=0.5),
        ),
        name=f"Required entry \u2014 for a target {stat_label} volume",
        customdata=p_at[ok] * 100.0,
        hovertemplate=(
            "to prove %{x:.2f} MMboe of mean proven volume<br>enter at " + DEPTH_HOVER
            + "<br>P<sub>well</sub> %{customdata:.1f}%<extra></extra>"
        ),
    )

    # **The other three statistics, thin and violet, no markers** (Lars, 2026-08-12).
    # The main curve carries whichever statistic the user chose, with markers coloured
    # by P_well; these are the remaining three drawn faintly beside it so the *spread
    # of the requirement itself* is visible. Deliberately marker-free: markers here
    # would be read as a second P_well scale, and there is only one.
    #
    # P90 is the low case and needs the deepest well; P10 the high case and the
    # shallowest. So the three fan out around the chosen one, and the width of that
    # fan is how much the answer depends on which discovery you are asking about.
    for other in ("p90", "p50", "p10"):
        if other == statistic:
            continue
        try:
            o_targets, o_z, _ = volume_target_curve(vsweep, n=n_targets, ts=ts,
                                                    statistic=other)
        except ValueError:
            continue                      # this sweep carries no such curve
        good = np.isfinite(o_z)
        if good.sum() < 2:
            continue
        fig.add_scatter(
            x=o_targets[good], y=o_z[good], mode="lines",
            line=dict(color=colour("well", dark), width=1.0,
                      dash="dot" if other != "p50" else "dash"),
            name=f"Required entry \u2014 {TARGET_STATISTIC_LABELS[other]}",
            hovertemplate=(f"to prove %{{x:.2f}} MMboe of {TARGET_STATISTIC_LABELS[other]}"
                           "<br>enter at " + DEPTH_HOVER + "<extra></extra>"),
        )

    if target is not None:
        res = invert_volume_target(vsweep, float(target), ts=ts)
        if res.achievable:
            # A right-angle leader: up the target volume, across to the depth. The
            # figure demonstrating its own reading, because "which axis do I start
            # on" is the first thing anyone has to guess at on an inverse plot and
            # a vertical rule alone answers only half of it.
            fig.add_scatter(
                x=[target, target, float(np.nanmin(targets))],
                y=[float(np.nanmax(z_req)), res.z_required, res.z_required],
                mode="lines", line=dict(color=p["text"], width=1.4, dash="dot"),
                showlegend=False, hoverinfo="skip",
            )
            fig.add_annotation(
                x=target, y=float(np.nanmax(z_req)), text=f"want {target:.0f} MMboe ",
                showarrow=False, xanchor="right", yanchor="bottom",
                font=dict(size=10, color=p["text"]),
            )
            fig.add_scatter(
                x=[target], y=[res.z_required], mode="markers+text",
                marker=dict(size=11, color=p["text"], symbol="circle-open",
                            line=dict(width=2.5)),
                text=[f" enter at {res.z_required:.0f} m \u00b7 P<sub>well</sub> "
                      f"{res.p_well_at:.1%}"],
                textposition="middle right", textfont=dict(size=10, color=p["text"]),
                showlegend=False, hoverinfo="skip",
            )

    fig.update_layout(
        title=f"B6 \u00b7 Inverse \u2014 how deep must the well go to prove a {stat_label} volume?",
        # Both readings named on the axis itself. One pair of axes carrying two
        # definitions of volume and two kinds of depth is only honest if the axis
        # says so -- unlabelled, this is the figure Lars could not read.
        xaxis_title=(f"Volume (MMboe) \u2014 target {stat_label} (curve) / "
                     "held by one trial (grey)"),
    )
    fig.update_xaxes(rangemode="tozero")
    apply_plotly(fig, dark, height)
    all_z = [z_req[ok], *spread_depths]
    all_z = np.concatenate([a for a in all_z if a.size])
    depth_axis_plotly(
        fig, zlim or (float(all_z.min()), float(all_z.max())),
        title=("Depth (m TVDSS) \u2014 required entry, or deeper (curve) / "
               "contact (grey)"),
        show_ticklabels=show_depth_labels,
    )
    return fig


#: Azimuths, in degrees, that successive contour labels are placed at. Stepped so
#: that neighbouring rings -- which crowd together where A(z) steepens -- do not
#: stack their labels into one unreadable column.
_LABEL_AZIMUTHS = (90.0, 55.0, 125.0, 20.0, 160.0, 70.0, 110.0, 40.0, 140.0)


# ------------------------------------------------- grid sizing
def suggest_grid(values: np.ndarray, depths: np.ndarray) -> tuple[int, int]:
    """A defensible default grid for :func:`pfig_a7_resource_grid`.

    **Freedman–Diaconis** on each axis independently: bin width ``2 IQR / n^(1/3)``,
    which is the rule that adapts to spread *and* sample size and is robust to the
    long right tail a resource distribution always has. Sturges would under-bin
    10 000 skewed trials badly; a fixed count would be wrong on any other file.

    Clamped to 15-90 bins per axis. Below 15 the structure this figure exists to
    show is averaged away; above 90 most cells hold one trial or none and the plot
    becomes a scatter drawn expensively.

    The workbook's own ``resource grid`` sheet is a fixed 100 x 100. That is inside
    this range at the top end, and on 10 000 trials it leaves most cells empty --
    which is why the default here is computed rather than copied.
    """
    def bins(x: np.ndarray) -> int:
        x = np.asarray(x, dtype=float)
        x = x[np.isfinite(x)]
        if x.size < 4:
            return 15
        q75, q25 = np.percentile(x, [75, 25])
        iqr = float(q75 - q25)
        span = float(np.ptp(x))
        if iqr <= 0 or span <= 0:
            return 15
        width = 2.0 * iqr / np.cbrt(x.size)
        return int(np.clip(round(span / width), 15, 90))

    return bins(values), bins(depths)


# ------------------------------------------------------------------- A8
def pfig_a8_contact_distribution(
    ts: TrialSet, *, n_bins: int = 40, current_entry: float | None = None,
    zlim: tuple[float, float] | None = None, show_depth_labels: bool = True,
    dark: bool = False, height: int | None = PANEL_HEIGHT,
):
    """A8 -- the contact distribution recovered from the trials, two ways at once.

    Bars: the **density** of sampled hydrocarbon-water contacts, as a horizontal
    histogram so depth stays on y where the depth rule requires it. This is the
    distribution the HCWC Builder produces and GeoX consumes, read back out of the
    trial file -- and the shape of it is what every location result in this tool
    ultimately rests on.

    Line: **P(contact deeper than z)**, the inverse cumulative. Read a depth off the
    y-axis and this is the fraction of *success* trials whose contact lies below it
    -- which is ``r_location`` at that entry depth, crest-referenced. So A8 is the
    raw material of A3 shown as a distribution rather than as a chance curve, and
    the two must agree at every depth.

    **Two x-axes, and this is the one place the project allows it.** Counts and
    probability share no units and no scale; the alternative is two panels, and the
    whole point is to see the mode of the distribution sitting against the steep
    part of the cumulative. The rule that matters -- CLAUDE.md's *"no dual y-axes,
    ever"* -- is about the *depth* axis meaning one thing, and it is untouched here:
    both series are read against the same y.
    """
    p = palette(dark)
    res = np.asarray(ts.col("resource"), dtype=float)
    contact = np.asarray(ts.col("contact"), dtype=float)
    ok = (res > 0) & np.isfinite(contact)
    contact = contact[ok]
    lo, hi = float(contact.min()), float(contact.max())

    counts, edges = np.histogram(contact, bins=int(n_bins), range=(lo, hi))
    centres = 0.5 * (edges[:-1] + edges[1:])
    width = float(edges[1] - edges[0])

    fig = go.Figure()
    fig.add_bar(
        x=counts, y=centres, orientation="h", width=width * 0.92,
        marker=dict(color=rgba("prospect", 0.55, dark),
                    line=dict(color=colour("prospect", dark), width=0.5)),
        name="Sampled contacts", xaxis="x2",
        hovertemplate="%{x:,.0f} trials with a contact near " + DEPTH_HOVER + "<extra></extra>",
    )

    # The inverse cumulative, from the sorted contacts rather than from the bars, so
    # it is exact rather than binned -- it has to equal r_location, which is computed
    # from the trials themselves.
    ordered = np.sort(contact)
    deeper = 100.0 * (ordered.size - np.arange(ordered.size)) / ordered.size
    fig.add_scatter(
        x=deeper, y=ordered, mode="lines", name="P(contact deeper than this)",
        line=dict(color=colour("well_associated", dark), width=2.6),
        hovertemplate="%{x:.1f}% of success trials have a contact below "
                      + DEPTH_HOVER + "<extra></extra>",
    )
    if current_entry is not None:
        _hline(fig, current_entry, p["well"], "dash", "well entry")

    fig.update_layout(
        title=f"A8 · Contact distribution and P(deeper) — {contact.size:,} success trials",
        xaxis=dict(title="P(contact deeper than this depth)  (%)", range=[0, 105]),
        xaxis2=dict(title="trials per bin", overlaying="x", side="top",
                    showgrid=False, rangemode="tozero"),
        bargap=0.05,
    )
    apply_plotly(fig, dark, height)
    depth_axis_plotly(fig, zlim or (lo, hi), show_ticklabels=show_depth_labels)
    return fig


# ------------------------------------------------------------------- B9
def pfig_b9_chance_weighted(
    vsweep: VolumeSweep, *, current_z: float | None = None,
    zlim: tuple[float, float] | None = None, show_depth_labels: bool = True,
    min_support: int = MIN_SUPPORT, dark: bool = False, height: int | None = PANEL_HEIGHT,
):
    """B9 -- chance-weighted resource against location: where the expectation peaks.

    ``P_well(z) x mean volume(z)``, swept. The planning question this answers is
    Lars's: *where do I target the most resource for the least risk?* -- and the
    answer is not the deepest location, nor the shallowest.

    **Every series here is P_well times a MEAN**, and the labels now say so (Lars,
    2026-08-12 asked which statistic "Well associated -- chance weighted" was). It is
    the arithmetic mean over the group the series conditions on, not a percentile:
    on a right-skewed resource distribution those are materially different numbers,
    and "chance weighted" alone did not distinguish them. The grey tails are the same
    weighting applied to P99/P90/P10/P1 of the *proven* volume, so they are labelled
    with their percentile explicitly.

    It is a product of a falling curve and a rising one. Chance falls down-dip
    because fewer contacts lie below the well; volume rises because a deeper well
    that does find hydrocarbons finds more of them. So the product usually has an
    **interior maximum**, and that depth is the expectation-maximising target.
    Drawn for both the proven volume and the whole well-associated volume, because
    they peak in different places and the difference is the exit depth's doing.

    **This is an expected value, and expected values describe no outcome that can
    happen.** The well either finds something near the success-case mean or it finds
    nothing; it never finds the chance-weighted number. It is the right quantity to
    *rank locations* with and the wrong one to quote as a volume -- which is why the
    success-case means stay on B1 and B7 beside it, and why this figure says so in
    its axis title rather than calling itself "resource".

    The same caution the source workbook's own *'Risked' Pmean* column deserves.
    """
    p = palette(dark)
    z = vsweep.z
    fig = go.Figure()
    pw = thin(vsweep.p_well, vsweep.n_discovery, min_support)
    series = [
        ("Proven MEAN × P_well", thin(vsweep.proven_mean, vsweep.n_discovery, min_support),
         "tested"),
    ]
    if vsweep.discovery_mean is not None:
        series.append(
            ("Well associated MEAN × P_well",
             thin(vsweep.discovery_mean, vsweep.n_discovery, min_support), "well_associated")
        )

    # The chance-weighted **spread**, drawn first so the mean lines sit on top of
    # it. P_well x the conditional percentiles of the proven volume: the same
    # weighting applied to the range rather than only to its centre, because an
    # expectation quoted without one is the number that gets argued about.
    #
    # The P90-P10 fill is the body of the distribution; **P99 and P1 are drawn as
    # thin grey lines outside it** (Lars, 2026-08-11) rather than widening the fill,
    # because on a right-skewed resource distribution P1 runs a long way above P10
    # and a fill out to it would swamp the mean lines this figure is about. Grey and
    # thin is the same convention A1's thickness family and A4's percentiles use:
    # context, not content.
    if vsweep.proven_p90 is not None and vsweep.proven_p10 is not None:
        lo = pw * thin(vsweep.proven_p90, vsweep.n_discovery, min_support)
        hi = pw * thin(vsweep.proven_p10, vsweep.n_discovery, min_support)
        band = np.isfinite(lo) & np.isfinite(hi)
        if band.any():
            fig.add_scatter(
                x=np.concatenate([lo[band], hi[band][::-1]]),
                y=np.concatenate([z[band], z[band][::-1]]),
                fill="toself", fillcolor=rgba("tested", 0.20, dark), mode="lines",
                line=dict(width=0), name="Proven P90–P10 × P_well",
                hoverinfo="skip",
            )
    for stat, label, dash in (
        ("proven_p99", "P99", "dot"), ("proven_p90", "P90", "dash"),
        ("proven_p10", "P10", "dash"), ("proven_p1", "P1", "dot"),
    ):
        values = getattr(vsweep, stat, None)
        if values is None:
            continue
        weighted = pw * thin(values, vsweep.n_discovery, min_support)
        if np.isfinite(weighted).sum() < 2:
            continue
        # Drawn full length, NaNs and all -- plotly breaks the line at a gap, and
        # keeping every series on the same grid means they can be compared index by
        # index instead of only by eye.
        fig.add_scatter(
            x=weighted, y=z, mode="lines",
            line=dict(color=p["muted"], width=1.0, dash=dash),
            name=f"Proven {label} × P_well",
            hovertemplate=(f"proven {label} x P_well"
                           "<br>%{x:.2f} MMboe at " + DEPTH_HOVER + "<extra></extra>"),
        )

    best_note = []
    for name, mean, role in series:
        weighted = pw * mean
        fig.add_scatter(
            x=weighted, y=z, mode="lines", name=name,
            line=dict(color=colour(role, dark), width=2.4),
            customdata=np.column_stack([pw * 100.0, mean]),
            hovertemplate=(name + "<br>%{x:.2f} MMboe expected"
                           "<br>= %{customdata[0]:.1f}% × %{customdata[1]:.1f} MMboe at "
                           + DEPTH_HOVER + "<extra></extra>"),
        )
        if np.any(np.isfinite(weighted)):
            i = int(np.nanargmax(weighted))
            fig.add_scatter(
                x=[weighted[i]], y=[z[i]], mode="markers+text",
                marker=dict(symbol="star", size=13, color=colour(role, dark)),
                text=[f"  {weighted[i]:.1f} MMboe at {z[i]:.0f} m"],
                textposition="middle right",
                textfont=dict(size=9, color=colour(role, dark)), showlegend=False,
                hovertemplate=f"maximum expectation<br>{weighted[i]:.2f} MMboe at "
                              f"{z[i]:.0f} m TVDSS<extra></extra>",
            )
            best_note.append(z[i])
    if current_z is not None:
        _hline(fig, current_z, p["text"], "dash")

    fig.update_layout(
        title="B9 · Chance-weighted resource vs location (expected, not a volume anyone finds)",
        xaxis_title="P_well × mean volume  (MMboe, expected)",
    )
    fig.update_xaxes(rangemode="tozero")
    apply_plotly(fig, dark, height)
    depth_axis_plotly(fig, zlim or (float(z.min()), float(z.max())),
                      show_ticklabels=show_depth_labels)
    return fig



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
    label_every: int = 4, volume_scale: str = "linear",
    dark: bool = False, height: int | None = PANEL_HEIGHT,
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

    ``volume_scale`` switches the volume axis between ``"linear"`` and ``"log"``
    (Lars, 2026-08-12). Both readings are worth having and they are not cosmetic
    variants of each other:

    * **linear** shows the *absolute* rate of exchange -- how many MMboe a point of
      chance buys -- which is what a well proposal argues about.
    * **log** shows the *proportional* one. On a closure whose volume spans an order
      of magnitude across the swept range, the shallow end of a linear frontier is
      crushed into the axis and the trade there is unreadable; on a log axis a
      straight segment means a constant *percentage* of volume per point of chance.

    The axis title says which is on, because a frontier that looks straight means
    different things in the two and there is no other cue.
    """
    if volume_scale not in ("linear", "log"):
        raise ValueError(f"unknown volume_scale {volume_scale!r}; expected 'linear' or 'log'")
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

    log = volume_scale == "log"
    fig.update_layout(
        title=(f"B7 · Chance against volume — the location trade-off "
               f"({reference_label(vsweep.reference)})"),
        xaxis_title=("Mean resource (MMboe, log scale — a straight segment is a "
                     "constant % per point of chance)" if log else
                     "Mean resource (MMboe, linear scale)"),
        yaxis_title="P_well  (%)",
    )
    if log:
        # rangemode="tozero" is meaningless on a log axis and plotly warns; the
        # range comes from the data instead. MEFS and the frontier are all positive
        # here, so nothing is lost.
        fig.update_xaxes(type="log")
    else:
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
def _well_track(fig, ad: AreaDepth, *, z_entry: float, z_exit: float,
                zlim: tuple[float, float] | None, dark: bool, transform):
    """Draw the well itself as a vertical line on an area-depth panel.

    The two horizontal rules say at what *depth* the well enters and leaves the
    reservoir; they do not show the well. This does, and on a section that is the
    thing the eye looks for first (Lars, 2026-08-11).

    **Where it goes on the x-axis, and why that is not arbitrary.** x here is
    enclosed area, which is not a spatial coordinate, so "vertical" is nominal --
    but there is exactly one honest anchor: ``A(z_entry)``, the area of the contour
    the well enters the reservoir on. That is the point where the entry rule meets
    the top-reservoir curve, so the track starts *on* the structure rather than
    floating beside it, and its foot at ``(A(z_entry), z_exit)`` sits inside the
    closure, since ``A(z_exit) > A(z_entry)``. Any other x would be a decoration.

    B0 puts the well at x = 0 because that panel's x is a schematic half-width and
    zero is the crest line. Here zero area *is* the apex, which is the one place the
    well demonstrably is not.

    Drawn in two pieces, following B0: thick between entry and exit -- the part in
    the reservoir, which is what the classes are cut on -- and thin above it, so the
    track reads as a borehole arriving from above rather than a floating segment.
    """
    x = float(transform(np.asarray([ad.area_at(z_entry)], dtype=float))[0])
    p = palette(dark)
    top = (zlim or (ad.shallowest, ad.deepest))[0]
    if z_entry > top:
        fig.add_scatter(
            x=[x, x], y=[top, z_entry], mode="lines", showlegend=False,
            line=dict(color=p["well"], width=1.6, dash="dot"), hoverinfo="skip",
        )
    fig.add_scatter(
        x=[x, x], y=[z_entry, z_exit], mode="lines", name="the well",
        line=dict(color=p["well"], width=6), showlegend=False,
        hovertemplate="the well " + DEPTH_HOVER + "<extra></extra>",
    )
    return fig


def pfig_c1_section(
    ad: AreaDepth, ts: TrialSet, *, z_entry: float, z_exit: float,
    area_scale: str = "area", zlim: tuple[float, float] | None = None,
    dark: bool = False, height: int | None = PANEL_HEIGHT,
):
    """C1 -- the structure, above C2's curves.

    The pair is the argument: this panel says *where* each volume sits in the
    structure, C2 says what it is worth and how likely it is, and the two carry the
    same four colours so a reader moves between them without a key.

    **Fully labelled** (Lars, 2026-08-11). It ran unlabelled for a while, on the
    reasoning that beside C2 its job was to be recognised rather than read. That was
    wrong in use: the pair only makes its argument if both halves carry a scale.
    C2 says what each volume is worth; C1 has to say where it sits, and "above the
    well at 2205 m" is not sayable without a depth axis. The class shading, the
    entry and exit rules and the depth scale are the content, not decoration.

    A1 higher up the same tab draws the same band with the thickness family and the
    area percentiles; this one stays simpler on purpose -- one base reservoir, no
    area curves -- so the classes are what the eye lands on.
    """
    p = palette(dark)
    _, transform = AREA_SCALES.get(area_scale, AREA_SCALES["area"])
    fig = go.Figure()
    note = _reservoir_band(fig, ad, ts, z_entry=z_entry, z_exit=z_exit, dark=dark,
                           transform=transform, labels=True)
    _hline(fig, z_entry, p["well"], "dash", "well entry")
    if z_exit != z_entry:
        _hline(fig, z_exit, p["well"], "dot", "well exit")
    _well_track(fig, ad, z_entry=z_entry, z_exit=z_exit, zlim=zlim,
                dark=dark, transform=transform)

    fig.update_layout(
        title=f"C1 · the structure, and the volumes a well at this depth divides it into{note}",
        xaxis_title=AREA_SCALES.get(area_scale, AREA_SCALES["area"])[0],
        showlegend=False,
    )
    fig.update_xaxes(rangemode="tozero")
    apply_plotly(fig, dark, height)
    depth_axis_plotly(fig, zlim or (ad.shallowest, ad.deepest))
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
            # **Markers on both families, values on the conditional one only**
            # (Lars, 2026-08-12). Both were labelled until then, on the reasoning
            # that seeing the same volume twice at two heights was the lesson. In
            # practice it doubled the text on the busiest figure in the app for no
            # new information: **the volumes are identical between the two readings**
            # -- risking scales the probability, never the volume -- so the second
            # copy of each number said nothing the first had not.
            #
            # The risked curve keeps its markers, because *where* it sits is the
            # whole point: the same P50 volume at a lower height is the location
            # penalty made visible. It is the redundant label that goes, not the mark.
            _mark_exceedance(
                fig, values, role, dark, chance=chance_used, size=6,
                show_text=reading == "conditional",
                textposition="middle right",
            )
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
