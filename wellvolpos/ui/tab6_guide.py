"""Tab ⑥ — the theory and guide, rendered from :mod:`wellvolpos.report.guide`."""

from __future__ import annotations


from ..report.guide import render as render_guide
from .common import split_caveat
from .context import Ctx


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

    render_guide(
        ts=ts, ad=ad if has_area else None, groups=groups,
        vc=vc if has_area else None, chance=chance, mefs=mefs,
        entry=entry, exit_=exit_, pos_source=pos_source,
    )
