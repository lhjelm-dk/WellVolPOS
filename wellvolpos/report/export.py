"""Export: one bundle, four formats, no second definition of the answer.

Everything here derives from a single :class:`Bundle`, assembled once from the
trial file and a :class:`~wellvolpos.report.case.Case`. That is the whole design
decision. The alternative -- each format computing what it needs -- is how an
XLSX comes to disagree with the PDF beside it, and this codebase has already
shipped three figures that disagreed with the number they were labelled with.
So the bundle computes; the writers only format.

Four outputs, each for a different reader:

``workbook_bytes``
    XLSX. The one a reviewer opens to check arithmetic: every KPI, both engines'
    summaries, the chance decomposition, both sweeps as columns, the QC verdict
    and the case settings. No formulas -- values only, because a formula in an
    exported workbook is a second implementation of the same calculation, and
    the source workbook this project replaces has 3 693 error cells to show
    where that leads.
``pdf_bytes``
    A single PDF: a stamped cover page, then every figure, one per page. The one
    that goes in a well proposal.
``figures_zip``
    PNG or SVG per figure, for dropping into slides.
``Case.to_json``
    The settings, in :mod:`wellvolpos.report.case`. Not a report -- a way to
    reopen this exact session.

**The figures come from the matplotlib set**, not from plotly, because static
plotly export needs kaleido and a browser and this must work offline on a
laptop. That is why every interactive figure has a matplotlib twin and why
``test_every_plotly_figure_has_an_export_twin`` guards it.

**Every export is stamped** with the POS in force and where it came from, the
reference contour and the allocation scheme. A figure can be cropped out of a
screenshot; a cover page and a `Case` sheet cannot be cropped out of a file.
"""

from __future__ import annotations

import numpy as np

import io
from functools import lru_cache
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timezone

import pandas as pd

from ..core.bands import banded_percentiles
from ..core.chance import (
    ChanceResult,
    ReferenceContour,
    allocate,
    expected_volume,
    p_well,
    waterfall_steps,
)
from ..core.classes import VolumeClasses, class_summary, split_trials
from ..core.groups import Groups, group_summary, group_trials
from ..core.reservoir import thickness_from_pay
from ..core.rose import commercial_chance, no_regrets
from ..core.structure import AreaDepth
from ..core.sweep import Sweep, VolumeSweep, run_sweep, run_volume_sweep
from ..io.adapters.base import TrialSet
from ..viz import figures as F
from ..viz.theme import new_figure
from ..ui.numbering import (EXPORT_FIGURE_KEYS, export_filename, export_number,
                            export_sort_key, renumber_title)
from .case import Case, fingerprint

FIGURE_FORMATS = ("png", "svg", "pdf")


# --------------------------------------------------------------------- bundle
@dataclass
class Bundle:
    """Everything an export needs, computed once.

    Assembled by :func:`assemble`. Holding the sweeps here rather than
    recomputing them per format matters in practice: ``run_volume_sweep``
    re-splits every trial at every depth and bootstraps each step, so it is the
    most expensive thing the app does.
    """

    case: Case
    ts: TrialSet
    ad: AreaDepth | None
    groups: Groups
    vc: VolumeClasses | None
    chance: ChanceResult
    sweep: Sweep
    vsweep: VolumeSweep | None
    pos: float
    pos_source: str
    qc: object | None = None
    warnings: list[str] = field(default_factory=list)

    @property
    def stamp(self) -> str:
        """The one line that must appear on every exported artefact."""
        return (
            f"POS_prospect {self.pos:.4f} from {self.pos_source} · "
            f"r_location {self.chance.r_location:.4f} · P_well {self.chance.p_well:.4f} · "
            f"entry {self.case.entry:.0f} m, exit {self.case.exit:.0f} m · "
            f"reference contour {self.case.reference} · allocation {self.case.scheme} · "
            f"MEFS {self.case.mefs:.1f} MMboe"
        )


def assemble(
    ts: TrialSet, case: Case, *, pos: float, pos_source: str,
    qc: object | None = None, n_sweep: int = 60, n_volume_sweep: int = 25,
) -> Bundle:
    """Compute the whole answer once, from the trials and the case.

    ``pos`` and ``pos_source`` are passed in rather than re-derived, because the
    POS in force depends on the risking convention *and* on what the failure
    detector found, and the app already resolved that. Re-deriving it here would
    be a second implementation of the risking branch that `app.py`
    section exists to keep singular.
    """
    entry, exit_ = case.entry, case.exit
    # The case stores the contour by its stable string value; the core takes the
    # enum, so the conversion happens once, here, and fails loudly on an unknown
    # value rather than defaulting to the crest.
    ref = ReferenceContour(case.reference)
    groups = group_trials(ts, entry, exit_)
    chance = p_well(ts, entry, pos, reference=ref)
    sweep = run_sweep(ts, pos, n=n_sweep, z_gap=exit_ - entry, reference=ref)

    ad = vc = vsweep = None
    if ts.has("area"):
        ad = AreaDepth.from_trials(ts.col("contact"), ts.col("area"))
        vc = split_trials(ts, ad, groups, entry, exit_)
        vsweep = run_volume_sweep(
            ts, ad, pos, z_gap=exit_ - entry, mefs=case.mefs,
            reference=ref, n=n_volume_sweep, n_boot=200,
        )

    warnings = list(case.check_against(ts))
    _, alloc_warnings = allocate(case.chance_table, chance.r_location, case.scheme)
    warnings.extend(alloc_warnings)
    if ad is None:
        warnings.append(
            "No productive-area column in this export, so the proven/possible split, the "
            "area-depth figures and the minimum-column-height mapping are absent from this export."
        )
    return Bundle(
        case=case, ts=ts, ad=ad, groups=groups, vc=vc, chance=chance, sweep=sweep,
        vsweep=vsweep, pos=pos, pos_source=pos_source, qc=qc, warnings=warnings,
    )


# --------------------------------------------------------------------- tables
def tables(b: Bundle) -> dict[str, pd.DataFrame]:
    """The workbook's sheets, as an ordered mapping of name to frame.

    Split out from :func:`workbook_bytes` so the numbers can be tested without
    parsing an XLSX, and so the same tables can be shown on screen beside the
    download button -- a reader who can see what is in the file before
    downloading it is a reader who notices when it is wrong.
    """
    out: dict[str, pd.DataFrame] = {}

    # --- Case ---------------------------------------------------------------
    settings = [
        ("Reservoir entry", f"{b.case.entry:.1f}", "m TVDSS"),
        ("Reservoir exit", f"{b.case.exit:.1f}", "m TVDSS"),
        ("MEFS / MCFS", f"{b.case.mefs:.2f}", "MMboe"),
        ("Risking convention", b.case.risking_convention, ""),
        ("POS_prospect in force", f"{b.pos:.6f}", f"from {b.pos_source}"),
        ("Reference contour", b.case.reference, ""),
        ("Allocation scheme", b.case.scheme, ""),
        ("Trial file", b.case.dataset or "—", ""),
        ("Trials", f"{b.ts.n_trials:,}", ""),
        ("Trial fingerprint", fingerprint(b.ts), ""),
        ("Adapter", b.ts.source, ""),
        ("Exported (UTC)", datetime.now(timezone.utc).isoformat(timespec="seconds"), ""),
    ]
    settings.append(("Play chance", f"{b.case.play_chance:.4f}", ""))
    for el, v in b.case.chance_table.items():
        settings.append((f"Chance · {el} | play", f"{v:.4f}", ""))
    out["Case"] = pd.DataFrame(settings, columns=["setting", "value", "units"])

    # --- KPIs ---------------------------------------------------------------
    # r_location and POS_prospect are kept on separate rows and never multiplied
    # into a single reported figure. That separation is the one idea the whole
    # tool rests on, so an exported sheet must not be the place it is lost.
    kpi = [
        ("POS_prospect", b.pos, "chance the prospect contains hydrocarbons"),
        ("r_location", b.chance.r_location,
         "P(contact deeper than the well | hydrocarbons present)"),
        ("P_well", b.chance.p_well, "POS_prospect x r_location — the chance THIS well finds HC"),
    ]
    gs = group_summary(b.ts, b.groups)
    kpi += [
        ("Prospect mean", gs["prospect"]["mean"], "MMboe, all trials incl. failures"),
        ("Discovery mean", gs["discovery"]["mean"], "MMboe, well-associated | discovery"),
        ("Attic mean (dry hole)", gs["attic_dry_hole"]["mean"], "MMboe, up-dip | dry with attic"),
    ]
    if b.vc is not None:
        cs = class_summary(b.vc, b.groups)
        kpi += [
            ("Proven mean", cs["proven"]["mean"], "MMboe — the headline KPI (decision 1)"),
            ("Unproven below LKH, mean", cs["below_lkh"]["mean"], "MMboe, not proven"),
        ]
    kpi.append(
        ("Expected volume at the well", expected_volume(gs["discovery"]["mean"], b.chance.p_well),
         "MMboe — discovery mean x P_well; not a volume anyone finds")
    )
    if b.ad is not None:
        kpi.append(("Apex (derived from A(z))", float(b.ad.apex_estimate()),
                    "m TVDSS — extrapolated, the trials do not reach the crest"))
    out["KPIs"] = pd.DataFrame(kpi, columns=["quantity", "value", "meaning"])

    # --- the two engines ----------------------------------------------------
    out["Reference engine"] = (
        pd.DataFrame(gs).T.rename_axis("group").reset_index()
    )
    if b.vc is not None:
        out["Extension split"] = (
            pd.DataFrame(class_summary(b.vc, b.groups)).T.rename_axis("class").reset_index()
        )

    # --- chance -------------------------------------------------------------
    steps = waterfall_steps(b.case.chance_table, b.chance.r_location, b.pos,
                            weights=b.case.scheme)
    running = 1.0
    rows = []
    for label, factor, role in steps:
        running *= factor
        rows.append({"step": label, "factor": factor, "running": running, "kind": role})
    out["Chance waterfall"] = pd.DataFrame(rows)

    # --- Rose's three quantities -------------------------------------------
    if b.ad is not None and b.vc is not None:
        nr = no_regrets(b.ts, b.ad, b.case.entry)
        cc = commercial_chance(b.ts, b.groups, b.vc.proven, b.chance.p_well, b.case.mefs)
        out["Rose quantities"] = pd.DataFrame(
            [
                ("No Regrets (deterministic)", nr.deterministic, "MMboe",
                 "area at entry x mean pay x mean yield — Rose calls it an oversimplification"),
                ("No Regrets (from the trials)", nr.at_entry_mean, "MMboe",
                 f"mean of the {nr.at_entry_n:,} trials with a contact within "
                 f"±{nr.window_m:.0f} m of the entry"),
                ("Pmcfs(well), whole volume", cc.p_mcfs_downdip, "fraction",
                 "Rose's definition: P(well-associated EUR > MCFS | discovery)"),
                ("Pmcfs, proven only", cc.p_mcfs_proven, "fraction",
                 "what B2 draws — entry-to-exit split, a different and smaller number"),
                ("Pc(well)", cc.pc_well, "fraction",
                 "P_well x Pmcfs(well) — the commercial chance Rose says to use for EMV"),
            ],
            columns=["quantity", "value", "units", "definition"],
        )

    # --- reservoir thickness ------------------------------------------------
    if b.ad is not None:
        tfp = thickness_from_pay(b.ts, b.ad)
        s = tfp.summary()
        out["Reservoir thickness"] = pd.DataFrame(
            [
                ("P90", s["p90"]), ("P50", s["p50"]), ("Mean", s["mean"]), ("P10", s["p10"]),
                ("Minimum", s["min"]), ("Maximum", s["max"]),
                ("Trials resolved", float(tfp.n_resolved)),
                ("Charged to base (excluded)", float(tfp.n_full_to_base)),
                ("Inconsistent (QC flag)", float(tfp.n_inconsistent)),
            ],
            columns=["statistic", "value"],
        )

    # --- sweeps -------------------------------------------------------------
    out["Depth sweep"] = pd.DataFrame({
        "entry_depth_m": b.sweep.z,
        "r_location": b.sweep.r_location,
        "p_well": b.sweep.p_well,
        "uncertainty_reduction": b.sweep.uncertainty_reduction,
        "share_chance_failure": b.sweep.share_chance_failure,
        "share_dry_with_attic": b.sweep.share_dry_with_attic,
        "share_contact_seen": b.sweep.share_contact_seen,
        "share_hc_to_exit": b.sweep.share_hc_to_exit,
    })
    if b.vsweep is not None:
        v = b.vsweep
        out["Volume sweep"] = pd.DataFrame({
            "entry_depth_m": v.z,
            "exit_depth_m": v.z_exit,
            "p_well": v.p_well,
            "proven_mean": v.proven_mean,
            "proven_mean_lo": v.proven_mean_lo,
            "proven_mean_hi": v.proven_mean_hi,
            "below_lkh_mean": v.below_lkh_mean,
            "attic_mean": v.attic_mean,
            "p_proven_exceeds_mefs": v.p_proven_exceeds_mefs,
            "p_attic_exceeds_mefs": v.p_attic_exceeds_mefs,
            "n_discovery": v.n_discovery,
            "n_dry": v.n_dry,
        })

    # --- QC -----------------------------------------------------------------
    if b.qc is not None and getattr(b.qc, "checks", None):
        out["QC"] = pd.DataFrame(
            [{"check": c.name, "level": c.level, "message": c.message} for c in b.qc.checks]
        )

    if b.warnings:
        out["Warnings"] = pd.DataFrame({"warning": b.warnings})
    return out


def workbook_bytes(b: Bundle) -> bytes:
    """The tables as one XLSX, values only.

    Sheet names are truncated to Excel's 31-character limit here rather than
    left to openpyxl, so a long name fails visibly in a test instead of raising
    at the user's download click.
    """
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        for name, frame in tables(b).items():
            frame.to_excel(writer, sheet_name=name[:31], index=False)
    return buf.getvalue()


# -------------------------------------------------------------------- figures
def build_figures(b: Bundle, *, dark: bool = False) -> dict[str, object]:
    """Draw the whole matplotlib set for this bundle.

    One place that knows which figures an export contains, so the PDF, the ZIP
    and the on-screen list cannot come to hold different sets. Figures that need
    a productive-area column are skipped rather than faked when the export has
    none.

    ``dark`` is plumbed through but never selected: the app dropped dark mode and
    draws light only. It is kept because ``viz/theme.py`` still carries the dark
    palette and an export is the one place a dark figure could make sense.

    The caller owns the returned figures and must close them --
    :func:`pdf_bytes` and :func:`figures_zip` do; a dozen live figures per rerun
    is how a Streamlit session runs matplotlib out of memory.

    Matplotlib's twenty-figure warning is lifted for the duration of this call. That
    warning exists to catch figures leaked in a loop, and holding every figure at
    once is precisely this function's contract -- the caller closes them, and
    ``test_the_export_closes_the_figures_it_draws`` proves it does. Scoped here
    rather than set globally, so a real leak anywhere else still warns.
    """
    import matplotlib.pyplot as plt

    with plt.rc_context({"figure.max_open_warning": 0}):
        return _draw_export_figures(b, dark=dark)


def _draw_export_figures(b: Bundle, *, dark: bool = False) -> dict[str, object]:
    c, ch, sc = b.case, b.chance, b.case.scheme
    figs: dict[str, object] = {}

    figs["colour_key"] = F.fig_colour_key(dark=dark)[0]
    figs["A2_outcome_tree"] = F.fig_a2_outcome_tree(b.sweep, current_z=c.entry, dark=dark)[0]
    # pos_trials is not threaded through: A3 draws the reconciliation only when
    # the two POS values differ, and the bundle carries the one in force.
    figs["A3_chance_decomposition"] = F.fig_a3_chance_decomposition(
        b.sweep, pos_prospect=b.pos, current_z=c.entry, dark=dark)[0]
    figs["A4_resource_vs_depth"] = F.fig_a4_resource_vs_depth(
        b.ts, current_entry=c.entry, mefs=c.mefs, dark=dark)[0]
    # The sensitivity fan needs the POS in force, which the bundle already carries
    # -- it is the one figure whose whole subject is that number being uncertain.
    figs["B11_pos_sensitivity"] = F.fig_b11_pos_sensitivity(
        b.sweep, pos_prospect=b.pos, current_z=c.entry, dark=dark)[0]
    figs["B3_uncertainty_reduction"] = F.fig_b3_uncertainty_reduction(
        b.sweep, current_z=c.entry, dark=dark)[0]
    figs["B4_chance_waterfall"] = F.fig_b4_chance_waterfall(
        c.chance_table, ch.r_location, b.pos, scheme=sc, dark=dark)[0]
    figs["B5_allocation_dumbbell"] = F.fig_b5_allocation_dumbbell(
        c.chance_table, ch.r_location, pos_prospect=b.pos, dark=dark)[0]

    figs["A8_contact_distribution"] = F.fig_a8_contact_distribution(
        b.ts, current_entry=c.entry, dark=dark)[0]
    # A9 had a twin in both backends and was never wired into the bundle, so it was
    # on screen and absent from every exported document. Found by audit,
    # 2026-08-11 -- the twin guard checks that a pair *agrees*, not that the export
    # path actually asks for it, which is a different hole.
    figs["A9_prospect_density"] = F.fig_a9_prospect_density(
        b.ts, mefs=c.mefs, dark=dark)[0]

    if b.ad is not None:
        figs["A1_area_depth"] = F.fig_a1_area_depth(
            b.ad, current_entry=c.entry, current_exit=c.exit, dark=dark)[0]
        figs["map_view"] = F.fig_map_view(
            b.ad, apex=float(b.ad.apex_estimate()), z_entry=c.entry, z_exit=c.exit,
            interval=c.map_interval, well_azimuth_deg=c.map_azimuth_deg, dark=dark)[0]
    if b.vc is not None and b.ad is not None:
        # Computed here rather than reused from the workbook sheet: this function is
        # called on its own by figures_zip, where that sheet was never built.
        _cc = commercial_chance(b.ts, b.groups, b.vc.proven, ch.p_well, c.mefs)
        figs["C1_section"] = F.fig_c1_section(
            b.ad, b.ts, z_entry=c.entry, z_exit=c.exit,
            area_scale=c.area_scale, dark=dark)[0]
        figs["C2_exceedance"] = F.fig_c2_exceedance(
            b.ts, b.groups, b.vc, pos_prospect=b.pos, p_well=ch.p_well,
            mefs=c.mefs, pc_well=_cc.pc_well, dark=dark)[0]
        # The wedge is schematic, but it is drawn to *this* prospect's recovered
        # thickness and contact, so it belongs in the bundle rather than in the docs.
        _thick = thickness_from_pay(b.ts, b.ad).thickness
        _t50 = float(np.nanpercentile(_thick[np.isfinite(_thick) & (_thick > 0)], 50))             if np.isfinite(_thick).any() else 50.0
        _dc = b.ts.col("contact")[(b.ts.col("resource") > 0)
                                  & (b.ts.col("contact") > c.entry)]
        if _dc.size:
            figs["C5_partitions"] = F.fig_c5_partitions(
                b.ad, z_entry=c.entry, z_exit=c.exit,
                z_contact=float(np.median(_dc)), area_scale=c.area_scale,
                dark=dark)[0]
        figs["C4_wedge"] = F.fig_c4_wedge(
            thickness=_t50, z_contact=float(np.nanmedian(
                b.ts.col("contact")[b.ts.col("resource") > 0])),
            z_entry=c.entry, z_exit=c.exit,
            apex=float(b.ad.apex_estimate()), dark=dark)[0]
        if b.vsweep is not None and b.vsweep.p_discovery_exceeds_mefs is not None:
            from ..core.utility import hurdle_curve
            figs["B14_hurdle_cost"] = F.fig_b14_hurdle_cost(
                hurdle_curve(b.vsweep), dark=dark)[0]
        figs["C6_outcome_tree"] = F.fig_c6_outcome_tree(
            b.groups, pos_prospect=b.pos, p_well=ch.p_well,
            pc_well=_cc.pc_well, dark=dark)[0]
        figs["C3_mefs_bars"] = F.fig_c3_mefs_bars(
            b.ts, b.groups, b.vc, pos_prospect=b.pos, p_well=ch.p_well,
            mefs=c.mefs, dark=dark)[0]
        figs["A5_exceedance"] = F.fig_a5_exceedance(
            b.ts, b.groups, b.vc, mefs=c.mefs,
            pos_prospect=b.pos, p_well=ch.p_well, dark=dark)[0]
        figs["A6_overlap"] = F.fig_a6_overlap(b.vc, b.groups, mefs=c.mefs, dark=dark)[0]
        figs["B0_section"] = F.fig_b0_section(
            b.ad, z_entry=c.entry, z_exit=c.exit, dark=dark)[0]
        # Banded percentiles: needs the classes, so it lives with the split figures
        # rather than with the sweeps. Drawn at the shipped defaults -- the app's
        # band mode and count are view controls, not settings the Case carries, and
        # a figure in an export has to be reproducible from the Case alone.
        figs["B12_banded_percentiles"] = F.fig_b12_banded_percentiles(
            banded_percentiles(b.ts, b.groups, b.vc, z_entry=c.entry, z_exit=c.exit),
            mefs=c.mefs, dark=dark)[0]
    if b.vsweep is not None:
        figs["B1_volume_split"] = F.fig_b1_volume_split(
            b.vsweep, current_z=c.entry, dark=dark)[0]
        # Its own figure since 2026-08-14: conditional on the well leaving the
        # reservoir in hydrocarbons, so not on the same footing as B1's three.
        figs["B13_below_exit"] = F.fig_b13_below_exit(
            b.vsweep, current_z=c.entry, dark=dark)[0]
        figs["B2_chance_vs_regret"] = F.fig_b2_chance_vs_regret(
            b.vsweep, current_z=c.entry, dark=dark)[0]
        figs["B6_inverse"] = F.fig_b6_inverse(
            b.vsweep, target=c.mefs, ts=b.ts, mefs=c.mefs, dark=dark)[0]
        figs["B7_frontier"] = F.fig_b7_frontier(b.vsweep, current_z=c.entry, dark=dark)[0]
        figs["B8_commercial_chance"] = F.fig_b8_commercial_chance(
            b.vsweep, current_z=c.entry, dark=dark)[0]
        figs["B9_chance_weighted"] = F.fig_b9_chance_weighted(
            b.vsweep, current_z=c.entry, dark=dark)[0]
    return _in_report_order(figs)


#: The two ways a report can be drawn.
#:
#: **Both, and neither is the correction of the other** (Lars, 2026-08-18: *"I want
#: both options to build the report"*). The matplotlib set was built for export and is
#: what every artefact has carried; the plotly set is what the app draws, so a reader
#: who recognises a figure from the screen gets that figure. They are the same
#: computations either way -- ``assemble`` does the arithmetic once and both backends
#: only format it, which is the convention that stops an XLSX disagreeing with the PDF
#: beside it, and it now stops the two image sets disagreeing too.
FIGURE_BACKENDS = ("matplotlib", "plotly")

#: Static images from plotly need a renderer, and it is not a hard dependency: it
#: drives a headless browser and costs a large download. Absent, the plotly backend
#: refuses with a message naming the package rather than failing somewhere inside
#: plotly's writer.
KALEIDO_HINT = (
    "**The plotly report is unavailable here.** It renders the interactive figures to "
    "static images with `kaleido`, which drives a headless browser, and neither is "
    "present.\n\n"
    "- **Running locally?** `pip install -r requirements-dev.txt`, or just "
    "`pip install kaleido`.\n"
    "- **On a hosted deployment?** There is no shell to install into. Add `kaleido` "
    "to `requirements.txt` and a file `packages.txt` containing `chromium`, then "
    "reboot the app. Both are deliberately left out of the default deployment "
    "because the browser install is large and easy to break.\n\n"
    "**Nothing else is affected.** The matplotlib report is the default, produces the "
    "same numbers, and needs no extra install."
)


@lru_cache(maxsize=1)
def kaleido_available() -> bool:
    """Can plotly write a static image here?

    **It renders one, rather than checking that the package imports.** The two are
    different questions and the difference is not academic: kaleido drives a headless
    browser, so on a machine with the package installed and no browser -- a CI runner,
    a slim container -- the import succeeds and every render fails. The docstring
    always asked the right question; the body answered an easier one.

    Cached for the process. The probe costs a second or so on first call and the answer
    cannot change while the app is running, so a Streamlit rerun pays nothing.
    """
    try:
        import plotly.graph_objects as _go

        _go.Figure().to_image(format="png", width=8, height=8)
    except Exception:
        return False
    return True


#: Pixel size for a plotly figure rendered to a page. Wider than the app draws, so the
#: text does not have to be scaled up in a document.
PLOTLY_IMAGE_SIZE = (1400, 900)


def build_plotly_figures(b: Bundle) -> dict[str, object]:
    """The same report, drawn by the interactive backend.

    **Keyed identically to :func:`build_figures`**, and a test asserts it: two builders
    emitting different sets is how an exported document comes to be missing a figure
    that is on screen, which has happened here before with A9.

    Heights are left to each figure. ``apply_plotly`` already sizes them for their
    legends, and overriding that would reintroduce the clipping the legend-aware
    margin exists to prevent.
    """
    import numpy as np

    from .. import viz as _V
    from ..core.bands import banded_percentiles
    from ..core.reservoir import thickness_from_pay
    from ..core.rose import commercial_chance

    c, ch, sc = b.case, b.chance, b.case.scheme
    figs: dict[str, object] = {}

    # The colour key leads both bundles: a reader who opens the archive first meets
    # what the colours mean, and it is the one page that is the same in every report.
    figs["colour_key"] = _V.pfig_colour_key()
    figs["A2_outcome_tree"] = _V.pfig_a2_outcome_tree(
        b.sweep, current_z=c.entry, current_exit=c.exit)
    figs["A3_chance_decomposition"] = _V.pfig_a3_chance_decomposition(
        b.sweep, pos_prospect=b.pos, current_z=c.entry, current_exit=c.exit)
    figs["A4_resource_vs_depth"] = _V.pfig_a4_resource_vs_depth(
        b.ts, current_entry=c.entry, current_exit=c.exit, mefs=c.mefs)
    figs["B11_pos_sensitivity"] = _V.pfig_b11_pos_sensitivity(
        b.sweep, pos_prospect=b.pos, current_z=c.entry, current_exit=c.exit)
    figs["B3_uncertainty_reduction"] = _V.pfig_b3_uncertainty_reduction(
        b.sweep, current_z=c.entry, current_exit=c.exit)
    figs["B4_chance_waterfall"] = _V.pfig_b4_chance_waterfall(
        c.chance_table, ch.r_location, b.pos, scheme=sc)
    figs["B5_allocation_dumbbell"] = _V.pfig_b5_allocation_dumbbell(
        c.chance_table, ch.r_location, pos_prospect=b.pos)
    figs["A8_contact_distribution"] = _V.pfig_a8_contact_distribution(
        b.ts, current_entry=c.entry)
    figs["A9_prospect_density"] = _V.pfig_a9_prospect_density(b.ts, mefs=c.mefs)

    if b.ad is not None:
        figs["A1_area_depth"] = _V.pfig_a1_area_depth(
            b.ad, ts=b.ts, current_entry=c.entry, current_exit=c.exit)
        figs["map_view"] = _V.pfig_map_view(
            b.ad, apex=float(b.ad.apex_estimate()), z_entry=c.entry, z_exit=c.exit,
            interval=c.map_interval, well_azimuth_deg=c.map_azimuth_deg)
    if b.vc is not None and b.ad is not None:
        _cc = commercial_chance(b.ts, b.groups, b.vc.proven, ch.p_well, c.mefs)
        figs["C1_section"] = _V.pfig_c1_section(
            b.ad, b.ts, z_entry=c.entry, z_exit=c.exit, area_scale=c.area_scale)
        figs["C2_exceedance"] = _V.pfig_c2_exceedance(
            b.ts, b.groups, b.vc, pos_prospect=b.pos, p_well=ch.p_well,
            mefs=c.mefs, pc_well=_cc.pc_well)
        _thick = thickness_from_pay(b.ts, b.ad).thickness
        _t50 = (float(np.nanpercentile(_thick[np.isfinite(_thick) & (_thick > 0)], 50))
                if np.isfinite(_thick).any() else 50.0)
        _pay = (np.asarray(b.ts.col("gross_pay"), dtype=float)
                if b.ts.has("gross_pay") else None)
        if _pay is not None:
            _pay = _pay[(np.asarray(b.ts.col("resource"), dtype=float) > 0)
                        & np.isfinite(_pay) & (_pay > 0)]
        _dc = b.ts.col("contact")[(b.ts.col("resource") > 0)
                                  & (b.ts.col("contact") > c.entry)]
        if _dc.size:
            figs["C5_partitions"] = _V.pfig_c5_partitions(
                b.ad, z_entry=c.entry, z_exit=c.exit,
                z_contact=float(np.median(_dc)), area_scale=c.area_scale)
        figs["C4_wedge"] = _V.pfig_c4_wedge(
            thickness=_t50,
            z_contact=float(np.nanmedian(
                b.ts.col("contact")[b.ts.col("resource") > 0])),
            z_entry=c.entry, z_exit=c.exit, apex=float(b.ad.apex_estimate()),
            mean_pay=(float(_pay.mean()) if _pay is not None and _pay.size else None))
        if b.vsweep is not None and b.vsweep.p_discovery_exceeds_mefs is not None:
            from ..core.utility import hurdle_curve
            figs["B14_hurdle_cost"] = _V.pfig_b14_hurdle_cost(hurdle_curve(b.vsweep))
        figs["C6_outcome_tree"] = _V.pfig_c6_outcome_tree(
            b.groups, pos_prospect=b.pos, p_well=ch.p_well, pc_well=_cc.pc_well)
        figs["C3_mefs_bars"] = _V.pfig_c3_mefs_bars(
            b.ts, b.groups, b.vc, pos_prospect=b.pos, p_well=ch.p_well, mefs=c.mefs)
        figs["A5_exceedance"] = _V.pfig_a5_exceedance(
            b.ts, b.groups, b.vc, mefs=c.mefs, pos_prospect=b.pos, p_well=ch.p_well)
        figs["A6_overlap"] = _V.pfig_a6_overlap(b.vc, b.groups, mefs=c.mefs)
        figs["B0_section"] = _V.pfig_b0_section(b.ad, z_entry=c.entry, z_exit=c.exit)
        figs["B12_banded_percentiles"] = _V.pfig_b12_banded_percentiles(
            banded_percentiles(b.ts, b.groups, b.vc, z_entry=c.entry, z_exit=c.exit),
            mefs=c.mefs)
    if b.vsweep is not None:
        figs["B1_volume_split"] = _V.pfig_b1_volume_split(
            b.vsweep, current_z=c.entry, current_exit=c.exit)
        figs["B13_below_exit"] = _V.pfig_b13_below_exit(
            b.vsweep, current_z=c.entry, current_exit=c.exit)
        figs["B2_chance_vs_regret"] = _V.pfig_b2_chance_vs_regret(
            b.vsweep, current_z=c.entry, current_exit=c.exit)
        figs["B6_inverse"] = _V.pfig_b6_inverse(
            b.vsweep, target=c.mefs, ts=b.ts, mefs=c.mefs)
        figs["B7_frontier"] = _V.pfig_b7_frontier(b.vsweep, current_z=c.entry)
        figs["B8_commercial_chance"] = _V.pfig_b8_commercial_chance(
            b.vsweep, current_z=c.entry, current_exit=c.exit)
        figs["B9_chance_weighted"] = _V.pfig_b9_chance_weighted(
            b.vsweep, current_z=c.entry, current_exit=c.exit)
    return _in_report_order(figs)


def plotly_figure_bytes(fig, fmt: str = "png", *, scale: float = 2.0) -> bytes:
    """One plotly figure as PNG, SVG or PDF, through kaleido."""
    if fmt not in FIGURE_FORMATS:
        raise ValueError(f"unknown figure format {fmt!r}; expected one of {FIGURE_FORMATS}")
    if not kaleido_available():
        raise RuntimeError(KALEIDO_HINT)
    w, h = PLOTLY_IMAGE_SIZE
    # The figure's own height wins where it has one -- ``apply_plotly`` sized it for
    # the legend, and forcing a common height is how a twelve-series legend gets
    # clipped out of an exported page.
    h = int(fig.layout.height) if fig.layout.height else h
    # SVG is already vector, so scaling it would only inflate the file.
    return fig.to_image(format=fmt, width=w, height=h,
                        scale=(1.0 if fmt == "svg" else scale))


def _renumber_matplotlib(fig, export_key: str) -> None:
    """Rewrite a drawn matplotlib figure's title to carry its displayed number.

    **Done to the figure rather than inside each of thirty drawing functions.** The
    figure functions are shared with the docs and the tests and know nothing about tabs;
    the number is a property of *where the figure sits in the app today*, which is
    exactly what ``ui/numbering`` owns. The same argument put ``renumber_title`` in
    ``ui.common.chart`` rather than in the plotly figures.

    A ``suptitle`` wins where there is one -- the multi-panel figures use it and their
    axes titles are panel labels, so rewriting an axes title there would renumber a
    panel and leave the figure's own heading alone.
    """
    number = export_number(export_key)
    if not number:
        return
    sup = getattr(fig, "_suptitle", None)
    if sup is not None and sup.get_text():
        sup.set_text(renumber_title(sup.get_text(), EXPORT_FIGURE_KEYS[export_key]))
        return
    for ax in fig.axes:
        if ax.get_title():
            ax.set_title(renumber_title(ax.get_title(),
                                        EXPORT_FIGURE_KEYS[export_key]))
            return


def _renumber_plotly(fig, export_key: str) -> None:
    """The same for a plotly figure. See :func:`_renumber_matplotlib`."""
    key = EXPORT_FIGURE_KEYS.get(export_key)
    if not key:
        return
    title = getattr(getattr(fig.layout, "title", None), "text", None)
    if title:
        fig.update_layout(title=dict(text=renumber_title(title, key)))


def _in_report_order(figs: dict) -> dict:
    """Sort a bundle by figure number, and renumber each title on the way through.

    **Order was dict-insertion order**, which was neither the app's nor numeric -- so
    the PDF ran 3.1, 3.2, 3.3, 3.4, 5.1, 5.2, 2.5, 2.4, 2.1 ... and a reader could not
    follow it against the screen. One place to fix it, because both backends build
    their dict in the order the code happens to be written in.
    """
    out = {}
    for key in sorted(figs, key=export_sort_key):
        fig = figs[key]
        if hasattr(fig, "layout"):
            _renumber_plotly(fig, key)
        else:
            _renumber_matplotlib(fig, key)
        out[key] = fig
    return out



def figure_bytes(fig, fmt: str = "png", *, dpi: int = 200) -> bytes:
    """One figure as PNG, SVG or PDF."""
    if fmt not in FIGURE_FORMATS:
        raise ValueError(f"unknown figure format {fmt!r}; expected one of {FIGURE_FORMATS}")
    buf = io.BytesIO()
    fig.savefig(buf, format=fmt, dpi=dpi, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    return buf.getvalue()


def figures_zip(b: Bundle, fmt: str = "png", *, dpi: int = 200,
                backend: str = "matplotlib") -> bytes:
    """Every figure in one archive, plus the stamp as a text file.

    The stamp travels as `README.txt` inside the archive because the figures
    themselves get separated from each other the moment they are dropped into
    slides -- and a figure separated from its POS provenance is the failure this
    project keeps guarding against.
    """
    if backend not in FIGURE_BACKENDS:
        raise ValueError(f"unknown backend {backend!r}; expected one of {FIGURE_BACKENDS}")
    if backend == "plotly":
        figs = build_plotly_figures(b)
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            z.writestr("README.txt", _cover_text(b))
            z.writestr("case.json", b.case.to_json())
            for name, fig in figs.items():
                z.writestr(f"{export_filename(name)}.{fmt}",
                           plotly_figure_bytes(fig, fmt))
        return buf.getvalue()

    figs = build_figures(b)
    try:
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            z.writestr("README.txt", _cover_text(b))
            z.writestr("case.json", b.case.to_json())
            for name, fig in figs.items():
                z.writestr(f"{export_filename(name)}.{fmt}",
                           figure_bytes(fig, fmt, dpi=dpi))
        return buf.getvalue()
    finally:
        _close(figs)


def build_figure_keys(b: Bundle) -> tuple[str, ...]:
    """Which figures this bundle will contain, without drawing any of them.

    The cover page needs the contents list and is itself drawn *into* the PDF, so
    building the figures to find out what they are would draw everything twice. Both
    builders emit the same keys -- a test asserts it -- so either one's key set is the
    answer, and the cheap way to get it is the same set of conditions they branch on.
    """
    keys = ["colour_key", "A2_outcome_tree", "A3_chance_decomposition",
            "A4_resource_vs_depth", "B11_pos_sensitivity", "B3_uncertainty_reduction",
            "B4_chance_waterfall", "B5_allocation_dumbbell", "A8_contact_distribution",
            "A9_prospect_density"]
    if b.ad is not None:
        keys += ["A1_area_depth", "map_view"]
    if b.vc is not None and b.ad is not None:
        keys += ["C1_section", "C2_exceedance", "C5_partitions", "C4_wedge",
                 "C6_outcome_tree", "C3_mefs_bars", "A5_exceedance", "A6_overlap",
                 "B0_section", "B12_banded_percentiles"]
        if b.vsweep is not None and b.vsweep.p_discovery_exceeds_mefs is not None:
            keys.append("B14_hurdle_cost")
    if b.vsweep is not None:
        keys += ["B1_volume_split", "B13_below_exit", "B2_chance_vs_regret",
                 "B6_inverse", "B7_frontier", "B8_commercial_chance",
                 "B9_chance_weighted"]
    return tuple(keys)


def _cover_text(b: Bundle) -> str:
    """The provenance page, as plain text so the PDF and the ZIP share it.

    The stamp is broken onto its own lines here rather than left as one long
    string: on an A4 page a single 160-character line runs off the edge, and a
    provenance stamp that has to be guessed at is not a provenance stamp.
    """
    lines = [
        "WellVolPOS — well location POS and volume",
        "",
    ]
    lines += [f"  {part.strip()}" for part in b.stamp.split(" · ")]
    lines += [
        "",
        f"Trial file: {b.case.dataset or '—'} · {b.ts.n_trials:,} trials · "
        f"fingerprint {fingerprint(b.ts)}",
        f"Exported {datetime.now(timezone.utc).isoformat(timespec='seconds')} UTC",
        "",
        "P_well = POS_prospect x r_location. The two are never multiplied into one",
        "reported number: r_location is the only quantity the well's position",
        "controls, and POS_prospect is the only one it does not.",
        "",
        "Both engines are shown and neither is the correction of the other: the",
        "reference engine groups whole trials after Schneider et al. (2023); the",
        "extension splits each trial into proven, possible and attic.",
    ]
    # **A contents list, in report order** (2026-08-18). The figure files are named
    # with the number the app shows -- 3.10, not B8 -- which is what a reader has in
    # front of them. The cost of an exact number is that "3.10" sorts before "3.2"
    # alphabetically, so the order lives here rather than being faked by zero-padding
    # a number that would then no longer match the screen.
    lines += ["", "Figures, in report order:"]
    for key in sorted(build_figure_keys(b), key=export_sort_key):
        number = export_number(key)
        lines.append(f"  {(number or '—'):>5}  {export_filename(key)}")

    if b.case.note:
        lines += ["", f"Note: {b.case.note}"]
    if b.warnings:
        lines += ["", "Warnings carried from this session:"]
        lines += [f"  - {w}" for w in b.warnings]
    return "\n".join(lines)


def pdf_bytes(b: Bundle, *, backend: str = "matplotlib") -> bytes:
    """Cover page, then one figure per page.

    Uses matplotlib's own ``PdfPages`` rather than assembling with pypdf: fewer
    moving parts, and the vector content survives, which is the reason to choose
    PDF over PNG in a well proposal.
    """
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    if backend not in FIGURE_BACKENDS:
        raise ValueError(f"unknown backend {backend!r}; expected one of {FIGURE_BACKENDS}")
    if backend == "plotly":
        return _plotly_pdf_bytes(b)

    figs = build_figures(b)
    buf = io.BytesIO()
    cover = None
    try:
        with PdfPages(buf) as pdf:
            cover, ax = new_figure(figsize=(11.7, 8.3))
            ax.axis("off")
            ax.text(0.0, 1.0, _cover_text(b), va="top", ha="left", fontsize=9,
                    family="monospace", transform=ax.transAxes)
            pdf.savefig(cover)
            for name, fig in figs.items():
                pdf.savefig(fig)
            info = pdf.infodict()
            info["Title"] = "WellVolPOS — well location POS and volume"
            info["Subject"] = b.stamp
            info["Creator"] = "WellVolPOS"
            info["CreationDate"] = datetime.now(timezone.utc)
        return buf.getvalue()
    finally:
        _close(figs)
        if cover is not None:
            plt.close(cover)


def _plotly_pdf_bytes(b: Bundle) -> bytes:
    """The plotly report as one PDF: kaleido writes the pages, pypdf staples them.

    The cover page is still drawn by matplotlib. It is a block of monospaced text, not
    a figure, and re-implementing it in plotly would give two covers that could drift
    -- the thing this module exists to prevent. So one cover, both reports.
    """
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from pypdf import PdfReader, PdfWriter

    if not kaleido_available():
        raise RuntimeError(KALEIDO_HINT)

    cover_buf = io.BytesIO()
    cover = None
    try:
        with PdfPages(cover_buf) as pdf:
            cover, ax = new_figure(figsize=(11.7, 8.3))
            ax.axis("off")
            ax.text(0.0, 1.0, _cover_text(b), va="top", ha="left", fontsize=9,
                    family="monospace", transform=ax.transAxes)
            pdf.savefig(cover)
    finally:
        if cover is not None:
            plt.close(cover)

    writer = PdfWriter()
    for page in PdfReader(io.BytesIO(cover_buf.getvalue())).pages:
        writer.add_page(page)
    for fig in build_plotly_figures(b).values():
        for page in PdfReader(io.BytesIO(plotly_figure_bytes(fig, "pdf"))).pages:
            writer.add_page(page)
    writer.add_metadata({
        "/Title": "WellVolPOS — well location POS and volume",
        "/Subject": b.stamp,
        "/Creator": "WellVolPOS",
    })
    out = io.BytesIO()
    writer.write(out)
    return out.getvalue()


def _close(figs: dict[str, object]) -> None:
    import matplotlib.pyplot as plt

    for fig in figs.values():
        plt.close(fig)
