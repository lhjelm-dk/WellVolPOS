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

import io
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np
import pandas as pd

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
from ..core.threshold import apply_min_column_height
from ..io.adapters.base import TrialSet
from ..viz import figures as F
from ..viz.theme import new_figure
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
    be a second implementation of the branch CLAUDE.md's "POS provenance"
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
        ("Minimum column height", f"{b.case.min_column_height:.1f}", "m"),
        ("Trial file", b.case.dataset or "—", ""),
        ("Trials", f"{b.ts.n_trials:,}", ""),
        ("Trial fingerprint", fingerprint(b.ts), ""),
        ("Adapter", b.ts.source, ""),
        ("Exported (UTC)", datetime.now(timezone.utc).isoformat(timespec="seconds"), ""),
    ]
    for el, v in b.case.chance_table.items():
        settings.append((f"Chance · {el}", f"{v:.4f}", ""))
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
            ("Possible mean (below exit)", cs["possible"]["mean"], "MMboe, not proven"),
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

    # --- minimum column height ---------------------------------------------
    if b.ad is not None:
        tm = apply_min_column_height(b.ts, b.ad, float(b.ad.apex_estimate()),
                                     b.case.min_column_height)
        out["Minimum column"] = pd.DataFrame(
            [
                ("Column height", b.case.min_column_height, "m below the derived apex"),
                ("Minimum admissible contact", tm.min_contact_depth, "m TVDSS"),
                ("Equivalent area", tm.min_area if tm.min_area is not None else np.nan, "km²"),
                ("Binds on this data", float(tm.binds), "1 = yes"),
                ("Trials excluded", float(tm.n_excluded), "count — mapping only, nothing filtered"),
            ],
            columns=["quantity", "value", "units"],
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
            "possible_mean": v.possible_mean,
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
def build_figures(b: Bundle, *, dark: bool | None = None) -> dict[str, object]:
    """Draw the whole matplotlib set for this bundle.

    One place that knows which figures an export contains, so the PDF, the ZIP
    and the on-screen list cannot come to hold different sets. Figures that need
    a productive-area column are skipped rather than faked when the export has
    none.

    The caller owns the returned figures and must close them --
    :func:`pdf_bytes` and :func:`figures_zip` do; a dozen live figures per rerun
    is how a Streamlit session runs matplotlib out of memory.
    """
    dark = b.case.dark if dark is None else dark
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
    figs["B3_uncertainty_reduction"] = F.fig_b3_uncertainty_reduction(
        b.sweep, current_z=c.entry, dark=dark)[0]
    figs["B4_chance_waterfall"] = F.fig_b4_chance_waterfall(
        c.chance_table, ch.r_location, b.pos, scheme=sc, dark=dark)[0]
    figs["B5_allocation_dumbbell"] = F.fig_b5_allocation_dumbbell(
        c.chance_table, ch.r_location, pos_prospect=b.pos, dark=dark)[0]

    if b.ad is not None:
        figs["A1_area_depth"] = F.fig_a1_area_depth(
            b.ad, current_entry=c.entry, current_exit=c.exit, dark=dark)[0]
        figs["map_view"] = F.fig_map_view(
            b.ad, apex=float(b.ad.apex_estimate()), z_entry=c.entry, z_exit=c.exit,
            interval=c.map_interval, well_azimuth_deg=c.map_azimuth_deg, dark=dark)[0]
    if b.vc is not None and b.ad is not None:
        figs["concepts"] = F.fig_concepts(
            b.ad, b.ts, b.groups, b.vc, z_entry=c.entry, z_exit=c.exit,
            pos_prospect=b.pos, p_well=ch.p_well, mefs=c.mefs,
            area_scale=c.area_scale, dark=dark)[0]
        figs["A5_exceedance"] = F.fig_a5_exceedance(b.ts, b.groups, b.vc, mefs=c.mefs, dark=dark)[0]
        figs["A6_overlap"] = F.fig_a6_overlap(b.vc, b.groups, mefs=c.mefs, dark=dark)[0]
        figs["B0_section"] = F.fig_b0_section(
            b.ad, z_entry=c.entry, z_exit=c.exit, dark=dark)[0]
    if b.vsweep is not None:
        figs["B1_volume_split"] = F.fig_b1_volume_split(
            b.vsweep, current_z=c.entry, dark=dark)[0]
        figs["B2_chance_vs_regret"] = F.fig_b2_chance_vs_regret(
            b.vsweep, current_z=c.entry, dark=dark)[0]
        figs["B6_inverse"] = F.fig_b6_inverse(b.vsweep, ts=b.ts, dark=dark)[0]
    return figs


def figure_bytes(fig, fmt: str = "png", *, dpi: int = 200) -> bytes:
    """One figure as PNG, SVG or PDF."""
    if fmt not in FIGURE_FORMATS:
        raise ValueError(f"unknown figure format {fmt!r}; expected one of {FIGURE_FORMATS}")
    buf = io.BytesIO()
    fig.savefig(buf, format=fmt, dpi=dpi, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    return buf.getvalue()


def figures_zip(b: Bundle, fmt: str = "png", *, dpi: int = 200) -> bytes:
    """Every figure in one archive, plus the stamp as a text file.

    The stamp travels as `README.txt` inside the archive because the figures
    themselves get separated from each other the moment they are dropped into
    slides -- and a figure separated from its POS provenance is the failure this
    project keeps guarding against.
    """
    figs = build_figures(b)
    try:
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            z.writestr("README.txt", _cover_text(b))
            z.writestr("case.json", b.case.to_json())
            for name, fig in figs.items():
                z.writestr(f"{name}.{fmt}", figure_bytes(fig, fmt, dpi=dpi))
        return buf.getvalue()
    finally:
        _close(figs)


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
    if b.case.note:
        lines += ["", f"Note: {b.case.note}"]
    if b.warnings:
        lines += ["", "Warnings carried from this session:"]
        lines += [f"  - {w}" for w in b.warnings]
    return "\n".join(lines)


def pdf_bytes(b: Bundle) -> bytes:
    """Cover page, then one figure per page.

    Uses matplotlib's own ``PdfPages`` rather than assembling with pypdf: fewer
    moving parts, and the vector content survives, which is the reason to choose
    PDF over PNG in a well proposal.
    """
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    figs = build_figures(b)
    buf = io.BytesIO()
    cover = None
    try:
        with PdfPages(buf) as pdf:
            cover, ax = F.new_figure(figsize=(11.7, 8.3), dark=b.case.dark) \
                if hasattr(F, "new_figure") else (None, None)
            if cover is not None:
                ax.axis("off")
                ax.text(0.0, 1.0, _cover_text(b), va="top", ha="left", fontsize=9,
                        family="monospace", wrap=True, transform=ax.transAxes)
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


def _close(figs: dict[str, object]) -> None:
    import matplotlib.pyplot as plt

    for fig in figs.values():
        plt.close(fig)
