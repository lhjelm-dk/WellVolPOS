"""The QC report that gates every analysis tab.

The point of a gate is that it stops things. A ``fail`` here blocks the rest of
the application rather than warning and continuing, because every number the
tool produces is a projection of the trial file and a misread file produces
confident nonsense.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..core.classes import check_area_pay_correlation
from ..core.reservoir import thickness_from_pay
from ..core.structure import AreaDepth
from .adapters.base import CANONICAL_FIELDS, TrialSet
from .failure import FailureReport, detect_failures
from .units import verdict as unit_verdict

LEVELS = ("pass", "warn", "fail")


@dataclass
class Check:
    name: str
    level: str
    message: str


@dataclass
class QCReport:
    checks: list[Check] = field(default_factory=list)
    failure: FailureReport | None = None
    area_depth: AreaDepth | None = None

    def add(self, name: str, level: str, message: str) -> None:
        assert level in LEVELS, level
        self.checks.append(Check(name, level, message))

    @property
    def blocked(self) -> bool:
        return any(c.level == "fail" for c in self.checks)

    @property
    def worst(self) -> str:
        for lvl in ("fail", "warn", "pass"):
            if any(c.level == lvl for c in self.checks):
                return lvl
        return "pass"

    def as_text(self) -> str:
        width = max((len(c.name) for c in self.checks), default=10)
        lines = [f"{c.level.upper():5s}  {c.name:<{width}}  {c.message}" for c in self.checks]
        return "\n".join(lines)


def run_qc(ts: TrialSet, *, min_trials_warn: int = 10_000) -> QCReport:
    rep = QCReport()

    # ---- required fields and trial count
    for fname, (unit, required) in CANONICAL_FIELDS.items():
        if required and not ts.has(fname):
            rep.add("required fields", "fail", f"Missing required field {fname!r} ({unit}).")
    if not any(c.name == "required fields" for c in rep.checks):
        present = [f for f in CANONICAL_FIELDS if ts.has(f)]
        rep.add("required fields", "pass", "Mapped: " + ", ".join(present) + ".")

    n = ts.n_trials
    if n < 1000:
        rep.add("trial count", "fail", f"{n:,} trials is too few for stable conditional statistics.")
    elif n < min_trials_warn:
        rep.add(
            "trial count", "warn",
            f"{n:,} trials. Conditional percentiles in narrow depth windows will be noisy; "
            f"50 000+ is recommended (Rose used 20 000, Milkov 5 000).",
        )
    else:
        rep.add("trial count", "pass", f"{n:,} trials.")

    # ---- sanity of the values themselves
    res, contact = ts.col("resource"), ts.col("contact")
    if not np.isfinite(res).all() or not np.isfinite(contact).all():
        rep.add("finite values", "fail", "Resource or contact contains non-finite values.")
    else:
        rep.add(
            "value ranges", "pass",
            f"Contact {contact.min():.1f}–{contact.max():.1f} m; "
            f"resource {res.min():.3f}–{res.max():.3f} MMboe.",
        )
    if (res < 0).any():
        rep.add("negative resource", "fail", f"{int((res < 0).sum()):,} trials have negative resource.")

    # ---- the volumetric identity, where it can be checked
    if ts.has("hc_grv") and ts.has("area") and ts.has("gross_pay"):
        a, g, v = ts.col("area"), ts.col("gross_pay"), ts.col("hc_grv")
        m = (v > 0) & (a > 0) & (g > 0)
        if m.sum() > 10:
            ratio = v[m] / (a[m] * g[m])
            spread = float(np.nanmax(ratio) - np.nanmin(ratio))
            # 1e-3 rather than machine epsilon: a CSV written at six significant
            # digits carries its own rounding, and that must not be mistaken for
            # the model departing from an area x thickness formulation.
            if spread < 1e-3:
                rep.add(
                    "GRV identity", "pass",
                    f"HC GRV = area x gross pay to within the file's stored precision "
                    f"(mean ratio {float(np.nanmean(ratio)):.6f}, spread {spread:.2e}). "
                    f"Area x thickness model confirmed.",
                )
            else:
                rep.add(
                    "GRV identity", "warn",
                    f"HC GRV / (area x gross pay) varies by {spread:.3g}. The model is not a plain "
                    f"area x thickness one; the per-trial split rests on a weaker assumption.",
                )

    # ---- units: reject, never convert (design plan 6.4 and 8)
    # Before anything numerical, because every check below reads the numbers as if
    # they were MMboe / m / km2, and a file in feet passes all of them.
    lvl, msg = unit_verdict(ts)
    rep.add("units", lvl, msg)

    # ---- failure-case detection
    rep.failure = detect_failures(ts)
    lvl = {"chance_failure": "pass", "none": "pass", "geometric": "warn", "ambiguous": "warn"}[
        rep.failure.verdict
    ]
    rep.add("failure cases", lvl, rep.failure.summary())

    # ---- area-depth curve
    if ts.has("area"):
        try:
            ad = AreaDepth.from_trials(ts.col("contact"), ts.col("area"))
            rep.area_depth = ad
            lvl, msg = ad.quality()
            rep.add("area-depth fit", lvl, msg)
        except ValueError as exc:
            rep.add("area-depth fit", "warn", str(exc))
    else:
        rep.add(
            "area-depth fit", "warn",
            "Productive area not exported, so A(z) cannot be recovered. The reference grouping "
            "engine still works; the proven/possible split does not.",
        )

    # ---- assumption behind the split
    lvl, msg, _ = check_area_pay_correlation(ts)
    rep.add("area / net-pay correlation", lvl, msg)

    # ---- reservoir thickness, recovered from pay by inverting the wedge
    if rep.area_depth is not None and (ts.has("hc_grv") or ts.has("gross_pay")):
        try:
            tfp = thickness_from_pay(ts, rep.area_depth)
        except ValueError:
            tfp = None
        if tfp is not None and tfp.n_resolved:
            lvl = "warn" if tfp.n_inconsistent else "pass"
            msg = tfp.message()
            if ts.has("thickness"):
                # Independent check: the inversion and the simulator's own
                # column are two routes to one number, so a disagreement means
                # the wedge geometry assumed here is not the one GeoX used.
                col = ts.col("thickness")[tfp.resolved]
                rec = tfp.thickness[tfp.resolved]
                bias = float(np.mean(rec - col))
                # A *deterministic* thickness has no variance to correlate against,
                # and prospect B is exactly that -- 50 m in every trial. Asking
                # numpy for a correlation there divides by zero, warns, and returns
                # NaN, so the agreement rests on the mean difference alone and the
                # message says which test was available rather than printing "nan".
                varies = float(np.ptp(col)) > 1e-9 and float(np.ptp(rec)) > 1e-9
                r = float(np.corrcoef(rec, col)[0, 1]) if (varies and rec.size > 2) else float("nan")
                agrees = abs(bias) < 1.0 and (not np.isfinite(r) or r > 0.99)
                how = (
                    f"mean difference {bias:+.2f} m, r = {r:.4f}" if np.isfinite(r) else
                    f"mean difference {bias:+.2f} m (the column is a constant "
                    f"{float(np.mean(col)):.1f} m, so there is no correlation to take)"
                )
                msg += (
                    f" Against the export's own reservoir-thickness column: {how}"
                    + (" — the wedge inversion and the simulator agree."
                       if agrees else
                       " — they DISAGREE, so the wedge geometry assumed here is not GeoX's.")
                )
                if not agrees:
                    lvl = "warn"
            rep.add("reservoir thickness from pay", lvl, msg)

    # ---- contact distribution spikes (a sentinel is expected; others are not)
    uniq, counts = np.unique(contact, return_counts=True)
    if counts.size:
        top = int(counts.max())
        if top > 0.02 * n:
            spike = float(uniq[int(counts.argmax())])
            expected = rep.failure and rep.failure.sentinel_contact == spike
            rep.add(
                "contact spikes",
                "pass" if expected else "warn",
                f"{top:,} trials ({top / n:.2%}) share contact {spike:.2f} m"
                + (" — the failure sentinel, as expected." if expected else
                   " — check whether the contact distribution is truncated there."),
            )

    if ts.notes:
        rep.add("import notes", "pass", " ".join(ts.notes))
    return rep
