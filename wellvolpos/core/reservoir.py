"""Reservoir thickness, recovered from pay.

**Pay thickness is not reservoir thickness**, and the difference is geometric
rather than a matter of definition-splitting. Reservoir thickness is a property
of the rock: top reservoir to base reservoir, present throughout the closure
whether or not anything is charged. Pay is a property of the *accumulation*: the
interval from top reservoir down to the hydrocarbon-water contact, capped by the
base where the column is thicker than the layer.

With a flat contact and a dipping layer the charged interval is a **wedge**. At
the crest the full reservoir thickness may be charged; down-dip the top surface
deepens and the interval thins; at the productive-area edge, where top reservoir
meets the contact, it is zero. So an area-averaged gross pay is an average over
that wedge -- always less than the reservoir thickness, and **varying with the
contact depth**. It is not a rock property at all, which is why shifting A(z)
down by it does not give a base-reservoir surface.

The inversion rests on one identity. For reservoir thickness ``T`` and a contact
at ``z_c``, the hydrocarbon-bearing gross rock volume is::

    GRV(z_c, T) = integral of A(z) dz, from (z_c - T) to z_c

The crestal region where the full ``T`` is charged contributes ``T * A(z_c - T)``
and the flank wedge contributes the remainder; integrating by parts, and using
``A(apex) = 0``, the two collapse into that single integral. Sanity: as ``T``
grows the lower limit reaches the apex and GRV becomes the whole closure volume
above the contact; as ``T`` goes to zero, so does GRV.

Because the right-hand side increases monotonically in ``T``, and because the
trial file carries GRV directly, ``T`` is **uniquely recoverable per trial** --
no fitting and no assumption beyond the wedge geometry:

    find z_lo such that  volume_above(z_c) - volume_above(z_lo) = GRV,
    then  T = z_c - z_lo

A corollary worth knowing, because it is checkable: the identity makes the ratio
of average gross pay to reservoir thickness equal the mean of A over the top
``T`` metres divided by A at the contact. On the reference file that ratio sits
at 0.825, which is a direct statement about how fast area grows over the last
~45 m of the structure rather than a coincidence.

**Always vertical.** Lars confirmed (2026-08-10) that the thickness GeoX reports
is true *vertical* thickness, not true stratigraphic thickness. So a base
reservoir is ``top + T`` with no ``1/cos(dip)`` correction, at any dip, and the
inversion above needs no dip term either.

**Gross, not net.** The inversion consumes a *gross* rock volume, so it returns a
*gross* reservoir thickness. In the reference export ``HC bearing gross rock
volume = Productive area x Average gross pay`` holds to stored precision in every
success trial, which makes that pay column a gross thickness and leaves
net/gross as the separate column it is. Where a file instead folds net/gross into
its pay, divide it out before calling this.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..io.adapters.base import TrialSet
from .structure import AreaDepth


@dataclass
class ThicknessFromPay:
    """Per-trial reservoir thickness recovered from the wedge geometry.

    ``thickness`` is NaN where the trial could not resolve one, for one of the
    two reasons below; ``resolved`` marks the ones that could.

    ``n_full_to_base``
        The recovered lower limit reached the apex: the closure is charged
        top-to-base everywhere, so the thickness is only bounded *below* by the
        column height and the trial cannot pin it down. Expected at deep
        contacts, and not a defect.
    ``n_inconsistent``
        The trial's HC-bearing GRV exceeds the entire closure volume above its
        own contact, which no thickness can produce.

        Read this as an **upper bound** on export problems, not a count of them.
        The closure volume is measured from the apex, and the apex is an
        extrapolation of A(z) to zero area -- so an apex estimated too deep makes
        the modelled closure too small and flags trials that are in fact
        consistent. On a synthetic cone whose apex is known, the true apex flags
        nothing while the derived apex (34 m too deep there) flags about 3 % of
        trials; ``tests/test_synthetic.py`` pins that. It is zero on the reference
        file, so nothing here is affected, but a closure with a thinner shallow
        tail would show it.
    """

    thickness: np.ndarray
    resolved: np.ndarray
    n_full_to_base: int
    n_inconsistent: int
    apex: float

    @property
    def n_resolved(self) -> int:
        return int(self.resolved.sum())

    def summary(self) -> dict[str, float]:
        t = self.thickness[self.resolved]
        if t.size == 0:
            return {k: float("nan") for k in ("p90", "p50", "mean", "p10", "min", "max")}
        return {
            "p90": float(np.percentile(t, 10.0)),
            "p50": float(np.percentile(t, 50.0)),
            "mean": float(t.mean()),
            "p10": float(np.percentile(t, 90.0)),
            "min": float(t.min()),
            "max": float(t.max()),
        }

    def message(self) -> str:
        s = self.summary()
        if self.n_resolved == 0:
            why = []
            if self.n_inconsistent:
                why.append(
                    f"{self.n_inconsistent:,} carry more HC gross rock volume than the closure "
                    f"holds above their own contact"
                )
            if self.n_full_to_base:
                why.append(f"{self.n_full_to_base:,} are charged to base")
            return (
                "No trial could resolve a reservoir thickness from its pay"
                + (" — " + "; ".join(why) + "." if why else ".")
            )
        parts = [
            f"Reservoir thickness back-calculated from pay on {self.n_resolved:,} trials: "
            f"P90 {s['p90']:.1f} · P50 {s['p50']:.1f} · mean {s['mean']:.1f} · "
            f"P10 {s['p10']:.1f} m."
        ]
        if self.n_full_to_base:
            parts.append(
                f"{self.n_full_to_base:,} trials are charged to base, so their thickness is "
                f"only bounded below and is excluded."
            )
        if self.n_inconsistent:
            parts.append(
                f"⚠ {self.n_inconsistent:,} trials carry more HC gross rock volume than the "
                f"closure holds above their own contact — check the export."
            )
        return " ".join(parts)


def thickness_from_pay(
    ts: TrialSet, ad: AreaDepth, *, apex: float | None = None, tol: float = 1e-9
) -> ThicknessFromPay:
    """Recover gross reservoir thickness per trial by inverting the wedge.

    Uses the trial's own HC-bearing gross rock volume where the export carries
    it, and otherwise ``productive area x average gross pay``, which is the same
    quantity -- the identity is verified by the QC gate.
    """
    res = ts.col("resource")
    contact = ts.col("contact")
    success = res > 0.0
    apex_v = float(ad.apex_estimate() if apex is None else apex)

    if ts.has("hc_grv"):
        grv = np.asarray(ts.col("hc_grv"), dtype=float)
    elif ts.has("area") and ts.has("gross_pay"):
        grv = np.asarray(ts.col("area"), dtype=float) * np.asarray(ts.col("gross_pay"), dtype=float)
    else:
        raise ValueError(
            "need HC-bearing gross rock volume, or productive area and average gross pay, "
            "to invert the wedge for reservoir thickness"
        )

    thickness = np.full(contact.shape, np.nan)
    resolved = np.zeros(contact.shape, dtype=bool)
    n_full = n_bad = 0

    above = ad.volume_above(contact, apex_v)
    remainder = above - grv          # closure volume left below (z_c - T)

    for i in np.flatnonzero(success):
        if not np.isfinite(remainder[i]) or grv[i] <= 0.0:
            continue
        if remainder[i] < -tol:
            n_bad += 1                                   # more pay than the closure holds
            continue
        if remainder[i] <= tol:
            n_full += 1                                  # charged to base; T unbounded above
            continue
        z_lo = float(ad.depth_for_volume(remainder[i], apex_v))
        t = float(contact[i]) - z_lo
        if t > 0.0:
            thickness[i] = t
            resolved[i] = True

    return ThicknessFromPay(
        thickness=thickness, resolved=resolved,
        n_full_to_base=n_full, n_inconsistent=n_bad, apex=apex_v,
    )
