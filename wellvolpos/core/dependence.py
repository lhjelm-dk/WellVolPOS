"""Which quantities move with the well's **exit**, and which only with its entry.

Lars, 2026-08-14: *"investigate what volume cases are dependent on exit depth and make
sure that well options plotted in the same plot are plotted with their relevant volume
case curves."*

Measured rather than reasoned about -- hold the entry at 2205 m on prospect B and move
only the exit:

    exit    proven   below LKH   well assoc   attic   n discovery
    2225 m  135.39      52.22       171.69    76.38        4701
    2255 m  152.54      32.76       171.69    76.38        4701
    2305 m  166.32      19.30       171.69    76.38        4701
    2405 m  171.69        —         171.69    76.38        4701

**Exactly two volume classes move.** The reason is structural rather than empirical:
``group_trials`` calls a discovery ``contact > z_entry``, so the *populations* --
discovery, dry-with-attic, and therefore ``r_location`` and ``P_well`` -- depend on the
entry alone. The exit enters only through ``LKH = min(contact, z_exit)``, which is
where ``split_trials`` cuts the accumulation. So the exit moves the **boundary between
proven and unproven** and nothing else; their sum, the well-associated volume, is
fixed.

That has a direct consequence for drawing several candidates on one figure. A curve
built from an entry-only quantity is **the same curve for every candidate** and drawing
it once is correct. A curve built from proven or from the unproven volume below LKH is
a *different curve per entry-to-exit spacing*, and drawing one of them at the selected
well's spacing quietly answers the wrong question for the others.

:data:`EXIT_DEPENDENT` and :data:`ENTRY_ONLY` name which is which, so a figure can say
so and a test can check that it does.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: Quantities whose value changes when only the reservoir **exit** moves. Every one of
#: them is downstream of ``LKH = min(contact, z_exit)``.
EXIT_DEPENDENT = (
    "proven",
    "below_lkh",
    "p_proven_exceeds_mefs",
    "p_below_lkh_exceeds_mefs",
    "p_well_exits_in_hc",
    "hc_to_exit",
)

#: Quantities that depend on the **entry** alone -- the grouping is
#: ``contact > z_entry`` and the exit never enters. One curve serves every candidate.
ENTRY_ONLY = (
    "r_location",
    "p_well",
    "discovery_total",
    "attic",
    "at_well",
    "n_discovery",
    "n_dry",
    "uncertainty_reduction",
)


@dataclass(frozen=True)
class DeviationCheck:
    """Is a candidate's entry-to-exit spacing consistent with a vertical well?

    The reservoir has a true vertical thickness, recovered per trial from the pay
    (``core.reservoir.thickness_from_pay``). A **vertical** well entering at the top
    and leaving at the base sees exactly that thickness, so its entry-to-exit spacing
    is *not a free choice* -- it is the reservoir.

    A larger spacing is still perfectly possible, and this is the point Lars raised:
    a **deviated or horizontal well drilled down-dip stays inside the layer over a
    greater vertical range**, so it tests a taller column than the formation is thick.
    A smaller spacing means the well stopped inside the reservoir -- TD'd early, or
    was cased off.

    None of the three is wrong. What is wrong is choosing a spacing without knowing
    which one you have chosen, because a 150 m spacing on a 50 m reservoir is a
    commitment to a deviated well and the volumes are computed as though you had
    drilled one.
    """

    gap: float
    thickness_p50: float
    thickness_p90: float
    thickness_p10: float
    n_resolved: int

    @property
    def ratio(self) -> float:
        """Spacing divided by the median reservoir thickness."""
        return self.gap / self.thickness_p50 if self.thickness_p50 > 0 else float("nan")

    @property
    def verdict(self) -> str:
        """``"vertical"``, ``"deviated"`` or ``"partial"``."""
        if not np.isfinite(self.ratio):
            return "unknown"
        if self.ratio > 1.05:
            return "deviated"
        if self.ratio < 0.95:
            return "partial"
        return "vertical"

    def message(self) -> str:
        if self.verdict == "unknown":
            return ("No reservoir thickness could be recovered from these trials, so "
                    "the entry-to-exit spacing cannot be checked against it.")
        t = (f"median reservoir thickness {self.thickness_p50:,.0f} m "
             f"(P90 {self.thickness_p90:,.0f} – P10 {self.thickness_p10:,.0f})")
        if self.verdict == "vertical":
            return (f"**Consistent with a vertical well**: the {self.gap:,.0f} m spacing "
                    f"matches the {t}.")
        if self.verdict == "deviated":
            return (f"**This spacing requires a deviated well.** {self.gap:,.0f} m of "
                    f"vertical section against a {t} — a factor of {self.ratio:.1f}. A "
                    f"vertical well cannot see that much column; one drilled down-dip "
                    f"inside the layer can. The volumes are computed as though you had "
                    f"drilled it.")
        return (f"**The well stops inside the reservoir**: {self.gap:,.0f} m of the "
                f"{t} — {self.ratio:.0%} of the section. Everything below the exit is "
                f"counted as unproven, which is right, but check that an early TD is "
                f"what you meant.")


def check_deviation(thickness: np.ndarray, gap: float) -> DeviationCheck:
    """Compare a candidate's entry-to-exit spacing with the reservoir it must sit in."""
    t = np.asarray(thickness, dtype=float)
    t = t[np.isfinite(t) & (t > 0)]
    if not t.size:
        return DeviationCheck(float(gap), float("nan"), float("nan"), float("nan"), 0)
    return DeviationCheck(
        gap=float(gap),
        thickness_p50=float(np.percentile(t, 50)),
        thickness_p90=float(np.percentile(t, 10)),
        thickness_p10=float(np.percentile(t, 90)),
        n_resolved=int(t.size),
    )
