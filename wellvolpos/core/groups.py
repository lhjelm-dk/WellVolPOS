"""The reference engine: whole-trial grouping, after Schneider et al. (2023).

Step 2 of the published workflow assigns each *whole trial* to one of two groups
by comparing its sampled area against the area at the well location:

    "If the sampled area is greater than or equal to the AAWL, then place that
     EUR value in the downdip group. This assumes a discovery at the well
     location, which includes the discovered updip volume."

So the "downdip" distribution is the total accumulation given a discovery, and
the "updip" distribution is the total accumulation in the dry-hole trials. This
is what the source workbook already does, and it is the reference against which
the finer proven/possible decomposition in :mod:`wellvolpos.core.classes` is
compared -- neither is "the correct one".

Grouping is expressed in terms of contact depth rather than area because the two
are equivalent through A(z) and the contact is the quantity every export carries.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..io.adapters.base import TrialSet


@dataclass
class Groups:
    """Boolean masks over the trial set, plus the counts the UI reports."""

    success: np.ndarray        # the prospect contains hydrocarbons at all
    discovery: np.ndarray      # HC present at the well's entry depth
    dry_with_attic: np.ndarray # HC present but entirely up-dip of the well
    chance_failure: np.ndarray # no hydrocarbons anywhere
    contact_seen: np.ndarray   # discovery AND the contact lies above the exit
    hc_to_exit: np.ndarray     # discovery AND hydrocarbons continue past the exit

    @property
    def n(self) -> int:
        return int(self.success.size)

    def shares(self) -> dict[str, float]:
        return {
            "chance_failure": float(self.chance_failure.mean()),
            "dry_with_attic": float(self.dry_with_attic.mean()),
            "contact_seen": float(self.contact_seen.mean()),
            "hc_to_exit": float(self.hc_to_exit.mean()),
            "p_well": float(self.discovery.mean()),
        }


def group_trials(ts: TrialSet, z_entry: float, z_exit: float | None = None) -> Groups:
    """Partition the trials for a well entering reservoir at ``z_entry``.

    ``z_exit`` is the depth at which the well leaves the reservoir. If omitted
    the contact-seen / HC-to-exit split is not made and both are reported as the
    discovery set.
    """
    res = ts.col("resource")
    contact = ts.col("contact")
    success = res > 0.0
    discovery = success & (contact > z_entry)
    dry = success & ~discovery
    failure = ~success
    if z_exit is None:
        seen = discovery.copy()
        past = np.zeros_like(discovery)
    else:
        seen = discovery & (contact <= z_exit)
        past = discovery & (contact > z_exit)
    return Groups(success, discovery, dry, failure, seen, past)


def group_summary(ts: TrialSet, groups: Groups) -> dict[str, dict[str, float]]:
    """Percentiles and mean for each reference group.

    Percentiles are reported in petroleum orientation: P90 is the value exceeded
    by 90 % of the group, P10 the value exceeded by 10 %.
    """
    res = ts.col("resource")
    out: dict[str, dict[str, float]] = {}
    for label, mask in (
        ("prospect", np.ones_like(groups.success, dtype=bool)),
        ("discovery", groups.discovery),
        ("attic_dry_hole", groups.dry_with_attic),
        ("attic_incl_failures", groups.dry_with_attic | groups.chance_failure),
    ):
        v = res[mask]
        if v.size == 0:
            out[label] = {k: float("nan") for k in ("n", "p90", "p50", "mean", "p10")}
            continue
        out[label] = {
            "n": float(v.size),
            "p90": float(np.percentile(v, 10)),
            "p50": float(np.percentile(v, 50)),
            "mean": float(v.mean()),
            "p10": float(np.percentile(v, 90)),
        }
    return out
