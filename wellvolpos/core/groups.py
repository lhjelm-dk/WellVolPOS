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
        """The trial file's *own* outcome shares -- 1 - this mask's chance_failure
        share is POS_trials, not the entered POS_prospect. Use
        :meth:`risked_shares` wherever the number is shown next to a P_well
        that came from an entered chance table, or the two will disagree
        exactly the way CLAUDE.md's "one idea" warns against.
        """
        return {
            "chance_failure": float(self.chance_failure.mean()),
            "dry_with_attic": float(self.dry_with_attic.mean()),
            "contact_seen": float(self.contact_seen.mean()),
            "hc_to_exit": float(self.hc_to_exit.mean()),
            "p_well": float(self.discovery.mean()),
        }

    def risked_shares(self, pos_prospect: float, p_well: float) -> dict[str, float]:
        """The four outcome shares, risked onto ``pos_prospect`` / ``p_well``.

        Discovery mass is pinned to ``p_well`` and split into contact-seen /
        HC-to-exit using this mask's own *proportions* (not its raw counts),
        so the result partitions to 1.0 and cannot disagree with a P_well
        computed from the same ``pos_prospect`` -- the fix behind A2's outcome
        tree, generalised so the same number can be shown anywhere else in the
        app without re-deriving it.

        Dry-with-attic absorbs the residual success mass, ``pos_prospect -
        p_well``. That is also exactly what the Rose reference contour means: at
        or up-dip of the P90-area contour ``r_location`` caps at 1.0, so the
        residual goes to zero and the dry-with-attic outcome disappears -- a
        well that high on the structure finds hydrocarbons whenever the prospect
        has any.

        Preconditions, unchecked because nothing in the app can violate them:
        ``0 <= p_well <= pos_prospect <= 1``. ``r_location <= 1`` always, so
        ``p_well`` cannot exceed ``pos_prospect``; a ``p_well`` larger than
        ``pos_prospect`` would return a negative dry-with-attic share.
        """
        n_disc = int(self.discovery.sum())
        seen_frac = float(self.contact_seen.sum()) / n_disc if n_disc else 0.0
        return {
            "chance_failure": 1.0 - pos_prospect,
            "dry_with_attic": pos_prospect - p_well,
            "contact_seen": p_well * seen_frac,
            "hc_to_exit": p_well * (1.0 - seen_frac),
            "p_well": p_well,
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


def boundary_ties(ts: "TrialSet", z_entry: float, tol_m: float = 0.5) -> tuple[int, float]:
    """Success trials whose contact sits within ``tol_m`` of the well entry.

    :func:`group_trials` calls a discovery ``contact > z_entry``, **strictly**, so a
    trial whose contact lands exactly on the entry is a dry hole. That is the right
    reading -- a contact at the entry is zero column at the well -- but it is
    invisible, and an invisible tie is where a boundary rule goes wrong quietly. One
    prospect-B trial sits exactly on 2205.0 m and it was enough to invert a whole
    band's percentiles in 3.12 before the band rule was made to match the engine's.

    Returns ``(count, fraction_of_success_trials)`` so a caller can say how much of
    the answer is sitting on the knife edge. Reported rather than acted on: nothing
    changes because of it, but a reader who moves the entry by a metre and sees a
    number move deserves to know why.
    """
    import numpy as _np

    res = _np.asarray(ts.col("resource"), dtype=float)
    contact = _np.asarray(ts.col("contact"), dtype=float)
    success = res > 0.0
    n_succ = int(success.sum())
    if not n_succ:
        return 0, 0.0
    near = success & (_np.abs(contact - float(z_entry)) <= float(tol_m))
    n = int(near.sum())
    return n, n / n_succ


def group_summary(ts: TrialSet, groups: Groups) -> dict[str, dict[str, float]]:
    """Percentiles and mean for each reference group.

    Percentiles are reported in petroleum orientation: P90 is the value exceeded
    by 90 % of the group, P10 the value exceeded by 10 %.

    **``prospect`` and ``prospect_success`` are different statistics and the
    difference matters.** ``prospect`` spans every trial including the chance
    failures, so on a file that has them it is already unconditional -- risking it
    again double-counts the geological chance, which is this codebase's recurring
    bug. ``prospect_success`` is the success case, and it is the one an expected
    volume multiplies. On a file with no zero-volume trials the two coincide, which
    is exactly why the error hid on one demo prospect and not the other.
    """
    res = ts.col("resource")
    out: dict[str, dict[str, float]] = {}
    for label, mask in (
        # ``prospect`` is over **every** trial, chance failures included, so on a file
        # that carries them it is already an *unconditional* statistic:
        # 13.5612 x 0.7605 = 10.3133 on the reference data, exactly. That is the right
        # thing for a KPI strip that says "over every trial" and the wrong thing to
        # multiply by a chance -- which is what "Expected prospect volume" was doing,
        # risking it a second time and reporting 7.84 where the answer is 10.31.
        ("prospect", np.ones_like(groups.success, dtype=bool)),
        # ...so the **conditional** prospect statistic is carried separately and
        # named. It is the success case: the distribution the percentiles belong to,
        # and the only prospect mean it is correct to multiply by POS_prospect.
        ("prospect_success", groups.success),
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
