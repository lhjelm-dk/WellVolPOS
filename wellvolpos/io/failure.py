"""Detect whether a trial export already contains chance-failure realisations.

This is the single most consequential question at import time, because getting
it wrong double-counts geological risk.

The source workbook multiplies ``1 - PERCENTRANK(all contacts, entry)`` -- which
already includes the failure trials -- by a separately entered prospect POS. In
the shipped file that POS is 1.0, so the answer is right by accident. Any real
chance table would have made it roughly 40 % too low.

The detector distinguishes two ways a trial can have zero volume:

``chance_failure``
    The simulator applied the geological chance factors and wrote the failures
    into the trial set. Signature: every zero-volume trial shares *one* contact
    value, all hydrocarbon quantities are collapsed to exactly zero, the
    non-hydrocarbon parameters are still sampled, and there is a clean gap
    between the sentinel and the shallowest real contact. POS is then readable
    directly as ``1 - n_zero / n_total``.

``geometric``
    The contact distribution itself puts mass above the crest. Signature: the
    zero-volume contacts are *spread* rather than identical. POS is not readable
    from the file and the chance table must be applied separately.

A continuous distribution clipped at a lower bound piles mass *at* the bound and
also scatters observations just above it. Many hundreds of hits on a single
value followed by a wide empty gap is not something it can produce, which is
what makes the distinction decidable rather than a matter of taste.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .adapters.base import TrialSet

VERDICTS = ("chance_failure", "geometric", "none", "ambiguous")


@dataclass
class FailureReport:
    n_total: int
    n_zero: int
    frac_zero: float
    verdict: str
    pos_trials: float | None
    sentinel_contact: float | None
    n_unique_zero_contacts: int
    gap_to_shallowest_success: float | None
    shallowest_success_contact: float | None
    collapsed_fields: list[str] = field(default_factory=list)
    still_sampled_fields: list[str] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)

    @property
    def has_failures(self) -> bool:
        return self.n_zero > 0

    def summary(self) -> str:
        if self.verdict == "none":
            return f"No zero-volume trials in {self.n_total:,} — the export looks success-case only."
        if self.verdict == "chance_failure":
            return (
                f"Detected {self.n_zero:,} zero-volume trials ({self.frac_zero:.2%}) of "
                f"{self.n_total:,}, all stamped with a single sentinel contact at "
                f"{self.sentinel_contact:.2f} m and all hydrocarbon quantities collapsed. "
                f"This is chance-failure coding. POS implied by the trials = {self.pos_trials:.4f}."
            )
        if self.verdict == "geometric":
            return (
                f"Detected {self.n_zero:,} zero-volume trials ({self.frac_zero:.2%}) whose contacts "
                f"are spread over {self.n_unique_zero_contacts:,} distinct values. That looks "
                f"geometric (contact above crest), not chance failure — POS is not readable "
                f"from this file."
            )
        return (
            f"Detected {self.n_zero:,} zero-volume trials ({self.frac_zero:.2%}) but the evidence "
            f"is mixed. Please choose the convention explicitly."
        )


# fields that a chance failure should zero out, vs those it should leave alone
_HC_FIELDS = ("resource", "area", "gross_pay", "hc_grv", "hc_pv")
_NON_HC_FIELDS = ("crest", "spill", "net_gross", "porosity", "thickness")


def detect_failures(ts: TrialSet, *, min_gap_m: float = 10.0) -> FailureReport:
    res = ts.col("resource")
    contact = ts.col("contact")
    n = len(res)
    zero = res <= 0.0
    n_zero = int(zero.sum())

    if n_zero == 0:
        return FailureReport(
            n_total=n, n_zero=0, frac_zero=0.0, verdict="none", pos_trials=None,
            sentinel_contact=None, n_unique_zero_contacts=0,
            gap_to_shallowest_success=None, shallowest_success_contact=None,
            evidence=["No zero-volume trials."],
        )

    zc = contact[zero]
    uniq = np.unique(zc[np.isfinite(zc)])
    n_uniq = int(uniq.size)
    sentinel = float(uniq[0]) if n_uniq == 1 else None

    succ_contacts = contact[~zero]
    shallowest = float(np.nanmin(succ_contacts)) if succ_contacts.size else None
    gap = (shallowest - float(np.nanmax(zc))) if (shallowest is not None and zc.size) else None

    collapsed, still_sampled = [], []
    for f in _HC_FIELDS:
        if ts.has(f):
            v = ts.col(f)[zero]
            if np.nanmax(np.abs(v)) == 0.0:
                collapsed.append(f)
    for f in _NON_HC_FIELDS:
        if ts.has(f):
            v = ts.col(f)[zero]
            if len(np.unique(v[np.isfinite(v)])) > max(10, n_zero // 100):
                still_sampled.append(f)

    ev: list[str] = []
    ev.append(f"{n_zero:,} of {n:,} trials ({n_zero / n:.2%}) have zero resource.")
    if n_uniq == 1:
        ev.append(f"All of them share one contact value: {sentinel:.2f} m.")
    else:
        ev.append(f"Their contacts take {n_uniq:,} distinct values (min {uniq.min():.2f}, max {uniq.max():.2f}).")
    if collapsed:
        ev.append("Collapsed to exactly zero: " + ", ".join(collapsed) + ".")
    if still_sampled:
        ev.append("Still fully sampled: " + ", ".join(still_sampled) + ".")
    if gap is not None:
        ev.append(
            f"Shallowest success contact is {shallowest:.2f} m, leaving a {gap:.1f} m gap "
            f"containing no trials."
        )

    single_value = n_uniq == 1
    clean_gap = gap is not None and gap > min_gap_m
    hc_collapsed = len(collapsed) >= 2 or (len(collapsed) == 1 and not ts.has("area"))

    if single_value and clean_gap and hc_collapsed:
        verdict = "chance_failure"
        ev.append(
            "A continuous distribution clipped at a bound would scatter values just above "
            "that bound; a single value followed by a wide empty gap cannot arise that way."
        )
    elif n_uniq > 5 and (gap is None or gap <= min_gap_m):
        verdict = "geometric"
    else:
        verdict = "ambiguous"

    pos = 1.0 - n_zero / n if verdict == "chance_failure" else None
    return FailureReport(
        n_total=n, n_zero=n_zero, frac_zero=n_zero / n, verdict=verdict, pos_trials=pos,
        sentinel_contact=sentinel, n_unique_zero_contacts=n_uniq,
        gap_to_shallowest_success=gap, shallowest_success_contact=shallowest,
        collapsed_fields=collapsed, still_sampled_fields=still_sampled, evidence=ev,
    )
