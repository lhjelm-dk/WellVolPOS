"""Two ways to pick a location that the expectation curve cannot express.

3.10 maximises ``P_well x mean volume``. That is the **risk-neutral** answer: it treats
a 10 % chance of 500 MMboe as worth exactly a 50 % chance of 100. No exploration company
behaves that way, and the two criteria here are the standard corrections.

**Certainty equivalent** (Cozzolino 1977, and every decision-analysis course since)
replaces the expectation with the value a risk-averse party would accept for certain
instead of the gamble. Under exponential utility with risk tolerance ``rho``:

.. math::

    CE = -\\rho \\ln\\big( p \\cdot E[e^{-V/\\rho}] + (1-p) \\big)

where ``p`` is the chance of the discovery happening at all and ``V`` the success-case
volume. Two properties make it the right shape: ``CE -> p x E[V]`` as ``rho -> inf``
(risk neutrality is the limiting case, so nothing is lost), and ``CE`` falls as ``rho``
falls, penalising exactly the low-chance / high-volume tail that a plain expectation
rewards. **The optimum moves up-dip** as a result.

**Chance-constrained** answers the question a mandate is usually written in: *"the best
odds I can get, subject to being confident it is commercial if it works."* Maximise
``P_well`` subject to ``P(volume > MEFS | discovery) >= q``. Both curves are monotone in
opposite directions -- chance falls down-dip, conditional commerciality rises -- so the
feasible set is a half-line and the answer is its shallow end.

**Both are applied to volumes, not to money, and that is a real limitation.** Risk
tolerance is properly a monetary quantity: it is about the size of loss a balance sheet
can absorb, and a dry hole's cost does not appear anywhere in this tool. Using volume as
the argument makes ``rho`` a *volume* people have to calibrate by feel. It still ranks
locations correctly under a fixed well cost, which is the case a single prospect is
usually in -- but do not carry a certainty equivalent in MMboe into an economic model
as though it were an expected value.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: Risk tolerance is scale-dependent, so there is no universal default. This is a
#: *fraction of the success-case mean*: at 1.0 the party is willing to risk about one
#: mean discovery, which is mildly risk-averse and a reasonable place to start an
#: argument. Smaller is more averse.
DEFAULT_RISK_FRACTION = 1.0

#: The conditional commerciality a chance-constrained search defaults to asking for.
DEFAULT_CONFIDENCE = 0.90


def certainty_equivalent(values, chance: float, rho: float) -> float:
    """CE of "with probability ``chance``, draw from ``values``; otherwise nothing".

    ``rho`` is the risk tolerance in the same units as ``values``. Returns the same
    units. ``rho <= 0`` is meaningless and raises rather than silently returning the
    expectation -- a zero tolerance would say every gamble is worth nothing, and a
    caller that reached that state has a bug worth seeing.
    """
    if not np.isfinite(rho) or rho <= 0:
        raise ValueError(f"risk tolerance must be positive and finite; got {rho!r}")
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0 or not np.isfinite(chance):
        return float("nan")
    p = float(np.clip(chance, 0.0, 1.0))
    # exp(-V/rho) underflows to 0 for a large volume against a small tolerance, which
    # is the correct limit -- that outcome contributes nothing a risk-averse party
    # would pay for -- so no guard is needed beyond keeping the log's argument positive.
    m = float(np.mean(np.exp(-v / rho)))
    inner = p * m + (1.0 - p)
    if inner <= 0.0:
        return float("nan")
    return float(-rho * np.log(inner))


@dataclass(frozen=True)
class RiskAdjusted:
    """The certainty-equivalent curve against depth, and where it peaks."""

    z: np.ndarray
    ce: np.ndarray
    rho: float
    #: Index of the maximum, or ``None`` when nothing is finite.
    best: int | None

    @property
    def best_depth(self) -> float:
        return float(self.z[self.best]) if self.best is not None else float("nan")


def ce_curve(ts, vsweep, *, rho: float, min_support: int = 30) -> RiskAdjusted:
    """Certainty equivalent of the well-associated volume, swept against entry depth.

    Re-splits nothing: at each depth the discovery group is ``contact > z``, and the
    volume is the trial's own recoverable resource -- the same population
    ``discovery_mean`` averages, so the CE and the expectation on 3.10 are two
    statistics of one distribution rather than two different ones.
    """
    from .groups import group_trials
    from .stats import thin

    contact = np.asarray(ts.col("contact"), dtype=float)
    res = np.asarray(ts.col("resource"), dtype=float)
    z = np.asarray(vsweep.z, dtype=float)
    pw = thin(vsweep.p_well, vsweep.n_discovery, min_support)

    ce = np.full(z.size, np.nan)
    for i, zi in enumerate(z):
        if not np.isfinite(pw[i]):
            continue
        disc = (res > 0.0) & (contact > zi)
        if disc.sum() < min_support:
            continue
        ce[i] = certainty_equivalent(res[disc], float(pw[i]), rho)

    best = int(np.nanargmax(ce)) if np.any(np.isfinite(ce)) else None
    return RiskAdjusted(z=z, ce=ce, rho=float(rho), best=best)


@dataclass(frozen=True)
class ConstrainedResult:
    """The best-odds location that still meets a commerciality confidence."""

    confidence: float
    feasible: bool
    depth: float | None
    p_well_at: float | None
    p_commercial_at: float | None
    pc_at: float | None

    def message(self) -> str:
        if not self.feasible:
            return (
                f"No supported location reaches {self.confidence:.0%} confidence that a "
                f"discovery clears MEFS — the deepest well-supported entry does not get "
                f"there, so this hurdle cannot be met on this prospect."
            )
        return (
            f"For {self.confidence:.0%} confidence that a discovery is commercial, enter "
            f"at **{self.depth:,.0f} m or deeper**. The best odds available under that "
            f"constraint are P_well **{self.p_well_at:.1%}**, giving "
            f"Pc **{self.pc_at:.1%}**."
        )


def constrained_best(vsweep, *, confidence: float = DEFAULT_CONFIDENCE,
                     min_support: int = 30) -> ConstrainedResult:
    """Maximise ``P_well`` subject to ``P(> MEFS | discovery) >= confidence``.

    **The constraint is read as a guarantee, not a first touch** -- the same rule B6's
    ``_required_depth`` uses, and for the same reason. ``P(commercial | discovery)`` is
    a sampled curve and dips wherever the discovery group is small, so inverting the
    first crossing returns depths that deeper locations contradict. A running minimum
    from the deep end gives "from here down it stays above the hurdle".

    ``P_well`` falls monotonically down-dip, so once the feasible set is a half-line its
    shallow end is the answer; no search is needed beyond finding that end.
    """
    from .stats import thin

    if not 0.0 < confidence <= 1.0:
        raise ValueError(f"confidence must be in (0, 1]; got {confidence!r}")
    if vsweep.p_discovery_exceeds_mefs is None:
        raise ValueError("this sweep carries no commerciality curve; it needs a mefs")

    z = np.asarray(vsweep.z, dtype=float)
    pw = thin(vsweep.p_well, vsweep.n_discovery, min_support)
    pm = thin(vsweep.p_discovery_exceeds_mefs, vsweep.n_discovery, min_support)

    # Running minimum from the deep end: guaranteed[i] is the worst commerciality at
    # or below z[i].
    guaranteed = np.full(z.size, np.nan)
    running = np.inf
    for i in range(z.size - 1, -1, -1):
        if np.isfinite(pm[i]):
            running = min(running, float(pm[i]))
            guaranteed[i] = running

    ok = np.isfinite(guaranteed) & (guaranteed >= confidence)
    if not ok.any():
        return ConstrainedResult(float(confidence), False, None, None, None, None)

    i = int(np.argmax(ok))          # shallowest feasible index
    return ConstrainedResult(
        confidence=float(confidence), feasible=True, depth=float(z[i]),
        p_well_at=float(pw[i]), p_commercial_at=float(pm[i]),
        pc_at=float(pw[i] * pm[i]),
    )


#: The hurdles 3.13 sweeps. Stops at 99 %: a curve that runs to 100 % spends its last
#: third on a region where the constraint is met only by the deepest supported step,
#: which says more about where the sweep ends than about the prospect.
HURDLE_LEVELS = np.round(np.arange(0.50, 0.9901, 0.01), 4)


@dataclass(frozen=True)
class HurdleCurve:
    """What each level of commercial confidence costs, swept.

    The table on tab ③ answers this at one confidence. Sweeping it is what shows the
    shape of the trade -- and one feature of that shape is genuinely counterintuitive
    and worth drawing rather than asserting: **``Pc`` falls as the hurdle tightens.**
    Demanding more confidence pushes the well deeper, deeper costs chance faster than
    it buys commerciality, and the product goes down. A reader who believes "more
    confident" means "better" is reading the table wrong, and the curve says so.
    """

    confidence: np.ndarray
    depth: np.ndarray
    p_well: np.ndarray
    pc: np.ndarray

    @property
    def feasible(self) -> np.ndarray:
        return np.isfinite(self.depth)


def hurdle_curve(vsweep, *, levels=None, min_support: int = 30) -> HurdleCurve:
    """Sweep the commerciality hurdle, and report what each level costs.

    Each point is one :func:`constrained_best` call, so the curve and the panel cannot
    disagree about any single confidence -- the panel is a point on this curve.
    """
    q = np.asarray(HURDLE_LEVELS if levels is None else levels, dtype=float)
    depth = np.full(q.size, np.nan)
    pw = np.full(q.size, np.nan)
    pc = np.full(q.size, np.nan)
    for i, level in enumerate(q):
        r = constrained_best(vsweep, confidence=float(level), min_support=min_support)
        if r.feasible:
            depth[i], pw[i], pc[i] = r.depth, r.p_well_at, r.pc_at
    return HurdleCurve(confidence=q, depth=depth, p_well=pw, pc=pc)
