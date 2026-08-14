"""What the reservoir exit moves, and whether the spacing is a well anyone can drill.

Rescued from ``test_multiwell.py`` when the candidate-well feature was removed
(2026-08-14). None of these is about comparing locations: they are about the *exit*,
which one well has as much as four did, and about the entry-to-exit spacing against
the reservoir thickness recovered from the trials.
"""

import numpy as np
import pytest

from wellvolpos.core import AreaDepth, run_volume_sweep
from wellvolpos.core.dependence import ENTRY_ONLY, EXIT_DEPENDENT, check_deviation


def test_only_two_volume_classes_move_when_the_exit_moves(full):
    """The structural fact the multi-well drawing rests on.

    ``group_trials`` calls a discovery ``contact > z_entry``, so the *populations*
    depend on the entry alone. The exit enters only through ``LKH = min(contact,
    z_exit)``, which is where ``split_trials`` cuts -- so it moves the **boundary
    between proven and unproven** and nothing else, and their sum is fixed.

    Which is why a candidate with a different entry-to-exit spacing needs its own
    proven and below-LKH curves and *shares* every other one.
    """
    from wellvolpos.core import (ENTRY_ONLY, EXIT_DEPENDENT, group_summary,
                                 group_trials, split_trials, thickness_from_pay)

    ad = AreaDepth.from_trials(full.col("contact"), full.col("area"))
    th = thickness_from_pay(full, ad).thickness
    apex = ad.apex_estimate()

    seen = []
    for exit_ in (3520.0, 3550.0, 3600.0):
        g = group_trials(full, 3500.0, exit_)
        vc = split_trials(full, ad, g, 3500.0, exit_, thickness=th, apex=apex)
        d = np.asarray(g.discovery, dtype=bool)
        hc = np.asarray(g.hc_to_exit, dtype=bool)
        seen.append({
            "proven": float(vc.proven[d].mean()),
            # The **additive** reading: over every discovery, so it is exactly
            # well-associated minus proven and therefore monotone by construction.
            # The conditional one is not monotone and must not be asserted as if it
            # were -- deepening the exit shrinks each trial's remainder but also
            # drops the shallow-contact trials from the population, and on prospect A
            # the second effect wins at first (2.24 -> 2.53).
            "below_lkh": float(vc.below_lkh[d].mean()),
            "below_lkh_cond": float(vc.below_lkh[hc].mean()) if hc.any() else float("nan"),
            "well_assoc": float(vc.discovery_total[d].mean()),
            "attic": float(vc.attic[np.asarray(g.dry_with_attic, dtype=bool)].mean()),
            "n_disc": int(d.sum()),
        })

    # Entry-only: identical at every exit.
    for key in ("well_assoc", "attic", "n_disc"):
        vals = [row[key] for row in seen]
        assert max(vals) - min(vals) < 1e-9, (key, vals)

    # Exit-dependent, and monotone in the direction the geometry demands: a deeper
    # exit proves more of the column and leaves less unproven.
    provens = [row["proven"] for row in seen]
    belows = [row["below_lkh"] for row in seen]
    assert provens == sorted(provens), provens
    assert belows == sorted(belows, reverse=True), belows
    # And they are exact complements at every exit, which is what makes the split a
    # decomposition rather than two independent estimates.
    for row in seen:
        assert row["proven"] + row["below_lkh"] == pytest.approx(row["well_assoc"],
                                                                abs=1e-9)
    assert provens[-1] - provens[0] > 0.2, "the exit should matter materially here"

    # And the two lists say so.
    assert "proven" in EXIT_DEPENDENT and "below_lkh" in EXIT_DEPENDENT
    assert "attic" in ENTRY_ONLY and "p_well" in ENTRY_ONLY
    assert not set(EXIT_DEPENDENT) & set(ENTRY_ONLY)


def test_the_spacing_is_checked_against_the_reservoir_it_must_sit_in(full):
    """A vertical well sees exactly the reservoir thickness, so the entry-to-exit
    spacing is not a free choice unless the well is deviated.

    Prospect B's reservoir is 50 m throughout, so a 150 m spacing is a commitment to
    a deviated well -- and the volumes are computed as though one had been drilled.
    Reported, never enforced: all three cases are legitimate wells.
    """
    from wellvolpos.core import check_deviation, thickness_from_pay

    ad = AreaDepth.from_trials(full.col("contact"), full.col("area"))
    th = thickness_from_pay(full, ad).thickness

    # Prospect A's reservoir has a *distribution* of thicknesses -- P90 35 m, P50 45 m,
    # P10 55 m -- unlike prospect B's constant 50 m, so the check is written against
    # its own median rather than against a number typed here.
    median = check_deviation(th, 1.0).thickness_p50
    assert 30.0 < median < 60.0, median
    assert check_deviation(th, median).verdict == "vertical"
    assert check_deviation(th, median * 3).verdict == "deviated"
    assert check_deviation(th, median * 0.5).verdict == "partial"
    assert check_deviation(th, median * 3).ratio == pytest.approx(3.0, abs=0.05)
    # No thickness recoverable -> says so rather than guessing.
    assert check_deviation(np.array([np.nan, np.nan]), 50.0).verdict == "unknown"


def test_the_at_the_well_window_reaches_the_swept_curve(reduced, area_depth):
    """Tab ④'s metric and 3.5's curve are the same quantity and must use one window.

    The curve kept the 2.0 m default while the metric used whatever had been typed,
    so the two could disagree with nothing on screen saying why (found 2026-08-14).
    """
    from wellvolpos.core import run_volume_sweep

    narrow = run_volume_sweep(reduced, area_depth, 0.76, n=12, z_gap=50.0,
                              at_well_window=1.0)
    wide = run_volume_sweep(reduced, area_depth, 0.76, n=12, z_gap=50.0,
                            at_well_window=10.0)
    assert narrow.at_well_window == 1.0 and wide.at_well_window == 10.0
    # A wider window takes in more trials at every depth that has any.
    ok = (narrow.at_well_n > 0) & (wide.at_well_n > 0)
    assert ok.any()
    assert np.all(wide.at_well_n[ok] >= narrow.at_well_n[ok])
    assert wide.at_well_n.sum() > narrow.at_well_n.sum()


# ------------------------------------------------- what the exit actually moves
