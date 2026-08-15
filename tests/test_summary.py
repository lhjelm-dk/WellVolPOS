"""The headline block: five numbers and a sentence, none of them re-derived.

A headline is the part a reader quotes into a well proposal, which makes it the worst
possible place for this project's recurring bug -- an unrisked number under a risked
label. These tests hold it to the same standard as the figures: every value must equal
the object the app already computed it in, never a parallel calculation that happens to
agree today.
"""

import numpy as np
import pytest

from wellvolpos.core import (
    AreaDepth,
    class_summary,
    group_trials,
    headline,
    p_well,
    split_trials,
)
from wellvolpos.core.rose import commercial_chance

from .conftest import ENTRY, EXIT

POS = 0.7605


@pytest.fixture(scope="module")
def parts(reduced):
    ad = AreaDepth.from_trials(reduced.col("contact"), reduced.col("area"))
    g = group_trials(reduced, ENTRY, EXIT)
    vc = split_trials(reduced, ad, g, ENTRY, EXIT)
    ch = p_well(reduced, ENTRY, POS)
    cs = class_summary(vc, g)
    cc = commercial_chance(reduced, g, vc.proven, ch.p_well, 14.0)
    h = headline(entry=ENTRY, exit_=EXIT, chance=ch, groups=g,
                 class_stats=cs, commercial=cc, mefs=14.0)
    return h, ch, cs, cc, g


def test_the_headline_reports_p_well_not_a_product_of_its_own(parts):
    """The fifth instance of this bug was B4 totalling its own steps. Not here."""
    h, ch, _, _, _ = parts
    assert h.p_well == pytest.approx(ch.p_well, abs=0)
    assert h.pos_prospect == pytest.approx(ch.pos_prospect, abs=0)
    assert h.r_location == pytest.approx(ch.r_location, abs=0)
    # And the decomposition still multiplies, so the sentence cannot drift from it.
    assert h.pos_prospect * h.r_location == pytest.approx(h.p_well, rel=1e-12)


def test_every_volume_is_the_one_class_summary_already_had(parts):
    h, _, cs, _, _ = parts
    assert h.proven_mean == pytest.approx(cs["proven"]["mean"], abs=0)
    assert h.well_associated_mean == pytest.approx(cs["discovery"]["mean"], abs=0)
    assert h.attic_mean == pytest.approx(cs["attic_dry_hole"]["mean"], abs=0)


def test_pc_is_at_or_below_p_well_and_comes_from_rose(parts):
    h, ch, _, cc, _ = parts
    assert h.pc_well == pytest.approx(cc.pc_well, abs=0)
    # Pc = P_well x Pmcfs, and a probability cannot exceed one, so Pc <= P_well.
    assert h.pc_well <= h.p_well + 1e-12


def test_the_counts_are_the_populations_not_a_guess(parts):
    """The first draft wrote "of 10,000 trials" as a literal -- prospect A's count,
    printed above whatever file happened to be loaded."""
    h, _, _, _, g = parts
    assert h.n_discovery == int(np.asarray(g.discovery).sum())
    assert h.n_dry_with_attic == int(np.asarray(g.dry_with_attic).sum())
    assert h.n_total == int(np.asarray(g.discovery).size)
    assert f"{h.n_total:,}" in h.sentence()


def test_the_sentence_names_the_conditioning_of_every_volume(parts):
    """It is written to be quoted, so a volume in it must not read as unconditional."""
    h, _, _, _, _ = parts
    text = h.sentence()
    assert "If it works" in text          # proven and well-associated
    assert "If it is dry" in text         # the attic, a different outcome
    assert f"{h.p_well:.1%}" in text
    assert f"{h.pc_well:.1%}" in text


def test_the_chance_half_survives_a_file_with_no_area(reduced):
    """A tab that sometimes has a headline is worse than one that always has the part
    it can stand behind."""
    g = group_trials(reduced, ENTRY, EXIT)
    h = headline(entry=ENTRY, exit_=EXIT, chance=p_well(reduced, ENTRY, POS), groups=g)
    assert h.p_well > 0
    assert h.pc_well is None and h.proven_mean is None
    assert "chance of finding hydrocarbons" in h.sentence()
    assert "If it works" not in h.sentence()


# ------------------------------------------------------------- candidate depths
def test_the_candidate_depths_agree_with_the_stars_the_figures_draw(reduced):
    """The panel is a second implementation of two figures' argmax, and stays one.

    ``candidate_depths`` mirrors B8's and B9's arithmetic rather than refactoring
    them, because moving their starred markers is a bigger change than the panel
    warrants. That is only safe with this test: if either figure changes how it
    weights or thins, the panel starts naming a depth the figure does not star.
    """
    from wellvolpos.core import candidate_depths, run_volume_sweep
    from wellvolpos.core.stats import thin

    ad = AreaDepth.from_trials(reduced.col("contact"), reduced.col("area"))
    sw = run_volume_sweep(reduced, ad, POS, z_gap=50.0, mefs=14.0, n=40)
    cands = {c.key: c for c in candidate_depths(sw)}
    assert {"chance", "expected", "commercial"} <= set(cands)

    pw = thin(sw.p_well, sw.n_discovery, 30)

    # B9 stars nanargmax(P_well x well-associated mean).
    weighted = pw * thin(sw.discovery_mean, sw.n_discovery, 30)
    assert cands["expected"].depth == pytest.approx(
        float(sw.z[int(np.nanargmax(weighted))]), abs=0)

    # B8 stars nanargmax(P_well x P(discovery > MEFS)).
    pc = pw * thin(sw.p_discovery_exceeds_mefs, sw.n_discovery, 30)
    assert cands["commercial"].depth == pytest.approx(
        float(sw.z[int(np.nanargmax(pc))]), abs=0)


def test_best_chance_is_the_shallow_end_and_says_so(reduced):
    """P_well falls monotonically down-dip, so its maximum is not a recommendation.

    Reporting it without that caveat would put the shallowest depth in the app at the
    top of a table headed "candidate depths", which reads as advice to drill the crest.
    """
    from wellvolpos.core import candidate_depths, run_volume_sweep

    ad = AreaDepth.from_trials(reduced.col("contact"), reduced.col("area"))
    sw = run_volume_sweep(reduced, ad, POS, z_gap=50.0, mefs=14.0, n=40)
    best = next(c for c in candidate_depths(sw) if c.key == "chance")
    supported = sw.z[np.asarray(sw.n_discovery) >= 30]
    assert best.depth == pytest.approx(float(supported.min()), abs=0)
    assert "by construction" in best.note


def test_a_required_depth_joins_the_panel_only_when_given(reduced):
    from wellvolpos.core import candidate_depths, run_volume_sweep

    ad = AreaDepth.from_trials(reduced.col("contact"), reduced.col("area"))
    sw = run_volume_sweep(reduced, ad, POS, z_gap=50.0, mefs=14.0, n=40)
    assert not any(c.key == "required" for c in candidate_depths(sw))
    with_req = candidate_depths(sw, required_depth=3520.0, required_target=15.0)
    req = next(c for c in with_req if c.key == "required")
    assert req.depth == 3520.0 and "15.0 MMboe" in req.value


def test_the_optimum_is_reported_as_a_plateau_and_survives_a_grid_change(reduced):
    """A depth that moves 51 m when the grid changes is false precision.

    Prospect B's commercial optimum came out at 2064 m on one sweep and 2115 m on
    another, both at Pc 21.9 % — because above the shallowest contact every success
    trial is a discovery, so r_location is 1 and the curve is exactly flat. argmax was
    breaking a genuine tie and reporting the winner as if it were a peak.
    """
    from wellvolpos.core import candidate_depths, run_volume_sweep

    ad = AreaDepth.from_trials(reduced.col("contact"), reduced.col("area"))
    coarse = {c.key: c for c in
              candidate_depths(run_volume_sweep(reduced, ad, POS, z_gap=50.0,
                                                mefs=14.0, n=40))}
    fine = {c.key: c for c in
            candidate_depths(run_volume_sweep(reduced, ad, POS, z_gap=50.0,
                                              mefs=14.0, n=60))}
    assert set(coarse) == set(fine)
    for key in coarse:
        a, b = coarse[key].plateau, fine[key].plateau
        assert a is not None and b is not None, key
        # The reported span must not depend on how finely the sweep was sampled.
        assert abs(a[0] - b[0]) <= 12.0, (key, a, b)
        assert abs(a[1] - b[1]) <= 12.0, (key, a, b)
        # And the argmax depth still lies inside the span it is reported with.
        assert a[0] - 1e-9 <= coarse[key].depth <= a[1] + 1e-9, key


def test_a_flat_optimum_prints_a_range_and_a_sharp_one_prints_a_depth(reduced):
    from wellvolpos.core import Candidate, candidate_depths, run_volume_sweep

    ad = AreaDepth.from_trials(reduced.col("contact"), reduced.col("area"))
    sw = run_volume_sweep(reduced, ad, POS, z_gap=50.0, mefs=14.0, n=40)
    for c in candidate_depths(sw):
        assert ("–" in c.describe_depth()) == c.is_flat, (c.key, c.describe_depth())
    # A candidate with no plateau — the required depth — states one number.
    only = Candidate(key="required", label="x", depth=3520.0, value="v", figure="b6")
    assert only.describe_depth() == "3,520 m" and not only.is_flat


# ------------------------------------------------------------- the outcome tree
def test_the_outcome_tree_partitions_to_one_and_agrees_with_p_well(reduced):
    """The sixth chance to reproduce this project's recurring bug, and the guard.

    An outcome tree that counts trial masks reports ``POS_trials`` under a ``P_well``
    label and looks entirely reasonable — A2 did exactly that, and B4 did the
    arithmetic equivalent. So this cross-checks the drawn leaves against
    ``core.chance.p_well``, never against the figure's own sum.
    """
    import matplotlib
    matplotlib.use("Agg")
    import wellvolpos.viz.interactive as I
    from wellvolpos.core import p_well as p_well_fn
    from wellvolpos.core.rose import commercial_chance

    ad = AreaDepth.from_trials(reduced.col("contact"), reduced.col("area"))
    g = group_trials(reduced, ENTRY, EXIT)
    vc = split_trials(reduced, ad, g, ENTRY, EXIT)
    ch = p_well_fn(reduced, ENTRY, POS)
    cc = commercial_chance(reduced, g, vc.proven, ch.p_well, 14.0)

    fig = I.pfig_c6_outcome_tree(g, pos_prospect=ch.pos_prospect,
                                 p_well=ch.p_well, pc_well=cc.pc_well)
    leaves = {t.name: float(t.x[0]) for t in fig.data if t.name}
    assert len(leaves) == 4, leaves
    assert sum(leaves.values()) == pytest.approx(1.0, abs=1e-12)

    # The discovery branch is P_well — from ChanceResult, not from the mask.
    discovery = sum(v for k, v in leaves.items() if k.startswith("Discovery"))
    assert discovery == pytest.approx(ch.p_well, abs=1e-12)
    # And it is *not* the trial file's own success rate, which is the number the bug
    # would have produced. On this file those differ by more than 15 points.
    pos_trials = float((reduced.col("resource") > 0).mean())
    assert abs(discovery - pos_trials) > 0.15, (discovery, pos_trials)

    # The commercial leaf is Rose's Pc, and the two discovery leaves split at it.
    assert leaves["Discovery, commercial"] == pytest.approx(cc.pc_well, abs=1e-12)
    assert leaves["Discovery, below MEFS"] == pytest.approx(
        ch.p_well - cc.pc_well, abs=1e-12)

    # Chance failure is 1 - POS, so a well cannot be luckier than its prospect.
    assert leaves["Chance failure — no hydrocarbons anywhere"] == pytest.approx(
        1.0 - ch.pos_prospect, abs=1e-12)


def test_the_outcome_tree_works_without_a_threshold(reduced):
    """No MEFS means no commercial split, and the tree still partitions to one."""
    import wellvolpos.viz.interactive as I
    from wellvolpos.core import p_well as p_well_fn

    g = group_trials(reduced, ENTRY, EXIT)
    ch = p_well_fn(reduced, ENTRY, POS)
    fig = I.pfig_c6_outcome_tree(g, pos_prospect=ch.pos_prospect, p_well=ch.p_well)
    leaves = {t.name: float(t.x[0]) for t in fig.data if t.name}
    assert set(leaves) == {"Chance failure — no hydrocarbons anywhere",
                           "Dry hole, hydrocarbons up-dip", "Discovery"}
    assert sum(leaves.values()) == pytest.approx(1.0, abs=1e-12)
    assert leaves["Discovery"] == pytest.approx(ch.p_well, abs=1e-12)


def test_the_starred_optima_report_a_band_not_a_grid_point(reduced):
    """3.9 and 3.10 starred an arbitrary tie-break until 2026-08-15.

    Above the shallowest contact every success trial is a discovery, so r_location is 1
    and both curves are exactly flat there. ``argmax`` picked whichever grid point won
    by a hair and the star claimed it as a peak -- prospect B's commercial optimum moved
    2064 -> 2115 m between two sweeps at an identical Pc of 21.9 %.
    """
    import re

    import wellvolpos.viz.interactive as I
    from wellvolpos.core import run_volume_sweep

    ad = AreaDepth.from_trials(reduced.col("contact"), reduced.col("area"))

    def band(n):
        sw = run_volume_sweep(reduced, ad, POS, z_gap=50.0, mefs=14.0, n=n)
        fig = I.pfig_b8_commercial_chance(sw, current_z=ENTRY)
        text = next(t.text[0] for t in fig.data
                    if t.text and "best Pc" in str(t.text[0]))
        found = re.findall(r"([\d,]+)–([\d,]+) m", text)
        assert found, f"the star still reports a single depth: {text!r}"
        return tuple(float(v.replace(",", "")) for v in found[0])

    coarse, fine = band(40), band(60)
    # The reported band must not depend on how finely the sweep was sampled.
    assert abs(coarse[0] - fine[0]) <= 12.0, (coarse, fine)
    assert abs(coarse[1] - fine[1]) <= 12.0, (coarse, fine)
    assert coarse[1] > coarse[0]


def test_the_figures_and_the_panel_use_one_plateau_definition(reduced):
    """``plateau_span`` is shared, so 3.9's band and the candidate panel's range for the
    same measure cannot disagree."""
    import numpy as np

    from wellvolpos.core import candidate_depths, plateau_span, run_volume_sweep
    from wellvolpos.core.stats import thin

    ad = AreaDepth.from_trials(reduced.col("contact"), reduced.col("area"))
    sw = run_volume_sweep(reduced, ad, POS, z_gap=50.0, mefs=14.0, n=40)
    pw = thin(sw.p_well, sw.n_discovery, 30)
    pc = pw * thin(sw.p_discovery_exceeds_mefs, sw.n_discovery, 30)
    direct = plateau_span(pc, sw.z, int(np.nanargmax(pc)))
    panel = next(c for c in candidate_depths(sw) if c.key == "commercial").plateau
    assert direct == pytest.approx(panel, abs=1e-9)
