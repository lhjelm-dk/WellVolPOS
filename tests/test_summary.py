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
