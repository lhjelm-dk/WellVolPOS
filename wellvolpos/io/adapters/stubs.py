"""Documented stubs for the four other simulators the design plan names (§8).

The plan's point was that *"adding one later is a file rather than a refactor"*, and
that only stays true if someone has checked it. These are that check: each stub is a
real :class:`~wellvolpos.io.adapters.base.TrialAdapter`, registered, sniffed and
skipped — so the protocol is exercised by four more implementations than the two that
work, and a change to it breaks here rather than in six months.

**Every stub sniffs to 0.0 and raises on read.** They are not partial importers, and a
partial importer is the worst thing this package could ship: a file that half-loads and
produces numbers that look reasonable is far more damaging than one that refuses. The
message names what is missing and what a contributor would have to write.

Why each is not built:

RoseRA / MMRA
    Rose & Associates' own tools, and the closest in spirit to this app — the reference
    contour and ``Pmcfs`` conventions come from their work. Their exports are not in
    ``Papers/`` and the file layout is not documented publicly, so a stub written from
    guesswork would be worse than none.

Petrel PPA
    Petrel's Play & Prospect Assessment writes a project database rather than a flat
    trial table, so this is not a parser but an extraction: the trials live behind an
    Ocean plug-in or a manual export. The plug-in is the real work; the adapter after
    it is small.

@RISK
    An Excel add-in, so the trials arrive as a sheet whose *shape* is whatever the
    modeller built. There is no canonical layout to target — which is precisely the
    case :mod:`wellvolpos.io.adapters.generic` already covers, mapping proposed with a
    confidence and a reason per field. Point an @RISK user there.

Crystal Ball
    Same as @RISK, and the same answer: its forecast export is a flat CSV that the
    generic adapter handles once the columns are mapped.

**So two of the four are already served** by the generic adapter and are listed here to
say so rather than to be written. The two that need real work are RoseRA, which needs a
sample file, and Petrel, which needs a plug-in.
"""

from __future__ import annotations

from .base import TrialSet


class _NotBuiltAdapter:
    """Base for the stubs: recognises nothing, refuses to read, explains why."""

    name = "unbuilt"
    #: What a contributor would need before this could be written.
    needs = ""

    def sniff(self, path) -> float:
        """Always 0.0.

        A stub must never outrank a working adapter, and returning anything positive
        on a format nobody has verified is how a real GeoX export ends up on a
        half-written path. :func:`wellvolpos.io.adapters.score_adapters` sorts on this,
        so 0.0 keeps them last without special-casing.
        """
        return 0.0

    def read(self, path) -> TrialSet:
        raise NotImplementedError(
            f"{self.name} is not implemented. {self.needs} "
            f"Until then, export the trials to a flat table and use the GeoX or "
            f"generic-CSV adapter — every canonical field is listed in "
            f"wellvolpos.io.adapters.base.CANONICAL_FIELDS."
        )


class RoseRAAdapter(_NotBuiltAdapter):
    name = "Rose & Associates RoseRA / MMRA"
    needs = ("Needs a sample export: the layout is not documented publicly and a "
             "guessed mapping would be worse than no adapter.")


class PetrelPPAAdapter(_NotBuiltAdapter):
    name = "Petrel Play & Prospect Assessment"
    needs = ("Needs an Ocean plug-in or a manual export first — PPA keeps trials in a "
             "project database rather than a flat table, so the adapter is the small "
             "half of that job.")


class AtRiskAdapter(_NotBuiltAdapter):
    name = "@RISK (Excel)"
    needs = ("Probably never needed: an @RISK sheet has whatever shape the modeller "
             "built, which is exactly what the generic-CSV adapter is for — it "
             "proposes a mapping with a confidence and a reason per field.")


class CrystalBallAdapter(_NotBuiltAdapter):
    name = "Oracle Crystal Ball"
    needs = ("Probably never needed: its forecast export is a flat CSV the generic-CSV "
             "adapter handles once the columns are mapped.")


#: The four, in the order the design plan lists them.
STUB_ADAPTERS = (RoseRAAdapter(), PetrelPPAAdapter(),
                 AtRiskAdapter(), CrystalBallAdapter())
