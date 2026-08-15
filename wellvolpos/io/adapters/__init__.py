"""Trial-file readers. Add a simulator by adding a file here and registering it below.

Two adapters ship. :class:`~wellvolpos.io.adapters.geox.GeoXAdapter` recognises an
SLB GeoX trial-browser export and knows its traps;
:class:`~wellvolpos.io.adapters.generic.GenericCsvAdapter` is the fallback for any
other delimited file and proposes a mapping rather than assuming one.

The order matters and is enforced by the confidence scores, not by this list:
GeoX scores up to 1.0 on a file it recognises, the generic adapter is capped at
0.3, so a real GeoX export is never read by the fallback. That cap is the whole
mechanism — see ``GenericCsvAdapter.sniff``.

Everything here reads from a :class:`~wellvolpos.io.adapters.source.Source`, which
is a name and some bytes. An uploaded file therefore never touches the disk
(design plan §10), and there is one code path rather than two.
"""
from pathlib import Path

from .base import CANONICAL_FIELDS, TrialAdapter, TrialSet
from .generic import GenericCsvAdapter, Proposal, propose, signature
from .stubs import STUB_ADAPTERS
from .geox import GeoXAdapter
from .source import Source

# **The stubs are registered, not omitted** (design plan §8). They sniff to 0.0 and
# raise on read, so they can never take a file from a working adapter -- but they are
# real TrialAdapter implementations, which is what keeps "adding a simulator is a file
# rather than a refactor" a checked claim instead of an aspiration.
ADAPTERS: list[TrialAdapter] = [GeoXAdapter(), GenericCsvAdapter(), *STUB_ADAPTERS]

__all__ = [
    "TrialSet",
    "TrialAdapter",
    "CANONICAL_FIELDS",
    "GeoXAdapter",
    "GenericCsvAdapter",
    "Proposal",
    "Source",
    "propose",
    "signature",
    "ADAPTERS",
    "STUB_ADAPTERS",
    "read_trials",
    "score_adapters",
]


def score_adapters(src) -> list[tuple[float, TrialAdapter]]:
    """Every adapter's confidence in ``src``, best first.

    Exposed because the app shows it: which adapter was chosen, and by how much,
    is provenance the reader should be able to see rather than infer from the
    numbers looking plausible.
    """
    source = Source.from_any(src)
    return sorted(((a.sniff(source), a) for a in ADAPTERS), key=lambda t: -t[0])


def read_trials(path, adapter=None) -> TrialSet:
    """Read a trial file, choosing the adapter by confidence unless one is given.

    ``path`` may be a path, an uploaded file, a file-like object or bytes; it is
    read into memory once and every adapter works from those bytes.
    """
    source = Source.from_any(path)
    if adapter is not None:
        return adapter.read(source)
    scored = score_adapters(source)
    best_score, best = scored[0]
    if best_score <= 0.0:
        raise ValueError(
            f"No adapter recognised {Path(source.name).name}. Tried: "
            + ", ".join(a.name for a in ADAPTERS)
            + ". The generic reader needs a contact column and a resource column it can "
            "identify; supply a mapping if the headers are unusual."
        )
    return best.read(source)
