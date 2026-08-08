"""Trial-file readers. Add a simulator by adding a file here and registering it below."""
from pathlib import Path

from .base import CANONICAL_FIELDS, TrialAdapter, TrialSet
from .geox import GeoXAdapter

ADAPTERS: list[TrialAdapter] = [GeoXAdapter()]

__all__ = ["TrialSet", "TrialAdapter", "CANONICAL_FIELDS", "GeoXAdapter", "ADAPTERS", "read_trials"]


def read_trials(path, adapter=None) -> TrialSet:
    """Read a trial file, choosing the adapter by confidence unless one is given."""
    path = str(path)
    if adapter is not None:
        return adapter.read(path)
    scored = sorted(((a.sniff(path), a) for a in ADAPTERS), key=lambda t: -t[0])
    best_score, best = scored[0]
    if best_score <= 0.0:
        raise ValueError(
            f"No adapter recognised {Path(path).name}. Tried: "
            + ", ".join(a.name for a in ADAPTERS)
        )
    return best.read(path)
