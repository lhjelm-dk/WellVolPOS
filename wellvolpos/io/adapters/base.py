"""The canonical trial container and the adapter protocol.

Everything downstream of import consumes a :class:`TrialSet` and nothing else.
Adding support for a new simulator is therefore a new file in this package, not
a change anywhere else in the codebase.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import pandas as pd

# Canonical field -> (unit, required). Units are fixed: MMboe, m, km2.
CANONICAL_FIELDS: dict[str, tuple[str, bool]] = {
    "trial": ("", False),
    "contact": ("m TVDSS", True),
    "resource": ("MMboe", True),
    "area": ("km2", False),
    "gross_pay": ("m", False),
    "hc_grv": ("1e6 m3", False),
    "hc_pv": ("1e6 m3", False),
    "crest": ("m TVDSS", False),
    "spill": ("m TVDSS", False),
    "net_gross": ("fraction", False),
    "porosity": ("fraction", False),
    "thickness": ("m", False),
}


@dataclass
class TrialSet:
    """One prospect segment's Monte Carlo trials, in canonical form.

    ``frame`` carries the canonical columns above; ``source_columns`` records
    which original column each one came from, so the QC report and the exported
    provenance stamp can show the user exactly what was mapped to what.
    """

    frame: pd.DataFrame
    source_columns: dict[str, str] = field(default_factory=dict)
    units: dict[str, str] = field(default_factory=dict)
    source: str = "unknown"
    prospect: str = ""
    notes: list[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.frame)

    @property
    def n_trials(self) -> int:
        return len(self.frame)

    def has(self, name: str) -> bool:
        return name in self.frame.columns and self.frame[name].notna().any()

    def col(self, name: str):
        """Return a canonical column as a numpy array, raising a useful error."""
        if name not in self.frame.columns:
            raise KeyError(
                f"canonical field {name!r} is not present in this trial set "
                f"(have: {sorted(self.frame.columns)})"
            )
        return self.frame[name].to_numpy()


@runtime_checkable
class TrialAdapter(Protocol):
    """A reader for one simulator's trial export."""

    name: str

    def sniff(self, path: str) -> float:
        """Return confidence in [0, 1] that this adapter can read ``path``."""

    def read(self, path: str) -> TrialSet:
        """Parse ``path`` into a :class:`TrialSet`."""
