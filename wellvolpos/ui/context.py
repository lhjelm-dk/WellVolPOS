"""Everything a tab needs to draw itself, resolved once by ``app.py``.

The point of collecting this into one object is not tidiness -- it is that the
things a tab must not re-derive are now visibly *given* to it. Three in
particular:

``pos`` and ``pos_source``
    Which POS is in force depends on the risking convention *and* on what the
    failure detector found in the file, and ``app.py`` resolves that branch once.
    CLAUDE.md's "POS provenance" section exists to keep it singular; a tab that
    recomputed it would be the second implementation.

``vc``
    The proven/possible split, apportioned on the wedge. Computed once per rerun
    because the thickness inversion is not free, and because two tabs splitting
    the same trials independently is how they come to disagree.

``elements`` / ``play_elements``
    Read out of ``st.session_state`` at the top of the script, because tab ②
    *creates* those widgets further down and Streamlit runs top to bottom. The
    widgets own the keys, so a change reruns and this read sees the new value at
    the top of it -- no lag, and no second copy of the state.

Frozen, so a tab cannot quietly reassign what it was handed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..core.chance import ChanceResult, ReferenceContour
from ..core.classes import VolumeClasses
from ..core.groups import Groups
from ..core.structure import AreaDepth
from ..io.adapters.base import TrialSet


@dataclass(frozen=True)
class Ctx:
    """One rerun's resolved state, shared by every tab."""

    # ------------------------------------------------------------------ the data
    ts: TrialSet
    #: The chosen demo's label, or the upload's file name. Stamped into a `Case`.
    dataset: str
    #: The bytes themselves, wrapped. Both `st.cache_data` loaders key on
    #: ``source.name`` and ``source.data`` rather than on a path, because an upload
    #: is never written to disk (design plan §10) and there is no path to key on.
    source: Any                   #: io.adapters.Source
    #: Manual column mapping for the generic reader, empty for a recognised format.
    #: A different mapping is a different import, so it is part of the cache key.
    overrides: dict[str, str]
    qc: Any                       #: QCReport -- Any, to keep io.qc out of the import graph

    # ------------------------------------------- the structure, where there is one
    ad: AreaDepth | None
    has_area: bool

    # ------------------------------------------------------------------ the well
    entry: float
    exit_: float

    # -------------------------------------------- conventions, never implicit (5)
    mefs: float
    ref: ReferenceContour
    scheme: str
    area_scale: str

    # ------------------------------------------------------------------ the risk
    elements: dict[str, float]
    play_elements: dict[str, float]
    play_chance: float
    risking_convention: str
    pos: float
    pos_source: str
    pos_from_table: float
    pos_trials: float | None

    # --------------------------------------------------------------- the results
    groups: Groups
    vc: VolumeClasses | None
    chance: ChanceResult

    # ------------------------------------ is the split defensible on this data?
    split_level: str
    split_message: str
    split_r: float

    @property
    def gap(self) -> float:
        """The well's own entry-to-exit spacing.

        Both sweeps take this, so a swept location is the same well moved up- or
        down-dip rather than a well of a different length. It cannot affect
        ``r_location`` or ``P_well`` -- ``group_trials``' discovery mask depends on
        the entry alone -- which is what makes it safe to sweep.
        """
        return self.exit_ - self.entry
    #: Half-width of the "contact lands on the well" window, in metres. Feeds both
    #: tab ④'s metric and 3.5's swept curve, which is why it is resolved once here
    #: rather than read separately in each -- they disagreed about the same quantity
    #: until 2026-08-14, the curve keeping the 2.0 m default while the metric used
    #: whatever had been typed.
    at_well_window: float = 2.0
