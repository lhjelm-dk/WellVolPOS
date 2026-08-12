"""The two cached readers, shared by ``app.py`` and tab ③.

Both key on the file's **bytes**, not a path: an upload is never written to disk
(design plan §10) so there is no path to key on, and ``st.cache_data`` hashes bytes
happily. At these sizes -- the largest demo export is under 3 MB -- holding them
costs nothing next to re-parsing 10 000 rows on every slider drag.
"""

from __future__ import annotations

import streamlit as st

from ..core import AreaDepth, ReferenceContour, run_volume_sweep
from ..io.adapters import GenericCsvAdapter, Source, read_trials
from ..io.qc import run_qc

__all__ = ["load", "volume_sweep"]


@st.cache_data(show_spinner=False)
def load(name: str, data: bytes, mapping_items: tuple = ()):
    """Read and QC one trial file, keyed on its bytes.

    ``mapping_items`` is a sorted tuple rather than a dict so it is hashable: it
    carries a manual column mapping for the generic reader, and a different
    mapping is a different import.
    """
    src = Source(name=name, data=data)
    adapter = GenericCsvAdapter(mapping=dict(mapping_items)) if mapping_items else None
    ts = read_trials(src, adapter=adapter)
    return ts, run_qc(ts)


@st.cache_data(show_spinner=False)
def volume_sweep(name: str, data: bytes, mapping_items: tuple,
                 pos: float, gap: float, mefs: float, reference: str):
    """The proven/possible sweep, cached on the settings that determine it.

    The most expensive computation on the page -- it re-splits every trial at every
    one of sixty depths and bootstraps each step -- and B6's slider does not change
    any of its inputs. Keyed on the file's bytes plus the scalars rather than on the
    TrialSet, because a dataclass holding a DataFrame is not hashable; ``load`` is
    itself cached, so re-reading here is free.
    """
    ts_, _ = load(name, data, mapping_items)
    ad_ = AreaDepth.from_trials(ts_.col("contact"), ts_.col("area"))
    return run_volume_sweep(
        ts_, ad_, pos, z_gap=gap, mefs=mefs,
        reference=ReferenceContour(reference), n_boot=400,
    )
