from pathlib import Path

import pytest

from wellvolpos.core import AreaDepth, group_trials
from wellvolpos.io.adapters import read_trials

DATA = Path(__file__).resolve().parents[1] / "data"

# The reference well, matching Results!E4 and Results!E5 of the source workbook.
ENTRY = 3500.0
EXIT = 3550.0


@pytest.fixture(scope="session")
def reduced():
    """The 7-column paste — the everyday case."""
    return read_trials(DATA / "demo_prospectA_reduced.csv")


@pytest.fixture(scope="session")
def full():
    """The 60-column GeoX export — exercises duplicate headers and prefixes."""
    return read_trials(DATA / "demo_prospectA_full.csv")


@pytest.fixture(scope="session")
def area_depth(reduced):
    return AreaDepth.from_trials(reduced.col("contact"), reduced.col("area"))


@pytest.fixture(scope="session")
def groups(reduced):
    return group_trials(reduced, ENTRY, EXIT)
