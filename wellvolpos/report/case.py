"""Save and reload a session: every setting, and nothing that was computed.

A "case" is the set of choices that turn one trial file into one answer — the
well geometry, the threshold volume, the five conventions of non-negotiable 5,
and the chance table. Write those down and the whole app is reproducible; write
down the answers instead and you have a document that can disagree with the code
that opens it.

**The case stores settings only, deliberately.** Every number this tool reports
is re-derived on load. A case file that carried its own KPIs would be a way to
display last month's numbers under this month's version, with nothing on screen
saying so — the same class of mistake as showing an unrisked number under a
risked label, which this codebase has made three times. The one thing that
travels besides the settings is *provenance*: which trial file the case was built
against, how many trials it had, and a fingerprint of its contents.

**The trial data itself is never embedded.** A GeoX export is the licensee's
data and can be 60 columns by 10 000 rows; a case is a few hundred bytes of
choices. So a case names its trial file and fingerprints it, and
:func:`check_against` says plainly when the file it is being reopened against is
not the one it was saved from. It will still open — re-running a case on new
trials is a legitimate thing to want — but it says so first rather than quietly
answering a different question.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import MISSING, asdict, dataclass, field, fields
from datetime import datetime, timezone
from typing import Any

import numpy as np

from ..core.chance import ELEMENTS, SHIPPED_SCHEMES, ReferenceContour

FORMAT = "wellvolpos-case"
FORMAT_VERSION = 1

#: Risking-convention keys. These are the stable identifiers ``app.py`` branches
#: on; the sentences shown beside them in the UI are presentation only, and a
#: case must survive them being reworded.
CONVENTION_KEYS = ("trials_risked", "success_case_only", "geometric")


@dataclass
class Case:
    """One session's settings, plus where its trials came from.

    Field names match the app's own variables so that the mapping between a
    control and its stored value is one to one and needs no translation table --
    a translation table being the place a save/load pair drifts apart.
    """

    # --- the well -----------------------------------------------------------
    entry: float
    exit: float
    mefs: float

    # --- the five explicit conventions (non-negotiable 5) -------------------
    risking_convention: str = "success_case_only"
    reference: str = ReferenceContour.CREST.value
    scheme: str = "equal_cube_root"
    chance_table: dict[str, float] = field(default_factory=lambda: {el: 1.0 for el in ELEMENTS})
    #: The chance the play works at all, one level above the four elements, which are
    #: read as conditional on it. Multiplies POS_prospect and therefore P_well.
    #: The play-level chance of each element, one level above ``chance_table``,
    #: whose four are read as conditional on the play working. Their product
    #: multiplies POS_prospect and therefore P_well.
    play_elements: dict[str, float] = field(
        default_factory=lambda: {el: 1.0 for el in ELEMENTS})

    # --- presentation, which changes no number -----------------------------
    area_scale: str = "area"
    map_interval: float = 50.0
    map_azimuth_deg: float = 35.0

    # --- provenance ---------------------------------------------------------
    dataset: str = ""
    n_trials: int = 0
    fingerprint: str = ""
    saved_utc: str = ""
    note: str = ""

    @property
    def play_chance(self) -> float:
        """The play's overall chance: the product of its four elements."""
        return float(np.prod(list(self.play_elements.values()))) if self.play_elements else 1.0

    def __post_init__(self) -> None:
        # Stamped at construction rather than at serialisation, so a case is
        # exactly what it will be written as. A ``to_json`` that filled in a
        # field the object did not hold made save/load asymmetric -- reloading a
        # case gave an object that no longer equalled the one saved.
        self.saved_utc = self.saved_utc or datetime.now(timezone.utc).isoformat(
            timespec="seconds")
        self.entry, self.exit = float(self.entry), float(self.exit)
        self.mefs = float(self.mefs)
        self.play_elements = {k: float(v) for k, v in dict(self.play_elements).items()}
        for _name, _value in self.play_elements.items():
            if not 0.0 < _value <= 1.0:
                raise ValueError(f"play chance for {_name!r} must be in (0, 1]; got {_value}")
        _missing_play = set(ELEMENTS) - set(self.play_elements)
        if _missing_play:
            raise ValueError(f"play chances are missing {sorted(_missing_play)}")
        self.chance_table = {k: float(v) for k, v in dict(self.chance_table).items()}
        if self.risking_convention not in CONVENTION_KEYS:
            raise ValueError(
                f"unknown risking convention {self.risking_convention!r}; "
                f"expected one of {CONVENTION_KEYS}"
            )
        # Validated by *value*, so a case written by a later version that added a
        # reference contour fails loudly here rather than silently falling back
        # to the crest and answering a different question.
        ReferenceContour(self.reference)
        if self.scheme not in SHIPPED_SCHEMES:
            raise ValueError(
                f"unknown allocation scheme {self.scheme!r}; expected one of {SHIPPED_SCHEMES}"
            )
        if self.exit < self.entry:
            raise ValueError(
                f"reservoir exit ({self.exit} m) is above the entry ({self.entry} m); "
                f"a trajectory cannot leave the reservoir shallower than it entered"
            )
        missing = set(ELEMENTS) - set(self.chance_table)
        if missing:
            raise ValueError(f"chance table is missing {sorted(missing)}")

    # ------------------------------------------------------------------ json
    def to_json(self, *, indent: int = 2) -> str:
        """Serialise, stamping the format and the time.

        Sorted keys and a fixed indent so two cases saved from the same settings
        are byte-identical apart from the timestamp, which makes a case file
        diffable in git -- the point of choosing JSON over a pickle.
        """
        payload: dict[str, Any] = {
            "format": FORMAT,
            "format_version": FORMAT_VERSION,
            "settings": asdict(self),
        }
        return json.dumps(payload, indent=indent, sort_keys=True)

    @classmethod
    def from_json(cls, text: str | bytes) -> "Case":
        """Rebuild a case, rejecting anything that is not one.

        Checked rather than trusted: a case file is a file the user picked off
        disk, and the failure mode of a permissive loader is an app that starts
        with half its settings silently defaulted.
        """
        if isinstance(text, bytes):
            text = text.decode("utf-8")
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as e:
            raise ValueError(f"not a valid case file: {e}") from e
        if not isinstance(payload, dict) or payload.get("format") != FORMAT:
            raise ValueError(
                f"not a WellVolPOS case file (expected format {FORMAT!r}, "
                f"got {payload.get('format') if isinstance(payload, dict) else type(payload).__name__!r})"
            )
        version = payload.get("format_version")
        if not isinstance(version, int) or version > FORMAT_VERSION:
            raise ValueError(
                f"case format version {version} is newer than this build understands "
                f"({FORMAT_VERSION}); upgrade WellVolPOS rather than reading it partially"
            )
        settings = payload.get("settings")
        if not isinstance(settings, dict):
            raise ValueError("case file has no settings block")
        known = {f.name for f in fields(cls)}
        unknown = set(settings) - known
        if unknown:
            # Same-version files with extra keys mean the writer knew something
            # this reader does not. Naming them beats dropping them in silence.
            raise ValueError(f"case file carries settings this build does not know: {sorted(unknown)}")
        required = {
            f.name for f in fields(cls)
            if f.default is MISSING and f.default_factory is MISSING
        }
        absent = required - set(settings)
        if absent:
            raise ValueError(f"case file is missing required settings: {sorted(absent)}")
        return cls(**settings)

    # ---------------------------------------------------------- provenance
    def check_against(self, ts) -> list[str]:
        """Warnings about reopening this case on a different trial set.

        Returns an empty list when the trials match. Never raises and never
        blocks: re-running a case against a new export is a reasonable thing to
        do -- the requirement is only that the app says so out loud instead of
        presenting the result as a reproduction.
        """
        out: list[str] = []
        if self.fingerprint and self.fingerprint != fingerprint(ts):
            out.append(
                f"These trials are not the ones this case was saved against "
                f"(`{self.dataset or 'unnamed'}`). The settings are restored, but the numbers "
                f"will not reproduce the saved session."
            )
        if self.n_trials and self.n_trials != ts.n_trials:
            out.append(
                f"Trial count differs: {self.n_trials:,} when saved, {ts.n_trials:,} now."
            )
        contact = ts.col("contact")
        lo, hi = float(contact.min()), float(contact.max())
        for name, z in (("entry", self.entry), ("exit", self.exit)):
            if not (lo <= z <= hi):
                out.append(
                    f"Reservoir {name} {z:.0f} m is outside this file's contact range "
                    f"({lo:.0f}–{hi:.0f} m), so it was clamped."
                )
        return out


#: The columns a fingerprint is taken over. Deliberately the three that every
#: downstream number depends on -- the contact that places the well, the resource
#: being cut, and the area that turns one into the other -- rather than every
#: column in the file. ``data/`` holds one GeoX run exported twice, at 7 columns
#: and at 60; hashing whatever happens to be present would call those two
#: different trials, which is the wrong answer to the question a case asks. The
#: question is "are these the same trials", not "is this the same file".
FINGERPRINT_FIELDS = ("contact", "resource", "area")


def fingerprint(ts) -> str:
    """A short content hash of the trial numbers a case depends on.

    Over :data:`FINGERPRINT_FIELDS` in a fixed order, so it is stable against
    column order and against the extra columns a full export carries. A file
    missing one of them fingerprints differently from one that has it, which is
    correct: the two are not interchangeable here, because without ``area`` there
    is no A(z) and no proven/possible split.
    """
    h = hashlib.sha256()
    for name in FINGERPRINT_FIELDS:
        h.update(name.encode("utf-8"))
        if not ts.has(name):
            h.update(b"\x00absent")
            continue
        h.update(np.asarray(ts.col(name), dtype="float64").tobytes())
    return h.hexdigest()[:16]
