"""Stable keys, their user-facing labels, and the chance-table defaults.

**The app branches on the keys, never on the label text.** Rewording user copy
must not be able to change which POS the whole tool uses -- see CLAUDE.md's "POS
provenance" section. Keeping the two in one module, side by side, is what makes
that separation visible instead of a convention someone has to know.
"""

from __future__ import annotations

__all__ = [
    "CHANCE_DEFAULTS", "CHANCE_HELP", "CONVENTION_KEYS", "CONVENTION_LABELS",
    "CONVENTION_PROVENANCE", "PLAY_DEFAULTS", "PLAY_HELP",
]

CONVENTION_KEYS = ("trials_risked", "success_case_only", "geometric")
CONVENTION_LABELS = {
    "trials_risked": (
        "Correct — trials are risked; use the implied POS and lock the chance table to display-only"
    ),
    "success_case_only": "No — trials are success-case only; apply my chance table on top",
    "geometric": (
        "The zeros are geometric (contact above crest), not chance failure; treat separately"
    ),
}
CONVENTION_PROVENANCE = {
    "trials_risked": "trials (chance table display-only)",
    "success_case_only": "chance table",
    "geometric": "chance table (geometric reading not yet implemented)",
}

#: Chance-table defaults (Lars, 2026-08-11). They were 1.0 each, which made
#: POS_prospect 1.0 and hid the whole conditional/unconditional distinction on the
#: default demo -- every risked curve coincided with its conditional twin. These
#: multiply to 0.432, so the app opens with a POS worth reasoning about.
CHANCE_DEFAULTS = {"charge": 0.90, "trap": 1.00, "reservoir": 0.60, "retention": 0.80}

#: Play-level defaults. 1.00 throughout, because a segment assessed on its own has
#: no play risk above it -- the user opts in by lowering one.
PLAY_DEFAULTS = {el: 1.00 for el in ("charge", "trap", "reservoir", "retention")}

PLAY_HELP = {
    "charge": "Chance the play has a working source and migration system at all.",
    "trap": "Chance the play develops mapped closures and a regional seal.",
    "reservoir": "Chance the play's reservoir interval is present and of quality regionally.",
    "retention": "Chance accumulations in this play are retained rather than lost regionally.",
}
CHANCE_HELP = {
    "charge": "Chance that hydrocarbons were generated and migrated into the closure.",
    "trap": "Chance that a valid closure and an effective seal exist.",
    "reservoir": "Chance of effective reservoir presence and quality. **Exempt from the "
                 "location penalty**: a well that misses the column still saw the rock.",
    "retention": "Chance the accumulation was retained rather than lost after charge.",
}
