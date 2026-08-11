"""Unit validation on import: reject, never convert.

The design plan (§6.4, §8) requires that *"the importer rejects a file whose units
do not match rather than silently converting"*. This module is that check, and it
was missing until 2026-08-10 — units were parsed into ``TrialSet.units`` and then
never looked at, so a file in feet or acres imported silently as metres and square
kilometres.

**Why reject rather than convert.** A conversion factor is a guess about what the
number means. Get it wrong and every figure is wrong by a constant, which is the
hardest kind of error to notice — the shapes all still look right. Refusing the
file puts the decision back where it belongs, with the person who exported it.

Two independent checks, because on real data neither is sufficient alone:

**Declared units.** Compared against :data:`CANONICAL_FIELDS` through an alias
table. A recognised *other* unit is a hard fail and the message names the factor
without applying it. An unrecognised string is a warning: it may be a spelling
nobody anticipated, and refusing a file over a spelling would be worse than
saying so.

**Magnitude plausibility**, which matters more in practice than the above,
because **neither demo CSV carries a unit row at all** — a 7-column paste out of
Excel has no room for one. With nothing declared, the only evidence is the numbers
themselves. Only two of the possible unit errors are actually detectable that way,
and this module claims exactly those two rather than pretending to more:

* **Depths in feet.** 3 500 m is 11 483 ft, so a file in feet reads as an
  implausible metre depth. Detectable, and worth catching.
* **Fractions written as percentages.** Net/gross of 74 rather than 0.74.
  Detectable, and a common export mistake.

Not detectable, and stated so nobody assumes otherwise: **area** in acres (3.2 km²
is 790 acres, and a 790 km² prospect is unusual but possible); **pay** in feet
(45 m is 148 ft, still a plausible thickness in metres); **resource** in any
volume unit, since MMboe has no characteristic scale. For those, an undeclared
file is taken at its word, and the QC line says which units were assumed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .adapters.base import CANONICAL_FIELDS, TrialSet

#: Accepted spellings per canonical unit, lowercased and whitespace-stripped
#: before matching. Deliberately generous: the point of the check is to catch a
#: file in the wrong *unit*, not a file in the wrong *typography*.
ALIASES: dict[str, tuple[str, ...]] = {
    "m TVDSS": (
        "m", "m tvdss", "mtvdss", "m tvd ss", "metre", "metres", "meter", "meters",
        "m ss", "mss", "m below msl", "m tvdss (m)",
    ),
    "m": ("m", "metre", "metres", "meter", "meters"),
    "MMboe": (
        "mmboe", "mm boe", "mmboe (mmboe)", "10^6 boe", "1e6 boe", "million boe",
        "mboe",              # SLB writes MMboe as Mboe in some locales
        # GeoX's own string for total resources, seen in prospect B's export. A
        # stock-tank barrel of oil equivalent *is* a boe, so 1e6 STB OE is MMboe
        # under another name -- and it must not be confused with "1e6 STB", which
        # is oil only and appears in the same file two columns away.
        "1e6 stb oe", "10^6 stb oe", "mmstb oe", "mmboe oe", "1e6 stboe",
    ),
    "km2": ("km2", "km^2", "km²", "sq km", "square km", "square kilometre", "km2 (km2)"),
    "1e6 m3": (
        "1e6 m3", "10^6 m3", "10^6 m³", "mm3", "million m3", "e6m3", "1e6m3", "m3*10^6",
    ),
    # GeoX writes fractions as "decimal"; already covered below but listed here so
    # the two real exports in data/ are both fully recognised.
    "fraction": ("fraction", "frac", "-", "ratio", "v/v", "dec", "decimal"),
}

#: Units that are *recognised and wrong*: the message names the factor so the
#: reader can fix the export, and the factor is never applied here.
KNOWN_OTHER: dict[str, dict[str, str]] = {
    "m TVDSS": {
        "ft": "feet (1 m = 3.2808 ft)", "feet": "feet (1 m = 3.2808 ft)",
        "ft tvdss": "feet (1 m = 3.2808 ft)", "f": "feet (1 m = 3.2808 ft)",
    },
    "m": {"ft": "feet (1 m = 3.2808 ft)", "feet": "feet (1 m = 3.2808 ft)"},
    "MMboe": {
        "mmbbl": "million barrels of oil, not oil-equivalent",
        "mmstb": "million stock-tank barrels, not oil-equivalent",
        "mstb": "thousand stock-tank barrels, not oil-equivalent",
        "bcf": "billion cubic feet of gas, not oil-equivalent",
        "mmscf": "million standard cubic feet of gas, not oil-equivalent",
        "tcf": "trillion cubic feet of gas, not oil-equivalent",
        "mmm3": "million cubic metres, not oil-equivalent",
    },
    "km2": {
        "acre": "acres (1 km² = 247.1 acres)", "acres": "acres (1 km² = 247.1 acres)",
        "ac": "acres (1 km² = 247.1 acres)",
        "m2": "square metres (1 km² = 1e6 m²)", "m^2": "square metres (1 km² = 1e6 m²)",
        "ha": "hectares (1 km² = 100 ha)", "hectare": "hectares (1 km² = 100 ha)",
        "sq mi": "square miles (1 km² = 0.3861 sq mi)",
    },
    "1e6 m3": {"bbl": "barrels", "acre-ft": "acre-feet", "acre ft": "acre-feet"},
    "fraction": {"%": "per cent, not a fraction", "percent": "per cent, not a fraction",
                 "pct": "per cent, not a fraction"},
}

#: A metre depth this large is not a prospect on Earth; it is a foot depth
#: mislabelled. 9 000 m TVDSS is already past the deepest well ever drilled, and
#: 3 500 m in feet reads as 11 483 -- so the two populations do not overlap.
IMPLAUSIBLE_DEPTH_M = 9_000.0


@dataclass
class UnitFinding:
    """One field's verdict. ``level`` is the QC vocabulary: pass / warn / fail."""

    field: str
    declared: str
    expected: str
    level: str
    message: str


def normalise(unit: str) -> str:
    return " ".join(str(unit).strip().lower().split())


def check_declared(ts: TrialSet) -> list[UnitFinding]:
    """Compare each field's declared unit against the canonical one.

    Fields with no declared unit are skipped entirely rather than warned about --
    :func:`check_plausibility` is what covers those, and warning twice about one
    absence would make the QC report noisier without saying anything new.
    """
    out: list[UnitFinding] = []
    for fname, (expected, _required) in CANONICAL_FIELDS.items():
        if not expected or fname not in ts.frame.columns:
            continue
        declared = normalise(ts.units.get(fname, ""))
        if not declared or declared in ("nan", "none", "-") and expected != "fraction":
            continue
        if declared in ALIASES.get(expected, ()):
            out.append(UnitFinding(fname, declared, expected, "pass",
                                   f"{fname}: {declared!r} matches {expected}."))
            continue
        wrong = KNOWN_OTHER.get(expected, {}).get(declared)
        if wrong is not None:
            out.append(UnitFinding(
                fname, declared, expected, "fail",
                f"{fname} is declared in {declared!r} — {wrong} — but this tool works in "
                f"{expected} only. Re-export in {expected}; nothing is converted here, because "
                f"a wrong factor makes every figure wrong by a constant and the shapes still "
                f"look right.",
            ))
            continue
        out.append(UnitFinding(
            fname, declared, expected, "warn",
            f"{fname} declares unit {declared!r}, which is not a spelling of {expected} that "
            f"this build recognises. It is being read as {expected} — check that it is.",
        ))
    return out


def check_plausibility(ts: TrialSet) -> list[UnitFinding]:
    """The two unit errors that are detectable from the numbers alone.

    See the module docstring for the ones that are not, and why claiming them
    would be dishonest rather than merely imprecise.
    """
    out: list[UnitFinding] = []
    for fname in ("contact", "crest", "spill"):
        if fname not in ts.frame.columns:
            continue
        v = np.asarray(ts.col(fname), dtype=float)
        v = v[np.isfinite(v) & (v != 0.0)]
        if v.size == 0:
            continue
        median = float(np.median(np.abs(v)))
        if median > IMPLAUSIBLE_DEPTH_M:
            out.append(UnitFinding(
                fname, ts.units.get(fname, "") or "(none declared)", "m TVDSS", "fail",
                f"{fname} has a median of {median:,.0f}, which is not a plausible depth in "
                f"metres — past the deepest well ever drilled. In feet it would be "
                f"{median / 3.2808:,.0f} m. Re-export in metres; nothing is converted here.",
            ))
    for fname in ("net_gross", "porosity"):
        if fname not in ts.frame.columns:
            continue
        v = np.asarray(ts.col(fname), dtype=float)
        v = v[np.isfinite(v)]
        if v.size == 0:
            continue
        top = float(np.nanmax(v))
        if top > 1.0:
            out.append(UnitFinding(
                fname, ts.units.get(fname, "") or "(none declared)", "fraction", "fail",
                f"{fname} reaches {top:,.3g}, so it is a percentage rather than a fraction. "
                f"Re-export as a fraction; nothing is divided by 100 here.",
            ))
    return out


def verdict(ts: TrialSet) -> tuple[str, str]:
    """One QC line: (level, message).

    Reports what was checked *and what was assumed*, because on a file with no
    unit row the honest statement is not "units pass" but "units were not
    declared, so these were assumed and here is why that is survivable".
    """
    findings = check_declared(ts) + check_plausibility(ts)
    failures = [f for f in findings if f.level == "fail"]
    if failures:
        return "fail", " ".join(f.message for f in failures)

    warnings = [f for f in findings if f.level == "warn"]
    declared = [f.field for f in findings if f.level == "pass"]
    undeclared = [
        f for f, (unit, _) in CANONICAL_FIELDS.items()
        if unit and f in ts.frame.columns and not normalise(ts.units.get(f, ""))
    ]
    if warnings:
        return "warn", " ".join(f.message for f in warnings)
    if declared and not undeclared:
        return "pass", f"Units confirmed against the file for {', '.join(declared)}."
    if declared:
        return "pass", (
            f"Units confirmed for {', '.join(declared)}; not declared for "
            f"{', '.join(undeclared)}, and taken as MMboe / m / km²."
        )
    return "pass", (
        f"This file declares no units, so MMboe, m and km² are assumed for all "
        f"{len(undeclared)} quantities. Depths and fractions were sanity-checked against those "
        f"units and are consistent; area, pay and resource cannot be checked from magnitude "
        f"alone, so they are taken at their word."
    )
