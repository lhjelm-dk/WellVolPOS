"""Reader for any delimited trial export that is not GeoX.

The second of the two adapters the design plan (§8) asks to ship. Its job is
different from :mod:`wellvolpos.io.adapters.geox`'s in one important way: GeoX
knows what a GeoX file looks like, and this one does not know what it is holding.
So it never guesses silently. It **proposes** a column mapping with a confidence
per field and the reasoning attached, and the caller — the app, or a test — either
accepts it or overrides it. A wrong mapping is the most damaging thing that can
happen at import, because every number downstream is then computed from the wrong
column and nothing looks broken.

What it handles, all of it observed in real exports rather than imagined:

* **Delimiter**: comma, semicolon, tab or pipe, chosen by which one gives a
  consistent field count across the first lines rather than by which is commonest
  — a text column full of commas otherwise wins on frequency alone.
* **Decimal comma**, the European convention. Sniffed from the fields, never
  assumed, and the sniff has to run after the delimiter is known because in a
  comma-separated file every separator looks like a comma decimal.
* **Thousands separators**, which pandas needs told about explicitly or it reads
  ``1 234,5`` as text and the column silently becomes NaN.
* **A units row** written under the header, as GeoX does. Detected as a first data
  row with no numbers in it, and kept as metadata for
  :mod:`wellvolpos.io.units` to check.
* **Blank leading rows and title rows**, which spreadsheet exports collect.
* **Duplicate column names**, which pandas mangles to ``x``, ``x.1``. Left
  mangled and reported, never resolved by guessing — that resolution is
  GeoX-specific knowledge and belongs in that adapter.

What it deliberately does not do: convert units (see
:mod:`wellvolpos.io.units` for why), join on a trial identifier (CLAUDE.md:
``TrialNumber`` is not a reliable key in a GeoX export and there is no reason to
trust another tool's more), or invent a column it cannot find. A file without a
contact and a resource is refused with a list of what it did see.

**The mapping profile.** :func:`signature` fingerprints a file by its header, so
a caller can remember one file's mapping and offer it again for the next export
with the same columns. The profile store itself belongs to the caller — in the app
it is session state — because this module has no business writing to disk.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .base import CANONICAL_FIELDS, TrialSet
from .source import Source

DELIMITERS = (",", ";", "\t", "|")

#: Canonical field -> patterns tried against the lowercased header, best first.
#: Broader than the GeoX table on purpose: this adapter is meeting the file for
#: the first time. Each pattern is a full-match regex.
PATTERNS: dict[str, tuple[str, ...]] = {
    "trial": (r"trial\s*(number|no|id)?", r"iteration", r"sim(ulation)?\s*(no|number)?", r"run", r"#"),
    "contact": (
        r"(hc|hydrocarbon|petroleum)?\s*[-_ ]?water\s*contact.*",
        r"hcwc", r"owc", r"gwc", r"fwl", r"free\s*water\s*level",
        r"contact(\s*depth)?", r"fill\s*depth", r"(hc\s*)?column\s*base",
    ),
    "resource": (
        r"recoverable.*(total\s*)?resource.*", r"(total\s*)?recoverable\s*(resource|volume)s?.*",
        r"eur", r"accumulation\s*size.*total.*", r"reserves?", r"recoverable",
        r"volume\s*\(?mmboe\)?", r"resource.*",
    ),
    "area": (r"(productive|closure|hc|reservoir)?\s*area.*", r"a\s*\(?km2\)?"),
    "gross_pay": (r"(average|avg|mean)?\s*gross\s*pay.*", r"pay\s*(thickness)?", r"h\s*gross"),
    "hc_grv": (r"(hc[ -]?bearing\s*)?gross\s*rock\s*volume.*", r"grv.*"),
    "hc_pv": (r"(hc\s*)?pore\s*volume.*", r"pv"),
    "crest": (r"(depth\s*of\s*)?crest.*", r"top\s*(structure|reservoir)\s*depth"),
    "spill": (r"spill\s*(point)?\s*(depth)?.*",),
    "net_gross": (r"net\s*[/:_ ]?\s*gross.*", r"n\s*[/:]\s*g", r"ntg"),
    "porosity": (r"poro(sity)?.*", r"phi"),
    "thickness": (r"(reservoir|gross)?\s*thickness.*", r"t\s*\(?m\)?"),
}

#: Patterns that must never win, however well they match. ``Inplace`` volumes are
#: the trap: they read as a resource and are the wrong quantity, and a file that
#: carries both must resolve to the recoverable one.
NEGATIVE = (r".*in\s*[-_ ]?place.*", r".*stoiip.*", r".*ghiip.*", r".*giip.*")


@dataclass
class Proposal:
    """A suggested mapping, with enough attached to argue with it.

    ``confidence`` is per field in [0, 1]; ``why`` says which rule fired. Both are
    surfaced in the app so the mapping is confirmed by a person rather than
    accepted because it appeared.
    """

    mapping: dict[str, str]
    confidence: dict[str, float] = field(default_factory=dict)
    why: dict[str, str] = field(default_factory=dict)
    columns: list[str] = field(default_factory=list)
    units: dict[str, str] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    @property
    def missing_required(self) -> list[str]:
        return [f for f, (_u, req) in CANONICAL_FIELDS.items() if req and f not in self.mapping]

    @property
    def needs_confirmation(self) -> list[str]:
        """Fields matched on a weak rule. Below 0.6 means the header did not say
        so plainly and a person should look."""
        return sorted(f for f, c in self.confidence.items() if c < 0.6)


def signature(src) -> str:
    """A short fingerprint of a file's header, for remembering its mapping.

    Over the *sorted* column names, so a tool that reorders its columns between
    runs still matches. Not over the data, because the point is to recognise the
    same *kind* of export, not the same export.
    """
    import hashlib

    source = Source.from_any(src)
    names, _, _, _ = _read_header(source)
    joined = "\x1f".join(sorted(n.strip().lower() for n in names))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:12]


def _choose_delimiter(lines: list[str]) -> str:
    """The delimiter that splits the first lines into a consistent field count.

    Consistency beats frequency: a free-text column full of commas makes ``,``
    the commonest character in a semicolon-separated file, and choosing on count
    alone then produces one ragged column per row.

    Lines with *no* occurrence of the candidate are dropped before judging
    consistency, because a spreadsheet export often opens with a title row that
    contains no delimiter at all. Counting it made every candidate look
    inconsistent, and a tab-separated file with a title row was then read as
    comma-separated and refused -- which is how this was found.
    """
    best, best_score = ",", (0, 0, 0)
    for d in DELIMITERS:
        counts = [ln.count(d) for ln in lines[:25] if ln.strip()]
        counts = [c for c in counts if c > 0]
        if len(counts) < 2:
            continue
        consistent = len(set(counts)) == 1
        # (consistent, how many lines agree, fields per line) -- the last is the
        # tie-break, so a file that is consistent under two delimiters resolves to
        # the one that actually splits it into columns.
        score = (1 if consistent else 0, len(counts), min(counts))
        if score > best_score:
            best, best_score = d, score
    return best


def _sniff_decimal(fields: list[str]) -> tuple[str, str | None]:
    """(decimal, thousands), decided on whole fields rather than raw text."""
    comma = sum(1 for f in fields if re.fullmatch(r"-?\d{1,3}(\.\d{3})*,\d+", f) or
                re.fullmatch(r"-?\d+,\d+", f))
    dot = sum(1 for f in fields if re.fullmatch(r"-?\d{1,3}(,\d{3})*\.\d+", f) or
              re.fullmatch(r"-?\d+\.\d+", f))
    if comma > dot:
        return ",", "."
    return ".", ","


def _looks_numeric(v) -> bool:
    try:
        float(str(v).strip().replace(",", "."))
        return True
    except (TypeError, ValueError):
        return False


def _read_header(src: Source) -> tuple[list[str], int, str, str]:
    """(names, header row index, delimiter, decimal).

    The header is the first row that has more than one field and is not all
    numbers -- which skips the title rows and blank rows a spreadsheet export
    collects above the real table.
    """
    if src.is_excel:
        head = pd.read_excel(src.buffer(), header=None, nrows=12)
        for i in range(len(head)):
            row = [str(v) for v in head.iloc[i].tolist() if str(v) not in ("nan", "")]
            if len(row) > 1 and not all(_looks_numeric(v) for v in row):
                return row, i, "", "."
        return [], 0, "", "."

    lines = [ln for ln in src.lines(40)]
    delim = _choose_delimiter(lines)
    for i, ln in enumerate(lines):
        cells = [c.strip().strip('"') for c in ln.split(delim)]
        cells = [c for c in cells if c != ""]
        if len(cells) > 1 and not all(_looks_numeric(c) for c in cells):
            fields = [c.strip() for l2 in lines[i + 1: i + 25] for c in l2.split(delim)]
            decimal, _ = _sniff_decimal(fields)
            return cells, i, delim, decimal
    return [], 0, delim, "."


def _score(name: str, patterns: tuple[str, ...]) -> tuple[float, str]:
    """Confidence that ``name`` is this field, and the reason.

    A full match on the first pattern is near-certain; later patterns are
    progressively weaker; a substring match is weakest of all and is exactly the
    case a person should confirm.
    """
    low = " ".join(str(name).strip().lower().split())
    low = re.sub(r"\s*\[[^\]]*\]|\s*\([^)]*\)$", "", low).strip()
    if any(re.fullmatch(p, low) for p in NEGATIVE) or any(re.match(p, low) for p in NEGATIVE):
        return 0.0, "excluded: in-place volume, not recoverable"
    for i, pat in enumerate(patterns):
        if re.fullmatch(pat, low):
            return max(0.95 - 0.1 * i, 0.55), f"header matches {pat!r}"
    for i, pat in enumerate(patterns):
        if re.search(pat, low):
            return max(0.5 - 0.05 * i, 0.2), f"header contains {pat!r}"
    return 0.0, ""


def propose(src, *, mapping: dict[str, str] | None = None) -> Proposal:
    """Suggest a canonical mapping for a delimited file.

    ``mapping`` overrides the suggestion field by field, which is how a
    remembered profile or a person's correction gets applied -- an override is
    recorded at confidence 1.0 with "set by hand" as its reason, so the app can
    show which fields were chosen rather than found.
    """
    source = Source.from_any(src)
    names, _hrow, _delim, _dec = _read_header(source)
    if not names:
        raise ValueError(f"{source.name}: could not find a header row")

    out, conf, why = {}, {}, {}
    taken: set[str] = set()
    # Fields in a fixed order, most identifiable first, so that when two fields
    # both match one column the more specific one claims it.
    for canon in ("contact", "resource", "area", "gross_pay", "hc_grv", "hc_pv",
                  "thickness", "crest", "spill", "net_gross", "porosity", "trial"):
        best_name, best_conf, best_why = None, 0.0, ""
        for n in names:
            if n in taken:
                continue
            c, reason = _score(n, PATTERNS[canon])
            if c > best_conf:
                best_name, best_conf, best_why = n, c, reason
        if best_name is not None and best_conf > 0.0:
            out[canon] = best_name
            conf[canon] = best_conf
            why[canon] = best_why
            taken.add(best_name)

    for canon, col in (mapping or {}).items():
        if canon not in CANONICAL_FIELDS:
            raise ValueError(f"{canon!r} is not a canonical field; expected one of "
                             f"{sorted(CANONICAL_FIELDS)}")
        if col not in names:
            raise ValueError(f"column {col!r} is not in {source.name} (have: {names[:12]}…)")
        out[canon] = col
        conf[canon] = 1.0
        why[canon] = "set by hand"

    notes = []
    dupes = sorted({n for n in names if names.count(n) > 1})
    if dupes:
        notes.append(
            f"Duplicate column names left as pandas mangled them ({', '.join(dupes)}); this "
            f"adapter does not guess which is which."
        )
    return Proposal(mapping=out, confidence=conf, why=why, columns=list(names), notes=notes)


class GenericCsvAdapter:
    """Reads a delimited trial export with an inferred or supplied mapping.

    ``sniff`` returns a *low* confidence on purpose. It is the fallback: any file
    with a plausible contact and resource column scores enough to be read, but
    never enough to outrank a simulator-specific adapter that actually recognised
    the format.
    """

    name = "Generic delimited trial export"

    def __init__(self, mapping: dict[str, str] | None = None):
        self.mapping = dict(mapping or {})

    def sniff(self, path) -> float:
        try:
            proposal = propose(path)
        except Exception:
            return 0.0
        if proposal.missing_required:
            return 0.0
        # Capped below the GeoX adapter's minimum useful score, so a real GeoX
        # export is never read by the fallback.
        mean_conf = float(np.mean([proposal.confidence[f] for f in ("contact", "resource")]))
        return min(0.15 + 0.15 * mean_conf, 0.3)

    def read(self, path) -> TrialSet:
        src = Source.from_any(path)
        names, hrow, delim, decimal = _read_header(src)
        proposal = propose(src, mapping=self.mapping)
        if proposal.missing_required:
            raise ValueError(
                f"{src.name}: could not identify " + ", ".join(proposal.missing_required)
                + f". Columns seen: {names[:12]}{'…' if len(names) > 12 else ''}. "
                f"Supply a mapping, e.g. GenericCsvAdapter(mapping={{'contact': '<column>'}})."
            )

        if src.is_excel:
            raw = pd.read_excel(src.buffer(), header=hrow)
        else:
            thousands = "." if decimal == "," else ","
            raw = pd.read_csv(src.buffer(), sep=delim, header=hrow, decimal=decimal,
                              thousands=thousands, skip_blank_lines=True)
        raw.columns = [str(c).strip().strip('"') for c in raw.columns]

        # A units row, GeoX-style: a first data row with nothing numeric in it.
        units: dict[str, str] = {}
        if len(raw) and raw.iloc[0].apply(lambda v: not _looks_numeric(v)).all():
            units = {c: str(raw.iloc[0][c]) for c in raw.columns}
            raw = raw.iloc[1:].reset_index(drop=True)

        data = {}
        for canon, col in proposal.mapping.items():
            if col in raw.columns:
                data[canon] = pd.to_numeric(raw[col], errors="coerce").to_numpy()
        frame = pd.DataFrame(data)
        missing = [f for f in ("contact", "resource") if f not in frame.columns]
        if missing:
            raise ValueError(f"{src.name}: mapped column(s) for {missing} not present after parsing")
        before = len(frame)
        frame = frame[frame["contact"].notna() & frame["resource"].notna()].reset_index(drop=True)

        notes = list(proposal.notes)
        if delim and delim != ",":
            notes.append(f"Delimiter detected as {delim!r}.")
        if decimal == ",":
            notes.append("Decimal comma detected; parsed accordingly rather than assumed.")
        if before != len(frame):
            notes.append(
                f"Dropped {before - len(frame):,} row(s) with no contact or no resource."
            )
        weak = proposal.needs_confirmation
        if weak:
            notes.append(
                "Mapped on a weak header match and worth confirming: "
                + ", ".join(f"{f} <- {proposal.mapping[f]!r}" for f in weak)
                + "."
            )
        return TrialSet(
            frame=frame,
            source_columns=proposal.mapping,
            units={k: units.get(v, "") for k, v in proposal.mapping.items()},
            source=self.name,
            notes=notes,
        )
