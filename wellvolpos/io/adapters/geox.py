"""Reader for SLB GeoX trial-browser exports.

The export has several traps, all of them observed in real files rather than
imagined:

* **Two header rows.** Row 3 carries the quantity name prefixed with the
  segment name (``prospect A TEST.Productive area``); row 4 carries the units;
  data starts at row 5.

* **Duplicate column names.** Six quantities appear *twice* with byte-identical
  names *and* byte-identical unit strings -- they are the in-place and
  recoverable variants and the distinguishing prefix is simply missing. They can
  only be told apart positionally, so this reader never matches on name alone
  where a duplicate exists.

* **Decimal convention.** GeoX writes ``.`` as the decimal separator regardless
  of locale, which bites anyone pasting into a comma-decimal Excel. The reader
  sniffs rather than assumes.

* **Failure sentinels.** Chance-failure trials are written with all hydrocarbon
  quantities set to exactly zero and the contact stamped with a placeholder
  above any possible crest. Detecting that is :mod:`wellvolpos.io.failure`, not
  this module -- the reader's job is to preserve the values faithfully.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from .base import TrialSet
from .source import Source

# canonical field -> ordered list of regexes tried against the cleaned header
SYNONYMS: dict[str, list[str]] = {
    "trial": [r"^trial\s*number$", r"^trial$", r"^iteration$"],
    "contact": [
        r"^hc water contact\s*-\s*result$",
        r"^(hc|petroleum|hydrocarbon)[ _-]?water contact",
        r"^contact( depth)?$",
        r"^hcwc$",
    ],
    "resource": [
        r"^recoverable\.accumulation size total resources$",
        r"^recoverable.*total resources$",
        r"^(eur|recoverable resource[s]?|total recoverable)$",
    ],
    "area": [r"^productive area$", r"^area$"],
    "gross_pay": [r"^average gross pay$", r"^gross pay$"],
    "hc_grv": [r"^hc bearing gross rock volume$", r"^hc.*gross rock volume$"],
    "hc_pv": [r"^hc pore volume$"],
    "crest": [r"^depth of crest$", r"^crest( depth)?$"],
    "spill": [r"^spill point depth$", r"^spill( depth)?$"],
    "net_gross": [r"^net/?gross( ratio)?$"],
    "porosity": [r"^porosity$"],
    "thickness": [r"^reservoir thickness$", r"^thickness$"],
}

# Quantities GeoX exports twice under one name (in-place first, recoverable second).
# Never resolved by name; see _resolve_duplicates.
KNOWN_DUPLICATE_STEMS = (
    "accumulation size dry gas",
    "accumulation size ngl",
)


def _clean_header(raw: str, prospect: str = "") -> str:
    s = str(raw).strip()
    if prospect and s.lower().startswith(prospect.lower() + "."):
        s = s[len(prospect) + 1 :]
    s = re.sub(r"^.*?\.", "", s) if "." in s and not s.lower().startswith("recoverable.") else s
    return s.strip()


def _detect_prospect(names: list[str]) -> str:
    """GeoX prefixes every column with '<segment name>.' -- recover it."""
    prefixes = [n.split(".", 1)[0] for n in names if "." in n]
    if not prefixes:
        return ""
    common = max(set(prefixes), key=prefixes.count)
    return common if prefixes.count(common) >= max(3, len(names) // 3) else ""


def _resolve_duplicates(names: list[str], frame: pd.DataFrame) -> list[str]:
    """Disambiguate GeoX's identically-named in-place / recoverable pairs.

    Positional order in the export is in-place then recoverable, and the
    in-place value is always >= the recoverable one. Both signals are checked;
    if they disagree the columns are left suffixed and flagged rather than
    guessed at.
    """
    out = list(names)
    seen: dict[str, list[int]] = {}
    for i, n in enumerate(names):
        seen.setdefault(n.lower(), []).append(i)
    for key, idx in seen.items():
        if len(idx) != 2 or not any(stem in key for stem in KNOWN_DUPLICATE_STEMS):
            continue
        a, b = idx
        va = pd.to_numeric(frame.iloc[:, a], errors="coerce")
        vb = pd.to_numeric(frame.iloc[:, b], errors="coerce")
        a_is_inplace = np.nansum(va.to_numpy()) >= np.nansum(vb.to_numpy())
        first, second = ("Inplace", "Recoverable") if a_is_inplace else ("Recoverable", "Inplace")
        out[a] = f"{first}.{names[a]}"
        out[b] = f"{second}.{names[b]}"
    return out


def _sniff_decimal(lines: list[str], sep: str) -> tuple[str, str | None]:
    """Return (decimal, thousands).

    Sniffing has to be done on individual *fields*, not the raw text: in a
    comma-separated file every separator looks like a comma decimal. A file that
    is already comma-separated cannot use a comma decimal without quoting, so
    that case is decided immediately.
    """
    if sep == ",":
        return ".", None
    fields: list[str] = []
    for line in lines:
        fields.extend(f.strip() for f in line.split(sep))
    comma_dec = sum(1 for f in fields if re.fullmatch(r"-?\d+,\d+", f))
    dot_dec = sum(1 for f in fields if re.fullmatch(r"-?\d+\.\d+", f))
    if comma_dec > dot_dec:
        return ",", "."
    return ".", None


class GeoXAdapter:
    """Reads a GeoX trial export in .xlsx, .csv, .txt or .tsv form."""

    name = "GeoX trial browser export"

    def sniff(self, path) -> float:
        src = Source.from_any(path)
        try:
            names = self._raw_header(src)
        except Exception:
            return 0.0
        if not names:
            return 0.0
        joined = " | ".join(names).lower()
        score = 0.0
        if "water contact" in joined:
            score += 0.4
        if "accumulation size" in joined:
            score += 0.3
        if "productive area" in joined:
            score += 0.2
        if any("." in n for n in names):
            score += 0.1
        return min(score, 1.0)

    # ------------------------------------------------------------------ read
    def read(self, path) -> TrialSet:
        src = Source.from_any(path)
        raw, units = self._read_raw(src)
        prospect = _detect_prospect([str(c) for c in raw.columns])
        cleaned = [_clean_header(c, prospect) for c in raw.columns]
        cleaned = _resolve_duplicates(cleaned, raw)
        raw.columns = cleaned

        mapping = self._map_columns(cleaned)
        missing = [f for f in ("contact", "resource") if f not in mapping]
        if missing:
            raise ValueError(
                "GeoX export is missing required field(s): "
                + ", ".join(missing)
                + f". Columns seen: {cleaned[:12]}{'...' if len(cleaned) > 12 else ''}"
            )

        data = {}
        for canon, src in mapping.items():
            col = pd.to_numeric(raw[src], errors="coerce")
            data[canon] = col.to_numpy()
        frame = pd.DataFrame(data)
        before = len(frame)
        frame = frame[frame["contact"].notna() & frame["resource"].notna()].reset_index(drop=True)

        notes = []
        if before != len(frame):
            # Said out loud rather than done quietly: a trial count that silently
            # differs from the one the exporter reports is the sort of discrepancy
            # that gets discovered halfway through arguing about a number.
            notes.append(
                f"Dropped {before - len(frame):,} row(s) with no contact or no resource."
            )
        if any(c.startswith(("Inplace.", "Recoverable.")) for c in cleaned):
            notes.append(
                "Resolved GeoX duplicate column names into Inplace./Recoverable. "
                "by position and magnitude."
            )
        return TrialSet(
            frame=frame,
            source_columns=mapping,
            units={k: units.get(v, "") for k, v in mapping.items()},
            source=self.name,
            prospect=prospect,
            notes=notes,
        )

    # ------------------------------------------------------------- internals
    def _raw_header(self, src: Source) -> list[str]:
        if src.is_excel:
            head = pd.read_excel(src.buffer(), header=None, nrows=8)
            row = self._header_row_index(head)
            return [str(x) for x in head.iloc[row].tolist() if str(x) != "nan"]
        text = src.lines(8)
        sep = "\t" if text and text[0].count("\t") >= text[0].count(",") else ","
        return [c.strip() for c in text[0].split(sep)] if text else []

    @staticmethod
    def _header_row_index(head: pd.DataFrame) -> int:
        """Find the row that names the quantities (GeoX pads with blank rows)."""
        for i in range(len(head)):
            vals = [str(v).lower() for v in head.iloc[i].tolist()]
            if any("water contact" in v or "trialnumber" in v for v in vals):
                return i
        return 0

    def _read_raw(self, src: Source) -> tuple[pd.DataFrame, dict[str, str]]:
        if src.is_excel:
            head = pd.read_excel(src.buffer(), header=None, nrows=8)
            hrow = self._header_row_index(head)
            names = [str(x) for x in head.iloc[hrow].tolist()]
            unit_row = head.iloc[hrow + 1].tolist() if hrow + 1 < len(head) else []
            units = {
                n: str(u)
                for n, u in zip(names, unit_row)
                if str(u) not in ("nan", "None", "")
            }
            df = pd.read_excel(src.buffer(), header=None, skiprows=hrow + 2)
            df = df.iloc[:, : len(names)]
            df.columns = names
            keep = [c for c in df.columns if str(c) != "nan"]
            return df[keep], units

        lines = src.lines()
        sep = "\t" if lines and lines[0].count("\t") >= lines[0].count(",") else ","
        decimal, thousands = _sniff_decimal(lines[1:60], sep)
        df = pd.read_csv(src.buffer(), sep=sep, decimal=decimal, thousands=thousands)
        # a units row written as the first data row (all non-numeric) is metadata
        units: dict[str, str] = {}
        if len(df) and df.iloc[0].apply(lambda v: not _is_number(v)).all():
            units = {c: str(df.iloc[0][c]) for c in df.columns}
            df = df.iloc[1:].reset_index(drop=True)
        return df, units

    @staticmethod
    def _map_columns(names: list[str]) -> dict[str, str]:
        mapping: dict[str, str] = {}
        lowered = [(n, n.lower().strip()) for n in names]
        for canon, patterns in SYNONYMS.items():
            for pat in patterns:
                hit = next((orig for orig, low in lowered if re.match(pat, low)), None)
                if hit is not None and canon not in mapping:
                    mapping[canon] = hit
                    break
        return mapping


def _is_number(v) -> bool:
    try:
        float(str(v).replace(",", "."))
        return True
    except (TypeError, ValueError):
        return False
