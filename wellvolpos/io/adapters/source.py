"""One input, whether it came from disk or from a browser upload.

The design plan (§10) promises that *"uploads stay in memory and are never
written to disk"*. Until 2026-08-10 the app broke that promise: every adapter
took a path, so an uploaded file had to be spilled to `.streamlit_uploads/`
before it could be read. It is the licensee's data; leaving copies of it in a
working directory is not ours to decide.

:class:`Source` is the fix. It holds a **name** (for the suffix, and for error
messages) and the **bytes**, and every adapter reads from it instead of from a
path. A path is read into memory once, up front, which costs nothing at these
sizes — the largest demo export is under 3 MB — and means there is exactly one
code path for both cases rather than two that can diverge.

:meth:`Source.from_any` accepts what actually turns up: a ``str`` or ``Path``, a
Streamlit ``UploadedFile``, any file-like object with ``read``, or raw bytes. So
``read_trials("file.csv")`` still works unchanged, which is why the test suite and
every existing caller did not have to be touched.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

TEXT_SUFFIXES = (".csv", ".txt", ".tsv", ".dat")
EXCEL_SUFFIXES = (".xlsx", ".xlsm", ".xls")


@dataclass
class Source:
    """A named blob of bytes: the only thing an adapter reads."""

    name: str
    data: bytes

    @classmethod
    def from_any(cls, obj) -> "Source":
        """Normalise whatever the caller has into a :class:`Source`.

        Already-a-``Source`` passes through, so a function taking "path or
        Source" can call this unconditionally at the top and not branch again.
        """
        if isinstance(obj, Source):
            return obj
        if isinstance(obj, (str, Path)):
            p = Path(obj)
            return cls(name=p.name, data=p.read_bytes())
        # Streamlit's UploadedFile: has .name and .getvalue(), and is also a
        # BytesIO -- getvalue() first, because it does not consume the stream and
        # so survives Streamlit replaying the script on every interaction.
        name = getattr(obj, "name", "") or "uploaded"
        if hasattr(obj, "getvalue"):
            return cls(name=str(name), data=bytes(obj.getvalue()))
        if hasattr(obj, "read"):
            if hasattr(obj, "seek"):
                obj.seek(0)
            raw = obj.read()
            return cls(name=str(name), data=raw.encode("utf-8") if isinstance(raw, str) else bytes(raw))
        if isinstance(obj, (bytes, bytearray)):
            return cls(name="uploaded", data=bytes(obj))
        raise TypeError(
            f"cannot read trials from {type(obj).__name__}; expected a path, an uploaded "
            f"file, a file-like object or bytes"
        )

    # ------------------------------------------------------------------ shape
    @property
    def suffix(self) -> str:
        return Path(self.name).suffix.lower()

    @property
    def is_excel(self) -> bool:
        """Excel by suffix, or by the zip magic number a renamed .xlsx still has.

        Checked both ways because an upload's name is whatever the user's browser
        supplied, and a spreadsheet saved as `trials.txt` is a real thing.
        """
        return self.suffix in EXCEL_SUFFIXES or self.data[:2] == b"PK"

    def buffer(self) -> io.BytesIO:
        """A fresh cursor over the bytes.

        Fresh on every call, deliberately: pandas leaves the cursor where it
        stopped,
        and a reader that peeks at the header and then parses would otherwise get
        an empty frame the second time.
        """
        return io.BytesIO(self.data)

    def text(self, encoding: str = "utf-8") -> str:
        return self.data.decode(encoding, errors="replace")

    def lines(self, limit: int | None = None) -> list[str]:
        out = self.text().splitlines()
        return out if limit is None else out[:limit]
