# Reference material (not tracked)

Put `WELL Location POS and Resources V10052017_prospect A.xlsx` here.

It is the **specification** for this port, not an input to it. The fifteen
numbers derived from it are hard-coded in `tests/test_excel_parity.py`, so the
test suite runs without the workbook present — which is why `.gitignore`
excludes `*.xlsx` from this folder. Keep the file here anyway so the derivation
can be re-checked by hand.
