# Getting started — step by step

Windows, PowerShell. Do these in order. Each step says what you should see if it
worked, and what to do if it did not.

Total time: about 15 minutes, most of it waiting for downloads.

---

## Step 1 — Unzip

The file `WellVolPOS_phase0.zip` is in `D:\Dokumenter\Lars\Pythonscripts\`.

1. Right-click it → **Extract All…**
2. Set the destination to **`D:\Dokumenter\Lars\Pythonscripts`** (not a new
   subfolder — Windows suggests `...\WellVolPOS_phase0`, change it).
3. Extract.

The zip contains a folder called `WellVolPOS`, so it merges into your existing
one. Nothing you already have gets overwritten.

**Check:** `D:\Dokumenter\Lars\Pythonscripts\WellVolPOS\` now also contains
`app.py`, `README.md`, `requirements.txt`, and folders `wellvolpos`, `tests`,
`data`, `docs`.

---

## Step 2 — Open PowerShell in the folder

Open the `WellVolPOS` folder in File Explorer, click the address bar, type
`powershell`, press Enter. A terminal opens already in the right place.

Confirm:

```powershell
pwd
```

Should print `D:\Dokumenter\Lars\Pythonscripts\WellVolPOS`.

---

## Step 3 — Check Python

```powershell
python --version
```

**If you see `Python 3.11` or higher** → go to step 4.

**If you see nothing, an error, or a version below 3.11** → install Python:
download from <https://www.python.org/downloads/>, and **tick "Add python.exe to
PATH"** on the first screen of the installer. Close PowerShell, reopen it (step
2), and check again.

**If it opens the Microsoft Store** → that is a Windows stub. Install from
python.org as above.

---

## Step 4 — Create a virtual environment

This keeps the project's packages separate from anything else on your machine.
Do it once.

```powershell
python -m venv .venv
```

Then activate it:

```powershell
.\.venv\Scripts\Activate.ps1
```

**Check:** your prompt now starts with `(.venv)`.

**If you get "running scripts is disabled on this system"** — Windows blocks
scripts by default. Run this once, then repeat the activate command:

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

> You must activate the environment (`.\.venv\Scripts\Activate.ps1`) **every time
> you open a new PowerShell window** in this folder. The `(.venv)` prefix is how
> you know.

---

## Step 5 — Install the packages

```powershell
pip install -r requirements.txt
```

Takes a couple of minutes. Some scrolling output is normal. A note about a newer
pip being available is not an error.

---

## Step 6 — Run the tests

This is the important one. The tests are the specification: they check that the
Python code reproduces your spreadsheet's numbers exactly.

```powershell
pytest
```

**You should see:**

```
69 passed in 0.6s
```

If all 69 pass, the port is faithful to your workbook and you can trust
everything built on top of it.

**If tests fail** — copy the output and send it to me. Do not carry on; a failure
here means something is genuinely wrong.

---

## Step 7 — Run the app

```powershell
streamlit run app.py
```

Your browser opens at `http://localhost:8501`.

- The **① Data** tab shows which spreadsheet column was mapped to which quantity.
- The **② QC & Risking** tab is the one to look at. It should report 2,395
  chance failures, POS 0.7605, and an area–depth fit of R² = 1.000000.
- Tabs ③–⑥ are placeholders. Building those is phase 1 onward.

Press **Ctrl+C** in PowerShell to stop it.

---

## Step 8 — Tidy the folder

These files are superseded or belong elsewhere. I can write to your disk but not
delete, so this bit is yours. In File Explorer, in the `WellVolPOS` folder:

**Delete:**
- `WellVolPOS_Design_Plan_v2.md`
- `WellVolPOS_Design_Plan_v3.md`
- `WellVolPOS_Design_Plan.md` *(the one in the root — the current copy now lives in `docs\`)*
- `graphics_mockup.png`
- `sheetA_structure_outcomes.png` and `sheetB_sweep_risk.png` *(also now in `docs\`)*
- `~$WELL Location POS and Resources V10052017_prospect A.xlsx` *(an Excel lock file)*
- all the loose `.pdf` files in the root *(they are already copied into `Papers\`)*
- `WellVolPOS_phase0.zip` in the parent folder, once step 1 worked

**Move:**
- `WELL Location POS and Resources V10052017_prospect A.xlsx` → into the
  `reference\` folder. It is the specification, not an input; the test suite runs
  without it, and it is deliberately kept out of git.

---

## Step 9 — Check git

The repository already exists inside the folder, with the first commit made.

```powershell
git log --oneline
```

**If you see** `... Phase 0: skeleton, import, failure detection, QC gate, parity suite` → good.

**If you get "git is not recognized"** → install Git for Windows from
<https://git-scm.com/download/win>, accept all the defaults, then reopen
PowerShell (step 2) and try again.

Now check what git thinks of the folder:

```powershell
git status
```

After step 8 this should say **"nothing to commit, working tree clean"**. If a
few stray files are listed as untracked, they are leftovers from step 8 — delete
them, or tell me and I will add them to `.gitignore`.

Tag this milestone:

```powershell
git tag phase-0
```

---

## Step 10 — Open it in Claude Code

```powershell
claude
```

**If "claude is not recognized"** → install it. You need Node.js first
(<https://nodejs.org>, take the LTS version), then:

```powershell
npm install -g @anthropic-ai/claude-code
```

Reopen PowerShell, `cd` back to the folder, and run `claude`.

Once it starts, paste this as your first message:

> Read `docs/WellVolPOS_Design_Plan.md` and `README.md`. Phase 0 is complete and
> all 69 tests pass. Start phase 1: the reference grouping engine figures — A3
> (chance decomposition vs location), A4 (resource vs contact depth), A5
> (exceedance curves) and B3 (uncertainty reduction), plus the depth sweep.
> Follow the depth-axis rule in `wellvolpos/viz/theme.py` and keep the parity
> suite green.

---

## Later: putting it on GitHub

Not needed to work, but worth doing before you have much history. Your demo data
is fictional, so the repository is safe to make public whenever you want.

1. Create an empty repository at <https://github.com/new>. Name it
   `wellvolpos`. **Do not** tick "add a README" — you already have one.
2. Then, in the folder:

```powershell
git remote add origin https://github.com/YOUR-USERNAME/wellvolpos.git
git push -u origin main
git push --tags
```

The included GitHub Action runs the 69 tests on every push, so you will see a
green tick when the port is still faithful and a red cross when it is not.

---

## The short version, once set up

Every time you come back:

```powershell
cd D:\Dokumenter\Lars\Pythonscripts\WellVolPOS
.\.venv\Scripts\Activate.ps1
pytest
streamlit run app.py
```

---

## If something goes wrong

Send me the exact text of the error. The most common ones:

| Symptom | Cause | Fix |
|---|---|---|
| `python is not recognized` | Python not on PATH | Reinstall from python.org, tick "Add to PATH" |
| `running scripts is disabled` | Windows script policy | `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned` |
| `No module named wellvolpos` | Wrong folder, or venv not active | `cd` to the folder; check for `(.venv)` in the prompt |
| `git is not recognized` | Git not installed | <https://git-scm.com/download/win> |
| `streamlit is not recognized` | Step 5 not run, or venv not active | Activate the venv, rerun step 5 |
| Port 8501 already in use | An app is still running | `streamlit run app.py --server.port 8502` |
