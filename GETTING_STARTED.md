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

**If you see `python is not recognized...`** → see *"I installed it but the
command isn't recognised"* below. This is the single most common snag and it is
usually a 10-second fix.

**If it opens the Microsoft Store** → a Windows stub is intercepting the
command. Fix it in *App execution aliases*, below.

---

## "I installed it, but the command isn't recognised"

`PATH` is the list of folders Windows searches when you type a command. If a
program isn't in one of those folders, Windows says *"not recognized"* even
though the program is sitting right there on your disk.

Two things cause it. Try them in this order.

### 1. Your terminal is holding a stale PATH — reopen it

**Windows reads PATH once, when a program starts.** A PowerShell window you
opened *before* installing still has the old list and will never see the new
entry, no matter how many times you retry.

**Close PowerShell completely** (the window, not just the command) and open a
new one from the folder (step 2). Try again.

This fixes it most of the time. It applies equally to Python, Git and Claude
Code.

### 2. The installer never added it

Run this to see what Windows can actually find:

```powershell
py --version
where.exe python
where.exe git
where.exe node
```

`where.exe` prints the full path if it finds the program, or *"Could not find
files"* if not.

**For Python — the shortcut.** The python.org installer always installs a
launcher called `py`, whether or not you ticked the PATH box. So if `py
--version` works but `python --version` doesn't, just use `py` for step 4:

```powershell
py -m venv .venv
```

**and everything after that is unaffected** — activating the virtual environment
puts its own folder at the front of PATH for that terminal, so `python`, `pip`,
`pytest` and `streamlit` all work normally from inside it. The PATH problem only
ever bites on that one command.

**To fix it properly:** re-run the Python installer from
<https://www.python.org/downloads/>, choose **Modify** → Next → tick **"Add
Python to environment variables"** → Install. Then reopen PowerShell.

**For Git:** reinstall from <https://git-scm.com/download/win> and keep the
default option *"Git from the command line and also from 3rd-party software"*.

**For Claude Code:** it needs Node.js first (<https://nodejs.org>, LTS version).
Install Node, **reopen PowerShell**, then `npm install -g
@anthropic-ai/claude-code`, then **reopen PowerShell again**.

### App execution aliases (the Microsoft Store trap)

If typing `python` opens the Microsoft Store, Windows has a stub intercepting
it. Turn the stub off:

**Settings → Apps → Advanced app settings → App execution aliases** → switch
**off** `python.exe` and `python3.exe`. Reopen PowerShell.

### Last resort: add the folder to PATH by hand

1. Press the Windows key, type `environment variables`, choose **"Edit the
   system environment variables"** → **Environment Variables…**
2. In the **upper** box (*User variables*), select **Path** → **Edit** →
   **New**, and paste the folder that contains the program. Typical locations:
   - Python: `C:\Users\<you>\AppData\Local\Programs\Python\Python312`
     and `...\Python312\Scripts`
   - Git: `C:\Program Files\Git\cmd`
   - npm global tools: `C:\Users\<you>\AppData\Roaming\npm`
3. OK out of all three dialogs, then **open a new PowerShell**.

To see the current list:

```powershell
$env:Path -split ';'
```

---

## Step 4 — Create a virtual environment

This keeps the project's packages separate from anything else on your machine.
Do it once.

```powershell
python -m venv .venv
```

*(If `python` isn't recognised, use `py -m venv .venv` — see the section above.)*

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
pip install -r requirements-dev.txt
```

Takes a couple of minutes. Some scrolling output is normal. A note about a newer
pip being available is not an error.

`requirements-dev.txt` installs everything: what the app needs to run, plus
pytest and `kaleido`. kaleido is the largest and the only optional one — it
lets tab ⑥ build the report from the *plotly* figures, the ones you see on
screen, by driving a headless browser. If it fails to install, or you would
rather not have it, install `requirements.txt` instead: everything still
works, and the app greys out that one option and tells you which package is
missing.

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
- `sheetA_structure_outcomes.png` and `sheetB_sweep_risk.png`
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

**Use the Claude desktop app you already have.** It includes Claude Code — no
Node.js, no CLI, nothing to install. (`SETUP_CLAUDE_CODE.bat` and
`RUN_CLAUDE.bat` in this folder are only for the terminal version; you can
ignore them.)

The desktop app has three tabs across the top:

| Tab | What it is |
|---|---|
| **Chat** | ordinary conversation, no file access |
| **Cowork** | the background agent in a sandbox — where this project was designed |
| **Code** | an interactive coding assistant with direct access to your files |

1. Click the **Code** tab at the top centre.
2. For the environment choose **Local** — Claude then works on your real files.
3. Click **Select folder** and choose
   `D:\Dokumenter\Lars\Pythonscripts\WellVolPOS`.
4. Pick a model from the dropdown next to the send button.
5. Type your instruction.

For the first message, paste this:

> Read CLAUDE.md, then start phase 1: figures A3, A4, A5 and B3 plus the depth
> sweep, wired into tabs 3 and 5. Keep the parity suite green.

**What to expect.** It starts in **Manual mode**: Claude proposes each change and
waits, showing a **diff view** of exactly what will change, with Accept and
Reject buttons. Your files are not touched until you accept. If you reject
something it asks how you would rather do it.

`CLAUDE.md` in this folder is read automatically at the start of every session —
it carries the project's decisions, the depth-axis rule, the parity constraint
and the domain traps, so Claude starts informed rather than cold. The same file
works for both the desktop app and the terminal version.

**Handy while you work:**

- **Ctrl+`** opens a terminal pane inside the app, so you can run the tests
  without leaving it.
- Type `@filename` to pull a specific file into the conversation.
- Click the `+12 -1` indicator after an edit to review the diff line by line and
  comment on it; Claude reads the comments and revises.
- Sessions run in parallel from the sidebar if you want two things at once.

Requirements: a Pro, Max, Team or Enterprise plan (clicking **Code** will say if
not), and Git for Windows, which is already installed on this machine.

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
| `python is not recognized` | Terminal opened before install, or PATH not set | Close and reopen PowerShell first; then see *"I installed it, but the command isn't recognised"* |
| `python` opens the Microsoft Store | App execution alias | Settings → Apps → Advanced app settings → App execution aliases → turn off python.exe |
| `running scripts is disabled` | Windows script policy | `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned` |
| `No module named wellvolpos` | Wrong folder, or venv not active | `cd` to the folder; check for `(.venv)` in the prompt |
| `git is not recognized` | Git not installed | <https://git-scm.com/download/win> |
| `streamlit is not recognized` | Step 5 not run, or venv not active | Activate the venv, rerun step 5 |
| Port 8501 already in use | An app is still running | `streamlit run app.py --server.port 8502` |
