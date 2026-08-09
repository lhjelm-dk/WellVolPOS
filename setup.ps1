# WellVolPOS -- automatic setup.
#
# Run this by double-clicking SETUP.bat. It does everything in one go:
#   1. finds a usable Python, or installs one
#   2. creates the project's private environment
#   3. installs the packages
#   4. runs the test suite
#   5. offers to start the app
#
# The one design decision worth knowing: every command below calls Python by its
# full path. Nothing here depends on PATH being set correctly, and nothing needs
# the environment to be "activated" -- which is exactly the problem this script
# exists to route around.

$ErrorActionPreference = 'Continue'
$ProgressPreference    = 'SilentlyContinue'   # makes downloads far faster
Set-Location -LiteralPath $PSScriptRoot

$MinMajor = 3
$MinMinor = 11
$PyVersion = '3.13.15'                        # used only if nothing is installed
$PyUrl = "https://www.python.org/ftp/python/$PyVersion/python-$PyVersion-amd64.exe"

function Say  ($m) { Write-Host $m }
function Good ($m) { Write-Host "  OK    $m" -ForegroundColor Green }
function Warn ($m) { Write-Host "  note  $m" -ForegroundColor Yellow }
function Bad  ($m) { Write-Host "  FAIL  $m" -ForegroundColor Red }
function Rule     { Write-Host ("-" * 68) -ForegroundColor DarkGray }

Say ""
Say "=================================================================="
Say "  WellVolPOS setup"
Say "=================================================================="
Say ""

# --------------------------------------------------------------- find Python
function Test-PythonExe {
    param([string]$Exe)
    if (-not $Exe) { return $null }
    # Never touch anything in WindowsApps. Those entries are Microsoft Store
    # execution aliases, and running one opens the Store rather than Python --
    # which would leave this script hanging on a shopfront.
    if ($Exe -like '*\WindowsApps\*') { return $null }
    # Belt and braces: an alias is a zero-length reparse point.
    try {
        $item = Get-Item -LiteralPath $Exe -ErrorAction Stop
        if ($item.Length -eq 0) { return $null }
    } catch { return $null }

    try {
        $out = & $Exe -c "import sys; print(sys.version_info[0], sys.version_info[1])" 2>$null
    } catch { return $null }
    if ($LASTEXITCODE -ne 0 -or -not $out) { return $null }

    $parts = ($out -split '\s+')
    if ($parts.Count -lt 2) { return $null }
    $maj = [int]$parts[0]; $min = [int]$parts[1]
    if ($maj -lt $MinMajor -or ($maj -eq $MinMajor -and $min -lt $MinMinor)) { return $null }
    return [pscustomobject]@{ Exe = $Exe; Version = "$maj.$min" }
}

function Find-Python {
    $found = @()

    # the py launcher, which the python.org installer always provides
    $pyLauncher = Get-Command py -ErrorAction SilentlyContinue
    if ($pyLauncher) {
        try {
            $real = & py -3 -c "import sys; print(sys.executable)" 2>$null
            if ($LASTEXITCODE -eq 0 -and $real) { $found += $real.Trim() }
        } catch { }
    }

    # anything named python on PATH, ignoring the Store stub folder
    Get-Command python -All -ErrorAction SilentlyContinue |
        Where-Object { $_.Source -and $_.Source -notlike '*\WindowsApps\*' } |
        ForEach-Object { $found += $_.Source }

    # the usual install locations, in case PATH was never set
    $globs = @(
        "$env:LOCALAPPDATA\Programs\Python\Python3*\python.exe",
        "$env:ProgramFiles\Python3*\python.exe",
        "${env:ProgramFiles(x86)}\Python3*\python.exe",
        "C:\Python3*\python.exe",
        # Anaconda and Miniconda: very common, install nowhere near the above,
        # and deliberately keep themselves off PATH unless you use their own
        # prompt -- which is exactly why 'python' appears to be missing.
        "$env:USERPROFILE\anaconda3\python.exe",
        "$env:USERPROFILE\miniconda3\python.exe",
        "$env:USERPROFILE\Anaconda3\python.exe",
        "$env:LOCALAPPDATA\anaconda3\python.exe",
        "$env:LOCALAPPDATA\Continuum\anaconda3\python.exe",
        "C:\ProgramData\anaconda3\python.exe",
        "C:\ProgramData\miniconda3\python.exe"
    )
    foreach ($g in $globs) {
        Get-ChildItem -Path $g -ErrorAction SilentlyContinue |
            Sort-Object FullName -Descending |
            ForEach-Object { $found += $_.FullName }
    }

    foreach ($exe in ($found | Select-Object -Unique)) {
        $ok = Test-PythonExe $exe
        if ($ok) { return $ok }
    }
    return $null
}

$venvPy = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'

Say "[1/5] Looking for the project environment..."
$venv = Test-PythonExe $venvPy
if ($venv) {
    Good "the .venv environment already exists and works (Python $($venv.Version))"
    Say  "        nothing to install; skipping straight to the packages"
    $py = $venv
    $haveVenv = $true
} else {
    $haveVenv = $false
    Say "      none yet. Looking for Python $MinMajor.$MinMinor or newer..."
    $py = Find-Python
}

if ($haveVenv) {
    # nothing to do
} elseif ($py) {
    Good "found Python $($py.Version)"
    Say  "        $($py.Exe)"
} else {
    Warn "no usable Python found (a Microsoft Store placeholder does not count)"
    Say  ""
    Say  "        Installing Python $PyVersion for your user account."
    Say  "        No administrator rights needed. This takes a few minutes."
    Say  ""

    $installed = $false

    # winget first: it handles versions and mirrors for us
    if (Get-Command winget -ErrorAction SilentlyContinue) {
        Say "        Trying winget..."
        foreach ($id in @('Python.Python.3.13', 'Python.Python.3.12')) {
            & winget install --id $id --exact --source winget `
                --accept-package-agreements --accept-source-agreements --silent 2>&1 |
                Out-String | Write-Host
            if (Find-Python) { $installed = $true; break }
        }
    }

    # fall back to the official installer
    if (-not $installed) {
        Say "        Downloading from python.org..."
        $exe = Join-Path $env:TEMP "python-$PyVersion-amd64.exe"
        try {
            Invoke-WebRequest -Uri $PyUrl -OutFile $exe -UseBasicParsing -ErrorAction Stop
            Say "        Installing (a progress window may appear)..."
            $p = Start-Process -FilePath $exe -Wait -PassThru -ArgumentList @(
                '/quiet', 'InstallAllUsers=0', 'PrependPath=1',
                'Include_launcher=1', 'Include_test=0', 'AssociateFiles=0'
            )
            if ($p.ExitCode -ne 0) { Warn "installer exit code $($p.ExitCode)" }
            Remove-Item $exe -ErrorAction SilentlyContinue
        } catch {
            Bad "could not download or run the installer: $($_.Exception.Message)"
        }
    }

    $py = Find-Python
    if (-not $py) {
        Say ""
        Bad "Python still not found after installing."
        Say ""
        Say "  Please install it by hand:"
        Say "    1. Go to  https://www.python.org/downloads/"
        Say "    2. Download Python $PyVersion and run the installer"
        Say "    3. TICK 'Add python.exe to PATH' on the first screen"
        Say "    4. Run this script again"
        Say ""
        Read-Host "  Press Enter to close"
        exit 1
    }
    Good "installed Python $($py.Version)"
}

# ------------------------------------------------------------- the environment
Rule
Say "[2/5] The project's private environment (.venv)..."

if ($haveVenv) {
    Good "reusing the existing one"
} else {
    & $py.Exe -m venv .venv
    if (-not (Test-Path $venvPy)) {
        Bad "could not create .venv"
        Read-Host "  Press Enter to close"
        exit 1
    }
    Good "created"
}

# ---------------------------------------------------------------- the packages
Rule
Say "[3/5] Installing packages. First run takes a few minutes..."
Say ""

& $venvPy -m pip install --upgrade pip --quiet
& $venvPy -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) {
    Say ""
    Bad "package installation failed. Send the message above to Claude."
    Read-Host "  Press Enter to close"
    exit 1
}
Good "packages installed"

# -------------------------------------------------------------------- the tests
Rule
Say "[4/5] Running the test suite."
Say ""
Say "      These check that the Python reproduces your spreadsheet's numbers."
Say "      You want to see: 69 passed"
Say ""

& $venvPy -m pytest
$testsPassed = ($LASTEXITCODE -eq 0)

Say ""
if ($testsPassed) {
    Good "all tests passed - the port is faithful to your workbook"
} else {
    Bad "some tests failed. Copy the output above and send it to Claude."
}

# ---------------------------------------------------------------------- the app
Rule
Say "[5/5] Done."
Say ""
Say "      From now on you can just double-click:"
Say "        RUN_APP.bat     to start the app"
Say "        RUN_TESTS.bat   to re-run the tests"
Say ""

$answer = Read-Host "      Start the app now? (y/n)"
if ($answer -match '^(y|yes)') {
    Say ""
    Say "      Your browser will open. Press Ctrl+C here to stop the app."
    Say ""
    & $venvPy -m streamlit run app.py
} else {
    Read-Host "      Press Enter to close"
}
