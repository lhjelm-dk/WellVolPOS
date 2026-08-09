@echo off
REM Re-run the test suite. Double-click this file. You want: 69 passed
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Setup has not been run yet. Double-click SETUP.bat first.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" -m pytest
echo.
pause
