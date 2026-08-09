@echo off
REM Start the WellVolPOS app. Double-click this file.
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Setup has not been run yet. Double-click SETUP.bat first.
  pause
  exit /b 1
)
echo Starting WellVolPOS. Your browser will open shortly.
echo Press Ctrl+C in this window to stop it.
echo.
".venv\Scripts\python.exe" -m streamlit run app.py
pause
