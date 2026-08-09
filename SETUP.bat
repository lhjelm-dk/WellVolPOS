@echo off
REM WellVolPOS - one-click setup. Double-click this file.
REM Runs setup.ps1 with the execution policy bypassed for this one process only,
REM so Windows does not need its script policy changed permanently.
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup.ps1"
if errorlevel 1 pause
