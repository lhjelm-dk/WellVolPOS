@echo off
REM Install Claude Code. Double-click this file.
REM
REM One line, no Node.js, no npm, no administrator rights. The native installer
REM places claude.exe under your user profile and keeps itself updated.
cd /d "%~dp0"
echo.
echo ==================================================================
echo   Installing Claude Code
echo ==================================================================
echo.
powershell -NoProfile -ExecutionPolicy Bypass -Command "irm https://claude.ai/install.ps1 | iex"
echo.
if exist "%USERPROFILE%\.local\bin\claude.exe" goto ok
echo   Could not confirm the install. If there is an error above, send it to Claude.
echo   To install by hand: open PowerShell and run
echo       irm https://claude.ai/install.ps1 ^| iex
goto end

:ok
echo   OK - Claude Code is installed.
echo.
echo   Next: double-click RUN_CLAUDE.bat to open it in this project.

:end
echo.
pause
