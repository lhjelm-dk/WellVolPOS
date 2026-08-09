@echo off
REM Open Claude Code in this project folder. Double-click this file.
REM Calls claude.exe by full path first, so a missing PATH entry cannot stop it
REM -- the same trap that made Python look missing on this machine.
cd /d "%~dp0"

set "CLAUDE_EXE=%USERPROFILE%\.local\bin\claude.exe"
if exist "%CLAUDE_EXE%" goto run

set "CLAUDE_EXE=claude"
where claude >nul 2>&1
if not errorlevel 1 goto run

echo.
echo   Claude Code is not installed yet.
echo   Double-click SETUP_CLAUDE_CODE.bat first.
echo.
pause
exit /b 1

:run
"%CLAUDE_EXE%"
