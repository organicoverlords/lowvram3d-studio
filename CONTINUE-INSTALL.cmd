@echo off
setlocal
echo Resume mode: verified completed stages will be skipped.
echo.
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0INSTALL-ONE-CLICK.ps1" -SkipPrerequisites
set EXITCODE=%ERRORLEVEL%
echo.
if not "%EXITCODE%"=="0" echo Installation failed with exit code %EXITCODE%.
pause
exit /b %EXITCODE%
