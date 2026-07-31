@echo off
setlocal
echo Retrying only optional degraded stages; verified core stages remain skipped.
echo.
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0INSTALL-ONE-CLICK.ps1" -SkipPrerequisites -RetryOptional
set EXITCODE=%ERRORLEVEL%
echo.
if not "%EXITCODE%"=="0" echo Optional retry finished with exit code %EXITCODE%.
pause
exit /b %EXITCODE%
