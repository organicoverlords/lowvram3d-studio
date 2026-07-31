@echo off
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0RESUME-LAST-JOB.ps1"
set EXITCODE=%ERRORLEVEL%
echo.
if not "%EXITCODE%"=="0" echo Resume did not complete. The job checkpoint was preserved.
pause
exit /b %EXITCODE%
