@echo off
setlocal
cd /d "%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\register-local-github-worker.ps1"
if errorlevel 1 (
  echo.
  echo Worker registration failed. See the message above.
  pause
  exit /b 1
)
echo.
echo Worker registration finished. The queued shaman run should start automatically.
pause
