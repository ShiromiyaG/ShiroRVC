@echo off
setlocal enabledelayedexpansion
title ShiroRVC TensorBoard

:: Resolve the repo root from this script's own location, so the launcher works
:: no matter what the current directory is when it runs.
set "SCRIPT_DIR=%~dp0"
pushd "%SCRIPT_DIR%.." >nul
set "REPO_ROOT=%cd%"
popd >nul

set "PORT=25565"
set "ADDRESS=http://localhost:%PORT%"

:: The folder comes from a drag-and-drop; fall back to a prompt so the script is
:: also usable by double-clicking it.
set "RAW=%~1"
if not defined RAW set /p "RAW=Key in model saved directory: "

:: Drop any quotes the user pasted around the path, then let `delims= ` trim the
:: leading blanks.  A blank or whitespace-only answer produces no token at all,
:: so LOGDIR stays undefined and is caught below.
set "LOGDIR="
if defined RAW set LOGDIR=!RAW:"=!
set "TRIMMED="
for /f "tokens=* delims= " %%A in ("!LOGDIR!") do set "TRIMMED=%%A"
set "LOGDIR=!TRIMMED!"
if not defined LOGDIR (
    echo No directory provided. Exiting.
    echo.
    pause
    exit /b 1
)
if not exist "!LOGDIR!" (
    echo Directory not found: !LOGDIR!
    echo.
    pause
    exit /b 1
)

:: The project ships its own Conda env, so a bare `tensorboard` would resolve to
:: whatever happens to be on PATH -- a different interpreter, or nothing at all.
set "PYTHON=%REPO_ROOT%\env\python.exe"
if not exist "%PYTHON%" (
    echo Environment not found at "%REPO_ROOT%\env".
    echo Please run 'run-install.bat' first to set up the environment.
    echo.
    pause
    exit /b 1
)

echo Log directory:       !LOGDIR!
echo TensorBoard address: %ADDRESS%
echo.

:: Bind on 0.0.0.0 so the board is also reachable from other machines, while the
:: browser gets a host name that always resolves.
start "ShiroRVC TensorBoard" "%PYTHON%" -m tensorboard.main --logdir="!LOGDIR!" --host=0.0.0.0 --port=%PORT%

timeout /t 3 /nobreak >nul

start "" "%ADDRESS%"
