@echo off

if /i "%cd%"=="C:\Windows\System32" (
    color 0C
    echo The fork shouldn't be run with admin permissions.
    echo.
    pause
    exit /b 1
)

setlocal
title ShiroRVC
:: Keep packages from %APPDATA%\Python (user site) out of the env; they
:: shadow the env's own and can be stale or half-uninstalled.
set "PYTHONNOUSERSITE=1"

if not exist env (
    echo Please run 'run-install.bat' first to set up the environment.
    pause
    exit /b 1
)

:: %* forwards anything the caller added -- --language pt_BR, --share, --port.
env\python.exe app.py --open %*
echo.
pause
