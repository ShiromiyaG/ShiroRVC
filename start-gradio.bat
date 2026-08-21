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

if not exist env (
    echo Please run 'run-install.bat' first to set up the environment.
    pause
    exit /b 1
)

:: %* forwards anything the caller added -- --language pt_BR, --share, --port.
env\python.exe app.py --open %*
echo.
pause
