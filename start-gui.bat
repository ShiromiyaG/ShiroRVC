@echo off
setlocal
title ShiroRVC

set "INSTALL_DIR=%~dp0"
set "ENV_DIR=%INSTALL_DIR%env"

if not exist "%ENV_DIR%\python.exe" (
    echo The environment was not found at "%ENV_DIR%".
    echo Run run-install.bat first.
    pause
    exit /b 1
)

cd /d "%INSTALL_DIR%"

rem pythonw has no console, so a crash before Qt starts would vanish silently.
rem Probe with the console interpreter first and only detach once we know the
rem imports resolve.
"%ENV_DIR%\python.exe" -c "import PySide6" 2>nul
if errorlevel 1 (
    echo The Qt interface needs its own dependencies.
    echo Installing them now...
    "%ENV_DIR%\python.exe" -m pip install -r "%INSTALL_DIR%gui\requirements-gui.txt" || goto :error
)

:: %* forwards anything the caller added, e.g. --language pt_BR.
start "" "%ENV_DIR%\pythonw.exe" -m gui %*
exit /b 0

:error
echo.
echo Could not install the Qt dependencies. See the output above.
pause
exit /b 1
