@echo off
setlocal enabledelayedexpansion
title ShiroRVC Installer

echo Welcome to the ShiroRVC Installer!
echo.

set "INSTALL_DIR=%cd%"
set "MINICONDA_DIR=%UserProfile%\Miniconda3"
set "ENV_DIR=%INSTALL_DIR%\env"
set "MINICONDA_URL=https://repo.anaconda.com/miniconda/Miniconda3-py312_26.5.3-2-Windows-x86_64.exe"
set "CONDA_EXE=%MINICONDA_DIR%\Scripts\conda.exe"
set "PYTHON_VERSION=3.12"
set "TORCH_VERSION=2.13.0"
set "TORCHVISION_VERSION=0.28.0"
set "TORCHAUDIO_VERSION=2.11.0"
set "PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cu130"

set "startTime=%TIME%"
set "startHour=%TIME:~0,2%"
set "startMin=%TIME:~3,2%"
set "startSec=%TIME:~6,2%"
set /a startHour=1%startHour% - 100
set /a startMin=1%startMin% - 100
set /a startSec=1%startSec% - 100
set /a startTotal = startHour*3600 + startMin*60 + startSec

call :cleanup
call :install_miniconda
call :create_conda_env
call :install_dependencies

set "endTime=%TIME%"
set "endHour=%TIME:~0,2%"
set "endMin=%TIME:~3,2%"
set "endSec=%TIME:~6,2%"
set /a endHour=1%endHour% - 100
set /a endMin=1%endMin% - 100
set /a endSec=1%endSec% - 100
set /a endTotal = endHour*3600 + endMin*60 + endSec
set /a elapsed = endTotal - startTotal
if %elapsed% lss 0 set /a elapsed += 86400
set /a hours = elapsed / 3600
set /a minutes = (elapsed %% 3600) / 60
set /a seconds = elapsed %% 60

echo Installation time: %hours% hours, %minutes% minutes, %seconds% seconds.
echo.

echo ShiroRVC has been installed successfully!
echo To start ShiroRVC, please run 'start.bat'.
echo.
pause
exit /b 0

:cleanup
echo Cleaning up unnecessary files...
:: `*.sh` is deliberately not in this list: run-install.sh / start.sh are the
:: Linux counterparts of these launchers and must survive a Windows install.
for %%F in (Makefile Dockerfile docker-compose.yaml) do if exist "%%F" del "%%F"
echo Cleanup complete.
echo.
exit /b 0

:install_miniconda
if exist "%CONDA_EXE%" (
    echo Miniconda already installed. Skipping installation.
    exit /b 0
)

echo Miniconda not found. Starting download and installation...
powershell -Command "& {Invoke-WebRequest -Uri '%MINICONDA_URL%' -OutFile 'miniconda.exe'}"
if not exist "miniconda.exe" goto :download_error

start /wait "" miniconda.exe /InstallationType=JustMe /RegisterPython=0 /S /D=%MINICONDA_DIR%
if errorlevel 1 goto :install_error

del miniconda.exe
echo Miniconda installation complete.
echo.
exit /b 0

:create_conda_env
echo Creating Conda environment...
call "%MINICONDA_DIR%\_conda.exe" create --no-shortcuts -y -k --prefix "%ENV_DIR%" python=%PYTHON_VERSION%
if errorlevel 1 goto :error
echo Conda environment created successfully.
echo.

if exist "%ENV_DIR%\python.exe" (
    echo Installing uv package installer...
    "%ENV_DIR%\python.exe" -m pip install uv
    if errorlevel 1 goto :error
    echo uv installation complete.
    echo.
)
exit /b 0

:install_dependencies
echo Installing dependencies...
call "%MINICONDA_DIR%\condabin\conda.bat" activate "%ENV_DIR%" || goto :error

echo Installing pip packages...
uv pip install --upgrade setuptools || goto :error
uv pip install torch==%TORCH_VERSION% torchvision==%TORCHVISION_VERSION% torchaudio==%TORCHAUDIO_VERSION% --upgrade --index-url %PYTORCH_INDEX_URL% || goto :error
uv pip install -r "%INSTALL_DIR%\requirements.txt" || goto :error

:: Translation catalogs: gettext falls back to English without raising
:: when a .mo is missing, so a skipped build looks like working software.
"%ENV_DIR%\python.exe" "%INSTALL_DIR%\tools\i18n_tool.py" compile

call "%MINICONDA_DIR%\condabin\conda.bat" deactivate
echo Dependencies installation complete.
echo.
exit /b 0


:download_error
echo Download failed. Please check your internet connection and try again.
goto :error

:install_error
echo Miniconda installation failed.
goto :error

:error
echo An error occurred during installation. Please check the output above for details.
pause
exit /b 1
