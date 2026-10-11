@echo off
cd /d "%~dp0"
echo DeeL Pyess - Installer
echo Python 3.12 or 3.13 is required.
echo [1] Install application only
 echo [2] Install GPU learning and generation
set /p studio_mode=Choose 1 or 2 (default 1):
set studio_gpu=
if "%studio_mode%"=="2" set studio_gpu=--gpu
where py >nul 2>nul
if errorlevel 1 goto use_python
py -3.12 --version >nul 2>nul
if not errorlevel 1 (py -3.12 install.py %studio_gpu% & goto done)
py -3.13 --version >nul 2>nul
if not errorlevel 1 (py -3.13 install.py %studio_gpu% & goto done)
:use_python
python install.py %studio_gpu%
:done
if errorlevel 1 echo Installation failed. Check the error above.
pause
