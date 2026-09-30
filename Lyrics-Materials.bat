@echo off
setlocal EnableExtensions
cd /d "%~dp0"

:menu
rem cls
echo.
echo  ================================================================
echo       Lyrics-Materials - Batch Menu
echo  ================================================================
echo.
echo.
echo.
echo.
echo.
echo.
echo.
echo.
echo.
echo     1. Launch Lyrics-Materials
echo.
echo     2. Installation / Setup
echo.
echo.
echo.
echo.
echo.
echo.
echo.
echo.
echo.
echo  ================================================================
set /p choice=  Selection; Menu Options = 1-2, Exit Batch = X: 

if /i "%choice%"=="1" goto launch
if /i "%choice%"=="2" goto install
if /i "%choice%"=="x" goto end
goto menu

:launch
cls
echo.
echo  ================================================================
echo       Lyrics-Materials - Main Program
echo  ================================================================
echo.
echo  Starting Lyrics-Materials...
if not exist "venv\Scripts\python.exe" (
  echo.
  echo  venv not found. Run option 2 first.
  pause
  goto menu
)
venv\Scripts\python.exe launcher.py
goto menu

:install
cls
echo.
echo  ================================================================
echo       Lyrics-Materials - Program Installer
echo  ================================================================
echo.
if exist "venv\Scripts\python.exe" (
  venv\Scripts\python.exe installer.py
) else (
  python installer.py
)
goto menu

:end
cls
echo.
echo  ================================================================
echo       Lyrics-Materials - Exit Program
echo  ================================================================
echo.

endlocal
exit /b 0
