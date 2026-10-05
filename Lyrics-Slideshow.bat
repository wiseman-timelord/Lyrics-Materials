@echo off
setlocal EnableExtensions
cd /d "%~dp0"

:menu
rem cls
echo.
echo  ================================================================
echo       Lyrics-Slideshow - Lyrics to Music Video
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
echo     1. Launch Lyrics-Slideshow
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
if not exist "venv\Scripts\python.exe" (
  echo.
  echo  venv not found. Run option 2 first.
  pause
  goto menu
)
echo.
echo  Starting Lyrics-Slideshow...
venv\Scripts\python.exe launcher.py
goto menu

:install
if exist "venv\Scripts\python.exe" (
  venv\Scripts\python.exe installer.py
) else (
  python installer.py
)
goto menu

:end
endlocal
exit /b 0
