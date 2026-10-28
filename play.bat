@echo off
setlocal
cd /d "%~dp0"
chcp 65001 >nul

python -c "import chess, numpy, torch, requests" >nul 2>nul
if errorlevel 1 (
  echo N10 needs some Python packages. Running setup...
  call setup.bat || exit /b 1
)

python main.py %*
pause
