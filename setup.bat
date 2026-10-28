@echo off
setlocal
cd /d "%~dp0"
chcp 65001 >nul

python -c "import sys" >nul 2>nul
if errorlevel 1 (
  echo Python could not be started from the command line.
  echo Install Python 3.10 or newer, then run setup.bat again.
  goto :error
)

echo Checking the Python packages already installed on this computer...
python -c "import chess, numpy, torch, requests" >nul 2>nul
if errorlevel 1 (
  echo One or more packages are missing. Installing only what N10 needs...
  python -m pip install --disable-pip-version-check -r requirements.txt || goto :error
) else (
  echo Found chess, NumPy, PyTorch, and requests. Nothing to download.
)

python -c "import chess, numpy, torch, requests" >nul 2>nul || goto :error

echo.
echo Running engine and integration self-test...
python main.py --self-test || goto :error
echo.
echo Setup complete. Run play.bat to start a physical game.
pause
exit /b 0

:error
echo.
echo Setup failed. Read the error above, then rerun setup.bat.
pause
exit /b 1
