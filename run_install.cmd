@echo off
REM One-time setup: creates the virtual environment and installs everything.
REM PyTorch comes from the CUDA index, not PyPI, so it is installed separately.
setlocal
cd /d "%~dp0"

if not exist "venv\Scripts\python.exe" (
    echo Creating virtual environment...
    python -m venv venv || goto :fail
)

echo Installing PyTorch ^(CUDA 13.0^)...
"venv\Scripts\python.exe" -m pip install torch==2.11.0 torchvision==0.26.0 ^
    --index-url https://download.pytorch.org/whl/cu130 --timeout 180 --retries 5 || goto :fail

echo Installing the rest...
"venv\Scripts\python.exe" -m pip install -r requirements.txt || goto :fail

echo.
echo Done. Start it with run_crowd_master.cmd
pause
exit /b 0

:fail
echo.
echo Setup failed. See the output above.
pause
exit /b 1
