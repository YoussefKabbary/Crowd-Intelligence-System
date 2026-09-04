@echo off
setlocal
cd /d "%~dp0"

if not exist "venv\Scripts\python.exe" (
    echo ERROR: The project virtual environment was not found.
    echo Expected: %CD%\venv\Scripts\python.exe
    echo.
    echo Create it with:
    echo     python -m venv venv
    echo     venv\Scripts\python.exe -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu130
    echo     venv\Scripts\python.exe -m pip install -r requirements.txt
    pause
    exit /b 1
)

"venv\Scripts\python.exe" "crowd_master_v2.py"
set "exit_code=%ERRORLEVEL%"

if not "%exit_code%"=="0" (
    echo.
    echo Crowd Master stopped with error code %exit_code%.
    pause
)

exit /b %exit_code%
