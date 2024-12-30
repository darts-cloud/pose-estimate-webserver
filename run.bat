@echo off
echo "checking if python is installed..."
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo "python is not installed."
    exit /b 1
)
echo "python is installed."

REM ‰¼‘zŠÂ‹«‚ğì¬
if not exist "venv" (
    python -m venv venv
    echo "venv environment has been created."
)

call .\venv\Scripts\activate

echo "Library loading started"
rem .\venv\Scripts\python -m pip install -r requirements.txt
python -m pip install -r requirements.txt

echo "Library loading complete"

REM main.py‚ğÀs
python main.py

pause