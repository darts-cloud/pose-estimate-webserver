@echo off

echo "checking if python is installed..."
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo "python is not installed."
    exit /b 1
)
echo "python is installed."

if not exist "venv" (
    python -m venv venv
    echo "venv environment has been created."
)

call .\venv\Scripts\activate
echo "Library loading started"

pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cu121

echo "Library loading complete"

rem cd ../
python main.py

pause