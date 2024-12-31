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
cd venv
Scripts\pip install -r ../requirements.txt
rem Scripts\pip install flask
rem python -m pip install -r requirements.txt

echo "Library loading complete"

cd ../
python main.py

pause