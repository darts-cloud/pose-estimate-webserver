@echo off
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo "python���C���X�g�[������Ă��܂���"
    exit /b 1
)

REM ���z�����쐬
if not exist "venv" (
    python -m venv venv
    echo "���z�����쐬���܂����B"
)

.\venv\Scripts\activate
pip install -r requirements.txt

echo "���C�u�����̃��[�h����"

REM main.py�����s
python main.py

pause