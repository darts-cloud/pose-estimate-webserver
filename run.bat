@echo off
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo "pythonがインストールされていません"
    exit /b 1
)

REM 仮想環境を作成
if not exist "venv" (
    python -m venv venv
    echo "仮想環境を作成しました。"
)

.\venv\Scripts\activate
pip install -r requirements.txt

echo "ライブラリのロード完了"

REM main.pyを実行
python main.py

pause