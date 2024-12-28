#!/bin/bash
# 仮想環境を作成
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "仮想環境を作成しました。"
fi

source venv/bin/activate
pip install -r requirements.txt

# main.pyを実行
python main.py
