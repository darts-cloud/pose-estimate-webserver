#!/bin/bash

# 仮想環境を作成
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "仮想環境を作成しました。"
fi

source venv/bin/activate

pip install -r ../requirements.txt
pip install pytest

pytest test_main.py -p no:warnings
pytest test_analysis.py -p no:warnings
pytest test_pose_adapter.py -p no:warnings

