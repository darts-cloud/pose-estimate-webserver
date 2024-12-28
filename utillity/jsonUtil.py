import json
import re

def ReadJSONC(filepath:str):
    with open(filepath, 'r', encoding='utf-8') as f:      # 開く
        text = f.read()                                   # 文字列を取得
    re_text = re.sub(r'/\*[\s\S]*?\*/|//.*', '', text)    # コメントを削除
    json_obj = json.loads(re_text)                        # JSONとして解釈
    return json_obj                                       # 辞書形式を返す
