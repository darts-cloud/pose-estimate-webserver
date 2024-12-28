# web.py

from flask import Flask, request, render_template, send_from_directory, redirect
from pose_estimate.analysis import *
from utillity.jsonUtil import *
import os

app = Flask(__name__, static_folder="static")

@app.route('/')
def index():

    config_folder = 'config'
    lines = [f for f in os.listdir(config_folder) if os.path.isfile(os.path.join(config_folder, f))]
    return render_template('index.html', lines=lines)  # インデックスページを表示

@app.route('/download', methods=['GET'])
def download():
    return send_from_directory(app.static_folder, "output_video.mp4")

@app.route('/analyze', methods=['POST'])
def analyze():
    video_file = request.files['video_urls']  # フォームから動画ファイルを取得
    results = []
    outputFile = 'static/output_video.mp4'  # 出力ファイル名を指定

    # 一時ファイルに保存
    temp_video_path = 'movie/temp_video.mp4'
    video_file.save(temp_video_path)  # 動画ファイルを一時保存

    print(f"config/{request.form['param']}")
    param = ReadJSONC(f"config/{request.form['param']}")
    od = AnalysisVideo(temp_video_path, outputFile, param)  # 一時ファイルのパスを渡す
    od.run()
    
    # 一時ファイルを削除する処理を追加することをお勧めします
    return redirect("/download")
    # return render_template('index.html', results=results, download_file=outputFile)  # 結果を表示し、ダウンロードを促す

# if __name__ == '__main__':
app.run(host="0.0.0.0", port=5002, debug=False)