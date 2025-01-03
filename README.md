# WEBサーバ＋姿勢推定アプリ

このアプリは、ユーザーが動画をアップロードし、姿勢推定を行うためのツールです。  
アプリケーションを起動するとWEBサーバが立ち上がります。  
ユーザーが動画をアップロードすると、分析が始まります。  
しばらくすると、分析結果がダウンロードできます。

## インストール

1. Gitをインストール（Windowsのみ）  
   https://gitforwindows.org/  
   すでにインストール済みの場合、不要。  

3. pythonをインストール（Windowsのみ）  
   2024年現在3.13は不安定であるため、3.12以下のインストールを推奨。  
   すでにインストール済みの場合、不要。  
   https://www.python.org/downloads/windows/
4. リポジトリをクローンします。  
   ```bash
   git clone https://github.com/darts-cloud/pose-estimate-webserver.git
   cd pose-estimate-webserver
   ```
5. WindowsのみVC＋のライブラリをインストール（Windowsのみ）  
https://aka.ms/vs/17/release/vc_redist.x64.exe

## 実行方法

アプリを実行するには、以下のコマンドを使用します。
```
(windowsの場合)
.¥run.bat

(mac, linuxの場合)
./run.sh
```

## 使用方法
http://(実行マシンのIPアドレス):5002/
でアクセスできます。

![image](https://github.com/user-attachments/assets/0e074e82-dc4a-41ed-99b0-140eb88cd73c)

動画ファイルを選択：姿勢推定を行いたい動画を選びます。  
設定ファイルを選択：設定ファイルを選択します。効果は以下。  

|設定ファイル|解析速度|解析レベル|
|----|----|----|
|1_low_quority.jsonc|速い|低|
|2_middle_quority.jsonc|やや遅い|中|
|3_high_quority.jsonc|遅い|高|

設定ファイルの実体はconfigフォルダ以下に配置してある

## トラブルシューティング
### Pythonをインストールしているのにpython is not installed.が表示される。
Pythonをインストールしているのにpython is not installed.が表示される。  
または、Microsoft Storeが開く場合はこちら  
https://loumo.jp/archives/26344


## パフォーマンス
### yolo11n-pose.pt
|CPU/GPU|モデル|FPS|
|----|----|----|
|Intel(R) Core(TM) i5-4300M CPU|yolo11n-pose.pt|10fps|
|Intel(R) Core(TM) i5-7300U CPU|yolo11n-pose.pt|12fps|
|Intel(R) Core(TM) i5-7300U CPU|yolo11n-pose.pt -> openvino|15.42fps|
|11th Gen Intel(R) Core(TM) i5-1135G7 CPU|yolo11n-pose.pt|20fps|
|11th Gen Intel(R) Core(TM) i5-1135G7 CPU|yolo11n-pose.pt -> openvino|31fps|
|11th Gen Intel(R) Core(TM) i5-1135G7 CPU|yolo11n-pose.pt -> openvino(int8)|38fps|
|Apple M1 Pro|yolo11n-pose.pt|44fps|

### yolo11x-pose.pt
|CPU/GPU|モデル|FPS|
|----|----|----|
|Intel(R) Core(TM) i5-4300M CPU|yolo11x-pose.pt|1.67fps|
|Intel(R) Core(TM) i5-7300U CPU|yolo11x-pose.pt|1.55fps|
|Intel(R) Core(TM) i5-7300U CPU|yolo11x-pose.pt -> openvino|1.62fps|
|11th Gen Intel(R) Core(TM) i5-1135G7 CPU|yolo11x-pose.pt|1.40fps|
|11th Gen Intel(R) Core(TM) i5-1135G7 CPU|yolo11x-pose.pt -> openvino|2.84fps|
|11th Gen Intel(R) Core(TM) i5-1135G7 CPU|yolo11x-pose.pt -> openvino(int8)|9.24fps|
|Apple M1 Pro|yolo11x-pose.pt|4.00fps|
