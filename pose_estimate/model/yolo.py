# FrameProcessor.py
import cv2
from ultralytics import YOLO
from enum import *
from abc import ABC, abstractmethod
import logging

class YoloModel():

    def __init__(self, pose_model_path, size, imgsz=None, threshold=0.85):
        # YoloV8のロガーを無効にする
        logging.getLogger("ultralytics").disabled = True

        # YOLOモデルのパスを取得し、モデルを初期化
        self._model = YOLO(pose_model_path)
        # print(self.model.device)  # デバイス情報を表示
        self._model.to("cuda:0")  # モデルをMPSデバイスに移動
        print(self._model.device)  # 再度デバイス情報を表示
        print(pose_model_path)
        self._size = size  # サイズを設定
        self._threshold = threshold
        self._imgsz = imgsz if imgsz is not None else 640
    
    def pose_estimation(self, frame):
        # フレームからポーズを推定
        result = self._model.predict(frame, stream=False, batch=10, imgsz=self._imgsz, conf=self._threshold)

        keypoints = result[0].keypoints

        # キーポイントが存在しない場合は元のフレームを返す
        if len(keypoints) == 0 or keypoints.conf is None:
            return frame

        xys = keypoints.xy[0].tolist()# キーポイントの座標をリストに変換
        plottedFrame = result[0].plot(conf=0.9, labels=False, boxes=False)

        return plottedFrame
    