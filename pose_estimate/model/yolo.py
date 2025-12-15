import cv2
from ultralytics import YOLO
from enum import *
from abc import ABC, abstractmethod
import logging
import openvino as ov
from pose_estimate.model.yolov8_processing import *
import numpy as np
import os
from queue import Queue
from collections import deque
from time import perf_counter
from quantization import *


class YoloBaseModel(ABC):

    def __init__(self, size, threshold):
        # YoloV8のロガーを無効にする
        logging.getLogger("ultralytics").disabled = True
        self._size = size  # サイズを設定
        self._threshold = threshold

class YoloModel(YoloBaseModel):

    def __init__(self, pose_model_path, size, imgsz=None, threshold=0.85, batch_size=10):
        super().__init__(size, threshold)

        # YOLOモデルのパスを取得し、モデルを初期化
        self._model = YOLO(pose_model_path)
        # self._model.to("mps")  # モデルをMPSデバイスに移動
        print(self._model.device)  # 再度デバイス情報を表示
        self._imgsz = imgsz if imgsz is not None else 640
        self._batch_size = batch_size
    
    def pose_estimation(self, frame):
        # フレームからポーズを推定
        result = self._model.predict(frame, stream=False, batch=self._batch_size, imgsz=self._imgsz, conf=self._threshold)

        keypoints = result[0].keypoints

        # キーポイントが存在しない場合は元のフレームを返す
        if len(keypoints) == 0 or keypoints.conf is None:
            return frame

        # xys = keypoints.xy[0].tolist()# キーポイントの座標をリストに変換
        plottedFrame = result[0].plot(conf=0.9, labels=False, boxes=False)

        return plottedFrame

    # yolo.py の YoloModel に追加
    def pose_estimation_batch(self, frames: list):
        # frames: list of np.ndarray
        # ultralytics はリストを受け取れて batch を渡せる
        results = self._model.predict(frames, stream=False, batch=self._batch_size, imgsz=self._imgsz, conf=self._threshold)
        out = []
        for i, r in enumerate(results):
            if len(r.keypoints) == 0 or r.keypoints.conf is None:
                out.append(frames[i])
            else:
                out.append(r.plot(conf=0.9, labels=False, boxes=False))
        return out

    def close(self):
        # self.det.close()
        pass

class YoloOpenVinoModel(YoloBaseModel): 

    def __init__(self, pose_model_path, size, imgsz=None, threshold=0.85, original_model=None, batch_size=1):
        super().__init__(size, threshold)
        self._batch_size = batch_size
        
        if original_model is not None and \
            not os.path.exists(pose_model_path):
            # YOLOモデルのパスを取得し、モデルを初期化
            self._model = YOLO(original_model)
            self._model.export(format='OpenVINO')
            # ov_model = ov.convert_model("model/yolo11n-pose.onnx")
            # ov.save_model(ov_model, model_path + "yolo11n-pose.xml")
        
        core = ov.Core()
        mdl = core.read_model(model=pose_model_path)

        # GPU用にコンパイル
        self._model = core.compile_model(model=mdl, device_name="GPU")
        self._imgsz = imgsz if imgsz is not None else 640
    
    def pose_estimation(self, frame):
        # フレームからポーズを推定（単体処理は既存の detect を利用）
        detections = self.detect(frame, self._model)[0]
        image_with_boxes = draw_results(detections, frame, label_map={0:"person"})

        return image_with_boxes
    
    def detect(self, image:np.ndarray, model:ov.Model):
        """
        YOLOv8-pose OpenVINOモデルの推論結果を辞書形式で返す
        Parameters:
            image (np.ndarray): 入力画像RGB
            model (Model): OpenVINO compiled model.
        Returns:
            detections (dict): box -  検出ボックスの[x1, y1, x2, y2, score, label]のリスト
                                kpt - 17点の[x1, y1, score1]のリスト
        """
        preprocessed_image = preprocess_image(image)
        input_tensor = image_to_tensor(preprocessed_image)
        result = model(input_tensor)
        boxes = result[model.output(0)]
        input_hw = input_tensor.shape[2:]
        detections = postprocess(pred_boxes=boxes, input_hw=input_hw, orig_img=image)
        return detections

    def pose_estimation_batch(self, frames: list):
        """
        バッチ入力で推論を行い、描画済みフレームをリストで返す（INT8モデル版）
        """
        pre_imgs = [preprocess_image(f) for f in frames]
        batch_input = np.stack(pre_imgs, axis=0)  # (N,C,H,W)
        input_tensor = images_to_tensor(batch_input)  # float32 normalized

        # 推論（バッチ）
        result = self._model(input_tensor)
        boxes = result[self._model.output(0)]
        input_hw = input_tensor.shape[2:]

        # 後処理（orig_img はリストを渡す）
        detections = postprocess(pred_boxes=boxes, input_hw=input_hw, orig_img=frames)

        # 各フレームに描画して返却
        out = []
        for i, det in enumerate(detections):
            out.append(draw_results(det, frames[i], label_map={0: "person"}))
        return out

    def close(self):
        # self.det.close()
        pass
    

class YoloOpenVinoInt8Model(YoloBaseModel): 
    def __init__(self, pose_model_path, size, imgsz=None, threshold=0.85, original_model=None, batch_size=1):
        super().__init__(size, threshold)
        self._batch_size = batch_size

        if original_model is not None and \
            not os.path.exists(pose_model_path):
            # YOLOモデルのパスを取得し、モデルを初期化
            conv = OpenVinoInt8Converter()
            conv.convert(pose_model_path, original_model)
        
        core = ov.Core()
        mdl = core.read_model(model=pose_model_path)

        # GPU用にコンパイル
        self._model = core.compile_model(model=mdl, device_name="GPU")
        self._imgsz = imgsz if imgsz is not None else 640
    
    def pose_estimation(self, frame):
        # フレームからポーズを推定（単体処理は既存の detect を利用）
        detections = self.detect(frame, self._model)[0]
        image_with_boxes = draw_results(detections, frame, label_map={0:"person"})

        return image_with_boxes

    def pose_estimation_batch(self, frames: list):
        """
        バッチ入力で推論を行い、描画済みフレームをリストで返す
        """
        # 前処理: 各フレームを (C,H,W) に変換してスタック -> (N,C,H,W)
        pre_imgs = [preprocess_image(f) for f in frames]
        batch_input = np.stack(pre_imgs, axis=0)  # (N,C,H,W)
        input_tensor = images_to_tensor(batch_input)  # float32 normalized

        # 推論（バッチ）
        result = self._model(input_tensor)
        boxes = result[self._model.output(0)]
        input_hw = input_tensor.shape[2:]

        # 後処理（orig_img はリストを渡す）
        detections = postprocess(pred_boxes=boxes, input_hw=input_hw, orig_img=frames)

        # 各フレームに描画して返却
        out = []
        for i, det in enumerate(detections):
            out.append(draw_results(det, frames[i], label_map={0: "person"}))
        return out
        
    def detect(self, image:np.ndarray, model:ov.Model):
        """
        YOLOv8-pose OpenVINOモデルの推論結果を辞書形式で返す
        Parameters:
            image (np.ndarray): 入力画像RGB
            model (Model): OpenVINO compiled model.
        Returns:
            detections (dict): box -  検出ボックスの[x1, y1, x2, y2, score, label]のリスト
                                kpt - 17点の[x1, y1, score1]のリスト
        """
        preprocessed_image = preprocess_image(image)
        input_tensor = image_to_tensor(preprocessed_image)
        result = model(input_tensor)
        boxes = result[model.output(0)]
        input_hw = input_tensor.shape[2:]
        detections = postprocess(pred_boxes=boxes, input_hw=input_hw, orig_img=image)
        return detections

    def close(self):
        # self.det.close()
        pass
