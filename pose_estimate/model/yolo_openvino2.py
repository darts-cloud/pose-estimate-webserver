# FrameProcessor.py
import cv2
from ultralytics import YOLO
from enum import *
from abc import ABC, abstractmethod
import logging
import openvino as ov
from pose_estimate.model.yolov8_processing_faster import *
import numpy as np
from queue import Queue
import openvino as ov
from collections import deque
from time import perf_counter

class YoloModel():  

    def __init__(self, pose_model_path, size, imgsz=None, threshold=0.85):
        # YoloV8のロガーを無効にする
        logging.getLogger("ultralytics").disabled = True

        model_path = "model\\yolo11n-pose_int8_openvino_model\\"
        if True:
            # YOLOモデルのパスを取得し、モデルを初期化
            self._model = YOLO(pose_model_path)
            self._model.export(format='OpenVINO')
        #     # ov_model = ov.convert_model("model/yolo11n-pose.onnx")
        #     # ov.save_model(ov_model, model_path + "yolo11n-pose.xml")
        
        # self._model = YOLO(model_path)

        # core = ov.Core()
        # mdl = core.read_model(model=model_path + "yolo11n-pose.xml")
                
        # CPU用にコンパイル
        # self._model = core.compile_model(model=mdl,device_name="GPU")

        # print(self.model.device)  # デバイス情報を表示
        # self._model.to("mps")  # モデルをMPSデバイスに移動
        # print(self._model.device)  # 再度デバイス情報を表示/
        self._size = size  # サイズを設定
        self._threshold = threshold
        self._imgsz = imgsz if imgsz is not None else 640
        self.det = AsyncDetector(model_path + "yolo11n-pose-int8.xml", "GPU")
        len_deque = 100 
    
    def pose_estimation(self, frame):
        # フレームからポーズを推定
        # result = self._model.predict(frame, stream=False, batch=10, imgsz=self._imgsz, conf=self._threshold)
        # detections = self.detect(frame, self._model)[0]
        # image_with_boxes = draw_results(detections, frame, label_map={0:"person"})
        self.det.infer(frame)
        detections,image_with_boxes = self.det.pop_result()
        if detections is None:
            return None
        # keypoints = result[0].keypoints

        # # キーポイントが存在しない場合は元のフレームを返す
        # if len(keypoints) == 0 or keypoints.conf is None:
        #     return frame

        # xys = keypoints.xy[0].tolist()# キーポイントの座標をリストに変換
        # plottedFrame = result[0].plot(conf=0.9, labels=False, boxes=False)

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

    def close(self):
        self.det.close()

class AsyncDetector:
    """非同期で姿勢推定を行うクラス"""

    def __init__(self,model_path,device_name):
        """
        Params:
            model_path(str):openvinoモデルのxmlファイルパス
            device_name(str):"CPU","GPU","AUTO"のいずれか
        """
        import openvino.properties.hint as hints

        # モデルコンパイル
        core = ov.Core()
        model = core.read_model(model=model_path)
        config = {hints.performance_mode: hints.PerformanceMode.THROUGHPUT}
        self.compiled_model = core.compile_model(model=model, device_name=device_name,config=config)

        # インプットレイヤー名
        self.input_name = model.input(0).any_name

        # 推論のキュー作成
        self.infer_queue = ov.AsyncInferQueue(self.compiled_model)

        # 後処理のコールバック関数を指定
        self.infer_queue.set_callback(self.callback)

        # 推論結果を入れるキュー
        self.results_queue = Queue()
        
    def callback(self,infer_request,info):
        """推論完了後に呼ばれる関数
        Params:
            infer_request(infer_request object) 
            info(tuple of np ndarray,list): フレーム画像と[高さ、幅]
        """
        cv2_img,input_hw = info
        # 推論結果取得
        results = infer_request.get_output_tensor(0).data
        
        detection = postprocess(pred_boxes=results, input_hw=input_hw, orig_img=cv2_img)[0]
        # 後処理
        # 結果を画像に描画
        image_with_boxes = draw_results(detection, cv2_img, label_map={0:"person"})

        # キューに結果を追加
        self.results_queue.put((detection,image_with_boxes))

    def infer(self,cv2_img):
        """推論する画像を保持、バッチサイズになったら非同期の推論を開始する
        Params:
            cv2_img(np ndarray): cv2画像 BGR
        """
        input_img_rgb = cv2_img[:,:,::-1] # bgr -> rgb

        # 画像前処理
        preprocessed_image = preprocess_image(input_img_rgb)
        input_tensor = image_to_tensor(preprocessed_image)
        input_hw = input_tensor.shape[2:]
        self.infer_queue.start_async({self.input_name: input_tensor}, (cv2_img,input_hw))
        
    def pop_result(self):
        """
        Returns:
            Tuple of list(dict), np ndarray : 検出結果と結果を描画した画像
        """
        if self.results_queue.qsize()==0:
            return None,None # 結果がない場合はNone
        else:
            return self.results_queue.get()
    
    def close(self):
        # 終了処理
        while self.results_queue.qsize()!=0:# キューが埋まっていたら流す
            self.results_queue.get()
        self.infer_queue.wait_all()
        print("Detector closed")
        del self.compiled_model
