import logging
from abc import ABC
from enum import *
from pathlib import Path

import numpy as np
import openvino as ov
from ultralytics import YOLO

from pose_estimate.model.yolov8_processing import *
from quantization import *


class YoloBaseModel(ABC):
    """YOLOベースモデルクラス。"""

    def __init__(self, size: tuple[int, int], threshold: float):
        """初期化処理。"""
        logging.getLogger("ultralytics").disabled = True
        self._size = size  # サイズを設定
        self._threshold = threshold

    def pose_estimation(self, frame: np.ndarray) -> np.ndarray:
        """フレームからポーズを推定し、描画済みフレームを返す"""
        result = self._model.predict(
            frame,
            stream=False,
            batch=self._batch_size,
            imgsz=self._imgsz,
            conf=self._threshold,
        )

        keypoints = result[0].keypoints

        # キーポイントが存在しない場合は元のフレームを返す
        if len(keypoints) == 0 or keypoints.conf is None:
            return frame

        # xys = keypoints.xy[0].tolist()# キーポイントの座標をリストに変換
        plotted_frame = result[0].plot(conf=0.9, labels=False, boxes=False)

        return plotted_frame

    # yolo.py の YoloModel に追加
    def pose_estimation_batch(self, frames: list[np.ndarray]) -> list[np.ndarray]:
        """バッチ入力で推論を行い、描画済みフレームをリストで返す"""
        results = self._model.predict(
            frames,
            stream=False,
            batch=self._batch_size,
            imgsz=self._imgsz,
            conf=self._threshold,
        )
        out = []
        for i, r in enumerate(results):
            if len(r.keypoints) == 0 or r.keypoints.conf is None:
                out.append(frames[i])
            else:
                out.append(r.plot(conf=0.9, labels=False, boxes=False))
        return out

    def close(self) -> None:
        """モデルをクローズする。"""
        self._model = None


class YoloModel(YoloBaseModel):
    """YOLOモデルクラス。"""

    def __init__(
        self,
        pose_model_path: str,
        size: tuple[int, int],
        imgsz: int | None = None,
        threshold: float = 0.85,
        batch_size: int = 10,
    ) -> None:
        """初期化処理。"""
        super().__init__(size, threshold)

        # YOLOモデルのパスを取得し、モデルを初期化
        self._model = YOLO(pose_model_path)
        # self._model.to("mps")  # モデルをMPSデバイスに移動
        print(self._model.device)  # 再度デバイス情報を表示
        self._imgsz = imgsz if imgsz is not None else 640
        self._batch_size = batch_size


class YoloOpenVinoModel(YoloBaseModel):
    """YOLO OpenVINOモデルクラス。"""

    def __init__(  # noqa: PLR0913
        self,
        pose_model_path: str,
        size: tuple[int, int],
        imgsz: int | None = None,
        threshold: float = 0.85,
        original_model: str | None = None,
        batch_size: int = 1,
    ) -> None:
        """初期化処理。

        Args:
            pose_model_path: ポーズモデルのパス
            size: 画像サイズ (width, height)
            imgsz: 入力画像サイズ
            threshold: 検出閾値
            original_model: 元のモデルパス(変換が必要な場合)
            batch_size: バッチサイズ

        """
        super().__init__(size, threshold)
        self._batch_size = batch_size

        if original_model is not None and not Path(pose_model_path).exists():
            # YOLOモデルのパスを取得し、モデルを初期化
            self._model = YOLO(original_model)
            self._model.export(format="OpenVINO")
            # エクスポートされたONNXモデルをOpenVINO形式に変換
            onnx_path = original_model.replace(".pt", ".onnx")
            if Path(onnx_path).exists():
                ov_model = ov.convert_model(onnx_path)
                ov.save_model(ov_model, pose_model_path)

        core = ov.Core()
        mdl = core.read_model(model=pose_model_path)

        # 利用可能なデバイスを自動検出(GPU優先、なければCPU)
        available_devices = core.available_devices
        device_name = "GPU" if "GPU" in available_devices else "CPU"

        # デバイスにコンパイル
        self._model = core.compile_model(model=mdl, device_name=device_name)
        self._imgsz = imgsz if imgsz is not None else 640

    def pose_estimation(self, frame: np.ndarray) -> np.ndarray:
        """フレームからポーズを推定し、描画済みフレームを返す。

        Args:
            frame: 入力フレーム

        Returns:
            描画済みフレーム

        """
        detections = self.detect(frame, self._model)[0]
        image_with_boxes = draw_results(detections, frame, label_map={0: "person"})

        return image_with_boxes

    def detect(self, image: np.ndarray, model: ov.Model) -> dict:
        """YOLOv8-pose OpenVINOモデルの推論結果を辞書形式で返す。

        Args:
            image: 入力画像RGB
            model: OpenVINO compiled model

        Returns:
            detections: box - 検出ボックスの[x1, y1, x2, y2, score, label]のリスト
                      kpt - 17点の[x1, y1, score1]のリスト

        """
        preprocessed_image = preprocess_image(image)
        input_tensor = image_to_tensor(preprocessed_image)
        result = model(input_tensor)
        boxes = result[model.output(0)]
        input_hw = input_tensor.shape[2:]
        detections = postprocess(pred_boxes=boxes, input_hw=input_hw, orig_img=image)
        return detections

    def pose_estimation_batch(self, frames: list[np.ndarray]) -> list[np.ndarray]:
        """バッチ入力で推論を行い、描画済みフレームをリストで返す。

        Args:
            frames: 入力フレームのリスト

        Returns:
            描画済みフレームのリスト

        """
        # OpenVINOモデルはバッチサイズ1でコンパイルされているため、フレームを1つずつ処理
        out = []
        for frame in frames:
            detections = self.detect(frame, self._model)[0]
            image_with_boxes = draw_results(detections, frame, label_map={0: "person"})
            out.append(image_with_boxes)
        return out

    def close(self) -> None:
        """モデルをクローズする。"""
        self._model = None


class YoloOpenVinoInt8Model(YoloBaseModel):
    """YOLO OpenVINO INT8モデルクラス。"""

    def __init__(  # noqa: PLR0913
        self,
        pose_model_path: str,
        size: tuple[int, int],
        imgsz: int | None = None,
        threshold: float = 0.85,
        original_model: str | None = None,
        batch_size: int = 1,
    ) -> None:
        """初期化処理。"""
        super().__init__(size, threshold)
        self._batch_size = batch_size

        if original_model is not None and not Path(pose_model_path).exists():
            # YOLOモデルのパスを取得し、モデルを初期化
            try:
                conv = OpenVinoInt8Converter()
                conv.convert(pose_model_path, original_model)
            except Exception as e:
                # 変換に失敗した場合、既存のモデルが存在するか確認
                if Path(pose_model_path).exists():
                    logging.warning(
                        "INT8モデルの変換に失敗しましたが、既存のモデルを使用します: %s",
                        e,
                    )
                else:
                    msg = (
                        f"INT8モデルの変換に失敗しました: {e}\n"
                        f"モデルパス: {pose_model_path}\n"
                        f"元のモデル: {original_model}\n"
                        "既存のINT8モデルを使用するか、変換環境を確認してください。"
                    )
                    raise RuntimeError(msg) from e

        if not Path(pose_model_path).exists():
            msg = f"INT8モデルが見つかりません: {pose_model_path}"
            raise FileNotFoundError(msg)

        core = ov.Core()
        mdl = core.read_model(model=pose_model_path)

        # 利用可能なデバイスを自動検出(GPU優先、なければCPU)
        available_devices = core.available_devices
        device_name = "GPU" if "GPU" in available_devices else "CPU"

        # デバイスにコンパイル
        self._model = core.compile_model(model=mdl, device_name=device_name)
        self._imgsz = imgsz if imgsz is not None else 640

    def pose_estimation(self, frame: np.ndarray) -> np.ndarray:
        """フレームからポーズを推定し、描画済みフレームを返す。

        Args:
            frame: 入力フレーム

        Returns:
            描画済みフレーム

        """
        detections = self.detect(frame, self._model)[0]
        image_with_boxes = draw_results(detections, frame, label_map={0: "person"})

        return image_with_boxes

    def pose_estimation_batch(self, frames: list[np.ndarray]) -> list[np.ndarray]:
        """バッチ入力で推論を行い、描画済みフレームをリストで返す。

        Args:
            frames: 入力フレームのリスト

        Returns:
            描画済みフレームのリスト

        """
        # OpenVINOモデルはバッチサイズ1でコンパイルされているため、フレームを1つずつ処理
        out = []
        for frame in frames:
            detections = self.detect(frame, self._model)[0]
            image_with_boxes = draw_results(detections, frame, label_map={0: "person"})
            out.append(image_with_boxes)
        return out

    def detect(self, image: np.ndarray, model: ov.Model) -> dict:
        """YOLOv8-pose OpenVINOモデルの推論結果を辞書形式で返す

        Parameters
        ----------
            image (np.ndarray): 入力画像RGB
            model (Model): OpenVINO compiled model.

        Returns
        -------
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
