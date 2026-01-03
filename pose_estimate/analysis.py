import cv2
from tqdm import tqdm

from pose_estimate.model.yolo import YoloModel, YoloOpenVinoInt8Model, YoloOpenVinoModel
from pose_estimate.video import VideoCapture
from utillity.logger import logger


class AnalysisVideo:
    """動画に対してポーズ推定分析を実行するクラス。"""

    PLUGIN_FOLDER_PATH = "plugin/"

    @logger
    def __init__(self, infile_name: str, outfile_name: str, param: dict | None) -> None:
        """初期化処理。

        Args:
            infile_name: 入力動画ファイルのパス
            outfile_name: 出力動画ファイルのパス
            param: 設定パラメータの辞書

        """
        # パラメータをロード
        self._load_params(param)

        print(infile_name)
        self._cap = VideoCapture(
            infile_name,
            dest := outfile_name,
            fps := self._fps,
            width := self._width,
            height := self._height,
        )

        ret, first_frame = self._cap.read()
        if not ret:
            msg = f"指定された入力ファイルが存在しません: {infile_name}"
            raise FileNotFoundError(msg)

        # 動画からFPSを取得
        self._fps = self._cap.fps
        height, width = first_frame.shape[:2]
        self._resize_flg = False
        if self._cap.width != width or self._cap.height != height:
            self._resize_flg = True

        self._size = (int(self._cap.width), int(self._cap.height))

        self._model = None
        if self._device in ("openvino"):
            self._model = YoloOpenVinoModel(
                self._pose_model_path,
                self._size,
                self._pose_imgsz,
                self._pose_threshold,
                self._original_model_path,
                self._batch_size,
            )
        elif self._device in ("openvino_int8"):
            self._model = YoloOpenVinoInt8Model(
                self._pose_model_path,
                self._size,
                self._pose_imgsz,
                self._pose_threshold,
                self._original_model_path,
                self._batch_size,
            )
        else:
            self._model = YoloModel(
                self._pose_model_path,
                self._size,
                self._pose_imgsz,
                self._pose_threshold,
                self._batch_size,
            )

        self._total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        msg = f"Total frames in the video: {self._total_frames}"
        logger.info(msg)

    def _init_default_params(self) -> None:
        """デフォルトパラメータを初期化する。"""
        self._fps = None
        self._width = None
        self._height = None
        self._pluginPath = None
        self._pose_enable_flg = False
        self._pose_model_path = None
        self._pose_threshold = None
        self._pose_imgsz = None
        self._display_video_flg = None
        self._original_model_path = None
        self._device = "cpu"
        self._batch_size = 10

    def _load_basic_params(self, param: dict | None) -> None:
        """基本パラメータを読み込む。"""
        if "fps" in param:
            self._fps = param["fps"]
        if "resolution" in param:
            if "width" in param["resolution"]:
                self._width = param["resolution"].get("width", None)
            if "height" in param["resolution"]:
                self._height = param["resolution"].get("height", None)
        if "plugins" in param:
            self._pluginPath = param["plugins"]
        if "display_video" in param:
            self._display_video_flg = param["display_video"]

    def _load_pose_analysis_params(self, pose_analysis: dict | None) -> None:
        """ポーズ分析パラメータを読み込む。"""
        self._pose_enable_flg = pose_analysis["enable"]
        self._pose_model_path = pose_analysis["model"]
        self._pose_threshold = pose_analysis["threshold"]
        self._pose_imgsz = pose_analysis["imgsz"]
        if "original_model" in pose_analysis:
            self._original_model_path = pose_analysis["original_model"]
        if "device" in pose_analysis:
            self._device = pose_analysis["device"]
        if "batch_size" in pose_analysis:
            self._batch_size = int(pose_analysis["batch_size"])

    def _load_params(self, param: dict | None) -> None:
        """パラメータを読み込んでインスタンス変数に設定する。"""
        self._init_default_params()
        if param is not None:
            self._load_basic_params(param)
            if "pose_analysis" in param:
                self._load_pose_analysis_params(param["pose_analysis"])

    @logger
    def run(self) -> None:
        """ポーズ推定分析を実行する。"""
        buf = []

        self._progress_bar = tqdm(
            total=self._total_frames,
            desc="Processing frames",
            unit="frame",
        )

        while True:
            ret, frame = self._cap.read()
            if ret:
                if self._resize_flg:
                    frame = cv2.resize(frame, self._size)
                buf.append(frame)

            if len(buf) >= self._batch_size or (not ret and len(buf) > 0):
                out_frames = self._model.pose_estimation_batch(buf)
                for of in out_frames:
                    self._cap.write(of)
                    self._progress_bar.update(1)
                buf = []

            if not ret:
                break

        self._model.close()
        self._cap.release()
        cv2.destroyAllWindows()
