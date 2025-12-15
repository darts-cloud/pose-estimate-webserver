import cv2
from pose_estimate.model.yolo import *
from utillity.logger import *
from tqdm import tqdm
from pose_estimate.video import *

class AnalysisVideo():
    PLUGIN_FOLDER_PATH = 'plugin/'
    @logger
    def __init__(self, infile_name, outfile_name, param):
        # パラメータをロード
        self._loadParams(param)

        print(infile_name)
        self._cap = VideoCapture(infile_name, dest:=outfile_name, 
                                 fps:=self._fps, 
                                 width:=self._width, 
                                 height:=self._height)

        ret, first_frame = self._cap.read()
        if not ret:
            raise FileNotFoundError(f'指定された入力ファイルが存在しません: {infile_name}')
        
        # 動画からFPSを取得
        self._fps = self._cap.fps
        height, width = first_frame.shape[:2]
        self._resize_flg = False
        if self._cap.width != width or self._cap.height != height:
            self._resize_flg = True

        self._size = (int(self._cap.width), int(self._cap.height))
        
        self._model = None
        if self._device in ('openvino'):
            self._model = YoloOpenVinoModel(self._pose_model_path, 
                                    self._size, 
                                    self._pose_imgsz, 
                                    self._pose_threshold,
                                    self._original_model_path,
                                    self._batch_size)
        elif self._device in ('openvino_int8'):
            self._model = YoloOpenVinoInt8Model(self._pose_model_path, 
                                    self._size, 
                                    self._pose_imgsz, 
                                    self._pose_threshold,
                                    self._original_model_path,
                                    self._batch_size)
        else:
            self._model = YoloModel(self._pose_model_path, 
                                    self._size, 
                                    self._pose_imgsz, 
                                    self._pose_threshold,
                                    self._batch_size)

        self._total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        logger.info(f"Total frames in the video: {self._total_frames}")

    def _loadParams(self, param):
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
        self._device = 'cpu'
        self._batch_size = 10
        if param is not None:
            if 'fps' in param:
                self._fps = param['fps']
            if 'resolution' in param:
                if 'width' in param['resolution']:
                    self._width = param['resolution'].get('width', None)
                if 'height' in param['resolution']:
                    self._height = param['resolution'].get('height', None)
            if 'plugins' in param:
                self._pluginPath = param["plugins"]
            if 'pose_analysis' in param:
                self._pose_enable_flg = param["pose_analysis"]["enable"]
                self._pose_model_path = param["pose_analysis"]["model"]
                self._pose_threshold = param["pose_analysis"]["threshold"]
                self._pose_imgsz = param["pose_analysis"]["imgsz"]
                if 'original_model' in param["pose_analysis"]:
                    self._original_model_path = param["pose_analysis"]["original_model"]
                if 'device' in param["pose_analysis"]:
                    self._device = param["pose_analysis"]["device"]
                if 'batch_size' in param["pose_analysis"]:
                    self._batch_size = int(param["pose_analysis"]["batch_size"])
            if 'display_video' in param:
                self._display_video_flg = param["display_video"]

    @logger
    def run(self):
        buf = []

        self._progress_bar = tqdm(total=self._total_frames, desc="Processing frames", unit="frame")

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

