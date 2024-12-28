import cv2
import numpy as np
import pytest
from pose_estimate.model.mediapipe import *
from pose_estimate.model.yolo import *

class TestYoloV8PoseAdapter:

    @pytest.fixture(autouse=True)
    def setup(self):
        self.size = (360, 640)
        self.model = Yolo11nModel(self.size)

    def test_pose_estimation_with_valid_frame1(self):
        # テスト用のフレームを作成
        cap = cv2.VideoCapture('test/file/input.mov')
        ret, frame = cap.read()
        assert ret

        result_frame = self.model.pose_estimation(frame)
        assert result_frame is not None
        assert result_frame.shape[0:2] == self.size

    def test_pose_estimation_with_valid_frame2(self):
        # テスト用のフレームを作成
        self.model = Yolo11mModel(self.size)
        cap = cv2.VideoCapture('test/file/input.mov')
        ret, frame = cap.read()
        assert ret

        result_frame = self.model.pose_estimation(frame)
        assert result_frame is not None
        assert result_frame.shape[0:2] == self.size

    def test_pose_estimation_with_valid_frame3(self):
        # テスト用のフレームを作成
        self.model = Yolo8nModel(self.size)
        cap = cv2.VideoCapture('test/file/input.mov')
        ret, frame = cap.read()
        assert ret

        result_frame = self.model.pose_estimation(frame)
        assert result_frame is not None
        assert result_frame.shape[0:2] == self.size

    def test_pose_estimation_with_valid_frame4(self):
        # テスト用のフレームを作成
        self.model = Yolo8mModel(self.size)
        cap = cv2.VideoCapture('test/file/input.mov')
        ret, frame = cap.read()
        assert ret

        result_frame = self.model.pose_estimation(frame)
        assert result_frame is not None
        assert result_frame.shape[0:2] == self.size

    def test_pose_estimation_with_empty_frame(self):
        frame = np.zeros((self.size[1], self.size[0], 3), dtype=np.uint8)
        result_frame = self.model.pose_estimation(frame)
        assert result_frame is not None

    def test_pose_estimation_with_no_keypoints(self):
        # モデルが何も検出しない場合のテスト
        frame = np.zeros((self.size[1], self.size[0], 3), dtype=np.uint8)
        result_frame = self.model.pose_estimation(frame)
        assert result_frame is not None

class TestPoseMediapipeAnalysis:

    @pytest.fixture(autouse=True)
    def setup(self):
        self.size = (360, 640)
        self.model = MediapipePoseModel()

    def test_pose_estimation_with_valid_frame(self):
        # テスト用のフレームを作成
        frame = np.zeros((self.size[0], self.size[1], 3), dtype=np.uint8)
        result_frame = self.model.pose_estimation(frame)
        assert result_frame is not None
        assert result_frame.shape[0:2] == self.size

    def test_pose_estimation_with_empty_frame(self):
        frame = np.zeros((self.size[1], self.size[0], 3), dtype=np.uint8)
        result_frame = self.model.pose_estimation(frame)
        assert result_frame is not None

    def test_pose_estimation_with_no_landmarks(self):
        # モデルが何も検出しない場合のテスト
        frame = np.zeros((self.size[1], self.size[0], 3), dtype=np.uint8)
        result_frame = self.model.pose_estimation(frame)
        assert result_frame is not None

if __name__ == "__main__":  # pragma: no cover
    pytest.main()