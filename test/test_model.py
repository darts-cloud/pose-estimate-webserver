"""YoloV8PoseAdapter クラスのテストファイル。"""

# pyright: reportMissingImports=false
# pylint: disable=use-of-assert,protected-access
import cv2
import numpy as np
import pytest

from pose_estimate.model.mediapipe import MediapipePoseModel
from pose_estimate.model.yolo import YoloModel


class TestYoloPoseAdapter:
    """YoloV8PoseAdapter クラスのテストクラス。"""

    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        """セットアップ処理。"""
        self.size = (720, 1280)
        self.model_path = "model/yolo11n-pose.pt"
        self.model = YoloModel(self.model_path, self.size)

    def test_pose_estimation_with_valid_frame1(self) -> None:
        """有効なフレームでポーズ推定を行うテスト"""
        cap = cv2.VideoCapture("test/file/input.mp4")
        ret, frame = cap.read()
        assert ret  # noqa: S101

        result_frame = self.model.pose_estimation(frame)
        assert result_frame is not None  # noqa: S101
        assert result_frame.shape[0:2] == self.size  # noqa: S101

    def test_pose_estimation_with_empty_frame(self) -> None:
        """空のフレームでポーズ推定を行うテスト"""
        frame = np.zeros((self.size[1], self.size[0], 3), dtype=np.uint8)
        result_frame = self.model.pose_estimation(frame)
        assert result_frame is not None  # noqa: S101

    def test_pose_estimation_with_no_keypoints(self) -> None:
        """モデルが何も検出しない場合のテスト"""
        frame = np.zeros((self.size[1], self.size[0], 3), dtype=np.uint8)
        result_frame = self.model.pose_estimation(frame)
        assert result_frame is not None  # noqa: S101


class TestPoseMediapipeAnalysis:
    """PoseMediapipeAnalysis クラスのテストクラス。"""

    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        """セットアップ処理。"""
        self.size = (1280, 720)
        self.model = MediapipePoseModel()

    def test_pose_estimation_with_valid_frame(self) -> None:
        """有効なフレームでポーズ推定を行うテスト"""
        frame = np.zeros((self.size[0], self.size[1], 3), dtype=np.uint8)
        result_frame = self.model.pose_estimation(frame)
        assert result_frame is not None  # noqa: S101
        assert result_frame.shape[0:2] == self.size  # noqa: S101

    def test_pose_estimation_with_empty_frame(self) -> None:
        """空のフレームでポーズ推定を行うテスト"""
        frame = np.zeros((self.size[1], self.size[0], 3), dtype=np.uint8)
        result_frame = self.model.pose_estimation(frame)
        assert result_frame is not None  # noqa: S101

    def test_pose_estimation_with_no_landmarks(self) -> None:
        """モデルが何も検出しない場合のテスト"""
        frame = np.zeros((self.size[1], self.size[0], 3), dtype=np.uint8)
        result_frame = self.model.pose_estimation(frame)
        assert result_frame is not None  # noqa: S101


if __name__ == "__main__":  # pragma: no cover
    pytest.main()
