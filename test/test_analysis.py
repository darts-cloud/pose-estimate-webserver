"""AnalysisVideo クラスのテストモジュール。"""

# pyright: reportMissingImports=false
# pylint: disable=use-of-assert,protected-access
import pytest

from pose_estimate.analysis import AnalysisVideo
from utillity.jsonUtil import ReadJSONC


class TestAnalysisVideo:
    """AnalysisVideo クラスのテストクラス。"""

    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        """セットアップ処理。"""
        self.param = ReadJSONC("test/file/1_cpu.jsonc")

    def test_init1(self) -> None:
        """存在しないファイルでエラーが発生することをテスト。"""
        input_file = "test/file/notexistfile.mov"
        output_file = "test/file/output.mp4"

        with pytest.raises(FileNotFoundError):
            AnalysisVideo(input_file, output_file, self.param)

    def test_init2(self) -> None:
        """正常な初期化をテスト。"""
        input_file = "test/file/input.mp4"
        output_file = "test/file/output.mp4"

        video = AnalysisVideo(input_file, output_file, self.param)

        assert video._fps == pytest.approx(60, 1.0)  # noqa: S101,SLF001
        assert video._size == (1280, 720)  # noqa: S101,SLF001

    def test_run1(self, mocker) -> None:  # noqa: ANN001
        """runメソッドの実行をテスト。"""
        input_file = "test/file/input.mp4"
        output_file = "test/file/output.mp4"

        video = AnalysisVideo(input_file, output_file, self.param)

        # モデルのcloseメソッドをモック
        mocker.patch.object(
            video._model,  # noqa: SLF001
            "close",
        )

        # モデルのpose_estimation_batchをモック
        mock_batch_result = [video._cap.read()[1] for _ in range(5)]  # noqa: SLF001
        mocker.patch.object(
            video._model,  # noqa: SLF001
            "pose_estimation_batch",
            return_value=mock_batch_result,
        )

        # runメソッドを実行
        video.run()

        # モデルのcloseが呼ばれたことを確認
        video._model.close.assert_called_once()  # noqa: SLF001

    def test_run_cpu(self) -> None:
        """runメソッドの実行をテスト。"""
        param = ReadJSONC("test/file/1_cpu.jsonc")
        input_file = "test/file/input.mp4"
        output_file = "test/file/output.mp4"

        video = AnalysisVideo(input_file, output_file, param)

        # runメソッドを実行
        video.run()

    def test_run_openvino(self) -> None:
        """runメソッドの実行をテスト。"""
        param = ReadJSONC("test/file/2_openvino.jsonc")
        input_file = "test/file/input.mp4"
        output_file = "test/file/output.mp4"

        video = AnalysisVideo(input_file, output_file, param)

        # runメソッドを実行
        video.run()

    def test_run_openvino_int8(self) -> None:
        """runメソッドの実行をテスト。"""
        param = ReadJSONC("test/file/3_openvino_int8.jsonc")
        input_file = "test/file/input.mp4"
        output_file = "test/file/output.mp4"

        video = AnalysisVideo(input_file, output_file, param)

        # runメソッドを実行
        video.run()


if __name__ == "__main__":  # pragma: no cover
    pytest.main()
