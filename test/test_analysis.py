from pose_estimate.analysis import AnalysisVideo
from plugin.darts_throw_plugin import *
from form import *
import pytest

class TestAnalysisVideo():

    def test_init1(self):
        input_file = 'test/file/notexistfile.mov'
        output_file = 'test/file/output.mov'
        
        plugins = []  # プラグインのリストを空に設定

        with pytest.raises(FileNotFoundError):
            video = AnalysisVideo(input_file, output_file, plugins)

    def test_init2(self):
        input_file = 'test/file/input.mov'
        output_file = 'test/file/output.mov'
        
        plugins = []  # プラグインのリストを空に設定
        video = AnalysisVideo(input_file, output_file, plugins)

        assert video._fps == pytest.approx(60, 1.0)
        assert video._size == (640, 360)
        assert video._plugins == plugins

    def test_run_0(self):
        # モックの設定
        input_file = 'test/file/input.mov'
        output_file = 'test/file/output.mov'

        video = AnalysisVideo(input_file, output_file, plugins)

        # runメソッドを実行
        video.run()

    def test_run_1(self):
        # モックの設定
        input_file = 'test/file/input.mov'
        output_file = 'test/file/output.mov'

        video = AnalysisVideo(input_file, output_file, plugins)

        # runメソッドを実行
        video.run()

    def test_run_2(self, mocker):
        # モックの設定
        input_file = 'test/file/input.mov'
        output_file = 'test/file/output.mov'

        plugins = [darts_throw_plugin()]  # プラグインを設定
        video = AnalysisVideo(input_file, output_file, plugins)
        
        # _process_frameメソッドがFalseを返すようにするために、モックを設定します。
        mocker.patch.object(video, '_process_frame', return_value=False)
        mocker.patch.object(video, '_merge_videos', return_value=False)

        # runメソッドを実行
        video.run()

    def test_estimate_pose1(self):
        # _estimate_poseメソッドのテスト
        input_file = 'test/file/input.mov'
        output_file = 'test/file/output.mov'

        plugins = []  # プラグインのリストを空に設定
        video = AnalysisVideo(input_file, output_file, plugins)
        video._pose_enable_flg = False

        # フレームをモック
        ret, frame = video._cap.read()
        assert ret

        # _estimate_poseメソッドを実行
        df, processed_frame = video._estimate_pose(frame)

        # 結果の確認
        assert df is None
        assert processed_frame.shape == frame.shape

    def test_estimate_pose2(self, mocker):
        # _estimate_poseメソッドのテスト
        input_file = 'test/file/input.mov'
        output_file = 'test/file/output.mov'
        
        plugins = []  # プラグインのリストを空に設定
        video = AnalysisVideo(input_file, output_file, plugins)
        video._pose_enable_flg = True
        
        # フレームをモック
        ret, frame = video._cap.read()
        assert ret

         # modelのモックを設定
        mocker.patch('pose_estimate.model.yolo.Yolo11nModel.pose_estimation', return_value=frame)
        
        # _estimate_poseメソッドを実行
        df, processed_frame = video._estimate_pose(frame)

        # 結果の確認
        assert df is not None
        assert processed_frame.shape == frame.shape

    def test_process_frame1(self):
        # _process_frameメソッドのテスト
        input_file = 'test/file/input.mov'
        output_file = 'test/file/output.mov'

        plugins = [darts_throw_plugin()]  # プラグインのリストを設定
        video = AnalysisVideo(input_file, output_file, plugins)

        # フレームをモック
        rt, frame = video._cap.read()
        video._pose_enable_flg = False
        df = DartsForm()

        # _process_frameメソッドを実行
        result = video._process_frame(frame, df)

        # 結果の確認
        assert result

    def test_process_frame2(self, mocker):
        # _process_frameメソッドのテスト
        input_file = 'test/file/input.mov'
        output_file = 'test/file/output.mov'

        plugins = [test_pligin()]  # プラグインのリストを設定
        video = AnalysisVideo(input_file, output_file, plugins)

        # フレームをモック
        rt, frame = video._cap.read()
        df = DartsForm()

        # _process_frameメソッドを実行
        result = video._process_frame(frame, df)

        # 結果の確認
        assert result
        
    def test_merge_videos1(self, mocker):
        # _merge_videosメソッドのテスト
        input_file = 'test/file/input.mov'
        output_file = 'test/file/output.mov'

        # プラグインのモックを設定
        mock_plugin = mocker.MagicMock()
        plugins = [mock_plugin]        
        video = AnalysisVideo(input_file, output_file, plugins)

        combined_frames = []
        while True:
            ret, frame = video._cap.read()
            if not ret:
                break
            combined_frames.append(frame)

        mock_plugin.merge_videos.return_value = combined_frames  # モックフレームを返す

        # _merge_videosメソッドを実行
        video._merge_videos()

        # merge_videosが呼ばれたことを確認
        mock_plugin.merge_videos.assert_called_once()
        # 出力ビデオにフレームが書き込まれたことを確認
        # self.assertTrue(mock_video_writer.write.called)
        
    def test_merge_videos2(self, mocker):
        # _merge_videosメソッドのテスト
        input_file = 'test/file/input.mov'
        output_file = 'test/file/output.mov'

        # プラグインのモックを設定
        plugins = [test_pligin()]
        video = AnalysisVideo(input_file, output_file, plugins)

        combined_frames = []
        while True:
            ret, frame = video._cap.read()
            if not ret:
                break
            combined_frames.append(frame)

        # _merge_videosメソッドを実行
        video._merge_videos()


class test_pligin(orbit_plugin):

    def __init__(self):
        super().__init__()

    def run(self, frame):
        return None

if __name__ == '__main__': # pragma: no cover
    pytest.main()