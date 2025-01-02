from glob import glob
import cv2
import nncf
import openvino as ov
from openvino.runtime import serialize
from pose_estimate.model.yolov8_processing import preprocess_image,image_to_tensor
import os

class DataLoader:
    """指定したディレクトリ内の画像を一つずつ前処理して入力テンソルとして返すイテレータ"""
    def __init__(self,dir):
        """
        Parameters:
            dir(str):画像が保存されているディレクトリパス
        """
        self.count = -1
        self.img_paths = glob(f"{dir}/*.*")
        self.max = len(self.img_paths)

    def __iter__(self):
        return self
    
    def __next__(self):
        self.count += 1
        if self.count == self.max: 
            raise StopIteration
        path = self.img_paths[self.count]
        # 画像読み込み
        input_img_rgb = cv2.imread(path)[:,:,::-1] # bgr -> rgb

        # 画像前処理
        preprocessed_image = preprocess_image(input_img_rgb)
        input_tensor = image_to_tensor(preprocessed_image)

        # 量子化のみであればラベルは不要なのでNone
        # 精度検証に用いるなら画像に応じたラベルデータを代入
        label = None 
        return input_tensor,label
    
class OpenVinoInt8Converter():
    def __init__(self):
        pass

    def transform_fn(self, data_item):
        """ラベルを除いて入力データのみを返す"""
        return data_item[0]

    def convert(self, int8_model_path, original_path):
        # 量子化したいモデルの読み込み
        core = ov.Core()
        ov_model = core.read_model(original_path)

        # データローダー作成
        data_loader= DataLoader("datasets/coco-pose/images/val2017")

        # Dataset作成
        calibration_dataset = nncf.Dataset(data_loader, self.transform_fn)

        # 量子化しない演算、名称を指定
        # ここでは後処理は量子化しないように指定
        ignored_scope = nncf.IgnoredScope(
            types=["Multiply", "Subtract", "Sigmoid"],  # 量子化しない演算
            # names=[
            #     "/model.23/dfl/conv/Conv",   # 後処理
            #     "/model.23/Add",
            #     "/model.23/Add_1",
            #     "/model.23/Add_2"
            # ]
        )

        # 量子化の実行
        quantized_pose_model = nncf.quantize(
            ov_model,
            calibration_dataset,
            preset=nncf.QuantizationPreset.MIXED,
            ignored_scope=ignored_scope
        )

        # int8_model_pathが格納されているディレクトリがなかった場合、ディレクトリを作成
        directory = os.path.dirname(int8_model_path)
        if not os.path.exists(directory):
            os.makedirs(directory)

        # モデル保存
        serialize(quantized_pose_model, int8_model_path)


