import cv2
import numpy as np

def draw_fan(frame, center, radius, angle_start, angle_end, color=(255, 0, 0), thickness=-1):
    """
    指定された中心、半径、角度範囲で扇を描画します。
    
    :param frame: 描画対象のフレーム
    :param center: 扇の中心 (x, y)
    :param radius: 扇の半径
    :param angle_start: 扇の開始角度
    :param angle_end: 扇の終了角度
    :param color: 扇の色
    :param thickness: 扇の厚さ
    """
    # 扇の外側の円を描画
    cv2.ellipse(frame, center, (radius, radius), 1, angle_start, angle_end, color, thickness)

# 使用例
# フレームを取得した後に以下のコードを追加
frame = np.zeros((480, 640, 3), dtype=np.uint8)  # 黒いフレームを作成
draw_fan(frame, (320, 240), 100, -1, -30)  # 中心(320, 240)に半径100の扇を描画
cv2.imshow("Fan", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
