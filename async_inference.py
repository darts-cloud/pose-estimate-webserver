from collections import deque
from time import perf_counter

import cv2

from pose_estimate.model.yolo_openvino import AsyncDetector
 
# 非同期インスタンス作成
# det = AsyncDetector("model/yolov8s-pose_openvino_model/yolov8s-pose.xml","GPU")
det = AsyncDetector("model/yolov8s-pose_int8_openvino_model/yolov8s-pose-int8.xml","GPU")

cap = cv2.VideoCapture("movie/temp_video.mp4")

# FPS測定用deque
len_deque = 100 
t_deque = deque([],maxlen=len_deque)

t_start = perf_counter()
count = 0

while True:
    # フレーム読み取り
    ret,frame = cap.read()
    if not ret:
        break

    # 推論リクエスト 
    det.infer(frame[40:680,400:1040])

    # (過去のフレームに対する)推論結果取得
    detections,image_with_boxes = det.pop_result()

    # 初めの数ループは推論が終わっていないのでNoneが返る
    if detections is None:
        continue 

    # FPS描画
    t_deque.append(perf_counter())
    if len(t_deque)==len_deque:
        cv2.putText(image_with_boxes, 
                    f'{(len_deque-1)/(t_deque[-1]-t_deque[0]):.1f}FPS',
                        (0, 50), cv2.FONT_HERSHEY_PLAIN, 3,
                    (255, 0, 0), 3, cv2.LINE_AA)
    count +=1

    # # 表示
    # cv2.imshow('Async Result' ,image_with_boxes)

    # # Escキーで中断
    # key =cv2.waitKey(1)
    # if key == 27:
    #     break

# 推論速度
t_total = perf_counter()-t_start
print(f"{t_total/count*1000:.1f}msec/loop, {1/(t_total/count):.2f}FPS")

# 終了処理
det.close()

