from ultralytics import YOLO
import time

# YOLO11 nanoモデルの読み込み
model = YOLO('yolo11n-pose.pt')

# ベースライン性能測定（COCO8データセット使用）
results = model.predict(imgsz=640, device=0)

# 主要メトリクスの記録
# pytorch_map50 = results.box.map50
# pytorch_map = results.box.map
# print(f"PyTorch mAP50: {pytorch_map50:.4f}")
# print(f"PyTorch mAP50-95: {pytorch_map:.4f}")

# 推論時間も測定
import time
start_time = time.time()
_ = model('bus.jpg')
pytorch_inference_time = (time.time() - start_time) * 1000
print(f"PyTorch推論時間: {pytorch_inference_time:.1f}ms")

# ONNXエクスポート（FP16最適化有効）
model.export(format='onnx', imgsz=640, optimize=True, half=True)

# ONNX版での性能測定
onnx_model = YOLO('yolo11n-pose.onnx')
onnx_results = model.predict(imgsz=640, device=0)

# # 精度比較
# onnx_map50 = onnx_results.box.map50
# onnx_map = onnx_results.box.map
# print(f"ONNX mAP50: {onnx_map50:.4f} (差分: {onnx_map50-pytorch_map50:+.4f})")
# print(f"ONNX mAP50-95: {onnx_map:.4f} (差分: {onnx_map-pytorch_map:+.4f})")

# 推論時間測定
start_time = time.time()
_ = onnx_model('bus.jpg')
onnx_inference_time = (time.time() - start_time) * 1000
print(f"ONNX推論時間: {onnx_inference_time:.1f}ms (高速化率: {pytorch_inference_time/onnx_inference_time:.1f}x)")

# TensorRT変換（FP16精度）
try:
    model.export(format='engine', imgsz=640, half=True, device=0)
    
    # TensorRT版での性能測定
    trt_model = YOLO('yolo11n-pose.engine')
    trt_results = trt_model.predict(imgsz=640, device=0)
    
    # # 精度比較
    # trt_map50 = trt_results.box.map50
    # trt_map = trt_results.box.map
    # print(f"TensorRT mAP50: {trt_map50:.4f} (差分: {trt_map50-pytorch_map50:+.4f})")
    # print(f"TensorRT mAP50-95: {trt_map:.4f} (差分: {trt_map-pytorch_map:+.4f})")
    
    # 推論時間測定
    start_time = time.time()
    _ = trt_model('bus.jpg')
    trt_inference_time = (time.time() - start_time) * 1000
    print(f"TensorRT推論時間: {trt_inference_time:.1f}ms (高速化率: {pytorch_inference_time/trt_inference_time:.1f}x)")
    
except Exception as e:
    print(f"TensorRT変換に失敗しました: {e}")
