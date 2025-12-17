# GPU/CUDA環境チェック
import torch
import ultralytics
import onnxruntime as ort
import subprocess

print("=== 環境情報 ===")
print(f"torch: {torch.__version__}")
print(f"CUDA: {torch.version.cuda}")
print(f"GPU利用可能: {torch.cuda.is_available()}")
print(f"ultralytics: {ultralytics.__version__}")
print(f"ONNXRuntime providers: {ort.get_available_providers()}")

# nvidia-smi確認
try:
    result = subprocess.check_output(["nvidia-smi"]).decode()
    print("NVIDIA-SMI:")
    print(result.splitlines()[2])  # GPUドライバー情報
except Exception as e:
    print(f"nvidia-smi not available: {e}")