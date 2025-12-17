from ultralytics import YOLO
from datetime import datetime

model = YOLO("yolo11n-pose.pt")
model.to("cuda")

print("Current time:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
result = model.predict(
    source="0",
    imgsz=320,
    # batch=48,
    half=True,
    device=0,
    stream=False,
    save=True,
    verbose=False,
)

print("Current time:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
