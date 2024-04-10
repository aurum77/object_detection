# Object detection with yolo
from ultralytics import YOLO
import cv2

model = YOLO("./yolo_weights/yolov8l.pt")
results = model("./images/trafik.jpg", show=True)

cv2.waitKey(0)
