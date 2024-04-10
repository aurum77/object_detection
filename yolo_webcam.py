from ultralytics import YOLO
import cv2

model = YOLO("./yolo_weights/yolov8n.pt")

camera = cv2.VideoCapture(0)
camera.set(3, 640)
camera.set(4, 480)

while True:
    success, image = camera.read()
    results = model(image, stream=True)
    for result in results:
        boxes = result.boxes

        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            print(x1, y1, x2, y2)
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 0), 1)

    cv2.imshow("Camera", image)
    cv2.waitKey(1)
