from ultralytics import YOLO
import cv2
import helper

model = YOLO("./yolo_weights/yolov8n.pt")

source = helper.setupSource("./videos/ugur_dundar.mp4")

while True:
    success, image = source.read()
    results = model(image, stream=True)

    for result in results:
        boxes = result.boxes
        names = result.names

        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            # print(x1, y1, x2, y2)
            cv2.rectangle(
                image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 255), 1
            )

            cls = int(box.cls[0])
            conf = box.conf[0]
            classname = helper.getClassname(cls)

            cv2.putText(
                image,
                f"{str(classname)}, {conf:.2f}",
                (int(x1), int(y1)),
                cv2.FONT_HERSHEY_PLAIN,
                3,
                (255, 255, 255),
            )

    cv2.imshow("Camera", image)
    cv2.waitKey(1)
