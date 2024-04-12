import helper
import cv2
import time
from ultralytics import YOLO

model = YOLO("./yolo_weights/yolov8n.pt")

kurtlarVadisi = "http://stream.tvcdn.net/dizi-youtube/kurtlar-vadisi.m3u8"

source = helper.setupSource(kurtlarVadisi)
fps = source.get(cv2.CAP_PROP_FPS)
wt = 1 / fps

while True:
    start_time = time.time()
    # Capture frame-by-frame
    ret, image = source.read()
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

    if image is not None:
        # Display the resulting frame
        cv2.namedWindow("frame", flags=cv2.WINDOW_GUI_NORMAL)
        cv2.imshow("frame", image)

        # Press q to close the video windows before it ends if you want
        if cv2.waitKey(22) & 0xFF == ord("q"):
            break
        dt = time.time() - start_time
        if wt - dt > 0:
            time.sleep(wt - dt)
    else:
        print("Frame is None")
        break

# When everything done, release the capture
source.release()
cv2.destroyAllWindows()
print("Video stop")
