# %%
import cv2
from ultralytics import YOLO


class Algorithm_Detection:
    def detectPeople(self):
        #model = YOLO('datasets\\weights\\best.pt')
        model = YOLO('yolo-Weights\yolov8n.pt')
        cap = cv2.VideoCapture(0)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (750, 590))
            results = model.track(frame, persist=True, conf=0.5)
            frame = results[0].plot()
            cv2.imshow('Webcam Detection', frame)

            if cv2.getWindowProperty('Webcam Detection', cv2.WND_PROP_VISIBLE) < 1:
                break

        cap.release()
        cv2.destroyAllWindows()