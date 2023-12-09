# %%
import cv2
from ultralytics import YOLO


class Algorithm_Detection:
    def detectPeople(self):

        model=YOLO('datasets\weights\\best.pt')

        # start webcam
        cap = cv2.VideoCapture(0)
        cap.set(3, 640)
        cap.set(4, 480)

        while True:
            ret, frame = cap.read()
            if ret:
                results = model.track(frame, persist=True, conf=0.5)

                frame_ = results[0].plot()

                cv2.imshow('Frame', frame_)
                if cv2.waitKey(int(25/3)) & 0xFF == ord('q'):
                    break
            else: break

        cap.release()
        cv2.destroyAllWindows()