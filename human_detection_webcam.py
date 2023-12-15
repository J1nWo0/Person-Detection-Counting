# %%
import cv2
import cvzone
from ultralytics import YOLO
from human_counting import Color 

class Algorithm_Detection:
    def detectPeople(self):
        #color = Color()
        #model = YOLO('datasets\\weights\\best.pt')
        model = YOLO('yolo-Weights\yolov8n.pt')
        cap = cv2.VideoCapture(0)
        cap.set(3, 1040)
        cap.set(4, 680)
        while True:
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.resize(frame, (1040, 680))
            results = model.track(frame, persist=True, conf=0.5, classes=[0])
            frame = results[0].plot()
            cvzone.putTextRect(frame,str('Pres [q] to quit'), (30,30), 1,1, Color().text1(), Color().text2())
            cv2.imshow('Webcam', frame)

            if cv2.waitKey(1) == ord('q'): #cv2.getWindowProperty('Webcam', cv2.WND_PROP_VISIBLE) < 1:
                break

        cap.release()
        cv2.destroyAllWindows()