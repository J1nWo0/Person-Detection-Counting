# %%
import os
import cv2
import cvzone
import time
import numpy as np
from ultralytics import YOLO
from tracker import *

model=YOLO('datasets\\weights\\best.pt')
#model = YOLO('yolo-Weights\yolov8s.pt')

class Color:
    def boundingBox1(self):
        green = (0,255,0)
        return green
    def boundingBox2(self):
        yellow = (0,255,255)
        return yellow
    def text1(self):
        white = (255,255,255)
        return white
    def text2(self):
        black = (0,0,0)
        return black
    def area1(self):
        blue = (255,0,0)
        return blue
    def area2(self):
        red = (0, 0, 255)
        return red
    def point(self):
        pink = (255,0,255)
        return pink 
    def center_point(self):
        cyan = (255,255,0)
        return cyan
    def rectangle(self):
        orange = (0,119,255)
        return orange

color = Color()
tracker = Tracker()
    
class Algorithm_Count:
    def __init__(self, a1, a2):
        self.people_entering = {}
        self.entering = set()
        self.people_exiting = {}
        self.exiting = set()
        self.area1 = a1
        self.area2 = a2
        self.paused = False
        self.coordinates = []
        self.start_time = time.time()

        cv2.namedWindow('Frame')

    def center_point(self, a, b):
        c = int((a+b)//2)
        return c

    def detect(self, frame):
        results = model(frame, conf=0.6, classes=[0])
        for result in results:
            detections = []
            for r in result.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = r
                detections.append([int(x1), int(y1), int(x2), int(y2), float(score)])
        return detections
    
    def draw_boxes(self, frame, detections):
        list = []
        for box in detections:
            if len(box) == 5:
                x1, y1, x2, y2, score = box
                class_id = 0  # Default class ID for a person
            elif len(box) == 6:
                x1, y1, x2, y2, score, class_id = box
            else: continue  # Skip boxes with unexpected length

            if class_id == 0:  # Assuming person class is 0
                label1 = f"Person: {score:.2f}"
                list.append([x1, y1, x2, y2])
                
        bbox_id = tracker.update(list)
        for bbox in bbox_id:
            x3, y3, x4, y4, id = bbox

            label2 = f"{id} Person: {score:.2f}"

            if id != -1:
                cv2.rectangle(frame, (x3, y3), (x4, y4), color.rectangle(), 2)
                #cvzone.putTextRect(frame, label2, (x3+10, y3-10), 1,1, color.text1(), color.text2()) 

            # People going right
            result_p1 = cv2.pointPolygonTest(np.array(self.area2,np.int32), ((x4,y4)), False)
            if result_p1 >= 0:
                self.people_entering[id] = (x4, y4) 
                cv2.rectangle(frame, (x3, y3), (x4, y4), color.boundingBox2(), 2)
                cvzone.putTextRect(frame, label2, (x3+10, y3-10), 1,1, color.text1(), color.text2()) 
                #cv2.putText(frame, label2, (x3, y3 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color.text1(), 2)
            if id in self.people_entering:
                result_p2 = cv2.pointPolygonTest(np.array(self.area1,np.int32), ((x4,y4)), False)
                if result_p2 >= 0:
                    cv2.rectangle(frame, (x3, y3), (x4, y4), color.boundingBox1(), 2)
                    cv2.circle(frame, (x4, y4), 4, color.point(), -1)  
                    #cv2.putText(frame, label2, (x3, y3 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color.text1(), 2)
                    cvzone.putTextRect(frame, label2, (x3+10, y3-10), 1,1, color.text1(), color.text2())
                    self.entering.add(id)
            # People going left
            result_p3 = cv2.pointPolygonTest(np.array(self.area1,np.int32), ((x4,y4)), False)
            if result_p3 >= 0:
                self.people_exiting[id] = (x4, y4) 
                cv2.rectangle(frame, (x3, y3), (x4, y4), color.boundingBox1(), 2)
                cvzone.putTextRect(frame, label2, (x3+10, y3-10), 1,1, color.text1(), color.text2()) 
            if id in self.people_exiting:
                result_p4 = cv2.pointPolygonTest(np.array(self.area2,np.int32), ((x4,y4)), False)
                if result_p4 >= 0:
                    cv2.rectangle(frame, (x3, y3), (x4, y4), color.boundingBox2(), 2)
                    cv2.circle(frame, (x4, y4), 4, color.point(), -1)  
                    #cv2.putText(frame, label2, (x3, y3 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color.text1(), 2)
                    cvzone.putTextRect(frame, label2, (x3+10, y3-10), 1,1, color.text1(), color.text2())
                    self.exiting.add(id)

        cv2.polylines(frame,[np.array(self.area1,np.int32)],True,color.area1(),2)
        #cvzone.putTextRect(frame,str('1'), (self.area1[3][0]+5, self.area1[3][1]+2), 1,1, color.text1(), color.text2())

        cv2.polylines(frame,[np.array(self.area2,np.int32)],True,color.area2(),2)
        #cvzone.putTextRect(frame,str('2'), (self.area2[3][0]+5, self.area2[3][1]+2), 1,1, color.text1(), color.text2())
        enter = len(self.entering)
        exit = len(self.exiting)
        cvzone.putTextRect(frame,str(f"Enter: {enter}"), (20,30), 1,1, color.text1(), color.text2())
        cvzone.putTextRect(frame,str(f"Exit: {exit}"), (20,60), 1,1, color.text1(), color.text2())

    def show_time(self, frame):
        elapsed_time = time.time() - self.start_time

        # Convert elapsed time to hours, minutes, seconds, and milliseconds
        milliseconds = int(elapsed_time * 1000) #/ 6.0001
        hours, milliseconds = divmod(milliseconds, 3600000)
        minutes, milliseconds = divmod(milliseconds, 60000)
        seconds = (milliseconds / 1000.0)

        # Display the time in the format "hour:minute:second.millisecond"
        time_str = "Running Time: {:02}:{:02}:{:06.3f}".format(int(hours), int(minutes), seconds)
        cvzone.putTextRect(frame,time_str, (20,480), 1,1, color.text1(), color.text2())

    def counting(self, video_path):
        cap = cv2.VideoCapture(video_path)

        downloads_path = os.path.join(os.path.expanduser('~'), 'Downloads')
        output_file_path = os.path.join(downloads_path, 'output_video.avi')
        out = cv2.VideoWriter(output_file_path,cv2.VideoWriter_fourcc(*'XVID'), 24.0, (1020,500))

        while True:
            if not self.paused:
                ret, frame = cap.read()
                if not ret: break

                frame = cv2.resize(frame,(1020,500))

                #results = model.track(frame, persist=True, conf=0.5)
                #frame_ = results[0].plot()

                detections = self.detect(frame)
                self.draw_boxes(frame, detections)

                out.write(frame)
                self.show_time(frame)
                cv2.imshow('Frame', frame)

            key = cv2.waitKey(1)&0xFF
            if cv2.getWindowProperty('Frame', cv2.WND_PROP_VISIBLE) < 1: 
                break
            elif key == ord('p'):
                self.paused = not self.paused


            #if cv2.waitKey()&0xFF == ord('q'): break
            #if cv2.waitKey(0)&0xFF == 27: continue

        cap.release()
        out.release()
        cv2.destroyAllWindows()