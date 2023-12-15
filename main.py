from human_counting import Algorithm_Count
from human_detection_webcam import Algorithm_Detection 
from set_coordinates import ClickPoints 


'''
# For Uploading a video files
a1=[(312,388),(289,390),(474,469),(497,462)]
a2=[(279,392),(250,397),(423,477),(454,469)]
#a1 = []
#a2 = []


in_video_path = 'Sample Test File\\test_video.mp4'


if not a1:
    coordinates1 = ClickPoints(in_video_path, a2)
    a1 = coordinates1.run()


if not a1:
    print("area 1 No coordinates")
    exit() # You can change it the flow 
elif len(a1) < 4:
    print("area 1 Incomplete")
    exit() # You can change it the flow 

if not a2:
    coordinates2 = ClickPoints(in_video_path, a1)
    a2 = coordinates2.run()

if not a2: 
    print("area 2 No coordinates")
    exit() # You can change it the flow 
elif len(a2) < 4:
    print("area 2 Incomplete")
    exit() # You can change it the flow 
else:
    print("Coordinates from ClickPoints:", a1)
    print("Coordinates from ClickPoints:", a2)

    algo = Algorithm_Count(a1, a2)
    algo.counting(in_video_path)


#----------------------------------------------------------------
'''
# For see the webcam
webcam = Algorithm_Detection()
webcam.detectPeople()


