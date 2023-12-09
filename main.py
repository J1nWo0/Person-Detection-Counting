

from human_counting import *
from human_detection_webcam import *



# For Uploading a video files
a1=[(312,388),(289,390),(474,469),(497,462)]
a2=[(279,392),(250,397),(423,477),(454,469)]

in_video_path = 'Sample Test File\\test_video.mp4'

algo = Algorithm_Count(a1, a2)


algo.counting(in_video_path)

'''
# For see the webcam
algo = Algorithm_Detection()

algo.detectPeople()

'''
