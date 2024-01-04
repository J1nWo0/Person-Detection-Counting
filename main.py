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
'''
# For see the webcam
webcam = Algorithm_Detection()
webcam.detectPeople()


from human_counting import Algorithm_Count
from human_detection_webcam import Algorithm_Detection
from set_coordinates import ClickPoints
'''

class VideoProcessor:
    def __init__(self, in_video_path, area1_coordinates=None, area2_coordinates=None):
        self.in_video_path = in_video_path
        self.area1_coordinates = area1_coordinates
        self.area2_coordinates = area2_coordinates

    def get_coordinates(self, current_coordinates, other_coordinates, area):
        if not current_coordinates:
            click_points = ClickPoints(self.in_video_path, other_coordinates)
            current_coordinates = click_points.run()

        if not current_coordinates:
            print(f"Area {area} No coordinates")
            exit()

        if len(current_coordinates) < 4:
            print(f"Area {area} Incomplete")
            exit()

        return current_coordinates

    def process_video(self):
        self.area1_coordinates = self.get_coordinates(self.area1_coordinates, self.area2_coordinates, 1)
        self.area2_coordinates = self.get_coordinates(self.area2_coordinates, self.area1_coordinates, 2)

        print("Coordinates from ClickPoints (Area 1):", self.area1_coordinates)
        print("Coordinates from ClickPoints (Area 2):", self.area2_coordinates)

        algo = Algorithm_Count(self.area1_coordinates, self.area2_coordinates)
        algo.counting(self.in_video_path)


if __name__ == "__main__":
    a1 = [(312,388),(289,390),(474,469),(497,462)]
    a2 = []
    in_video_path = 'Sample Test File\\test_video.mp4'

    video_processor = VideoProcessor(in_video_path, a1, a2)
    video_processor.process_video()

    # Uncomment the following block if you want to use the webcam
    '''
    webcam_processor = Algorithm_Detection()
    webcam_processor.detectPeople()
    '''
