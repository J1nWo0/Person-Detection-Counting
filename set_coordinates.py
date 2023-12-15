import cv2
import cvzone
import numpy as np

class ClickPoints:
    def __init__(self, video_path, predefined_list):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        self.clicked_points = []
        self.original_image = None
        self.image = None
        self.return_coordinates = False
        self.predefined_list = predefined_list
 
        # Check if the video is opened successfully
        if not self.cap.isOpened():
            print("Error: Could not open the video.")
            exit()

        # Read the first frame
        ret, self.original_image = self.cap.read()

        # Check if the frame is read successfully
        if not ret:
            print("Error: Could not read the first frame.")
            exit()

        # Create a copy of the original image for processing
        self.image = self.original_image.copy()

        # Set the callback function for mouse events
        cv2.namedWindow("Set Coordinates")
        cv2.setMouseCallback("Set Coordinates", self.mouse_callback)

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # Append the coordinates to the list
            self.clicked_points.append((x, y))
            #print(f"Clicked at ({x}, {y})")

            # Draw a point on the image at the clicked coordinates
            cv2.circle(self.image, (x, y), 4, (0, 0, 255), -1)

            # Check if more than 4 points have been clicked
            if len(self.clicked_points) > 4:
                # Clear the list and reset the image
                self.clicked_points = []
                self.image = self.original_image.copy()

        # Check if 4 points have been clicked
        if len(self.clicked_points) == 4:
            # Create a box using the clicked points
            self.create_box()

    def display_coordinates(self):
        p, i = 0, 1
        cvzone.putTextRect(self.image,'Press [s] to save coordinates', (750,30), 1,1, (255,255,255), (0,0,0))
        cvzone.putTextRect(self.image,'Press [r] to reset coordinates', (750,60), 1,1, (255,255,255), (0,0,0))
        
        for x, y in self.clicked_points:
            cvzone.putTextRect(self.image,str(f"Point {i}: X: {x}, Y: {y}"), (20,30+p), 1,1, (255,255,255), (0,0,0))
            i+=1
            p+=30

    def create_box(self):
        # Convert the list of points to NumPy array
        points_array = np.array(self.clicked_points, dtype=np.int32)

        # Reshape the array to shape (4, 1, 2)
        points_array = points_array.reshape((4, 1, 2))

        # Draw the box on the image
        cv2.polylines(self.image, [points_array], isClosed=True, color=(255, 0, 0), thickness=2)

    def save_coordinates(self):
        self.return_coordinates = True

    def get_coordinates(self):
        return self.clicked_points
    
    def reset_coordinates(self):
        self.clicked_points = []
        self.image = self.original_image.copy()

    def is_have_predefined_list(self):
        if self.predefined_list:
            # Convert the list of points to NumPy array
            points_array = np.array(self.predefined_list, dtype=np.int32)

            # Reshape the array to shape (4, 1, 2)
            points_array = points_array.reshape((4, 1, 2))

            # Draw the box on the image
            cv2.polylines(self.image, [points_array], isClosed=True, color=(255, 0, 0), thickness=2)

    def run(self):
        
        while True:
            self.image = cv2.resize(self.image, (1020,500))
            self.is_have_predefined_list()
            self.display_coordinates()
            cv2.imshow("Set Coordinates", self.image)

            # Press 'Esc' to exit the loop
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q') or cv2.getWindowProperty('Set Coordinates', cv2.WND_PROP_VISIBLE) < 1:
                break
            elif key == ord('s'):
                # Save coordinates when 's' is pressed
                self.save_coordinates()
            elif key == ord('r'):
                self.reset_coordinates()

            # Check if 's' was pressed to return the coordinates
            if self.return_coordinates:
                # Reset the flag
                self.return_coordinates = False
                # Release the VideoCapture object
                self.cap.release()
                # Print the stored coordinates
                #print("Clicked points:", self.clicked_points)
                cv2.destroyAllWindows()
                # Return the coordinates
                return self.clicked_points
'''
if __name__ == "__main__":
    video_path = "D:\Project Programming\Person Detection Counting\Sample Test File\\test_video.mp4"  # Replace with the path to your video
    list1 = []
    list2 = []
    #list=[(312,388),(289,390),(474,469),(497,462)]
    #list=[(279,392),(250,397),(423,477),(454,469)]
    click_points_app = ClickPoints(video_path, list2)
    list1 = click_points_app.run()

    # Access the coordinates returned after the run method is called
    if not list1 or len(list1) < 4:
        print("No coordinates or Incomplete")
    else: 
        print("Coordinates from ClickPoints:", list1)
'''