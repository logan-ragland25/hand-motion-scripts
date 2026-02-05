import math

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os

model_path = '/home/logan/Documents/Projects/handMotionScripts/hand_landmarker.task'
minimizedIds = []

base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
detector = vision.HandLandmarker.create_from_options(options)


# Try index 2 if 0 continues to fail
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 600)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 500)

pinched = False

while True:
    success, frame = cap.read()
    if not success:
        break

    RGB_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=RGB_frame)
    result = detector.detect(mp_image)

    if result.hand_landmarks:
        for hand_landmarks in result.hand_landmarks:
            # Loop through each of the 21 landmarks
            for landmark in hand_landmarks:
                # Convert normalized coordinates (0.0 to 1.0) to pixel coordinates
                x = int(landmark.x * frame.shape[1])
                y = int(landmark.y * frame.shape[0])

                # Draw a small circle at each joint
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

            # Optional: Draw a line between index tip (8) and thumb tip (4)
            # This is how you'd start building a 'pinch' detector
            thumb = hand_landmarks[4]
            index = hand_landmarks[8]
            thumb_coords = (int(thumb.x * frame.shape[1]), int(thumb.y * frame.shape[0]))
            index_coords = (int(index.x * frame.shape[1]), int(index.y * frame.shape[0]))
            cv2.line(frame, thumb_coords, index_coords, (255, 0, 0), 2)

            distance = math.sqrt((thumb.x * frame.shape[1] - index.x * frame.shape[1])**2 + (thumb.y * frame.shape[0] - index.y * frame.shape[0])**2)

            if distance < 30 and not pinched:
                pinched = True
                #minimizedIds.append() whatever the id of window being closed is

            if 80 < distance < 120:
                pinched = False

            if 150 > distance:
                if len(minimizedIds) > 0:
                    mostRecentID = minimizedIds.pop()
                    print(mostRecentID)

    cv2.imshow("capture image", frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
