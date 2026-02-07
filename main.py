import time
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import subprocess

minimizedIds = []

def get_focused_window_name():
    return subprocess.run(["kdotool", "getactivewindow"], capture_output=True, text=True).stdout.strip()

def control_window(action):
    cmd = ""

    if action == "minimize":
        current_window = get_focused_window_name()
        cmd = ["kdotool", "windowminimize", current_window]
        minimizedIds.append(current_window)

    elif action == "maximize":
        if len(minimizedIds) > 0:
            current_window = minimizedIds.pop()
            cmd = ["kdotool", "windowactivate", current_window]

    else:
        return

    if len(cmd) > 0:
        subprocess.run(cmd, text=True)

# Set up parameters for trained model
base_options = python.BaseOptions(model_asset_path='/home/logan/Documents/Projects/handMotionScripts/hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
detector = vision.HandLandmarker.create_from_options(options)
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 600)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 500)

analyze_timeout = 0
command_timeout = 0

# This program is intended to run in the background so no end condition needed
while True:
    success, frame = cap.read()
    if not success:
        break

    # Convert BGR to RGB and store data
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    result = detector.detect(mp_image)

    # If a hand is found, analyze
    if len(result.hand_landmarks) > 1 and time.time() - command_timeout > 1 and time.time() - analyze_timeout > 0.2:
        # For each hand detected
        for hand_landmarks in result.hand_landmarks:
            # Loop through each of the 21 hand landmarks
            for landmark in hand_landmarks:
                # Convert normalized coordinates (0.0 to 1.0) to pixel coordinates
                # Without this, all dot will be in top left corner at (0,0)
                x = int(landmark.x * frame.shape[1])
                y = int(landmark.y * frame.shape[0])

                # Create circle at each landmark
                cv2.circle(frame, (x, y), 3, (52, 21, 57))

            # Hand Position Logic
            # Only care about fingertip landmarks
            # Thumb
            thumb_pos_x = hand_landmarks[4].x * frame.shape[1]
            thumb_pos_y = hand_landmarks[4].y * frame.shape[0]

            # Index
            index_pos_x = hand_landmarks[8].x * frame.shape[1]
            index_pos_y = hand_landmarks[8].y * frame.shape[0]

            # Middle
            middle_pos_x = hand_landmarks[12].x * frame.shape[1]
            middle_pos_y = hand_landmarks[12].y * frame.shape[0]

            # Ring
            ring_pos_x = hand_landmarks[16].x * frame.shape[1]
            ring_pos_y = hand_landmarks[16].y * frame.shape[0]

            # Pinky
            pinky_pos_x = hand_landmarks[20].x * frame.shape[1]
            pinky_pos_y = hand_landmarks[20].y * frame.shape[0]

            # Thumb and Pointer Pinch --> Minimize Window
            if thumb_pos_y - index_pos_y < 20 and abs(thumb_pos_x - index_pos_x) < 10:
                control_window("minimize")
                command_timeout = time.time()
                # Set timeout of 1 second

            if thumb_pos_y - index_pos_y > 225:
                control_window("maximize")
                command_timeout = time.time()



        analyze_timeout = time.time()


    cv2.imshow("capture image", frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()