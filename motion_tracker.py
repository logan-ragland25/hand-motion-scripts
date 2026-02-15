import mediapipe as mp
import cv2
import time
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import subprocess
import pynput

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

base_options = python.BaseOptions(model_asset_path='/home/logan/Documents/Projects/handMotionScripts/hand_gesture_final/gesture_recognizer.task')
options = vision.GestureRecognizerOptions(base_options=base_options)
detector = vision.GestureRecognizer.create_from_options(options)

cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 600)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 500)

mouse = pynput.mouse.Controller()
cam = cv2.VideoCapture(0)

analyze_timeout = 0
command_timeout = 0
save_path = "/home/logan/Pictures/Selfies/"

while True:
    success, frame = cap.read()
    if not success:
        break

    # Convert BGR to RGB and store data
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    result = detector.recognize(mp_image)

    if len(result.gestures) > 0 and time.time() - command_timeout > 1 and time.time() - analyze_timeout > 0.2:
        top_gesture = result.gestures[0][0]
        hand_landmarks = result.hand_landmarks

        if top_gesture.category_name == "peace":
            filename = f"{save_path}selfie_{int(time.time())}.png"
            cv2.imwrite(filename, frame)
            command_timeout = time.time()

        elif top_gesture.category_name == "closed_fist":
            control_window("minimize")
            command_timeout = time.time()

        elif top_gesture.category_name == "open_fist":
            control_window("maximize")
            command_timeout = time.time()

        elif top_gesture.category_name == "one_finger_up":
            mouse.scroll(0, 1)

        elif top_gesture.category_name == "two_fingers_up":
            mouse.scroll(0, 1)
            mouse.scroll(0, 1)
            mouse.scroll(0, 1)

        elif top_gesture.category_name == "one_finger_down":
            mouse.scroll(0, -1)

        elif top_gesture.category_name == "two_fingers_down":
            mouse.scroll(0, -1)
            mouse.scroll(0, -1)
            mouse.scroll(0, -1)

        analyze_timeout = time.time()