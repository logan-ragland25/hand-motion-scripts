import os
from mediapipe_model_maker import gesture_recognizer

model_asset_path = "/home/logan/Documents/Projects/handMotionScripts/assets"

image_path = os.path.join(model_asset_path, 'hand_position_photos')

data = gesture_recognizer.Dataset.from_folder(image_path)
train_data, remaining_data = data.split(0.8)
test_data, validation_data = remaining_data.split(0.5)

hparams = gesture_recognizer.HParams(export_dir="hand_gesture_final")
options = gesture_recognizer.GestureRecognizerOptions(hparams=hparams)
model = gesture_recognizer.GestureRecognizer.create(
    train_data = train_data,
    validation_data = validation_data,
    options=options,
)
model.export_model()