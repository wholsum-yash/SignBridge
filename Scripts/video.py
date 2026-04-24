import os
from collections import deque

import cv2
import numpy as np
from landmarks import get_landmarks
from tensorflow.keras.models import load_model  # pyright: ignore[reportMissingModuleSource] 

from prediction_filter import Stabilizer
from state_machine import StateMachine
from sentence_builder import SentenceBuilder
from UI_UX import draw_ui 

def is_no_hand_sequence(sequence, threshold=0.6):
    zero_frames = sum(np.all(frame == 0) for frame in sequence)
    return (zero_frames / len(sequence) >= threshold)


# loading model
model = load_model("model/best_model.h5")

# labels
DATA_PATH = "dataset"
actions = sorted(os.listdir(DATA_PATH))

# buffers
sequence = deque(maxlen=32)

# display hold
display_word = None
display_timer = 0
DISPLAY_FRAMES = 6  #  for words to be snappy on screen

confidence = 0.0 # pre-defined to prevent crashes

# prediction_filter module
stabilizer = Stabilizer(
    maxlen=10,
    conf_threshold=0.6,
    score_threshold=4.0
)

state_machine = StateMachine(cooldown_frames=15)
sentence_builder = SentenceBuilder(max_pause_frames=20)

# video input
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)

if not cap.isOpened():
    print("Camera still not accessible")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # extracting landmarks
    landmarks = get_landmarks(frame)

    if landmarks is not None:
        sequence.append(landmarks)

    if len(sequence) == 32:

        seq_array = np.array(sequence)

        no_hand = is_no_hand_sequence(seq_array)

        if no_hand:
            stabilizer.reset()
        else:
            res = model.predict(np.expand_dims(seq_array, axis=0))[0]

            confidence = float(np.max(res))
            pred = int(np.argmax(res))

            if confidence >= 0.75:
                stabilizer.update(pred, confidence)

        final_pred = stabilizer.get_output()

        # STATE MACHINE
        has_hand = not no_hand
        emitted = state_machine.update(has_hand, final_pred)

        # SENTENCE BUILDER 
        final_sentence = sentence_builder.update(emitted, has_hand, actions)
        current_sentence = sentence_builder.get_current_sentence()

        if final_sentence is not None:
            print(f"SENTENCE: {final_sentence}")

        # WORD DISPLAY 
        if emitted is not None:
            print(f"FINAL: {actions[emitted]}")
            display_word = actions[emitted]
            display_timer = DISPLAY_FRAMES

        # reduce timer (snappy persistence)
        if display_timer > 0:
            display_timer -= 1
        else:
            display_word = None

    else:
        current_sentence = ""

    # UI RENDER (ONLY PLACE FOR VISUALS) 
    frame = draw_ui(
        frame,
        display_word,
        current_sentence,
        confidence,
        state_machine.get_state()
    )

    cv2.imshow("SignBridge Demo", frame)

    if cv2.waitKey(10) & 0xFF == ord("q"):
        break

# CLEANUP 
cap.release()
cv2.destroyAllWindows()
