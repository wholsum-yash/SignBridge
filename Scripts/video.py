import os
from collections import deque

import cv2
import numpy as np
from landmarks import get_landmarks
from tensorflow.keras.models import load_model  # pyright: ignore

from prediction_filter import Stabilizer
from state_machine import StateMachine
from sentence_builder import SentenceBuilder


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
DISPLAY_FRAMES = 6  # snappy

# modules
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

        # show word briefly
        if display_word is not None and display_timer > 0:
            cv2.putText(
                frame,
                display_word,
                (50, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )
            display_timer -= 1

        # SENTENCE DISPLAY 
        if current_sentence:
            cv2.putText(
                frame,
                current_sentence,
                (50, 160),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )

    cv2.imshow("SignBridge Demo", frame)

    if cv2.waitKey(10) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
