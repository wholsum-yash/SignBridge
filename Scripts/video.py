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
from tts import TTS
tts = TTS()


def add_velocity(sequence):
    velocity = np.diff(sequence, axis=0)
    velocity = np.vstack([np.zeros_like(velocity[0]), velocity])
    return np.concatenate([sequence, velocity], axis=1)


def is_no_hand_sequence(sequence, threshold=0.6):
    zero_frames = sum(np.all(frame[:416] == 0) for frame in sequence)
    return (zero_frames / len(sequence) >= threshold)


# load model
model = load_model("model/fine_tuned_model.h5")

data_path = "dataset"
actions = sorted(os.listdir(data_path))

sequence = deque(maxlen=32)

display_word = None
display_timer = 0
display_frames = 6

confidence = 0.0

last_emitted = None

# modules 
stabilizer = Stabilizer(
    maxlen=5,
    conf_threshold=0.6,
    score_threshold=2.5
)

state_machine = StateMachine(cooldown_frames=8)
sentence_builder = SentenceBuilder(max_pause_frames=20, expiry_frames=300)

# camera setup
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)

if not cap.isOpened():
    print("camera not accessible")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    landmarks = get_landmarks(frame)

    if landmarks is not None:
        sequence.append(landmarks)

    if len(sequence) == 32:

        seq_array = np.array(sequence)
        seq_array = add_velocity(seq_array)

        no_hand = is_no_hand_sequence(seq_array)

        if no_hand:
            stabilizer.reset()
            last_emitted = None
        else:
            res = model.predict(np.expand_dims(seq_array, axis=0))[0]

            confidence = float(np.max(res))
            pred = int(np.argmax(res))

            print("raw:", actions[pred], confidence)

            if confidence >= 0.6:
                stabilizer.update(pred, confidence)
            else:
                stabilizer.update(None, 0.0)

        final_pred = stabilizer.get_output()

        has_hand = not no_hand
        emitted = state_machine.update(has_hand, final_pred)

        sentence_builder.update(emitted, has_hand, actions)
        current_lines = sentence_builder.get_display_lines()

        # emission control 
        if emitted is not None:

            if emitted == last_emitted:
                emitted = None
            else:
                last_emitted = emitted

                word = actions[emitted]

                print(f"final: {actions[emitted]}")
                display_word = actions[emitted]
                display_timer = display_frames

                # adding tts
                tts.speak(word)

                sequence.clear()
                stabilizer.reset()

        if display_timer > 0:
            display_timer -= 1
        else:
            display_word = None

    else:
        current_lines = []

    # ui 
    frame = draw_ui(
        frame,
        display_word,
        current_lines,
        confidence,
        state_machine.get_state()
    )

    cv2.imshow("signbridge demo", frame)

    if cv2.waitKey(10) & 0xff == ord("q"):
        break

# cleanup
cap.release()
cv2.destroyAllWindows()
