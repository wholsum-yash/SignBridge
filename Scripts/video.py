import os
from collections import deque

import cv2
import numpy as np
from landmarks import get_landmarks
from tensorflow.keras.models import load_model # pyright: ignore[reportMissingModuleScource]

# loading model
model = load_model("model/best_model.h5")

# TARGET WORDS labels in Training Order
DATA_PATH = "dataset"
actions = sorted(os.listdir(DATA_PATH))

# buffers
sequence = deque(maxlen=32)
predictions = deque(maxlen=10)

# video input (VIDEO FOR DEMO)
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
    sequence.append(landmarks)

    # prediction
    if len(sequence) == 32:
        res = model.predict(np.expand_dims(np.array(sequence), axis=0))[0]

        confidence = np.max(res)
        pred = np.argmax(res)

        # confidence filter
        if confidence > 0.7:
            predictions.append(pred)
        else:
            predictions.append(None)

        print(f"Pred: {actions[pred]} | Conf: {confidence:.2f}")

        # smoothing
        final_pred = None

        if len(predictions) == 10:
            valid_preds = [p for p in predictions if p is not None]

            if len(valid_preds) > 0:
                most_common = max(set(valid_preds), key=valid_preds.count)

                if valid_preds.count(most_common) > 7:
                    final_pred = most_common

        # display
        if final_pred is not None:
            cv2.putText(
                frame,
                f"{actions[final_pred]} ({confidence:.2f})",
                (50, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2,
            )

    cv2.imshow("SignBridge Demo", frame)

    if cv2.waitKey(10) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
