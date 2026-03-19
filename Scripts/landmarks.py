import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()


# Getting Mediapipe Lankmarks
def get_landmarks(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        hand = results.multi_hand_landmarks[0]

        landmarks = []
        for point in hand.landmark:
            landmarks.extend([point.x, point.y, point.z])

        return landmarks  # expected length = 63
    else:
        return [0] * 63
