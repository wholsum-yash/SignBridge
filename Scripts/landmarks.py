import cv2
import mediapipe as mp

CONFIDENCE_THRESHOLD = 0.5 # the minimum confidence to detect a hand sign

"""NOTE: mediapipe API doesn't expose clean confidence per frame; so we approx confidence using detection presence; no-hands -> 0, detected-hands -> 1, binary nature for now; Mediapipe handmarker for real confidence scores (Future Consideration). """

mp_hands = mp.solutions.hands #pyright: ignore[reportAttributeAccessIssue]
hands = mp_hands.Hands()

# confidence detection helper functions

def zero_frame(landmarks): # checks if frame is a zero frame
    return all(v == 0 for v in landmarks)

# Getting Mediapipe Landmarks
def get_landmarks(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    all_landmarks = []
    
    frame_confidence = 1.0 if results.multi_hand_landmarks else 0.0

    if frame_confidence >= CONFIDENCE_THRESHOLD:

        # Process up to 2 hands
        for i in range(min(len(results.multi_hand_landmarks), 2)):
            hand_landmarks = results.multi_hand_landmarks[i]
            landmarks = []
            for point in hand_landmarks.landmark:
                landmarks.extend([point.x, point.y, point.z])

            # Normalize relative to wrist (first landmark)
            wrist_x = landmarks[0]
            wrist_y = landmarks[1]
            
            for j in range(0, len(landmarks), 3):
                landmarks[j] -= wrist_x
                landmarks[j + 1] -= wrist_y
            
            all_landmarks.extend(landmarks)

    # Padding with zeros if less than 2 hands detected (126 = 2 hands * 21 landmarks * 3 coords)
    padding_needed = 126 - len(all_landmarks)
    if padding_needed > 0:
        all_landmarks.extend([0] * padding_needed)

    return all_landmarks[:126]  # Ensure exact length 126

def repair_frames(sequence): # Fixes broken (zero) frames in a sequence by copying adjacent valid frames.

    """
     This is called in build_dataset.py before validation.
    The 'repaired_count' returned is used to calculate 'fallback_ratio' for quality checks.
    """

    last_valid = None # checks for last valid frame
    repaired_count = 0 # checks for repaired frames

    for i in range(len(sequence)):

        if zero_frame(sequence[i]):
            if last_valid is not None:
                sequence[i] = last_valid # carrying forward the last valid frame
                repaired_count += 1

            else:
                # Edge Case: First few frames are broken; Look ahead for first valid frame
                for j in range( i + 1, len(sequence)):
                    if not zero_frame(sequence[j]):
                        sequence[i] = sequence[j]
                        last_valid = sequence[j]
                        repaired_count += 1
                        break
        else:
            last_valid = sequence[i]

    return sequence, repaired_count 
