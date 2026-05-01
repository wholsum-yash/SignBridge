import cv2
import mediapipe as mp
import numpy as np
CONFIDENCE_THRESHOLD = 0.5 # the minimum confidence to detect a hand sign

"""NOTE: mediapipe API doesn't expose clean confidence per frame; so we approx confidence using detection presence; no-hands -> 0, detected-hands -> 1, binary nature for now; Mediapipe handmarker for real confidence scores (Future Consideration). """

mp_hands = mp.solutions.hands #pyright: ignore[reportAttributeAccessIssue]

""" Initializing Mediapipe models for hands, face, pose detection

NOTE: These run per frame and provide raw landmark data """

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

HAND_FEATURES = 416

# temporal decay memory
last_valid_hands = None
missing_frame_count = 0
DECAY_WINDOW = 7

# helper functions
def zero_frame(landmarks): # checks if an invalid frame is a zero frame (all features are zero) used later for repairing frames
    return all(v == 0 for v in landmarks)

def distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def compute_hand_center(hand_landmarks): # computes geometric center of hands; used for spacial positioning relative to face
    xs =[p.x for p in hand_landmarks.landmark]
    ys =[p.y for p in hand_landmarks.landmark]
    return (np.mean(xs), np.mean(ys))

# extracting features: hands 
def extract_hand_landmarks(results):

    global last_valid_hands, missing_frame_count

    all_landmarks = []
    hand_centers = []

    # adding feature engineering
    def engineer_features(landmarks):
        lm = np.array(landmarks).reshape(2, 21, 3)

        features = []

        for hand in lm:
            if np.all(hand == 0):
                #  FIX: match full feature size per hand (145)
                features.extend([0] * 145)
                continue

            wrist = hand[0]
            norm = hand - wrist

            scale = np.linalg.norm(hand[9] - hand[0])
            if scale < 1e-3:
                scale = 1.0
            norm = norm / scale

            # 63 coords
            features.extend(norm.flatten())

            # 60 bone vectors
            for i in range(1, 21):
                features.extend(norm[i] - norm[i - 1])

            # 19 angles
            for i in range(1, 20):
                v1 = norm[i] - norm[i - 1]
                v2 = norm[i + 1] - norm[i]
                cos_angle = np.dot(v1, v2) / (
                    np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6
                )
                features.append(cos_angle)

            # 3 centroid
            centroid = np.mean(norm, axis=0)
            features.extend(centroid)

        features = np.array(features)

        #  FINAL SAFETY: force fixed size
        if len(features) < HAND_FEATURES:
            features = np.pad(features, (0, HAND_FEATURES - len(features)))
        elif len(features) > HAND_FEATURES:
            features = features[:HAND_FEATURES]

        return features

    # CASE 1: NO HAND 
    if not results.multi_hand_landmarks:
        missing_frame_count += 1

        if last_valid_hands is not None:
            prev_landmarks, _ = last_valid_hands # pyright: ignore[reportGeneralTypeIssues]

            # This preserves motion consistency instead of flattening it
            noise = np.random.normal(0, 0.01, size=len(prev_landmarks))
            drift = missing_frame_count * 0.002
            noisy = [v + n + drift for v, n in zip(prev_landmarks, noise)]
            return engineer_features(noisy)

        else:
            return np.zeros(HAND_FEATURES)

    # CASE 2: HAND DETECTED 
    missing_frame_count = 0

    for i in range(min(len(results.multi_hand_landmarks), 2)):
        hand = results.multi_hand_landmarks[i]

        landmarks = []
        for p in hand.landmark:
            landmarks.extend([p.x, p.y, p.z])

        # normalize relative to wrist
        wrist_x = landmarks[0]
        wrist_y = landmarks[1]

        for j in range(0, len(landmarks), 3):
            landmarks[j] -= wrist_x
            landmarks[j + 1] -= wrist_y

        all_landmarks.extend(landmarks)
        hand_centers.append(compute_hand_center(hand))

    while len(hand_centers) < 2:
        hand_centers.append((0, 0))

    RAW_FEATURES = 126

#  two hand structure -> important
    if len(all_landmarks) == 63:  # only one hand detected
        all_landmarks.extend([0] * 63)

    if len(all_landmarks) < RAW_FEATURES:
        all_landmarks.extend([0] * (RAW_FEATURES - len(all_landmarks)))

    final_landmarks = all_landmarks[:RAW_FEATURES]
    # print("RAW SUM:", np.sum(final_landmarks))

    #  STORE RAW FOR DECAY
    last_valid_hands = (final_landmarks, hand_centers)

    return engineer_features(final_landmarks)

# Getting Mediapipe Landmarks
def get_landmarks(frame):

    """ Main feature extraction pipeline; Converts frames into fixed length feature vectors of dim """
    
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_rgb = cv2.resize(frame_rgb, (640, 480)) # upscaling helps mediapipe in hand detection

    hand_results = hands.process(frame_rgb)
    # print("HAND DETECTED:", hand_results.multi_hand_landmarks is not None)

    # FIX: unpacking tuple correctly
    hand_features  = extract_hand_landmarks(hand_results)

    if len(hand_features) != HAND_FEATURES:
        return [0] * HAND_FEATURES

    return hand_features
 
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
