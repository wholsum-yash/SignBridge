import cv2
import mediapipe as mp
import numpy as np
CONFIDENCE_THRESHOLD = 0.5 # the minimum confidence to detect a hand sign

"""NOTE: mediapipe API doesn't expose clean confidence per frame; so we approx confidence using detection presence; no-hands -> 0, detected-hands -> 1, binary nature for now; Mediapipe handmarker for real confidence scores (Future Consideration). """

mp_face = mp.solutions.face_mesh #pyright: ignore[reportAttributeAccessIssue]
mp_hands = mp.solutions.hands #pyright: ignore[reportAttributeAccessIssue]
mp_pose = mp.solutions.pose #pyright: ignore[reportAttributeAccessIssue]

""" Initializing Mediapipe models for hands, face, pose detection

NOTE: These run per frame and provide raw landmark data """

hands = mp_hands.Hands()
face_mesh = mp_face.FaceMesh()
pose = mp_pose.Pose()

HAND_FEATURES = 126
TOTAL_FEATURES = 148

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

# extracting features: hands, face, neck
def extract_hand_landmarks(results): # Extract hand landmarks and normalize them relative to wrist

    global last_valid_hands, missing_frame_count

    all_landmarks = []
    hand_centers= []

    # CASE 1: HAND NOT DETECTED (DECAYED AWAY)
    if not results.multi_hand_landmarks:
        # return [0]*126, [(0,0), (0,0)]  --(OLD-LOGIC: INSTANT ZERO)--
        missing_frame_count += 1

        # NEW-LOGIC: Decaying of missing hand
        if last_valid_hands is not None:

            if missing_frame_count <= DECAY_WINDOW:

                # smoothly decay
                prev_landmarks, prev_centers = last_valid_hands # pyright: ignore[reportGeneralTypeIssues]
                
                decay_factor = max(0, 1 - (missing_frame_count / DECAY_WINDOW))

                decayed_landmarks = [v * decay_factor for v in prev_landmarks]

                return decayed_landmarks, prev_centers

            else:
                # fully gone after decay
                last_valid_hands = None
                return [0]*126, [(0,0), (0,0)]

        else:
            return [0]*126, [(0,0), (0,0)] 

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
            landmarks[j+1] -= wrist_y

        all_landmarks.extend(landmarks)

        center = compute_hand_center(hand)
        hand_centers.append(center)

    # padding if 1 hand; ensuring consistent 2 hand representation; padding missing hand with zeros 
    while len(hand_centers) < 2:
        hand_centers.append((0,0))

    padding = HAND_FEATURES - len(all_landmarks)
    if padding > 0:
        all_landmarks.extend([0] * padding)

    final_landmarks = all_landmarks[:126]


    if not zero_frame(final_landmarks):
        last_valid_hands = (final_landmarks, hand_centers)

    return final_landmarks, hand_centers

def extract_face_points(face_results): # extract key face points used for spacial reasoning (not complete face mesh)
    if not face_results.multi_face_landmarks:
        return None

    face = face_results.multi_face_landmarks[0]

    def get(idx):
        p = face.landmark[idx]

        return (p.x, p.y)

    return{
        "nose": get(1),
        "chin": get(152),
        "forehead": get(10),
        "left_cheek": get(234),
        "right_cheek": get(454)
    }

def extract_neck_points(pose_results): # estiamtes neck position as midpoint between shoulders
    if not pose_results.pose_landmarks:
        return (0,0)

    lm = pose_results.pose_landmarks.landmark

    left = lm[mp_pose.PoseLandmark.LEFT_SHOULDER]
    right = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER]

    return ((left.x + right.x )/2, (left.y + right.y)/2)

# spacial features engine
def compute_spatial_features(center, face, neck):

    """Converts hand position into spatial features relative to face and neck;
    This logic allows model to distinguish between similar sign gestures by location """    

    if face is None:
        return [0] * 9

    # distances
    d_nose = distance(center, face["nose"])
    d_chin = distance(center, face["chin"])
    d_forehead = distance(center, face["forehead"])
    d_left_cheek = distance(center, face["left_cheek"])
    d_right_cheek = distance(center, face["right_cheek"])

    # normalization: normalizing distance to make it invariant to camera distance
    face_h = distance(face["nose"], face["chin"]) + 1e-6 # 1e-6: a tiny number added to prevent division by zero
    face_w = distance(face["left_cheek"], face["right_cheek"]) + 1e-6

    # relative position
    cx, cy = face["nose"]

    rel_x = (center[0] - cx)/ face_w
    rel_y = (center[1] - cy)/ face_h

    # neck
    d_neck = distance(center, neck)
    rel_neck_y = (center[1] - neck[1]) / face_h

    return [
        d_nose, d_chin, d_forehead, d_left_cheek, d_right_cheek, d_neck,
        rel_x, rel_y,
        rel_neck_y
    ] 
# Getting Mediapipe Landmarks
def get_landmarks(frame):

    """ Main feature extraction pipeline; Converts frames into fixed length feature vectors of dim(148) """
    
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    hand_results = hands.process(frame_rgb)
    face_results = face_mesh.process(frame_rgb)
    pose_results = pose.process(frame_rgb)

    # hands
    hand_features, hand_centers = extract_hand_landmarks(hand_results)

    # face
    face = extract_face_points(face_results)

    # neck
    neck = extract_neck_points(pose_results)

    spacial_features = []

    for center in hand_centers:
        spacial = compute_spatial_features(center, face, neck)
        spacial_features.extend(spacial)

    final_features = hand_features + spacial_features

    # SAFETY: CHECKING -> consistent feature size for model input
    if(len(final_features) != TOTAL_FEATURES):
        return [0] * TOTAL_FEATURES

    return final_features 
   

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
