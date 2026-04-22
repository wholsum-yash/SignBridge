import json
import os

import numpy as np
from augment import augment_sequence
from frames import extract_frames
from landmarks import get_landmarks, repair_frames # pyright: ignore[reportAttributeAccessIssue]
from clean_dataset import validate_sequence


with open("/home/wholsum/projects/SignBridge/WLASL-complete/WLASL_v0.3.json") as words:
    data = json.load(words)

# checking the words in the dataset
words = [entry["gloss"].lower() for entry in data]

TARGET_WORDS = [
    "yes",
    "no",
    "wait",
    "good",
    "bad",
    #"book",
    #"drink",
    #"computer",
    #"before",
    #"go",
    #"chair",
    #"clothes",
    #"who",
    #"deaf",
    #"help",
    #"walk"
]
MAX_SAMPLES_PER_CLASS = 150 # safety cap

filtered = []  # Holds the TARGET_WORDS data from the availble data.

# Selecting the TARGET_WORD classes from the available data
for entry in data:
    if entry["gloss"].lower() in TARGET_WORDS:
        filtered.append(entry)

# Your Path to dataset videos as BASE_PATH
BASE_PATH = "/home/wholsum/projects/SignBridge/WLASL-complete/videos/"

for label in TARGET_WORDS:
    os.makedirs(f"dataset/{label}", exist_ok= True)


counters = { label:0
    for label in TARGET_WORDS
}  # dictionary to count samples per class


for entry in filtered:
    label = entry["gloss"].lower()

    if counters[label] >= MAX_SAMPLES_PER_CLASS:
        continue

    for inst in entry["instances"]:

        if counters[label] >= MAX_SAMPLES_PER_CLASS:
            break
        video_id = inst["video_id"]
        video_path = f"{BASE_PATH}/{video_id}.mp4"


        # Skip missing files in the dataset
        if not os.path.exists(video_path):
            continue

        frames = extract_frames(video_path)

        sequence = []

        for frame in frames:
            landmarks = get_landmarks(frame)
            sequence.append(landmarks)
       
        # Repair broken frames using landmarks.py
        sequence, repaired_count = repair_frames(sequence)

        total_frames = len(sequence)

        if total_frames == 0:
            print("[REJECTED: EMPTY SEQUENCE]")
            continue
        
        # Calculate fallback_ratio
        # represents the percentage of the video that was 'repaired'.
        # MUST: pass this to validate_sequence to check for "too much repair".
        fallback_ratio = repaired_count / total_frames

        # converting frames to numpy
        sequence = np.array(sequence)

        # DEBUG: Checking Sequence Shape
        print(f"[DEBUG] Sequence shape before validation: {sequence.shape}")


        # Safety check
        if sequence.ndim != 2 or sequence.shape[1] != 144:
            print(f"[REJECTED] Invalid shape of sequence before validation: {sequence.shape}")
            continue       

        # Validate the sequence using clean_dataset.py
        # MUST: pass 'fallback_ratio' here so clean_dataset can enforce FALLBACK_THRESHOLD.
        is_valid, reason, metrics = validate_sequence(sequence, fallback_ratio)
        print(
            f"{video_id} → "
            f"valid: {metrics['valid_ratio']:.2f}, "
            f"diversity: {metrics['diversity_ratio']:.2f}, "
            f"repair: {fallback_ratio:.2f}"
        )

        if not is_valid:
            print(f"[REJECTED] ->{reason}\n")
            continue
      
        # Ensuring Shape at dataset level
        if sequence.shape != (32, 144):
            print(f"[SKIP] Invalid Shape:{sequence.shape}")
            continue
        
        # adding augmentation
        augment_sequences = augment_sequence(sequence)
        
        # path creation
        save_dir = f"dataset/{label}"
        count = counters[label]

        for idx, aug_seq in enumerate(augment_sequences):
            save_path = f"{save_dir}/sample_{count:03d}_aug{idx}.npy"
        
            # saving samples
            print(f"Saving to: {save_path}")
            np.save(save_path, aug_seq)

        # incrementing counter AFTER saving all augmented versions
        counters[label] += 1

print("\n __DATASET SUMMARY__")
print(f"Total number of classes: {len(TARGET_WORDS)}")
for label in TARGET_WORDS:
    print(f"{label}: {counters[label]} .npy files saved")
print(f"Total samples saved: {sum(counters.values())}")
