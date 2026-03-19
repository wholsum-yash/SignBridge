import json
import os

from frames import extract_frames
from landmarks import get_landmarks

with open("/home/wholsum/projects/SignBridge/WLASL-dataset/WLASL_v0.3.json") as words:
    data = json.load(words)

# checking the words in the dataset
words = [entry["gloss"].lower() for entry in data]

TARGET_WORDS = [
    "hello",
    "thank you",
    "help",
    "yes",
    "no",
    "please",
    "sorry",
    "good",
    "bad",
    "stop",
    "go",
    "eat",
    "drink",
    "come",
    "wait",
]

filtered = []  # Holds the TARGET_WORDS data from the availble data.

# Selecting the TARGET_WORD classes from the available data
for entry in data:
    if entry["gloss"].lower() in TARGET_WORDS:
        filtered.append(entry)

# building the dataset
x = []
y = []

# Your Path to dataset videos as BASE_PATH
BASE_PATH = "/home/wholsum/projects/SignBridge/WLASL-dataset/videos"

for entry in filtered[:3]:
    label = entry["gloss"].lower()

    for inst in entry["instances"]:
        video_id = inst["video_id"]
        video_path = f"{BASE_PATH}/{video_id}.mp4"

        # print(video_path)
        # print(os.path.exists(video_path))

        # Skip missing files in the dataset
        if not os.path.exists(video_path):
            continue

        frames = extract_frames(video_path)

        sequence = []
        for frame in frames:
            landmarks = get_landmarks(frame)
            sequence.append(landmarks)

        x.append(sequence)
        y.append(label)
print(len(x))
print(len(y))
