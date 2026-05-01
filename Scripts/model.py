import os
import sys
from collections import defaultdict

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from tensorflow.keras.callbacks import (
    Callback,
    EarlyStopping,
    LearningRateScheduler,
    ModelCheckpoint,
) # pyright: ignore[reportMissingModuleSource]
from tensorflow.keras.layers import LSTM, Dense, Dropout # pyright: ignore[reportMissingModuleSource]
from tensorflow.keras.losses import CategoricalCrossentropy # pyright: ignore[reportMissingModuleSource]
from tensorflow.keras.models import Sequential, load_model # pyright: ignore[reportMissingModuleSource]
from tensorflow.keras.optimizers import Adam # pyright: ignore[reportMissingModuleSource]
from tensorflow.keras.utils import to_categorical # pyright: ignore[reportMissingImports]

# mode control
if len(sys.argv) < 2:
    print("\n[ERROR] No mode provided.")
    print("Usage: python model.py [train | finetune]")
    sys.exit(1)

MODE = sys.argv[1].lower()

if MODE not in ["train", "finetune"]:
    print("\n[ERROR] Invalid mode.")
    print("Use: train OR finetune")
    sys.exit(1)

print(f"\n[MODE SELECTED]: {MODE}")

# config
BASE_FEATURES = 416
USE_VELOCITY = True
TOTAL_FEATURES = BASE_FEATURES * (2 if USE_VELOCITY else 1)
DATA_PATH = "dataset"

# DEBUG CALLBACK 
class DebugCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        print(f"\n[DEBUG] Epoch {epoch + 1}")
        print(f"Loss: {logs.get('loss'):.4f}")
        print(f"Accuracy: {logs.get('categorical_accuracy'):.4f}")
        print(f"Val Loss: {logs.get('val_loss'):.4f}")
        print(f"Val Acc: {logs.get('val_categorical_accuracy'):.4f}")

# data prep
actions = sorted(os.listdir(DATA_PATH))
label_map = {label: num for num, label in enumerate(actions)}

grouped = defaultdict(list)

for action in actions:
    action_path = os.path.join(DATA_PATH, action)

    for file in os.listdir(action_path):
        if not file.endswith(".npy"):
            continue

        parts = file.replace(".npy", "").split("_")
        if len(parts) < 3:
            continue

        key = f"{parts[0]}_{parts[1]}"
        grouped[key].append(os.path.join(action_path, file))

video_keys = list(grouped.keys())

train_keys, val_keys = train_test_split(
    video_keys, test_size=0.2, random_state=42
)

train_files, val_files = [], []

for key in train_keys:
    train_files.extend(grouped[key])

for key in val_keys:
    for file in grouped[key]:
        if file.endswith("Orig.npy"):
            val_files.append(file)

X_train, y_train, X_val, y_val = [], [], [], []

def extract_label(path):
    return os.path.basename(os.path.dirname(path))

for file in train_files:
    X_train.append(np.load(file))
    y_train.append(label_map[extract_label(file)])

for file in val_files:
    X_val.append(np.load(file))
    y_val.append(label_map[extract_label(file)])

X_train = np.array(X_train)
X_val = np.array(X_val)

num_classes = len(actions)

y_train = to_categorical(np.array(y_train), num_classes=num_classes)
y_val = to_categorical(np.array(y_val), num_classes=num_classes)

labels_flat = np.argmax(y_train, axis=1)

class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(labels_flat),
    y=labels_flat
)

class_weights = dict(enumerate(class_weights))

# learning rate scheduler
def scheduler(epoch):
    warmup_epochs = 5

    if MODE == "train":
        max_lr = 1e-3
    else:
        max_lr = 1e-4

    if epoch < warmup_epochs:
        return max_lr * (epoch + 1) / warmup_epochs
    return max_lr

lr_callback = LearningRateScheduler(scheduler)

# model 
if MODE == "train":
    print("\n[MODE] Training from scratch")

    model = Sequential([
        LSTM(64, return_sequences=True, activation="relu", input_shape=(32, TOTAL_FEATURES)),
        Dropout(0.3),

        LSTM(64, return_sequences=False, activation="relu"),
        Dropout(0.3),

        Dense(64, activation="relu"),
        Dropout(0.3),

        Dense(num_classes, activation="softmax")
    ])

    lr = 1e-3
    epochs = 50
    save_path = "model/gesture_model.h5"

elif MODE == "finetune":
    print("\n[MODE] Fine-tuning existing model")

    model = load_model("model/gesture_model.h5")

    lr = 1e-4
    epochs = 25
    save_path = "model/fine_tuned_model.h5"

# compile
model.compile(
    optimizer=Adam(learning_rate=lr),
    loss=CategoricalCrossentropy(label_smoothing=0.1),
    metrics=["categorical_accuracy"],
) # pyright: ignore[reportPossiblyUnboundVariable]

# callbacks
if MODE == "train":
    early_stop = EarlyStopping(monitor="val_loss", patience=7, restore_best_weights=True)
else:
    early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

checkpoint = ModelCheckpoint(
    save_path,
    monitor="val_loss",
    save_best_only=True
) # pyright: ignore[reportPossiblyUnboundVariable]

# train
model.fit(
    X_train,
    y_train,
    epochs=epochs,
    batch_size=32,
    shuffle=True,
    validation_data=(X_val, y_val),
    class_weight=class_weights,
    callbacks=[early_stop, checkpoint, lr_callback, DebugCallback()],
) # pyright: ignore[reportPossiblyUnboundVariable]

# save
model.save(save_path) # pyright: ignore[reportPossiblyUnboundVariable]

print(f"\n[INFO] Model saved to {save_path}") # pyright: ignore[reportPossiblyUnboundVariable]
