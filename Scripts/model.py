import os
from datetime import datetime
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
from tensorflow.keras.models import Sequential # pyright: ignore[reportMissingModuleSource]
from tensorflow.keras.optimizers import Adam # pyright: ignore[reportMissingModuleSource]
from tensorflow.keras.utils import to_categorical # pyright: ignore[reportMissingImports]

BASE_FEATURES = 126
USE_VELOCITY = True

TOTAL_FEATURES = BASE_FEATURES * (2 if USE_VELOCITY else 1)


# DEBUG CALLBACK 
class DebugCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        print(f"\n[DEBUG] Epoch {epoch + 1}")
        print(f"Loss: {logs.get('loss'):.4f}")
        print(f"Accuracy: {logs.get('categorical_accuracy'):.4f}")
        print(f"Val Loss: {logs.get('val_loss'):.4f}")
        print(f"Val Acc: {logs.get('val_categorical_accuracy'):.4f}")


# path setup
DATA_PATH = "dataset"
actions = sorted(os.listdir(DATA_PATH))

# label encoding
label_map = {label: num for num, label in enumerate(actions)}

# grouping files by video 
grouped = defaultdict(list)

for action in actions:
    action_path = os.path.join(DATA_PATH, action)

    for file in os.listdir(action_path):
        if not file.endswith(".npy"):
            continue

        # expected format: yes_abc123_orig.npy
        parts = file.replace(".npy", "").split("_")

        if len(parts) < 3:
            continue  # safety skip if malformed

        label = parts[0]
        video_id = parts[1]

        key = f"{label}_{video_id}"

        full_path = os.path.join(action_path, file)
        grouped[key].append(full_path)

# split by video 
video_keys = list(grouped.keys())

train_keys, val_keys = train_test_split(
    video_keys,
    test_size=0.2,
    random_state=42
)

# build train/val 
train_files = []
val_files = []

# train
for key in train_keys:
    train_files.extend(grouped[key])

# val
for key in val_keys:
    for file in grouped[key]:
        if file.endswith("Orig.npy"):
            val_files.append(file)

# load data 
X_train, y_train = [], []
X_val, y_val = [], []

def extract_label(path):
    return os.path.basename(os.path.dirname(path))

# training
for file in train_files:
    X_train.append(np.load(file))
    y_train.append(label_map[extract_label(file)])

# validation
for file in val_files:
    X_val.append(np.load(file))
    y_val.append(label_map[extract_label(file)])

X_train = np.array(X_train)
X_val = np.array(X_val)

y_train = to_categorical(np.array(y_train))
y_val = to_categorical(np.array(y_val))

# ASSERTS 
assert X_train.shape[2] == TOTAL_FEATURES, \
    f"Feature mismatch: expected {TOTAL_FEATURES}, got {X_train.shape[2]}"

# DEBUG CHECKS 
print("\n[DEBUG] Train videos:", len(train_keys))
print("[DEBUG] Val videos:", len(val_keys))
print("[DEBUG] Train samples:", len(train_files))
print("[DEBUG] Val samples:", len(val_files))

# CRITICAL: no leakage
overlap = set(train_keys).intersection(set(val_keys))
print("[DEBUG] Overlap:", overlap)


# SANITY CHECKS 
assert len(X_train) > 0, "Train dataset empty"
assert len(X_val) > 0, "Validation dataset empty"
assert X_train.shape[1:] == (32, TOTAL_FEATURES), "Shape mismatch"

if np.isnan(X_train).any():
    print("[WARNING] NaN values in train set")

if np.isinf(X_train).any():
    print("[WARNING] Inf values in train set")


# class weights 
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
    max_lr = 1e-3
    if epoch < warmup_epochs:
        return max_lr * (epoch + 1) / warmup_epochs
    return max_lr


lr_callback = LearningRateScheduler(scheduler)


# model
model = Sequential()

model.add(LSTM(64, return_sequences=True, activation="relu", input_shape=(32, TOTAL_FEATURES)))
model.add(Dropout(0.3))

model.add(LSTM(64, return_sequences=False, activation="relu"))
model.add(Dropout(0.3))

model.add(Dense(64, activation="relu"))
model.add(Dropout(0.3))

model.add(Dense(len(actions), activation="softmax"))

# compile
model.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss=CategoricalCrossentropy(label_smoothing=0.1),
    metrics=["categorical_accuracy"],
)

print("Min:", np.min(X_train), "Max:", np.max(X_train))


# callbacks/early stopping
early_stop = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)

checkpoint = ModelCheckpoint(
    "model/best_model.h5",
    monitor="val_loss",
    save_best_only=True
)


# train model
model.fit(
    X_train,
    y_train,
    epochs=50,
    shuffle=True,
    validation_data=(X_val, y_val),
    class_weight=class_weights,
    callbacks=[early_stop, checkpoint, lr_callback, DebugCallback()],
)


# SANITY PREDICTION 
print("\n[DEBUG] Prediction sanity check")

sample = X_train[0:1]
prediction = model.predict(sample)

print("Prediction vector:", prediction)
print("Predicted class:", np.argmax(prediction))
print("Actual class:", np.argmax(y_train[0]))

# checking total features:
print(f"[CONFIG] TOTAL_FEATURES = {TOTAL_FEATURES}")

# saving model
model.save("model/gesture_model.h5")
