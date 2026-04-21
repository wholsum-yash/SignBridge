import os
from datetime import datetime

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import ( # pyright: ignore[reportMissingModuleSource]
    Callback,
    EarlyStopping,
    LearningRateScheduler,
    ModelCheckpoint,
)  
from tensorflow.keras.layers import (  # pyright: ignore[reportMissingModuleSource]
    LSTM,
    Dense,
    Dropout,
)
from tensorflow.keras.losses import CategoricalCrossentropy  # pyright: ignore[reportMissingModuleSource]
from tensorflow.keras.models import Sequential  # pyright: ignore[reportMissingModuleSource]
from tensorflow.keras.optimizers import Adam  # pyright: ignore[reportMissingModuleSource]
from tensorflow.keras.utils import to_categorical  # pyright: ignore[reportMissingImports]


class DebugCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        print(f"\n[DEBUG] Epoch {epoch + 1}")
        print(f"Loss: {logs.get('loss'):.4f}")
        print(f"Accuracy: {logs.get('categorical_accuracy'):.4f}")

        print(f"Val Loss: {logs.get('val_loss'):.4f}")
        print(f"Val Acc: {logs.get('val_categorical_accuracy'):.4f}")


# creating log directory
log_dir = os.path.join("logs", datetime.now().strftime("%Y%m%d-%H%M%S"))

# loading dataset
DATA_PATH = "dataset"
actions = sorted(os.listdir(DATA_PATH))

sequences = []
labels = []

label_map = {label: num for num, label in enumerate(actions)}

for action in actions:
    for file in os.listdir(os.path.join(DATA_PATH, action)):
        path = os.path.join(DATA_PATH, action, file)
        sequence = np.load(path)
        sequences.append(sequence)
        labels.append(label_map[action])

x = np.array(sequences)
y = np.array(labels)

# class weights
class_weights = compute_class_weight(
    class_weight="balanced", classes=np.unique(labels), y=labels
)

class_weights = dict(enumerate(class_weights))

# DEBUG: Dataset Size Sanity
print("\n[DEBUG] Total samples:", len(x))
print("[DEBUG] Number of classes:", len(actions))

# DEBUG: Empty Dataset Check
assert len(x) > 0, "Dataset is empty. Something failed upstream."
# DEBUG: Data Shape Check
assert x.shape[1:] == (32, 126), "Dataset shape mismatch. Expected (30, 126)"

# DEBUG: Data sanity check
if np.isnan(x).any():
    print("[WARNING] NaN values found in dataset")

if np.isinf(x).any():
    print("[WARNING] Infinite values found in dataset")

y = to_categorical(y)

# DEBUG: Data Validation Block
print("\n[DEBUG] Dataset Info")
print("X shape:", x.shape)
print("y shape:", y.shape)

# Check one sample
print("\n[DEBUG] Sample check")
print("Sample shape:", x[0].shape)

# Check label distribution
label_indices = np.argmax(y, axis=1)
unique, counts = np.unique(label_indices, return_counts=True)

print("\n[DEBUG] Label distribution:")
for u, c in zip(unique, counts):
    print(f"{actions[u]}: {c} samples")


# warmup scheduler
def scheduler(epoch):
    warmup_epochs = 5
    max_lr = 1e-3
    if epoch < warmup_epochs:
        return max_lr * (epoch + 1) / warmup_epochs
    return max_lr


lr_callback = LearningRateScheduler(scheduler)

# train and test splits
X_train, X_val, y_train, y_val = train_test_split(x, y, test_size=0.2, stratify=y)

# building model
model = Sequential()

model.add(LSTM(64, return_sequences=True, activation="relu", input_shape=(32, 126)))
model.add(Dropout(0.3))

# model.add(LSTM(128, return_sequences=True, activation="relu"))

model.add(LSTM(64, return_sequences=False, activation="relu"))
model.add(Dropout(0.3))

model.add(Dense(64, activation="relu"))
model.add(Dropout(0.3))

# model.add(Dense(32, activation="relu"))
model.add(Dense(len(actions), activation="softmax"))

# compiling model
model.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss=CategoricalCrossentropy(label_smoothing=0.1),
    metrics=["categorical_accuracy"],
)

print("Min:", np.min(x), "Max:", np.max(x))

# early stopping and checkpoint
early_stop = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
checkpoint = ModelCheckpoint(
    "model/best_model.h5", monitor="val_loss", save_best_only=True
)

# train model
model.fit(
    X_train,
    y_train,
    epochs=50,
    shuffle=True,
    validation_data=(X_val, y_val),
    # class_weight=class_weights,
    callbacks=[early_stop, checkpoint, lr_callback],
)

# DEBUG: Prediction Sanity Check
print("\n[DEBUG] Prediction sanity check")

sample = X_train[0:1]
prediction = model.predict(sample)

print("Prediction vector:", prediction)
print("Predicted class:", np.argmax(prediction))
print("Actual class:", np.argmax(y_train[0]))

# saving model
model.save("model/gesture_model.h5")
