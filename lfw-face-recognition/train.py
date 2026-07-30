"""
Face Recognition using CNN on the LFW ("Labeled Faces in the Wild") dataset.
------------------------------------------------------------------------------
Classifies grayscale face images of well-known people using a CNN.
Only people with a minimum number of images are kept, to have enough
samples per class for training.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

tf.random.set_seed(42)
np.random.seed(42)

# ---------------------------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------------------------
# min_faces_per_person filters out people with too few images so every class
# has a reasonable number of training examples. resize keeps images small
# and fast to train on.
lfw = fetch_lfw_people(min_faces_per_person=40, resize=0.5, color=False)

X = lfw.images  # shape: (n_samples, h, w)
y = lfw.target
target_names = lfw.target_names
n_classes = target_names.shape[0]
h, w = X.shape[1], X.shape[2]

print(f"Samples: {X.shape[0]}, Image size: {h}x{w}, Classes: {n_classes}")
print("Class distribution:")
for i, name in enumerate(target_names):
    print(f"  {name}: {(y == i).sum()}")

# ---------------------------------------------------------------------------
# 2. PREPROCESS
# ---------------------------------------------------------------------------
X = X.astype("float32") / 255.0
X = X[..., np.newaxis]  # add channel dim -> (N, h, w, 1)
y_cat = to_categorical(y, n_classes)

X_train, X_test, y_train, y_test, y_train_idx, y_test_idx = train_test_split(
    X, y_cat, y, test_size=0.2, stratify=y, random_state=42
)

# Preview some faces
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
for i, ax in enumerate(axes.flat):
    ax.imshow(X_train[i].squeeze(), cmap="gray")
    ax.set_title(target_names[y_train_idx[i]], fontsize=9)
    ax.axis("off")
plt.tight_layout()
plt.savefig("sample_faces.png", dpi=150)
plt.close()

# ---------------------------------------------------------------------------
# 3. DATA AUGMENTATION (LFW is small & imbalanced across identities)
# ---------------------------------------------------------------------------
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
)
datagen.fit(X_train)

# ---------------------------------------------------------------------------
# 4. BUILD CNN
# ---------------------------------------------------------------------------
def build_model(input_shape, n_classes):
    model = models.Sequential([
        layers.Input(shape=input_shape),

        layers.Conv2D(32, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        layers.Conv2D(128, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),

        layers.Flatten(),
        layers.Dense(256, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(n_classes, activation="softmax"),
    ])
    return model

model = build_model((h, w, 1), n_classes)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)
model.summary()

# ---------------------------------------------------------------------------
# 5. TRAIN
# ---------------------------------------------------------------------------
callbacks = [
    EarlyStopping(monitor="val_accuracy", patience=10, restore_best_weights=True),
    ModelCheckpoint("best_lfw_model.keras", monitor="val_accuracy", save_best_only=True),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-6),
]

EPOCHS = 60
BATCH_SIZE = 32

history = model.fit(
    datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
    steps_per_epoch=max(1, len(X_train) // BATCH_SIZE),
    epochs=EPOCHS,
    validation_data=(X_test, y_test),
    callbacks=callbacks,
)

# ---------------------------------------------------------------------------
# 6. EVALUATE
# ---------------------------------------------------------------------------
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest Accuracy: {test_acc:.4f} | Test Loss: {test_loss:.4f}")

y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)

print("\n--- Classification Report ---")
print(classification_report(y_test_idx, y_pred, target_names=target_names))

cm = confusion_matrix(y_test_idx, y_pred)
plt.figure(figsize=(8, 7))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=target_names, yticklabels=target_names)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("LFW Face Recognition Confusion Matrix")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.close()

# Training curves
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(history.history["accuracy"], label="train")
axes[0].plot(history.history["val_accuracy"], label="val")
axes[0].set_title("Accuracy")
axes[0].legend()
axes[1].plot(history.history["loss"], label="train")
axes[1].plot(history.history["val_loss"], label="val")
axes[1].set_title("Loss")
axes[1].legend()
plt.tight_layout()
plt.savefig("training_curves.png", dpi=150)
plt.close()

# Show a few predictions vs actual
fig, axes = plt.subplots(2, 5, figsize=(14, 6))
for i, ax in enumerate(axes.flat):
    ax.imshow(X_test[i].squeeze(), cmap="gray")
    pred_name = target_names[y_pred[i]]
    true_name = target_names[y_test_idx[i]]
    color = "green" if pred_name == true_name else "red"
    ax.set_title(f"Pred: {pred_name}\nTrue: {true_name}", fontsize=8, color=color)
    ax.axis("off")
plt.tight_layout()
plt.savefig("sample_predictions.png", dpi=150)
plt.close()

# ---------------------------------------------------------------------------
# 7. SAVE FINAL MODEL
# ---------------------------------------------------------------------------
model.save("lfw_face_recognition_final.keras")
np.save("target_names.npy", target_names)
print("Saved lfw_face_recognition_final.keras and target_names.npy")
print("Done.")
