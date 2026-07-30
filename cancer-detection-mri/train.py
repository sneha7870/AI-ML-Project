"""
Cancer Detection using MRI Images
------------------------------------
Classifies brain MRI scans as tumor / no-tumor (or multi-class tumor type)
using a CNN with transfer learning (VGG16 backbone).

Expects data in ImageFolder-style structure:

    data/
        Training/
            glioma/
            meningioma/
            pituitary/
            notumor/
        Testing/
            glioma/
            meningioma/
            pituitary/
            notumor/

This matches the popular "Brain Tumor MRI Dataset" on Kaggle
(https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset).
Download it, unzip so it matches the structure above, and point DATA_DIR to it.
Works fine for binary (tumor/notumor) too — just use 2 subfolders instead of 4.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix

tf.random.set_seed(42)
np.random.seed(42)

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
DATA_DIR = "data"                      # expects data/Training and data/Testing
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 30

TRAIN_DIR = os.path.join(DATA_DIR, "Training")
TEST_DIR = os.path.join(DATA_DIR, "Testing")

if not os.path.isdir(TRAIN_DIR):
    raise FileNotFoundError(
        f"'{TRAIN_DIR}' not found. Download the Brain Tumor MRI Dataset from Kaggle "
        "and arrange it as data/Training/<class>/*.jpg and data/Testing/<class>/*.jpg "
        "(see the docstring at the top of this file)."
    )

# ---------------------------------------------------------------------------
# 1. DATA GENERATORS
# ---------------------------------------------------------------------------
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    validation_split=0.15,
)
test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_gen = train_datagen.flow_from_directory(
    TRAIN_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode="categorical", subset="training", shuffle=True, seed=42,
)
val_gen = train_datagen.flow_from_directory(
    TRAIN_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode="categorical", subset="validation", shuffle=False, seed=42,
)
test_gen = test_datagen.flow_from_directory(
    TEST_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode="categorical", shuffle=False,
)

class_names = list(train_gen.class_indices.keys())
n_classes = len(class_names)
print(f"Classes: {class_names}")
print(f"Train: {train_gen.samples}, Val: {val_gen.samples}, Test: {test_gen.samples}")

# Preview a batch
imgs, labels = next(train_gen)
fig, axes = plt.subplots(2, 5, figsize=(14, 6))
for i, ax in enumerate(axes.flat):
    img = (imgs[i] - imgs[i].min()) / (imgs[i].max() - imgs[i].min() + 1e-8)
    ax.imshow(img)
    ax.set_title(class_names[np.argmax(labels[i])], fontsize=9)
    ax.axis("off")
plt.tight_layout()
plt.savefig("sample_mri_images.png", dpi=150)
plt.close()

# ---------------------------------------------------------------------------
# 2. BUILD MODEL (Transfer Learning: VGG16 backbone)
# ---------------------------------------------------------------------------
base_model = VGG16(weights="imagenet", include_top=False, input_shape=(*IMG_SIZE, 3))
base_model.trainable = False  # freeze for phase 1

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation="relu"),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(n_classes, activation="softmax"),
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
)
model.summary()

callbacks = [
    EarlyStopping(monitor="val_accuracy", patience=8, restore_best_weights=True),
    ModelCheckpoint("best_mri_model.keras", monitor="val_accuracy", save_best_only=True),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-7),
]

# ---------------------------------------------------------------------------
# 3. PHASE 1: TRAIN CLASSIFIER HEAD (backbone frozen)
# ---------------------------------------------------------------------------
history1 = model.fit(
    train_gen, validation_data=val_gen, epochs=EPOCHS, callbacks=callbacks,
)

# ---------------------------------------------------------------------------
# 4. PHASE 2: FINE-TUNE (unfreeze last few VGG16 blocks)
# ---------------------------------------------------------------------------
base_model.trainable = True
for layer in base_model.layers[:-4]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss="categorical_crossentropy",
    metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
)

history2 = model.fit(
    train_gen, validation_data=val_gen, epochs=15, callbacks=callbacks,
)

# ---------------------------------------------------------------------------
# 5. EVALUATE ON TEST SET
# ---------------------------------------------------------------------------
test_loss, test_acc, test_auc = model.evaluate(test_gen, verbose=0)
print(f"\nTest Accuracy: {test_acc:.4f} | Test AUC: {test_auc:.4f} | Test Loss: {test_loss:.4f}")

y_pred_probs = model.predict(test_gen)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = test_gen.classes

print("\n--- Classification Report ---")
print(classification_report(y_true, y_pred, target_names=class_names))

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("MRI Tumor Classification - Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.close()

# Combined training curves (phase 1 + phase 2)
acc = history1.history["accuracy"] + history2.history["accuracy"]
val_acc = history1.history["val_accuracy"] + history2.history["val_accuracy"]
loss = history1.history["loss"] + history2.history["loss"]
val_loss = history1.history["val_loss"] + history2.history["val_loss"]

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(acc, label="train"); axes[0].plot(val_acc, label="val")
axes[0].axvline(len(history1.history["accuracy"]), color="gray", linestyle="--", label="fine-tune start")
axes[0].set_title("Accuracy"); axes[0].legend()
axes[1].plot(loss, label="train"); axes[1].plot(val_loss, label="val")
axes[1].axvline(len(history1.history["loss"]), color="gray", linestyle="--")
axes[1].set_title("Loss"); axes[1].legend()
plt.tight_layout()
plt.savefig("training_curves.png", dpi=150)
plt.close()

# ---------------------------------------------------------------------------
# 6. SAVE MODEL
# ---------------------------------------------------------------------------
model.save("mri_cancer_detection_final.keras")
with open("class_names.txt", "w") as f:
    f.write("\n".join(class_names))
print("Saved mri_cancer_detection_final.keras and class_names.txt")
print("Done.")
