"""
Train a simple CNN using a folder-based image dataset.

Expected structure:
dataset/
  train/
    Tomato___healthy/
    Tomato___Early_blight/
    ...
  validation/
    Tomato___healthy/
    Tomato___Early_blight/
    ...

The folder names become the model's class labels. The app's CLASS_NAMES list must
match the alphabetically sorted class folder names printed after training.
"""
import os
import tensorflow as tf

DATASET_DIR = "dataset"
TRAIN_DIR = os.path.join(DATASET_DIR, "train")
VAL_DIR = os.path.join(DATASET_DIR, "validation")
MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "plant_disease_model.keras")
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10

if not os.path.isdir(TRAIN_DIR) or not os.path.isdir(VAL_DIR):
    raise SystemExit(
        "Dataset folders not found. Create dataset/train and dataset/validation, "
        "with one subfolder per class. See README.md."
    )

train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR, image_size=IMG_SIZE, batch_size=BATCH_SIZE,
    label_mode="categorical", shuffle=True, seed=42
)
val_ds = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR, image_size=IMG_SIZE, batch_size=BATCH_SIZE,
    label_mode="categorical", shuffle=False
)

class_names = train_ds.class_names
print("Class names in model order:", class_names)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.map(lambda x, y: (tf.cast(x, tf.float32) / 255.0, y)).prefetch(AUTOTUNE)
val_ds = val_ds.map(lambda x, y: (tf.cast(x, tf.float32) / 255.0, y)).prefetch(AUTOTUNE)

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.08),
    tf.keras.layers.RandomZoom(0.1),
])

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(224, 224, 3)),
    data_augmentation,
    tf.keras.layers.Conv2D(32, 3, activation="relu"),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, 3, activation="relu"),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(128, 3, activation="relu"),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(len(class_names), activation="softmax")
])

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
os.makedirs(MODEL_DIR, exist_ok=True)
callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True, monitor="val_loss"),
    tf.keras.callbacks.ModelCheckpoint(MODEL_PATH, save_best_only=True, monitor="val_loss")
]
model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=callbacks)
model.save(MODEL_PATH)

with open(os.path.join(MODEL_DIR, "class_names.txt"), "w", encoding="utf-8") as file:
    file.write("\n".join(class_names))

print(f"Model saved to {MODEL_PATH}")
print("Copy the class names above into CLASS_NAMES in app.py, in exactly the same order.")
