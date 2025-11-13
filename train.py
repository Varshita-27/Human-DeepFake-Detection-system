# ====== train.py (Windows safe, no multiprocessing) ======
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ---- Paths ----
BASE_DIR = Path(__file__).resolve().parent.parent # project root
TRAIN_DIR = BASE_DIR / "Dataset" / "Dataset" / "Train"
VAL_DIR   = BASE_DIR / "Dataset" / "Dataset" / "Validation"
MODEL_OUT = BASE_DIR / "deepfake_detection_model.h5"

BATCH = 32        # stable
EPOCHS = 1
STEPS = 100       # faster for first run
VAL_STEPS = 30

def run_training():
    print("Train dir :", TRAIN_DIR)
    print("Val dir   :", VAL_DIR)

    train_gen = ImageDataGenerator(rescale=1./255, horizontal_flip=True)
    val_gen   = ImageDataGenerator(rescale=1./255)

    train = train_gen.flow_from_directory(
        TRAIN_DIR, target_size=(96, 96), batch_size=BATCH, class_mode="binary")

    val = val_gen.flow_from_directory(
        VAL_DIR, target_size=(96, 96), batch_size=BATCH, class_mode="binary")

    base = MobileNetV2(include_top=False, weights="imagenet", input_shape=(96, 96, 3))
    base.trainable = False

    model = Sequential([
        base,
        GlobalAveragePooling2D(),
        Dense(256, activation="relu"),
        BatchNormalization(),
        Dropout(0.3),
        Dense(1, activation="sigmoid"),
    ])

    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    model.summary()

    history = model.fit(
        train,
        epochs=EPOCHS,
        steps_per_epoch=STEPS,
        validation_data=val,
        validation_steps=VAL_STEPS,
        workers=1,                    # ✅ IMPORTANT
        use_multiprocessing=False     # ✅ IMPORTANT
    )

    model.save(MODEL_OUT)
    print("✅ Model saved at:", MODEL_OUT)

if __name__ == "__main__":
    run_training()

