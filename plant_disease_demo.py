"""
plant_disease_demo.py

Single-file script to:
 - load dataset from ./data/{train,val,test}
 - train MobileNetV2 (or load saved model)
 - save artifacts to ./saved_model and ./results
 - start a Gradio demo (robust prediction wrapper + logging)

Usage examples:
 - Train quickly: python plant_disease_demo.py --epochs 2 --batch_size 8
 - Serve only (use existing saved model): python plant_disease_demo.py --serve_only --share
"""

import os
import argparse
from pathlib import Path
import traceback
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import itertools

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.models import load_model

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

import gradio as gr
from PIL import Image as PILImage

# ---------- Config ----------
ROOT = Path(".")
DATA_DIR = ROOT / "data"
TRAIN_DIR = DATA_DIR / "train"
VAL_DIR = DATA_DIR / "val"
TEST_DIR = DATA_DIR / "test"

RESULTS_DIR = ROOT / "results"
MODEL_DIR = ROOT / "saved_model"
SAVED_H5 = MODEL_DIR / "plant_mobilenetv2.h5"
SAVED_KERAS = MODEL_DIR / "plant_mobilenetv2.keras"

IMG_SIZE = (224, 224)
INPUT_SHAPE = IMG_SIZE + (3,)

# PlantVillage dataset class names.
# Used as a fallback when the training dataset is not available locally.
PLANTVILLAGE_CLASSES = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
]

# ---------- Helpers ----------
def ensure_dirs():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

def build_model(num_classes):
    base = MobileNetV2(
        input_shape=INPUT_SHAPE,
        include_top=False,
        weights="imagenet"
    )

    # 🔓 Unfreeze top layers only (safe fine-tuning)
    for layer in base.layers[:-30]:
        layer.trainable = False
    for layer in base.layers[-30:]:
        layer.trainable = True

    x = layers.GlobalAveragePooling2D()(base.output)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    out = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs=base.input, outputs=out)

    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-5),  # very important
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


def plot_confusion_matrix(cm, classes, out_file):
    plt.figure(figsize=(6,6))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fmt = 'd'
    thresh = cm.max() / 2. if cm.size else 0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(out_file, bbox_inches='tight')
    plt.close()

def save_text_file(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")

# ---------- Predict wrapper (robust) ----------
def make_predict_fn(model_ref, class_names_ref):
    """
    Returns a robust predict_image function for Gradio that:
      - always returns (label_dict, details_string)
      - logs exceptions to results/predict_error.log
    """
    def predict_image(img):
        try:
            if img is None:
                return {"error": 1.0}, "No image provided."

            # Accept either numpy array or PIL or file-like
            if hasattr(img, "read"):  # file-like object
                pil = PILImage.open(img).convert("RGB")
            elif isinstance(img, np.ndarray):
                pil = PILImage.fromarray(img).convert("RGB")
            else:
                # last resort convert
                pil = PILImage.fromarray(np.array(img)).convert("RGB")

            pil = pil.resize(IMG_SIZE)
            arr = np.array(pil).astype("float32") / 255.0

            # ensure channels
            if arr.ndim == 2:
                arr = np.stack([arr]*3, axis=-1)
            if arr.shape[-1] == 4:
                arr = arr[..., :3]

            batch = np.expand_dims(arr, axis=0)

            # Ensure model loaded
            model = model_ref.get("model", None)
            if model is None:
                # try to load from saved h5 if present
                if Path(SAVED_H5).exists():
                    try:
                        model_ref["model"] = load_model(str(SAVED_H5))
                        model = model_ref["model"]
                    except Exception as e:
                        # log and return
                        raise RuntimeError(f"Failed to load saved model: {e}")
                else:
                    raise RuntimeError("Model not loaded and no saved model file found.")

            preds = model.predict(batch, verbose=0)
            preds = np.asarray(preds).squeeze()
            if preds.ndim != 1:
                preds = preds.reshape(-1)

            # Convert logits -> softmax probabilities (safe)
            if (preds < 0).any() or preds.sum() > 1.0001:
                e = np.exp(preds - np.max(preds))
                probs = e / e.sum()
            else:
                probs = preds / (preds.sum() + 1e-12)

            # Build class mapping
            class_names = class_names_ref.get("class_names", None)
            if not class_names:
                # fallback named indices
                class_names = [f"class_{i}" for i in range(len(probs))]

            out_dict = {class_names[i]: float(probs[i]) for i in range(len(probs))}

            top_idx = int(np.argmax(probs))
            top_class = class_names[top_idx]
            top_conf = float(probs[top_idx])

            top3_idx = np.argsort(probs)[::-1][:3]
            top3 = ", ".join([f"{class_names[i]}: {probs[i]:.3f}" for i in top3_idx])

            details = f"Top class: {top_class} ({top_conf:.3f})\nTop-3: {top3}\nRaw: {np.round(probs,4).tolist()}"

            return out_dict, details

        except Exception as exc:
            # log full traceback with timestamp
            tb = traceback.format_exc()
            log_path = RESULTS_DIR / "predict_error.log"
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"\n[{datetime.now()}] Exception in predict_image:\n{tb}\n")
            return {"error": 1.0}, f"Prediction error: {str(exc)} (see {log_path})"

    return predict_image

# ---------- Main ----------
def main(args):
    ensure_dirs()

    # If serve_only, skip training and try to load model
    serve_only = args.serve_only

    # Build / check dataset presence
    if not serve_only:
        if not TRAIN_DIR.exists() or not VAL_DIR.exists():
            print("ERROR: dataset folders missing. Expect ./data/train and ./data/val (subfolders per class).")
            return

        # Generators
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True,
            fill_mode="nearest"
        )
        val_datagen = ImageDataGenerator(rescale=1./255)

        train_gen = train_datagen.flow_from_directory(
            str(TRAIN_DIR), target_size=IMG_SIZE, batch_size=args.batch_size, class_mode="categorical", shuffle=True
        )
        val_gen = val_datagen.flow_from_directory(
            str(VAL_DIR), target_size=IMG_SIZE, batch_size=args.batch_size, class_mode="categorical", shuffle=False
        )

        num_classes = len(train_gen.class_indices)
        class_names = [k for k,v in sorted(train_gen.class_indices.items(), key=lambda x:x[1])]
        print("Classes:", train_gen.class_indices)

        # Build model and train
        model = build_model(num_classes)
        model.summary()

        checkpoint = ModelCheckpoint(str(SAVED_H5), monitor="val_accuracy", save_best_only=True, verbose=1)
        early = EarlyStopping(monitor="val_accuracy", patience=4, restore_best_weights=True, verbose=1)

        history = model.fit(train_gen, validation_data=val_gen, epochs=args.epochs, callbacks=[checkpoint, early])

        # Save model artifacts (HDF5 and .keras)
        try:
            model.save(str(SAVED_H5))
        except Exception as e:
            print("Warning: couldn't save .h5:", e)
        try:
            model.save(str(SAVED_KERAS))
        except Exception:
            pass

        # Evaluate on validation set
        val_gen.reset()
        y_true, y_pred = [], []
        for i in range(len(val_gen)):
            Xb, yb = val_gen[i]
            preds_b = model.predict(Xb)
            y_true.extend(np.argmax(yb, axis=1).tolist())
            y_pred.extend(np.argmax(preds_b, axis=1).tolist())
            if len(y_true) >= val_gen.samples:
                break

        acc = accuracy_score(y_true, y_pred) if len(y_true) else 0.0
        print(f"Validation Accuracy: {acc:.4f}")

        target_names = class_names
        report = classification_report(y_true, y_pred, target_names=target_names, digits=4)
        save_text_file(RESULTS_DIR / "classification_report.txt", report)

        cm = confusion_matrix(y_true, y_pred)
        plot_confusion_matrix(cm, target_names, RESULTS_DIR / "confusion_matrix.png")

        # Save training plots
        try:
            plt.figure()
            plt.plot(history.history.get("accuracy", []), label="train_acc")
            plt.plot(history.history.get("val_accuracy", []), label="val_acc")
            plt.legend()
            plt.title("Accuracy")
            plt.savefig(RESULTS_DIR / "training_accuracy.png")
            plt.close()
        except Exception:
            pass

    else:
        # serve_only = True
        # If serving only, attempt to find class mapping by scanning saved model or data folder
        class_names = None
        if TRAIN_DIR.exists():
            # if training data still present, build a quick generator to get class indices
            tmp = ImageDataGenerator(rescale=1./255).flow_from_directory(
                TRAIN_DIR, target_size=IMG_SIZE, batch_size=1, class_mode="categorical", shuffle=False
            )
            class_names = [k for k,v in sorted(tmp.class_indices.items(), key=lambda x:x[1])]
        else:
            # Dataset is not available, so use the built-in PlantVillage class mapping.
            class_names = PLANTVILLAGE_CLASSES
            print("Training dataset not found. Using built-in PlantVillage class names.")
            

    # Prepare model_ref and class_names_ref for predict wrapper to access and mutate
    model_ref = {"model": None}
    class_names_ref = {"class_names": class_names}

    # Try loading saved model if available
    if Path(SAVED_H5).exists():
        try:
            model_ref["model"] = load_model(str(SAVED_H5))
            print("Loaded model from", SAVED_H5)
        except Exception as e:
            print("Warning: failed to load saved .h5 model:", e)
            model_ref["model"] = None
    elif Path(SAVED_KERAS).exists():
        try:
            model_ref["model"] = load_model(str(SAVED_KERAS))
            print("Loaded model from", SAVED_KERAS)
        except Exception as e:
            print("Warning: failed to load saved .keras model:", e)
            model_ref["model"] = None
    else:
        if serve_only:
            print("Serve-only requested but no saved model found. Aborting.")
            return

    # If training was run but class_names_ref empty, try to compute it from the train generator (if present)
    if class_names_ref["class_names"] is None:
        # attempt to read from TRAIN_DIR
        if TRAIN_DIR.exists():
            dirs = sorted([d.name for d in TRAIN_DIR.iterdir() if d.is_dir()])
            if dirs:
                class_names_ref["class_names"] = dirs

    # Final fallback names if still None
    if class_names_ref["class_names"] is None and model_ref.get("model") is not None:
        # try to infer class count from model output shape
        try:
            out_shape = model_ref["model"].output_shape
            num_classes = out_shape[-1] if isinstance(out_shape, tuple) else (out_shape[0][-1] if isinstance(out_shape, list) else None)
            if num_classes:
                class_names_ref["class_names"] = [f"class_{i}" for i in range(num_classes)]
        except Exception:
            class_names_ref["class_names"] = []

    # Create the gradio predict function
    predict_image = make_predict_fn(model_ref, class_names_ref)

    # ---- Launch Gradio ----
    demo = gr.Interface(
        fn=predict_image,
        inputs=gr.Image(type="numpy", label="Upload a leaf image"),
        outputs=[
            gr.Label(num_top_classes=3, label="Predictions"),
            gr.Textbox(label="Details")
        ],
        title="Plant Leaf Disease Detector (MobileNetV2)",
        description="Upload a leaf image and the model will predict disease/health class."
    )

    print("Launching Gradio demo...")
    demo.launch(share=args.share, server_name="0.0.0.0", server_port=args.port)

# ---------- CLI ----------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=6, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--share", action="store_true", help="Create a public Gradio share link")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--serve_only", action="store_true", help="Skip training and only serve an existing saved model")
    args = parser.parse_args()
    main(args)
