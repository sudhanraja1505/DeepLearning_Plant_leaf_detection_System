import os
import numpy as np
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from PIL import Image
import tensorflow as tf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "plant_disease_model.keras")
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}
IMG_SIZE = (224, 224)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Replace this list with the exact class names and order used when training your model.
CLASS_NAMES = [
    "Healthy",
    "Leaf Disease"
]

model = None
if os.path.exists(MODEL_PATH):
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
    except Exception as exc:
        print(f"Could not load model: {exc}")
else:
    print("No trained model found. Follow README.md to train one.")

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def prepare_image(image_path):
    image = Image.open(image_path).convert("RGB").resize(IMG_SIZE)
    array = np.asarray(image, dtype=np.float32) / 255.0
    return np.expand_dims(array, axis=0)

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    image_url = None
    error = None

    if request.method == "POST":
        uploaded = request.files.get("leaf_image")
        if not uploaded or uploaded.filename == "":
            error = "Please choose a leaf image."
        elif not allowed_file(uploaded.filename):
            error = "Upload a PNG, JPG, JPEG, or WEBP image."
        elif model is None:
            error = "The trained model is not installed yet. Follow the training steps in README.md."
        else:
            filename = secure_filename(uploaded.filename)
            # Add a unique prefix to avoid overwriting uploads with the same filename.
            import uuid
            filename = f"{uuid.uuid4().hex}_{filename}"
            path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            uploaded.save(path)

            try:
                predictions = model.predict(prepare_image(path), verbose=0)[0]
                class_index = int(np.argmax(predictions))
                result = {
                    "label": CLASS_NAMES[class_index] if class_index < len(CLASS_NAMES) else f"Class {class_index}",
                    "confidence": float(predictions[class_index]) * 100
                }
                image_url = f"/static/uploads/{filename}"
            except Exception as exc:
                error = f"Prediction failed. Check that the model and class labels match. Details: {exc}"

    return render_template("index.html", result=result, image_url=image_url, error=error)

@app.errorhandler(413)
def file_too_large(_error):
    return render_template("index.html", result=None, image_url=None,
                           error="File is too large. Maximum upload size is 8 MB."), 413

if __name__ == "__main__":
    # Development only. Use a production WSGI server when deploying.
    app.run(debug=True)
