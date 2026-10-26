# Plant Leaf Disease Detection Using Deep Learning

A beginner-friendly web application that uses a Convolutional Neural Network (CNN) to classify plant leaf images. The user uploads an image in the Flask web interface and receives the predicted class and model confidence.

> **Important:** The ZIP contains the complete application code and a training script, but does not include a trained model or image dataset. Train the model using your own compatible dataset before predictions will work. Predictions are educational and should not be treated as definitive agricultural diagnoses.

## Features

- Clean responsive web interface
- Upload PNG, JPG, JPEG, or WEBP leaf images
- CNN model training script
- Predicted class and confidence display
- File size and extension checks
- TensorFlow/Keras model support

## Technologies

- Python
- Flask
- TensorFlow / Keras
- NumPy and Pillow
- HTML and CSS

## Folder Structure

```text
plant_leaf_disease_detection/
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
├── templates/
│   └── index.html
├── static/
│   ├── style.css
│   └── uploads/
├── model/
└── dataset/
    ├── train/
    │   ├── Tomato___healthy/
    │   └── Tomato___Early_blight/
    └── validation/
        ├── Tomato___healthy/
        └── Tomato___Early_blight/
```

The dataset example shows only two classes. You can use a dataset with more classes; the training script reads class names from the folder names.

## 1. Install Python

Install a Python version compatible with the TensorFlow version available for your operating system. Python 3.10 or 3.11 is a practical choice for many TensorFlow setups. Check TensorFlow's official installation instructions if installation fails.

## 2. Open the project

Extract the ZIP file and open the extracted `plant_leaf_disease_detection` folder in VS Code.

## 3. Create and activate a virtual environment

Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, use Command Prompt:

```bat
py -3.11 -m venv .venv
.venv\Scripts\activate.bat
```

If the `py -3.11` command is not available, use `python` if a compatible Python version is installed.

## 4. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

TensorFlow can be a large installation and may require a compatible Python version and operating system. If installation fails, check the official TensorFlow installation guide.

## 5. Prepare the dataset

Place your labeled leaf images in the following folders:

```text
dataset/
├── train/
│   ├── Tomato___healthy/
│   └── Tomato___Early_blight/
└── validation/
    ├── Tomato___healthy/
    └── Tomato___Early_blight/
```

Put training images in `train` and separate validation images in `validation`. Every class folder should contain examples of that class. You can obtain a plant leaf image dataset, such as PlantVillage, from a reputable source. Follow the dataset's license and usage terms.

Use the same class folders in both `train` and `validation`. The folder names are used as labels, sorted alphabetically by Keras.

## 6. Train the model

From the project root, run:

```bash
python train_model.py
```

The script trains a basic CNN and saves the best model to:

```text
model/plant_disease_model.keras
```

It also writes the class names to `model/class_names.txt`.

### Match class labels in the application

After training, open `model/class_names.txt`. Copy its class names in the same order into `CLASS_NAMES` in `app.py`. For example, if the file contains:

```text
Tomato___Early_blight
Tomato___healthy
```

then set:

```python
CLASS_NAMES = [
    "Tomato___Early_blight",
    "Tomato___healthy"
]
```

**The label order must exactly match the order printed by the training script.** Otherwise, the displayed label may be incorrect.

## 7. Run the web application

```bash
python app.py
```

Open `http://127.0.0.1:5000` in your browser, select a leaf image, and click **Analyze image**.

## Troubleshooting

- **Model not installed:** Run `python train_model.py` after preparing the dataset.
- **Wrong class shown:** Ensure `CLASS_NAMES` in `app.py` matches `model/class_names.txt` exactly.
- **Dataset not found:** Check the folder names and run the training script from the project root.
- **TensorFlow installation error:** Check Python and TensorFlow compatibility for your operating system.
- **Low accuracy:** Use clean, correctly labeled images, keep a separate validation set, and train for more epochs if appropriate.
- **Upload error:** Use a supported image format under 8 MB.

## Security and deployment notes

This application is intended for local educational use. Before public deployment, disable Flask debug mode, configure a production WSGI server, add authentication and rate limits, validate uploads more thoroughly, and set upload retention/cleanup rules. Do not upload confidential images. Never expose secrets in source control.

## Future Enhancements

- Transfer learning with MobileNetV2 or EfficientNet
- More plant species and disease classes
- Prediction history and downloadable reports
- User authentication
- Cloud deployment and cloud image storage
- Multilingual interface

## Disclaimer

This project is for learning and demonstration. Model predictions depend on the training dataset, image quality, and model performance. Consult a qualified agricultural expert before making crop-treatment decisions.
