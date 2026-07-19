# Plant Leaf Disease Detector 🌿

A deep learning-based plant leaf disease detection system built using **TensorFlow**, **MobileNetV2**, and **Gradio**.

The application allows users to upload an image of a plant leaf and predicts the most likely disease or healthy class along with confidence scores.

## Features

- Detects plant leaf diseases from uploaded images
- Supports 38 plant disease and healthy classes
- Uses MobileNetV2 transfer learning
- Displays Top-3 predictions with confidence scores
- Interactive Gradio web interface
- Includes a pre-trained model for direct inference
- Supports optional model retraining

## Tech Stack

- Python
- TensorFlow / Keras
- MobileNetV2
- NumPy
- scikit-learn
- Matplotlib
- Gradio
- Pillow

## Project Structure

```text
plant-disease/
├── plant_disease_demo.py
├── requirements.txt
├── README.md
├── saved_model/
├── results/
├── counts.py
├── inspect_data.py
├── make_dummy_data.py
└── split_dataset.py
```

### Main Files

- `plant_disease_demo.py` - Main training and Gradio application
- `saved_model/` - Pre-trained MobileNetV2 model
- `results/` - Model evaluation results and sample predictions
- `split_dataset.py` - Dataset splitting utility
- `inspect_data.py` - Dataset inspection utility
- `counts.py` - Dataset counting utility
- `make_dummy_data.py` - Utility for generating dummy/test data
- `requirements.txt` - Python dependencies

The full dataset is not included in this repository because of its large size.

## Quick Start - Run the Pre-trained Model

### 1. Clone the repository

```bash
git clone https://github.com/Naimish314/Plant_leaf_disease_detection.git
cd Plant_leaf_disease_detection
```

### 2. Install the dependencies

```bash
pip install -r requirements.txt
```

### 3. Start the application

The repository includes a pre-trained model, so you do not need to train the model again.

Run:

```bash
python plant_disease_demo.py --serve_only
```

The Gradio application will start locally. Open the URL shown in the terminal, usually:

```text
http://localhost:7860
```

Upload a plant leaf image and click **Submit** to view the Top-3 predicted disease or healthy classes along with their confidence scores.

## Training the Model

Retraining is optional and requires the PlantVillage dataset.

Prepare the dataset using the following directory structure:

```text
data/
├── train/
│   ├── class_1/
│   ├── class_2/
│   └── ...
├── val/
│   ├── class_1/
│   ├── class_2/
│   └── ...
└── test/
    ├── class_1/
    ├── class_2/
    └── ...
```

Then run:

```bash
python plant_disease_demo.py --epochs 6 --batch_size 16
```

Training can take a significant amount of time depending on the hardware being used.

## Dataset

The model was trained using the **PlantVillage dataset**, containing images across **38 plant disease and healthy classes**.

The full dataset is not included in this repository due to its large size.

## Model

The project uses **MobileNetV2** with transfer learning for plant leaf image classification.

A trained model is included in the `saved_model/` directory, allowing the Gradio application to run directly without retraining the model.

## How It Works

1. The user uploads an image of a plant leaf.
2. The image is resized and preprocessed.
3. The trained MobileNetV2 model processes the image.
4. The model generates probabilities for the supported classes.
5. The application displays the Top-3 predictions with confidence scores.

## Disclaimer

This project is intended for educational and experimental purposes. Model predictions should not be considered a substitute for professional agricultural or plant pathology advice.