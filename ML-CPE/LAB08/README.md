# 🐾 ML-08: Deep Convolutional Neural Network (DCNN)

This project implements a VGG-based Deep Convolutional Neural Network (DCNN) for image classification. It is designed to classify images into three distinct categories: **Cat**, **Dog**, and **Wild**. The project features a fully modular Python pipeline covering everything from automatic dataset loading and preprocessing to model training, evaluation, and prediction.

## 💾 Dataset
The image dataset used in this project is **Animal Faces (AFHQ)**, sourced from Kaggle. The dataset contains high-quality images of animal faces divided into three categories: dogs, cats, and wildlife.
- **Source:** [Kaggle - Animal Faces (AFHQ)](https://www.kaggle.com/datasets/andrewmvd/animal-faces)

## 📂 Project Structure

The repository is structured to separate the dataset from the source code for better organization:

```text
LAB08/
│
├── Dataset/                     
│   └── Animal/
│       ├── cat/            # Contains cat images
│       ├── dog/            # Contains dog images
│       └── wild/           # Contains wild animal images
│
└── mini-proj/
    ├── main.py             # Main execution pipeline (Entry point)
    ├── data_loader.py      # Automatically detects classes and loads images
    ├── preprocessing.py    # Resizes images and converts BGR to RGB
    ├── split_data.py       # Splits data into Training, Validation, and Test sets
    ├── vgg_model.py        # Defines the VGG architecture and training logic
    ├── evaluate.py         # Generates accuracy metrics, classification report, confusion matrix, and plots
    ├── test_vgg.py         # Tests the trained model using random unseen images
    │
    └── outputs/            # Directory for generated files (Auto-generated)
        ├── confusion_matrix.png   # Visualized confusion matrix with colorbar
        ├── training_history.png   # Accuracy and loss graphs
        └── prediction_sample.png  # Sample predictions output
```

## 🚀 How to Use

### 1. Training and Evaluation
To start the data loading, preprocessing, model training, and evaluation process, run the main script. The script is configured to use a `64x64` image size and will automatically detect the classes in the `Dataset/Animal` directory.

```bash
python main.py
```
*Note: You can modify hyperparameters such as `EPOCHS`, `BATCH_SIZE`, and `IMG_SIZE` directly inside `main.py`.*

### 2. Testing the Model (Prediction)
After the model is successfully trained and saved, you can run the test script. This will randomly select 4 images from the test set, predict their classes, display the confidence percentages, and generate a result image (`prediction_sample.png`).

```bash
python test_vgg.py
```

## 📊 Outputs & Evaluation

Upon running `main.py`, the pipeline evaluates the model and generates visual reports in the `outputs/` folder:
- **Training History:** Graphs displaying accuracy and loss over epochs (useful for identifying overfitting or early stopping points).
- **Confusion Matrix:** A grid showing the true vs. predicted classifications to analyze the model's performance on individual classes.
- **Classification Report:** Precision, recall, and f1-score printed in the console.
