import os
import json
import numpy as np

from data_loader import load_data
from preprocessing import preprocess_data
from split_data import split_dataset
from cnn_model import train_model, predict_model
from evaluate import evaluate_model, plot_history

# ตั้งค่า Path ให้อ้างอิงตามโฟลเดอร์ Dataset และ m-project
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "Dataset")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

# ไฟล์ Dataset ของเรา
FILE_1 = os.path.join(DATA_DIR, "zoo2.csv")
FILE_2 = os.path.join(DATA_DIR, "zoo3.csv")

def main():
    print("--" * 25)
    print("Zoo Dataset Classification (DNN Pipeline)")
    print("--" * 25)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. Load Data
    print("\n[Step 1] Loading data...")
    df = load_data(FILE_1, FILE_2)
    
    # 2. Preprocess Data
    print("\n[Step 2] Preprocessing data...")
    X, y, classes, num_classes = preprocess_data(df)
    
    # บันทึก classes name ไว้ใช้เทสทีหลัง
    with open(f"{OUTPUT_DIR}/classes.json", "w") as f:
        json.dump(classes, f)
        
    print(f"Features shape: {X.shape}")
    print(f"Classes ({num_classes}): {classes}")
        
    # 3. Split Dataset
    print("\n[Step 3] Splitting dataset...")
    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(X, y)
    
    # บันทึก Test set ไว้ใช้กับไฟล์ test_cnn.py
    np.save(f"{OUTPUT_DIR}/X_test.npy", X_test)
    np.save(f"{OUTPUT_DIR}/y_test.npy", y_test)
    
    print(f"Training set   : {len(X_train)} samples")
    print(f"Validation set : {len(X_val)} samples")
    print(f"Testing set    : {len(X_test)} samples")
    
    # 4. Train Model
    print("\n[Step 4] Training model...")
    model, history = train_model(X_train, y_train, X_val, y_val, num_classes, OUTPUT_DIR, epochs=500)
    
    # 5. Evaluate Model
    print("\n[Step 5] Evaluating model...")
    predictions = predict_model(model, X_test)
    evaluate_model(y_test, predictions, classes, save_path=f"{OUTPUT_DIR}/confusion_matrix.png")
    plot_history(history, f"{OUTPUT_DIR}/training_history.png")
    
    print("\nPipeline completed successfully!")

if __name__ == "__main__":
    main()