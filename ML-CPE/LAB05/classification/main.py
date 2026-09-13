import json
import os
import joblib
import numpy as np
import matplotlib

# ใช้โหมด "Agg" เพื่อให้ Matplotlib สร้างกราฟและเซฟเป็นไฟล์ภาพได้เลย 
# โดยไม่ต้องเปิดหน้าต่าง GUI ขึ้นมา (เหมาะสำหรับการรันสคริปต์อัตโนมัติ)
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# นำเข้าฟังก์ชันจากไฟล์อื่นๆ ในโปรเจกต์ที่เราสร้างไว้
from data_load import load_data
from preprocess import preprocess_data
from split_data import split_dataset
from svm_model import train_svm, predict_svm
from evaluate import evaluate_model

# กำหนดเส้นทางไฟล์ (Path) แบบ Dynamic เพื่อให้รันเครื่องไหนก็ไม่พัง
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # ดึงที่อยู่ปัจจุบันของไฟล์ main.py
DATA_PATH = os.path.join(BASE_DIR, "..", "Dataset")   # ถอยกลับไป 1 โฟลเดอร์เพื่อเข้าถึงโฟลเดอร์ Dataset
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")        # กำหนดโฟลเดอร์สำหรับเก็บผลลัพธ์

TEST_SIZE = 0.2 # กำหนดสัดส่วนข้อมูลทดสอบที่ 20% (Train 80% / Test 20%)

def main():
    print("=" * 60)
    print("SVM Classification: Zoo Animals")
    print("=" * 60)

    # สร้างโฟลเดอร์ outputs/ หากยังไม่มี (exist_ok=True คือถ้ามีอยู่แล้วไม่ต้องสร้างใหม่และไม่แจ้ง Error)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # =========================================
    # Step 1: Load Dataset (โหลดข้อมูล)
    # =========================================
    print("\n[Step 1] Loading dataset...")

    # รับค่า Features (X), Labels (y) และรายชื่อคลาสทั้งหมด
    X, y, classes = load_data(DATA_PATH)

    print("\nDataset loaded successfully.")
    print(f"Total samples : {len(X)}")       # จำนวนข้อมูลสัตว์ทั้งหมด
    print(f"Features      : {X.shape[1]}")   # จำนวนคุณลักษณะ (Features) ของสัตว์แต่ละตัว
    print(f"Classes       : {classes}")      # ประเภทของสัตว์ทั้งหมด

    # บันทึกข้อมูลดิบที่โหลดมาเป็นไฟล์ .npy (Numpy Array) เพื่อให้โหลดใช้งานครั้งต่อไปได้ไวขึ้น
    np.save(f"{OUTPUT_DIR}/features.npy", X)
    np.save(f"{OUTPUT_DIR}/labels.npy", y)

    # บันทึกรายชื่อคลาสเป็นไฟล์ .json
    with open(f"{OUTPUT_DIR}/classes.json", "w") as f:
        json.dump([int(c) for c in classes], f)

    # =========================================
    # Step 2: Preprocessing (เตรียมข้อมูล)
    # =========================================
    print("\n[Step 2] Preprocessing data...")

    # ปรับชนิดข้อมูลให้เป็น float32 (อยู่ในไฟล์ preprocess.py)
    X = preprocess_data(X)
    print(f"Feature shape: {X.shape}")

    # =========================================
    # Step 3: Split Dataset (แบ่งชุดข้อมูล)
    # =========================================
    print("\n[Step 3] Splitting dataset...")

    # แบ่งข้อมูลเป็นชุด Train และ Test ตามสัดส่วน 80:20
    X_train, X_test, y_train, y_test = split_dataset(X, y, TEST_SIZE)

    # บันทึกชุดข้อมูลที่ถูกแบ่งแล้ว เพื่อใช้สำหรับการทดสอบ (test_svm.py) หรือวิเคราะห์ย้อนหลัง
    np.save(f"{OUTPUT_DIR}/X_train.npy", X_train)
    np.save(f"{OUTPUT_DIR}/X_test.npy", X_test)
    np.save(f"{OUTPUT_DIR}/y_train.npy", y_train)
    np.save(f"{OUTPUT_DIR}/y_test.npy", y_test)

    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples : {len(X_test)}")

    # =========================================
    # Step 4: Train SVM (สอนโมเดล)
    # =========================================
    print("\n[Step 4] Training SVM models...")

    # โค้ดส่วนนี้จะคืนค่า (Return) เป็น Dictionary ที่เก็บโมเดล SVM ทั้ง 3 แบบ (Linear, Poly, RBF)
    models = train_svm(X_train, y_train)

    # วนลูปเพื่อบันทึกโมเดล (Serialization) เก็บไว้เป็นไฟล์ .pkl ด้วยไลบรารี joblib
    # เพื่อให้นำโมเดลไปใช้งานจริง (Deploy/Test) ได้โดยไม่ต้องเทรนใหม่
    for name, model in models.items():
        joblib.dump(model, f"{OUTPUT_DIR}/svm_{name}.pkl")

    print("\nSVM training completed.")

    # =========================================
    # Step 5: Prediction (ทำนายผล)
    # =========================================
    print("\n[Step 5] Predicting test data...")

    # นำข้อมูล Test Set เข้าไปทดสอบกับโมเดลทั้ง 3 ตัว
    predictions = predict_svm(models, X_test)

    # =========================================
    # Step 6: Evaluation (ประเมินผลประสิทธิภาพ)
    # =========================================
    print("\n[Step 6] Evaluating models...")

    results = {}

    # นำผลทำนาย (predictions) มาเทียบกับความจริง (y_test) เพื่อหาค่าความแม่นยำ
    for name in models:
        accuracy = evaluate_model(
            y_test,
            predictions[name],
            classes,
            name,
            # สั่งให้สร้างและเซฟรูป Confusion Matrix ของแต่ละโมเดล
            save_path=f"{OUTPUT_DIR}/confusion_matrix_{name}.png"
        )
        # เก็บค่าความแม่นยำของแต่ละโมเดลไว้ใน Dictionary
        results[name] = accuracy

    # =========================================
    # Step 7: Compare Accuracy (เปรียบเทียบและสร้างกราฟ)
    # =========================================
    print("\n" + "=" * 60)
    print("SVM KERNEL COMPARISON")
    print("=" * 60)

    # พรินต์ผลลัพธ์ความแม่นยำออกมาที่หน้าจอ
    for kernel, accuracy in results.items():
        print(f"{kernel.upper():<12} {accuracy * 100:.2f}%")

    # --- ส่วนของการสร้างกราฟแท่ง (Bar Chart) ---
    kernels = list(results.keys())       # ชื่อโมเดลแกน X (Linear, Poly, RBF)
    accuracies = list(results.values())  # ค่าความแม่นยำแกน Y

    plt.figure(figsize=(8, 5))           # กำหนดขนาดกราฟ

    plt.bar(kernels, accuracies)         # สร้างกราฟแท่ง

    plt.title("SVM Kernel Accuracy Comparison")
    plt.xlabel("Kernel")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1.0)                     # กำหนดแกน Y ให้อยู่ระหว่าง 0 ถึง 1 (0% - 100%)

    # วนลูปเพื่อเขียนตัวเลขเปอร์เซ็นต์กำกับไว้ด้านบนสุดของแต่ละแท่ง
    for i, value in enumerate(accuracies):
        plt.text(
            i, 
            value + 0.02,                # ขยับตัวเลขขึ้นไปเหนือแท่งเล็กน้อย
            f"{value * 100:.2f}%",       # แปลงเป็นเปอร์เซ็นต์ทศนิยม 2 ตำแหน่ง
            ha="center"                  # จัดให้อยู่กึ่งกลาง
        )

    plt.tight_layout() # จัดขอบกราฟให้พอดี
    
    # เซฟกราฟเป็นไฟล์ภาพ PNG
    plt.savefig(f"{OUTPUT_DIR}/accuracy_comparison.png", dpi=150)
    plt.close() # ปิดการทำงานของกราฟเพื่อคืนหน่วยความจำ

    print("\nSaved: outputs/accuracy_comparison.png")

if __name__ == "__main__":
    # เช็กว่าไฟล์นี้ถูกกดรันโดยตรงหรือไม่ ถ้าใช่ให้เรียกฟังก์ชัน main() ทำงาน
    main()
