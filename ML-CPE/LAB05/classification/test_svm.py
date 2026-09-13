import json
import os
import joblib # ไลบรารีสำหรับโหลดไฟล์โมเดลที่บันทึกไว้
import numpy as np

# กำหนดเส้นทาง (Path) ไปยังโฟลเดอร์ outputs ที่เก็บไฟล์จากการเทรนไว้
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
N_SAMPLES = 10 # กำหนดจำนวนสัตว์ที่จะสุ่มขึ้นมาทดสอบ

def test_svm(n_samples=N_SAMPLES):

    # 1. โหลดโมเดล (Load Models)
    # ใช้ joblib.load ดึงสมองของ AI ที่เทรนเสร็จแล้ว (ไฟล์ .pkl) กลับมาใช้งาน
    models = {
        "linear": joblib.load(f"{OUTPUT_DIR}/svm_linear.pkl"),
        "poly": joblib.load(f"{OUTPUT_DIR}/svm_poly.pkl"),
        "rbf": joblib.load(f"{OUTPUT_DIR}/svm_rbf.pkl")
    }

    # 2. โหลดชุดข้อมูลทดสอบ (Load Test Data)
    # ดึงไฟล์ Features (X_test) และคำตอบที่ถูกต้อง (y_test) ที่ถูกแบ่งไว้ตอนแรก
    X_test = np.load(f"{OUTPUT_DIR}/X_test.npy")
    y_test = np.load(f"{OUTPUT_DIR}/y_test.npy")

    # โหลดรายชื่อคลาสเพื่อความสมบูรณ์ของระบบ
    with open(f"{OUTPUT_DIR}/classes.json") as f:
        classes = json.load(f)

    # ป้องกันกรณีที่ขอสุ่มทดสอบมากกว่าจำนวนข้อมูลที่มีจริง
    n_samples = min(n_samples, len(X_test))

    # 3. สุ่มข้อมูล (Random Sampling)
    # สุ่มเลือก "หมายเลขลำดับ (Index)" ของสัตว์จำนวน 10 ตัว โดยไม่ซ้ำกัน (replace=False)
    index = np.random.choice(len(X_test), n_samples, replace=False)

    # ดึงข้อมูลและเฉลยของสัตว์ 10 ตัวนั้นออกมา
    X_sample = X_test[index]
    y_sample = y_test[index]

    print("\n========================================")
    print("SVM PREDICTION TEST")
    print("========================================")

    # 4. ทดสอบและเปรียบเทียบผล (Inference & Comparison)
    for i in range(n_samples):
        print(f"\nSample {i + 1}")
        print(f"True Class: {y_sample[i]}") # พรินต์เฉลยที่ถูกต้องออกมาดูก่อน

        # นำข้อมูลสัตว์ตัวนี้ไปให้โมเดลทั้ง 3 Kernel ทำนาย
        for kernel, model in models.items():

            # *** จุดสำคัญทางเทคนิค: การใช้ .reshape(1, -1) ***
            # โมเดลของ Scikit-learn ถูกออกแบบมาให้รับข้อมูลเป็นกลุ่ม (2D Array)
            # แต่เราส่งข้อมูลเข้าไปทำนายทีละ 1 ตัว (1D Array) จึงต้องใช้ reshape 
            # เพื่อแปลงรูปร่างข้อมูลให้เป็น [1 แถว, คอลัมน์เท่าเดิม] โมเดลถึงจะยอมอ่าน
            prediction = model.predict(X_sample[i].reshape(1, -1))[0]

            # ตรวจคำตอบว่าทายถูก (OK) หรือทายผิด (WRONG)
            result = (
                "OK"
                if prediction == y_sample[i]
                else "WRONG"
            )

            # พรินต์ผลลัพธ์การทำนายจัดรูปแบบให้สวยงามอ่านง่าย
            print(
                f"{kernel.upper():<10} "
                f"Prediction: {prediction:<3} "
                f"{result}"
            )

if __name__ == "__main__":
    test_svm()
