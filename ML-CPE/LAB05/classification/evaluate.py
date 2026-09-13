# นำเข้าไลบรารี Matplotlib สำหรับการวาดกราฟ
import matplotlib

# ใช้โหมด "Agg" เพื่อบอกให้ Matplotlib สร้างและเซฟกราฟเป็นไฟล์รูปภาพ (Backend) โดยไม่ต้องพยายามเปิดหน้าต่างกราฟิก (GUI) ขึ้นมาแสดงผล
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

# นำเข้าเครื่องมือสำหรับวัดผลโมเดลจาก scikit-learn
from sklearn.metrics import (
    accuracy_score,        # คำนวณความแม่นยำรวม (ทายถูกกี่เปอร์เซ็นต์)
    classification_report, # สร้างรายงานเจาะลึก (Precision, Recall, F1-Score)
    confusion_matrix       # สร้างตารางเมทริกซ์ความสับสน (ดูว่าโมเดลสับสนคลาสไหนกับคลาสไหน)
)

# ฟังก์ชันหลักสำหรับการประเมินผลโมเดล
def evaluate_model(y_test, predictions, classes, kernel,
                   save_path=None):

    # 1. คำนวณความแม่นยำรวม (Accuracy) โดยเทียบผลเฉลย (y_test) กับผลที่โมเดลทาย (predictions)
    accuracy = accuracy_score(y_test, predictions)

    # พรินต์ส่วนหัวของรายงานใน Terminal
    print("\n----------------------------------------")
    print(f"{kernel.upper()} SVM") # แสดงชื่อ Kernel (เช่น LINEAR SVM, RBF SVM)
    print("----------------------------------------")
    print(f"Accuracy: {accuracy * 100:.2f}%") # แสดงความแม่นยำเป็นเปอร์เซ็นต์ ทศนิยม 2 ตำแหน่ง

    print("\nClassification Report:")
    
    # 2. พรินต์ Classification Report
    # รายงานนี้จะบอกค่า Precision (ความแม่นยำเมื่อทายคลาสนี้), Recall (ความสามารถในการหาคลาสนี้เจอ), และ F1-score
    print(
        classification_report(
            y_test,
            predictions,
            labels=classes,                         # กำหนดรายชื่อคลาสที่มี
            target_names=[str(c) for c in classes], # แปลงชื่อคลาสให้เป็น String
            zero_division=0                         # ป้องกัน Error หารด้วยศูนย์ในกรณีที่โมเดลทายคลาสบางคลาสไม่ได้เลย
        )
    )

    # 3. สร้าง Confusion Matrix (ตารางเปรียบเทียบผลทาย vs ความจริง)
    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=classes
    )

    print("Confusion Matrix:")
    print(matrix) # พรินต์ตารางตัวเลขลงใน Terminal

    # 4. หากมีการส่ง Path สำหรับเซฟรูปภาพมาด้วย ให้เรียกฟังก์ชันวาดกราฟ
    if save_path:
        plot_confusion_matrix(
            matrix,
            classes,
            kernel,
            save_path
        )

    # ส่งคืนค่าความแม่นยำรวม เพื่อให้ main.py นำไปสร้างกราฟแท่งเปรียบเทียบต่อไป
    return accuracy

# ฟังก์ชันสำหรับวาดรูป Confusion Matrix และเซฟเป็นไฟล์
def plot_confusion_matrix(
    matrix,
    classes,
    kernel,
    save_path
):

    # สร้างหน้ากระดาษ (Figure) ขนาด 7x6 นิ้ว
    fig, ax = plt.subplots(figsize=(7, 6))

    # วาดตารางเมทริกซ์ โดยใช้โทนสีฟ้า (Blues) ช่องที่ตัวเลขเยอะสีจะเข้ม ช่องที่ตัวเลขน้อยสีจะอ่อน
    ax.imshow(matrix, cmap="Blues")

    labels = [str(c) for c in classes]

    # กำหนดให้แกน X และ Y แสดงชื่อคลาส (ประเภทสัตว์)
    ax.set_xticks(
        np.arange(len(classes)),
        labels
    )
    ax.set_yticks(
        np.arange(len(classes)),
        labels
    )

    # ตั้งชื่อแกนเพื่อไม่ให้สับสน (แกน X = ผลที่ทายออกมา, แกน Y = ความจริง)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

    # ตั้งชื่อหัวกราฟตามประเภทของ Kernel
    ax.set_title(
        f"Confusion Matrix - {kernel.upper()} SVM"
    )

    # คำนวณค่ากึ่งกลางของตัวเลขในเมทริกซ์ เพื่อใช้กำหนดสีของตัวหนังสือ
    threshold = matrix.max() / 2

    # วนลูปเพื่อเขียนตัวเลขกำกับลงไปในแต่ละช่องของตาราง
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(
                j,                      # พิกัดแกน X
                i,                      # พิกัดแกน Y
                matrix[i, j],           # ตัวเลขผลลัพธ์ที่จะเขียนลงไป
                ha="center",            # จัดกึ่งกลางแนวนอน
                va="center",            # จัดกึ่งกลางแนวตั้ง
                # หากช่องนั้นสีเข้ม (ค่าเกิน threshold) ให้ใช้ตัวหนังสือสีขาว 
                # หากช่องสีอ่อน ให้ใช้ตัวหนังสือสีดำ เพื่อให้อ่านง่าย
                color=(
                    "white"
                    if matrix[i, j] > threshold
                    else "black"
                )
            )

    fig.tight_layout()               # จัดขอบกราฟให้สวยงาม ไม่ให้ตัวหนังสือตกขอบ
    fig.savefig(save_path, dpi=150)  # เซฟรูปลงใน Path ที่กำหนด ด้วยความละเอียด 150 DPI
    plt.close(fig)                   # ปิดการวาดเพื่อคืนหน่วยความจำ (Memory) คืนให้ระบบ
