# ใบงานที่ 5: Support Vector Machine (SVM)
## SVM on a Dataset of Your Choice — Zoo Animal Classification
 
โปรเจกต์นี้ประยุกต์ใช้ Support Vector Machine (SVM) เพื่อจำแนกประเภทของสัตว์ (Animal Classification) โดยเปรียบเทียบประสิทธิภาพของ SVM ทั้ง 3 kernel ได้แก่ **Linear**, **Polynomial**, และ **RBF**
 
---
 
## Objective
 
- นำ Support Vector Machine (SVM) มาประยุกต์ใช้ในการจำแนก (classification) ข้อมูล
- เปรียบเทียบประสิทธิภาพของ SVM แต่ละ kernel (Linear, Polynomial, RBF)
- ประเมินผลด้วยค่า Accuracy, Classification Report และ Confusion Matrix
---
 
## Dataset
 
**Zoo Animals Extended Dataset**
🔗 ที่มา: [Kaggle - agajorte/zoo-animals-extended-dataset](https://www.kaggle.com/datasets/agajorte/zoo-animals-extended-dataset)
 
ข้อมูลประกอบด้วยไฟล์ 2 ไฟล์ที่ถูกนำมารวมกัน (concat):
 
| ไฟล์ | รายละเอียด |
|---|---|
| `zoo2.csv` | ข้อมูลสัตว์ชุดแรก |
| `zoo3.csv` | ข้อมูลสัตว์ชุดเพิ่มเติม |
 
**Features (16 คุณลักษณะ):** ลักษณะทางกายภาพและพฤติกรรมของสัตว์ เช่น มีขน (hair), มีขนนก (feathers), วางไข่ (eggs), มีนม (milk), บินได้ (airborne), อาศัยในน้ำ (aquatic), เป็นสัตว์นักล่า (predator), มีฟัน (toothed), จำนวนขา (legs) ฯลฯ
 
**Target:** `class_type` — ประเภทของสัตว์ 7 class (1–7) เช่น สัตว์เลี้ยงลูกด้วยนม, นก, สัตว์เลื้อยคลาน, ปลา, สัตว์สะเทินน้ำสะเทินบก, แมลง, สัตว์ไม่มีกระดูกสันหลังอื่นๆ
 
**ขนาดข้อมูลหลัง preprocessing:** 113 samples (หลังลบข้อมูลซ้ำและค่าว่างออก)
 
---
 
## Project Structure
 
```
classification/
├── main.py            # จุดเริ่มการทำงานหลัก (pipeline ทั้งหมด)
├── data_load.py        # โหลดและรวมข้อมูลจาก zoo2.csv, zoo3.csv
├── preprocess.py        # แปลงข้อมูลเป็น float32 array
├── split_data.py        # แบ่งข้อมูล train/test
├── svm_model.py         # เทรนโมเดล SVM ทั้ง 3 kernel
├── evaluate.py         # ประเมินผลและสร้างกราฟ confusion matrix
├── test_svm.py          # ทดสอบโมเดลกับตัวอย่างข้อมูลแบบสุ่ม
└── outputs/            # ผลลัพธ์ที่ได้จากการรัน (สร้างอัตโนมัติ)
    ├── features.npy
    ├── labels.npy
    ├── classes.json
    ├── X_train.npy / X_test.npy
    ├── y_train.npy / y_test.npy
    ├── svm_linear.pkl / svm_poly.pkl / svm_rbf.pkl
    ├── confusion_matrix_linear.png
    ├── confusion_matrix_poly.png
    ├── confusion_matrix_rbf.png
    └── accuracy_comparison.png
 
Dataset/
├── zoo2.csv
└── zoo3.csv
```
 
---
 
## Pipeline การทำงาน
 
1. **Load Dataset** — โหลด `zoo2.csv` และ `zoo3.csv` มารวมกัน, ลบข้อมูลซ้ำและค่าว่างออก
2. **Preprocessing** — แปลงข้อมูลเป็น `numpy array` ชนิด `float32`
3. **Split Dataset** — แบ่งข้อมูลเป็น Training set 80% / Testing set 20% (แบบ stratified)
4. **Standardization** — ปรับสเกลข้อมูลด้วย `StandardScaler` ก่อนเข้าโมเดล (อยู่ใน Pipeline ของแต่ละ kernel)
5. **Train SVM** — เทรนโมเดล SVM 3 kernel: Linear, Polynomial (degree=3), RBF
6. **Prediction** — ทำนายผลลัพธ์บน Testing set
7. **Evaluation** — คำนวณ Accuracy, Classification Report, Confusion Matrix ของแต่ละ kernel
8. **Compare Accuracy** — สรุปและแสดงกราฟเปรียบเทียบ accuracy ของทั้ง 3 kernel
---
 
## Requirements
 
```
pandas
numpy
scikit-learn
matplotlib
joblib
```
 
ติดตั้งด้วยคำสั่ง:
```bash
pip install pandas numpy scikit-learn matplotlib joblib
```
 
---
 
## วิธีการรัน
 
**1. เทรนโมเดลและประเมินผล**
```bash
python main.py
```
 
**2. ทดสอบโมเดลกับตัวอย่างข้อมูลแบบสุ่ม**
```bash
python test_svm.py
```
 
---
 
## ผลลัพธ์ (Results)
 
### Accuracy เปรียบเทียบระหว่าง Kernel
 
| Kernel | Accuracy |
|---|---|
| **Linear** | 91.30% |
| **Poly**   | 91.30% |
| **RBF**    | 86.96% |
 
### สรุปผล
 
- **Linear SVM** และ **Polynomial SVM** ให้ผลลัพธ์ accuracy สูงสุดเท่ากันที่ 91.30% ส่วน **RBF SVM** ให้ผล accuracy ต่ำสุดที่ 86.96%
- จาก Classification Report และ Confusion Matrix พบว่า **class 5** เป็นจุดที่โมเดลทำนายผิดพลาดบ่อยที่สุดในทุก kernel โดยเฉพาะ Poly และ RBF ที่ทำนายผิดเป็น class 3 แทนทั้งหมด (recall = 0%)
- สาเหตุหลักมาจาก class 5 มีจำนวนตัวอย่างในชุดข้อมูลค่อนข้างน้อย (มีเพียง 2 samples ใน test set) ทำให้โมเดลเรียนรู้ลักษณะเฉพาะของ class นี้ได้ไม่ดีพอ
- แนวทางปรับปรุงที่เป็นไปได้: ใช้ `class_weight="balanced"` ใน SVC เพื่อถ่วงน้ำหนักให้ class ที่มีข้อมูลน้อย หรือหาข้อมูลเพิ่มเติมสำหรับ class ดังกล่าว
---
 
