
Claude finished the response

Analyzed SVM classification lab assignment structure and objectives
Analyzed SVM classification lab assignment structure and objectives
ดูจากใบงานนี้เข้าใจว่านักศึกษาจะทำ Lab SVM ใหม่ (LEB) โดยเปลี่ยนจาก dataset zoo (ตัวเลข/tabular) มาเป็น รูปภาพ Cat vs Dog ซึ่งต่างจากงานก่อนหน้าพอสมควร เพราะต้องมีขั้นตอน "แปลงรูปภาพเป็น feature vector" ก่อนเข้า SVM (SVM รับ input เป็นตัวเลขล้วนๆ ไม่ใช่รูปภาพตรงๆ)

ก่อนเริ่มเขียนโค้ดให้ครบตามโครงสร้างนี้ ขอเช็ค 2 เรื่องก่อนครับ เพื่อไม่ให้เขียนโค้ดผิดทางแล้วต้องแก้ทีหลัง


test_svm.py
76 lines

py


svm_model.py
38 lines

py


split_data.py
16 lines

py


preprocess.py
42 lines

py


main.py
83 lines

py


evaluate.py
72 lines

py


data_load.py
59 lines

py

ตัวอย่าง


ตรวจสอบตัวอย่างและคาดเดาความตั้งใจของผู้ใช้
ตรวจสอบตัวอย่างและคาดเดาความตั้งใจของผู้ใช้
โค้ดตัวอย่างชุดนี้ครบตามที่ใบงานกำหนดเลยครับ (แปลงรูปเป็น grayscale + resize 100x100 + normalize 0-1 แล้วค่อยเข้า SVM แบบ RBF ที่มี PCA ช่วยลดมิติก่อน) โครงสร้างตรงกับ diagram ที่ให้มาเกือบทั้งหมด

สิ่งที่น่าสังเกตจากโค้ดตัวอย่างนี้:

1. วิธี preprocess ที่เขาเลือกใช้

Flatten grayscale pixel (ไม่ใช้ HOG/color histogram) แต่ใช้ PCA ลดมิติเหลือ 150 features ก่อนเข้า SVM — เป็นทางออกที่ฉลาด เพราะรูป 100×100 = 10,000 pixels ถ้าไม่ลดมิติ RBF SVM จะช้ามาก PCA ช่วยทั้งเรื่องความเร็วและช่วยตัด noise ออกด้วย
2. จุดที่ต้องระวัง (ปัญหาเดิมที่เจอมาก่อน)

python
DATA_PATH = "Cats-and-Dogs/PetImages"
เป็น relative path เหมือนเคสก่อนหน้าเป๊ะ — ถ้ารันจากคนละ working directory จะหาไฟล์ไม่เจอ และ path นี้คาดหวังโฟลเดอร์ชื่อ Cats-and-Dogs/PetImages/Cat, Cats-and-Dogs/PetImages/Dog ซึ่งต่างจาก tree diagram ที่ให้มาตอนแรก (ที่วาง PetImages/ ไว้ที่ root ของโปรเจกต์เลย ไม่มีโฟลเดอร์ Cats-and-Dogs ครอบ) ต้องเช็คให้ตรงกับที่เครื่องคุณมีจริง

3. test_svm.py มี IMG_SIZE = 100 แยกต่างหาก — ต้องตรงกับ IMG_SIZE ใน main.py เสมอ ไม่งั้นตอน reshape รูปกลับมาแสดงผลจะพัง (ตอนนี้ตรงกันอยู่ ok)

ตอนนี้คุณมีไฟล์ครบแล้ว อยากให้ผมช่วยเรื่องไหนต่อครับ


data_load.py
66 lines

py


evaluate.py
107 lines

py


main.py
227 lines

py


preprocess.py
11 lines

py


split_data.py
17 lines

py


svm_model.py
69 lines

py


test_svm.py
93 lines

py

อันนี้สมบูรณ์รึยัง


ตรวจสอบและยืนยันโครงการเสร็จสมบูรณ์ตามข้อกำหนด
ตรวจสอบและยืนยันโครงการเสร็จสมบูรณ์ตามข้อกำหนด
ใช่ครับ ชุดนี้ สมบูรณ์แล้ว และคุณก็รันผ่านจริงมาแล้วก่อนหน้านี้ (ผลลัพธ์ accuracy Linear 91.30%, Poly 91.30%, RBF 86.96% ที่โชว์ไปก่อนหน้า) มาเช็ค checklist เทียบกับใบงานให้ชัดๆ อีกรอบ:

✅ ตรงตามข้อกำหนดใบงานครบ
ข้อกำหนดใบงาน	อยู่ในไฟล์ไหน	สถานะ
Select & load dataset	data_load.py โหลด zoo2.csv + zoo3.csv	✅
Explore & preprocess (ลบ duplicate, ลบ missing)	data_load.py (drop_duplicates, dropna)	✅
Standardize features ก่อนเทรน	svm_model.py — StandardScaler() อยู่ใน Pipeline ทุก kernel	✅
เทรน 3 kernels: Linear, Poly, RBF	svm_model.py	✅
Evaluate ด้วย accuracy	evaluate.py (accuracy_score + classification report + confusion matrix)	✅
Output: accuracy score แต่ละ kernel	main.py Step 7 print + accuracy_comparison.png	✅
Output: predictions บน dataset	test_svm.py สุ่มทำนายและโชว์ผล	✅
Path bug (relative path ผิด)	main.py, test_svm.py ใช้ BASE_DIR แล้ว	✅ แก้แล้ว
📋 ไฟล์ output ที่ควรมีครบใน outputs/ (เช็คจากที่รันจริงไปแล้ว)
features.npy, labels.npy, classes.json, X_train/X_test/y_train/y_test.npy, svm_linear.pkl, svm_poly.pkl, svm_rbf.pkl, confusion_matrix_linear/poly/rbf.png, accuracy_comparison.png — ครบตามที่รันมาก่อนหน้า ✅

สิ่งที่อาจเสริมได้ (ไม่บังคับ แต่ทำให้ครบเนียนขึ้นสำหรับส่งงาน)
requirements.txt — ยังไม่มีในโปรเจกต์นี้ ถ้าต้องส่งพร้อมโค้ด ควรมีไว้ระบุ dependency (pandas, numpy, scikit-learn, matplotlib, joblib)
รายงานผล/สรุปใน README สั้นๆ ว่า kernel ไหนดีที่สุดและทำไม (มีข้อมูลจาก class 5 ที่ทายพลาดบ่อยที่คุยกันไปแล้ว)
อยากให้ผมช่วยสร้าง requirements.txt ให้เลยไหมครับ จะได้ครบเซ็ตพร้อมส่ง

ทำ readme ให้หน่อย

Claude’s response was interrupted.





Claude is AI and can make mistakes. Please double-check responses.


Readme · MD
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
 
