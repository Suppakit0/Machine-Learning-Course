# ใบงานที่ 6: Neural Network และการประยุกต์ใช้งาน
## Neural Network (NN) on a Dataset of Your Choice — Zoo Animal Classification

โปรเจกต์นี้ประยุกต์ใช้ Neural Network (Multi-Layer Perceptron - MLP) เพื่อจำแนกประเภทของสัตว์ (Animal Classification) โดยเปรียบเทียบประสิทธิภาพระหว่างจำนวน Epochs และสถาปัตยกรรม (Configurations) ที่แตกต่างกัน

---

## Objective

- นำ Neural Network (NN) มาประยุกต์ใช้ในการจำแนก (classification) ข้อมูล
- เปรียบเทียบประสิทธิภาพของสถาปัตยกรรม NN ที่มีขนาดแตกต่างกัน (Small, Medium, Large)
- เปรียบเทียบผลลัพธ์การฝึกสอนด้วยจำนวน Epochs ที่หลากหลาย (10, 20, 30 Epochs)
- ประเมินผลด้วยค่า Accuracy, Classification Report, Confusion Matrix และกราฟการเรียนรู้

---

## Dataset

**Zoo Animals Extended Dataset**
🔗 ที่มา: [Kaggle - agajorte/zoo-animals-extended-dataset](https://www.kaggle.com/datasets/agajorte/zoo-animals-extended-dataset)

ข้อมูลประกอบด้วยไฟล์ 2 ไฟล์ที่ถูกนำมารวมกัน (concat):

| ไฟล์ | รายละเอียด |
|---|---|
| `zoo2.csv` | ข้อมูลสัตว์ชุดแรก |
| `zoo3.csv` | ข้อมูลสัตว์ชุดเพิ่มเติม |

**Features (16 คุณลักษณะ):** ลักษณะทางกายภาพและพฤติกรรมของสัตว์ เช่น มีขน (`hair`), มีขนนก (`feathers`), วางไข่ (`eggs`), มีนม (`milk`), บินได้ (`airborne`), อาศัยในน้ำ (`aquatic`), เป็นสัตว์นักล่า (`predator`), มีฟัน (`toothed`), จำนวนขา (`legs`) ฯลฯ

**Target:** `class_type` — ประเภทของสัตว์ 7 คลาส (1–7) ได้แก่ สัตว์เลี้ยงลูกด้วยนม (Mammal), นก (Bird), สัตว์เลื้อยคลาน (Reptile), ปลา (Fish), สัตว์สะเทินน้ำสะเทินบก (Amphibian), แมลง (Insect), และสัตว์ไม่มีกระดูกสันหลังอื่นๆ (Invertebrate) โดยถูกแปลงค่าเป็น 0–6

**ขนาดข้อมูลหลัง preprocessing:** 113 samples (แบ่งเป็น Training set 78 samples, Validation set 12 samples, และ Testing set 23 samples)

---

## Project Structure

```text
Dataset/
├── zoo2.csv
└── zoo3.csv

classification/
├── main.py                     # จุดเริ่มการทำงานหลักของ Neural Network
├── data_loader.py              # โหลดและรวมข้อมูลจาก zoo2.csv, zoo3.csv
├── preprocessing.py            # แปลงข้อมูลและทำ StandardScaler
├── split_data.py               # แบ่งข้อมูล Train/Val/Test แบบ Stratified
├── nn_model.py                 # สร้างและเทรนโมเดล MLP ด้วย Keras
├── evaluate.py                 # ประเมินผล, สร้าง Confusion Matrix และกราฟ
├── test_nn.py                  # ทดสอบโมเดล NN กับตัวอย่างข้อมูลแบบสุ่ม
└── outputs/                    # ผลลัพธ์ที่ได้จากการรัน (สร้างอัตโนมัติ)
    ├── features.npy
    ├── labels.npy
    ├── classes.json
    ├── X_train.npy / X_val.npy / X_test.npy
    ├── y_train.npy / y_val.npy / y_test.npy
    ├── nn_model.keras
    ├── history.json
    ├── results.json
    ├── confusion_matrix.png
    └── training_history.png
