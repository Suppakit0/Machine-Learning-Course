# ♻️ EcoSort AI – AI Waste Classification

ระบบจำแนกประเภทขยะด้วยปัญญาประดิษฐ์ (Artificial Intelligence) โดยใช้โมเดล Convolutional Neural Network (CNN) ในการวิเคราะห์รูปภาพและทำนายประเภทของขยะ พร้อมแสดงผลผ่านเว็บไซต์ที่พัฒนาด้วย Streamlit

## 📌 Project Overview

โปรเจกต์นี้มีวัตถุประสงค์เพื่อพัฒนาระบบ AI ที่สามารถจำแนกประเภทขยะจากรูปภาพได้โดยอัตโนมัติ เพื่อช่วยให้ผู้ใช้งานสามารถแยกประเภทขยะได้สะดวกและถูกต้องมากขึ้น

ระบบสามารถจำแนกขยะได้ทั้งหมด 6 ประเภท ได้แก่

1. **Cardboard** – กระดาษลัง
2. **Glass** – แก้ว
3. **Metal** – โลหะ
4. **Paper** – กระดาษ
5. **Plastic** – พลาสติก
6. **Trash** – ขยะทั่วไป

## 🎯 Project Objectives

- เพื่อศึกษาและประยุกต์ใช้ Machine Learning ในการจำแนกประเภทขยะ
- เพื่อพัฒนาโมเดล CNN สำหรับจำแนกภาพขยะ
- เพื่อสร้างเว็บไซต์สำหรับอัปโหลดรูปภาพและแสดงผลการทำนาย
- เพื่อส่งเสริมการคัดแยกขยะอย่างถูกต้อง

## 📚 Dataset Source

โปรเจกต์นี้ใช้ชุดข้อมูล **Garbage Classification** จาก Kaggle เพื่อนำมาใช้ในการฝึกและทดสอบโมเดล AI สำหรับจำแนกประเภทขยะจากรูปภาพ

ชุดข้อมูลประกอบด้วยขยะ 6 ประเภท ได้แก่

- Cardboard
- Glass
- Metal
- Paper
- Plastic
- Trash

ชุดข้อมูลที่นำมาใช้ในโปรเจกต์มีทั้งหมด **2,527 รูปภาพ**

- **Dataset Platform:** Kaggle
- **Dataset Name:** Garbage Classification
- **Dataset URL:** [Kaggle – Garbage Classification](https://www.kaggle.com/datasets/asdasdasasdas/garbage-classification)

ข้อมูลจาก Dataset ถูกนำมาแบ่งเป็น Training Set, Validation Set และ Testing Set เพื่อใช้ในการพัฒนาและประเมินประสิทธิภาพของโมเดล CNN

**Dataset Attribution:** เครดิตและสิทธิ์การใช้งานเป็นไปตามเงื่อนไขของเจ้าของ Dataset โปรดตรวจสอบ License จากหน้า Kaggle ต้นฉบับ

### Dataset Preparation

แบ่งข้อมูลออกเป็น 3 ส่วน

| Dataset | Percentage |
|---|---:|
| Training Set | 70% |
| Validation Set | 15% |
| Testing Set | 15% |

รายละเอียดการเตรียมข้อมูล:

- ปรับขนาดภาพเป็น 180 × 180 pixels
- Normalize ค่าพิกเซลให้อยู่ในช่วง 0–1
- ใช้ Data Augmentation เพื่อเพิ่มความหลากหลายของข้อมูล
- ใช้ Oversampling เพื่อปรับสมดุลข้อมูลในชุด Training

## 🛠️ Technologies Used

- Python
- TensorFlow / Keras
- Convolutional Neural Network (CNN)
- Streamlit
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Pillow (PIL)

## 📁 Project Structure

```text
AI-Waste-Classification/
│
├── dataset/
│   ├── cardboard/
│   ├── glass/
│   ├── metal/
│   ├── paper/
│   ├── plastic/
│   └── trash/
│
├── models/
│   ├── waste_cnn_model.keras
│   └── class_names.json
│
├── reports/
│
├── train.py
├── app.py
├── requirements.txt
└── README.md
```

## ⚙️ Installation

### 1. Clone Project

```bash
git clone <repository-url>
```

เข้าไปยังโฟลเดอร์โปรเจกต์

```bash
cd AI-Waste-Classification
```

> หมายเหตุ: เปลี่ยน `<repository-url>` เป็น URL ของ GitHub Repository จริง หากอัปโหลดโปรเจกต์ขึ้น GitHub แล้ว

### 2. Create Virtual Environment (Optional)

สร้าง Virtual Environment

```bash
python -m venv venv
```

เปิดใช้งาน Virtual Environment บน Windows

```bash
venv\Scripts\activate
```

### 3. Install Dependencies

ติดตั้ง Library ที่จำเป็นสำหรับโปรเจกต์

```bash
pip install -r requirements.txt
```

## 🧠 Model Training

ใช้คำสั่งต่อไปนี้เพื่อฝึกโมเดล CNN

```bash
python train.py --data_dir dataset --epochs 40
```

เมื่อฝึกโมเดลเสร็จ ระบบจะบันทึกโมเดลและไฟล์ที่เกี่ยวข้องไว้ในโฟลเดอร์ `models/` รวมถึงรายงานผลการประเมินไว้ในโฟลเดอร์ `reports/`

## 🚀 Run Web Application

เริ่มต้นใช้งานเว็บไซต์ด้วย Streamlit

```bash
python -m streamlit run app.py
```

จากนั้นเปิด Browser ตาม URL ที่ Streamlit แสดงขึ้นมา โดยปกติจะเป็น

```text
http://localhost:8501
```

## 📊 Model Evaluation

ผลการทดสอบโมเดล V2 บนชุดข้อมูล Testing

| Evaluation Metric | Result |
|---|---:|
| Test Accuracy | 70.00% |
| Test Loss | 0.8085 |
| Macro F1-Score | 68.75% |
| Weighted F1-Score | 71.52% |

### Classification Report

| Class | Precision | Recall | F1-Score |
|---|---:|---:|---:|
| Cardboard | 93.88% | 76.67% | 84.40% |
| Glass | 74.19% | 60.53% | 66.67% |
| Metal | 70.00% | 67.74% | 68.85% |
| Paper | 81.08% | 67.42% | 73.62% |
| Plastic | 72.60% | 72.60% | 72.60% |
| Trash | 30.65% | 95.00% | 46.34% |

## 🌐 Application Features

- Upload รูปภาพขยะผ่านหน้าเว็บไซต์
- แสดงรูปภาพที่ผู้ใช้งานอัปโหลด
- วิเคราะห์ประเภทขยะด้วยโมเดล CNN
- แสดงประเภทขยะที่โมเดลทำนาย
- แสดงค่าความมั่นใจ (Confidence)
- แสดงคะแนนความน่าจะเป็นของขยะแต่ละประเภท
- แสดงคำแนะนำในการคัดแยกขยะ

## ⚠️ Limitations

- ความแม่นยำของโมเดลยังไม่สมบูรณ์ 100%
- ภาพที่มีแสงน้อยหรือวัตถุไม่ชัดเจนอาจทำให้ทำนายผิดพลาด
- ขยะบางประเภทมีลักษณะใกล้เคียงกัน ทำให้โมเดลสับสน
- ประเภท Trash มีจำนวนข้อมูลน้อยกว่าประเภทอื่น ส่งผลต่อประสิทธิภาพการจำแนก

## 🔮 Future Improvements

- เพิ่มจำนวนรูปภาพใน Dataset
- ปรับปรุงโมเดล CNN เพื่อเพิ่มความแม่นยำ
- ทดลองใช้ Transfer Learning เช่น MobileNetV2
- ปรับปรุงการจำแนกขยะที่มีลักษณะคล้ายกัน
- พัฒนาระบบให้รองรับการใช้งานบนโทรศัพท์มือถือ

## 👨‍💻 Project Information

| Detail | Information |
|---|---|
| Project Name | EcoSort AI – AI Waste Classification |
| Project Type | Machine Learning Mini Project |
| Model | Convolutional Neural Network (CNN) |
| Framework | TensorFlow / Keras |
| Web Application | Streamlit |

---

*EcoSort AI – Making Waste Sorting Smarter with AI.*