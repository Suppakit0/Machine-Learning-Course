# ML-07-Convolutional Neural Network (CNN) / Deep Neural Network (DNN)

โปรเจกต์นี้พัฒนาขึ้นสำหรับใบงานที่ 7 (Lab 07) รายวิชา Machine Learning ของมหาวิทยาลัยเทคโนโลยีราชมงคลธัญบุรี โดยสร้างไปป์ไลน์ Deep Learning ด้วย Python สำหรับการจำแนกประเภทข้อมูลตาราง (Tabular Data Classification) จากชุดข้อมูลสัตว์ (Zoo Dataset) ซึ่งครอบคลุมกระบวนการตั้งแต่การโหลดข้อมูล, การประมวลผลล่วงหน้า, การแบ่งชุดข้อมูล, การสร้างและเทรนโมเดล, ไปจนถึงการประเมินผลและการทดสอบโมเดลอย่างเป็นระบบในรูปแบบโครงสร้างโค้ดแบบโมดูล

# ข้อมูล (Data)

ชุดข้อมูลสัตว์ (Zoo Dataset) ประกอบด้วยไฟล์ CSV สองไฟล์ ได้แก่:
- `zoo2.csv`
- `zoo3.csv`
ข้อมูลจะถูกโหลดและรวมเข้าด้วยกันอัตโนมัติ โดยประกอบด้วยคุณลักษณะทางชีวภาพ (Features) ต่างๆ ของสัตว์ และคลาสประเภทของสัตว์ (`class_type`) สำหรับการจำแนกแบบหลายคลาส (Multiclass Classification)

# โครงสร้างโปรเจกต์ (Project Structure)

```text
LAB07/
│
├── Dataset/                  
│   ├── zoo2.csv
│   └── zoo3.csv
│
├── m-project/
│   ├── __pycache__/
│   ├── outputs/                    
│   │   ├── classes.json
│   │   ├── X_test.npy
│   │   ├── y_test.npy
│   │   ├── dnn_model.keras
│   │   ├── history.json
│   │   ├── confusion_matrix.png
│   │   ├── test_confusion_matrix.png
│   │   └── training_history.png
│   ├── main.py                     # ไฟล์หลักสำหรับรันกระบวนการเทรน (Training Pipeline)
│   ├── data_loader.py              # โหลดและรวมไฟล์ชุดข้อมูล CSV เข้าด้วยกัน
│   ├── preprocessing.py            # แยกฟีเจอร์และปรับสเกลข้อมูลด้วย Standard Scaling
│   ├── split_data.py               # แบ่งข้อมูลเป็นชุดฝึกสอน (Train), ตรวจสอบ (Validation) และทดสอบ (Test)
│   ├── cnn_model.py                # สร้าง, เทรน, บันทึก และทำนายผลด้วยโมเดล DNN
│   ├── evaluate.py                 # คำนวณความแม่นยำ, รายงานการจำแนกประเภท, Confusion Matrix และกราฟประวัติการเทรน
│   ├── test_cnn.py                 # โหลดโมเดลที่บันทึกไว้มาทดสอบกับชุดข้อมูลทดสอบจริง
│   └── README.md                   # เอกสารอธิบายรายละเอียดโปรเจกต์
└── README.md
```

# สรุป (Summary)

โปรเจกต์นี้ประยุกต์ใช้โครงข่ายประสาทเทียมเชิงลึก (Deep Neural Network - DNN) ในการจำแนกประเภทสัตว์จากชุดข้อมูล `zoo2.csv` และ `zoo3.csv` ข้อมูลทั้งหมดจะถูกจัดการผ่านโมดูลแยกส่วนอย่างเป็นระเบียบ ทำการสเกลข้อมูลด้วย `StandardScaler` และแบ่งสัดส่วนข้อมูลอย่างแม่นยำด้วยเทคนิค Stratified Split โมเดลจะถูกเทรนพร้อมกลไก `EarlyStopping` เพื่อป้องกันการเกิด Overfitting และประเมินผลผ่านค่า Accuracy, Classification Report, Confusion Matrix รวมถึงกราฟแสดง Loss และ Accuracy ในแต่ละ Epoch เพื่อตรวจสอบประสิทธิภาพและความถูกต้องของโมเดลอย่างรอบด้าน
