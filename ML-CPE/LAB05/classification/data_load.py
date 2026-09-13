import os
import pandas as pd # นำเข้าไลบรารี Pandas สำหรับจัดการข้อมูลแบบตาราง (DataFrame)
import numpy as np  # นำเข้า NumPy สำหรับจัดการข้อมูลตัวเลขและอาร์เรย์

# ฟังก์ชันสำหรับโหลดและทำความสะอาดชุดข้อมูล
def load_data(data_path):

    # 1. กำหนดรายชื่อไฟล์เป้าหมาย
    # ใช้ os.path.join เพื่อเชื่อม Path ให้รองรับได้ทุกระบบปฏิบัติการ (Windows/Mac/Linux)
    files = [
        os.path.join(data_path, "zoo2.csv"),
        os.path.join(data_path, "zoo3.csv")
    ]

    dataframes = [] # สร้าง List ว่างไว้รอเก็บข้อมูลจากไฟล์ที่อ่านได้

    # 2. วนลูปอ่านไฟล์ทีละไฟล์
    for file in files:

        # ดักจับ Error: เช็กว่ามีไฟล์นี้อยู่จริงไหม ถ้าไม่มีให้แจ้งเตือน FileNotFoundError
        if not os.path.exists(file):
            raise FileNotFoundError(
                f"File not found: {file}"
            )

        # ใช้ Pandas อ่านข้อมูลจากไฟล์ CSV เข้ามาเป็น DataFrame
        df = pd.read_csv(file)

        # นำ DataFrame ที่อ่านได้ไปเก็บสะสมไว้ใน List
        dataframes.append(df)

        # พรินต์แจ้งเตือนว่าโหลดไฟล์สำเร็จและมีข้อมูลกี่บรรทัด (กี่ Samples)
        print(
            f"Loaded: {file} "
            f"({len(df)} samples)"
        )

    # 3. รวมชุดข้อมูล (Data Integration)
    # นำข้อมูลจากทั้ง 2 ไฟล์มาต่อกันเป็นตารางเดียว (ignore_index=True เพื่อจัดเรียงเลขแถวใหม่ตั้งแต่ 0)
    data = pd.concat(
        dataframes,
        ignore_index=True
    )

    # 4. ทำความสะอาดข้อมูล (Data Cleaning)
    # ลบข้อมูลบรรทัดที่ซ้ำกันออก (เพื่อป้องกันไม่ให้โมเดลจำข้อสอบซ้ำ)
    data = data.drop_duplicates()

    # ลบข้อมูลบรรทัดที่มีค่าว่าง (Missing Values หรือ NaN) ออก เพื่อป้องกัน Error ตอนเทรนโมเดล
    data = data.dropna()

    print("\nDataset combined successfully.")
    print(f"Total samples: {len(data)}")

    # 5. แยกฟีเจอร์และเป้าหมาย (Feature & Target Separation)
    # X คือ "คุณลักษณะ" (Features) สร้างโดยการตัดคอลัมน์ชื่อสัตว์และประเภทสัตว์ทิ้งไป
    # (เพราะชื่อสัตว์ไม่มีผลต่อลักษณะทางกายภาพ และคลาสคือสิ่งที่เราต้องการทาย)
    X = data.drop(
        columns=["animal_name", "class_type"]
    )

    # y คือ "คำตอบ" (Target) ดึงมาเฉพาะคอลัมน์ประเภทสัตว์ (class_type)
    y = data["class_type"]

    # 6. แปลงประเภทข้อมูลให้พร้อมใช้งาน (Data Type Conversion)
    # แปลง X ให้เป็น NumPy Array แบบทศนิยม 32 บิต เพื่อความรวดเร็วและประหยัดหน่วยความจำ
    X = X.to_numpy(
        dtype=np.float32
    )

    # แปลง y ให้เป็น NumPy Array เช่นกัน
    y = y.to_numpy()

    # 7. ค้นหาคลาสทั้งหมดที่มี
    # ใช้ np.unique เพื่อดูว่าคลาสทั้งหมดมีกี่ประเภท และใช้ sorted เพื่อเรียงลำดับให้สวยงาม
    classes = sorted(
        np.unique(y).tolist()
    )

    # ส่งคืนค่า X (ฟีเจอร์), y (คำตอบ), และ classes (รายชื่อประเภทสัตว์) กลับไปให้ main.py
    return X, y, classes
