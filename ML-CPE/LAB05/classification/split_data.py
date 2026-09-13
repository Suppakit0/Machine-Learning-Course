import numpy as np

from sklearn.model_selection import train_test_split

# ฟังก์ชันสำหรับแบ่งข้อมูลเป็นชุด Train (สอน) และ Test (ทดสอบ)
def split_dataset(X, y, test_size=0.2):

    # แปลงข้อมูลเป้าหมาย (Labels/y) ให้เป็น NumPy Array เพื่อความชัวร์และป้องกัน Error
    y = np.asarray(y)

    # ฟังก์ชัน train_test_split จะทำการสับเปลี่ยน (Shuffle) ข้อมูล และแบ่งออกเป็น 4 ส่วน ได้แก่ X_train, X_test, y_train, y_test
    X_train, X_test, y_train, y_test = train_test_split(
        X,                  # ข้อมูลฟีเจอร์ (Features)
        y,                  # ข้อมูลเป้าหมาย (Labels) ว่าสัตว์ตัวนี้คือประเภทไหน
        test_size=test_size,# สัดส่วนชุดทดสอบ (ค่า Default คือ 0.2 หรือ 20%)
        random_state=42,    # ล็อกค่า Seed ของการสุ่ม เพื่อให้รันกี่ครั้ง ข้อมูลก็ถูกแบ่งแบบเดิมเสมอ
        stratify=y          # *** จุดสำคัญที่สุด: บังคับให้การสุ่มรักษาสัดส่วนของคลาส (Class Distribution)
    )

    # ส่งคืนชุดข้อมูลทั้ง 4 ตัวกลับไปให้ main.py
    return X_train, X_test, y_train, y_test
