import numpy as np
from sklearn.preprocessing import StandardScaler

def preprocess_data(df):
    """แยก Features และ Target พร้อมทำ Standard Scaling"""
    # X ตัดคอลัมน์ชื่อสัตว์และคลาสออก
    X = df.drop(['animal_name', 'class_type'], axis=1).values
    
    # y ปรับให้คลาสเริ่มที่ 0 (จากเดิม 1-7 เป็น 0-6 เพื่อให้ Keras คำนวณได้ถูกต้อง)
    y = df['class_type'].values - 1 
    
    num_classes = len(np.unique(y))
    classes = [f"Type_{i+1}" for i in range(num_classes)]
    
    # สเกลข้อมูล (Standardization)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, classes, num_classes