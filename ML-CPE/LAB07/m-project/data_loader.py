import pandas as pd
import os

def load_data(file1_path, file2_path):
    """โหลดและรวมไฟล์ CSV 2 ไฟล์เข้าด้วยกัน"""
    if not os.path.exists(file1_path):
        raise FileNotFoundError(f"หาไฟล์ไม่พบ กรุณาเช็ก Path: {file1_path}")
    if not os.path.exists(file2_path):
        raise FileNotFoundError(f"หาไฟล์ไม่พบ กรุณาเช็ก Path: {file2_path}")
    
    print(f"Loading: {file1_path} and {file2_path}")
    df1 = pd.read_csv(file1_path)
    df2 = pd.read_csv(file2_path)
    
    # รวมข้อมูลทั้งสองไฟล์
    df = pd.concat([df1, df2], ignore_index=True)
    print(f"Total loaded rows: {len(df)}")
    return df