import numpy as np


def preprocess_data(X):

    # ใช้คำสั่ง np.asarray เพื่อแปลงโครงสร้างข้อมูลต้นทาง ให้กลายเป็น NumPy Array ซึ่งเป็นฟอร์แมตมาตรฐานที่โมเดล Machine Learning ต้องการ
    X = np.asarray(
        X,
        dtype=np.float32
    )

    return X
