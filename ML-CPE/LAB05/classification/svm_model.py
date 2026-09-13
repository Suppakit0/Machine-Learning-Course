from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# ฟังก์ชันสำหรับกำหนดค่าและสอนโมเดล (Training)
def train_svm(X_train, y_train):

    models = {} # สร้าง Dictionary เปล่าไว้เก็บโมเดลที่เทรนเสร็จแล้ว

    # กำหนดค่า Hyperparameters ให้กับ SVM ใน 3 รูปแบบ (Kernels) เพื่อนำมาแข่งกัน
    kernels = {
        # 1. Linear Kernel: สร้างเส้นแบ่งแบบเส้นตรง (เหมาะกับข้อมูลที่แยกได้ง่ายๆ)
        "linear": SVC(
            kernel="linear",
            C=1.0 # ค่า C (Regularization) คือค่ายอมรับความผิดพลาด ยิ่งน้อยยิ่งยอมให้มีจุดที่ทายผิดได้บ้างเพื่อไม่ให้โมเดลจำข้อสอบเกินไป (Overfitting)
        ),

        # 2. Polynomial Kernel: สร้างเส้นแบ่งแบบโค้งเว้า 
        "poly": SVC(
            kernel="poly",
            C=1.0,
            degree=3 # กำหนดความโค้งเป็นสมการพหุนามดีกรี 3
        ),

        # 3. RBF (Radial Basis Function) Kernel: สร้างเส้นแบ่งแบบวงล้อมรอบกลุ่มข้อมูล (เหมาะกับข้อมูลที่ซับซ้อนมาก)
        "rbf": SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale" # ให้ระบบคำนวณรัศมีอิทธิพลของจุดข้อมูลโดยอัตโนมัติ
        )
    }

    # วนลูปเพื่อนำแต่ละ Kernel มาสร้าง Pipeline และทำการสอน (Train)
    for name, svm in kernels.items():

        # *** จุดสำคัญ: การใช้ Pipeline ***
        # ระบบจะบังคับให้ข้อมูลวิ่งผ่าน StandardScaler เพื่อปรับสเกลก่อน แล้วค่อยส่งเข้าโมเดล SVM เสมอ
        pipeline = Pipeline([
            (
                "scaler",
                StandardScaler() 
            ),
            (
                "svm",
                svm
            )
        ])

        print(
            f"Training {name.upper()} SVM..."
        )

        # เริ่มต้นกระบวนการเรียนรู้ โดยป้อนข้อมูลฟีเจอร์ (X_train) และเฉลย (y_train) ให้โมเดล
        pipeline.fit(
            X_train,
            y_train
        )

        # เก็บโมเดลที่ฉลาดแล้ว (เทรนเสร็จแล้ว) ลงใน Dictionary
        models[name] = pipeline

    return models

# ฟังก์ชันสำหรับนำโมเดลไปทำนายผล (Prediction)
def predict_svm(models, X_test):

    predictions = {} # สร้าง Dictionary ไว้เก็บคำตอบที่โมเดลทายออกมา

    # วนลูปเรียกใช้งานโมเดลทั้ง 3 ตัว
    for name, model in models.items():
        
        # ให้โมเดลทำนายข้อมูลชุดทดสอบ (X_test)
        # ข้อดีของ Pipeline คือ X_test จะถูกปรับสเกล (Scale) อัตโนมัติก่อนเข้า SVM โดยใช้ค่ามาตรฐานเดิมจากตอน Train
        predictions[name] = model.predict(
            X_test
        )

    return predictions
