from pathlib import Path
import cv2
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent

def main():
    weights = ROOT / 'runs' / 'aquarium_model_30' / 'weights' / 'best.pt'

    if not weights.exists():
        raise FileNotFoundError(f'Model not found: {weights}')

    model = YOLO(str(weights))
    
    # เปิดกล้อง Webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise RuntimeError('Cannot open webcam')

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            # ตรวจจับแบบเรียลไทม์
            results = model.predict(
                source=frame,
                conf=0.4,
                imgsz=416,
                device=0,
                verbose=False
            )

            annotated_frame = results[0].plot()
            cv2.imshow('Real-time Aquarium Detection', annotated_frame)

            # กด Q เพื่อออก
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()