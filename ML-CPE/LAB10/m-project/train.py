from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent

def main():
    # โหลด Pretrained YOLO model
    model = YOLO("yolov8n.pt")

    # ระบุ Path ของไฟล์ data.yaml (แก้ปัญหา FileNotFoundError)
    yaml_path = str(ROOT / 'data.yaml')

    # เริ่มเทรนโมเดล
    model.train(
        data=yaml_path,
        epochs=30,        # ปรับจำนวน Epoch ตามใบงานกำหนด (เช่น 10, 25, 50)
        imgsz=416,
        batch=8,          
        device=0,         # ใช้ GPU RTX 4060 ของเครื่อง Nitro 
        workers=0,
        project=str(ROOT / "runs"),
        name="aquarium_model_30",
        exist_ok=True,
        save=True,
        plots=True
    )

    print("\nTraining completed!")
    print(f"Best Model saved at: {ROOT / 'runs' / 'aquarium_model' / 'weights' / 'best.pt'}")

if __name__ == "__main__":
    main()