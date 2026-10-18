from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent

def main():
    # โหลดโมเดลที่ดีที่สุดจากการเทรน
    weights = ROOT / 'runs' / 'aquarium_model_30' / 'weights' / 'best.pt'

    if not weights.exists():
        raise FileNotFoundError(f'Model not found: {weights}')

    model = YOLO(str(weights))
    yaml_path = str(ROOT / 'data.yaml')

    # ประเมินผลบนชุดข้อมูล test เพื่อเก็บค่า mAP, Precision, Recall ไปทำรายงาน
    metrics = model.val(
        data=yaml_path,
        split='test',
        imgsz=416,
        batch=8,
        device=0,
        workers=0,
        plots=True,
        project=str(ROOT / 'runs'),
        name='evaluation',
        exist_ok=True,
    )

    # แสดงผลประเมิน
    print('\n===== Evaluation Results =====')
    print(f'Precision: {metrics.box.mp:.4f}')
    print(f'Recall: {metrics.box.mr:.4f}')
    print(f'mAP50: {metrics.box.map50:.4f}')
    print(f'mAP50-95: {metrics.box.map:.4f}')

if __name__ == '__main__':
    main()