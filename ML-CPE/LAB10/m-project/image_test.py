from pathlib import Path
import random
import cv2
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent

def main():
    weights = ROOT / 'runs' / 'aquarium_model_30' / 'weights' / 'best.pt'
    
    # บรรทัดที่ 12: ชี้ไปที่โฟลเดอร์ภาพ Test 
    image_dir = ROOT.parent / 'Dataset' / 'Aquarium' / 'test' / 'images'
    
    if not image_dir.exists():
        # บรรทัดที่ 16: เผื่อกรณีไฟล์ภาพอยู่แค่ในโฟลเดอร์ test โดยไม่มีโฟลเดอร์ย่อย images
        image_dir = ROOT.parent / 'Dataset' / 'Aquarium' / 'test'

    if not weights.exists():
        raise FileNotFoundError(f'Model not found: {weights}')
    if not image_dir.exists():
        raise FileNotFoundError(f'Dataset not found: {image_dir}')

    model = YOLO(str(weights))

    extensions = {'.jpg', '.jpeg', '.png'}
    images = [p for p in image_dir.iterdir() if p.suffix.lower() in extensions]

    if not images:
        raise RuntimeError('No test images found')

    # สุ่มเลือกภาพมาทดสอบ 5 ภาพ
    selected_images = random.sample(images, min(5, len(images)))

    for image in selected_images:
        results = model.predict(
            source=str(image),
            conf=0.25,
            imgsz=416,
            device=0,
            save=True,
            project=str(ROOT / 'runs'),
            name='image_test',
            exist_ok=True
        )

        for result in results:
            print(f'\nImage: {image.name}')
            for box in result.boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                print(f'{model.names[class_id]}: {confidence:.2f}')

            annotated = result.plot()
            cv2.imshow('YOLO Aquarium Detection', annotated)
            
            # กดปุ่มใดๆ เพื่อดูภาพถัดไป หรือกด 'q' เพื่อออก
            if cv2.waitKey(0) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                return

    cv2.destroyAllWindows()
    print('\nImage testing completed!')

if __name__ == '__main__':
    main()