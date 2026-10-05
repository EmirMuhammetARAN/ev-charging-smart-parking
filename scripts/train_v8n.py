import os
from pathlib import Path
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def main():
    yaml_file = PROJECT_ROOT / "datasets" / "dataset v3" / "data.yaml"
    print("[*] YOLOv8n egitimi baslatiliyor (v8m ile birebir ayni ortam ve ayarlar)...")
    print(f"[*] Veri seti: {yaml_file}")
    print("[*] Ayarlar: 100 epoch, imgsz=1024, batch=8, device='0', workers=8")
    
    model_weight = PROJECT_ROOT / "yolov8n.pt"
    model = YOLO(str(model_weight))
    
    results = model.train(
        data=str(yaml_file),
        epochs=100,
        imgsz=1024,
        batch=8,
        name="ev_charging_v8n",
        project=str((PROJECT_ROOT / "runs").resolve()),
        exist_ok=True,
        device="0",
        workers=8,
        verbose=True
    )
    print("[+] YOLOv8n egitimi basariyla tamamlandi!")

if __name__ == "__main__":
    main()
