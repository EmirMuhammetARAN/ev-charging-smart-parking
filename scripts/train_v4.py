import os
import shutil
import random
import yaml
from pathlib import Path
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def setup_dataset():
    dataset_dir = PROJECT_ROOT / "datasets" / "dataset v3"
    train_images_dir = dataset_dir / "train" / "images"
    train_labels_dir = dataset_dir / "train" / "labels"
    val_images_dir = dataset_dir / "valid" / "images"
    val_labels_dir = dataset_dir / "valid" / "labels"
    yaml_path = dataset_dir / "data.yaml"

    if not val_images_dir.exists():
        print("Validasyon klasoru bulunamadi, train'den %20 ayriliyor...")
        val_images_dir.mkdir(parents=True, exist_ok=True)
        val_labels_dir.mkdir(parents=True, exist_ok=True)

        images = list(train_images_dir.glob("*.jpg")) + list(train_images_dir.glob("*.png"))
        random.shuffle(images)
        val_count = int(len(images) * 0.2)
        val_images = images[:val_count]

        for img in val_images:
            shutil.move(str(img), str(val_images_dir / img.name))
            label = train_labels_dir / (img.stem + ".txt")
            if label.exists():
                shutil.move(str(label), str(val_labels_dir / label.name))
        print(f"{val_count} resim validasyon klasorune tasindi.")

    # Fix data.yaml paths
    with open(yaml_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    
    data['path'] = str(dataset_dir.resolve()).replace('\\', '/')
    data['train'] = "train/images"
    data['val'] = "valid/images"
    if 'test' in data:
        del data['test']
        
    with open(yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
    
    return yaml_path

if __name__ == "__main__":
    yaml_file = setup_dataset()
    print("[*] Egitim basliyor (YOLOv8m 1024x1024)...")
    
    model = YOLO("yolov8m.pt")
    
    results = model.train(
        data=str(yaml_file),
        epochs=100,
        imgsz=1024,
        batch=8,
        name="ev_charging_v4",
        project=str((PROJECT_ROOT / "runs").resolve()),
        exist_ok=True,
        device="0"
    )
    print("[+] Egitim tamamlandi!")
