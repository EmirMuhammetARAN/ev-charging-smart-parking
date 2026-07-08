import os
import shutil
import random
import yaml
from pathlib import Path
from ultralytics import YOLO

def setup_dataset():
    dataset_dir = Path(r"c:\Users\emir_\Documents\GitHub\tübitak\datasets\dataset v3")
    train_images_dir = dataset_dir / "train" / "images"
    train_labels_dir = dataset_dir / "train" / "labels"
    val_images_dir = dataset_dir / "valid" / "images"
    val_labels_dir = dataset_dir / "valid" / "labels"
    yaml_path = dataset_dir / "data.yaml"

    if not val_images_dir.exists():
        print("Validasyon klasörü bulunamadı, train'den %20 ayırılıyor...")
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
        print(f"{val_count} resim validasyon klasörüne taşındı.")

    # Fix data.yaml paths
    with open(yaml_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    
    data['train'] = str(train_images_dir.resolve())
    data['val'] = str(val_images_dir.resolve())
    if 'test' in data:
        del data['test']
        
    with open(yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(data, f, default_flow_style=False)
    
    return yaml_path

if __name__ == "__main__":
    yaml_file = setup_dataset()
    print("Eğitim başlıyor...")
    
    # Geçmiş model varsa oradan fine-tune et, yoksa yolov8m
    # Kullanıcı "v4" modeli olacağını söylemişti.
    model = YOLO("yolov8m.pt")
    
    results = model.train(
        data=str(yaml_file),
        epochs=100,
        imgsz=1024,
        batch=8,
        name="ev_charging_v4",
        project=r"c:\Users\emir_\Documents\GitHub\tübitak\runs",
        exist_ok=True,
        device="0"
    )
    print("Eğitim tamamlandı!")
