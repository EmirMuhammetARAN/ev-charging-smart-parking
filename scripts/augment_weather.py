"""
HAVA DURUMU SİMÜLASYONU (Data Augmentation)
================================================
Mevcut V2 datasetindeki görüntülere yapay zeka ve görüntü işleme
ile Yağmur (Rain), Sis (Fog) ve Gece (Karanlık) efektleri ekler.

Bu script, datasetin sayısını artırarak modeli gerçek dünya 
koşullarına hazırlamak için tasarlanmıştır.

Kullanım:
    python scripts/augment_weather.py
"""

import os
import cv2
import glob
import shutil
import numpy as np
import albumentations as A

# Yollar
SOURCE_DATASET = r"C:\Users\emir_\Documents\GitHub\tübitak\datasets\dataset v2"
TARGET_DATASET = r"C:\Users\emir_\Documents\GitHub\tübitak\datasets\dataset v3 weather"

# Albumentations Augmentasyon Pipeline'ları
# Gece (Düşük Işık + Biraz Gürültü)
night_transform = A.Compose([
    A.RandomBrightnessContrast(brightness_limit=(-0.6, -0.4), contrast_limit=(-0.2, 0.2), p=1.0),
    A.GaussNoise(p=1.0)
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

# Sis (Fog)
fog_transform = A.Compose([
    A.RandomFog(p=1.0)
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

# Yağmur (Rain)
rain_transform = A.Compose([
    A.RandomRain(p=1.0)
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

# Kar (Snow)
snow_transform = A.Compose([
    A.RandomSnow(p=1.0)
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))


def parse_yolo_labels(label_path):
    """YOLO label dosyasını okur."""
    bboxes = []
    class_labels = []
    if not os.path.exists(label_path):
        return bboxes, class_labels
    
    with open(label_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                class_id = int(parts[0])
                x_center, y_center, width, height = map(float, parts[1:5])
                bboxes.append([x_center, y_center, width, height])
                class_labels.append(class_id)
    return bboxes, class_labels

def save_yolo_labels(bboxes, class_labels, out_path):
    """YOLO etiketlerini kaydeder."""
    with open(out_path, 'w', encoding='utf-8') as f:
        for bbox, cls_id in zip(bboxes, class_labels):
            f.write(f"{cls_id} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\n")


def augment_and_save(image, bboxes, class_labels, transform, effect_name, base_name, target_img_dir, target_lbl_dir):
    """Verilen efekti uygular ve kaydeder."""
    if not bboxes:  # Bbox yoksa sadece resmi augment et
        # Albumentations boş bbox kabul etmez, dummy verip geri alalım
        dummy_bbox = [[0.5, 0.5, 0.1, 0.1]]
        dummy_label = [0]
        try:
            transformed = transform(image=image, bboxes=dummy_bbox, class_labels=dummy_label)
            aug_img = transformed['image']
            aug_bboxes = []
            aug_labels = []
        except Exception as e:
            print(f"Hata ({effect_name}): {e}")
            return
    else:
        try:
            transformed = transform(image=image, bboxes=bboxes, class_labels=class_labels)
            aug_img = transformed['image']
            aug_bboxes = transformed['bboxes']
            aug_labels = transformed['class_labels']
        except Exception as e:
            print(f"Hata ({effect_name}): {e}")
            return

    # Kayıt yolları
    new_name = f"{base_name}_{effect_name}"
    img_out = os.path.join(target_img_dir, f"{new_name}.jpg")
    lbl_out = os.path.join(target_lbl_dir, f"{new_name}.txt")

    # cv2.imwrite(img_out, aug_img) -> Türkçe karakter desteği için
    is_success, buffer = cv2.imencode(".jpg", aug_img)
    if is_success:
        with open(img_out, 'wb') as f:
            f.write(buffer)
    if aug_bboxes:
        save_yolo_labels(aug_bboxes, aug_labels, lbl_out)


def main():
    print("HAVA DURUMU SIMULASYONU BASLIYOR...")
    
    src_images = os.path.join(SOURCE_DATASET, "train", "images")
    src_labels = os.path.join(SOURCE_DATASET, "train", "labels")
    
    tgt_images = os.path.join(TARGET_DATASET, "train", "images")
    tgt_labels = os.path.join(TARGET_DATASET, "train", "labels")
    
    os.makedirs(tgt_images, exist_ok=True)
    os.makedirs(tgt_labels, exist_ok=True)
    
    image_paths = glob.glob(os.path.join(src_images, "*.*"))
    print(f"Toplam {len(image_paths)} orijinal gorsel bulundu.")
    
    for idx, img_path in enumerate(image_paths):
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        lbl_path = os.path.join(src_labels, f"{base_name}.txt")
        
        # Orijinal resmi kopyala (v3 datasetinde orijinaller de olsun)
        shutil.copy(img_path, os.path.join(tgt_images, f"{base_name}_orig.jpg"))
        if os.path.exists(lbl_path):
            shutil.copy(lbl_path, os.path.join(tgt_labels, f"{base_name}_orig.txt"))
        
        # Görüntüyü oku (Türkçe karakter destekli)
        with open(img_path, 'rb') as f:
            file_bytes = np.asarray(bytearray(f.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            
        if image is None:
            print(f"Okunamadı: {img_path}")
            continue
            
        bboxes, class_labels = parse_yolo_labels(lbl_path)
        
        # 1. Yağmur Efekti
        augment_and_save(image, bboxes, class_labels, rain_transform, "rain", base_name, tgt_images, tgt_labels)
        
        # 2. Sis Efekti
        augment_and_save(image, bboxes, class_labels, fog_transform, "fog", base_name, tgt_images, tgt_labels)
        
        # 3. Gece Efekti
        augment_and_save(image, bboxes, class_labels, night_transform, "night", base_name, tgt_images, tgt_labels)
        
        # 4. Kar Efekti
        augment_and_save(image, bboxes, class_labels, snow_transform, "snow", base_name, tgt_images, tgt_labels)

        if (idx + 1) % 10 == 0:
            print(f"[{idx + 1}/{len(image_paths)}] Islendi...")

    # data.yaml dosyasını kopyala
    src_yaml = os.path.join(SOURCE_DATASET, "data.yaml")
    if os.path.exists(src_yaml):
        shutil.copy(src_yaml, os.path.join(TARGET_DATASET, "data.yaml"))

    print("\nIslem Tamamlandi!")
    print(f"Yeni dataset suraya kaydedildi: {TARGET_DATASET}")
    print("Artik bu yeni dataset ile hava kosullarina dayanikli V4 modelini egitebiliriz!")


if __name__ == "__main__":
    main()
