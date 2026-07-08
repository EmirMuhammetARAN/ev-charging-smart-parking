"""
Veri Setini Train / Valid / Test olarak böl.
Roboflow'dan gelen 150 resmi %70 Train, %20 Valid, %10 Test olarak ayırır.
"""
import os
import shutil
import random
import yaml

# =================== AYARLAR ===================
DATASET_DIR = r"C:\Users\emir_\Documents\GitHub\tübitak\datasets"
SEED = 42          # Tekrarlanabilirlik için sabit tohum
TRAIN_RATIO = 0.70
VALID_RATIO = 0.20
TEST_RATIO  = 0.10
# ================================================

random.seed(SEED)

src_images = os.path.join(DATASET_DIR, "train", "images")
src_labels = os.path.join(DATASET_DIR, "train", "labels")

# Tüm resim dosyalarını al ve karıştır
all_images = sorted([f for f in os.listdir(src_images) if f.endswith((".png", ".jpg", ".jpeg"))])
random.shuffle(all_images)

total = len(all_images)
n_train = int(total * TRAIN_RATIO)
n_valid = int(total * VALID_RATIO)
# Geri kalan test'e gider

splits = {
    "train": all_images[:n_train],
    "valid": all_images[n_train:n_train + n_valid],
    "test":  all_images[n_train + n_valid:],
}

print(f"Toplam resim: {total}")
for split_name, files in splits.items():
    print(f"  {split_name}: {len(files)} resim")

# Klasörleri oluştur ve dosyaları taşı
for split_name, files in splits.items():
    img_dir = os.path.join(DATASET_DIR, split_name, "images")
    lbl_dir = os.path.join(DATASET_DIR, split_name, "labels")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)

    for img_file in files:
        # Kaynak dosya yolları
        src_img = os.path.join(src_images, img_file)
        label_file = os.path.splitext(img_file)[0] + ".txt"
        src_lbl = os.path.join(src_labels, label_file)

        # Hedef dosya yolları
        dst_img = os.path.join(img_dir, img_file)
        dst_lbl = os.path.join(lbl_dir, label_file)

        # Train split'i zaten yerinde, sadece valid ve test'e taşı
        if split_name != "train":
            if os.path.exists(src_img):
                shutil.move(src_img, dst_img)
            if os.path.exists(src_lbl):
                shutil.move(src_lbl, dst_lbl)

# data.yaml'ı güncelle
data_yaml_path = os.path.join(DATASET_DIR, "data.yaml")
data = {
    "path": DATASET_DIR,
    "train": "train/images",
    "val": "valid/images",
    "test": "test/images",
    "nc": 4,
    "names": ["Fis_Bosta", "Fis_Takili", "car", "station"],
}

with open(data_yaml_path, "w", encoding="utf-8") as f:
    yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

print(f"\n✅ data.yaml güncellendi: {data_yaml_path}")
print("✅ Veri seti bölme işlemi tamamlandı!")
