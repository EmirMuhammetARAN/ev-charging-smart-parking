import random
import shutil
import cv2
import yaml
import numpy as np
import albumentations as A
from pathlib import Path

# Sabit Rastgelelik Cekirdegi (Reproducible Seed)
GLOBAL_SEED = 42
random.seed(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)

base_dir = Path(r"D:\GitHub\ev-charging-smart-parking")
val_img_dir = base_dir / "datasets" / "dataset v3" / "valid" / "images"
val_lbl_dir = base_dir / "datasets" / "dataset v3" / "valid" / "labels"

rain_dir = base_dir / "datasets" / "rain_test"
rain_img_dir = rain_dir / "images"
rain_lbl_dir = rain_dir / "labels"

rain_img_dir.mkdir(parents=True, exist_ok=True)
rain_lbl_dir.mkdir(parents=True, exist_ok=True)

# Guclu ve Belirgin Torrential (Saganak) Yagmur Transformu
rain_transform = A.RandomRain(
    slant_range=(-12, -6),
    drop_length=36,
    drop_width=1,
    drop_color=(220, 225, 235),
    blur_value=3,
    brightness_coefficient=0.78,
    rain_type="torrential",
    p=1.0
)

img_paths = sorted(list(val_img_dir.glob("*.png")) + list(val_img_dir.glob("*.jpg")))
print(f"Generating Torrential Rain with SEED={GLOBAL_SEED} for {len(img_paths)} validation images...")

for idx, p in enumerate(img_paths):
    img = cv2.imread(str(p))
    if img is None:
        continue
    
    # Her gorsel icin deterministik seed ata
    img_seed = GLOBAL_SEED + idx
    random.seed(img_seed)
    np.random.seed(img_seed)
    rain_transform.set_random_seed(img_seed)
    
    # Apply torrential rain
    rainy = rain_transform(image=img)["image"]
    
    # Save image
    out_img = rain_img_dir / p.name
    cv2.imwrite(str(out_img), rainy)
    
    # Copy label
    lbl_name = p.stem + ".txt"
    src_lbl = val_lbl_dir / lbl_name
    if src_lbl.exists():
        shutil.copy(str(src_lbl), str(rain_lbl_dir / lbl_name))

# Create data.yaml
v3_yaml = base_dir / "datasets" / "dataset v3" / "data.yaml"
with open(v3_yaml, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

rain_cfg = {
    "path": "D:/GitHub/ev-charging-smart-parking/datasets/rain_test",
    "train": "images",
    "val": "images",
    "nc": cfg["nc"],
    "names": cfg["names"]
}

with open(rain_dir / "data.yaml", "w", encoding="utf-8") as f:
    yaml.dump(rain_cfg, f, default_flow_style=False, allow_unicode=True)

img_count = len(list(rain_img_dir.glob("*.*")))
lbl_count = len(list(rain_lbl_dir.glob("*.txt")))
print(f"[+] datasets/rain_test (SEED={GLOBAL_SEED}) hazir: {img_count} gorsel, {lbl_count} etiket olusturuldu.")
