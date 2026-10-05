from pathlib import Path
from ultralytics import YOLO

def main():
    base_dir = Path(r"D:\GitHub\ev-charging-smart-parking")
    model_path = base_dir / "runs" / "ev_charging_v4" / "weights" / "best.pt"
    rain_yaml = base_dir / "datasets" / "rain_test" / "data.yaml"

    print(f"[*] Model yukleniyor: {model_path}")
    model = YOLO(str(model_path))

    print("[*] Yagmurlu veri seti uzerinde validasyon baslatiliyor...")
    metrics = model.val(
        data=str(rain_yaml),
        imgsz=1024,
        batch=8,
        conf=0.25,
        iou=0.6,
        device="0",
        workers=0,
        name="val_rain_v4",
        project=str(base_dir / "runs"),
        exist_ok=True
    )

    print("\n" + "="*60)
    print("       YAGMURLU TEST SONUCLARI (48 GORSEL - SEED 42)")
    print("="*60)
    print(f"Genel Precision : %{metrics.box.mp * 100:.2f}")
    print(f"Genel Recall    : %{metrics.box.mr * 100:.2f}")
    print(f"Genel mAP50     : %{metrics.box.map50 * 100:.2f}")
    print(f"Genel mAP50-95  : %{metrics.box.map * 100:.2f}")
    print("-" * 60)
    print(f"{'Sinif Adi':<16} | {'mAP50':<10} | {'mAP50-95':<10}")
    print("-" * 60)
    
    # metrics.box.ap50 -> Gercek Sinif Bazli AP50 degerleri
    # metrics.box.maps -> Sinif Bazli AP50-95 degerleri
    for cls_idx, cls_name in enumerate(metrics.names.values()):
        ap50_val = metrics.box.ap50[cls_idx] if len(metrics.box.ap50) > cls_idx else 0.0
        map_val = metrics.box.maps[cls_idx] if len(metrics.box.maps) > cls_idx else 0.0
        print(f"{cls_name:<16} | %{ap50_val * 100:<9.2f} | %{map_val * 100:<9.2f}")
    print("="*60)

if __name__ == "__main__":
    main()
