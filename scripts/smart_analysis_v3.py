"""
=================================================================
  AKILLI OTOPARK ANALİZ SİSTEMİ v3 (Saf Yapay Zeka - No Algoritma)
  Bu versiyonda HİÇBİR kural tabanlı algoritma, mesafe ölçümü veya
  eşleştirme (DP) YAZILMAMIŞTIR. Her şey %100 yapay zekanın kendi
  doğrudan kararıdır.
  
  Sınıflar (7 Sınıf):
  0: Fis_Bosta
  1: Fis_Takili
  2: car_charging
  3: car_parked
  4: station_bosta
  5: station_park
  6: station_sarj
=================================================================
"""
from ultralytics import YOLO
import cv2
import os
import sys

# =================== AYARLAR ===================
MODEL_PATH = r"C:\Users\emir_\Documents\GitHub\tübitak\runs\ev_charging_v3\weights\best.pt"
CONFIDENCE_THRESHOLD = 0.25   
IOU_THRESHOLD = 0.45          

OUTPUT_DIR = r"C:\Users\emir_\Documents\GitHub\tübitak\runs\smart_analysis_v3"

# Renkler (BGR)
COLOR_SARJ  = (0, 150, 0)      # Koyu yeşil
COLOR_PARK  = (0, 0, 255)      # Kırmızı
COLOR_BOS   = (200, 200, 200)  # Gri

# =================== YARDIMCI FONKSİYONLAR ===================

def get_color(cls_name):
    if "sarj" in cls_name or "charging" in cls_name:
        return COLOR_SARJ
    elif "park" in cls_name or "parked" in cls_name:
        return COLOR_PARK
    return COLOR_BOS

def get_label_text(cls_name, conf=None):
    if cls_name == "car_charging":
        return f"SARJ (%{int(conf*100)})" if conf else "SARJ"
    elif cls_name == "car_parked":
        return f"PARK (%{int(conf*100)})" if conf else "PARK"
    elif cls_name == "station_sarj":
        return "Istasyon: SARJ"
    elif cls_name == "station_park":
        return "Istasyon: PARK"
    elif cls_name == "station_bosta":
        return "Istasyon: BOS"
    return cls_name

# =================== GÖRSEL ÇİZİM ===================

def draw_analysis(image, detections):
    """Saf YOLO çıktılarını görsel üzerine çizer."""
    img = image.copy()
    h, w = img.shape[:2]
    
    # Tespitleri sınıflandır
    cars = [d for d in detections if "car" in d["class"]]
    stations = [d for d in detections if "station" in d["class"]]
    fisler = [d for d in detections if "Fis" in d["class"]]
    
    # Arabaları ve İstasyonları soldan sağa sırala (Raporlama için)
    cars.sort(key=lambda c: c["box"][0])
    stations.sort(key=lambda s: s["box"][0])
    
    # 1. İstasyonları çiz
    for visual_idx, station in enumerate(stations):
        box = station["box"]
        x1, y1, x2, y2 = map(int, box)
        color = get_color(station["class"])
        
        # Basit isim
        if station["class"] == "station_sarj": label = f"Ist.{visual_idx+1}: SARJ"
        elif station["class"] == "station_park": label = f"Ist.{visual_idx+1}: PARK"
        else: label = f"Ist.{visual_idx+1}: BOS"
        
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
        cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 2, y1), color, -1)
        cv2.putText(img, label, (x1 + 1, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)
    
    # 2. Arabaları çiz
    for car in cars:
        box = car["box"]
        x1, y1, x2, y2 = map(int, box)
        color = get_color(car["class"])
        label = get_label_text(car["class"], car["conf"])
        
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        label_y = min(y2 + th + 8, h - 5)
        cv2.rectangle(img, (x1, label_y - th - 6), (x1 + tw + 4, label_y + 2), color, -1)
        cv2.putText(img, label, (x1 + 2, label_y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        
        aciklama = "YAPAY ZEKA TESPITI"
        (aw, ah), _ = cv2.getTextSize(aciklama, cv2.FONT_HERSHEY_SIMPLEX, 0.25, 1)
        cv2.putText(img, aciklama, (x1 + 2, min(label_y + ah + 6, h - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (255, 255, 255), 1)
    
    # 3. Fişleri çiz (Sadece dekoratif)
    for fis in fisler:
        box = fis["box"]
        x1, y1, x2, y2 = map(int, box)
        c = (0, 165, 255) if fis["class"] == "Fis_Bosta" else (255, 0, 255)
        text = "bos" if fis["class"] == "Fis_Bosta" else "takili"
        cv2.rectangle(img, (x1, y1), (x2, y2), c, 1)
        cv2.putText(img, text, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.3, c, 1)
    
    # 4. Sol üst özet panel (Hiçbir eşleştirme algoritması yok, sadece sayılar)
    num_stations = len(stations)
    panel_h = 30 + 4 * 18 + 10 + num_stations * 18 + 15
    panel_w = 280
    panel_h = min(panel_h, h - 10)
    
    overlay = img.copy()
    cv2.rectangle(overlay, (5, 5), (panel_w, panel_h), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.8, img, 0.2, 0, img)
    
    y_pos = 22
    cv2.putText(img, "OTOPARK DURUM RAPORU V3 (SAF AI)", (12, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 255, 255), 1)
    y_pos += 6
    cv2.line(img, (12, y_pos), (panel_w - 10, y_pos), (0, 255, 255), 1)
    y_pos += 16
    
    sarj_count = sum(1 for s in stations if s["class"] == "station_sarj")
    bos_count = sum(1 for s in stations if s["class"] == "station_bosta")
    park_count = sum(1 for s in stations if s["class"] == "station_park")
    
    cv2.putText(img, f"Istasyon: {num_stations}  Arac: {len(cars)}", (12, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
    y_pos += 18
    cv2.putText(img, f"  Sarj: {sarj_count}  Park: {park_count}  Bos: {bos_count}", (12, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
    y_pos += 6
    cv2.line(img, (12, y_pos), (panel_w - 10, y_pos), (100, 100, 100), 1)
    y_pos += 14
    
    for visual_idx, station in enumerate(stations):
        color = get_color(station["class"])
        if station["class"] == "station_sarj": status_text = f"Ist.{visual_idx+1}: SARJ"
        elif station["class"] == "station_park": status_text = f"Ist.{visual_idx+1}: PARK"
        else: status_text = f"Ist.{visual_idx+1}: BOS"
        
        cv2.circle(img, (20, y_pos - 3), 4, color, -1)
        cv2.putText(img, status_text, (30, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.33, color, 1)
        y_pos += 18
    
    return img

# =================== KONSOL ÇIKTISI ===================

def print_report(detections, image_name):
    print("\n" + "=" * 60)
    print(f"  AKILLI OTOPARK ANALIZ RAPORU V3 (SAF YAPAY ZEKA)")
    print(f"  Gorsel: {image_name}")
    print("=" * 60)
    
    cars = [d for d in detections if "car" in d["class"]]
    stations = [d for d in detections if "station" in d["class"]]
    
    cars.sort(key=lambda c: c["box"][0])
    stations.sort(key=lambda s: s["box"][0])
    
    sarj_count = sum(1 for s in stations if s["class"] == "station_sarj")
    bos_count = sum(1 for s in stations if s["class"] == "station_bosta")
    park_count = sum(1 for s in stations if s["class"] == "station_park")
    
    print(f"\n  Toplam Istasyon : {len(stations)}")
    print(f"  Toplam Arac     : {len(cars)}")
    print(f"  Sarj Eden       : {sarj_count}")
    print(f"  Park (sarjsiz)  : {park_count}")
    print(f"  Bos (musait)    : {bos_count}")
    
    print("\n" + "-" * 60)
    print("  ISTASYON DETAYLARI:")
    print("-" * 60)
    
    for visual_idx, station in enumerate(stations):
        if station["class"] == "station_sarj":
            icon = "  [ZAP]"
            status = "SARJ EDIYOR"
        elif station["class"] == "station_park":
            icon = "  [P]  "
            status = "PARK ETMIS"
        else:
            icon = "  [ ]  "
            status = "BOS - Musait"
        print(f"  {icon} Istasyon {visual_idx+1}: {status}")
    
    if cars:
        print("\n" + "-" * 60)
        print("  ARAC DETAYLARI:")
        print("-" * 60)
        
        for i, car in enumerate(cars):
            durum = "SARJ OLUYOR" if car["class"] == "car_charging" else "PARK ETMIS"
            print(f"\n  Arac {i+1}:")
            print(f"    Yapay Zeka Karari : {durum} (Guven: %{int(car['conf']*100)})")
            
    print("\n" + "=" * 60 + "\n")

# =================== ANA FONKSİYON ===================

def main():
    if len(sys.argv) > 1:
        source = sys.argv[1]
    else:
        source = r"C:\Users\emir_\Documents\GitHub\tübitak\datasets\dataset v2\train\images"
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("Model yukleniyor...")
    model = YOLO(MODEL_PATH)
    print(f"Model yuklendi: {MODEL_PATH}\n")
    
    if os.path.isdir(source):
        image_files = []
        for root, dirs, files in os.walk(source):
            for f in sorted(files):
                if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_files.append(os.path.join(root, f))
    else:
        image_files = [source]
    
    print(f"Toplam {len(image_files)} gorsel analiz edilecek...\n")
    
    for img_path in image_files:
        img_name = os.path.basename(img_path)
        results = model(img_path, conf=CONFIDENCE_THRESHOLD, iou=IOU_THRESHOLD, imgsz=1024, verbose=False)
        
        detections = []
        for result in results:
            boxes = result.boxes
            for i in range(len(boxes)):
                box = boxes.xyxy[i].cpu().numpy()
                cls_id = int(boxes.cls[i].cpu().numpy())
                conf = float(boxes.conf[i].cpu().numpy())
                cls_name = result.names[cls_id]
                detections.append({"box": box, "class": cls_name, "conf": conf})
        
        print_report(detections, img_name)
        
        image = cv2.imread(img_path)
        annotated = draw_analysis(image, detections)
        
        out_path = os.path.join(OUTPUT_DIR, f"analiz_{img_name}")
        if not out_path.lower().endswith('.png'):
            out_path = out_path.rsplit('.', 1)[0] + '.png'
        cv2.imwrite(out_path, annotated)
    
    print(f"\nTum analizler tamamlandi!")
    print(f"Gorseller kaydedildi: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
