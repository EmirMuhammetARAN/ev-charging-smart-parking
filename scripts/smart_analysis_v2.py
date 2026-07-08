"""
=================================================================
  AKILLI OTOPARK ANALİZ SİSTEMİ v2 (End-to-End Öğrenme)
  YOLO tespitlerini alır, istasyon ve araç durumlarını
  doğrudan modelin kendi kararlarından okur.
  
  Sınıflar:
  0: Fis_Bosta
  1: Fis_Takili
  2: car (kullanım dışı)
  3: car_charging
  4: car_parked
  5: station_bosta
  6: station_park
  7: station_sarj
=================================================================
"""
from ultralytics import YOLO
import cv2
import numpy as np
import os
import sys

# =================== AYARLAR ===================
# Yeni eğittiğimiz modelin yolu (ev_charging_v3 klasörü altındaki best.pt)
MODEL_PATH = r"C:\Users\emir_\Documents\GitHub\tübitak\runs\ev_charging_v3\weights\best.pt"
CONFIDENCE_THRESHOLD = 0.25   # Minimum tespit güveni
IOU_THRESHOLD = 0.45          # NMS eşiği

# Çıktı klasörü
OUTPUT_DIR = r"C:\Users\emir_\Documents\GitHub\tübitak\runs\smart_analysis_v2"

# Renkler (BGR formatında) - Riskli vs Yüksek ayrımı (buna V2'de gerek kalmadı ama görsel olarak koruyoruz)
COLOR_YUKSEK_SARJ  = (0, 150, 0)      # Koyu yeşil
COLOR_YUKSEK_PARK  = (0, 0, 255)      # Kırmızıya yakın (Red)
COLOR_BOS          = (200, 200, 200)  # Gri
COLOR_BELIRSIZ     = (100, 100, 100)  # Koyu gri


# =================== YARDIMCI FONKSİYONLAR ===================

def box_center(box):
    """Bounding box'ın merkez noktasını hesaplar. box = [x1, y1, x2, y2]"""
    cx = (box[0] + box[2]) / 2
    cy = (box[1] + box[3]) / 2
    return (cx, cy)


def distance(center1, center2):
    """İki merkez noktası arasındaki Öklid mesafesini hesaplar."""
    return np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)


def get_car_color(cls_name):
    """Arabanın durumuna göre renk döndürür."""
    if cls_name == "car_charging":
        return COLOR_YUKSEK_SARJ
    elif cls_name == "car_parked":
        return COLOR_YUKSEK_PARK
    return COLOR_BELIRSIZ


def get_station_color(cls_name):
    """İstasyonun durumuna göre renk döndürür."""
    if cls_name == "station_sarj":
        return COLOR_YUKSEK_SARJ
    elif cls_name == "station_park":
        return COLOR_YUKSEK_PARK
    elif cls_name == "station_bosta":
        return COLOR_BOS
    return COLOR_BELIRSIZ


# =================== ANA ANALİZ FONKSİYONU ===================

def analyze_detections(detections, class_names):
    """
    YOLO tespitlerini alır ve akıllı analiz yapar.
    V2 Mimarisi: Durum doğrudan modelden gelir! Karar matrisi yoktur.
    Sadece UI için araba ve istasyonları görsel olarak eşleştiririz.
    """
    
    cars = []
    stations = []
    fis_bosta_list = []
    fis_takili_list = []
    
    # Tespitleri sınıflara göre ayır
    for det in detections:
        box = det["box"]
        cls = det["class"]
        conf = det["conf"]
        center = box_center(box)
        
        obj = {"box": box, "conf": conf, "center": center, "class_name": cls}
        
        if cls in ["car_charging", "car_parked"]:
            obj["station"] = None
            cars.append(obj)
        elif cls in ["station_bosta", "station_park", "station_sarj"]:
            obj["car"] = None
            stations.append(obj)
        elif cls == "Fis_Bosta":
            fis_bosta_list.append(obj)
        elif cls == "Fis_Takili":
            fis_takili_list.append(obj)
    
    # Arabaları X merkez koordinatına göre soldan sağa sırala
    cars.sort(key=lambda c: (c["box"][0] + c["box"][2]) / 2)
    stations.sort(key=lambda s: (s["box"][0] + s["box"][2]) / 2)
    
    # ---- ADIM 1: Her arabayı en yakın istasyona eşle (Sadece görsellik ve UI için) ----
    n_cars = len(cars)
    n_stats = len(stations)
    MAX_DIST = 400  # Araba alt-sağ ile istasyon altı arası maksimum mesafe
    
    # dp[i][j] = ilk i araba ile ilk j istasyonun minimum eşleşme maliyeti
    dp = [[float('inf')] * (n_stats + 1) for _ in range(n_cars + 1)]
    for j in range(n_stats + 1):
        dp[0][j] = 0
    for i in range(1, n_cars + 1):
        dp[i][0] = i * MAX_DIST
        
    for i in range(1, n_cars + 1):
        for j in range(1, n_stats + 1):
            car_br = (cars[i-1]["box"][2], cars[i-1]["box"][3])
            st_bottom = ((stations[j-1]["box"][0] + stations[j-1]["box"][2]) / 2, stations[j-1]["box"][3])
            dist = distance(car_br, st_bottom)
            
            cost_match = float('inf')
            if dist <= MAX_DIST:
                cost_match = dp[i-1][j-1] + dist
                
            cost_skip_st = dp[i][j-1]
            cost_skip_car = dp[i-1][j] + MAX_DIST
            
            dp[i][j] = min(cost_match, cost_skip_st, cost_skip_car)
            
    # Backtracking ile eşleşmeleri bul
    i, j = n_cars, n_stats
    while i > 0 and j > 0:
        car_br = (cars[i-1]["box"][2], cars[i-1]["box"][3])
        st_bottom = ((stations[j-1]["box"][0] + stations[j-1]["box"][2]) / 2, stations[j-1]["box"][3])
        dist = distance(car_br, st_bottom)
        
        if dp[i][j] == dp[i-1][j-1] + dist and dist <= MAX_DIST:
            cars[i-1]["station"] = stations[j-1]
            stations[j-1]["car"] = cars[i-1]
            i -= 1
            j -= 1
        elif dp[i][j] == dp[i][j-1]:
            j -= 1
        else:
            i -= 1
    
    return {
        "cars": cars,
        "stations": stations,
        "fis_bosta": fis_bosta_list,
        "fis_takili": fis_takili_list,
    }


# =================== GÖRSEL ÇİZİM ===================

def draw_analysis(image, analysis):
    """Analiz sonuçlarını görsel üzerine çizer."""
    img = image.copy()
    h, w = img.shape[:2]
    
    # İstasyonları soldan sağa sırala (görsel tutarlılık)
    sorted_stations = sorted(enumerate(analysis["stations"]), key=lambda x: x[1]["center"][0])
    
    # 1. İstasyonları çiz
    for visual_idx, (orig_idx, station) in enumerate(sorted_stations):
        box = station["box"]
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        
        color = get_station_color(station["class_name"])
        
        if station["class_name"] == "station_sarj":
            label = f"Ist.{visual_idx+1}: SARJ"
        elif station["class_name"] == "station_park":
            label = f"Ist.{visual_idx+1}: PARK"
        else:
            label = f"Ist.{visual_idx+1}: BOS"
        
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)
        
        # İstasyon etiketini üste yaz
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
        cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 2, y1), color, -1)
        cv2.putText(img, label, (x1 + 1, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)
    
    # 2. Arabaları ve durumlarını çiz
    for car in analysis["cars"]:
        box = car["box"]
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        
        color = get_car_color(car["class_name"])
        
        if car["class_name"] == "car_charging":
            label = f"SARJ (%{int(car['conf']*100)})"
        elif car["class_name"] == "car_parked":
            label = f"PARK (%{int(car['conf']*100)})"
        else:
            label = f"ARAC (%{int(car['conf']*100)})"
        
        # Araba kutusu
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        
        # Durum etiketi (arabanın altına)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        label_y = min(y2 + th + 8, h - 5)
        cv2.rectangle(img, (x1, label_y - th - 6), (x1 + tw + 4, label_y + 2), color, -1)
        cv2.putText(img, label, (x1 + 2, label_y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        
        # Açıklama (AI Tahmini)
        aciklama = "YAPAY ZEKA TESPITI"
        (aw, ah), _ = cv2.getTextSize(aciklama, cv2.FONT_HERSHEY_SIMPLEX, 0.25, 1)
        acy = min(label_y + ah + 6, h - 3)
        cv2.putText(img, aciklama, (x1 + 2, acy), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (255, 255, 255), 1)
        
        # İstasyona eşleşmişse çizgi çiz
        if car["station"]:
            sc = tuple(map(int, car["station"]["center"]))
            cc = tuple(map(int, car["center"]))
            line_color = color
            cv2.line(img, cc, sc, line_color, 1, cv2.LINE_AA)
    
    # 3. Fişleri çiz (Sadece görsel detay)
    for fb in analysis["fis_bosta"]:
        box = fb["box"]
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 165, 255), 1)
        cv2.putText(img, "bos", (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 165, 255), 1)
    
    for ft in analysis["fis_takili"]:
        box = ft["box"]
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 1)
        cv2.putText(img, "takili", (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 255), 1)
    
    # 4. Özet panel
    num_stations = len(analysis["stations"])
    panel_h = 30 + 4 * 18 + 10 + num_stations * 18 + 15
    panel_w = 280
    panel_h = min(panel_h, h - 10)
    
    overlay = img.copy()
    cv2.rectangle(overlay, (5, 5), (panel_w, panel_h), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.8, img, 0.2, 0, img)
    
    y_pos = 22
    cv2.putText(img, "OTOPARK DURUM RAPORU V2 (END-TO-END AI)", (12, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 255, 255), 1)
    y_pos += 6
    cv2.line(img, (12, y_pos), (panel_w - 10, y_pos), (0, 255, 255), 1)
    y_pos += 16
    
    sarj_count = sum(1 for s in analysis["stations"] if s["class_name"] == "station_sarj")
    bos_count = sum(1 for s in analysis["stations"] if s["class_name"] == "station_bosta")
    park_count = sum(1 for s in analysis["stations"] if s["class_name"] == "station_park")
    
    cv2.putText(img, f"Istasyon: {num_stations}  Arac: {len(analysis['cars'])}", (12, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
    y_pos += 18
    cv2.putText(img, f"  Sarj: {sarj_count}  Park: {park_count}  Bos: {bos_count}", (12, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
    y_pos += 6
    cv2.line(img, (12, y_pos), (panel_w - 10, y_pos), (100, 100, 100), 1)
    y_pos += 14
    
    for visual_idx, (orig_idx, station) in enumerate(sorted_stations):
        color = get_station_color(station["class_name"])
        
        car_idx_text = ""
        if station.get("car") is not None:
            for c_idx, c in enumerate(analysis["cars"]):
                if c is station["car"]:
                    car_idx_text = f" (Arac {c_idx + 1})"
                    break
        
        if station["class_name"] == "station_sarj":
            status_text = f"Ist.{visual_idx+1}: SARJ{car_idx_text}"
        elif station["class_name"] == "station_park":
            status_text = f"Ist.{visual_idx+1}: PARK{car_idx_text}"
        else:
            status_text = f"Ist.{visual_idx+1}: BOS"
        
        cv2.circle(img, (20, y_pos - 3), 4, color, -1)
        cv2.putText(img, status_text, (30, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.33, color, 1)
        y_pos += 18
    
    return img


# =================== KONSOL ÇIKTISI ===================

def print_report(analysis, image_name):
    print("\n" + "=" * 60)
    print(f"  AKILLI OTOPARK ANALIZ RAPORU V2")
    print(f"  Gorsel: {image_name}")
    print("=" * 60)
    
    cars = analysis["cars"]
    stations = analysis["stations"]
    
    sarj_count = sum(1 for s in stations if s["class_name"] == "station_sarj")
    bos_count = sum(1 for s in stations if s["class_name"] == "station_bosta")
    park_count = sum(1 for s in stations if s["class_name"] == "station_park")
    
    print(f"\n  Toplam Istasyon : {len(stations)}")
    print(f"  Toplam Arac     : {len(cars)}")
    print(f"  Sarj Eden       : {sarj_count}")
    print(f"  Park (sarjsiz)  : {park_count}")
    print(f"  Bos (musait)    : {bos_count}")
    
    print("\n" + "-" * 60)
    print("  ISTASYON DETAYLARI:")
    print("-" * 60)
    
    sorted_stations = sorted(enumerate(stations), key=lambda x: x[1]["center"][0])
    
    for visual_idx, (orig_idx, station) in enumerate(sorted_stations):
        if station["class_name"] == "station_sarj":
            icon = "  [ZAP]"
            status = "SARJ EDIYOR"
        elif station["class_name"] == "station_park":
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
            if car["class_name"] == "car_charging":
                durum = "SARJ OLUYOR"
            elif car["class_name"] == "car_parked":
                durum = "PARK ETMIS"
            else:
                durum = "BELIRSIZ"
                
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
                
                detections.append({
                    "box": box,
                    "class": cls_name,
                    "conf": conf,
                })
        
        analysis = analyze_detections(detections, model.names)
        print_report(analysis, img_name)
        
        image = cv2.imread(img_path)
        annotated = draw_analysis(image, analysis)
        
        out_path = os.path.join(OUTPUT_DIR, f"analiz_{img_name}")
        if not out_path.lower().endswith('.png'):
            out_path = os.path.splitext(out_path)[0] + '.png'
        cv2.imwrite(out_path, annotated)
    
    print(f"\nTum analizler tamamlandi!")
    print(f"Gorseller kaydedildi: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
