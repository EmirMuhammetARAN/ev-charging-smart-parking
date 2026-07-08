"""
=================================================================
  AKILLI OTOPARK ANALİZ SİSTEMİ v2
  YOLO tespitlerini alır, nesneleri birbirleriyle ilişkilendirir
  ve her istasyonun/arabanın şarj durumunu belirler.
  
  FIX LOG:
  - Exclusive 1:1 eşleştirme (bir fiş/istasyon sadece 1 araca)
  - DURUM 4: İkisi de yoksa → RİSKLİ ŞARJ (fis_bosta bulmak kolay)
  - Riskli vs Yüksek durumlar için farklı renkler
  - Panel boyutu dinamik hesaplama (taşma yok)
  - Belirsiz istasyon sorunu düzeltildi
=================================================================
"""
from ultralytics import YOLO
import cv2
import numpy as np
import os
import sys

# =================== AYARLAR ===================
MODEL_PATH = r"C:\Users\emir_\Documents\GitHub\tübitak\runs\ev_charging_v1\weights\best.pt"
CONFIDENCE_THRESHOLD = 0.25   # Minimum tespit güveni
IOU_THRESHOLD = 0.45          # NMS eşiği

# Nesneleri eşleştirmek için maksimum piksel mesafesi
MAX_PLUG_CAR_DISTANCE = 120       # Fiş-Araba eşleştirme mesafesi
MAX_PLUG_STATION_DISTANCE = 150   # Fiş-İstasyon eşleştirme mesafesi

# Çıktı klasörü
OUTPUT_DIR = r"C:\Users\emir_\Documents\GitHub\tübitak\runs\smart_analysis"

# Renkler (BGR formatında) - Riskli vs Yüksek ayrımı
COLOR_YUKSEK_SARJ  = (0, 150, 0)      # Koyu yeşil
COLOR_RISKLI_SARJ  = (0, 255, 170)    # Sarıya yakın açık yeşil / lime
COLOR_YUKSEK_PARK  = (0, 0, 255)      # Kırmızıya yakın (Red)
COLOR_RISKLI_PARK  = (0, 165, 255)    # Turuncuya yakın (Orange)
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


def get_car_color(car):
    """Arabanın durumuna göre renk döndürür (Riskli vs Yüksek ayrımı)."""
    aciklama = car.get("aciklama", "")
    durum = car.get("durum", "")
    
    if durum == "sarj_oluyor":
        if "DUSUK" in aciklama:
            return COLOR_RISKLI_SARJ
        return COLOR_YUKSEK_SARJ
    elif durum == "park_etmis":
        if "DUSUK" in aciklama:
            return COLOR_RISKLI_PARK
        return COLOR_YUKSEK_PARK
    return COLOR_BELIRSIZ


def get_station_color(station):
    """İstasyonun durumuna göre renk döndürür."""
    durum = station.get("durum", "")
    riskli = station.get("riskli", False)
    
    if durum == "sarj":
        return COLOR_RISKLI_SARJ if riskli else COLOR_YUKSEK_SARJ
    elif durum == "dolu_sarjsiz":
        return COLOR_RISKLI_PARK if riskli else COLOR_YUKSEK_PARK
    elif durum == "bos_musait":
        return COLOR_BOS
    return COLOR_BELIRSIZ


# =================== ANA ANALİZ FONKSİYONU ===================

def analyze_detections(detections, class_names):
    """
    YOLO tespitlerini alır ve akıllı analiz yapar.
    
    KARAR MATRİSİ:
    ===============
    1. Fis_Takili VAR + Fis_Bosta VAR → Güvenleri karşılaştır (RİSKLİ)
    2. Fis_Takili VAR + Fis_Bosta YOK → YÜKSEK OLASILIK ŞARJ
    3. Fis_Takili YOK + Fis_Bosta VAR → YÜKSEK OLASILIK PARK
    4. Fis_Takili YOK + Fis_Bosta YOK → RİSKLİ ŞARJ (çünkü Fis_Bosta bulmak kolay)
    5. Araba YOK                      → BOŞ
    
    EXCLUSIVE MATCHING (valid-bit):
    ===============================
    - Her Fis_Takili sadece 1 arabaya atanır (en yakın)
    - Her istasyon sadece 1 arabaya atanır (en yakın)
    - Atanan nesneler tekrar kullanılmaz
    """
    
    # Tespitleri sınıflara göre ayır
    cars = []
    stations = []
    fis_bosta_list = []
    fis_takili_list = []
    
    for det in detections:
        box = det["box"]
        cls = det["class"]
        conf = det["conf"]
        center = box_center(box)
        
        obj = {"box": box, "conf": conf, "center": center}
        
        if cls == "car":
            cars.append(obj)
        elif cls == "station":
            stations.append(obj)
        elif cls == "Fis_Bosta":
            fis_bosta_list.append(obj)
        elif cls == "Fis_Takili":
            fis_takili_list.append(obj)
    
    # ---- ADIM 1: Fişleri en yakın istasyonlara ata (exclusive: her fiş 1 istasyona) ----
    for station in stations:
        station["fis_bosta"] = None
        station["fis_takili"] = None
        station["car"] = None
        station["durum"] = "bos"
        station["riskli"] = False
        
    # Her Fis_Bosta'yı en yakın istasyona ata (exclusive)
    used_stations_bosta = set()
    # Tüm (fiş, istasyon) çiftlerini mesafeye göre sırala
    bosta_pairs = []
    for fb_idx, fb in enumerate(fis_bosta_list):
        for st_idx, station in enumerate(stations):
            d = distance(station["center"], fb["center"])
            if d < MAX_PLUG_STATION_DISTANCE:
                bosta_pairs.append((d, fb_idx, st_idx))
    bosta_pairs.sort(key=lambda x: x[0])
    
    used_fis_bosta = set()
    for d, fb_idx, st_idx in bosta_pairs:
        if fb_idx not in used_fis_bosta and st_idx not in used_stations_bosta:
            stations[st_idx]["fis_bosta"] = fis_bosta_list[fb_idx]
            used_fis_bosta.add(fb_idx)
            used_stations_bosta.add(st_idx)
            

    
    # ---- ADIM 2: Her arabayı en yakın istasyona eşle (EXCLUSIVE 1:1 matching) ----
    for car in cars:
        car["station"] = None
        car["fis_takili_yakin"] = None
        car["durum"] = "belirsiz"
        car["olasilik"] = 0
        car["aciklama"] = ""
    
    # Arabaları X merkez koordinatına göre soldan sağa sırala
    cars.sort(key=lambda c: (c["box"][0] + c["box"][2]) / 2)
    stations.sort(key=lambda s: (s["box"][0] + s["box"][2]) / 2)
    
    # ---- Dinamik Programlama (Sequence Alignment) ile Sıra Korumalı Eşleştirme ----
    # Soldan sağa sırayı KESİNLİKLE koruyarak (çizgilerin kesişmesini önler) eşleştirir.
    n_cars = len(cars)
    MAX_DIST = 400  # Top-Right köşe ile istasyon altı arası maksimum mesafe
    
    # ---- Dinamik Programlama (Sequence Alignment) ile Sıra Korumalı Eşleştirme ----
    # Soldan sağa sırayı KESİNLİKLE koruyarak (çizgilerin kesişmesini önler) eşleştirir.
    n_cars = len(cars)
    n_stats = len(stations)
    MAX_DIST = 400  # Araba merkezi ile istasyon altı arası maksimum mesafe
    
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
            
            # Seçenek 1: Bu arabayı bu istasyonla eşle
            cost_match = float('inf')
            if dist <= MAX_DIST:
                cost_match = dp[i-1][j-1] + dist
                
            # Seçenek 2: İstasyonu boş bırak (atla)
            cost_skip_st = dp[i][j-1]
            
            # Seçenek 3: Arabayı atla (hiçbir istasyona atama, otopark dışı say)
            cost_skip_car = dp[i-1][j] + MAX_DIST
            
            dp[i][j] = min(cost_match, cost_skip_st, cost_skip_car)
            
    # Backtracking ile eşleşmeleri bul
    i, j = n_cars, n_stats
    while i > 0 and j > 0:
        car_br = (cars[i-1]["box"][2], cars[i-1]["box"][3])
        st_bottom = ((stations[j-1]["box"][0] + stations[j-1]["box"][2]) / 2, stations[j-1]["box"][3])
        dist = distance(car_br, st_bottom)
        
        if dp[i][j] == dp[i-1][j-1] + dist and dist <= MAX_DIST:
            # Eşleştir!
            cars[i-1]["station"] = stations[j-1]
            stations[j-1]["car"] = cars[i-1]
            i -= 1
            j -= 1
        elif dp[i][j] == dp[i][j-1]:
            # İstasyon boş bırakıldı
            j -= 1
        else:
            # Araba otopark dışı kaldı
            i -= 1
    
    # ---- ADIM 2.5: Her Fis_Takili'yi en yakın arabaya ata (exclusive) ----
    used_cars_takili = set()
    takili_pairs = []
    for ft_idx, ft in enumerate(fis_takili_list):
        for car_idx, car in enumerate(cars):
            car_top_center = ((car["box"][0] + car["box"][2]) / 2, car["box"][1])
            d = distance(ft["center"], car_top_center)
            
            penalty = 0
            # Eger arabanin istasyonunda zaten bos fis varsa, bu araba sarj olamaz!
            if car.get("station") and car["station"].get("fis_bosta"):
                penalty = 1000
                
            if d < MAX_PLUG_CAR_DISTANCE:
                takili_pairs.append((d + penalty, ft_idx, car_idx, d)) # gercek d'yi de sakla
                
    takili_pairs.sort(key=lambda x: x[0])
    
    used_fis_takili = set()
    for penalized_d, ft_idx, car_idx, real_d in takili_pairs:
        if ft_idx not in used_fis_takili and car_idx not in used_cars_takili:
            cars[car_idx]["fis_takili"] = fis_takili_list[ft_idx]
            cars[car_idx]["fis_takili_yakin"] = {
                "box": fis_takili_list[ft_idx]["box"],
                "conf": fis_takili_list[ft_idx]["conf"],
                "dist": real_d
            }
            # EĞER arabanın bir istasyonu varsa, o istasyonun fişi de bu olur!
            if cars[car_idx]["station"] is not None:
                cars[car_idx]["station"]["fis_takili"] = fis_takili_list[ft_idx]
            used_fis_takili.add(ft_idx)
            used_cars_takili.add(car_idx)
    
    # ---- ADIM 3: Şarj durumunu belirle ----
    for car in cars:
        has_fis_takili = car["fis_takili_yakin"] is not None
        
        # İstasyondaki Fis_Bosta durumu
        station_has_fis_bosta = False
        if car["station"]:
            station_has_fis_bosta = car["station"]["fis_bosta"] is not None
        
        # ===== KARAR MATRİSİ =====
        
        if has_fis_takili and station_has_fis_bosta:
            # araba var + fiş takılı classı var + fis bosta classı var
            conf_takili = car["fis_takili_yakin"]["conf"]
            conf_bosta = car["station"]["fis_bosta"]["conf"]
            
            if conf_bosta > conf_takili:
                car["durum"] = "park_etmis"
                car["olasilik"] = int(conf_bosta * 100)
                car["aciklama"] = "DUSUK OLASILIK PARK"
                if car["station"]:
                    car["station"]["durum"] = "dolu_sarjsiz"
                    car["station"]["riskli"] = True
            else:
                car["durum"] = "sarj_oluyor"
                car["olasilik"] = int(conf_takili * 100)
                car["aciklama"] = "DUSUK OLASILIK SARJ"
                if car["station"]:
                    car["station"]["durum"] = "sarj"
                    car["station"]["riskli"] = True
                
        elif has_fis_takili and not station_has_fis_bosta:
            # araba var + fiş takılı classı var + fis bosta classı yok
            car["durum"] = "sarj_oluyor"
            car["olasilik"] = int(car["fis_takili_yakin"]["conf"] * 100)
            car["aciklama"] = "YUKSEK OLASILIK SARJ"
            if car["station"]:
                car["station"]["durum"] = "sarj"
            
        elif not has_fis_takili and station_has_fis_bosta:
            # araba var + fiş takılı classı yok + fis bosta classı var
            car["durum"] = "park_etmis"
            car["olasilik"] = int(car["station"]["fis_bosta"]["conf"] * 100)
            car["aciklama"] = "YUKSEK OLASILIK PARK"
            if car["station"]:
                car["station"]["durum"] = "dolu_sarjsiz"
            
        elif not has_fis_takili and not station_has_fis_bosta:
            # araba var + fiş takılı classı yok + fis bosta classı yok
            car["durum"] = "sarj_oluyor"
            car["olasilik"] = int(car["conf"] * 100)
            car["aciklama"] = "DUSUK OLASILIK SARJ"
            if car["station"]:
                car["station"]["durum"] = "sarj"
                car["station"]["riskli"] = True
    
    # Araba olmayan istasyonları işaretle
    for station in stations:
        if station["car"] is None:
            if station["fis_bosta"]:
                station["durum"] = "bos_musait"
            elif station["fis_takili"]:
                # İstasyonda fiş takılı ama araba yok (garip ama mümkün)
                station["durum"] = "bos_musait"
            else:
                station["durum"] = "bos_musait"
    
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
        
        color = get_station_color(station)
        
        if station["durum"] == "sarj":
            label = f"Ist.{visual_idx+1}: SARJ"
        elif station["durum"] == "dolu_sarjsiz":
            label = f"Ist.{visual_idx+1}: PARK"
        elif station["durum"] == "bos_musait":
            label = f"Ist.{visual_idx+1}: BOS"
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
        
        color = get_car_color(car)
        
        if car["durum"] == "sarj_oluyor":
            label = f"SARJ (%{car['olasilik']})"
        else:
            label = f"PARK (%{car['olasilik']})"
        
        # Araba kutusu
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        
        # Durum etiketi (arabanın altına)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        label_y = min(y2 + th + 8, h - 5)
        cv2.rectangle(img, (x1, label_y - th - 6), (x1 + tw + 4, label_y + 2), color, -1)
        cv2.putText(img, label, (x1 + 2, label_y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        
        # Açıklama (daha küçük)
        aciklama = car["aciklama"]
        (aw, ah), _ = cv2.getTextSize(aciklama, cv2.FONT_HERSHEY_SIMPLEX, 0.25, 1)
        acy = min(label_y + ah + 6, h - 3)
        cv2.putText(img, aciklama, (x1 + 2, acy), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (255, 255, 255), 1)
        
        # Eğer şarj oluyorsa, arabadan istasyona çizgi çiz
        if car["durum"] == "sarj_oluyor" and car["station"]:
            sc = tuple(map(int, car["station"]["center"]))
            cc = tuple(map(int, car["center"]))
            line_color = color
            cv2.line(img, cc, sc, line_color, 1, cv2.LINE_AA)
    
    # 3. Fişleri çiz
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
    
    # 4. Özet panel (sol üst köşe) - DİNAMİK BOYUT
    num_stations = len(analysis["stations"])
    panel_h = 30 + 4 * 18 + 10 + num_stations * 18 + 15  # Sadece istasyon satırları
    panel_w = 280
    panel_h = min(panel_h, h - 10)  # Resim boyutunu aşmasın
    
    overlay = img.copy()
    cv2.rectangle(overlay, (5, 5), (panel_w, panel_h), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.8, img, 0.2, 0, img)
    
    y_pos = 22
    cv2.putText(img, "OTOPARK DURUM RAPORU", (12, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
    y_pos += 6
    cv2.line(img, (12, y_pos), (panel_w - 10, y_pos), (0, 255, 255), 1)
    y_pos += 16
    
    # İstasyon özeti
    sarj_count = sum(1 for s in analysis["stations"] if s["durum"] == "sarj")
    bos_count = sum(1 for s in analysis["stations"] if "bos" in s["durum"])
    park_count = sum(1 for s in analysis["stations"] if s["durum"] == "dolu_sarjsiz")
    
    cv2.putText(img, f"Istasyon: {num_stations}  Arac: {len(analysis['cars'])}", (12, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
    y_pos += 18
    cv2.putText(img, f"  Sarj: {sarj_count}  Park: {park_count}  Bos: {bos_count}", (12, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
    y_pos += 6
    cv2.line(img, (12, y_pos), (panel_w - 10, y_pos), (100, 100, 100), 1)
    y_pos += 14
    
    # Her istasyonun detayı (soldan sağa sıralı)
    for visual_idx, (orig_idx, station) in enumerate(sorted_stations):
        color = get_station_color(station)
        riskli_tag = " !" if station.get("riskli", False) else ""
        
        car_idx_text = ""
        if station.get("car") is not None:
            for c_idx, c in enumerate(analysis["cars"]):
                if c is station["car"]:
                    car_idx_text = f" (Arac {c_idx + 1})"
                    break
        
        if station["durum"] == "sarj":
            status_text = f"Ist.{visual_idx+1}: SARJ{riskli_tag}{car_idx_text}"
        elif station["durum"] == "dolu_sarjsiz":
            status_text = f"Ist.{visual_idx+1}: PARK{riskli_tag}{car_idx_text}"
        elif station["durum"] == "bos_musait":
            status_text = f"Ist.{visual_idx+1}: BOS"
        else:
            status_text = f"Ist.{visual_idx+1}: BOS"
        
        cv2.circle(img, (20, y_pos - 3), 4, color, -1)
        cv2.putText(img, status_text, (30, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.33, color, 1)
        y_pos += 18
    
    return img


# =================== KONSOL ÇIKTISI ===================

def print_report(analysis, image_name):
    """Konsola detaylı rapor yazdırır."""
    print("\n" + "=" * 60)
    print(f"  AKILLI OTOPARK ANALIZ RAPORU")
    print(f"  Gorsel: {image_name}")
    print("=" * 60)
    
    cars = analysis["cars"]
    stations = analysis["stations"]
    
    sarj_count = sum(1 for s in stations if s["durum"] == "sarj")
    bos_count = sum(1 for s in stations if "bos" in s["durum"])
    park_count = sum(1 for s in stations if s["durum"] == "dolu_sarjsiz")
    
    print(f"\n  Toplam Istasyon : {len(stations)}")
    print(f"  Toplam Arac     : {len(cars)}")
    print(f"  Sarj Eden       : {sarj_count}")
    print(f"  Park (sarjsiz)  : {park_count}")
    print(f"  Bos (musait)    : {bos_count}")
    
    print("\n" + "-" * 60)
    print("  ISTASYON DETAYLARI:")
    print("-" * 60)
    
    # İstasyonları soldan sağa sırala
    sorted_stations = sorted(enumerate(stations), key=lambda x: x[1]["center"][0])
    
    for visual_idx, (orig_idx, station) in enumerate(sorted_stations):
        riskli_tag = " !" if station.get("riskli", False) else ""
        
        if station["durum"] == "sarj":
            icon = "  [ZAP]"
            status = f"SARJ EDIYOR{riskli_tag}"
        elif station["durum"] == "dolu_sarjsiz":
            icon = "  [P]  "
            status = f"PARK ETMIS{riskli_tag}"
        elif station["durum"] == "bos_musait":
            icon = "  [ ]  "
            status = "BOS - Musait"
        else:
            icon = "  [ ]  "
            status = "BOS"
        
        print(f"  {icon} Istasyon {visual_idx+1}: {status}")
    
    if cars:
        print("\n" + "-" * 60)
        print("  ARAC DETAYLARI:")
        print("-" * 60)
        
        for i, car in enumerate(cars):
            durum = "SARJ OLUYOR" if car["durum"] == "sarj_oluyor" else "PARK ETMIS"
            print(f"\n  Arac {i+1}:")
            print(f"    Durum      : {durum}")
            print(f"    Olasilik   : %{car['olasilik']}")
            print(f"    Aciklama   : {car['aciklama']}")
            
            if car["fis_takili_yakin"]:
                print(f"    Fis Takili  : EVET (guven: %{car['fis_takili_yakin']['conf']*100:.0f})")
            else:
                print(f"    Fis Takili  : HAYIR")
            
            if car["station"]:
                has_bosta = "EVET" if car["station"]["fis_bosta"] else "HAYIR"
                print(f"    Ist. Bos Fis: {has_bosta}")
    
    print("\n" + "=" * 60 + "\n")


# =================== ANA FONKSİYON ===================

def main():
    # Komut satırından resim yolu al (opsiyonel)
    if len(sys.argv) > 1:
        source = sys.argv[1]
    else:
        # Varsayılan: test klasörü
        source = r"C:\Users\emir_\Documents\GitHub\tübitak\datasets\test\images"
    
    # Çıktı klasörünü oluştur
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Modeli yükle
    print("Model yukleniyor...")
    model = YOLO(MODEL_PATH)
    print(f"Model yuklendi: {MODEL_PATH}\n")
    
    # Kaynak bir klasör mü yoksa tek bir dosya mı?
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
        
        # YOLO ile tespit yap
        results = model(img_path, conf=CONFIDENCE_THRESHOLD, iou=IOU_THRESHOLD, imgsz=1024, verbose=False)
        
        # Tespitleri listeye çevir
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
        
        # Akıllı analiz yap
        analysis = analyze_detections(detections, model.names)
        
        # Konsola rapor yazdır
        print_report(analysis, img_name)
        
        # Görsel üzerine çiz ve kaydet
        image = cv2.imread(img_path)
        annotated = draw_analysis(image, analysis)
        
        out_path = os.path.join(OUTPUT_DIR, f"analiz_{img_name}")
        # PNG olarak kaydet (kalite kaybı olmasın)
        if not out_path.lower().endswith('.png'):
            out_path = os.path.splitext(out_path)[0] + '.png'
        cv2.imwrite(out_path, annotated)
    
    print(f"\nTum analizler tamamlandi!")
    print(f"Gorseller kaydedildi: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
