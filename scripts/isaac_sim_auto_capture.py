"""
=================================================================
  ISAAC SIM - Sentetik EV Şarj İstasyonu Veri Seti Üretici
=================================================================
  Bu kodu Isaac Sim'de:
    Window -> Script Editor
  penceresine yapıştırıp çalıştır (Run butonuna bas).

  150 adet fotoğraf otomatik çekilecek.
  Arabalar rastgele park yerlerine yerleşir, bazıları gizlenir,
  bazıları ters park eder. Işık da değişir.
=================================================================
"""
import omni.usd
import omni.kit.commands
import omni.kit.app
from pxr import UsdGeom, Gf, Sdf
import random
import math
import os
import asyncio
import carb

# =================== AYARLAR ===================
TEST_MODE = False  # True yaparsan resim çekmez, beklemez, 1 kere kurar ve bırakır!
NUM_FRAMES = 1 if TEST_MODE else 40
# ONEMLI: Yol icinde Turkce karakter OLMAMALI (Omniverse renderercapture desteklemiyor)
OUTPUT_DIR = r"C:\Users\emir_\Documents\omniverse_captures_anomali"
# RTX Path Tracing'in pürüzsüzleşmesi (noise kalkması) için saniye cinsinden bekleme süresi:
# (1080p için ortalama 30-35 saniye, 1024x1024 için 15-20 saniye önerilir)
RENDER_WAIT_TIME = 18.0

# Anomali senaryoları (Kablo yerde, yanlış park) açık mı kapalı mı?
# Normal gece/sis çekimleri için False, Edge Case çekimleri için True yap!
ENABLE_ANOMALIES = True

# ---- Araç bilgileri (Stage yolları, scale, Z offset, şarj kapağı offsetleri) ----
CARS = {
    "/World/_020_Mazda_3_Hatchback": {
        "scale": Gf.Vec3d(30, 30, 30),
        "z_offset": 0.0,
        "plug_offset": Gf.Vec3d(0.2896, 0.4667, 0.30471),
        "plug_orient": Gf.Vec3d(180, -90, 90),
        "plug_scale": Gf.Vec3d(0.8, 0.8, 0.8),
        "cable_offset": Gf.Vec3d(0.07, -0.01, -0.08)  # Fişin kuyruğuna tam oturması için (Dışarı, Geriye, Aşağıya)
    },
    "/World/_026_Ferrari_849_Testarossa": {
        "scale": Gf.Vec3d(30, 30, 30),
        "z_offset": 0.0,
        "plug_offset": Gf.Vec3d(0.1287, 0.3946, 0.3397),
        "plug_orient": Gf.Vec3d(90, 0, -180),
        "plug_scale": Gf.Vec3d(0.8, 0.8, 0.8),
        "cable_offset": Gf.Vec3d(0.079, -0.01, 0.07)
    },
    "/World/Generic_Sedan_Car": {
        "scale": Gf.Vec3d(0.3, 0.3, 0.3),
        "z_offset": 0.0,
        "plug_offset": Gf.Vec3d(0.2894, 0.4285, 0.2799),
        "plug_orient": Gf.Vec3d(180, -90, 90),
        "plug_scale": Gf.Vec3d(0.8, 0.8, 0.8),
        "cable_offset": Gf.Vec3d(0.07, -0.01, -0.08)
    },
    "/World/Nissan_4x4": {
        "scale": Gf.Vec3d(0.3, 0.3, 0.3),
        "z_offset": 0.266,
        "plug_offset": Gf.Vec3d(0.3528, 0.3471, 0.2671),
        "plug_orient": Gf.Vec3d(0, -90, -90),
        "plug_scale": Gf.Vec3d(0.8, 0.8, 0.8),
        "cable_offset": Gf.Vec3d(0.07, -0.01, -0.08)
    },
    "/World/Suv_Car": {
        "scale": Gf.Vec3d(0.3, 0.3, 0.3),
        "z_offset": 0.0,
        "plug_offset": Gf.Vec3d(0.2783, 0.4039, 0.3091),
        "plug_orient": Gf.Vec3d(180, -90, 90),
        "plug_scale": Gf.Vec3d(0.8, 0.8, 0.8),
        "cable_offset": Gf.Vec3d(0.07, -0.01, -0.08)
    },
    "/World/Suv_Car_01": {
        "scale": Gf.Vec3d(0.3, 0.3, 0.3),
        "z_offset": 0.0,
        "plug_offset": Gf.Vec3d(0.2751, 0.4089, 0.3091),
        "plug_orient": Gf.Vec3d(180, -90, 90),
        "plug_scale": Gf.Vec3d(0.8, 0.8, 0.8),
        "cable_offset": Gf.Vec3d(0.07, -0.01, -0.08)
    },
}

# ---- İstasyon ve Park Bilgileri ----
STATIONS = [
    {
        "spot": (1.8758, 5.6641),
        "idle_plug": (1.5775, 4.2043, 0.5630),
        "cable_start": (1.6279, 4.2016, 0.3803)
    },
    {
        "spot": (1.0180, 5.6779),
        "idle_plug": (0.8686, 4.2043, 0.5630),
        "cable_start": (0.9189, 4.2016, 0.3803)
    },
    {
        "spot": (0.1525, 5.6565),
        "idle_plug": (0.1462, 4.2043, 0.5630),
        "cable_start": (0.1965, 4.2016, 0.3803)
    },
    {
        "spot": (-0.7173, 5.6394),
        "idle_plug": (-1.0602, 4.2043, 0.5630),
        "cable_start": (-1.0098, 4.2016, 0.3803)
    },
    {
        "spot": (-1.5677, 5.7125),
        "idle_plug": (-1.7684, 4.2043, 0.5630),
        "cable_start": (-1.7180, 4.2016, 0.3803)
    },
    {
        "spot": (-2.4362, 5.7075),
        "idle_plug": (-2.4803, 4.2043, 0.5630),
        "cable_start": (-2.4299, 4.2016, 0.3803)
    }
]

# Boştaki fişlerin kuyruğuna kabloyu oturtmak için X, Y, Z ofseti
IDLE_CABLE_OFFSET = Gf.Vec3d(-0.07, 0.0, -0.08)

CAR_PATHS = list(CARS.keys())


# =================== YARDIMCI FONKSİYONLAR ===================

def get_stage():
    return omni.usd.get_context().get_stage()


def move_car(car_path, x, y, rotate_z=0.0):
    """Bir aracı belirtilen konuma taşır ve Z ekseninde döndürür (scale ve Z offset korunur)."""
    stage = get_stage()
    prim = stage.GetPrimAtPath(car_path)
    if not prim.IsValid():
        carb.log_warn(f"HATA: {car_path} bulunamadi!")
        return

    info = CARS[car_path]
    z = info["z_offset"]

    xformable = UsdGeom.Xformable(prim)

    # Translate'i guncelle
    found_translate = False
    for op in xformable.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
            curr = op.Get()
            if isinstance(curr, Gf.Vec3f):
                op.Set(Gf.Vec3f(x, y, z))
            else:
                op.Set(Gf.Vec3d(x, y, z))
            found_translate = True
            break
    if not found_translate:
        xformable.AddTranslateOp().Set(Gf.Vec3d(x, y, z))
        
    # Rotasyon guncelle (180 derece dondurmek icin)
    for op in xformable.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeOrient:
            rot = Gf.Rotation(Gf.Vec3d(0, 0, 1), rotate_z)
            curr = op.Get()
            if isinstance(curr, Gf.Quatd):
                op.Set(Gf.Quatd(rot.GetQuat().GetReal(), *rot.GetQuat().GetImaginary()))
            else:
                op.Set(Gf.Quatf(rot.GetQuat().GetReal(), *rot.GetQuat().GetImaginary()))
        elif op.GetOpType() == UsdGeom.XformOp.TypeRotateXYZ and "unitsResolve" not in op.GetName():
            curr = op.Get() or Gf.Vec3d(0, 0, 0)
            if isinstance(curr, Gf.Vec3d):
                op.Set(Gf.Vec3d(curr[0], curr[1], rotate_z))
            else:
                op.Set(Gf.Vec3f(curr[0], curr[1], rotate_z))


def set_car_visible(car_path, visible):
    """Aracı görünür veya gizli yapar."""
    stage = get_stage()
    prim = stage.GetPrimAtPath(car_path)
    if not prim.IsValid():
        return
    imageable = UsdGeom.Imageable(prim)
    if visible:
        imageable.MakeVisible()
    else:
        imageable.MakeInvisible()


def randomize_sun():
    """Güneş açısını rastgele değiştirir (sabah/öğle/akşam efekti)."""
    stage = get_stage()
    light_path = "/Environment/DistantLight"
    prim = stage.GetPrimAtPath(light_path)
    if not prim.IsValid():
        carb.log_warn("DistantLight bulunamadi, isik degistirilmiyor.")
        return

    xformable = UsdGeom.Xformable(prim)
    sun_elevation = random.uniform(25, 75)  # 25=sabah/aksam, 75=ogle
    sun_azimuth = random.uniform(-40, 40)   # Sag-sol sapma

    # Mevcut rotateXYZ op'unu bul ve değiştir (Vec3d = double precision)
    for op in xformable.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeRotateXYZ:
            op.Set(Gf.Vec3d(-sun_elevation, sun_azimuth, 0))
            return


async def capture_screenshot(filepath):
    """Viewport'tan ekran görüntüsü alır."""
    captured = False

    # Yöntem 1: capture_viewport_to_file
    try:
        from omni.kit.viewport.utility import get_active_viewport
        viewport = get_active_viewport()
        import omni.kit.viewport.utility as vp_utils
        vp_utils.capture_viewport_to_file(viewport, filepath)
        captured = True
    except Exception as e1:
        pass

    # Yöntem 2: renderer_capture
    if not captured:
        try:
            import omni.renderer_capture
            rc = omni.renderer_capture.acquire_renderer_capture_interface()
            rc.capture_next_frame_swapchain(filepath)
            await omni.kit.app.get_app().next_update_async()
            captured = True
        except Exception as e2:
            pass

    # Yöntem 3: Kit capture (en yeni sürümler)
    if not captured:
        try:
            import omni.kit.capture
            cap = omni.kit.capture.CaptureOptions()
            cap.file_name = filepath
            omni.kit.capture.capture_next_frame(cap)
            await omni.kit.app.get_app().next_update_async()
            captured = True
        except Exception:
            pass

    if not captured:
        carb.log_warn(f"UYARI: Fotograf cekilemedi: {filepath}")

    return captured


def move_plug(plug_path, pos, rot, scale=None):
    """Fiş objesini belirtilen konuma, açıya ve boyuta getirir."""
    stage = get_stage()
    prim = stage.GetPrimAtPath(plug_path)
    if not prim.IsValid(): 
        carb.log_warn(f"move_plug: {plug_path} bulunamadi!")
        return
        
    xformable = UsdGeom.Xformable(prim)
    
    # 1. xformOpOrder listesini temizle
    xformable.ClearXformOpOrder()
    
    # 2. Eski xformOp attribute'larini fiziksel olarak sil (type cakismasi onlenir)
    attrs_to_remove = []
    for attr in prim.GetAttributes():
        if attr.GetName().startswith("xformOp:"):
            attrs_to_remove.append(attr.GetName())
    for attr_name in attrs_to_remove:
        prim.RemoveProperty(attr_name)
    
    # 3. Sifirdan dogru sirada ve dogru tipte (PrecisionDouble) olustur
    xformable.AddTranslateOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*pos))
    xformable.AddRotateXYZOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*rot))
    # Eger ozel scale verilmisse ekle (ornegin 0.8x kucultme)
    if scale is not None:
        xformable.AddScaleOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*scale))
    # Orijinal modelin birim donusumu (cm -> m)
    xformable.AddRotateXYZOp(opSuffix="unitsResolve", precision=UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(90, 0, 0))
    xformable.AddScaleOp(opSuffix="unitsResolve", precision=UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(0.01, 0.01, 0.01))


def draw_catenary_cable(stage, prim_path, start_pos, end_pos, is_charging=True, car_x=None):
    """Istasyondan fise sarkan siyah 3D kablo cizer.
    Kavisli bir Bezier egrisi kullanarak araba tamponunu ve sari dubalari (bollards) es gecer."""
    import math
    curves = UsdGeom.BasisCurves.Get(stage, prim_path)
    if not curves:
        curves = UsdGeom.BasisCurves.Define(stage, prim_path)
    
    num_points = 20
    points = []
    x1, y1, z1 = start_pos
    x2, y2, z2 = end_pos
    
    min_z = min(z1, z2)
    max_sag = max(0.0, min_z - 0.05)
    sag = 0.4 if is_charging else 0.15
    sag = min(sag, max_sag)
    
    # Dubalari ve tamponu asmak icin kontrol noktasi (Control Point)
    # Yanal kavis (bow_x) bazi araclarda ters S cizdigi icin iptal edildi.
    # Sadece arabaya degmemesi icin istasyona dogru (geriye) hafif kavis (bow_y) veriyoruz.
    bow_x = 0.0
    bow_y = -0.15 if is_charging else 0.0
    
    cp_x = x1 + (x2 - x1) * 0.5 + bow_x
    cp_y = y1 + (y2 - y1) * 0.5 + bow_y
    cp_z = min_z - sag
    
    for i in range(num_points):
        t = i / (num_points - 1)
        # Quadratic Bezier formu
        x = (1 - t)**2 * x1 + 2 * (1 - t) * t * cp_x + t**2 * x2
        y = (1 - t)**2 * y1 + 2 * (1 - t) * t * cp_y + t**2 * y2
        z = (1 - t)**2 * z1 + 2 * (1 - t) * t * cp_z + t**2 * z2
        points.append(Gf.Vec3f(x, y, z))
        
    curves.GetPointsAttr().Set(points)
    curves.GetCurveVertexCountsAttr().Set([len(points)])
    curves.GetTypeAttr().Set(UsdGeom.Tokens.linear)
    curves.GetWidthsAttr().Set([0.02])  # 2cm kalinlik
    
    color_attr = curves.GetPrim().GetAttribute('primvars:displayColor')
    if not color_attr:
        color_attr = curves.GetPrim().CreateAttribute('primvars:displayColor', Sdf.ValueTypeNames.Color3fArray)
    color_attr.Set([Gf.Vec3f(0.05, 0.05, 0.05)])


# =================== SENARYO ÜRETİCİ ===================

def setup_random_scenario(frame_idx):
    """
    Her kare için rastgele bir otopark ve KABLO senaryosu kurar.
    """
    num_cars = random.randint(1, 6)

    selected_cars = random.sample(CAR_PATHS, min(num_cars, len(CAR_PATHS)))
    
    # 0'dan 5'e kadar istasyon indeksleri seç
    selected_spot_indices = random.sample(range(6), min(num_cars, 6))

    # Seçilen arabaları yerleştir ve kablo tak
    for car, station_idx in zip(selected_cars, selected_spot_indices):
        station = STATIONS[station_idx]
        sx, sy = station["spot"]

        # Her zaman ANOMALI: Yanlis park!
        bad_park = True if ENABLE_ANOMALIES else False
        if bad_park:
            jitter_x = random.uniform(-0.8, 0.8) # Yan park alanina tasiyor
            jitter_y = random.uniform(-0.6, 0.6) # Cizgiden disari cikiyor
            rotate_z = 180.0 + random.uniform(-15, 15) # Capraz park
        else:
            jitter_x = random.uniform(-0.04, 0.04)
            jitter_y = random.uniform(-0.03, 0.03)
            rotate_z = 180.0

        car_x = sx + jitter_x
        car_y = sy + jitter_y

        # Arabalari 180 (veya capraz) derece dondurerek GERI GERI park ettir
        move_car(car, car_x, car_y, rotate_z=rotate_z)
        set_car_visible(car, True)

        # Test modunda hepsi sarj olsun ki kablolari gorelim, normalde %60
        is_charging = True if TEST_MODE else (random.random() < 0.6)
        plug_path = f"/World/Plug_{station_idx+1}"
        
        if is_charging:
            car_info = CARS[car]
            offset = car_info["plug_offset"]
            
            # Araba 180 derece dondugu icin (geri park), fisin dunya koordinatlari da tersine doner
            # Fis artik arabanin OTEKI tarafinda ve on-arka olarak TERS yonde
            plug_pos = (car_x - offset[0], car_y - offset[1], offset[2])
            
            # Fisin kendi rotasyonu da 180 derece donmeli ki arabaya dogru baksin
            orig_rot = car_info["plug_orient"]
            plug_rot_x = orig_rot[0]
            plug_rot_y = orig_rot[1]
            plug_rot_z = orig_rot[2] + 180.0
            
            # Ferrari icin fis z acisi her zaman 0 olmali (cunku tum istasyonlar solda)
            if "Ferrari" in car:
                plug_rot_z = 0.0

            plug_rot = (plug_rot_x, plug_rot_y, plug_rot_z)
            
            plug_scale = car_info.get("plug_scale", None)
            move_plug(plug_path, plug_pos, plug_rot, scale=plug_scale)
            
            # Kablo bitisini fisin lastik kuyruguna uzat (offset de 180 derece dondugu icin X ve Y ters cevrildi)
            c_off = car_info.get("cable_offset", Gf.Vec3d(0, 0, 0))
            cable_end_pos = (plug_pos[0] - c_off[0], plug_pos[1] - c_off[1], plug_pos[2] + c_off[2])
            
            # Istasyondan arabaya kisa ve puruzsuz sarkan kablo ciz
            draw_catenary_cable(get_stage(), f"/World/Cable_{station_idx+1}", station["cable_start"], cable_end_pos, is_charging=True, car_x=car_x)
        else:
            # Araba var ama şarj olmuyor (boşta)
            # Her zaman ANOMALI: Kullanıcı kabloyu yere atmış!
            drop_cable = True if ENABLE_ANOMALIES else False
            if drop_cable:
                # Fiş istasyonun (Y=4.2) önüne, arabanın (Y=5.6) tarafına düşmeli
                drop_x = station["idle_plug"][0] + random.uniform(-0.4, 0.4)
                drop_y = station["idle_plug"][1] + random.uniform(0.3, 1.2) # Kablo boyutu kadar öne
                drop_z = 0.044 # Asfalt yüksekliği (istasyon platformundan daha alçak)
                drop_rot = (0, -90, random.uniform(0, 360))
                move_plug(plug_path, (drop_x, drop_y, drop_z), drop_rot, scale=(0.8, 0.8, 0.8))
                draw_catenary_cable(get_stage(), f"/World/Cable_{station_idx+1}", station["cable_start"], (drop_x, drop_y, drop_z), is_charging=False)
            else:
                move_plug(plug_path, station["idle_plug"], (100, -90, 0), scale=(0.8, 0.8, 0.8))
                cable_end_pos = (station["idle_plug"][0] + IDLE_CABLE_OFFSET[0],
                                 station["idle_plug"][1] + IDLE_CABLE_OFFSET[1],
                                 station["idle_plug"][2] + IDLE_CABLE_OFFSET[2])
                draw_catenary_cable(get_stage(), f"/World/Cable_{station_idx+1}", station["cable_start"], cable_end_pos, is_charging=False)

    # Seçilmeyen arabaları GİZLE
    for car in CAR_PATHS:
        if car not in selected_cars:
            set_car_visible(car, False)

    # Boş kalan istasyonların fişlerini ve kısa kablolarını yerine as
    for station_idx in range(6):
        if station_idx not in selected_spot_indices:
            station = STATIONS[station_idx]
            plug_path = f"/World/Plug_{station_idx+1}"
            
            # Her zaman boş istasyonun da kablosu yere atılmış olsun
            drop_cable = True if ENABLE_ANOMALIES else False
            if drop_cable:
                # İstasyon boşken de kablo ön tarafa (park alanına doğru) düşmeli
                drop_x = station["idle_plug"][0] + random.uniform(-0.4, 0.4)
                drop_y = station["idle_plug"][1] + random.uniform(0.3, 1.2)
                drop_z = 0.044
                drop_rot = (0, -90, random.uniform(0, 360))
                move_plug(plug_path, (drop_x, drop_y, drop_z), drop_rot, scale=(0.8, 0.8, 0.8))
                draw_catenary_cable(get_stage(), f"/World/Cable_{station_idx+1}", station["cable_start"], (drop_x, drop_y, drop_z), is_charging=False)
            else:
                move_plug(plug_path, station["idle_plug"], (100, -90, 0), scale=(0.8, 0.8, 0.8))
                cable_end_pos = (station["idle_plug"][0] + IDLE_CABLE_OFFSET[0],
                                 station["idle_plug"][1] + IDLE_CABLE_OFFSET[1],
                                 station["idle_plug"][2] + IDLE_CABLE_OFFSET[2])
                draw_catenary_cable(get_stage(), f"/World/Cable_{station_idx+1}", station["cable_start"], cable_end_pos, is_charging=False)

    return num_cars


# =================== ANA FONKSİYON ===================

async def generate_dataset():
    """Ana veri üretim döngüsü - Bunu çalıştır!"""

    # Çıktı klasörünü oluştur
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 6 tane fişin olduğundan emin ol (otomatik çoğalt)
    stage = get_stage()
    root_layer = stage.GetRootLayer()
    
    base_plug_paths = ["/World/Electric_Vehicle_Cable", "/Electric_Vehicle_Cable"]
    valid_base_path = None
    for bp in base_plug_paths:
        if stage.GetPrimAtPath(bp).IsValid():
            valid_base_path = bp
            carb.log_warn(f">>> Fis modeli bulundu: {bp}")
            break
    
    if not valid_base_path:
        carb.log_warn("HATA: Electric_Vehicle_Cable bulunamadi! Stage icindeki yolunu kontrol et.")
    else:
        plugs_created = 0
        for i in range(1, 7):
            target_path = f"/World/Plug_{i}"
            if stage.GetPrimAtPath(target_path).IsValid():
                carb.log_warn(f"  Plug_{i} zaten var, atliyor.")
                plugs_created += 1
                continue
            
            created = False
            
            # Yontem 1: Sdf.CopySpec (en guvenilir USD yontemi)
            try:
                result = Sdf.CopySpec(root_layer, valid_base_path, root_layer, target_path)
                if result:
                    created = True
                    carb.log_warn(f"  Plug_{i} olusturuldu (Sdf.CopySpec)")
                else:
                    carb.log_warn(f"  Plug_{i} Sdf.CopySpec basarisiz, diger yontem deneniyor...")
            except Exception as e:
                carb.log_warn(f"  Plug_{i} Sdf.CopySpec hata: {e}")
            
            # Yontem 2: CopyPrims komutu (farkli parametre isimleri)
            if not created:
                try:
                    omni.kit.commands.execute('CopyPrims',
                        paths_from=[valid_base_path],
                        paths_to=[target_path])
                    if stage.GetPrimAtPath(target_path).IsValid():
                        created = True
                        carb.log_warn(f"  Plug_{i} olusturuldu (CopyPrims paths_to)")
                except Exception as e:
                    carb.log_warn(f"  Plug_{i} CopyPrims paths_to hata: {e}")
            
            # Yontem 3: CopyPrims komutu (duplicate_paths parametresi)
            if not created:
                try:
                    omni.kit.commands.execute('CopyPrims',
                        paths_from=[valid_base_path],
                        duplicate_paths=[target_path])
                    if stage.GetPrimAtPath(target_path).IsValid():
                        created = True
                        carb.log_warn(f"  Plug_{i} olusturuldu (CopyPrims duplicate_paths)")
                except Exception as e:
                    carb.log_warn(f"  Plug_{i} CopyPrims duplicate_paths hata: {e}")
            
            # Yontem 4: Manuel USD kopyalama
            if not created:
                try:
                    source_prim = stage.GetPrimAtPath(valid_base_path)
                    stage.DefinePrim(target_path, source_prim.GetTypeName())
                    # Referans olarak ekle
                    target_prim = stage.GetPrimAtPath(target_path)
                    if target_prim.IsValid():
                        target_prim.GetReferences().AddInternalReference(valid_base_path)
                        created = True
                        carb.log_warn(f"  Plug_{i} olusturuldu (Internal Reference)")
                except Exception as e:
                    carb.log_warn(f"  Plug_{i} manuel kopyalama hata: {e}")
            
            if created:
                plugs_created += 1
            else:
                carb.log_warn(f"  !!! Plug_{i} HICBIR YONTEMLE OLUSTURULAMADI !!!")
                
        carb.log_warn(f">>> Fis durumu: {plugs_created}/6 hazir")
    
    # Klonlama sonrasi stage'in guncellenmesini bekle
    await omni.kit.app.get_app().next_update_async()
    await omni.kit.app.get_app().next_update_async()

    carb.log_warn("=" * 55)
    carb.log_warn("  SENTETIK VERI URETIMI BASLIYOR")
    carb.log_warn(f"  Hedef: {NUM_FRAMES} fotograf")
    carb.log_warn(f"  Cikti:  {OUTPUT_DIR}")
    carb.log_warn("=" * 55)

    success_count = 0

    # İstenildiğinde durdurabilmek için global bir ayar (flag) oluştur
    carb.settings.get_settings().set_bool("/tubitak/stop_script", False)

    for i in range(NUM_FRAMES):
        # Durdurma komutu gelmiş mi kontrol et
        if carb.settings.get_settings().get("/tubitak/stop_script"):
            carb.log_warn("!!! KULLANICI TARAFINDAN URETIM DURDURULDU !!!")
            break

        # 1. Rastgele senaryo kur
        num_cars = setup_random_scenario(i)

        # 2. Her 15 karede bir güneşi değiştir
        if i % 15 == 0:
            randomize_sun()

        # 3. Test modu değilse render'ı bekle ve fotoğraf çek
        if not TEST_MODE:
            for _ in range(15):
                await omni.kit.app.get_app().next_update_async()

            # RTX Path Tracing kalitesi için (512/512 sample) ekstra bekleme suresi. 
            # RENDER_WAIT_TIME ayarindan ceker.
            await asyncio.sleep(RENDER_WAIT_TIME)

            # 4. Fotoğraf çek
            filepath = os.path.join(OUTPUT_DIR, f"frame_{i:04d}.png")
            ok = await capture_screenshot(filepath)
            if ok:
                success_count += 1

        # 5. İlerleme yazdır
        if (i + 1) % 10 == 0 or TEST_MODE:
            carb.log_warn(f"  [{i+1}/{NUM_FRAMES}]  Arabalar: {num_cars}  Basarili: {success_count}")
            
        if TEST_MODE:
            # Test modunda ekranın (viewport) güncellenebilmesi için motoru 1 kare ilerlet
            await omni.kit.app.get_app().next_update_async()
            carb.log_warn("TEST MODU CALISTI, EKRANA BAKABILIRSIN!")

    # Bittiğinde tüm arabaları tekrar göster (Test modundaysak sakın gizleme ki görelim)
    if not TEST_MODE:
        for car in CAR_PATHS:
            set_car_visible(car, True)

    carb.log_warn("=" * 55)
    carb.log_warn(f"  TAMAMLANDI!  {success_count}/{NUM_FRAMES} fotograf")
    carb.log_warn(f"  Konum: {OUTPUT_DIR}")
    carb.log_warn("=" * 55)


# =============== ÇALIŞTIR! ===============
asyncio.ensure_future(generate_dataset())
