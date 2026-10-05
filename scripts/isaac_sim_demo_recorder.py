"""
=================================================================
  ISAAC SIM - TÜBİTAK Demo Videosu İçin Animasyon Scripti
=================================================================
  Bu kodu Isaac Sim'de Window -> Script Editor'e yapıştır ve çalıştır.
  
  Senaryo:
  1. Sabit araçlar istasyonda durur.
  2. Ana araç uzaktan istasyona yavaşça gelir.
  3. Araç durur, şarj kablosu araca takılır.
  4. Şarj kablosu yere atılır (Kritik Anomali!).
  5. Araç geri geri istasyondan çıkar.
  
  Tüm bu aşamalarda saniyede/karede bir fotoğraf çekilir.
  Sonra bu fotoğrafları birleştirip video yapacağız.
=================================================================
"""
import omni.usd
import omni.kit.app
from pxr import UsdGeom, Gf, Sdf
import asyncio
import carb
import os

# Ayarlar
OUTPUT_DIR = r"C:\Users\emir_\Documents\omniverse_demo_video"
RENDER_WAIT_TIME = 0.5  # Video için Path Tracing çok beklenmez, yarım saniye yeter
FPS = 10  # Saniyede kaç adım atacak

# --- Araç ve İstasyon Yolları (Kendi sahnene göre burayı düzelt) ---
# Hareket edecek olan ana araç
MAIN_CAR = "/World/Nissan_4x4" 
MAIN_STATION_SPOT = (1.8758, 5.6641) # Aracın park edeceği tam nokta
MAIN_STATION_CABLE_START = (1.6279, 4.2016, 0.3803)
MAIN_PLUG = "/World/Plug_1"

# Fiş ofsetleri (Nissan 4x4 için)
PLUG_OFFSET = Gf.Vec3d(0.3528, 0.3471, 0.2671)
PLUG_ORIENT = Gf.Vec3d(0, -90, -90)
CABLE_OFFSET = Gf.Vec3d(0.07, -0.01, -0.08)

# Diğer sabit araçlar (Sahnede süs olarak duracaklar)
STATIC_CARS = [
    {"path": "/World/Generic_Sedan_Car", "spot": (-0.7173, 5.6394)},
    {"path": "/World/Suv_Car", "spot": (-2.4362, 5.7075)}
]

# =================== YARDIMCI FONKSİYONLAR (Eski Kodundan Alıntı) ===================

def get_stage():
    return omni.usd.get_context().get_stage()

def move_car(car_path, x, y, z_offset=0.0, rotate_z=0.0):
    stage = get_stage()
    prim = stage.GetPrimAtPath(car_path)
    if not prim.IsValid(): return
    xformable = UsdGeom.Xformable(prim)
    
    found_translate = False
    for op in xformable.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
            op.Set(Gf.Vec3d(x, y, z_offset))
            found_translate = True
            break
    if not found_translate:
        xformable.AddTranslateOp().Set(Gf.Vec3d(x, y, z_offset))
        
    for op in xformable.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeOrient:
            rot = Gf.Rotation(Gf.Vec3d(0, 0, 1), rotate_z)
            op.Set(Gf.Quatd(rot.GetQuat().GetReal(), *rot.GetQuat().GetImaginary()))
        elif op.GetOpType() == UsdGeom.XformOp.TypeRotateXYZ and "unitsResolve" not in op.GetName():
            curr = op.Get() or Gf.Vec3d(0, 0, 0)
            op.Set(Gf.Vec3d(curr[0], curr[1], rotate_z))

def move_plug(plug_path, pos, rot):
    stage = get_stage()
    prim = stage.GetPrimAtPath(plug_path)
    if not prim.IsValid(): return
    xformable = UsdGeom.Xformable(prim)
    xformable.ClearXformOpOrder()
    for attr in list(prim.GetAttributes()):
        if attr.GetName().startswith("xformOp:"): prim.RemoveProperty(attr.GetName())
    xformable.AddTranslateOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*pos))
    xformable.AddRotateXYZOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*rot))
    xformable.AddScaleOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(0.8, 0.8, 0.8))
    xformable.AddRotateXYZOp(opSuffix="unitsResolve", precision=UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(90, 0, 0))
    xformable.AddScaleOp(opSuffix="unitsResolve", precision=UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(0.01, 0.01, 0.01))

def draw_catenary_cable(stage, prim_path, start_pos, end_pos, bow_y=0.0):
    curves = UsdGeom.BasisCurves.Get(stage, prim_path)
    if not curves: curves = UsdGeom.BasisCurves.Define(stage, prim_path)
    points = []
    x1, y1, z1 = start_pos
    x2, y2, z2 = end_pos
    min_z = min(z1, z2)
    cp_x = x1 + (x2 - x1) * 0.5
    cp_y = y1 + (y2 - y1) * 0.5 + bow_y
    cp_z = min_z - 0.4
    for i in range(20):
        t = i / 19.0
        x = (1 - t)**2 * x1 + 2 * (1 - t) * t * cp_x + t**2 * x2
        y = (1 - t)**2 * y1 + 2 * (1 - t) * t * cp_y + t**2 * y2
        z = (1 - t)**2 * z1 + 2 * (1 - t) * t * cp_z + t**2 * z2
        points.append(Gf.Vec3f(x, y, z))
    curves.GetPointsAttr().Set(points)
    curves.GetCurveVertexCountsAttr().Set([len(points)])
    curves.GetTypeAttr().Set(UsdGeom.Tokens.linear)
    curves.GetWidthsAttr().Set([0.02])
    color_attr = curves.GetPrim().GetAttribute('primvars:displayColor')
    if not color_attr: color_attr = curves.GetPrim().CreateAttribute('primvars:displayColor', Sdf.ValueTypeNames.Color3fArray)
    color_attr.Set([Gf.Vec3f(0.05, 0.05, 0.05)])

async def capture_screenshot(filepath):
    try:
        import omni.kit.capture
        cap = omni.kit.capture.CaptureOptions()
        cap.file_name = filepath
        omni.kit.capture.capture_next_frame(cap)
        await omni.kit.app.get_app().next_update_async()
    except:
        pass

# =================== ANİMASYON DÖNGÜSÜ ===================

async def run_demo_animation():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    stage = get_stage()
    frame_idx = 0

    def save_frame():
        nonlocal frame_idx
        filepath = os.path.join(OUTPUT_DIR, f"frame_{frame_idx:04d}.png")
        frame_idx += 1
        return filepath

    carb.log_warn("--- DEMO ANIMASYON BASLIYOR ---")
    
    # 1. SAHNE HAZIRLIĞI
    # Sabit araçları yerleştir
    for sc in STATIC_CARS:
        move_car(sc["path"], sc["spot"][0], sc["spot"][1], rotate_z=180.0)
    
    # Fişi başlangıçta istasyona as (Boşta)
    idle_plug_pos = (1.5775, 4.2043, 0.5630)
    move_plug(MAIN_PLUG, idle_plug_pos, (100, -90, 0))
    draw_catenary_cable(stage, "/World/Cable_1", MAIN_STATION_CABLE_START, (idle_plug_pos[0]-0.07, idle_plug_pos[1], idle_plug_pos[2]-0.08), 0.0)

    # Ana aracı istasyonun epey uzağına koy (Y=12.0)
    start_y = 12.0
    end_y = MAIN_STATION_SPOT[1]
    car_x = MAIN_STATION_SPOT[0]
    
    # 2. ARAÇ YANAŞIYOR (İleri doğru milim milim)
    carb.log_warn("Asama 1: Arac Yanasiyor...")
    steps = 40
    for step in range(steps):
        current_y = start_y - ((start_y - end_y) * (step / steps))
        move_car(MAIN_CAR, car_x, current_y, rotate_z=180.0)
        await asyncio.sleep(RENDER_WAIT_TIME)
        await capture_screenshot(save_frame())
    
    # Araba park etti, 10 kare bekle (1 saniye)
    for _ in range(10):
        await asyncio.sleep(RENDER_WAIT_TIME)
        await capture_screenshot(save_frame())

    # 3. KABLO ARACA TAKILIYOR (Şarj Başlıyor)
    carb.log_warn("Asama 2: Kablo Takiliyor...")
    plug_pos = (car_x - PLUG_OFFSET[0], end_y - PLUG_OFFSET[1], PLUG_OFFSET[2])
    plug_rot = (PLUG_ORIENT[0], PLUG_ORIENT[1], PLUG_ORIENT[2] + 180.0)
    move_plug(MAIN_PLUG, plug_pos, plug_rot)
    cable_end = (plug_pos[0] - CABLE_OFFSET[0], plug_pos[1] - CABLE_OFFSET[1], plug_pos[2] + CABLE_OFFSET[2])
    draw_catenary_cable(stage, "/World/Cable_1", MAIN_STATION_CABLE_START, cable_end, -0.15)
    
    # Şarj olurken 15 kare bekle
    for _ in range(15):
        await asyncio.sleep(RENDER_WAIT_TIME)
        await capture_screenshot(save_frame())

    # 4. KABLO YERE ATILIYOR (Kritik Anomali!)
    carb.log_warn("Asama 3: Fis Yerde!")
    drop_pos = (car_x - 0.2, end_y - 1.0, 0.044) # Yerde
    move_plug(MAIN_PLUG, drop_pos, (0, -90, 45))
    draw_catenary_cable(stage, "/World/Cable_1", MAIN_STATION_CABLE_START, drop_pos, 0.0)
    
    # Anomali varken 20 kare bekle
    for _ in range(20):
        await asyncio.sleep(RENDER_WAIT_TIME)
        await capture_screenshot(save_frame())
        
    # 5. ARAÇ GERİ GERİ ÇIKIYOR
    carb.log_warn("Asama 4: Arac Ayriliyor...")
    for step in range(steps):
        current_y = end_y + ((start_y - end_y) * (step / steps))
        move_car(MAIN_CAR, car_x, current_y, rotate_z=180.0)
        await asyncio.sleep(RENDER_WAIT_TIME)
        await capture_screenshot(save_frame())

    carb.log_warn(f"--- DEMO BITTI! {frame_idx} fotograf cekildi. Klasor: {OUTPUT_DIR} ---")

# Çalıştır
asyncio.ensure_future(run_demo_animation())
