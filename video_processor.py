import os
import sys

# --- ORTAM DEGISKENLERI ---
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

import tensorflow as tf
tf.get_logger().setLevel('ERROR')

import time
import math
import numpy as np
import cv2
import pickle 
import json 
import queue #İş kuyruğu için
from collections import deque 

from deepface import DeepFace
import torch
from ultralytics import YOLO
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtGui import QImage

# ==============================================================
#                      SISTEM AYARLARI
# ==============================================================

if getattr(sys, 'frozen', False):
    # Eğer program .exe olarak çalışıyorsa (PyInstaller)
    ANA_DIZIN = os.path.dirname(sys.executable)
else:
    # Eğer normal python dosyası olarak çalışıyorsa
    try:
        ANA_DIZIN = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        ANA_DIZIN = os.path.abspath('.')

# --- MODEL YOLLARI ---
MODEL_YUZ = os.path.join(ANA_DIZIN, 'yolov8n-face-lindevs.pt') 
MODEL_VUCUT = os.path.join(ANA_DIZIN, 'yolov8n.pt')            

# --- EKRAN AYARLARI ---
EKRAN_GENISLIK = 480
EKRAN_YUKSEKLIK = 360

# --- KAYNAK LISTESI ---
KAMERA_KAYNAKLARI = {
    1: 0,               # Yüz Tanıma
    2: "kosma.mp4",     # Koşma Analizi
    3: "bekleme.mp4",   # Bekleme Analizi
    4: "topluluk.mp4",  # Kalabalık Analizi
    5: "esya.mp4",      # Şüpheli Paket
    6: "araba.mp4"      # Hatalı Park
}

# --- VERITABANI AYARLARI ---
YETKILI_KLASORU = os.path.join(ANA_DIZIN, "Gorevli") 
OZNITELIK_DB_YOLU = os.path.join(ANA_DIZIN, "face_database.pkl") 
DB_INDEKS_YOLU = os.path.join(ANA_DIZIN, "face_db_index.json") 

# ==============================================================
#                      YARDIMCI SINIFLAR
# ==============================================================

def turkce_dosya_oku(yol):
    """Türkçe karakterli dosya yollarını okumak için."""
    yol = str(yol)
    if not os.path.exists(yol): return None
    veri = np.fromfile(yol, dtype=np.uint8)
    resim = cv2.imdecode(veri, cv2.IMREAD_COLOR)
    return resim

class YuzDogrulayici:
    def __init__(self, referans_klasor_yolu):
        self.referans_klasor_yolu = referans_klasor_yolu
        self.referans_oznitelikler_ort = {} 
        self._veritabani_yukle_veya_olustur()

    def _veritabani_yukle_veya_olustur(self):
        if self._veritabani_gecerli_mi():
            try:
                with open(OZNITELIK_DB_YOLU, 'rb') as f: self.referans_oznitelikler_ort = pickle.load(f)
                print(f"[YuzDogrulayici] Veritabanı Yüklendi: {len(self.referans_oznitelikler_ort)} kişi.")
            except: self._veritabani_yeniden_olustur()
        else: self._veritabani_yeniden_olustur()

    def _veritabani_gecerli_mi(self):
        if not os.path.exists(OZNITELIK_DB_YOLU): return False
        try:
            with open(DB_INDEKS_YOLU, 'r') as f: indeks_sayimlari = json.load(f)
            return indeks_sayimlari == self._disk_sayimi_yap()
        except: return False

    def _disk_sayimi_yap(self):
        sayimlar = {}
        if not os.path.isdir(self.referans_klasor_yolu): return sayimlar
        for d in os.listdir(self.referans_klasor_yolu):
            p = os.path.join(self.referans_klasor_yolu, d)
            if os.path.isdir(p):
                sayimlar[d] = len([f for f in os.listdir(p) if f.lower().endswith(('.png','.jpg','.jpeg'))])
        return sayimlar

    def _veritabani_yeniden_olustur(self):
        print("[YuzDogrulayici] Veritabanı oluşturuluyor (Facenet512)...") 
        tum_oznitelikler, sayimlar = {}, {}
        
        if not os.path.isdir(self.referans_klasor_yolu): 
            print(f"UYARI: '{self.referans_klasor_yolu}' klasörü bulunamadı!")
            return

        for kisi in os.listdir(self.referans_klasor_yolu):
            kisi_dizin = os.path.join(self.referans_klasor_yolu, kisi)
            if not os.path.isdir(kisi_dizin): continue
            
            tum_oznitelikler[kisi] = []
            resimler = [f for f in os.listdir(kisi_dizin) if f.lower().endswith(('.png','.jpg','.jpeg'))]
            sayimlar[kisi] = len(resimler)
            
            for resim_adi in resimler:
                resim_yolu = os.path.join(kisi_dizin, resim_adi)
                resim = turkce_dosya_oku(resim_yolu)
                
                if resim is None: continue
                try:
                    sonuc = DeepFace.represent(resim, model_name="Facenet512", detector_backend="opencv", enforce_detection=False)
                    if sonuc:
                        oznitelik = sonuc[0]["embedding"]
                        tum_oznitelikler[kisi].append(np.array(oznitelik, dtype=np.float32))
                except Exception:
                    continue
        
        self.referans_oznitelikler_ort = {k: np.mean(v, axis=0) for k, v in tum_oznitelikler.items() if v}
        
        try:
            with open(OZNITELIK_DB_YOLU, 'wb') as f: pickle.dump(self.referans_oznitelikler_ort, f)
            with open(DB_INDEKS_YOLU, 'w', encoding='utf-8') as f: json.dump(sayimlar, f)
            print("[YuzDogrulayici] Veritabanı kaydedildi!")
        except Exception as e:
            print(f"Veritabanı kayıt hatası: {e}")

    def dogrula(self, yuz_resmi):
        try:
            sonuc = DeepFace.represent(yuz_resmi, model_name="Facenet512", detector_backend="opencv", enforce_detection=False)
            if not sonuc: return (False, 999.0, None)
            
            anlik_oznitelik = np.array(sonuc[0]["embedding"], dtype=np.float32)
            DOGRULAMA_ESIGI = 0.40 

            min_mesafe, kimlik = 999.0, None
            for isim, ref_oznitelik in self.referans_oznitelikler_ort.items():
                mesafe = 1.0 - (np.dot(anlik_oznitelik, ref_oznitelik) / (np.linalg.norm(anlik_oznitelik) * np.linalg.norm(ref_oznitelik) + 1e-12))
                if mesafe < min_mesafe: min_mesafe, kimlik = mesafe, isim
            
            return (min_mesafe <= DOGRULAMA_ESIGI, min_mesafe, kimlik)
        except: return (False, 999.0, None)

# =========================================================================
# ARKA PLAN THREAD
# =========================================================================
class AnalizIschisi(QThread):
    sonuc_sinyali = pyqtSignal(int, str, float) # ID, Isim, Mesafe

    def __init__(self, yuz_dogrulayici):
        super().__init__()
        self.yuz_dogrulayici = yuz_dogrulayici
        self.kuyruk = queue.Queue()
        self._calisiyor = True

    def istek_ekle(self, takip_id, yuz_resmi):
        if self.kuyruk.empty(): # Yığılmayı önle
            self.kuyruk.put((takip_id, yuz_resmi))

    def run(self):
        while self._calisiyor:
            try:
                takip_id, yuz_resmi = self.kuyruk.get(timeout=0.1)
                dogru, mesafe, isim = self.yuz_dogrulayici.dogrula(yuz_resmi)
                
                if dogru:
                    self.sonuc_sinyali.emit(takip_id, isim, mesafe)
                else:
                    self.sonuc_sinyali.emit(takip_id, "Bilinmiyor", mesafe)
            except queue.Empty:
                continue

    def stop(self):
        self._calisiyor = False
        self.wait(3000)

# =========================================================================
# THREAD 1: YUZ TANIMA (KAMERA 1) - GÜNCELLENMİŞ VERSİYON
# =========================================================================
class YuzTanimaThread(QThread):
    goruntu_sinyali = pyqtSignal(QImage, int)
    anomali_sinyali = pyqtSignal(str)
    kamera_durum_sinyali = pyqtSignal(int, bool)

    def __init__(self, kamera_indeks, kaynak):
        super().__init__()
        self._calisiyor = True
        self.kamera_indeks = kamera_indeks
        
        self.video_dosyasi_mi = isinstance(kaynak, str)
        if self.video_dosyasi_mi:
            olasi_yol = os.path.join(ANA_DIZIN, "Videolar", str(kaynak))
            self.video_kaynagi = olasi_yol if os.path.exists(olasi_yol) else kaynak
        else:
            self.video_kaynagi = kaynak

        self.cihaz = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Modelleri burada yükleme (Donmayı önlemek için None yapıyoruz)
        self.model = None
        self.yuz_dogrulayici = None
        self.analiz_iscisi = None

        self.onbellek_kutular = [] 
        self.takip_edilen_kisiler = {}
        self.alarm_aktif = True
        self.kare_sayaci = 0
        self.MIN_YUZ_BOYUTU = 30

    def analiz_sonucu_islesin(self, takip_id, isim, mesafe):
        """İşçi Thread'den gelen sonuçları işler"""
        if takip_id in self.takip_edilen_kisiler:
            kisi = self.takip_edilen_kisiler[takip_id]
            kisi["son_kontrol"] = time.time()
            
            if isim != "Bilinmiyor":
                if kisi["kimlik"] != isim:
                    self.anomali_sinyali.emit(f"✅ YETKILI: {isim}")
                kisi["kimlik"] = isim
                kisi["kilitli"] = True
            else:
                kisi["hata_sayisi"] += 1
                if kisi["hata_sayisi"] > 4:
                    if kisi["kimlik"] != "Onaylanmadi":
                        self.anomali_sinyali.emit(f"❌ YETKISIZ (ID: {takip_id})")
                    kisi["kimlik"] = "Onaylanmadi"

    def run(self):
        # Modelleri Arka Planda Yükle (Donma Çözümü 1)
        if self.model is None:
            self.model = YOLO(MODEL_YUZ).to(self.cihaz)
        if self.yuz_dogrulayici is None:
            self.yuz_dogrulayici = YuzDogrulayici(YETKILI_KLASORU)
        
        # İşçiyi Başlat
        if self.analiz_iscisi is None:
            self.analiz_iscisi = AnalizIschisi(self.yuz_dogrulayici)
            self.analiz_iscisi.sonuc_sinyali.connect(self.analiz_sonucu_islesin)
            self.analiz_iscisi.start()

        if isinstance(self.video_kaynagi, int) and os.name == 'nt':
            kamera = cv2.VideoCapture(self.video_kaynagi, cv2.CAP_DSHOW)
        else:
            kamera = cv2.VideoCapture(self.video_kaynagi)
        
        while self._calisiyor:
            basarili, kare = kamera.read()
            if not basarili:
                if self.video_dosyasi_mi: kamera.set(cv2.CAP_PROP_POS_FRAMES, 0); continue
                else: break
            
            kare_boyutlandirilmis = cv2.resize(kare, (EKRAN_GENISLIK, EKRAN_YUKSEKLIK))
            cikti_karesi = kare_boyutlandirilmis.copy()
            self.kare_sayaci += 1
            
            # Takibi sıklaştırdık (%3) daha akıcı olsun diye
            if self.kare_sayaci % 3 == 0:
                kucuk_kare = cv2.resize(kare_boyutlandirilmis, (320, 240))
                sonuclar = self.model.track(kucuk_kare, persist=True, verbose=False, classes=[0], conf=0.5, imgsz=320)
                
                olcek_x = EKRAN_GENISLIK / 320
                olcek_y = EKRAN_YUKSEKLIK / 240
                
                if sonuclar[0].boxes.id is not None:
                    kutular = sonuclar[0].boxes.xyxy.cpu().tolist()
                    takip_idleri = sonuclar[0].boxes.id.int().cpu().tolist()
                    
                    yeni_onbellek_kutular = []
                    mevcut_idler = set()
                    
                    for kutu, takip_id in zip(kutular, takip_idleri):
                        x1, y1, x2, y2 = kutu
                        x1, x2 = int(x1 * olcek_x), int(x2 * olcek_x)
                        y1, y2 = int(y1 * olcek_y), int(y2 * olcek_y)
                        
                        yeni_onbellek_kutular.append(((x1,y1,x2,y2), takip_id))
                        mevcut_idler.add(takip_id)
                        
                        if takip_id not in self.takip_edilen_kisiler:
                            self.takip_edilen_kisiler[takip_id] = {
                                "kimlik": "Bilinmiyor", "hata_sayisi": 0, "kilitli": False, "son_kontrol": 0
                            }
                        
                        kisi = self.takip_edilen_kisiler[takip_id]
                        
                        # --- ANALİZ İSTEĞİ (DONMA ÇÖZÜMÜ 2) ---
                        if self.alarm_aktif and not kisi["kilitli"]:
                            su_an = time.time()
                            if su_an - kisi["son_kontrol"] > 1.0:
                                yuz_img = kare_boyutlandirilmis[max(0,y1):min(EKRAN_YUKSEKLIK,y2), max(0,x1):min(EKRAN_GENISLIK,x2)]
                                if yuz_img.size > 0 and (x2-x1) > self.MIN_YUZ_BOYUTU:
                                    # Beklemek yok, işçiye sipariş ver
                                    self.analiz_iscisi.istek_ekle(takip_id, yuz_img.copy())
                                     
                    self.onbellek_kutular = yeni_onbellek_kutular
                    for tid in list(self.takip_edilen_kisiler.keys()):
                        if tid not in mevcut_idler: del self.takip_edilen_kisiler[tid]
                else:
                    self.onbellek_kutular = []

            # Çizim İşlemleri (Değişmedi)
            yetkisiz_var = False
            if self.alarm_aktif:
                for (kutu, takip_id) in self.onbellek_kutular:
                    if takip_id in self.takip_edilen_kisiler:
                        kisi = self.takip_edilen_kisiler[takip_id]
                        renk = (0, 255, 0) # Yeşil
                        etiket = kisi["kimlik"]
                        
                        if etiket in ["Bilinmiyor", "Onaylanmadi"]:
                            renk = (0, 0, 255) # Kırmızı
                            yetkisiz_var = True
                        
                        x1, y1, x2, y2 = kutu
                        cv2.rectangle(cikti_karesi, (x1, y1), (x2, y2), renk, 2)
                        cv2.putText(cikti_karesi, etiket, (x1, y1-5), 0, 0.6, renk, 2)

            self.kamera_durum_sinyali.emit(self.kamera_indeks, yetkisiz_var)
            
            rgb = cv2.cvtColor(cikti_karesi, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            qimg = QImage(rgb.data, w, h, ch*w, QImage.Format_RGB888).copy()
            self.goruntu_sinyali.emit(qimg, self.kamera_indeks)
            
            time.sleep(0.03) # Akıcılık için bekleme
            
        kamera.release()
        if self.analiz_iscisi: self.analiz_iscisi.stop()

    def stop(self): self._calisiyor = False; self.wait(3000)

# =========================================================================
# THREAD 2: KOSMA ANALIZI (KAMERA 2)
# =========================================================================
class HareketAnalizThread(QThread):
    goruntu_sinyali = pyqtSignal(QImage, int)
    anomali_sinyali = pyqtSignal(str)
    kamera_durum_sinyali = pyqtSignal(int, bool)

    def __init__(self, kamera_indeks, kaynak):
        super().__init__()
        self._calisiyor = True
        self.kamera_indeks = kamera_indeks
        
        self.video_dosyasi_mi = isinstance(kaynak, str)
        if self.video_dosyasi_mi:
            olasi_yol = os.path.join(ANA_DIZIN, "Videolar", str(kaynak))
            self.gercek_kaynak = olasi_yol if os.path.exists(olasi_yol) else kaynak
        else:
            self.gercek_kaynak = kaynak

        self.cihaz = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = YOLO(MODEL_VUCUT).to(self.cihaz)
        
        self.takip_edilen_nesneler = {} 
        self.alarm_aktif = True
        self.onbellek_kutular = [] 
        self.kare_sayaci = 0
        self.ANALIZ_SIKLIGI = 3
        self.BAGIL_HIZ_ESIGI = 0.8

    def run(self):
        kamera = cv2.VideoCapture(self.gercek_kaynak)
        fps = kamera.get(cv2.CAP_PROP_FPS)
        if fps is None or fps == 0 or fps > 120: fps = 30 
        kare_suresi = 1.0 / fps 
        
        while self._calisiyor:
            baslangic = time.time() 
            basarili, kare = kamera.read()
            if not basarili:
                if self.video_dosyasi_mi: kamera.set(cv2.CAP_PROP_POS_FRAMES, 0); continue
                else: break
            
            kare_boyutlandirilmis = cv2.resize(kare, (EKRAN_GENISLIK, EKRAN_YUKSEKLIK))
            cikti_karesi = kare_boyutlandirilmis.copy()
            self.kare_sayaci += 1
            
            if self.kare_sayaci % self.ANALIZ_SIKLIGI == 0:
                su_an = time.time()
                kucuk = cv2.resize(kare_boyutlandirilmis, (320, 240))
                sonuclar = self.model.track(kucuk, persist=True, verbose=False, classes=[0], conf=0.3, imgsz=320, tracker="bytetrack.yaml")
                
                olcek_x = EKRAN_GENISLIK / 320
                olcek_y = EKRAN_YUKSEKLIK / 240
                
                self.onbellek_kutular = []
                mevcut_idler = set()
                alarm_tetiklendi = False
                
                if sonuclar[0].boxes.id is not None:
                    kutular = sonuclar[0].boxes.xyxy.cpu().tolist()
                    takip_idleri = sonuclar[0].boxes.id.int().cpu().tolist()
                    
                    for kutu, takip_id in zip(kutular, takip_idleri):
                        x1, y1, x2, y2 = kutu
                        x1, x2 = int(x1*olcek_x), int(x2*olcek_x)
                        y1, y2 = int(y1*olcek_y), int(y2*olcek_y)
                        
                        cx, cy = int((x1+x2)/2), int((y1+y2)/2)
                        kisi_boyu = abs(y2 - y1) 
                        mevcut_idler.add(takip_id)
                        
                        if takip_id not in self.takip_edilen_nesneler:
                            self.takip_edilen_nesneler[takip_id] = {
                                "gecmis": deque(maxlen=5), "durum": "Analiz...", "uyarildi": False
                            }
                        
                        nesne = self.takip_edilen_nesneler[takip_id]
                        nesne["gecmis"].append((cx, cy, su_an, kisi_boyu))
                        
                        if self.alarm_aktif and len(nesne["gecmis"]) >= 3:
                            onceki_x, onceki_y, onceki_zaman, onceki_boy = nesne["gecmis"][0]
                            simdiki_x, simdiki_y, simdiki_zaman, simdiki_boy = nesne["gecmis"][-1]
                            
                            mesafe = math.sqrt((simdiki_x - onceki_x)**2 + (simdiki_y - onceki_y)**2)
                            zaman_farki = simdiki_zaman - onceki_zaman
                            
                            if zaman_farki > 0.1:
                                piksel_hizi = mesafe / zaman_farki
                                ortalama_boy = (onceki_boy + simdiki_boy) / 2
                                bagil_hiz = piksel_hizi / ortalama_boy
                                
                                dinamik_esik = self.BAGIL_HIZ_ESIGI
                                if ortalama_boy < 50: dinamik_esik += 0.5 

                                if bagil_hiz > dinamik_esik:
                                    hiz_degeri = int(bagil_hiz * 10) 
                                    nesne["durum"] = f"KOSUYOR ({hiz_degeri})"
                                    alarm_tetiklendi = True
                                    
                                    if not nesne["uyarildi"]: 
                                        self.anomali_sinyali.emit(f"🏃 KOSMA (ID: {takip_id}) - Hiz: {hiz_degeri}")
                                        nesne["uyarildi"] = True
                                    
                                    self.onbellek_kutular.append(((x1,y1,x2,y2), takip_id, "KOSUYOR", (0, 0, 255)))
                                else:
                                    nesne["durum"] = "Yuruyor"
                                    nesne["uyarildi"] = False 
                                    self.onbellek_kutular.append(((x1,y1,x2,y2), takip_id, "Yuruyor", (0, 255, 0)))
                        else:
                            self.onbellek_kutular.append(((x1,y1,x2,y2), takip_id, "Analiz...", (255, 255, 0)))

                    for tid in list(self.takip_edilen_nesneler.keys()):
                        if tid not in mevcut_idler: del self.takip_edilen_nesneler[tid]
                    self.kamera_durum_sinyali.emit(self.kamera_indeks, alarm_tetiklendi)
                else:
                    self.onbellek_kutular = []

            if self.alarm_aktif:
                 for (kutu, takip_id, durum, renk) in self.onbellek_kutular:
                        x1, y1, x2, y2 = kutu
                        cv2.rectangle(cikti_karesi, (x1, y1), (x2, y2), renk, 2)
                        cv2.putText(cikti_karesi, durum, (x1, y1-5), 0, 0.6, renk, 2)

            rgb = cv2.cvtColor(cikti_karesi, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            qimg = QImage(rgb.data, w, h, ch*w, QImage.Format_RGB888).copy()
            self.goruntu_sinyali.emit(qimg, self.kamera_indeks)
            
            bekleme = kare_suresi - (time.time() - baslangic)
            if bekleme > 0: time.sleep(bekleme)

        kamera.release()

    def stop(self): self._calisiyor = False; self.wait(3000)

# =========================================================================
# THREAD 3: BEKLEME / DOLASMA ALGISI (KAMERA 3)
# =========================================================================
class BeklemeAnalizThread(QThread):
    goruntu_sinyali = pyqtSignal(QImage, int)
    anomali_sinyali = pyqtSignal(str)
    kamera_durum_sinyali = pyqtSignal(int, bool)

    def __init__(self, kamera_indeks, kaynak):
        super().__init__()
        self._calisiyor = True
        self.kamera_indeks = kamera_indeks
        
        self.video_dosyasi_mi = isinstance(kaynak, str)
        if self.video_dosyasi_mi:
            olasi_yol = os.path.join(ANA_DIZIN, "Videolar", str(kaynak))
            self.gercek_kaynak = olasi_yol if os.path.exists(olasi_yol) else kaynak
        else:
            self.gercek_kaynak = kaynak
            
        self.cihaz = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = YOLO(MODEL_VUCUT).to(self.cihaz)
        self.takip_edilen_kisiler = {} 
        self.kayip_kisi_hafizasi = {} 
        self.onbellek_kutular = [] 
        self.kare_sayaci = 0
        self.ANALIZ_SIKLIGI = 7
        self.BEKLEME_SURE_ESIGI = 15.0 # Saniye

    def run(self):
        kamera = cv2.VideoCapture(self.gercek_kaynak)
        fps = kamera.get(cv2.CAP_PROP_FPS)
        if fps is None or fps == 0 or fps > 120: fps = 30
        kare_suresi = 1.0 / fps
        
        while self._calisiyor:
            baslangic = time.time() 
            basarili, kare = kamera.read()
            if not basarili:
                if self.video_dosyasi_mi: kamera.set(cv2.CAP_PROP_POS_FRAMES, 0); continue
                else: break
            
            kare_boyutlandirilmis = cv2.resize(kare, (EKRAN_GENISLIK, EKRAN_YUKSEKLIK))
            cikti_karesi = kare_boyutlandirilmis.copy()
            self.kare_sayaci += 1
            
            if self.kare_sayaci % self.ANALIZ_SIKLIGI == 0:
                su_an = time.time()
                kucuk = cv2.resize(kare_boyutlandirilmis, (320, 240))
                sonuclar = self.model.track(kucuk, persist=True, verbose=False, classes=[0], conf=0.35, imgsz=320, tracker="bytetrack.yaml")
                
                olcek_x = EKRAN_GENISLIK / 320
                olcek_y = EKRAN_YUKSEKLIK / 240
                mevcut_kare_idleri = set()
                
                if sonuclar[0].boxes.id is not None:
                    kutular = sonuclar[0].boxes.xyxy.cpu().tolist()
                    takip_idleri = sonuclar[0].boxes.id.int().cpu().tolist()
                    
                    for kutu, takip_id in zip(kutular, takip_idleri):
                        x1, y1, x2, y2 = kutu
                        x1, x2 = int(x1*olcek_x), int(x2*olcek_x)
                        y1, y2 = int(y1*olcek_y), int(y2*olcek_y)
                        cx, cy = (x1+x2)//2, (y1+y2)//2
                        
                        final_id = takip_id
                        
                        # ID Kurtarma (Kayıp hafızasından)
                        if final_id not in self.takip_edilen_kisiler:
                            en_iyi_eslesme = None
                            min_mesafe = 120.0 
                            for kayip_id, veri in self.kayip_kisi_hafizasi.items():
                                if su_an - veri["son_gorulme"] < 1.0:
                                    lcx, lcy = veri["merkez"]
                                    mesafe = math.sqrt((cx-lcx)**2 + (cy-lcy)**2)
                                    if mesafe < min_mesafe:
                                        min_mesafe = mesafe
                                        en_iyi_eslesme = kayip_id
                            if en_iyi_eslesme is not None:
                                final_id = en_iyi_eslesme
                                self.takip_edilen_kisiler[final_id] = self.kayip_kisi_hafizasi[en_iyi_eslesme]
                                del self.kayip_kisi_hafizasi[en_iyi_eslesme]
                        
                        mevcut_kare_idleri.add(final_id)
                        
                        if final_id not in self.takip_edilen_kisiler:
                            self.takip_edilen_kisiler[final_id] = {
                                "giris_zamani": su_an, "son_gorulme": su_an,
                                "uyarildi": False, "kutu": (x1, y1, x2, y2), "merkez": (cx, cy)
                            }
                        else:
                            self.takip_edilen_kisiler[final_id]["son_gorulme"] = su_an
                            self.takip_edilen_kisiler[final_id]["kutu"] = (x1, y1, x2, y2)
                            self.takip_edilen_kisiler[final_id]["merkez"] = (cx, cy)

                alarm_aktif = False
                silinecek_idler = []
                
                for tid, veri in self.takip_edilen_kisiler.items():
                    if tid not in mevcut_kare_idleri:
                        self.kayip_kisi_hafizasi[tid] = veri
                        silinecek_idler.append(tid)
                    else:
                        gecen_sure = su_an - veri["giris_zamani"]
                        if gecen_sure > self.BEKLEME_SURE_ESIGI: 
                            alarm_aktif = True
                            if not veri["uyarildi"]:
                                self.anomali_sinyali.emit(f"⏳ UZUN SURELI BEKLEME (ID: {tid})")
                                veri["uyarildi"] = True

                for tid in silinecek_idler: del self.takip_edilen_kisiler[tid]
                
                suresi_dolanlar = [k for k,v in self.kayip_kisi_hafizasi.items() if (su_an - v["son_gorulme"] > 1.5)]
                for k in suresi_dolanlar: del self.kayip_kisi_hafizasi[k]

                self.onbellek_kutular = [] 
                cizilecekler = {**self.takip_edilen_kisiler, **self.kayip_kisi_hafizasi}
                
                for tid, veri in cizilecekler.items():
                    if (su_an - veri["son_gorulme"]) < 0.5:
                        gecen = su_an - veri["giris_zamani"]
                        renk = (0, 255, 255) 
                        metin = f"{int(gecen)}s"
                        
                        if gecen > self.BEKLEME_SURE_ESIGI:
                            renk = (0, 0, 255)
                            metin = f"BEKLEME! ({int(gecen)}s)"
                        
                        self.onbellek_kutular.append((veri["kutu"], metin, renk))
                
                self.kamera_durum_sinyali.emit(self.kamera_indeks, alarm_aktif)

            for (kutu, metin, renk) in self.onbellek_kutular:
                x1, y1, x2, y2 = kutu
                cv2.rectangle(cikti_karesi, (x1, y1), (x2, y2), renk, 2)
                cv2.putText(cikti_karesi, metin, (x1, y1-10), 0, 0.6, renk, 2)

            rgb = cv2.cvtColor(cikti_karesi, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            qimg = QImage(rgb.data, w, h, ch*w, QImage.Format_RGB888).copy()
            self.goruntu_sinyali.emit(qimg, self.kamera_indeks)
            
            bekleme = kare_suresi - (time.time() - baslangic)
            if bekleme > 0: time.sleep(bekleme)
            
        kamera.release()

    def stop(self): self._calisiyor = False; self.wait(3000)


# =========================================================================
# THREAD 4: TOPLU INSAN / YOGUNLUK ALGISI (KAMERA 4)
# =========================================================================
class KalabalikAnalizThread(QThread):
    goruntu_sinyali = pyqtSignal(QImage, int)
    anomali_sinyali = pyqtSignal(str)
    kamera_durum_sinyali = pyqtSignal(int, bool)

    def __init__(self, kamera_indeks, kaynak):
        super().__init__()
        self._calisiyor = True
        self.kamera_indeks = kamera_indeks
        
        self.video_dosyasi_mi = isinstance(kaynak, str)
        if self.video_dosyasi_mi:
            olasi_yol = os.path.join(ANA_DIZIN, "Videolar", str(kaynak))
            self.gercek_kaynak = olasi_yol if os.path.exists(olasi_yol) else kaynak
        else:
            self.gercek_kaynak = kaynak
            
        self.cihaz = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = YOLO(MODEL_VUCUT).to(self.cihaz)
        self.onbellek_kutular = []
        self.kare_sayaci = 0
        self.KALABALIK_ESIGI = 8
        self.KRITIK_YOGUNLUK_SEVIYESI = 15
        self.MIN_YAKINLIK_PIKSEL = 100
        self.ANALIZ_SIKLIGI = 3  # Her 3. kare analiz edilir (CPU optimizasyonu)
        self.kritik_uyari_verildi = False
        self.kalabalik_uyari_verildi = False
        # Titremeyi önlemek için son analiz değerlerini önbellekte tut
        self.onbellek_max_kume = 0
        self.onbellek_toplam_kisi = 0
        self.onbellek_alarm = False

    def run(self):
        kamera = cv2.VideoCapture(self.gercek_kaynak)
        fps = kamera.get(cv2.CAP_PROP_FPS)
        if fps is None or fps == 0 or fps > 120: fps = 30
        kare_suresi = 1.0 / fps
        
        while self._calisiyor:
            baslangic = time.time()
            basarili, kare = kamera.read()
            if not basarili:
                if self.video_dosyasi_mi: kamera.set(cv2.CAP_PROP_POS_FRAMES, 0); continue
                else: break
            
            kare_boyutlandirilmis = cv2.resize(kare, (EKRAN_GENISLIK, EKRAN_YUKSEKLIK))
            cikti_karesi = kare_boyutlandirilmis.copy()
            self.kare_sayaci += 1
            # Her karede önbellekteki son değeri kullan (titreme önleme)
            alarm_tetiklendi = self.onbellek_alarm
            
            if self.kare_sayaci % self.ANALIZ_SIKLIGI == 0:
                kucuk = cv2.resize(kare_boyutlandirilmis, (480, 360))
                sonuclar = self.model(kucuk, verbose=False, classes=[0], conf=0.25, imgsz=480)
                
                olcek_x = EKRAN_GENISLIK / 480
                olcek_y = EKRAN_YUKSEKLIK / 360
                
                self.onbellek_kutular = []
                max_kume_boyutu = 0
                mevcut_kisi_sayisi = 0
                
                if len(sonuclar[0].boxes) > 0:
                    kutular = sonuclar[0].boxes.xyxy.cpu().tolist()
                    kisi_verileri = []
                    toplam_boy = 0 
                    
                    for kutu in kutular:
                        x1, y1, x2, y2 = kutu
                        x1, x2 = int(x1*olcek_x), int(x2*olcek_x)
                        y1, y2 = int(y1*olcek_y), int(y2*olcek_y)
                        cx, cy = (x1+x2)//2, (y1+y2)//2
                        kisi_boyu = abs(y2 - y1)
                        toplam_boy += kisi_boyu 
                        
                        kisi_verileri.append({
                            "kutu": (x1, y1, x2, y2), "merkez": (cx, cy), "boy": kisi_boyu
                        })

                    ortalama_boy = toplam_boy / len(kisi_verileri) if len(kisi_verileri) > 0 else 100
                    mevcut_kisi_sayisi = len(kisi_verileri)

                    for i in range(len(kisi_verileri)):
                        p1 = kisi_verileri[i]
                        p1_kume_boyutu = 1 
                        boy_orani = p1["boy"] / ortalama_boy
                        dinamik_yakinlik_esigi = self.MIN_YAKINLIK_PIKSEL * boy_orani
                        
                        for j in range(i + 1, len(kisi_verileri)):
                            p2 = kisi_verileri[j]
                            mesafe = math.sqrt((p1["merkez"][0] - p2["merkez"][0])**2 + (p1["merkez"][1] - p2["merkez"][1])**2)
                            if mesafe < dinamik_yakinlik_esigi: p1_kume_boyutu += 1

                        max_kume_boyutu = max(max_kume_boyutu, p1_kume_boyutu)
                    
                    if max_kume_boyutu >= self.KALABALIK_ESIGI:
                        alarm_tetiklendi = True
                        renk = (0, 165, 255) 
                        if max_kume_boyutu >= self.KRITIK_YOGUNLUK_SEVIYESI:
                            renk = (0, 0, 255) 
                            if not self.kritik_uyari_verildi:
                                self.anomali_sinyali.emit(f"🔴 KRITIK KUME! ({max_kume_boyutu} Kisi)")
                                self.kritik_uyari_verildi = True
                                self.kalabalik_uyari_verildi = False 
                        else:
                            self.kritik_uyari_verildi = False
                            if not self.kalabalik_uyari_verildi:
                                self.anomali_sinyali.emit(f"⚠️ YAKIN KUME IHLALI ({max_kume_boyutu} Kisi)")
                                self.kalabalik_uyari_verildi = True
                    else:
                        renk = (0, 255, 0) 
                        self.kritik_uyari_verildi = False
                        self.kalabalik_uyari_verildi = False

                    for veri in kisi_verileri:
                        self.onbellek_kutular.append((veri["kutu"], renk))

                # Analiz sonuçlarını önbelleğe al (titreme önleme)
                self.onbellek_max_kume = max_kume_boyutu
                self.onbellek_toplam_kisi = mevcut_kisi_sayisi
                self.onbellek_alarm = alarm_tetiklendi

            # --- GELİŞMİŞ OVERLAY (önbellekten, titremeye karşı) ---
            mk = self.onbellek_max_kume
            tk = self.onbellek_toplam_kisi

            if mk >= self.KRITIK_YOGUNLUK_SEVIYESI:
                overlay_renk = (0, 0, 200)      # Koyu kırmızı
                yazi_rengi = (255, 255, 255)
                durum_yazisi = "KRITIK"
            elif mk >= self.KALABALIK_ESIGI:
                overlay_renk = (0, 100, 220)    # Turuncu-kahve
                yazi_rengi = (255, 255, 255)
                durum_yazisi = "UYARI"
            else:
                overlay_renk = (30, 30, 30)     # Nötr koyu
                yazi_rengi = (200, 200, 200)
                durum_yazisi = "NORMAL"

            # Yarı saydam arka plan kutusu
            overlay = cikti_karesi.copy()
            cv2.rectangle(overlay, (0, 0), (260, 58), overlay_renk, -1)
            cv2.addWeighted(overlay, 0.62, cikti_karesi, 0.38, 0, cikti_karesi)

            # İnce üst çizgi aksanı
            cv2.rectangle(cikti_karesi, (0, 0), (260, 3), yazi_rengi, -1)

            # Durum etiketi (küçük, üstte)
            cv2.putText(cikti_karesi, durum_yazisi, (10, 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, yazi_rengi, 1, cv2.LINE_AA)

            # Ana metinler
            cv2.putText(cikti_karesi, f"MAX KUME : {mk}",
                        (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.62, yazi_rengi, 1, cv2.LINE_AA)
            cv2.putText(cikti_karesi, f"TOPLAM   : {tk}",
                        (10, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.62, yazi_rengi, 1, cv2.LINE_AA)

            for (kutu, renk) in self.onbellek_kutular:
                x1, y1, x2, y2 = kutu
                cv2.rectangle(cikti_karesi, (x1, y1), (x2, y2), renk, 2)
            
            self.kamera_durum_sinyali.emit(self.kamera_indeks, alarm_tetiklendi)
            rgb = cv2.cvtColor(cikti_karesi, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            qimg = QImage(rgb.data, w, h, ch*w, QImage.Format_RGB888).copy()
            self.goruntu_sinyali.emit(qimg, self.kamera_indeks)
            
            bekleme = kare_suresi - (time.time() - baslangic)
            if bekleme > 0: time.sleep(bekleme)
            
        kamera.release()

    def stop(self): self._calisiyor = False; self.wait(3000)


# =========================================================================
# THREAD 5: ŞÜPHELİ PAKET - HASSAS HAREKET TAKİBİ MODU
# =========================================================================
class SupheliPaketThread(QThread):
    goruntu_sinyali = pyqtSignal(QImage, int)
    anomali_sinyali = pyqtSignal(str)
    kamera_durum_sinyali = pyqtSignal(int, bool)

    def __init__(self, kamera_indeks, kaynak):
        super().__init__()
        self._calisiyor = True
        self.kamera_indeks = kamera_indeks
        
        self.video_dosyasi_mi = isinstance(kaynak, str)
        if self.video_dosyasi_mi:
            olasi_yol = os.path.join(ANA_DIZIN, "Videolar", str(kaynak))
            self.gercek_kaynak = olasi_yol if os.path.exists(olasi_yol) else kaynak
        else:
            self.gercek_kaynak = kaynak

        self.cihaz = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Algılama başarısı için Medium model (yolov8m)
        self.model_adi = 'yolov8m.pt' 
        self.model = YOLO(os.path.join(ANA_DIZIN, self.model_adi)).to(self.cihaz)
        
        # --- AYARLAR ---
        self.TERK_EDILME_LIMITI = 5.0  # 5 saniye hareketsiz kalırsa alarm
        self.HAREKET_ESIGI = 5.0       # Hareket toleransı (düşük tutarak hassasiyet arttı)
        self.ANALIZ_SIKLIGI = 3 
        
        self.nesne_gecmisi = {} 
        self.onbellek_sonuclar = [] 
        self.HEDEF_SINIFLAR = [24, 26, 28] # Çantalar

    def run(self):
        kamera = cv2.VideoCapture(self.gercek_kaynak)
        fps = kamera.get(cv2.CAP_PROP_FPS)
        if not fps or fps > 120: fps = 30
        kare_suresi = 1.0 / fps
        kare_sayaci = 0

        while self._calisiyor:
            baslangic = time.time()
            basarili, kare = kamera.read()
            if not basarili:
                if self.video_dosyasi_mi: 
                    kamera.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    self.nesne_gecmisi.clear()
                    continue
                else: break

            orig_h, orig_w = kare.shape[:2]
            kare_ekran = cv2.resize(kare, (EKRAN_GENISLIK, EKRAN_YUKSEKLIK))
            cikti_karesi = kare_ekran.copy()
            
            olcek_x = EKRAN_GENISLIK / orig_w
            olcek_y = EKRAN_YUKSEKLIK / orig_h

            kare_sayaci += 1
            
            if kare_sayaci % self.ANALIZ_SIKLIGI == 0:
                su_an = time.time()
                
                sonuclar = self.model.track(
                    kare, # Orijinal boyut
                    persist=True, 
                    verbose=False, 
                    classes=self.HEDEF_SINIFLAR, 
                    conf=0.15,      
                    imgsz=640,      
                    tracker="bytetrack.yaml"
                )
                
                self.onbellek_sonuclar = [] 
                mevcut_idler = set()
                alarm_aktif = False

                if sonuclar[0].boxes.id is not None:
                    kutular = sonuclar[0].boxes.xyxy.cpu().tolist()
                    idler = sonuclar[0].boxes.id.int().cpu().tolist()
                    
                    for kutu, takip_id in zip(kutular, idler):
                        ox1, oy1, ox2, oy2 = map(int, kutu)
                        cx, cy = (ox1+ox2)//2, (oy1+oy2)//2
                        mevcut_idler.add(takip_id)
                        
                        dx1, dx2 = int(ox1 * olcek_x), int(ox2 * olcek_x)
                        dy1, dy2 = int(oy1 * olcek_y), int(oy2 * olcek_y)
                        
                        bilgi_metni = ""      
                        
                        if takip_id not in self.nesne_gecmisi:
                            self.nesne_gecmisi[takip_id] = {
                                'ilk_durma_zamani': None, 
                                'son_konum': (cx, cy), 
                                'son_gorulme': su_an, 
                                'uyarildi': False
                            }

                        veri = self.nesne_gecmisi[takip_id]
                        onceki_cx, onceki_cy = veri['son_konum']
                        veri['son_gorulme'] = su_an 
                        
                        # Hareket mesafesi
                        mesafe = math.hypot(cx - onceki_cx, cy - onceki_cy)
                        
                        # --- HAREKET MANTIGI ---
                        if mesafe < self.HAREKET_ESIGI:
                            if veri['ilk_durma_zamani'] is None: 
                                veri['ilk_durma_zamani'] = su_an 
                            
                            bekleme_suresi = su_an - veri['ilk_durma_zamani']
                            
                            if bekleme_suresi > self.TERK_EDILME_LIMITI:
                                renk = (0, 0, 255) # KIRMIZI
                                bilgi_metni = f"SUPHELI! ({int(bekleme_suresi)}s)"
                                alarm_aktif = True
                                if not veri['uyarildi']:
                                    self.anomali_sinyali.emit(f"💣 SUPHELI PAKET (ID: {takip_id})")
                                    veri['uyarildi'] = True
                            else:
                                renk = (0, 255, 255) # SARI
                                bilgi_metni = f"Hareketsiz {int(bekleme_suresi)}s"
                        else:
                            # Hareketli (Mavi)
                            veri['ilk_durma_zamani'] = None 
                            veri['uyarildi'] = False
                            renk = (255, 0, 0) # MAVİ
                            bilgi_metni = "TASINIYOR"

                        veri['son_konum'] = (cx, cy)
                        self.onbellek_sonuclar.append((dx1, dy1, dx2, dy2, renk, bilgi_metni))

                silinecekler = []
                for tid, veri in self.nesne_gecmisi.items():
                    if tid not in mevcut_idler:
                        if su_an - veri['son_gorulme'] > 3.0: silinecekler.append(tid)
                for k in silinecekler: del self.nesne_gecmisi[k]
                
                self.kamera_durum_sinyali.emit(self.kamera_indeks, alarm_aktif)

            for (x1, y1, x2, y2, renk, metin) in self.onbellek_sonuclar:
                cv2.rectangle(cikti_karesi, (x1, y1), (x2, y2), renk, 2)
                if metin: 
                    (w, h), _ = cv2.getTextSize(metin, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(cikti_karesi, (x1, y1 - 20), (x1 + w, y1), renk, -1)
                    metin_rengi = (255,255,255) if renk == (255,0,0) else (0,0,0)
                    cv2.putText(cikti_karesi, metin, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, metin_rengi, 2)

            rgb = cv2.cvtColor(cikti_karesi, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            qimg = QImage(rgb.data, w, h, ch*w, QImage.Format_RGB888).copy()
            self.goruntu_sinyali.emit(qimg, self.kamera_indeks)
            
            bekleme = kare_suresi - (time.time() - baslangic)
            if bekleme > 0: time.sleep(bekleme)
        
        kamera.release()

    def stop(self): self._calisiyor = False; self.wait(3000)

# =========================================================================
# THREAD 6: HATALI PARK / DURAN ARAC TESPITI (KAMERA 6)
# =========================================================================
class HataliParkThread(QThread):
    goruntu_sinyali = pyqtSignal(QImage, int)
    anomali_sinyali = pyqtSignal(str)
    kamera_durum_sinyali = pyqtSignal(int, bool)

    def __init__(self, kamera_indeks, kaynak):
        super().__init__()
        self._calisiyor = True
        self.kamera_indeks = kamera_indeks
        
        self.video_dosyasi_mi = isinstance(kaynak, str)
        if self.video_dosyasi_mi:
            olasi_yol = os.path.join(ANA_DIZIN, "Videolar", str(kaynak))
            self.gercek_kaynak = olasi_yol if os.path.exists(olasi_yol) else kaynak
        else:
            self.gercek_kaynak = kaynak

        self.cihaz = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = YOLO(MODEL_VUCUT).to(self.cihaz)
        self.SURE_LIMITI = 10.0 
        self.HAREKET_ESIGI = 15.0 
        self.ANALIZ_SIKLIGI = 3
        self.arac_gecmisi = {} 
        self.onbellek_sonuclar = [] 
        self.HEDEF_SINIFLAR = [2, 3, 5, 7] 

    def run(self):
        kamera = cv2.VideoCapture(self.gercek_kaynak)
        fps = kamera.get(cv2.CAP_PROP_FPS)
        if not fps or fps > 120: fps = 30
        kare_suresi = 1.0 / fps
        kare_sayaci = 0

        while self._calisiyor:
            baslangic = time.time()
            basarili, kare = kamera.read()
            if not basarili:
                if self.video_dosyasi_mi: 
                    kamera.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    self.arac_gecmisi.clear()
                    continue
                else: break

            kare_boyutlandirilmis = cv2.resize(kare, (EKRAN_GENISLIK, EKRAN_YUKSEKLIK))
            cikti_karesi = kare_boyutlandirilmis.copy()
            kare_sayaci += 1
            
            if kare_sayaci % self.ANALIZ_SIKLIGI == 0:
                su_an = time.time()
                sonuclar = self.model.track(kare_boyutlandirilmis, persist=True, verbose=False, classes=self.HEDEF_SINIFLAR, conf=0.25, imgsz=320, tracker="bytetrack.yaml")
                
                self.onbellek_sonuclar = [] 
                mevcut_idler = set()
                alarm_aktif = False

                if sonuclar[0].boxes.id is not None:
                    kutular = sonuclar[0].boxes.xyxy.cpu().tolist()
                    idler = sonuclar[0].boxes.id.int().cpu().tolist()
                    
                    for kutu, takip_id in zip(kutular, idler):
                        x1, y1, x2, y2 = map(int, kutu)
                        cx, cy = (x1+x2)//2, (y1+y2)//2
                        mevcut_idler.add(takip_id)
                        renk = (0, 255, 0) 
                        bilgi_metni = ""      
                        
                        if takip_id not in self.arac_gecmisi:
                            self.arac_gecmisi[takip_id] = {
                                'giris_zamani': su_an, 'ilk_durma_zamani': None, 'son_konum': (cx, cy), 'son_gorulme': su_an, 'uyarildi': False
                            }

                        veri = self.arac_gecmisi[takip_id]
                        onceki_cx, onceki_cy = veri['son_konum']
                        veri['son_gorulme'] = su_an
                        mesafe = math.hypot(cx - onceki_cx, cy - onceki_cy)
                        
                        if mesafe < self.HAREKET_ESIGI:
                            if veri['ilk_durma_zamani'] is None: veri['ilk_durma_zamani'] = su_an 
                            durma_suresi = su_an - veri['ilk_durma_zamani']
                            if durma_suresi > self.SURE_LIMITI:
                                renk = (0, 0, 255) 
                                bilgi_metni = f"BEKLIYOR ({int(durma_suresi)}s)"
                                alarm_aktif = True
                                if not veri['uyarildi']:
                                    self.anomali_sinyali.emit(f"⛔ UZUN SURELI BEKLEME (ID: {takip_id})")
                                    veri['uyarildi'] = True
                            else:
                                renk = (0, 255, 0)
                                veri['uyarildi'] = False 
                        else:
                            veri['ilk_durma_zamani'] = None 
                            veri['uyarildi'] = False
                            renk = (0, 255, 0)
                            if (su_an - veri['giris_zamani']) > self.SURE_LIMITI: bilgi_metni = "YAVAS"

                        veri['son_konum'] = (cx, cy)
                        self.onbellek_sonuclar.append((x1, y1, x2, y2, renk, bilgi_metni))

                for tid in list(self.arac_gecmisi.keys()):
                    if tid not in mevcut_idler: del self.arac_gecmisi[tid]
                
                self.kamera_durum_sinyali.emit(self.kamera_indeks, alarm_aktif)

            for (x1, y1, x2, y2, renk, metin) in self.onbellek_sonuclar:
                cv2.rectangle(cikti_karesi, (x1, y1), (x2, y2), renk, 2)
                if metin: cv2.putText(cikti_karesi, metin, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, renk, 2)

            rgb = cv2.cvtColor(cikti_karesi, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            qimg = QImage(rgb.data, w, h, ch*w, QImage.Format_RGB888).copy()
            self.goruntu_sinyali.emit(qimg, self.kamera_indeks)
            
            bekleme = kare_suresi - (time.time() - baslangic)
            if bekleme > 0: time.sleep(bekleme)
        
        kamera.release()

    def stop(self): self._calisiyor = False; self.wait(3000)