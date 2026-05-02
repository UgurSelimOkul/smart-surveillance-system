import os
import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
                             QGridLayout, QLabel, QPushButton, QListWidget, QFrame,
                             QDialog, QSizePolicy)
from PyQt5.QtCore import Qt, QTime
from PyQt5.QtGui import QFont, QGuiApplication, QPixmap

# --- ORTAM AYARLARI ---
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

if hasattr(os, 'add_dll_directory'):
    try:
        os.add_dll_directory(r"C:\Windows\System32")
    except:
        pass

# --- BASLANGIC KONTROLU ---
from startup_check import baslangic_kontrolu_yap
_SISTEM_DURUMU = baslangic_kontrolu_yap()

# PyTorch Kontrolü
try:
    import torch
    print(f"PyTorch Önden Yüklendi: {torch.__version__}")
except ImportError as e:
    print(f"PyTorch yükleme hatası: {e}")
except OSError as e:
    print(f"DLL Hatası Yakalandı: {e}")

# video_processor.py dosyasındaki sınıfları çağırıyoruz.
try:
    from video_processor import (
        YuzTanimaThread, 
        HareketAnalizThread, 
        BeklemeAnalizThread, 
        KalabalikAnalizThread, 
        SupheliPaketThread, 
        HataliParkThread, 
        KAMERA_KAYNAKLARI
    )
except ImportError as e:
    print("HATA: video_processor.py dosyasi bulunamadi veya içinde hata var!")
    print(f"Detay: {e}")
    sys.exit(1)

# --- ANA UYGULAMA SINIFI ---
class AkilliGozetlemeApp(QMainWindow):
    
    # Kameraların konfigürasyonu
    KAMERA_AYARLARI = [
        {"kamera_no": 1, "tip": "YUZ",      "baslik": "KAMERA 1 | GIRIS (YUZ KONTROL)"},
        {"kamera_no": 2, "tip": "KOSMA",    "baslik": "KAMERA 2 | HIZ IHLALI"},
        {"kamera_no": 3, "tip": "BEKLEME",  "baslik": "KAMERA 3 | BEKLEME IHLALI"},
        {"kamera_no": 4, "tip": "KALABALIK","baslik": "KAMERA 4 | TOPLU INSAN YOGUNLUGU"},
        {"kamera_no": 5, "tip": "PAKET",    "baslik": "KAMERA 5 | SUPHELI CANTA KONTROL"},
        {"kamera_no": 6, "tip": "PARK",     "baslik": "KAMERA 6 | HATALI PARK KONTROL"}
    ]

    KAMERA_SAYISI = len(KAMERA_AYARLARI)

    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("KRITIK GOZETLEME SISTEMI | UĞUR SELİM OKUL")
        self.setGeometry(QGuiApplication.primaryScreen().geometry())
        self.showMaximized() 
        
        # Stil dosyasını yükle
        try:
            ana_klasor = os.path.dirname(os.path.abspath(__file__)) 
        except NameError:
            ana_klasor = os.path.abspath('.') 
            
        self.stilleri_yukle(os.path.join(ana_klasor, "style.qss")) 

        merkezi_pencere = QWidget()
        self.setCentralWidget(merkezi_pencere)
        
        self.ana_duzen = QHBoxLayout(merkezi_pencere)
        self.ana_duzen.setContentsMargins(0, 0, 0, 0)

        self.yan_menuyu_olustur()
        self.icerik_alanini_olustur() 

        self.ana_duzen.addWidget(self.yan_menu)
        self.ana_duzen.addWidget(self.icerik_alani)
        
        # Aktif thread listesi
        self.video_islemcileri = []
        
        # Başlangıçta tüm kameraları aktif et
        for ayar in self.KAMERA_AYARLARI:
            self.kamera_baslat(ayar["kamera_no"])

    def stilleri_yukle(self, dosya_yolu):
        try:
            with open(dosya_yolu, "r", encoding="utf-8") as dosya: 
                self.setStyleSheet(dosya.read())
        except FileNotFoundError:
            self.setStyleSheet("QMainWindow { background-color: #050505; color: #FFFFFF; }")

    def yan_menuyu_olustur(self):
        self.yan_menu = QFrame()
        self.yan_menu.setFixedWidth(320)
        self.yan_menu.setObjectName("Sidebar")
        
        menu_duzeni = QVBoxLayout(self.yan_menu)
        menu_duzeni.setAlignment(Qt.AlignTop) # Her şeyi yukarı yasla

        logo_etiketi = QLabel("KOMUTA GUVENLIK")
        logo_etiketi.setFont(QFont("Arial", 18, QFont.ExtraBold))
        logo_etiketi.setAlignment(Qt.AlignCenter)
        logo_etiketi.setObjectName("MenuTitle")
        
        menu_duzeni.addWidget(logo_etiketi)

        bosluk_etiketi = QLabel("KAMERA KONTROL")
        bosluk_etiketi.setStyleSheet("color: #666; font-size: 10pt; font-weight: bold; margin-top: 10px;")
        bosluk_etiketi.setAlignment(Qt.AlignCenter)
        menu_duzeni.addWidget(bosluk_etiketi)

        for i in range(1, 7): 
            btn = QPushButton(f"✅ KAMERA {i} AKTİF")
            btn.clicked.connect(lambda checked, numara=i: self.kamera_ac_kapa(numara)) #Butona basıldığı zaman buton numarası ile birlikte kamera ac kapaya git.
            menu_duzeni.addWidget(btn)
        
        menu_duzeni.addStretch(1) #Baska Seyler ekleyecek olursan bunun altına ekle.

    def kamera_ac_kapa(self, kamera_no):
        tiklanan_buton = self.sender() #Hangi buton basti.
        
        if ("AKTİF" in tiklanan_buton.text()):
            # KAPATMA
            tiklanan_buton.setText(f"❌ KAMERA {kamera_no} PASİF")
            tiklanan_buton.setStyleSheet("color: #ff5555; border: 1px solid #ff5555;")
            
            self.kamera_durdur(kamera_no)
            
            if kamera_no in self.kamera_gorunumleri:
                etiket = self.kamera_gorunumleri[kamera_no]
                etiket.clear()
                etiket.setText("KAMERA KAPATILDI")
                etiket.setStyleSheet("color: red; font-weight: bold; font-size: 14pt;")
                etiket.setAlignment(Qt.AlignCenter)

            self.uyari_kaydet(f"SİSTEM: Kamera {kamera_no} KAPATILDI.")
            self.kamera_durumunu_yonet(kamera_no, False)
            
        else:
            # AÇMA
            tiklanan_buton.setText(f"✅ KAMERA {kamera_no} AKTİF")
            tiklanan_buton.setStyleSheet("")

            if kamera_no in self.kamera_gorunumleri:
                self.kamera_gorunumleri[kamera_no].setStyleSheet("")
                self.kamera_gorunumleri[kamera_no].clear()
                self.kamera_gorunumleri[kamera_no].setText("SINYAL BEKLENIYOR...")

            # Anomali çerçevesini sıfırla
            self.kamera_durumunu_yonet(kamera_no, False)

            self.kamera_baslat(kamera_no)
            self.uyari_kaydet(f"SİSTEM: Kamera {kamera_no} BAŞLATILDI.")

    def kamera_baslat(self, kamera_no):
        ayar = next((item for item in self.KAMERA_AYARLARI if item["kamera_no"] == kamera_no), None)
        if not ayar: return

        # --- GUVENLIK KONTROL: Model veya video eksikse thread baslatma ---
        if not _SISTEM_DURUMU["modeller_tamam"]:
            # Hiçbir model yoksa hiçbir kamerayı başlatma
            if kamera_no in self.kamera_gorunumleri:
                self.kamera_gorunumleri[kamera_no].setText("MODEL DOSYASI EKSİK\nKamera devre dışı")
                self.kamera_gorunumleri[kamera_no].setStyleSheet(
                    "color: #ff6600; font-weight: bold; font-size: 11pt;")
                self.kamera_gorunumleri[kamera_no].setAlignment(Qt.AlignCenter)
            return

        kaynak = KAMERA_KAYNAKLARI.get(kamera_no, -1)

        # Video dosyası gerektiren kameralar için ek kontrol (kamera_no 2-6)
        if isinstance(kaynak, str):
            import os as _os
            dosya_adi = _os.path.basename(str(kaynak))
            if dosya_adi in _SISTEM_DURUMU["eksik_videolar"]:
                if kamera_no in self.kamera_gorunumleri:
                    self.kamera_gorunumleri[kamera_no].setText(f"VİDEO EKSİK\n{dosya_adi}\nVideolar/ klasörüne koyun")
                    self.kamera_gorunumleri[kamera_no].setStyleSheet(
                        "color: #ff6600; font-weight: bold; font-size: 10pt;")
                    self.kamera_gorunumleri[kamera_no].setAlignment(Qt.AlignCenter)
                return

        if kaynak == -1: return 

        if kamera_no in self.kamera_basliklari:
            self.kamera_basliklari[kamera_no].setText(ayar["baslik"])

        islemci = None
        tip = ayar["tip"]

        # --- SINIF EŞLEŞTİRMESİ ---
        if tip == "YUZ":
            islemci = YuzTanimaThread(kamera_no, kaynak)
        elif tip == "KOSMA":
            islemci = HareketAnalizThread(kamera_no, kaynak)
        elif tip == "BEKLEME":
            islemci = BeklemeAnalizThread(kamera_no, kaynak)
        elif tip == "KALABALIK": 
            islemci = KalabalikAnalizThread(kamera_no, kaynak)
        elif tip == "PAKET":
            islemci = SupheliPaketThread(kamera_no, kaynak)
        elif tip == "PARK":
            islemci = HataliParkThread(kamera_no, kaynak)
        
        if islemci:
            # --- SİNYAL BAĞLAMA ---
            islemci.goruntu_sinyali.connect(self.goruntuyu_guncelle)
            islemci.anomali_sinyali.connect(self.uyari_kaydet)
            islemci.kamera_durum_sinyali.connect(self.kamera_durumunu_yonet)
            
            islemci.start()
            self.video_islemcileri.append(islemci)

    def kamera_durdur(self, kamera_no):
        islemci_silinecek = None
        
        for islemci in self.video_islemcileri:
            if islemci.kamera_indeks == kamera_no:
                islemci_silinecek = islemci
                break
        
        if islemci_silinecek:
            try:
                # --- SİNYALLERİ KOPARIYORUZ ---
                islemci_silinecek.goruntu_sinyali.disconnect()
                islemci_silinecek.anomali_sinyali.disconnect()
                islemci_silinecek.kamera_durum_sinyali.disconnect()
            except Exception:
                pass 

            if islemci_silinecek.isRunning():
                islemci_silinecek.stop() 
            
            if islemci_silinecek in self.video_islemcileri:
                self.video_islemcileri.remove(islemci_silinecek)

    def kamera_kutusu_olustur(self, index):
        kutu = QFrame()
        kutu.setObjectName("CameraContainer")
        kutu.setProperty("anomaly_active", "false")
        
        dikey_duzen = QVBoxLayout(kutu)
        dikey_duzen.setContentsMargins(0, 0, 0, 0)
        dikey_duzen.setSpacing(0) 

        baslik_kutusu = QWidget()
        baslik_duzeni = QHBoxLayout(baslik_kutusu)
        baslik_duzeni.setContentsMargins(5, 5, 5, 5) 
        baslik_duzeni.setSpacing(5)

        baslik_etiketi = QLabel(f"KAMERA {index} | BAGLANIYOR...") 
        baslik_etiketi.setObjectName("CameraHeader")
        baslik_etiketi.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        
        buyutme_butonu = QPushButton("⤢")
        buyutme_butonu.setFixedSize(30, 30)
        buyutme_butonu.setObjectName(f"MaximizeButton_{index}")
        buyutme_butonu.setToolTip("Tam Ekran")
        buyutme_butonu.clicked.connect(lambda checked, idx=index: self.kameray_buyut(idx))
        
        baslik_duzeni.addWidget(baslik_etiketi)
        baslik_duzeni.addStretch(1) 
        baslik_duzeni.addWidget(buyutme_butonu)
        
        baslik_kutusu.setObjectName("CameraHeaderWidget")

        kamera_goruntusu = QLabel("SINYAL BEKLENIYOR...")
        kamera_goruntusu.setObjectName("CameraLabel")
        kamera_goruntusu.setAlignment(Qt.AlignCenter)
        kamera_goruntusu.setMinimumHeight(150)
        kamera_goruntusu.setScaledContents(True)
        
        dikey_duzen.addWidget(baslik_kutusu, 0) 
        dikey_duzen.addWidget(kamera_goruntusu, 1)
        
        return kutu, kamera_goruntusu, baslik_etiketi

    def icerik_alanini_olustur(self):
        self.icerik_alani = QFrame()
        self.icerik_alani.setObjectName("ContentArea")
        
        icerik_duzeni = QVBoxLayout(self.icerik_alani)

        self.kamera_izgarasi_alani = QWidget()
        self.kamera_izgara_duzeni = QGridLayout(self.kamera_izgarasi_alani)
        self.kamera_izgara_duzeni.setSpacing(10) 
        
        self.kamera_izgara_duzeni.setRowStretch(0, 1)
        self.kamera_izgara_duzeni.setRowStretch(1, 1)
        self.kamera_izgara_duzeni.setColumnStretch(0, 1)
        self.kamera_izgara_duzeni.setColumnStretch(1, 1)
        self.kamera_izgara_duzeni.setColumnStretch(2, 1)
        
        self.kamera_gorunumleri = {}
        self.kamera_basliklari = {}
        self.kamera_kutulari = {} 
        
        for i in range(self.KAMERA_SAYISI):
            cam_idx = i + 1 
            kamera_kutusu, kamera_resmi, baslik_yazisi = self.kamera_kutusu_olustur(cam_idx) 
            
            ayar_verisi = next((oge for oge in self.KAMERA_AYARLARI if oge["kamera_no"] == cam_idx), None)
            
            if ayar_verisi:
                baslik_yazisi.setText(ayar_verisi["baslik"])
            
            self.kamera_gorunumleri[cam_idx] = kamera_resmi
            self.kamera_basliklari[cam_idx] = baslik_yazisi
            self.kamera_kutulari[cam_idx] = kamera_kutusu 
            
            satir, sutun = i // 3, i % 3
            self.kamera_izgara_duzeni.addWidget(kamera_kutusu, satir, sutun, 1, 1) 

        uyari_basligi = QLabel("KRITIK ALARM VE OLAY BILDIRIMLERI")
        uyari_basligi.setObjectName("WarningTitle")
        
        self.uyari_listesi = QListWidget()
        self.uyari_listesi.setMaximumHeight(150)
        self.uyari_listesi.addItem(QTime.currentTime().toString('[hh:mm:ss] ') + " SISTEM BASLATILDI.")

        icerik_duzeni.addWidget(self.kamera_izgarasi_alani, 6)
        icerik_duzeni.addWidget(uyari_basligi)
        icerik_duzeni.addWidget(self.uyari_listesi, 1) 

    def goruntuyu_guncelle(self, gelen_resim, kamera_no):
        if kamera_no in self.kamera_gorunumleri:
            hedef_gorunum = self.kamera_gorunumleri[kamera_no]
            hedef_gorunum.setPixmap(QPixmap.fromImage(gelen_resim))

    def kameray_buyut(self, kamera_no):
        """Seçili kamerayı ayrı bir tam ekran diyalog penceresinde göster."""
        if kamera_no not in self.kamera_gorunumleri:
            return

        dialog = QDialog(self)
        dialog.setWindowTitle(f"KAMERA {kamera_no} — TAM EKRAN")
        dialog.setWindowFlags(Qt.Window | Qt.WindowCloseButtonHint | Qt.WindowMaximizeButtonHint)
        dialog.resize(960, 720)
        dialog.setStyleSheet("background-color: #050505;")

        duzen = QVBoxLayout(dialog)
        duzen.setContentsMargins(0, 0, 0, 0)

        buyuk_gorunum = QLabel("SINYAL BEKLENIYOR...")
        buyuk_gorunum.setAlignment(Qt.AlignCenter)
        buyuk_gorunum.setStyleSheet("color: #AAAAAA; font-size: 14pt;")
        buyuk_gorunum.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        buyuk_gorunum.setScaledContents(True)
        duzen.addWidget(buyuk_gorunum)

        # Şu anki kare varsa hemen göster
        mevcut_pixmap = self.kamera_gorunumleri[kamera_no].pixmap()
        if mevcut_pixmap:
            buyuk_gorunum.setPixmap(mevcut_pixmap)

        # Sinyal bağlama: diyalog açık olduğu sürece görüntüyü güncelle
        def _guncelle(img, no):
            if no == kamera_no:
                buyuk_gorunum.setPixmap(QPixmap.fromImage(img))

        # İlgili thread sinyalini geçici bağla
        for islemci in self.video_islemcileri:
            if islemci.kamera_indeks == kamera_no:
                islemci.goruntu_sinyali.connect(_guncelle)
                dialog.finished.connect(lambda _, isl=islemci: isl.goruntu_sinyali.disconnect(_guncelle))
                break

        dialog.exec_()
    
    def uyari_kaydet(self, mesaj):
        zaman_damgasi = QTime.currentTime().toString('[hh:mm:ss] ')
        self.uyari_listesi.addItem(zaman_damgasi + mesaj)
        # Bellek taşmasını önlemek için en eski kaydı sil
        if self.uyari_listesi.count() > 500:
            self.uyari_listesi.takeItem(0)
        self.uyari_listesi.scrollToBottom()

    def kamera_durumunu_yonet(self, kamera_no, durum_aktif_mi):
        if kamera_no in self.kamera_kutulari:
            kutu = self.kamera_kutulari[kamera_no]
            mevcut_durum = kutu.property("anomaly_active")
            yeni_durum_yazisi = "true" if durum_aktif_mi else "false"
            
            if mevcut_durum != yeni_durum_yazisi:
                kutu.setProperty("anomaly_active", yeni_durum_yazisi)
                kutu.style().polish(kutu)

    def closeEvent(self, olay):
        for islemci in self.video_islemcileri:
            if islemci.isRunning():
                islemci.stop()
                islemci.wait(3000)  # En fazla 3 saniye bekle
        olay.accept()

if __name__ == "__main__":
    uygulama = QApplication(sys.argv)
    pencere = AkilliGozetlemeApp()
    pencere.show()
    sys.exit(uygulama.exec_())