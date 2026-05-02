"""
startup_check.py — Sistem Başlangıç Kontrol Modülü
Bu dosya main.py tarafından import edilir.
Eksik kütüphane / model / video dosyası varsa kullanıcıya açıklayıcı bir popup gösterir.
Arayüz her zaman açılır, yalnızca ilgili kameralar devre dışı kalır.
"""

import os
import sys

# --- DİZİN TESPİTİ (video_processor.py ile aynı mantık) ---
if getattr(sys, 'frozen', False):
    ANA_DIZIN = os.path.dirname(sys.executable)
else:
    try:
        ANA_DIZIN = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        ANA_DIZIN = os.path.abspath('.')

# ============================================================
#  KONTROL EDİLECEK ŞEYLER
# ============================================================

# 1) Python kütüphaneleri  →  (import_adi, pip_adi, açıklama)
GEREKLI_KUTUPHANELER = [
    ("cv2",         "opencv-python",        "Video okuma ve görüntü işleme"),
    ("numpy",       "numpy",                "Sayısal hesaplama"),
    ("torch",       "torch torchvision torchaudio",
                                            "YOLOv8 modelleri için PyTorch"),
    ("ultralytics", "ultralytics",          "YOLO nesne tespiti"),
    ("deepface",    "deepface",             "Yüz tanıma (Kamera 1)"),
    ("tensorflow",  "tensorflow",           "DeepFace arka ucu"),
    ("PyQt5",       "PyQt5",                "Arayüz kütüphanesi"),
]

# 2) Model dosyaları  →  (dosya_yolu, açıklama, indirme_notu)
GEREKLI_MODELLER = [
    (
        os.path.join(ANA_DIZIN, "yolov8n-face-lindevs.pt"),
        "Yüz Tespit Modeli (Kamera 1)",
        "https://github.com/lindevs/yolov8-face adresinden indirip proje klasörüne koyun."
    ),
    (
        os.path.join(ANA_DIZIN, "yolov8n.pt"),
        "Genel Nesne Tespit Modeli (Kamera 2, 3, 4, 6)",
        "Terminal'de:  python -c \"from ultralytics import YOLO; YOLO('yolov8n.pt')\"  komutu ile otomatik indirilir."
    ),
    (
        os.path.join(ANA_DIZIN, "yolov8m.pt"),
        "Medium Nesne Tespit Modeli (Kamera 5 — Şüpheli Çanta)",
        "Terminal'de:  python -c \"from ultralytics import YOLO; YOLO('yolov8m.pt')\"  komutu ile otomatik indirilir."
    ),
]

# 3) Video dosyaları  →  (dosya_adi, kamera_aciklamasi)
GEREKLI_VIDEOLAR = [
    ("kosma.mp4",    "Kamera 2 — Hız İhlali"),
    ("bekleme.mp4",  "Kamera 3 — Bekleme İhlali"),
    ("topluluk.mp4", "Kamera 4 — Kalabalık Analizi"),
    ("esya.mp4",     "Kamera 5 — Şüpheli Çanta"),
    ("araba.mp4",    "Kamera 6 — Hatalı Park"),
]

VIDEO_KLASORU = os.path.join(ANA_DIZIN, "Videolar")

# ============================================================
#  KONTROL FONKSİYONLARI
# ============================================================

def _kutuphaneleri_kontrol_et():
    eksikler = []
    for import_adi, pip_adi, aciklama in GEREKLI_KUTUPHANELER:
        try:
            __import__(import_adi)
        except ImportError:
            eksikler.append((import_adi, pip_adi, aciklama))
    return eksikler


def _modelleri_kontrol_et():
    eksikler = []
    for yol, aciklama, not_ in GEREKLI_MODELLER:
        if not os.path.isfile(yol):
            eksikler.append((os.path.basename(yol), aciklama, not_))
    return eksikler


def _videolari_kontrol_et():
    eksikler = []
    for dosya_adi, aciklama in GEREKLI_VIDEOLAR:
        tam_yol = os.path.join(VIDEO_KLASORU, dosya_adi)
        if not os.path.isfile(tam_yol):
            eksikler.append((dosya_adi, aciklama, tam_yol))
    return eksikler


# ============================================================
#  POPUP OLUŞTURMA (PyQt5 yoksa konsola yaz)
# ============================================================

def _popup_goster(baslik, html_icerik):
    """Detaylı HTML içerikli kaydırılabilir bir uyarı penceresi gösterir."""
    try:
        from PyQt5.QtWidgets import (
            QDialog, QVBoxLayout, QHBoxLayout, QTextBrowser,
            QPushButton, QLabel, QApplication
        )
        from PyQt5.QtCore import Qt
        from PyQt5.QtGui import QFont

        # QApplication zaten varsa yeniden oluşturma
        app = QApplication.instance()
        _app_olusturuldu = False
        if app is None:
            app = QApplication(sys.argv)
            _app_olusturuldu = True

        dialog = QDialog()
        dialog.setWindowTitle(baslik)
        dialog.setMinimumWidth(760)
        dialog.setMinimumHeight(520)
        dialog.setWindowFlags(Qt.Dialog | Qt.WindowCloseButtonHint)
        dialog.setStyleSheet("""
            QDialog {
                background-color: #0d0d0d;
                color: #e0e0e0;
            }
            QTextBrowser {
                background-color: #111111;
                color: #e0e0e0;
                border: 1px solid #2a2a2a;
                border-radius: 6px;
                font-size: 10pt;
                font-family: Consolas, Courier New, monospace;
                padding: 10px;
            }
            QPushButton {
                background-color: #1a1a2e;
                color: #aaaaff;
                border: 1px solid #3a3aff;
                border-radius: 5px;
                padding: 8px 24px;
                font-size: 10pt;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #2a2a4e;
            }
            QLabel#baslik {
                color: #ff4444;
                font-size: 13pt;
                font-weight: bold;
                padding: 6px 0;
            }
            QLabel#alt_not {
                color: #666666;
                font-size: 9pt;
                font-style: italic;
            }
        """)

        duzen = QVBoxLayout(dialog)
        duzen.setContentsMargins(18, 18, 18, 14)
        duzen.setSpacing(10)

        # Başlık
        baslik_etiketi = QLabel("⚠  EKSİK BAĞIMLILIKLAR TESPİT EDİLDİ")
        baslik_etiketi.setObjectName("baslik")
        baslik_etiketi.setAlignment(Qt.AlignLeft)
        duzen.addWidget(baslik_etiketi)

        # İçerik alanı
        icerik_kutusu = QTextBrowser()
        icerik_kutusu.setOpenExternalLinks(True)
        icerik_kutusu.setHtml(html_icerik)
        duzen.addWidget(icerik_kutusu, 1)

        # Alt not
        alt_not = QLabel("Arayüz açılmaya devam edecek. Eksik bileşenlere ait kameralar görüntü göstermeyecektir.")
        alt_not.setObjectName("alt_not")
        alt_not.setWordWrap(True)
        duzen.addWidget(alt_not)

        # Butonlar
        buton_satiri = QHBoxLayout()
        buton_satiri.addStretch()
        tamam_btn = QPushButton("Anladım, Devam Et")
        tamam_btn.setDefault(True)
        tamam_btn.clicked.connect(dialog.accept)
        buton_satiri.addWidget(tamam_btn)
        duzen.addLayout(buton_satiri)

        dialog.exec_()

        if _app_olusturuldu:
            pass  # exec_() zaten event loop'u yönetir

    except Exception as e:
        # PyQt5 bile kurulamadıysa konsola yaz
        print("\n" + "="*60)
        print(baslik)
        # HTML taglerini temizleyip konsola bas
        import re
        temiz = re.sub(r'<[^>]+>', '', html_icerik).strip()
        print(temiz)
        print("="*60 + "\n")


# ============================================================
#  HTML RAPOR OLUŞTURMA
# ============================================================

def _html_olustur(eksik_kutuphaneler, eksik_modeller, eksik_videolar):
    bloklar = []

    # --- Stil sabitleri ---
    BASLIK_STIL  = "color:#ff6666; font-size:12pt; font-weight:bold; margin-top:14px; margin-bottom:4px;"
    SATIR_STIL   = "margin: 3px 0; padding: 4px 8px; background:#1a1a1a; border-left: 3px solid {renk}; border-radius:3px;"
    KOD_STIL     = "background:#0a0a0a; color:#7ec8e3; padding:4px 10px; border-radius:4px; font-family:Consolas,monospace; display:block; margin:4px 0 8px 0;"
    NOT_STIL     = "color:#888888; font-size:9pt; margin-left:8px;"

    # ---- 1. EKSİK KÜTÜPHANELER ----
    if eksik_kutuphaneler:
        bloklar.append(f'<p style="{BASLIK_STIL}">📦  Eksik Python Kütüphaneleri ({len(eksik_kutuphaneler)} adet)</p>')
        bloklar.append('<p style="color:#aaaaaa; font-size:9pt;">Aşağıdaki komutu terminalde (cmd / PowerShell) çalıştırın:</p>')

        pip_listesi = " ".join(pip for _, pip, _ in eksik_kutuphaneler)
        bloklar.append(f'<code style="{KOD_STIL}">pip install {pip_listesi}</code>')

        bloklar.append('<table width="100%" cellspacing="4" style="margin-top:6px;">')
        for import_adi, pip_adi, aciklama in eksik_kutuphaneler:
            bloklar.append(
                f'<tr>'
                f'<td style="color:#ff4444; font-weight:bold; width:130px;">{import_adi}</td>'
                f'<td style="color:#aaaaaa;">{aciklama}</td>'
                f'<td style="color:#555555; font-size:9pt; text-align:right;">pip install {pip_adi}</td>'
                f'</tr>'
            )
        bloklar.append('</table>')

        # PyTorch özel notu
        if any(k == "torch" for k, _, _ in eksik_kutuphaneler):
            bloklar.append(
                f'<p style="{NOT_STIL}">⚡ PyTorch için GPU desteği istiyorsanız '
                f'<a href="https://pytorch.org/get-started/locally/" style="color:#7ec8e3;">pytorch.org</a> '
                f'adresinden sisteminize uygun komutu alın.</p>'
            )

    # ---- 2. EKSİK MODEL DOSYALARI ----
    if eksik_modeller:
        bloklar.append(f'<p style="{BASLIK_STIL}">🧠  Eksik Model Dosyaları ({len(eksik_modeller)} adet)</p>')
        bloklar.append('<p style="color:#aaaaaa; font-size:9pt;">Model dosyaları proje ana klasöründe bulunmalıdır.</p>')
        for dosya, aciklama, not_ in eksik_modeller:
            bloklar.append(
                f'<div style="{SATIR_STIL.format(renk="#e67e22")}">'
                f'<span style="color:#e67e22; font-weight:bold;">{dosya}</span> '
                f'<span style="color:#888;">&nbsp;—&nbsp;{aciklama}</span><br>'
                f'<span style="{NOT_STIL}">{not_}</span>'
                f'</div>'
            )

    # ---- 3. EKSİK VİDEO DOSYALARI ----
    if eksik_videolar:
        bloklar.append(f'<p style="{BASLIK_STIL}">🎞  Eksik Video Dosyaları ({len(eksik_videolar)} adet)</p>')
        bloklar.append(
            f'<p style="color:#aaaaaa; font-size:9pt;">'
            f'Video dosyaları <code style="color:#7ec8e3;">Videolar/</code> alt klasörüne koyulmalıdır.'
            f'&nbsp; Tam yol: <code style="color:#555;">{VIDEO_KLASORU}</code></p>'
        )
        bloklar.append('<table width="100%" cellspacing="4">')
        for dosya, aciklama, tam_yol in eksik_videolar:
            bloklar.append(
                f'<tr>'
                f'<td style="color:#f39c12; font-weight:bold; width:140px;">{dosya}</td>'
                f'<td style="color:#aaaaaa;">{aciklama}</td>'
                f'</tr>'
            )
        bloklar.append('</table>')
        bloklar.append(
            f'<p style="{NOT_STIL}">Videolar klasörü mevcut değilse manuel oluşturun: '
            f'<code style="color:#7ec8e3;">mkdir Videolar</code></p>'
        )

    # ---- GENEL KURULUM NOTU ----
    bloklar.append(
        '<hr style="border:none; border-top:1px solid #222; margin-top:18px;"/>'
        '<p style="color:#555555; font-size:9pt; margin-top:8px;">'
        '📂 Proje klasör yapısı şu şekilde olmalıdır:<br>'
        '<code style="color:#666; line-height:1.8;">'
        'proje/<br>'
        '&nbsp;&nbsp;├── main.py<br>'
        '&nbsp;&nbsp;├── video_processor.py<br>'
        '&nbsp;&nbsp;├── startup_check.py<br>'
        '&nbsp;&nbsp;├── style.qss<br>'
        '&nbsp;&nbsp;├── yolov8n.pt<br>'
        '&nbsp;&nbsp;├── yolov8m.pt<br>'
        '&nbsp;&nbsp;├── yolov8n-face-lindevs.pt<br>'
        '&nbsp;&nbsp;├── Gorevli/  &nbsp;(yetkili kişi fotoğrafları)<br>'
        '&nbsp;&nbsp;└── Videolar/<br>'
        '&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── kosma.mp4<br>'
        '&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── bekleme.mp4<br>'
        '&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── topluluk.mp4<br>'
        '&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── esya.mp4<br>'
        '&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└── araba.mp4'
        '</code></p>'
    )

    return "\n".join(bloklar)


# ============================================================
#  ANA FONKSİYON — main.py buradan çağırır
# ============================================================

def baslangic_kontrolu_yap():
    """
    Tüm eksiklikleri kontrol eder.
    Herhangi bir eksik varsa ayrıntılı popup gösterir.
    Döndürdüğü sözlük:
      {
        "modeller_tamam": bool,   # .pt dosyaları mevcut mu
        "videolar_tamam": bool,   # video dosyaları mevcut mu
        "eksik_videolar": list,   # eksik video dosya adları
      }
    Arayüz her zaman açılır — bu fonksiyon asla crash etmez.
    """
    eksik_kutuphaneler = _kutuphaneleri_kontrol_et()
    eksik_modeller     = _modelleri_kontrol_et()
    eksik_videolar     = _videolari_kontrol_et()

    hersey_tamam = (
        not eksik_kutuphaneler
        and not eksik_modeller
        and not eksik_videolar
    )

    if not hersey_tamam:
        html = _html_olustur(eksik_kutuphaneler, eksik_modeller, eksik_videolar)
        _popup_goster("Sistem Kontrol Raporu — Eksik Bileşenler", html)

    eksik_video_adlari = {dosya for dosya, _, _ in eksik_videolar}

    return {
        "modeller_tamam": len(eksik_modeller) == 0,
        "videolar_tamam": len(eksik_videolar) == 0,
        "eksik_videolar": eksik_video_adlari,
    }
