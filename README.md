# YOLOv8 Cat & Oyuncak Detection

## Proje Bilgileri
- Ders: Makine Öğrenmesi
---

## Proje Açıklaması
Bu projede YOLOv8 kullanılarak iki sınıflı (Cat ve Oyuncak) bir nesne tespit modeli
eğitilmiştir. Eğitilen model PyQt5 tabanlı bir masaüstü arayüz ile entegre edilmiştir.

---

## Kullanılan Teknolojiler
- Python
- YOLOv8 (Ultralytics)
- PyQt5
- OpenCV
- Google Colab

---

## Dosyalar
- `Untitled6.ipynb` : Model eğitimi ve veri hazırlama
- `best.pt` : Eğitilmiş YOLOv8 modeli
- `cats_dolls.yaml` : YOLO yapılandırma dosyası
- `main.py` : PyQt5 grafik arayüz uygulaması

---

## Çalıştırma
```bash
pip install ultralytics pyqt5 opencv-python pillow
python main.py
