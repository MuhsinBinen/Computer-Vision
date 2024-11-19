# -*- coding: utf-8 -*-
"""predict.py"""

# YOLO kütüphanesini yükle
from ultralytics import YOLO
import cv2

# Eğitimden gelen kendi modelinizi yükleyin
custom_model_path = "C:\\Users\\Muhsin21\\PycharmProjects\\Vize\\FinalSonuc\\train\\weights\\best.pt"
custom_model = YOLO(custom_model_path)

# COCO için önceden eğitilmiş modeli yükleyin
coco_model = YOLO("yolov8n.pt")  # COCO için YOLOv8 nano model (daha hızlı)

# Webcam'den görüntü al ve tahmin yap
def webcam_detection():
    # Webcam'i başlat
    cap = cv2.VideoCapture(0)  # 0, varsayılan kamerayı ifade eder
    if not cap.isOpened():
        print("Kamera açılamadı!")
        return

    print("Tahmin için webcam açık. Çıkmak için 'q' tuşuna basın.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Kamera akışı alınamadı!")
            break

        # Kendi modelinizle tahmin yap
        custom_results = custom_model.predict(source=frame, conf=0.3, show=False)
        custom_annotated_frame = custom_results[0].plot()  # Kendi modelinizin tahminleri

        # COCO modeliyle tahmin yap
        coco_results = coco_model.predict(source=frame, conf=0.5, show=False)
        coco_annotated_frame = coco_results[0].plot()  # COCO modelinin tahminleri

        # Çerçeveyi ekranda göster (iki ayrı pencere)
        cv2.imshow("Custom Model Detection", custom_annotated_frame)
        cv2.imshow("COCO Model Detection", coco_annotated_frame)

        # Çıkış için 'q' tuşuna basılırsa döngüyü kır
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Kaynakları serbest bırak ve pencereleri kapat
    cap.release()
    cv2.destroyAllWindows()

# Ana fonksiyon
if __name__ == "__main__":
    print("1: Webcam ile tahmin")
    print("2: Görüntü/Videodan tahmin")
    choice = input("Seçiminizi yapın (1 veya 2): ")

    if choice == "1":
        webcam_detection()
    elif choice == "2":
        input_path = input("Tahmin yapılacak görüntü veya video dosyasının yolunu girin: ")
        # Kendi modelinizle tahmin yap
        custom_results = custom_model.predict(source=input_path, save=True, save_txt=True, save_crop=True)
        print("Kendi modelinizle tahmin tamamlandı! Sonuçlar kaydedildi.")

        # COCO modeliyle tahmin yap
        coco_results = coco_model.predict(source=input_path, save=True, save_txt=True, save_crop=True)
        print("COCO model tahmini tamamlandı! Sonuçlar kaydedildi.")
    else:
        print("Geçersiz seçim!")
