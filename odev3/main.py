import cv2
import numpy as np
from ultralytics import YOLO
from matplotlib import pyplot as plt
import matplotlib

# Matplotlib'in arka ucunu Agg olarak ayarla
matplotlib.use('Agg')

# Kenar tespiti yapılacak resmin yolu
image_path = "C:\\Users\\Muhsin21\\Desktop\\Edge Detection.png"

# Resmi yükle
image = cv2.imread(image_path)
if image is None:
    raise ValueError("Resim bulunamadı veya yüklenemedi. Lütfen geçerli bir yol belirtin.")

# YOLOv8 modelini yükle
model = YOLO('yolov8s.pt')

# YOLOv8 kullanarak nesne algılama yap
results = model.predict(image, save=False, save_txt=False)

# Algılanan nesnelerin etrafına bounding box (sınır kutuları) çiz
for result in results:
    for box in result.boxes.xyxy:
        x1, y1, x2, y2 = map(int, box[:4])  # Koordinatları tam sayıya çevir
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

# Kenar tespiti için resmi gri tonlamaya çevir ve Canny algoritmasını uygula
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 100, 200)

# Sonuçları görselleştir ve dosya olarak kaydet
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Algılanan Nesneler (YOLOv8 ile)")
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

plt.subplot(1, 2, 2)
plt.title("Kenar Tespiti (Canny)")
plt.imshow(edges, cmap='gray')

# Grafikleri bir dosyaya kaydet
plt.savefig("output.png")
print("Grafikler output.png dosyasına kaydedildi.")
