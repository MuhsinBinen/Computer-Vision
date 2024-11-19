import cv2
import numpy as np

# 1. Görüntüyü okuma
image = cv2.imread('images.jpg')  # 'gorsel.jpg' yerine kendi görüntünüzü kullanın
if image is None:
    print("Görüntü yüklenemedi. Lütfen doğru dosya yolunu kontrol edin.")
    exit()

# Pencereyi tam ekran modunda aç
cv2.namedWindow('Orijinal - Ortalama Filtre - Laplace Filtre', cv2.WINDOW_NORMAL)
cv2.setWindowProperty('Orijinal - Ortalama Filtre - Laplace Filtre', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# 2. Ortalama Filtre (Blurring) uygulama
average_filtered = cv2.blur(image, (5, 5))  # 5x5 kernel boyutunda ortalama filtre

# 3. Laplace filtresi uygulama
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Laplace filtresi için gri tonlamaya çeviriyoruz
laplace_filtered = cv2.Laplacian(gray_image, cv2.CV_64F)  # Laplace filtresi
laplace_filtered = cv2.convertScaleAbs(laplace_filtered)  # Mutlak değere çevirip, 8-bit formatına dönüştürme

# Gri tonlamalı Laplace filtresi sonucu, diğer BGR formatındaki görüntülerle aynı formatta olmalı
laplace_bgr = cv2.cvtColor(laplace_filtered, cv2.COLOR_GRAY2BGR)

# 4. Görüntüleri yan yana birleştirme
combined_image = np.hstack((image, average_filtered, laplace_bgr))

# Pencere boyutuna göre görüntüyü yeniden boyutlandırma
while True:
    # Pencerenin şu anki boyutlarını al
    _, _, win_width, win_height = cv2.getWindowImageRect('Orijinal - Ortalama Filtre - Laplace Filtre')

    # Görüntüyü pencere boyutuna göre yeniden boyutlandır
    resized_combined_image = cv2.resize(combined_image, (win_width, win_height))

    # Yeniden boyutlandırılmış görüntüyü göster
    cv2.imshow('Orijinal - Ortalama Filtre - Laplace Filtre', resized_combined_image)

    # ESC tuşuna basılana kadar bekler
    key = cv2.waitKey(1) & 0xFF  # Basılan tuşu kontrol et
    if key == 27:  # ESC tuşu ASCII değeri 27
        break

# Pencereleri kapat
cv2.destroyAllWindows()