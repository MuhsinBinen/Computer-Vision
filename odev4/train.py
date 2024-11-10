from ultralytics import YOLO
import matplotlib

matplotlib.use('Agg')  # For non-interactive backend


# Roboflow'dan indirilen veri seti dosyasının yolunu belirleyin
dataset_yaml = 'C:\\Users\\Muhsin21\\PycharmProjects\\odev4\\Dataset\\Odev4.v1i.yolov8\\data.yaml'

# YOLOv8 modelini başlatın
model = YOLO('yolov8s.pt')

# Modeli eğitin
model.train(data=dataset_yaml, epochs=10, imgsz=640, batch=8, name='apricot_model')
