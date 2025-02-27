import os
import torch
import shutil
from ultralytics import YOLO
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Modeli Yükle (YOLOv11X-CLS)
model = YOLO("yolo11x-cls.pt")  # Eğitilmiş önceden var olan model

# Modeli Eğit
model.train(
    data="/content/drive/MyDrive/stroke3/son_veriler3",  # Eğitim veri setinin yolu
    epochs=5,               # Epoch sayısı (gerekirse artırılabilir)
    imgsz=224,              # Görüntü boyutu (224x224)
    batch=16,               # Mini-batch boyutu
    device="cuda"           # GPU kullanımı (CPU için "cpu" yaz)
)

# En iyi eğitilmiş modelin yolunu belirle
best_model_path = "runs/classify/train/weights/best.pt" #burası colabta çalıştırılan makinedeki yol
"""
# Eğer dosya yoksa, başka bir yere kaydedilmiş olabilir. Otomatik bulma:
if not os.path.exists(best_model_path):
    search_result = !find runs/ -name "best.pt"
    if search_result:
        best_model_path = search_result[0]  # Bulunan ilk 'best.pt' dosyasını kullan
"""
# Model dosyasının gerçekten var olup olmadığını tekrar kontrol et
if os.path.exists(best_model_path):
    best_model_drive_path = "/content/drive/MyDrive/YoloV12-Cls/best_model.pt"
    os.makedirs(os.path.dirname(best_model_drive_path), exist_ok=True)
    shutil.copy(best_model_path, best_model_drive_path)
    print(f"✅ Model ağırlıkları Google Drive'a kaydedildi: {best_model_drive_path}")
else:
    print("❌ Hata: 'best.pt' bulunamadı. Lütfen model eğitiminin tamamlandığından emin olun.")

# Modeli yükle
best_model = YOLO(best_model_drive_path)

# Test veri klasörünü tanımla
test_folder = "/content/drive/MyDrive/stroke3/son_veriler3/test"  # Test klasörünün yolu

# Test görüntülerini ve etiketlerini al
test_images = []
true_labels = []

for label, class_name in enumerate(["normal", "stroke"]):  # 0 = normal, 1 = stroke
    class_path = os.path.join(test_folder, class_name)
    if os.path.exists(class_path):
        for img in os.listdir(class_path):
            if img.endswith((".jpg", ".png", ".jpeg")):
                test_images.append(os.path.join(class_path, img))
                true_labels.append(label)

# Modeli test verisi üzerinde çalıştır ve tahminleri al
results = best_model.predict(source=test_images)

# Tahmin edilen etiketleri al
pred_labels = [result.probs.top1 for result in results]

# Metrikleri hesapla
accuracy = accuracy_score(true_labels, pred_labels)
precision = precision_score(true_labels, pred_labels, average="binary")
recall = recall_score(true_labels, pred_labels, average="binary")
f1 = f1_score(true_labels, pred_labels, average="binary")

# Sonuçları yazdır
print(f"✅ Accuracy (Doğruluk): {accuracy:.4f}")
print(f"✅ Precision (Kesinlik): {precision:.4f}")
print(f"✅ Recall (Duyarlılık): {recall:.4f}")
print(f"✅ F1 Skoru: {f1:.4f}")

# Checkpoint kaydet (Model ağırlıklarını .pth olarak sakla)
checkpoint_path = "/content/drive/MyDrive/YoloV12-Cls/model_checkpoint.pth"
torch.save(best_model.model.state_dict(), checkpoint_path)
print(f"📌 Model checkpoint kaydedildi: {checkpoint_path}")
