import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import os
from sklearn.metrics import precision_score, recall_score, f1_score


# Modeli yükleme fonksiyonu
def load_model(weights_path, device):
    model = models.resnet50()
    # CBAM eklendiğini varsayarak CBAM içeren modelinize uygun şekilde düzenleyin
    model.fc = torch.nn.Linear(model.fc.in_features, 2)  # Binary classification için çıkış katmanı
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()
    return model


# Görüntü işleme fonksiyonu
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)  # Batch boyutu ekleme


# Test fonksiyonu
def test_model(model, test_folder, device):
    image_paths = [os.path.join(test_folder, img) for img in os.listdir(test_folder) if
                   img.endswith(('png', 'jpg', 'jpeg'))]

    y_true = []  # Gerçek etiketler
    y_pred = []  # Tahmin edilen etiketler

    for img_path in image_paths:
        label = 1 if "positive" in img_path else 0  # Gerçek etiketi belirleme (dosya ismine göre)
        y_true.append(label)

        image_tensor = preprocess_image(img_path).to(device)
        with torch.no_grad():
            output = model(image_tensor)
            predicted_class = torch.argmax(output, dim=1).item()
            y_pred.append(predicted_class)

        print(f"{img_path}: Tahmin edilen sınıf {predicted_class}")

    # Precision, Recall, F1 Score hesaplama
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights_path = "model_weights.pth"  # Eğittiğin ağırlık dosyanın yolu
    test_folder = "test_images"  # Test görüntülerinin bulunduğu klasör

    model = load_model(weights_path, device)
    test_model(model, test_folder, device)