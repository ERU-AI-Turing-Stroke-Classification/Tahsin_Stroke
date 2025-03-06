import torch
import numpy as np
import random
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision import models
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
import os

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Google Colab kullanıyorsan GPU'yu kontrol et
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Veri dönüşümleri (Ölçekleme, Normalizasyon vb.)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Veri kümelerini yükle (Örnek: Klasör Yapısı)
data_dir = "/content/drive/MyDrive/stroke3/son_veriler3"  # Buraya veri setinin yolunu yaz
train_dataset = datasets.ImageFolder(root=f"{data_dir}/train", transform=transform)
val_dataset = datasets.ImageFolder(root=f"{data_dir}/validation", transform=transform)
test_dataset = datasets.ImageFolder(root=f"{data_dir}/test", transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False, worker_init_fn=lambda _: np.random.seed(seed))
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# ResNet-50 Modelini Yükle (ImageNet ile Önceden Eğitilmiş)
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)  # 2 sınıflı çıktı (İskemik İnme: Var/Yok)
model = model.to(device)

# Kayıp fonksiyonu ve optimizasyon
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Checkpoint kaydetme fonksiyonu
def save_checkpoint(epoch, model, optimizer, filename="/content/drive/MyDrive/Resnet/stroke_resnet50_checkpoint.pth"):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }
    torch.save(checkpoint, filename)
    print(f"Checkpoint saved at epoch {epoch}")

# Modeli eğitme fonksiyonu
def train_model(model, train_loader, val_loader, criterion, optimizer, epochs=10):
    best_val_acc = 0.0
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        train_acc = 100 * correct / total
        val_acc = evaluate_model(model, val_loader)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {running_loss/len(train_loader):.4f} - Train Acc: {train_acc:.2f}% - Val Acc: {val_acc:.2f}%")

        # En iyi doğrulama doğruluğu elde edilirse modeli kaydet
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(epoch, model, optimizer)

# Modeli doğrulama ve detaylı metrikleri hesaplama fonksiyonu
def evaluate_model(model, loader):
    model.eval()
    correct = 0
    total = 0
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    accuracy = 100 * correct / total
    print("Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=['iskemik_yok', 'iskemik_var']))
    return accuracy

# Modeli eğit
train_model(model, train_loader, val_loader, criterion, optimizer, epochs=10)

# Test seti ile değerlendirme
test_acc = evaluate_model(model, test_loader)
print(f"Test Accuracy: {test_acc:.2f}%")

# Eğitilmiş modeli kaydet
torch.save(model.state_dict(), "/content/drive/MyDrive/Resnet/stroke_resnet50_best.pth")