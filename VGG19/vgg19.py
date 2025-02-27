import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torchvision import models
from torch.utils.data import DataLoader
from torchvision.models import VGG19_Weights
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os

# **Cihazı belirle (GPU varsa kullan)**
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Kullanılan cihaz: {device}")

# **Veri Dönüşümleri (Preprocessing)**
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# **Dataset ve DataLoader**
train_dataset = ImageFolder(root="/content/drive/MyDrive/stroke2/son_veriler2/train", transform=transform)
val_dataset = ImageFolder(root="/content/drive/MyDrive/stroke2/son_veriler2/validation", transform=transform)
test_dataset = ImageFolder(root="/content/drive/MyDrive/stroke2/son_veriler2/test", transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# **VGG19 Modelini Yükle ve Özelleştir**
model = models.vgg19(weights=VGG19_Weights.IMAGENET1K_V1)
model.classifier[6] = nn.Linear(4096, 1)
model = model.to(device)

# **Kayıp fonksiyonu ve optimizasyon**
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# **Checkpoint dosya yolu**
checkpoint_path = "/content/drive/MyDrive/Vgg19/model_checkpoint.pth"
best_model_path = "/content/drive/MyDrive/Vgg19/best_model.pth"

# **Modeli Yükleme Fonksiyonu**
def load_checkpoint(model, optimizer, checkpoint_path):
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_accuracy = checkpoint.get('best_accuracy', 0)
        print(f"Checkpoint yüklendi. Kaldığı yerden devam ediyor: Epoch {start_epoch}")
    else:
        start_epoch = 0
        best_accuracy = 0
    return model, optimizer, start_epoch, best_accuracy

# **Modeli yükle (varsa)**
model, optimizer, start_epoch, best_accuracy = load_checkpoint(model, optimizer, checkpoint_path)

# **Eğitim ve Değerlendirme Fonksiyonu**
def train_or_validate(model, dataloader, criterion, optimizer=None):
    if optimizer:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    all_labels = []
    all_preds = []

    with torch.set_grad_enabled(optimizer is not None):
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device).float()
            outputs = model(inputs).view(-1)
            loss = criterion(outputs, labels)

            if optimizer:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            preds = torch.round(torch.sigmoid(outputs))
            all_labels.extend(labels.cpu().detach().numpy())
            all_preds.extend(preds.cpu().detach().numpy())

    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds) * 100

    return avg_loss, accuracy

# **Eğitim Döngüsü**
epochs = 30
for epoch in range(start_epoch, epochs):
    print(f"Epoch {epoch+1} started")
    train_loss, train_accuracy = train_or_validate(model, train_loader, criterion, optimizer)
    val_loss, val_accuracy = train_or_validate(model, val_loader, criterion)

    print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%")
    print(f"Validation Loss: {val_loss:.4f}, Validation Acc: {val_accuracy:.2f}%")

    # **Checkpoint kaydetme**
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_accuracy': best_accuracy
    }, checkpoint_path)
    print(f"Checkpoint kaydedildi: Epoch {epoch+1}")

    # **En iyi modeli kaydet**
    if val_accuracy > best_accuracy:
        best_accuracy = val_accuracy
        torch.save(model.state_dict(), best_model_path)
        print("Yeni en iyi model kaydedildi!")

# **Test Aşaması**
print("\nTest Verisi ile Modeli Değerlendiriyoruz...")
test_loss, test_accuracy = train_or_validate(model, test_loader, criterion)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.2f}%")