import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision.models import resnet50

# Google Colab için GPU kullanımı
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Veri seti yolunu belirle
data_dir = "/content/drive/MyDrive/stroke2/son_veriler2"

# Veri ön işleme
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 2.0)),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Veri setini yükle
train_dataset = datasets.ImageFolder(root=os.path.join(data_dir, "train"), transform=transform)
val_dataset = datasets.ImageFolder(root=os.path.join(data_dir, "validation"), transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=2)

# CBAM modülü
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x

# CBAM'lı ResNet50 modelini yükle
class ResNet50_CBAM(nn.Module):
    def __init__(self, num_classes=2):
        super(ResNet50_CBAM, self).__init__()
        self.model = resnet50(weights='IMAGENET1K_V2')
        self.model.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)  # Siyah-beyaz için
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

        # CBAM ekleyelim
        self.cbam1 = CBAM(256)  # İlk ResNet bloğu çıkışı
        self.cbam2 = CBAM(512)  # İkinci ResNet bloğu çıkışı
        self.cbam3 = CBAM(1024) # Üçüncü ResNet bloğu çıkışı

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.cbam1(x)  # CBAM ekledik

        x = self.model.layer2(x)
        x = self.cbam2(x)

        x = self.model.layer3(x)
        x = self.cbam3(x)

        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.model.fc(x)
        return x

model = ResNet50_CBAM().to(device)

# Kayıp fonksiyonu ve optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10) # Öğrenme oranı planlayıcı

# Checkpoint dosyaları
checkpoint_last = "/content/drive/MyDrive/ResNet50CbamGaus/checkpoint_last.pth"
checkpoint_best = "/content/drive/MyDrive/ResNet50CbamGaus/best_model.pth"

# Eğer checkpoint varsa yükle
start_epoch = 0
best_acc = 0

if os.path.exists(checkpoint_last):
    print("Checkpoint bulundu! Kaldığı yerden devam ediliyor...")
    checkpoint = torch.load(checkpoint_last)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    start_epoch = checkpoint["epoch"] + 1
    best_acc = checkpoint["best_acc"]

# Model eğitimi
num_epochs = 50
for epoch in range(start_epoch, num_epochs):
    model.train()
    train_loss, correct, total = 0, 0, 0
    for images, labels in tqdm(train_loader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    train_acc = 100. * correct / total
    print(f"Epoch {epoch+1}: Train Loss: {train_loss/len(train_loader):.4f}, Accuracy: {train_acc:.2f}%")

    scheduler.step()

    # Modeli validasyon setiyle değerlendir
    model.eval()
    correct, total, val_loss = 0, 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    val_acc = 100. * correct / total
    print(f"Epoch {epoch+1}: Validation Loss: {val_loss/len(val_loader):.4f}, Validation Accuracy: {val_acc:.2f}%")

    # Checkpoint kaydetme (Her epoch sonunda)
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_acc": best_acc,
    }, checkpoint_last)

    # En iyi modeli kaydet
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), checkpoint_best)
        print("Best model saved!")

print("Training finished! Best validation accuracy:", best_acc)
