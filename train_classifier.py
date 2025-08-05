import os
import torch
from torch import nn, optim
from torchvision import transforms, models
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
from PIL import Image
from collections import Counter
from tqdm import tqdm

data_dir = "labeled_cells"
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.RandomApply([
        transforms.RandomRotation(degrees=5),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2)
    ], p=0.7),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

full_dataset = ImageFolder(root=data_dir, transform=transform)
num_classes = len(full_dataset.classes)
class_names = full_dataset.classes

labels = [s[1] for s in full_dataset.samples]
label_counts = Counter(labels)
weights = [1.0 / label_counts[label] for label in labels]
sampler = WeightedRandomSampler(weights, len(weights))

val_pct = 0.1
val_size = int(len(full_dataset) * val_pct)
train_size = len(full_dataset) - val_size
train_ds, val_ds = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=32)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(pretrained=True)
model.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
epochs = 30

for epoch in range(epochs):
    model.train()
    total_loss = 0
    correct = 0
    for xb, yb in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
        xb, yb = xb.to(device), yb.to(device)
        pred = model(xb)
        loss = criterion(pred, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct += (pred.argmax(dim=1) == yb).sum().item()
    acc = correct / len(train_ds)
    print(f"[INFO] Epoch {epoch+1}: Loss={total_loss:.3f}, Accuracy={acc*100:.2f}%")
    model.eval()
    val_correct = 0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            val_correct += (pred.argmax(dim=1) == yb).sum().item()
    val_acc = val_correct / len(val_ds)
    print(f"[INFO] Val Accuracy: {val_acc*100:.2f}%")
    scheduler.step()

torch.save(model.state_dict(), "sfenizer.pth")
print("[SUCCESS] Model saved!")