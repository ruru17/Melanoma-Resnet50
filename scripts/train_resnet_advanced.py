import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, WeightedRandomSampler, Dataset
from torchvision import models, transforms
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt

# ============================
# CONFIGURACION GENERAL
# ============================

CSV_PATH = "data/preprocessing_report.csv"
IMG_DIR = "data/images_preprocessed"
MODEL_DIR = "models"
METRICS_DIR = "reports"
EPOCHS = 15
BATCH_SIZE = 16
LR = 1e-4

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)

device = torch.device("cpu")
print(f"Usando dispositivo: {device}")

# ============================
# DATASET PERSONALIZADO
# ============================

class SkinDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        row = self.data.iloc[idx]
        img_name = row["image_id"] + ".jpg"
        img_path = os.path.join(self.img_dir, img_name)

        image = Image.open(img_path).convert("RGB")
        label = int(row["label"])

        if self.transform:
            image = self.transform(image)

        return image, label

# ============================
# TRANSFORMACIONES
# (suaves porque ya hay preprocesamiento)
# ============================

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(
        brightness=0.10,
        contrast=0.10,
        saturation=0.05,
        hue=0.01
    ),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# ============================
# CARGAR DATASET
# ============================

df = pd.read_csv(CSV_PATH)
dataset = SkinDataset(CSV_PATH, IMG_DIR, transform=train_transform)

# split 80/20
val_size = int(0.2 * len(dataset))
train_size = len(dataset) - val_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

val_dataset.dataset.transform = val_transform

# ============================
# WEIGHTED SAMPLER (solo train)
# ============================

train_indices = train_dataset.indices
train_labels = df.iloc[train_indices]["label"].values

class_counts = np.bincount(train_labels)
class_weights = 1.0 / class_counts

train_sample_weights = np.array([class_weights[l] for l in train_labels])

sampler = WeightedRandomSampler(
    weights=train_sample_weights,
    num_samples=len(train_sample_weights),
    replacement=True
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ============================
# MODELO RESNET50
# ============================

model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 2)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)
scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)

best_val_acc = 0.0
train_losses = []
val_losses = []
val_accuracies = []

# ============================
# FUNCION DE VALIDACION
# ============================

def evaluate(model, loader):
    model.eval()
    total = 0
    correct = 0
    loss_total = 0

    all_labels = []
    all_preds = []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)

            loss = criterion(outputs, labels)
            loss_total += loss.item()

            _, preds = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (preds == labels).sum().item()

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    acc = correct / total
    return loss_total / len(loader), acc, all_labels, all_preds

# ============================
# ENTRENAMIENTO PRINCIPAL
# ============================

print("Iniciando entrenamiento avanzado...")

for epoch in range(1, EPOCHS + 1):
    model.train()
    running_loss = 0.0

    print(f"\n==============================")
    print(f" Epoch {epoch}/{EPOCHS}")
    print(f"==============================")

    for imgs, labels in tqdm(train_loader, desc="Entrenando"):
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_train_loss = running_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    val_loss, val_acc, y_true, y_pred = evaluate(model, val_loader)
    val_losses.append(val_loss)
    val_accuracies.append(val_acc)

    print(f"Train Loss: {avg_train_loss:.4f}")
    print(f"Val Loss:   {val_loss:.4f}")
    print(f"Val Acc:    {val_acc*100:.2f}%")

    scheduler.step()

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), f"{MODEL_DIR}/resnet50_preprocessed_best.pth")
        print("Nuevo mejor modelo guardado.")

# ============================
# REPORTES Y METRICAS
# ============================

print("\nGenerando reportes...")

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(5, 4))
plt.imshow(cm, cmap="Blues")
plt.title("Matriz de Confusion")
plt.colorbar()
plt.xlabel("Prediccion")
plt.ylabel("Real")
plt.savefig(f"{METRICS_DIR}/confusion_matrix.png")
plt.close()

report = classification_report(y_true, y_pred, output_dict=True)
df_report = pd.DataFrame(report).transpose()
df_report.to_csv(f"{METRICS_DIR}/classification_report.csv")

print("Entrenamiento completado.")
print(f"Matriz de confusion guardada en: {METRICS_DIR}/confusion_matrix.png")
print(f"Reporte guardado en: {METRICS_DIR}/classification_report.csv")
print(f"Mejor modelo guardado en: {MODEL_DIR}/resnet50_preprocessed_best.pth")
