import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler, random_split
from torchvision import models, transforms
from tqdm import tqdm
import pandas as pd
import numpy as np
from PIL import Image
import os

##############################################
# DATASET PERSONALIZADO
##############################################

class SkinDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        img_name = row["image_id"] + ".jpg"
        img_path = os.path.join(self.root_dir, img_name)

        image = Image.open(img_path).convert("RGB")
        label = int(row["label"])

        if self.transform:
            image = self.transform(image)

        return image, label

##############################################
# TRANSFORMACIONES
##############################################

def get_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ColorJitter(
            brightness=0.20,
            contrast=0.20,
            saturation=0.15,
            hue=0.02
        ),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    return train_transform, val_transform


##############################################
# ENTRENAMIENTO PRINCIPAL
##############################################

def main():

    device = torch.device("cpu")
    print(f"Usando dispositivo: {device}")

    # Transformaciones
    train_transform, val_transform = get_transforms()

    # Dataset completo
    full_dataset = SkinDataset(
        csv_file="data/labels_final.csv",
        root_dir="data/images",
        transform=train_transform
    )

    # Dividir en entrenamiento y validación
    val_size = int(0.2 * len(full_dataset))
    train_size = len(full_dataset) - val_size

    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    val_dataset.dataset.transform = val_transform  # aplicar transformaciones de validación

    ##############################################
    # WEIGHTED RANDOM SAMPLER CORREGIDO
    ##############################################

    full_labels = np.array(full_dataset.data["label"])

    # Frecuencia por clase
    class_counts = np.bincount(full_labels)
    class_weights = 1.0 / class_counts

    # Pesos SOLO para los índices del train_dataset
    train_weights = [class_weights[full_labels[i]] for i in train_dataset.indices]

    sampler = WeightedRandomSampler(
        weights=train_weights,
        num_samples=len(train_weights),
        replacement=True
    )

    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        sampler=sampler
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=16,
        shuffle=False
    )

    ##############################################
    # MODELO RESNET50
    ##############################################

    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 2)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    ##############################################
    # ENTRENAMIENTO
    ##############################################

    best_val_acc = 0.0
    epochs = 10

    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")
        print("------------------------------")

        model.train()
        running_loss = 0.0

        for inputs, labels in tqdm(train_loader, desc="Entrenando"):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        print(f"Train Loss: {avg_loss:.4f}")

        ##############################################
        # VALIDACIÓN
        ##############################################

        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_acc = correct / total
        print(f"Val Acc: {val_acc:.4f}")

        ##############################################
        # GUARDAR MEJOR MODELO
        ##############################################

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "models/resnet50_melanoma_best.pth")
            print("✔ Mejor modelo guardado!")

    print("\nEntrenamiento finalizado.")
    print(f"Mejor exactitud de validación: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()
