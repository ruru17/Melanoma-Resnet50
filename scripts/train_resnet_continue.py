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

#################################################################
# DATASET PERSONALIZADO
#################################################################

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


#################################################################
# TRANSFORMACIONES
#################################################################

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


#################################################################
# ENTRENAMIENTO CONTINUADO
#################################################################

def main():

    device = torch.device("cpu")
    print(f"Usando dispositivo: {device}")

    train_transform, val_transform = get_transforms()

    #############################################################
    # DATASET PREPROCESADO
    #############################################################

    full_dataset = SkinDataset(
        csv_file="data/labels_final.csv",
        root_dir="data/images_preprocessed",
        transform=train_transform
    )

    # Dividir 80% train / 20% val
    val_size = int(0.2 * len(full_dataset))
    train_size = len(full_dataset) - val_size

    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # La parte de validación NO debe tener augmentación
    val_dataset.dataset.transform = val_transform

    #############################################################
    # WEIGHTED SAMPLER — CORRECCIÓN
    #############################################################

    # Índices del subconjunto de entrenamiento
    train_indices = train_dataset.indices

    # Etiquetas solo del set de entrenamiento
    train_labels = full_dataset.data.iloc[train_indices]["label"].values

    # Contar cuántos ejemplos hay por clase
    class_counts = np.bincount(train_labels)
    class_weights = 1.0 / class_counts

    # Peso por muestra individual
    sample_weights = [class_weights[label] for label in train_labels]

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

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

    #############################################################
    # CARGAR MODELO PREVIO
    #############################################################

    model_path = "models/resnet50_preprocessed_best.pth"
    print("Cargando modelo previo...")

    model = models.resnet50(weights=None)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 2)
    model.load_state_dict(torch.load(model_path, map_location=device))

    model = model.to(device)
    print("Modelo cargado correctamente.")
    print("Continuando entrenamiento...")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-5)

    #############################################################
    # ENTRENAMIENTO
    #############################################################

    best_val_acc = 0.0
    epochs = 15

    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")
        print("------------------------------")

        model.train()
        running_loss = 0.0

        for imgs, labels in tqdm(train_loader, desc="Entrenando"):
            imgs, labels = imgs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        print(f"Train Loss: {avg_loss:.4f}")

        #############################################################
        # VALIDACIÓN
        #############################################################

        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)

                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        val_acc = correct / total
        print(f"Validation Accuracy: {val_acc:.4f}")

        #############################################################
        # GUARDAR MEJOR MODELO
        #############################################################

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "models/resnet50_preprocessed_best.pth")
            print("Nuevo mejor modelo guardado.")

    print("Entrenamiento continuo finalizado.")


if __name__ == "__main__":
    main()
