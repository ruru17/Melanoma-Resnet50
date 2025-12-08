import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from torchvision import models, transforms
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ======== Rutas de modelos ========
OLD_MODEL_PATH = "models/resnet50_melanoma_best.pth"
NEW_MODEL_PATH = "models/resnet50_preprocessed_best.pth"

# ======== Ruta de etiquetas reales ========
# Este CSV fue generado directamente desde HAM10000 metadata
LABELS_CSV = "data/labels_final.csv"

# ======== Rutas de imágenes ========
IMG_DIR_ORIGINAL = "data/images"
IMG_DIR_PREPROCESSED = "data/images_preprocessed"

# ======== Carpeta donde se guardarán los resultados ========
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

DEVICE = torch.device("cpu")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def load_model(path):
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.eval()
    return model

def load_image(path):
    img = Image.open(path).convert("RGB")
    return transform(img).unsqueeze(0)

def evaluate(model, image_paths):
    preds = []
    for p in image_paths:
        x = load_image(p)
        with torch.no_grad():
            out = model(x)
            preds.append(torch.argmax(out, dim=1).item())
    return np.array(preds)

def save_confusion_matrix(y_true, y_pred, name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, cmap="Blues", fmt="d")
    plt.xlabel("Predicción del modelo")
    plt.ylabel("Clasificación real")
    plt.title(name)
    plt.savefig(f"{RESULTS_DIR}/{name}.png", dpi=300)
    plt.close()

def main():
    df = pd.read_csv(LABELS_CSV)

    # Dividir dataset para evaluación (20% fija)
    df_test = df.sample(frac=0.2, random_state=42)

    print(f"Total test: {len(df_test)}")

    image_ids = df_test["image_id"].values
    labels = df_test["label"].values  # Ground truth real proveniente de HAM10000

    # Rutas de imágenes
    paths_old = [f"{IMG_DIR_ORIGINAL}/{img}.jpg" for img in image_ids]
    paths_new = [f"{IMG_DIR_PREPROCESSED}/{img}.jpg" for img in image_ids]

    # Cargar modelos
    old_model = load_model(OLD_MODEL_PATH)
    new_model = load_model(NEW_MODEL_PATH)

    # Evaluar
    old_preds = evaluate(old_model, paths_old)
    new_preds = evaluate(new_model, paths_new)

    # Métricas
    metrics = {
        "metric": ["accuracy", "precision", "recall", "f1"],
        "old_model": [
            accuracy_score(labels, old_preds),
            precision_score(labels, old_preds, zero_division=0),
            recall_score(labels, old_preds),
            f1_score(labels, old_preds)
        ],
        "new_model": [
            accuracy_score(labels, new_preds),
            precision_score(labels, new_preds, zero_division=0),
            recall_score(labels, new_preds),
            f1_score(labels, new_preds)
        ]
    }

    df_metrics = pd.DataFrame(metrics)
    df_metrics.to_csv(f"{RESULTS_DIR}/model_metrics_comparison.csv", index=False)

    print(df_metrics)

    save_confusion_matrix(labels, old_preds, "confusion_matrix_old")
    save_confusion_matrix(labels, new_preds, "confusion_matrix_new")

    print("\nResultados guardados en carpeta:")
    print(f"→ {RESULTS_DIR}")

if __name__ == "__main__":
    main()
