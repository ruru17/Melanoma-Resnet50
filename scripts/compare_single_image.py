import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2
import os
import sys
import matplotlib.pyplot as plt
from skimage import exposure


# --------------------------------------------------------
# CONFIGURACIÓN
# --------------------------------------------------------

OLD_MODEL_PATH = "models/resnet50_melanoma_best.pth"
NEW_MODEL_PATH = "models/resnet50_preprocessed_best.pth"

IMAGE_DIR = "data/images_indv"
SAVE_DIR = "results/single_compare"
os.makedirs(SAVE_DIR, exist_ok=True)

DEVICE = torch.device("cpu")


# --------------------------------------------------------
# TRANSFORM PARA ENTRADA A LOS MODELOS
# --------------------------------------------------------
model_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])
])


# --------------------------------------------------------
# CARGAR MODELOS
# --------------------------------------------------------
def load_model(path):
    model = models.resnet50(weights=None)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 2)
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.eval()
    return model


old_model = load_model(OLD_MODEL_PATH)
new_model = load_model(NEW_MODEL_PATH)


# --------------------------------------------------------
# CLAHE EN LAB — Mantiene el color
# --------------------------------------------------------
def apply_clahe(img_rgb):
    img_rgb = img_rgb.astype(np.float32) / 255.0

    # Convertimos RGB → LAB
    lab = cv2.cvtColor((img_rgb * 255).astype(np.uint8), cv2.COLOR_RGB2LAB)

    L, A, B = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    L_eq = clahe.apply(L)

    lab_eq = cv2.merge((L_eq, A, B))

    # LAB → RGB
    img_eq = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2RGB)

    return img_eq


# --------------------------------------------------------
# CLASIFICAR IMAGEN
# --------------------------------------------------------
def infer(model, img_np):
    pil_img = Image.fromarray(img_np)
    tensor = model_transform(pil_img).unsqueeze(0)

    with torch.no_grad():
        out = model(tensor)
        prob = torch.softmax(out, dim=1)
        pred = torch.argmax(prob).item()

    return pred, prob[0][pred].item()


# --------------------------------------------------------
# BUSCAR ARCHIVO
# --------------------------------------------------------
def find_image(base_name):
    exts = [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]

    for ext in exts:
        candidate = os.path.join(IMAGE_DIR, base_name + ext)
        if os.path.exists(candidate):
            return candidate

    return None


# --------------------------------------------------------
# PROCESO PRINCIPAL
# --------------------------------------------------------
def compare_image(base_name):

    img_path = find_image(base_name)
    if img_path is None:
        print("ERROR: No se encontró la imagen en data/images_indv")
        return

    print(f"Imagen encontrada: {img_path}")

    # Cargar original
    img_original = np.array(Image.open(img_path).convert("RGB"))

    # Aplicar CLAHE
    img_preprocessed = apply_clahe(img_original.copy())

    # Inference
    old_pred, old_conf = infer(old_model, img_original)
    new_pred, new_conf = infer(new_model, img_preprocessed)

    old_label = "Melanoma" if old_pred == 1 else "Benigno"
    new_label = "Melanoma" if new_pred == 1 else "Benigno"

    # --------------------------------------------------------
    # Crear imagen comparativa
    # --------------------------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    ax[0].imshow(img_original)
    ax[0].set_title(f"Original\nModelo viejo: {old_label}\nConf: {old_conf:.2f}")
    ax[0].axis("off")

    ax[1].imshow(img_preprocessed)
    ax[1].set_title(f"Preprocesada\nModelo nuevo: {new_label}\nConf: {new_conf:.2f}")
    ax[1].axis("off")

    save_path = os.path.join(SAVE_DIR, f"{base_name}_comparison.png")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"Comparación guardada en:\n{save_path}")


# --------------------------------------------------------
# EJECUCIÓN DESDE TERMINAL
# --------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python compare_single_image.py nombre_imagen_sin_extension")
        sys.exit(1)

    base = sys.argv[1]
    compare_image(base)
