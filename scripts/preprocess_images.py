import os
import cv2
import numpy as np
import pandas as pd
from skimage import exposure
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm

# ================================================
# CONFIGURACIÓN DE RUTAS
# ================================================
CSV_PATH = "data/labels_final.csv"
IMG_DIR = "data/images"
OUT_DIR = "data/images_preprocessed"
REPORT_PATH = "data/preprocessing_report.csv"

os.makedirs(OUT_DIR, exist_ok=True)

# ================================================
# FUNCIONES DE PREPROCESAMIENTO
# ================================================
def apply_clahe(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    l2 = clahe.apply(l)

    lab2 = cv2.merge((l2, a, b))
    enhanced = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)
    return enhanced

def normalize_intensity(img):
    img_norm = cv2.normalize(img, None, alpha=0, beta=255,
                             norm_type=cv2.NORM_MINMAX)
    return img_norm

# ================================================
# MÉTRICAS DE MEJORA
# ================================================
def entropy(img):
    histogram, _ = np.histogram(img.flatten(), bins=256, range=(0, 255))
    histogram = histogram / np.sum(histogram)
    histogram = histogram[histogram > 0]
    return -np.sum(histogram * np.log2(histogram))

def contrast(img):
    return img.std()

def ssim_measure(before, after):
    before_gray = cv2.cvtColor(before, cv2.COLOR_BGR2GRAY)
    after_gray = cv2.cvtColor(after, cv2.COLOR_BGR2GRAY)
    return ssim(before_gray, after_gray)

# ================================================
# CARGAR LABELS
# ================================================
df_labels = pd.read_csv(CSV_PATH)

# Solo necesitamos: image_id, label
df = df_labels[["image_id", "label"]]

print(f"Procesando {len(df)} imágenes...\n")

rows = []

# ================================================
# PROCESAMIENTO PRINCIPAL
# ================================================
for _, row in tqdm(df.iterrows(), total=len(df), desc="Procesando imágenes"):
    img_id = row["image_id"]
    label = row["label"]  # 0 = benigno, 1 = melanoma

    input_path = os.path.join(IMG_DIR, img_id + ".jpg")
    output_path = os.path.join(OUT_DIR, img_id + ".jpg")

    if not os.path.exists(input_path):
        continue

    img = cv2.imread(input_path)
    if img is None:
        continue

    # Métricas antes
    ent_before = entropy(img)
    cont_before = contrast(img)

    # Aplicar mejoras
    img2 = apply_clahe(img)
    img3 = normalize_intensity(img2)

    # Métricas después
    ent_after = entropy(img3)
    cont_after = contrast(img3)
    sim = ssim_measure(img, img3)

    # Guardar imagen procesada
    cv2.imwrite(output_path, img3)

    # Guardar datos en el reporte
    rows.append([
        img_id,
        int(label),
        ent_before, ent_after,
        cont_before, cont_after,
        sim
    ])

# ================================================
# GUARDAR REPORTE CSV
# ================================================
df_report = pd.DataFrame(rows, columns=[
    "image_id",
    "label",
    "entropy_before", "entropy_after",
    "contrast_before", "contrast_after",
    "ssim_similarity"
])

df_report.to_csv(REPORT_PATH, index=False)

print("\n✔ Preprocesamiento completado.")
print(f"✔ Imágenes procesadas guardadas en: {OUT_DIR}")
print(f"✔ Reporte generado: {REPORT_PATH}")
