import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

# Directorios
ORIG_DIR = "data/images"
PRE_DIR = "data/images_preprocessed"
OUT_DIR = "results/improvement_report"

os.makedirs(OUT_DIR, exist_ok=True)

def compute_metrics(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    brightness = np.mean(gray)
    contrast = np.std(gray)
    entropy = -np.sum((gray/255.0) * np.log2((gray/255.0) + 1e-10))

    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist = hist / hist.sum()
    hist_var = np.var(hist)

    noise_est = np.mean(cv2.GaussianBlur(gray, (5,5), 0) - gray)

    return brightness, contrast, entropy, hist_var, noise_est

results = []

all_images = sorted(os.listdir(ORIG_DIR))

print(f"Analizando {len(all_images)} imágenes...")

for fname in tqdm(all_images):
    if not fname.lower().endswith(".jpg"):
        continue

    orig_path = os.path.join(ORIG_DIR, fname)
    pre_path  = os.path.join(PRE_DIR, fname)

    if not os.path.exists(pre_path):
        continue

    orig = cv2.imread(orig_path)
    pre  = cv2.imread(pre_path)

    b_o, c_o, e_o, v_o, n_o = compute_metrics(orig)
    b_p, c_p, e_p, v_p, n_p = compute_metrics(pre)

    results.append([
        fname,
        b_o, b_p, b_p - b_o,
        c_o, c_p, c_p - c_o,
        e_o, e_p, e_p - e_o,
        v_o, v_p, v_p - v_o,
        n_o, n_p, n_p - n_o
    ])

columns = [
    "image",
    "brightness_orig", "brightness_pre", "brightness_diff",
    "contrast_orig", "contrast_pre", "contrast_diff",
    "entropy_orig", "entropy_pre", "entropy_diff",
    "hist_var_orig", "hist_var_pre", "hist_var_diff",
    "noise_orig", "noise_pre", "noise_diff"
]

df = pd.DataFrame(results, columns=columns)

csv_path = os.path.join(OUT_DIR, "improvement_metrics.csv")
df.to_csv(csv_path, index=False)
print(f"CSV guardado en: {csv_path}")

# Summary
summary = df.describe()
summary_path = os.path.join(OUT_DIR, "summary_statistics.txt")
with open(summary_path, "w") as f:
    f.write("===== RESUMEN ESTADÍSTICO =====\n\n")
    f.write(str(summary))
print(f"Resumen guardado en: {summary_path}")

# Generate simple comparison plots
plt.figure(figsize=(10,5))
plt.hist(df["contrast_orig"], bins=50, alpha=0.5, label="Original")
plt.hist(df["contrast_pre"], bins=50, alpha=0.5, label="Preprocesada")
plt.legend()
plt.title("Distribución del contraste")
plt.savefig(os.path.join(OUT_DIR, "contrast_distribution.png"))
plt.close()

plt.figure(figsize=(10,5))
plt.hist(df["entropy_orig"], bins=50, alpha=0.5, label="Original")
plt.hist(df["entropy_pre"], bins=50, alpha=0.5, label="Preprocesada")
plt.legend()
plt.title("Distribución de entropía")
plt.savefig(os.path.join(OUT_DIR, "entropy_distribution.png"))
plt.close()

print("Gráficas guardadas en carpeta de reporte.")
