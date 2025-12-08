import os
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2

IMG_DIR_ORIGINAL = "data/images"
IMG_DIR_PREP = "data/images_preprocessed"

RESULTS_DIR = "results/preprocessing_analysis"
os.makedirs(RESULTS_DIR, exist_ok=True)

CSV_PATH = "data/labels_final.csv"


def load_image(path):
    return np.array(Image.open(path).convert("RGB"))


def compute_cdf_metrics(gray):
    hist, _ = np.histogram(gray.flatten(), bins=256, range=[0, 256], density=True)
    cdf = hist.cumsum()
    cdf = cdf / cdf[-1]
    std_cdf = cdf.std()
    return cdf, std_cdf


def generate_comparison_figure(image_id, orig, prep, gray_orig, gray_prep, cdf_o, cdf_p):

    fig = plt.figure(figsize=(14, 10))

    # ------------------ 1) Imagen original ------------------
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.imshow(orig)
    ax1.set_title("Imagen Original")
    ax1.axis("off")

    # ------------------ 2) Imagen preprocesada ------------------
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.imshow(prep)
    ax2.set_title("Imagen Preprocesada")
    ax2.axis("off")

    # ------------------ 3) Histograma comparado ------------------
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.hist(gray_orig.flatten(), bins=256, alpha=0.5, label="Original")
    ax3.hist(gray_prep.flatten(), bins=256, alpha=0.5, label="Preprocesada")
    ax3.set_title("Histograma Comparado")
    ax3.legend()

    # ------------------ 4) CDF comparada ------------------
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.plot(cdf_o, label="CDF Original")
    ax4.plot(cdf_p, label="CDF Preprocesada")
    ax4.set_title("CDF Normalizada")
    ax4.legend()

    plt.tight_layout()

    out_path = f"{RESULTS_DIR}/{image_id}_FULL_COMPARISON.png"
    plt.savefig(out_path, dpi=300)
    plt.close()

    return out_path


def analyze_image(image_id):
    orig_path = f"{IMG_DIR_ORIGINAL}/{image_id}.jpg"
    prep_path = f"{IMG_DIR_PREP}/{image_id}.jpg"

    if not os.path.exists(orig_path) or not os.path.exists(prep_path):
        print("Falta imagen:", image_id)
        return None

    orig = load_image(orig_path)
    prep = load_image(prep_path)

    gray_orig = cv2.cvtColor(orig, cv2.COLOR_RGB2GRAY)
    gray_prep = cv2.cvtColor(prep, cv2.COLOR_RGB2GRAY)

    contrast_o = gray_orig.std()
    contrast_p = gray_prep.std()

    bright_o = gray_orig.mean()
    bright_p = gray_prep.mean()

    cdf_o, std_o = compute_cdf_metrics(gray_orig)
    cdf_p, std_p = compute_cdf_metrics(gray_prep)

    final_img = generate_comparison_figure(
        image_id, orig, prep,
        gray_orig, gray_prep,
        cdf_o, cdf_p
    )

    return {
        "image_id": image_id,
        "contrast_original": contrast_o,
        "contrast_preprocessed": contrast_p,
        "brightness_original": bright_o,
        "brightness_preprocessed": bright_p,
        "cdf_std_original": std_o,
        "cdf_std_preprocessed": std_p,
        "comparison_image_path": final_img
    }


def main():
    df = pd.read_csv(CSV_PATH)

    # toma 10 imágenes aleatorias
    subset = df.sample(10, random_state=42)

    results = []

    for _, row in subset.iterrows():
        r = analyze_image(row["image_id"])
        if r:
            results.append(r)

    df_out = pd.DataFrame(results)
    df_out.to_csv(f"{RESULTS_DIR}/preprocessing_report_comparative.csv", index=False)

    print("\nComparaciones generadas en la carpeta:")
    print(RESULTS_DIR)
    print("\nCSV final creado:")
    print("preprocessing_report_comparative.csv")


if __name__ == "__main__":
    main()
