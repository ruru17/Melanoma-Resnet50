import os
import csv
import requests
from tqdm import tqdm

# Ruta donde guardaste tus imágenes descargadas
IMAGES_DIR = "ISIC_subset/images"

# Archivo CSV de salida
OUTPUT_CSV = "data/labels.csv"

# Diccionario de diagnósticos aceptados como melanoma
MELANOMA_LABELS = ["melanoma", "invasive melanoma", "melanoma in situ"]

def get_diagnosis(image_id):
    url = f"https://api.isic-archive.com/api/v2/images/{image_id}"
    
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        diagnosis = data.get("meta", {}).get("clinical", {}).get("diagnosis", None)
        return diagnosis
    
    except Exception as e:
        print(f"Error obteniendo metadata de {image_id}: {e}")
        return None


def main():
    images = [f for f in os.listdir(IMAGES_DIR) if f.endswith(".jpg")]

    os.makedirs("data", exist_ok=True)

    with open(OUTPUT_CSV, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["image", "label"])

        for img in tqdm(images, desc="Consultando etiqueta"):
            image_id = img.replace(".jpg", "")

            diagnosis = get_diagnosis(image_id)

            if diagnosis is None:
                print(f"No se encontró diagnóstico para {img}. Se excluirá.")
                continue

            label = 1 if diagnosis.lower() in MELANOMA_LABELS else 0

            writer.writerow([img, label])

    print("\nArchivo labels.csv generado correctamente:")
    print(f"- Ruta: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
