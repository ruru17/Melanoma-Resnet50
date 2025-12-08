import os
import requests
from tqdm import tqdm
import time

# ======== CONFIGURACIÓN ========
TOTAL = 1000       # total a descargar
PAGE_SIZE = 100    # imágenes por página
REINTENTOS = 5
# ===============================

os.makedirs("ISIC_subset/images", exist_ok=True)

def descargar(url, ruta):
    """Descarga con reintentos."""
    for intento in range(REINTENTOS):
        try:
            with requests.get(url, stream=True, timeout=20) as r:
                r.raise_for_status()
                with open(ruta, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
            return True
        except Exception as e:
            print(f"Error intento {intento+1}: {e}")
            time.sleep(1)
    return False


def obtener_pagina(offset):
    """Obtiene una página de resultados."""
    url = f"https://api.isic-archive.com/api/v2/images?limit={PAGE_SIZE}&offset={offset}"
    res = requests.get(url)
    res.raise_for_status()
    return res.json().get("results", [])


descargadas = 0
offset = 0

print(f"Iniciando descarga de {TOTAL} imágenes...")

while descargadas < TOTAL:
    items = obtener_pagina(offset)

    if not items:
        print("⚠ No hay más imágenes disponibles en esta API.")
        break

    print(f"\nPágina offset {offset}, recibidas: {len(items)}")

    for item in tqdm(items, desc=f"Descargando pág {offset}", leave=False):

        # Validar que exista archivo "full"
        if "files" not in item or "full" not in item["files"]:
            continue

        img_id = item["isic_id"]
        img_url = item["files"]["full"]["url"]
        ruta = f"ISIC_subset/images/{img_id}.jpg"

        ok = descargar(img_url, ruta)

        if ok:
            descargadas += 1

        if descargadas >= TOTAL:
            break

    offset += PAGE_SIZE  # pasar a la siguiente página

print(f"\n✔ Descarga completada. Total descargadas: {descargadas}")
