import os
import pandas as pd

# Rutas
metadata_path = "data/HAM10000_metadata.csv"
images_folder = "data/images"
output_csv = "data/labels.csv"

print("Leyendo metadata...")
df = pd.read_csv(metadata_path)

# Normalizar columnas si vienen con otros nombres
df.columns = [c.lower() for c in df.columns]

# Revisar qué columnas existen
print("Columnas detectadas:", df.columns.tolist())

# Buscar columnas clave
if "image_id" not in df.columns or "dx" not in df.columns:
    raise ValueError("ERROR: El archivo no contiene 'image_id' o 'dx'.")

print("Filtrando imágenes que existen en data/images/...")
image_files = os.listdir(images_folder)
image_files = [f.replace(".jpg", "") for f in image_files if f.lower().endswith(".jpg")]

# Filtrar metadata a imágenes existentes
df_filtered = df[df["image_id"].isin(image_files)].copy()

print(f"Total imágenes encontradas en metadata: {len(df_filtered)}")

# Convertir dx a etiqueta binaria
df_filtered["label"] = df_filtered["dx"].apply(lambda x: 1 if x == "mel" else 0)

# Crear labels.csv
df_filtered[["image_id", "label"]].to_csv(output_csv, index=False)

print(f"✔ labels.csv generado correctamente en: {output_csv}")
print(df_filtered.head())
