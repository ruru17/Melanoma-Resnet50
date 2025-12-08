import os
import pandas as pd

# Paths
metadata_path = "data/HAM10000_metadata.csv"
images_path = "data/images/"

print("Leyendo metadata completa de 10,000 imágenes...")
df = pd.read_csv(metadata_path)

# Crear columna label:
# mel = melanoma (1)
# todo lo demás = benigno (0)
df["label"] = df["dx"].apply(lambda x: 1 if x == "mel" else 0)

# Verificar qué imágenes existen realmente en la carpeta de imágenes
existing_files = set(os.listdir(images_path))

rows = []
for _, row in df.iterrows():
    image_name = row["image_id"] + ".jpg"
    if image_name in existing_files:   # solo si existe la imagen
        rows.append([row["image_id"], row["label"]])

# Guardar labels finales
df_final = pd.DataFrame(rows, columns=["image_id", "label"])
df_final.to_csv("data/labels_final.csv", index=False)

print("✔ labels_final.csv generado correctamente.")
print("Total de imágenes encontradas:", len(df_final))
print(df_final['label'].value_counts())
