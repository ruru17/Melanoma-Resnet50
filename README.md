Melanoma Classification with ResNet50 & Enhanced Preprocessing

Este repositorio contiene un sistema completo para clasificación de melanoma usando ResNet50, entrenado primero con imágenes originales y posteriormente con imágenes preprocesadas (CLAHE + Iluminación + Normalización) para mejorar el rendimiento en pieles oscuras.

Incluye:

Entrenamiento del modelo original y del modelo mejorado

Preprocesamiento avanzado de las 10,000 imágenes de HAM10000

Comparación entre el modelo viejo y el nuevo

Evaluación cuantitativa y visual

Predicción de imágenes nuevas (incluso fuera de la base de datos)

Scripts organizados y notebook demostrativo

Estructura del Proyecto
melanoma-resnet/
│
├── data/
│   ├── images/                     # 10,000 imágenes originales HAM10000
│   ├── images_preprocessed/        # 10,000 imágenes mejoradas
│   ├── HAM10000_metadata.csv       # Metadata oficial
│   ├── labels_final.csv            # IDs + etiquetas (0 benigno, 1 melanoma)
│   └── preprocessing_report.csv    # Métricas de mejora por imagen
│
├── models/
│   ├── resnet50_melanoma_best.pth          # Modelo entrenado con imágenes originales
│   └── resnet50_preprocessed_best.pth      # Modelo entrenado con imágenes preprocesadas
│
├── results/
│   ├── evaluation/                 # Métricas viejo vs nuevo
│   ├── preprocessing_vis/          # Comparaciones visuales
│   └── single_image_eval/          # Comparación de una imagen individual
│
├── notebooks/
│   └── melanoma_pipeline.ipynb     # Notebook principal
│
├── scripts/
│   ├── preprocess_images.py
│   ├── train_resnet_advanced.py
│   ├── evaluate_models.py
│   ├── visualize_preprocessing.py
│   └── compare_single_image.py
│
├── README.md
└── requirements.txt

Objetivo del Proyecto

Desarrollar un sistema capaz de:

Clasificar lesiones como melanoma (1) o benignas (0).

Mejorar la robustez del modelo en imágenes de pieles oscuras mediante:

CLAHE

Corrección Gamma

Normalización de iluminación

Comparar objetivamente:

modelo viejo vs modelo preprocesado

mejoras en contraste, histograma, entropía, CDF, y calidad visual.

Requisitos de instalación

Ejecutar:

pip install -r requirements.txt

1. Preprocesar las imágenes

Este script genera:

Nuevas imágenes mejoradas

Reporte de mejora por imagen

python scripts/preprocess_images.py


Salida en:

data/images_preprocessed/
data/preprocessing_report.csv

2. Entrenar el modelo preprocesado
python scripts/train_resnet_advanced.py


El mejor modelo se guarda como:

models/resnet50_preprocessed_best.pth

3. Comparación entre modelos
python scripts/evaluate_models.py


Genera:

results/evaluation/model_metrics_comparison.csv
results/evaluation/confusion_matrix_old.png
results/evaluation/confusion_matrix_new.png


Incluye accuracy, F1, precision y recall.

4. Visualizar mejoría del preprocesamiento
python scripts/visualize_preprocessing.py nombre_imagen


Salida:

results/preprocessing_vis/nombre_imagen_comparison.png


Incluye:

Imagen original vs preprocesada

Histogramas

Entropía comparativa

5. Evaluar una imagen individual (modelo viejo vs modelo nuevo)
python scripts/compare_single_image.py nombre_sin_extension


Ejemplo:

python scripts/compare_single_image.py foto1


Debe existir en:

data/images_indv/


Salida en:

results/single_image_eval/foto1_comparison.png


Contenido:

Imagen original y preprocesada

Predicción del modelo original

Predicción del modelo preprocesado


Notas importantes

Las etiquetas provienen de HAM10000_metadata.csv, usando dx como etiqueta.

Las clases están balanceadas durante el entrenamiento usando WeightedRandomSampler.

Las imágenes externas se procesan automáticamente antes de ser clasificadas.

Licencia

Proyecto para fines académicos y de investigación. No debe usarse como diagnóstico médico.

