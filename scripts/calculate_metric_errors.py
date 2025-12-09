import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

# =====================================================
# FUNCIONES PARA IMPORTAR DESDE NOTEBOOK O TERMINAL
# =====================================================

def compute_metric_stats(csv_path):
    """
    Lee un CSV con columnas:
        accuracy_per_image
        precision_per_image
        recall_per_image
        f1_per_image
    
    Calcula media y desviación estándar de cada métrica.
    """
    df = pd.read_csv(csv_path)

    metrics = ["accuracy_per_image", "precision_per_image", 
               "recall_per_image", "f1_per_image"]

    stats = {
        "metric": [],
        "mean": [],
        "std": []
    }

    for m in metrics:
        if m not in df.columns:
            raise ValueError(f"ERROR: el archivo no contiene la columna {m}")

        stats["metric"].append(m.replace("_per_image", ""))
        stats["mean"].append(df[m].mean())
        stats["std"].append(df[m].std())

    return pd.DataFrame(stats)


def plot_metric_stats(stats, output_path):
    """
    Genera una gráfica donde cada métrica aparece con su media y desviación estándar.
    """
    plt.figure(figsize=(8, 5))

    x = np.arange(len(stats))
    means = stats["mean"].values
    errors = stats["std"].values

    plt.bar(x, means, yerr=errors, capsize=8)
    plt.xticks(x, stats["metric"].values)
    plt.ylim(0, 1)
    plt.ylabel("Valor")
    plt.title("Comparación de métricas con desviación estándar")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def save_stats(stats, output_csv):
    """Guarda la tabla de estadísticas."""
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    stats.to_csv(output_csv, index=False)


# =====================================================
# EJECUCIÓN DESDE TERMINAL
# =====================================================

if __name__ == "__main__":
    print("Calculando estadísticas de métricas individuales...")

    # Archivo generado previamente al evaluar imagen por imagen
    input_csv = "results/model_precision_by_image.csv"

    # Archivos de salida
    output_csv = "results/metric_stats_all.csv"
    output_plot = "results/metric_stats_all.png"

    # Validar archivo
    if not os.path.exists(input_csv):
        print(f"ERROR: No se encontró {input_csv}")
        print("Ejecuta primero: evaluate_models.py")
        sys.exit(1)

    # Calcular estadísticas
    stats = compute_metric_stats(input_csv)
    print("\nResultados de estadísticas:")
    print(stats)

    # Guardar resultados
    save_stats(stats, output_csv)
    plot_metric_stats(stats, output_plot)

    print("\nArchivos generados:")
    print(f"- {output_csv}")
    print(f"- {output_plot}")
    print("Proceso completado.")
