# ========================================
# VISUALIZACIONES
# ========================================
# Script unificado para generar gráficos de densidad y cajas.
# Mejoras: Unificado en un solo archivo, genera HTML con imágenes embebidas.

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
import time
from algoritmos_ordenamiento import SortingAlgorithms

# Función df_partition incluida
def df_partition(df, criterion, percentage=0.25, partition_type="best",
                 sorting_algorithm="quick_sort"):
    if df.empty:
        raise ValueError("El DataFrame no puede estar vacío.")
    if criterion not in df.columns:
        raise ValueError(f"La columna '{criterion}' no existe en el DataFrame.")
    if not 0 < percentage <= 1:
        raise ValueError("El porcentaje debe estar entre 0 y 1 (exclusivo de 0).")
    if partition_type not in ["best", "worst"]:
        raise ValueError("partition_type debe ser 'best' o 'worst'.")

    sorter = SortingAlgorithms()
    sort_method = getattr(sorter, sorting_algorithm, None)

    if sort_method is None:
        raise ValueError(f"Algoritmo '{sorting_algorithm}' no disponible en SortingAlgorithms.")

    ascending_order = (partition_type == "worst")

    start_time = time.time()

    df_sorted = sort_method(df, criterion, ascending=ascending_order)

    partition_size = int(len(df) * percentage)

    df_result = df_sorted.iloc[:partition_size].copy()

    end_time = time.time()
    execution_time = end_time - start_time

    print(f"Algoritmo usado: {sorting_algorithm}")
    print(f"Tiempo de ejecución: {execution_time:.6f} segundos")
    print(f"Tamaño de la partición '{partition_type}': {len(df_result)} filas")

    return df_result, execution_time

# Cargar el dataset
df = pd.read_csv('../data/dataset_ejemplo.csv')

# Crear particiones
selected_algorithm = 'quick_sort'
B2C, _ = df_partition(df, 'y', 0.50, "best", selected_algorithm)
W2C, _ = df_partition(df, 'y', 0.50, "worst", selected_algorithm)
B4C, _ = df_partition(df, 'y', 0.25, "best", selected_algorithm)
W4C, _ = df_partition(df, 'y', 0.25, "worst", selected_algorithm)
B8C, _ = df_partition(df, 'y', 0.125, "best", selected_algorithm)
W8C, _ = df_partition(df, 'y', 0.125, "worst", selected_algorithm)
B16C, _ = df_partition(df, 'y', 0.0625, "best", selected_algorithm)
W16C, _ = df_partition(df, 'y', 0.0625, "worst", selected_algorithm)

print("\n" + "="*80)
print("PASO 3.3: GRÁFICO DE DENSIDAD DE 'y' POR PARTICIÓN")
print("="*80)

plt.figure(figsize=(18, 8))
sns.kdeplot(data=df['y'], label='Original', fill=True, alpha=0.2, color='grey')
sns.kdeplot(data=B4C['y'], label='B4C (Mejores 25%)', fill=True, alpha=0.5)
sns.kdeplot(data=W4C['y'], label='W4C (Peores 25%)', fill=True, alpha=0.5)
sns.kdeplot(data=B16C['y'], label='B16C (Mejores 6.25%)', fill=True, alpha=0.5)
sns.kdeplot(data=W16C['y'], label='W16C (Peores 6.25%)', fill=True, alpha=0.5)
sns.kdeplot(data=B2C['y'], label='B2C (Mejores 50%)', fill=True, alpha=0.8)
sns.kdeplot(data=W2C['y'], label='W2C (Peores 50%)', fill=True, alpha=0.8)
sns.kdeplot(data=B8C['y'], label='B8C (Mejores 12.50%)', fill=True, alpha=0.3)
sns.kdeplot(data=W8C['y'], label='W8C (Peores 12.50%)', fill=True, alpha=0.3)

plt.title('Distribución de la Variable "y" en Diferentes Particiones', fontsize=16)
plt.xlabel('Valor de y', fontsize=12)
plt.ylabel('Densidad', fontsize=12)
plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.5)

plt.savefig('../results/grafico_densidad.png', dpi=300, bbox_inches='tight')
data_densidad = io.BytesIO()
plt.savefig(data_densidad, format='png', dpi=300, bbox_inches='tight')
densidad_b64 = base64.b64encode(data_densidad.getvalue()).decode()
plt.close()

print("Gráfico de densidad guardado.")

print("\n" + "="*80)
print("PASO 3.4: DIAGRAMA DE CAJAS DE 'y' POR PARTICIÓN")
print("="*80)

data_to_plot = [B2C['y'], W2C['y'], B4C['y'], W4C['y'], B8C['y'], W8C['y'], B16C['y'], W16C['y']]
labels = ['B2C', 'W2C', 'B4C', 'W4C', 'B8C', 'W8C', 'B16C', 'W16C']

plt.figure(figsize=(16, 8))
plt.boxplot(data_to_plot, tick_labels=labels, vert=False, patch_artist=True)
plt.title('Distribución de "y" por Partición', fontsize=16)
plt.xlabel('Valor de y', fontsize=12)
plt.grid(axis='x', linestyle='--', alpha=0.6)

plt.savefig('../results/diagrama_cajas.png', dpi=300, bbox_inches='tight')
data_cajas = io.BytesIO()
plt.savefig(data_cajas, format='png', dpi=300, bbox_inches='tight')
cajas_b64 = base64.b64encode(data_cajas.getvalue()).decode()
plt.close()

print("Diagrama de cajas guardado.")

# Crear HTML con ambas imágenes
html_content = f"""
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <title>Resultados de Visualizaciones</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1 {{ color: #333; }}
        img {{ max-width: 100%; height: auto; margin: 10px 0; }}
    </style>
</head>
<body>
    <h1>Resultados de las Visualizaciones</h1>
    <h2>Gráfico de Densidad</h2>
    <img src="data:image/png;base64,{densidad_b64}" alt="Gráfico de Densidad">
    <h2>Diagrama de Cajas</h2>
    <img src="data:image/png;base64,{cajas_b64}" alt="Diagrama de Cajas">
</body>
</html>
"""

with open('../results/visualizaciones.html', 'w', encoding='utf-8') as f:
    f.write(html_content)

print("HTML de visualizaciones generado en '../results/visualizaciones.html'.")