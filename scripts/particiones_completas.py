# ========================================
# PARTICIONES COMPLETAS
# ========================================
# Script unificado para comparación, creación múltiple y análisis estadístico de particiones.
# Mejoras: Unificado en un solo archivo para reducir cantidad de scripts.

import pandas as pd
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

print("\n" + "="*80)
print("PASO 2.1: COMPARANDO RENDIMIENTO DE ALGORITMOS DE ORDENAMIENTO")
print("="*80)

algorithms_to_compare = ['quick_sort']
for algo in algorithms_to_compare:
    print(f"\n--- Probando: {algo.replace('_', ' ').title()} ---")
    df_partition(df, 'y', percentage=0.25, partition_type="best", sorting_algorithm=algo)

print("\n\n" + "="*80)
print("PASO 2.2: CREANDO PARTICIONES CON EL ALGORITMO ELEGIDO (quick_sort)")
print("="*80)

selected_algorithm = 'quick_sort'
B4C, _ = df_partition(df, 'y', 0.25, "best", selected_algorithm)
W4C, _ = df_partition(df, 'y', 0.25, "worst", selected_algorithm)

print("\nPartición de los MEJORES 25% (B4C):")
print(B4C.head())
print("\nPartición de los PEORES 25% (W4C):")
print(W4C.head())

B4C.to_csv('../data/particion_mejores.csv', index=False)
W4C.to_csv('../data/particion_peores.csv', index=False)
print("\nParticiones guardadas en '../data/particion_mejores.csv' y '../data/particion_peores.csv'.")

print("\n" + "="*80)
print("PASO 3.1: CREANDO MÚLTIPLES PARTICIONES CON 'quick_sort'")
print("="*80)

B2C, _ = df_partition(df, 'y', 0.50, "best", selected_algorithm)
W2C, _ = df_partition(df, 'y', 0.50, "worst", selected_algorithm)
B8C, _ = df_partition(df, 'y', 0.125, "best", selected_algorithm)
W8C, _ = df_partition(df, 'y', 0.125, "worst", selected_algorithm)
B16C, _ = df_partition(df, 'y', 0.0625, "best", selected_algorithm)
W16C, _ = df_partition(df, 'y', 0.0625, "worst", selected_algorithm)

print("\nResumen de tamaños de las particiones:")
print(f"B2C (Best 50%): {B2C.shape}")
print(f"W2C (Worst 50%): {W2C.shape}")
print(f"B4C (Best 25%): {B4C.shape}")
print(f"W4C (Worst 25%): {W4C.shape}")
print(f"B8C (Best 12.5%): {B8C.shape}")
print(f"W8C (Worst 12.5%): {W8C.shape}")
print(f"B16C (Best 6.25%): {B16C.shape}")
print(f"W16C (Worst 6.25%): {W16C.shape}")

B2C.to_csv('../data/B2C.csv', index=False)
W2C.to_csv('../data/W2C.csv', index=False)
B8C.to_csv('../data/B8C.csv', index=False)
W8C.to_csv('../data/W8C.csv', index=False)
B16C.to_csv('../data/B16C.csv', index=False)
W16C.to_csv('../data/W16C.csv', index=False)
print("\nTodas las particiones guardadas como archivos CSV en '../data/'.")

print("\n" + "="*80)
print("PASO 3.2: ANÁLISIS ESTADÍSTICO DE LAS PARTICIONES")
print("="*80)

particiones = {
    'df_original': df, 'B2C': B2C, 'W2C': W2C, 'B4C': B4C, 'W4C': W4C,
    'B8C': B8C, 'W8C': W8C, 'B16C': B16C, 'W16C': W16C
}

estadisticas = []
for nombre, partition_df in particiones.items():
    stats = {
        'Partición': nombre,
        'Tamaño': len(partition_df),
        'Min_y': partition_df['y'].min(),
        'Max_y': partition_df['y'].max(),
        'Media_y': partition_df['y'].mean(),
        'Mediana_y': partition_df['y'].median(),
        'Std_Dev_y': partition_df['y'].std()
    }
    estadisticas.append(stats)

df_stats = pd.DataFrame(estadisticas).set_index('Partición')
print(df_stats.round(2))

df_stats.to_csv('../data/estadisticas_particiones.csv')
print("\nEstadísticas guardadas en '../data/estadisticas_particiones.csv'.")

print("Script de particiones completas ejecutado correctamente.")