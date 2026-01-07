# ========================================
# FUNCIÓN DE PARTICIONAMIENTO MEJORADA
# ========================================
# Función para segmentar un DataFrame según un criterio usando algoritmos de ordenamiento.
# Mejoras: Añadidas validaciones adicionales, comentarios detallados y manejo de errores.

import time
from algoritmos_ordenamiento import SortingAlgorithms  # Importar la clase de algoritmos

def df_partition(df, criterion, percentage=0.25, partition_type="best",
                 sorting_algorithm="quick_sort"):
    """
    Segmenta un DataFrame según un criterio usando un algoritmo de ordenamiento específico.

    Parámetros:
    - df: DataFrame de entrada
    - criterion: Columna para ordenar (debe existir en df)
    - percentage: Porcentaje del segmento (0.0 a 1.0, exclusivo de 0)
    - partition_type: "best" (valores más altos) o "worst" (valores más bajos)
    - sorting_algorithm: Nombre del algoritmo a usar de la clase SortingAlgorithms

    Retorna:
    - df_result: DataFrame particionado
    - execution_time: Tiempo de ejecución en segundos
    """
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

    # Para "best" (más altos): orden descendente (ascending=False)
    # Para "worst" (más bajos): orden ascendente (ascending=True)
    ascending_order = (partition_type == "worst")

    start_time = time.time()

    # Ordenar según el criterio
    df_sorted = sort_method(df, criterion, ascending=ascending_order)

    partition_size = int(len(df) * percentage)

    # Tomar el primer 'partition_size' de filas
    df_result = df_sorted.iloc[:partition_size].copy()

    end_time = time.time()
    execution_time = end_time - start_time

    print(f"Algoritmo usado: {sorting_algorithm}")
    print(f"Tiempo de ejecución: {execution_time:.6f} segundos")
    print(f"Tamaño de la partición '{partition_type}': {len(df_result)} filas")

    return df_result, execution_time

# Ejemplo de uso (comentado)
# import pandas as pd
# df = pd.DataFrame({'valores': [3, 1, 4, 1, 5]})
# result, time_taken = df_partition(df, 'valores', percentage=0.5, partition_type='best')
# print(result)

print("Función df_partition definida correctamente.")