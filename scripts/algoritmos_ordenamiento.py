# ========================================
# ALGORITMOS DE ORDENAMIENTO
# ========================================
# Clase que implementa 10 algoritmos de ordenamiento para DataFrames de pandas.
# Mejoras: Añadidos comentarios detallados, validaciones básicas y manejo de errores.

import pandas as pd
import numpy as np

class SortingAlgorithms:
    """Clase que implementa 10 algoritmos de ordenamiento para DataFrames."""

    @staticmethod
    def bubble_sort(df, column, ascending=True):
        """Bubble Sort - O(n²) - Intercambia elementos adyacentes si están en orden incorrecto."""
        if df.empty or column not in df.columns:
            raise ValueError("DataFrame vacío o columna no encontrada.")
        df_copy = df.copy().reset_index(drop=True)
        n = len(df_copy)
        for i in range(n):
            for j in range(0, n-i-1):
                if ascending:
                    if df_copy.loc[j, column] > df_copy.loc[j+1, column]:
                        df_copy.iloc[j], df_copy.iloc[j+1] = df_copy.iloc[j+1].copy(), df_copy.iloc[j].copy()
                else:  # Descending
                    if df_copy.loc[j, column] < df_copy.loc[j+1, column]:
                        df_copy.iloc[j], df_copy.iloc[j+1] = df_copy.iloc[j+1].copy(), df_copy.iloc[j].copy()
        return df_copy

    @staticmethod
    def selection_sort(df, column, ascending=True):
        """Selection Sort - O(n²) - Selecciona el mínimo/máximo y lo coloca en su posición."""
        if df.empty or column not in df.columns:
            raise ValueError("DataFrame vacío o columna no encontrada.")
        df_copy = df.copy().reset_index(drop=True)
        n = len(df_copy)
        for i in range(n):
            idx = i
            for j in range(i+1, n):
                if ascending:
                    if df_copy.loc[j, column] < df_copy.loc[idx, column]:
                        idx = j
                else:  # Descending
                    if df_copy.loc[j, column] > df_copy.loc[idx, column]:
                        idx = j
            df_copy.iloc[i], df_copy.iloc[idx] = df_copy.iloc[idx].copy(), df_copy.iloc[i].copy()
        return df_copy

    @staticmethod
    def insertion_sort(df, column, ascending=True):
        """Insertion Sort - O(n²) - Inserta cada elemento en su posición correcta."""
        if df.empty or column not in df.columns:
            raise ValueError("DataFrame vacío o columna no encontrada.")
        df_copy = df.copy().reset_index(drop=True)
        for i in range(1, len(df_copy)):
            key_row = df_copy.iloc[i].copy()
            key_value = df_copy.loc[i, column]
            j = i - 1
            if ascending:
                while j >= 0 and df_copy.loc[j, column] > key_value:
                    df_copy.iloc[j + 1] = df_copy.iloc[j].copy()
                    j -= 1
            else:  # Descending
                while j >= 0 and df_copy.loc[j, column] < key_value:
                    df_copy.iloc[j + 1] = df_copy.iloc[j].copy()
                    j -= 1
            df_copy.iloc[j + 1] = key_row
        return df_copy

    @staticmethod
    def merge_sort(df, column, ascending=True):
        """Merge Sort - O(n log n) - Divide y conquista."""
        if df.empty or column not in df.columns:
            raise ValueError("DataFrame vacío o columna no encontrada.")
        if len(df) <= 1:
            return df.copy()

        mid = len(df) // 2
        left_half = df.iloc[:mid]
        right_half = df.iloc[mid:]

        left = SortingAlgorithms.merge_sort(left_half, column, ascending)
        right = SortingAlgorithms.merge_sort(right_half, column, ascending)

        return SortingAlgorithms._merge(left, right, column, ascending)

    @staticmethod
    def _merge(left, right, column, ascending):
        """Función auxiliar para merge sort."""
        result = []
        i = j = 0
        left_records = left.to_dict('records')
        right_records = right.to_dict('records')

        while i < len(left_records) and j < len(right_records):
            if ascending:
                if left_records[i][column] <= right_records[j][column]:
                    result.append(left_records[i])
                    i += 1
                else:
                    result.append(right_records[j])
                    j += 1
            else:  # Descending
                if left_records[i][column] >= right_records[j][column]:
                    result.append(left_records[i])
                    i += 1
                else:
                    result.append(right_records[j])
                    j += 1

        result.extend(left_records[i:])
        result.extend(right_records[j:])

        return pd.DataFrame(result)

    @staticmethod
    def quick_sort(df, column, ascending=True):
        """Quick Sort - O(n log n) promedio - Usa pivote para dividir."""
        if column not in df.columns:
            raise ValueError("La columna no existe en el DataFrame.")
        if len(df) <= 1:
            return df.copy()

        pivot_row = df.iloc[len(df) // 2]
        pivot_value = pivot_row[column]

        if ascending:
            left = df[df[column] < pivot_value]
            middle = df[df[column] == pivot_value]
            right = df[df[column] > pivot_value]
        else:  # Descending
            left = df[df[column] > pivot_value]
            middle = df[df[column] == pivot_value]
            right = df[df[column] < pivot_value]

        return pd.concat([
            SortingAlgorithms.quick_sort(left, column, ascending),
            middle,
            SortingAlgorithms.quick_sort(right, column, ascending)
        ]).reset_index(drop=True)

    @staticmethod
    def heap_sort(df, column, ascending=True):
        """Heap Sort - O(n log n) - Usa estructura de heap."""
        if df.empty or column not in df.columns:
            raise ValueError("DataFrame vacío o columna no encontrada.")
        df_copy = df.copy().reset_index(drop=True)
        n = len(df_copy)

        def heapify(df_local, n_local, i_local):
            largest_or_smallest = i_local
            left = 2 * i_local + 1
            right = 2 * i_local + 2

            if ascending:
                if left < n_local and df_local.loc[i_local, column] < df_local.loc[left, column]:
                    largest_or_smallest = left
                if right < n_local and df_local.loc[largest_or_smallest, column] < df_local.loc[right, column]:
                    largest_or_smallest = right
            else:  # Descending
                if left < n_local and df_local.loc[i_local, column] > df_local.loc[left, column]:
                    largest_or_smallest = left
                if right < n_local and df_local.loc[largest_or_smallest, column] > df_local.loc[right, column]:
                    largest_or_smallest = right

            if largest_or_smallest != i_local:
                df_local.iloc[i_local], df_local.iloc[largest_or_smallest] = df_local.iloc[largest_or_smallest].copy(), df_local.iloc[i_local].copy()
                heapify(df_local, n_local, largest_or_smallest)

        for i in range(n // 2 - 1, -1, -1):
            heapify(df_copy, n, i)

        for i in range(n - 1, 0, -1):
            df_copy.iloc[i], df_copy.iloc[0] = df_copy.iloc[0].copy(), df_copy.iloc[i].copy()
            heapify(df_copy, i, 0)

        return df_copy

    @staticmethod
    def shell_sort(df, column, ascending=True):
        """Shell Sort - Complejidad variable, mejor que O(n²) - Usa gaps."""
        if df.empty or column not in df.columns:
            raise ValueError("DataFrame vacío o columna no encontrada.")
        df_copy = df.copy().reset_index(drop=True)
        n = len(df_copy)
        gap = n // 2
        while gap > 0:
            for i in range(gap, n):
                temp_row = df_copy.iloc[i].copy()
                temp_val = temp_row[column]
                j = i
                if ascending:
                    while j >= gap and df_copy.loc[j - gap, column] > temp_val:
                        df_copy.iloc[j] = df_copy.iloc[j - gap].copy()
                        j -= gap
                else:  # Descending
                    while j >= gap and df_copy.loc[j - gap, column] < temp_val:
                        df_copy.iloc[j] = df_copy.iloc[j - gap].copy()
                        j -= gap
                df_copy.iloc[j] = temp_row
            gap //= 2
        return df_copy

    @staticmethod
    def counting_sort(df, column, ascending=True):
        """Counting Sort - O(n + k) - Solo para enteros no negativos."""
        if df.empty or column not in df.columns:
            raise ValueError("DataFrame vacío o columna no encontrada.")
        if not pd.api.types.is_integer_dtype(df[column]) or (df[column] < 0).any():
            raise ValueError("Counting Sort solo funciona con enteros no negativos.")

        df_copy = df.copy()
        max_val = df_copy[column].max()
        count = np.zeros(max_val + 1, dtype=int)

        for val in df_copy[column]:
            count[val] += 1

        for i in range(1, len(count)):
            count[i] += count[i-1]

        output = [None] * len(df_copy)
        for i in range(len(df_copy) - 1, -1, -1):
            row = df_copy.iloc[i]
            val = row[column]
            output[count[val] - 1] = row
            count[val] -= 1

        result_df = pd.DataFrame(output)
        if not ascending:
            result_df = result_df.iloc[::-1]

        return result_df.reset_index(drop=True)

    @staticmethod
    def radix_sort(df, column, ascending=True):
        """Radix Sort - O(d * (n + k)) - Solo para enteros no negativos."""
        if df.empty or column not in df.columns:
            raise ValueError("DataFrame vacío o columna no encontrada.")
        if not pd.api.types.is_integer_dtype(df[column]) or (df[column] < 0).any():
            raise ValueError("Radix Sort solo funciona con enteros no negativos.")

        df_copy = df.copy()
        max_val = df_copy[column].max()
        exp = 1
        while max_val // exp > 0:
            df_copy = SortingAlgorithms._counting_sort_for_radix(df_copy, column, exp)
            exp *= 10

        if not ascending:
            df_copy = df_copy.iloc[::-1]

        return df_copy.reset_index(drop=True)

    @staticmethod
    def _counting_sort_for_radix(df, column, exp):
        n = len(df)
        output = [None] * n
        count = [0] * 10

        for i in range(n):
            index = df.loc[i, column] // exp
            count[index % 10] += 1

        for i in range(1, 10):
            count[i] += count[i - 1]

        i = n - 1
        while i >= 0:
            index = df.loc[i, column] // exp
            output[count[index % 10] - 1] = df.iloc[i].to_dict()
            count[index % 10] -= 1
            i -= 1

        return pd.DataFrame(output)

    @staticmethod
    def tim_sort(df, column, ascending=True):
        """Tim Sort (implementación de Pandas) - O(n log n) - Usa mergesort estable."""
        if df.empty or column not in df.columns:
            raise ValueError("DataFrame vacío o columna no encontrada.")
        # kind='mergesort' o 'stable' es similar a Timsort
        return df.sort_values(by=column, ascending=ascending, kind='stable').reset_index(drop=True)

# Ejemplo de uso (comentado para no ejecutar automáticamente)
# df = pd.DataFrame({'valores': [3, 1, 4, 1, 5]})
# sorted_df = SortingAlgorithms.bubble_sort(df, 'valores')
# print(sorted_df)

print("Clase SortingAlgorithms definida correctamente.")