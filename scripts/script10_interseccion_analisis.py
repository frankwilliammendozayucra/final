import traceback
import pandas as pd
from collections import deque # Necesario para la lógica de BFS

# --- PASO 1: Lógica de Script 8 (Unión de Recorridos) ---
# Necesitamos la función auxiliar de Script 8
def _get_union_for_partition(partition_name, results_dict):
    """
    Función auxiliar para obtener la unión BFS/DFS de UNA partición.
    (Versión silenciosa, solo devuelve el set)
    """
    if partition_name not in results_dict:
        return set() # Devuelve un set vacío

    data = results_dict[partition_name]
    bfs_list = data.get('bfs')
    dfs_list = data.get('dfs')

    if bfs_list is None or dfs_list is None:
        return set() # Devuelve un set vacío

    union_set = set(bfs_list).union(set(dfs_list))
    return union_set

def generar_uniones_de_recorridos(resultados_arboles):
    """
    Ejecuta la lógica de Script 8 para generar el dict de uniones.
    """
    print("\n--- PASO 1: Generando 'Array 1' (Unión de Recorridos)... ---")

    pares_a_comparar = [
        ('B2C', 'W2C'), ('B4C', 'W4C'),
        ('B8C', 'W8C'), ('B16C', 'W16C')
    ]
    uniones_finales = {}

    # Procesar Pares
    for nombre_b, nombre_w in pares_a_comparar:
        pair_name = nombre_b[1:] # Ej: "2C"
        set_b = _get_union_for_partition(nombre_b, resultados_arboles)
        set_w = _get_union_for_partition(nombre_w, resultados_arboles)

        # Calcular la UNIÓN TOTAL entre B y W
        total_union_set = set_b.union(set_w)
        uniones_finales[pair_name] = total_union_set

    # Procesar 'df_original'
    set_orig = _get_union_for_partition('df_original', resultados_arboles)
    uniones_finales['df_original'] = set_orig

    print("✓ 'Array 1' (Recorridos) generado.")
    return uniones_finales

# --- PASO 2: Lógica de Script 9 (Nodos Discrepantes) ---
def generar_nodos_discrepantes(todas_las_diferencias):
    """
    Ejecuta la lógica de Script 9 para generar el dict de discrepancias.
    """
    print("\n--- PASO 2: Generando 'Array 2' (Nodos con Discrepancias)... ---")

    nodos_discrepantes_por_par = {}

    for comparacion, df_diff in todas_las_diferencias.items():
        if not isinstance(df_diff, pd.DataFrame) or 'Nodo' not in df_diff.columns:
            continue

        # Obtener nodos únicos y guardarlos como un set
        nodos_con_discrepancia_set = set(df_diff['Nodo'].unique())
        nodos_discrepantes_por_par[comparacion] = nodos_con_discrepancia_set

    print("✓ 'Array 2' (Discrepancias) generado.")
    return nodos_discrepantes_por_par

# --- PASO 3: INTERSECCIÓN (¡Nuevo!) ---
def encontrar_interseccion_final(uniones_finales, nodos_discrepantes):
    """
    Compara los dos diccionarios y encuentra la intersección.
    """
    print("\n" + "="*80)
    print("PASO 3: REALIZANDO INTERSECCIÓN DE RESULTADOS")
    print("="*80)

    # Mapa para alinear los keys de los dos diccionarios
    # Llave = (Key de Discrepancias), Valor = (Key de Recorridos)
    key_map = {
        'B2C vs W2C': '2C',
        'B4C vs W4C': '4C',
        'B8C vs W8C': '8C',
        'B16C vs W16C': '16C'
        # df_original no tiene discrepancias, se ignora
    }

    resultados_de_interseccion = {}

    # Iteramos sobre los resultados de las Discrepancias (Array 2)
    for comparacion_key, set_discrepantes in nodos_discrepantes.items():

        print(f"\n{'='*70}")
        print(f"COMPARACIÓN: {comparacion_key.upper()}")
        print(f"{'='*70}")

        # 1. Encontrar la key de Recorridos (Array 1)
        if comparacion_key not in key_map:
            print(f"  - No se encontró un par para '{comparacion_key}' en el mapa. Saltando.")
            continue

        recorrido_key = key_map[comparacion_key] # Ej: '2C'

        # 2. Obtener los dos sets
        set_recorridos = uniones_finales.get(recorrido_key)

        if set_recorridos is None:
            print(f"  - No se encontraron datos de recorrido para '{recorrido_key}'. Saltando.")
            continue

        # 3. Imprimir las dos listas de entrada
        print(f"  Array 1 (Recorridos {recorrido_key}):")
        print(f"    {sorted(list(set_recorridos))}")

        print(f"\n  Array 2 (Discrepancias {comparacion_key}):")
        print(f"    {sorted(list(set_discrepantes))}")

        # 4. Calcular la INTERSECCIÓN
        interseccion = set_recorridos.intersection(set_discrepantes)
        lista_interseccion = sorted(list(interseccion))

        resultados_de_interseccion[comparacion_key] = lista_interseccion

        # 5. Imprimir el resultado final
        print(f"\n  --- INTERSECCIÓN FINAL ({len(lista_interseccion)} nodos) ---")
        print(f"  (Nodos que son CERCANOS a 'y' Y ESTRUCTURALMENTE INESTABLES)")
        print(f"  {lista_interseccion}")

    return resultados_de_interseccion

# --- Función Principal ---
def ejecutar_analisis_de_interseccion():

    print("\n" + "="*80)
    print("INICIANDO SCRIPT 10: ANÁLISIS DE INTERSECCIÓN")
    print("="*80)

    # 1. Verificar dependencias
    try:
        if ('resultados_arboles' not in locals() and 'resultados_arboles' not in globals()) or \
           ('todas_las_diferencias' not in locals() and 'todas_las_diferencias' not in globals()):
             print("\n⚠️ ERROR: Faltan variables de scripts anteriores.")
             print("Por favor, asegúrate de haber ejecutado:")
             print("  1. 'PASO 6' (para generar 'resultados_arboles')")
             print("  2. 'ETAPA 4' (para generar 'todas_las_diferencias')")
             return

        global resultados_arboles, todas_las_diferencias

        # --- PASO 1 ---
        # (Lógica de Script 8)
        uniones_finales = generar_uniones_de_recorridos(resultados_arboles)

        # --- PASO 2 ---
        # (Lógica de Script 9)
        nodos_discrepantes = generar_nodos_discrepantes(todas_las_diferencias)

        # --- PASO 3 ---
        # (Nueva Lógica de Intersección)
        resultados_finales = encontrar_interseccion_final(uniones_finales, nodos_discrepantes)

        print("\n" + "="*80)
        print("SCRIPT 10: ANÁLISIS DE INTERSECCIÓN COMPLETADO")
        print("="*80)

        # 'resultados_finales' está disponible para su uso
        # return resultados_finales

    except NameError as e:
        print(f"\n❌ ERROR CRÍTICO: Faltan variables: {e}")
        print("Por favor, asegúrate de ejecutar las celdas 'PASO 6' y 'ETAPA 4' primero.")
    except Exception as e:
        print(f"\n❌ Ocurrió un error inesperado: {e}")
        traceback.print_exc()

# --- Ejecutar la función ---
ejecutar_analisis_de_interseccion()