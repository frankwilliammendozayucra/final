import traceback
import pandas as pd

def extraer_nodos_discrepantes():

    print("\n" + "="*80)
    print("INICIANDO SCRIPT 9: EXTRACCIÓN DE NODOS CON DISCREPANCIAS")
    print("="*80)

    # 1. Verificar si 'todas_las_diferencias' existe
    try:
        # La variable debe existir globalmente de la celda anterior
        if 'todas_las_diferencias' not in locals() and 'todas_las_diferencias' not in globals():
             print("\n⚠️ ERROR: La variable 'todas_las_diferencias' no se encontró.")
             print("Por favor, asegúrate de ejecutar la 'ETAPA 4' (comparación de árboles) primero.")
             return

        global todas_las_diferencias

        if not todas_las_diferencias:
             print("\n⚠️ ADVERTENCIA: La variable 'todas_las_diferencias' está vacía.")
             print("   (No se encontraron diferencias en la Etapa 4).")
             return

        print(f"\nSe encontraron {len(todas_las_diferencias)} comparaciones en 'todas_las_diferencias'.")

        # Diccionario para guardar los arrays
        nodos_discrepantes_por_par = {}

        # 2. Iterar sobre cada par de comparación
        for comparacion, df_diff in todas_las_diferencias.items():

            print(f"\n{'='*70}")
            print(f"PROCESANDO COMPARACIÓN: {comparacion.upper()}")
            print(f"{'='*70}")

            if not isinstance(df_diff, pd.DataFrame) or 'Nodo' not in df_diff.columns:
                print(f"  ⚠️  Datos para '{comparacion}' no son un DataFrame válido o falta la columna 'Nodo'. Saltando.")
                continue

            # --- Tarea Principal ---
            # 1. Obtener la columna 'Nodo'
            # 2. Usar .unique() para obtener cada nodo una sola vez
            # 3. Convertir a una lista (array)
            nodos_con_discrepancia = df_diff['Nodo'].unique().tolist()

            # 4. Ordenar alfabéticamente para una salida limpia
            nodos_con_discrepancia.sort()

            # 5. Guardar el resultado
            nodos_discrepantes_por_par[comparacion] = nodos_con_discrepancia

            # 6. Imprimir el array
            print(f"  Total de nodos con discrepancias: {len(nodos_con_discrepancia)}")
            print(f"  ARRAY DE NODOS:")
            print(f"  {nodos_con_discrepancia}")

        print("\n" + "="*80)
        print("EXTRACCIÓN DE NODOS COMPLETADA")
        print("="*80)

        # Puedes usar 'nodos_discrepantes_por_par' en celdas futuras
        # return nodos_discrepantes_por_par

    except NameError:
        print("\n❌ ERROR CRÍTICO: La variable 'todas_las_diferencias' no está definida.")
        print("Por favor, asegúrate de ejecutar la 'ETAPA 4' primero.")
    except Exception as e:
        print(f"\n❌ Ocurrió un error inesperado: {e}")
        traceback.print_exc()

# --- Ejecutar la función ---
extraer_nodos_discrepantes()