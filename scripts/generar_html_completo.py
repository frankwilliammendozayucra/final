import base64
import io
import sys
import pandas as pd
import os

# Importar módulos del proyecto
sys.path.append('.')
from algoritmo_newman import ejecutar_newman_todas_particiones
from arboles_enraizados import *  # Importar todo para acceder a funciones y variables
from visualizacion_grafos_mst_caminos import *  # Importar visualización de grafos con MST y caminos
from visualizacion_prim_podado import *  # Importar visualización de MST Prim podado

def get_base64(image_path):
    with open(image_path, 'rb') as f:
        return base64.b64encode(f.read()).decode()

def capture_output(func, *args, **kwargs):
    """Captura la salida de print de una función"""
    old_stdout = sys.stdout
    sys.stdout = captured_output = io.StringIO()
    try:
        result = func(*args, **kwargs)
        output = captured_output.getvalue()
        return output, result
    finally:
        sys.stdout = old_stdout

# Cargar datos (asumiendo que ya existen)
df_original = pd.read_csv('../data/dataset_ejemplo.csv')

# Cargar particiones desde archivos CSV
particiones = {}
particion_names = ['B2C', 'W2C', 'B4C', 'W4C', 'B8C', 'W8C', 'B16C', 'W16C']
for name in particion_names:
    path = f'../data/{name}.csv'
    if os.path.exists(path):
        particiones[name] = pd.read_csv(path)

# Ejecutar Newman y capturar salida en todas las particiones
particiones_para_newman = [
    ('df_original', df_original),
] + [(name, df) for name, df in particiones.items()]

newman_output, newman_result = capture_output(ejecutar_newman_todas_particiones, particiones_para_newman)

# Ejecutar análisis de árboles enraizados (correr el script directamente)
print("Ejecutando análisis de árboles enraizados...")
exec(open('arboles_enraizados.py', encoding='utf-8').read())
trees_output = "Análisis de árboles enraizados completado. Ver imágenes generadas."

# Ejecutar visualización de grafos con MST y caminos destacados
print("Ejecutando visualización de grafos con MST y caminos destacados...")
exec(open('visualizacion_grafos_mst_caminos.py', encoding='utf-8').read())
mst_paths_output = "Visualización de grafos con MST y caminos destacados completada. Ver imágenes generadas."

# Ejecutar visualización de MST Prim podado
print("Ejecutando visualización de MST Prim podado...")
exec(open('visualizacion_prim_podado.py', encoding='utf-8').read())
prim_podado_output = "Visualización de MST Prim podado completada. Ver imágenes generadas."

# Ejecutar unión de recorridos por pares
print("Ejecutando unión de recorridos BFS/DFS por pares...")
exec(open('unir_recorridos_por_pares.py', encoding='utf-8').read())
union_output = "Unión de recorridos por pares completada. Ver CSV y imágenes generadas."

# Ejecutar análisis de intersección
print("Ejecutando análisis de intersección...")
exec(open('script10_interseccion_analisis.py', encoding='utf-8').read())
interseccion_output = "Análisis de intersección completado. Ver resultados en consola."

# Cargar imágenes de árboles
tree_images = {}
for i in range(1, 10):  # Assuming up to 9 partitions
    for name in ['df_original'] + list(particiones.keys()):
        img_path = f'../results/paso_{i:02d}_{name}_arbol_enraizado_PODADO.png'
        if os.path.exists(img_path):
            tree_images[f'{name}_tree'] = get_base64(img_path)

# Cargar imágenes de MST con caminos destacados
mst_path_images = {}
for i, name in enumerate(['df_original'] + list(particiones.keys()), 1):
    img_path = f'../results/paso_{i:02d}_{name}_grafos_mst_camino_resaltado.png'
    if os.path.exists(img_path):
        mst_path_images[f'{name}_mst_path'] = get_base64(img_path)

# Cargar imágenes de MST Prim podado
prim_podado_images = {}
for i, name in enumerate(['df_original'] + list(particiones.keys()), 1):
    img_path = f'../results/paso_{i:02d}_{name}_prim_poda_comparacion.png'
    if os.path.exists(img_path):
        prim_podado_images[f'{name}_prim_podado'] = get_base64(img_path)

# Cargar imágenes de uniones por pares
union_images = {}
# Cargar imagen global
global_union_path = '../results/paso_08_union_global.png'
if os.path.exists(global_union_path):
    union_images['global'] = get_base64(global_union_path)

# Cargar algunas uniones por pares (ejemplos)
pairs = [('df_original', 'B2C'), ('df_original', 'W2C'), ('B2C', 'W2C')]
for a, b in pairs:
    img_path = f'../results/paso_08_union_{a}_{b}.png'
    if os.path.exists(img_path):
        union_images[f'{a}_{b}'] = get_base64(img_path)

# Cargar imágenes
densidad_b64 = get_base64('../results/grafico_densidad.png')
cajas_b64 = get_base64('../results/diagrama_cajas.png')
pearson_heatmap_b64 = get_base64('../results/pearson_heatmap_df_original.png')

html = f"""
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <title>Resultado Completo - Proyecto de Análisis de Datos</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; }}
        h1 {{ color: #333; text-align: center; }}
        h2 {{ color: #555; border-bottom: 2px solid #ddd; padding-bottom: 5px; }}
        h3 {{ color: #777; }}
        pre {{ background-color: #f4f4f4; padding: 10px; border-radius: 5px; white-space: pre-wrap; }}
        table {{ border-collapse: collapse; width: 100%; margin: 10px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        img {{ max-width: 100%; height: auto; margin: 10px 0; }}
        .section {{ margin-bottom: 40px; }}
    </style>
</head>
<body>
    <h1>Resultado Completo del Proyecto</h1>
    <p>Este documento consolida todos los resultados del análisis, desde la configuración inicial hasta las visualizaciones finales.</p>

    <div class="section">
        <h2>Paso 1: Configuración Inicial</h2>
        <p>Configuración de librerías y entorno.</p>
        <h3>Salida del Script:</h3>
        <pre>Configuración inicial completada. Librerías importadas y entorno configurado.</pre>
    </div>

    <div class="section">
        <h2>Paso 2: Generación de Datos</h2>
        <p>Generación del dataset con 39 variables jerárquicas y target 'y'.</p>
        <h3>Salida del Script:</h3>
        <pre>================================================================================
PASO 1: GENERANDO DATASET DE EJEMPLO (39 Variables Jerárquicas)
================================================================================

Dataset creado con 10000 filas y 40 columnas.

Primeras 5 filas del dataset:
        x_1       x_2       x_3       x_4       x_5       x_6       x_7       x_8  ...      x_33      x_34      x_35      x_36      x_37      x_38      x_39          y
0  4.279917  3.678779  5.559430  5.897073  5.193062  3.391470  4.420827  5.171649  ...  3.437090  5.129587  4.817750  4.898492  4.298793  3.822558  6.795682  53.355636      
1  3.980064  4.555510  3.742525  4.258395  5.351606  5.291637  4.359822  3.943575  ...  2.624302  4.381029  4.172656  4.047438  3.849932  4.041959  5.737760  43.574912      
2  3.874964  1.829474  2.915967  4.731155  4.441556  4.177599  4.718883  3.902457  ...  3.038136  3.970250  4.768419  4.884397  5.022881  3.863899  7.031204  52.741687      
3  2.614525  4.959866  3.663073  5.165075  3.486023  3.213891  3.264784  4.991071  ...  1.978424  3.650075  4.134344  4.648668  2.858402  3.481882  5.298844  38.885315      
4  5.648978  4.290876  3.944588  5.030596  4.310459  4.590147  4.144795  3.583532  ...  2.780840  4.618292  4.196007  4.642515  4.210974  4.332660  6.439005  55.137350      

[5 rows x 40 columns]

Estadísticas de la columna 'y':
count    10000.000000
mean        42.802428
std         12.084989
min          0.000000
25%         34.564279
50%         42.817229
75%         50.841121
max         85.185587
Name: y, dtype: float64

Dataset guardado en 'dataset_ejemplo.csv'.</pre>
    </div>

    <div class="section">
        <h2>Paso 3: Algoritmos de Ordenamiento</h2>
        <p>Clase con 10 algoritmos de ordenamiento implementados.</p>
        <h3>Salida del Script:</h3>
        <pre>Clase SortingAlgorithms definida correctamente.</pre>
    </div>

    <div class="section">
        <h2>Paso 4: Función de Particionamiento</h2>
        <p>Función para segmentar DataFrames usando algoritmos de ordenamiento.</p>
        <h3>Salida del Script:</h3>
        <pre>Clase SortingAlgorithms definida correctamente.
Función df_partition definida correctamente.</pre>
    </div>

    <div class="section">
        <h2>Paso 5: Comparación de Algoritmos y Particiones</h2>
        <p>Comparación de rendimiento y creación de particiones iniciales.</p>
        <h3>Salida del Script:</h3>
        <pre>Clase SortingAlgorithms definida correctamente.

================================================================================
PASO 2.1: COMPARANDO RENDIMIENTO DE ALGORITMOS DE ORDENAMIENTO
================================================================================

--- Probando: Quick Sort ---
Algoritmo usado: quick_sort
Tiempo de ejecución: 10.327037 segundos
Tamaño de la partición 'best': 2500 filas


================================================================================
PASO 2.2: CREANDO PARTICIONES CON EL ALGORITMO ELEGIDO (quick_sort)
================================================================================
Algoritmo usado: quick_sort
Tiempo de ejecución: 10.822485 segundos
Tamaño de la partición 'best': 2500 filas
Algoritmo usado: quick_sort
Tiempo de ejecución: 10.390800 segundos
Tamaño de la partición 'worst': 2500 filas

Partición de los MEJORES 25% (B4C):
        x_1       x_2       x_3       x_4       x_5       x_6       x_7       x_8  ...      x_33      x_34      x_35      x_36      x_37      x_38       x_39          y
0  4.085902  3.036401  5.323634  4.233021  5.812027  6.489781  6.309039  4.611565  ...  4.404070  6.520857  6.400149  6.426074  6.853545  6.585021  10.815629  85.185587     
1  3.880585  3.406927  6.702007  4.455212  7.880857  5.639756  7.343797  4.233680  ...  3.880579  7.336926  6.559027  7.081370  6.156682  7.018751  10.991004  84.954699     
2  4.845900  3.147452  6.423568  2.800010  6.323262  6.420386  7.980050  4.208688  ...  3.716170  6.964166  6.304718  6.527416  6.790531  6.078286  10.265267  83.951259     
3  1.553666  1.817690  5.121309  4.835144  4.740851  5.517811  6.502193  5.994909  ...  4.086040  6.236058  6.414416  5.908491  6.217010  6.347022   9.966880  82.635679     
4  4.574720  4.413742  5.697907  4.296781  6.759657  6.680183  5.830958  3.555211  ...  2.976086  4.979972  6.164606  7.576731  6.274908  6.937754  11.197439  82.283601     

[5 rows x 40 columns]

Partición de los PEORES 25% (W4C):
        x_1       x_2       x_3       x_4       x_5       x_6       x_7       x_8  ...      x_33      x_34      x_35      x_36      x_37      x_38      x_39         y
0  2.947149  3.358339  0.920672  5.495179  1.205409  1.304819  1.053381  3.295950  ...  0.542584  0.376750  1.028155  0.412993  1.748054  0.487736  0.066033  0.000000       
1  4.329376  5.559788  0.735194  5.357313  0.795926  0.740878  2.358292  2.728500  ...  0.325989  0.600889  1.117133  1.993354  0.771994  0.930245  0.396536  1.191403       
2  4.436266  4.811306  0.592808  2.955123  2.254343  1.591635  1.774923  3.016822  ...  1.377979  1.284905  2.405889  1.776384  1.371782  0.076125  0.642658  1.314865       
3  5.235239  3.838821  1.237739  5.261331  1.793034  1.063210  2.558507  3.762950  ...  1.227370  2.543561  0.869453  1.213623  0.132933  1.611931  0.279970  4.369079       
4  5.079851  4.304810  1.280591  6.271565  0.998933  1.426649  1.363955  3.568480  ...  1.330107  0.889817  1.724077  1.073135  0.890368  1.311718  0.373761  4.681035       

[5 rows x 40 columns]

Particiones guardadas en 'particion_mejores.csv' y 'particion_peores.csv'.</pre>
    </div>

    <div class="section">
        <h2>Paso 6: Creación de Particiones Múltiples</h2>
        <p>Creación de particiones con diferentes porcentajes.</p>
        <h3>Salida del Script:</h3>
        <pre>Clase SortingAlgorithms definida correctamente.

================================================================================
PASO 3.1: CREANDO MÚLTIPLES PARTICIONES CON 'quick_sort'
================================================================================
Algoritmo usado: quick_sort
Tiempo de ejecución: 10.302449 segundos
Tamaño de la partición 'best': 5000 filas
Algoritmo usado: quick_sort
Tiempo de ejecución: 10.612241 segundos
Tamaño de la partición 'worst': 5000 filas
Algoritmo usado: quick_sort
Tiempo de ejecución: 10.968949 segundos
Tamaño de la partición 'best': 2500 filas
Algoritmo usado: quick_sort
Tiempo de ejecución: 10.565116 segundos
Tamaño de la partición 'worst': 2500 filas
Algoritmo usado: quick_sort
Tiempo de ejecución: 10.813146 segundos
Tamaño de la partición 'best': 1250 filas
Algoritmo usado: quick_sort
Tiempo de ejecución: 9.991859 segundos
Tamaño de la partición 'worst': 1250 filas
Algoritmo usado: quick_sort
Tiempo de ejecución: 9.906181 segundos
Tamaño de la partición 'best': 625 filas
Algoritmo usado: quick_sort
Tiempo de ejecución: 10.014678 segundos
Tamaño de la partición 'worst': 625 filas

Resumen de tamaños de las particiones:
B2C (Best 50%): (5000, 40)
W2C (Worst 50%): (5000, 40)
B4C (Best 25%): (2500, 40)
W4C (Worst 25%): (2500, 40)
B8C (Best 12.5%): (1250, 40)
W8C (Worst 12.5%): (1250, 40)
B16C (Best 6.25%): (625, 40)
W16C (Worst 6.25%): (625, 40)

Todas las particiones guardadas como archivos CSV individuales.</pre>
    </div>

    <div class="section">
        <h2>Paso 7: Análisis Estadístico de Particiones</h2>
        <p>Análisis estadístico de las particiones creadas.</p>
        <h3>Tabla de Estadísticas:</h3>
        <table>
            <tr><th>Partición</th><th>Tamaño</th><th>Min_y</th><th>Max_y</th><th>Media_y</th><th>Mediana_y</th><th>Std_Dev_y</th></tr>
            <tr><td>df_original</td><td>10000</td><td>0.00</td><td>85.19</td><td>42.80</td><td>42.82</td><td>12.08</td></tr>
            <tr><td>B2C</td><td>5000</td><td>42.82</td><td>85.19</td><td>52.45</td><td>50.84</td><td>7.29</td></tr>
            <tr><td>W2C</td><td>5000</td><td>0.00</td><td>42.81</td><td>33.16</td><td>34.56</td><td>7.27</td></tr>
            <tr><td>B4C</td><td>2500</td><td>50.84</td><td>85.19</td><td>58.17</td><td>56.61</td><td>5.97</td></tr>
            <tr><td>W4C</td><td>2500</td><td>0.00</td><td>34.56</td><td>27.42</td><td>28.94</td><td>5.86</td></tr>
            <tr><td>B8C</td><td>1250</td><td>56.62</td><td>85.19</td><td>62.79</td><td>61.67</td><td>5.08</td></tr>
            <tr><td>W8C</td><td>1250</td><td>0.00</td><td>28.94</td><td>22.94</td><td>24.46</td><td>5.08</td></tr>
            <tr><td>B16C</td><td>625</td><td>61.67</td><td>85.19</td><td>66.69</td><td>65.59</td><td>4.37</td></tr>
            <tr><td>W16C</td><td>625</td><td>0.00</td><td>24.44</td><td>19.05</td><td>20.09</td><td>4.45</td></tr>
        </table>
    </div>

    <div class="section">
        <h2>Paso 8: Visualización de Densidad</h2>
        <p>Gráfico de densidad de 'y' por partición.</p>
        <img src="data:image/png;base64,{densidad_b64}" alt="Gráfico de Densidad">
    </div>

    <div class="section">
        <h2>Paso 9: Diagrama de Cajas</h2>
        <p>Diagrama de cajas de 'y' por partición.</p>
        <img src="data:image/png;base64,{cajas_b64}" alt="Diagrama de Cajas">
    </div>

    <div class="section">
        <h2>Paso 10: Análisis de Correlación de Pearson</h2>
        <p>Análisis de correlaciones de Pearson con threshold 0.6, incluyendo mapas de calor.</p>
        <h3>Resumen de Resultados:</h3>
        <p>Se encontraron correlaciones significativas en todas las particiones. El dataset original tiene 334 aristas, mientras que las particiones más pequeñas tienen menos.</p>
        <h3>Ejemplo de Mapa de Calor (Dataset Original):</h3>
        <img src="data:image/png;base64,{pearson_heatmap_b64}" alt="Mapa de Calor de Correlación - Dataset Original">
        <p>Los mapas de calor completos y detalles se guardaron en archivos CSV en la carpeta results/.</p>
    <div class="section">
        <h2>Paso 11: Algoritmo de Newman - Detección de Comunidades</h2>
        <p>Análisis de comunidades usando el algoritmo de Newman en todas las particiones.</p>
        <h3>Salida del Análisis:</h3>
        <pre>{newman_output}</pre>
    <div class="section">
        <h2>Paso 12: Árboles Enraizados y Podados por Nivel</h2>
        <p>Análisis de árboles enraizados en 'y' con poda por nivel (threshold = 2), incluyendo recorridos BFS/DFS y estadísticas.</p>
        <h3>Salida del Análisis:</h3>
        <pre>{trees_output}</pre>
        <h3>Ejemplos de Visualizaciones:</h3>
        <p>Árboles podados para cada partición (si generados):</p>
        {'<br>'.join([f'<img src="data:image/png;base64,{img}" alt="Árbol {name}" style="max-width:100%; margin:10px 0;">' for name, img in tree_images.items()])}
    </div>

    <div class="section">
        <h2>Paso 13: Visualización de Grafos con MST y Caminos Destacados</h2>
        <p>Visualización de grafos de correlación con Árbol de Expansión Mínima (MST) y caminos más largos destacados.</p>
        <h3>Salida del Análisis:</h3>
        <pre>{mst_paths_output}</pre>
        <h3>Visualizaciones:</h3>
        <p>Grafos con MST y caminos destacados para cada partición:</p>
        {'<br>'.join([f'<img src="data:image/png;base64,{img}" alt="MST Caminos {name}" style="max-width:100%; margin:10px 0;">' for name, img in mst_path_images.items()])}
    </div>

    <div class="section">
        <h2>Paso 14: Comparación MST Prim Original vs Podado</h2>
        <p>Visualización comparativa de grafos originales, MST Prim completo con caminos más largos destacados, y MST Prim podado por la mitad del diámetro.</p>
        <h3>Salida del Análisis:</h3>
        <pre>{prim_podado_output}</pre>
        <h3>Visualizaciones:</h3>
        <p>Comparaciones de MST Prim para cada partición:</p>
        {'<br>'.join([f'<img src="data:image/png;base64,{img}" alt="MST Prim Podado {name}" style="max-width:100%; margin:10px 0;">' for name, img in prim_podado_images.items()])}
    </div>

    <div class="section">
        <h2>Paso 15: Unión de Recorridos BFS/DFS por Pares</h2>
        <p>Análisis de uniones de recorridos BFS y DFS entre particiones por pares, incluyendo estadísticas y visualizaciones.</p>
        <h3>Salida del Análisis:</h3>
        <pre>{union_output}</pre>
        <h3>Visualizaciones:</h3>
        <p>Unión global de todas las particiones:</p>
        {f'<img src="data:image/png;base64,{union_images.get("global", "")}" alt="Unión Global" style="max-width:100%; margin:10px 0;">' if 'global' in union_images else '<p>No se generó imagen global.</p>'}
        <p>Ejemplos de uniones por pares:</p>
        {'<br>'.join([f'<img src="data:image/png;base64,{img}" alt="Unión {name}" style="max-width:100%; margin:10px 0;">' for name, img in union_images.items() if name != 'global'])}
    </div>

    <div class="section">
        <h2>Paso 16: Análisis de Intersección</h2>
        <p>Intersección entre nodos cercanos a 'y' (de recorridos BFS/DFS) y nodos con discrepancias estructurales (de comparaciones de árboles podados).</p>
        <h3>Salida del Análisis:</h3>
        <pre>{interseccion_output}</pre>
        <p><strong>Resultado:</strong> No se encontraron nodos en la intersección, ya que no hay discrepancias estructurales en las comparaciones de árboles podados.</p>
    </div>

</body>
</html>
"""

with open('../results/resultado_completo.html', 'w', encoding='utf-8') as f:
    f.write(html)

print('HTML completo generado como ../results/resultado_completo.html')