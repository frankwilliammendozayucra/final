# Proyecto Final: Análisis de Datos con Árboles de Expansión Mínima y Algoritmos de Grafos

## Descripción
Este proyecto realiza un análisis exhaustivo de un dataset de ejemplo utilizando técnicas de correlación de Pearson, algoritmos de grafos (MST Kruskal y Prim), detección de comunidades (Newman), árboles enraizados con poda, y análisis de recorridos BFS/DFS. El resultado final es un reporte HTML completo con visualizaciones y estadísticas.

## Estructura del Proyecto

```
d:\Proyecto final\
├── data\                    # Datos de entrada
│   ├── dataset_ejemplo.csv  # Dataset original (10,000 filas)
│   ├── B2C.csv             # Partición Bootstrap 2C
│   ├── W2C.csv             # Partición Window 2C
│   ├── B4C.csv             # Partición Bootstrap 4C
│   ├── W4C.csv             # Partición Window 4C
│   ├── B8C.csv             # Partición Bootstrap 8C
│   ├── W8C.csv             # Partición Window 8C
│   ├── B16C.csv            # Partición Bootstrap 16C
│   └── W16C.csv            # Partición Window 16C
├── scripts\                 # Scripts de Python
│   ├── configuracion_proyecto.py
│   ├── algoritmos_ordenamiento.py
│   ├── funcion_particionamiento.py
│   ├── generacion_datos.py
│   ├── particiones_completas.py
│   ├── visualizaciones.py
│   ├── analisis_correlacion_pearson.py
│   ├── algoritmo_newman.py
│   ├── arboles_enraizados.py
│   ├── visualizacion_grafos_mst_caminos.py
│   ├── visualizacion_prim_podado.py
│   ├── unir_recorridos_por_pares.py
│   ├── script9_extraer_nodos_discrepantes.py
│   ├── guardar_archivos_finales.py
│   └── generar_html_completo.py
├── results\                 # Resultados y salidas
│   ├── resultado_completo.html    # Reporte HTML final
│   ├── df_original.csv            # Dataset original guardado
│   ├── *_partition.csv            # Particiones guardadas
│   ├── estadisticas_particiones.csv
│   ├── pearson_*.csv              # Matrices y aristas de correlación
│   ├── grafos_mst_*.png           # Visualizaciones de grafos MST
│   ├── paso_*_*.png               # Imágenes de análisis
│   ├── union_recorridos_pares.csv # CSV de uniones BFS/DFS
│   └── resumen_*.csv              # Resúmenes de análisis
└── README.md               # Este archivo
```

## Scripts y Orden de Ejecución

### Scripts Principales (ordenados por ejecución recomendada):

1. **configuracion_proyecto.py** - Configuración inicial del proyecto
2. **algoritmos_ordenamiento.py** - Utilidades de ordenamiento
3. **funcion_particionamiento.py** - Funciones de particionamiento
4. **generacion_datos.py** - Generación de datos de ejemplo
5. **particiones_completas.py** - Creación y análisis de particiones
6. **visualizaciones.py** - Gráficos de densidad y cajas
7. **analisis_correlacion_pearson.py** - Análisis de correlación Pearson
8. **algoritmo_newman.py** - Detección de comunidades Newman
9. **arboles_enraizados.py** - Creación de árboles enraizados y podados
10. **visualizacion_grafos_mst_caminos.py** - Visualización de MSTs con caminos
11. **visualizacion_prim_podado.py** - Comparación MST Prim completo vs podado
12. **unir_recorridos_por_pares.py** - Unión de recorridos BFS/DFS por pares
13. **script9_extraer_nodos_discrepantes.py** - Extracción de nodos discrepantes
14. **guardar_archivos_finales.py** - Guardado de archivos finales
15. **generar_html_completo.py** - Generación del reporte HTML completo

### Scripts Especiales:
- **script9_extraer_nodos_discrepantes.py** - Requiere que se ejecute `visualizacion_prim_podado.py` primero
- **unir_recorridos_por_pares.py** - Requiere que se ejecute `arboles_enraizados.py` primero

## Cómo Ejecutar

### Opción 1: Ejecución Completa Automática
```bash
cd "d:\Proyecto final\scripts"
python generar_html_completo.py
```
Esto ejecuta todos los análisis y genera el HTML completo.

### Opción 2: Ejecución Paso a Paso
Ejecuta los scripts en orden desde VS Code o terminal:
```bash
cd "d:\Proyecto final\scripts"
python analisis_correlacion_pearson.py
python arboles_enraizados.py
# ... continuar con los demás
python generar_html_completo.py
```

## Dependencias
- Python 3.x
- pandas
- numpy
- matplotlib
- seaborn
- networkx
- scikit-learn (para correlaciones)

Instalar con:
```bash
pip install pandas numpy matplotlib seaborn networkx scikit-learn
```

## Resultados Generados

### HTML Principal
- `results/resultado_completo.html` - Reporte completo con todas las secciones

### Archivos CSV
- Matrices de correlación por partición
- Aristas filtradas (r ≥ 0.6)
- Resúmenes de MSTs y caminos
- Estadísticas de particiones
- Uniones de recorridos BFS/DFS

### Imágenes PNG
- Mapas de calor de correlaciones
- Visualizaciones de grafos MST
- Árboles enraizados y podados
- Comparaciones MST completo vs podado
- Uniones de grafos por pares

## Análisis Realizados

1. **Análisis de Correlación Pearson** - Matrices completas y filtradas
2. **Visualización de Grafos** - Grafos originales, MST Kruskal y Prim
3. **Árboles Enraizados** - Árboles BFS enraizados en 'y' con poda por nivel
4. **Detección de Comunidades** - Algoritmo de Newman
5. **Comparación MST** - MST Prim completo vs podado por la mitad del diámetro
6. **Unión de Recorridos** - Análisis de uniones BFS/DFS entre particiones
7. **Extracción de Discrepancias** - Nodos con diferencias estructurales
8. **Intersección de Resultados** - Nodos cercanos a 'y' y estructuralmente inestables

## Notas Técnicas
- Los scripts están diseñados para ejecutarse en secuencia
- Algunos scripts dependen de variables globales generadas por scripts anteriores
- El HTML final integra todos los resultados automáticamente
- Las visualizaciones usan NetworkX y Matplotlib
- Los análisis se ejecutan en particiones del dataset original

## Contacto
Proyecto desarrollado para análisis de datos con algoritmos de grafos y aprendizaje automático.