# Configuración inicial del proyecto: imports y ajustes
# Este script verifica e importa las librerías necesarias sin instalar automáticamente.

try:
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import networkx as nx
    from typing import Literal
    import time
    from collections import deque
    import traceback
    import heapq
    print("Todas las librerías importadas correctamente.")
except ImportError as e:
    print(f"Error al importar librerías: {e}")
    print("Asegúrate de que pandas, numpy, matplotlib, seaborn, networkx estén instalados.")
    raise

# Configuración para visualización en pandas: mostrar todas las columnas
pd.set_option('display.max_columns', None)

# Configuración de estilo para seaborn
sns.set(style="whitegrid")

# Mensaje de confirmación
print("Configuración inicial completada. Librerías importadas y entorno configurado.")