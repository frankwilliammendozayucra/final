# =================================================================
# PASO 6: GENERANDO ÁRBOLES ENRAIZADOS Y PODADOS POR NIVEL
# =================================================================

import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
from collections import deque
import traceback
import os

print("\n" + "="*90)
print("PASO 6: GENERANDO ÁRBOLES ENRAIZADOS EN 'y' (PODADOS POR NIVEL)")
print("="*90)

# ⭐ PARÁMETRO DE UMBRAL DE NIVEL ⭐
# Nivel 0 = 'y'
# Nivel 1 = Hijos directos de 'y'
# Nivel 2 = Hijos de los hijos
LEVEL_THRESHOLD = 2

print(f"✓ Umbral de Nivel (Profundidad) fijado en: {LEVEL_THRESHOLD}\n")

# Verificar disponibilidad de graphviz
layout_jerarquico_disponible = False
try:
    from networkx.drawing import nx_pydot
    layout_jerarquico_disponible = True
    print("✓ Se encontró 'nx_pydot'. Usando layout jerárquico 'dot'.\n")
except ImportError:
    print("⚠️  'nx_pydot' no disponible. Usando Shell Layout.\n")

# -----------------------------------------------------------------
# FUNCIONES DE RECORRIDO (Sin cambios)
# -----------------------------------------------------------------

def custom_bfs(graph, start_node):
    """Recorrido BFS desde nodo inicial."""
    if start_node not in graph.nodes():
        return []

    visited = set()
    queue = deque([start_node])
    bfs_order = []

    while queue:
        node = queue.popleft()
        if node not in visited:
            visited.add(node)
            bfs_order.append(node)
            # Asegurarse de que el grafo tenga la función neighbors
            if hasattr(graph, 'neighbors'):
                for neighbor in graph.neighbors(node):
                    if neighbor not in visited:
                        queue.append(neighbor)

    return bfs_order

def custom_dfs(graph, start_node):
    """Recorrido DFS desde nodo inicial."""
    if start_node not in graph.nodes():
        return []

    visited = set()
    stack = [start_node]
    dfs_order = []

    while stack:
        node = stack.pop()
        if node not in visited:
            visited.add(node)
            dfs_order.append(node)
            if hasattr(graph, 'neighbors'):
                neighbors = list(graph.neighbors(node))
                for neighbor in reversed(neighbors):
                    if neighbor not in visited:
                        stack.append(neighbor)

    return dfs_order


def ordered_tree_layout(tree, root_node, x_spacing=2.5, y_spacing=2.5, order='bfs'):
    """
    Layout de árbol ordenado: asigna posiciones (x,y) por nivel (y = -depth)
    y distribuye los nodos de cada nivel equidistantemente en x según 'order':
    'bfs' (orden de BFS), 'alphabetical', 'degree'.
    """
    # Obtener nodos en orden BFS (mantenemos el orden natural del árbol)
    try:
        bfs_nodes = list(nx.bfs_tree(tree, source=root_node))
    except Exception:
        bfs_nodes = list(tree.nodes())

    depths = dict(nx.shortest_path_length(tree, source=root_node))
    levels = {}
    for node in bfs_nodes:
        depth = depths.get(node, 0)
        levels.setdefault(depth, []).append(node)

    pos = {}
    for depth in sorted(levels.keys()):
        nodes = levels[depth]
        # Ordenar según criterio
        if order == 'alphabetical':
            nodes = sorted(nodes)
        elif order == 'degree':
            nodes = sorted(nodes, key=lambda n: tree.degree(n), reverse=True)
        # else keep BFS order

        n = len(nodes)
        if n == 1:
            xs = [0.0]
        else:
            span = (n - 1) * x_spacing
            start_x = -span / 2
            xs = [start_x + i * x_spacing for i in range(n)]
        y = -depth * y_spacing
        for x, node in zip(xs, nodes):
            pos[node] = (x, y)
    return pos

# Diccionario para almacenar resultados
resultados_arboles = {}

def crear_arbol_enraizado_particion(partition_name, mst_prim_graph, numero_paso):
    """
    Crea árbol enraizado, lo PODA según el LEVEL_THRESHOLD,
    y luego ejecuta el análisis (layout, viz, BFS/DFS) sobre el árbol podado.
    """

    print(f"\n{'='*90}")
    print(f"PASO {numero_paso}: PARTICIÓN '{partition_name.upper()}'")
    print(f"{'='*90}\n")

    # ==========================================
    # 1. VALIDAR QUE EXISTA NODO 'y'
    # ==========================================
    root_node = 'y'
    if root_node not in mst_prim_graph.nodes():
        print(f"  ✗ Nodo raíz '{root_node}' no encontrado. Saltando.")
        return None

    print(f"1. Creando árbol enraizado en '{root_node}'...")

    # ==========================================
    # 2. CREAR ÁRBOL ENRAIZADO (BFS TREE)
    # ==========================================
    # Primero creamos el árbol completo para saber todas las profundidades
    tree_prim_y = nx.bfs_tree(mst_prim_graph, source=root_node)
    print(f"  ✓ Árbol BFS completo creado ({tree_prim_y.number_of_nodes()} nodos, {tree_prim_y.number_of_edges()} aristas)")

    # ==========================================
    # 2.5 APLICAR PODA POR NIVEL (THRESHOLD)
    # ==========================================
    print(f"\n2. Aplicando poda (Nivel <= {LEVEL_THRESHOLD})...")

    # 1. Encontrar todos los nodos dentro del umbral
    nodes_to_keep = []
    # Calculamos profundidades desde el árbol completo
    heights_full = nx.shortest_path_length(tree_prim_y, source=root_node)

    for node, depth in heights_full.items():
        if depth <= LEVEL_THRESHOLD:
            nodes_to_keep.append(node)

    # 2. Crear el subgrafo basado en esos nodos
    # Usamos .subgraph() y lo copiamos para tener un grafo independiente
    pruned_tree = tree_prim_y.subgraph(nodes_to_keep).copy()

    print(f"  ✓ Árbol podado: {pruned_tree.number_of_nodes()} nodos, {pruned_tree.number_of_edges()} aristas")


    # ==========================================
    # 3. CALCULAR LAYOUT JERÁRQUICO Y ORDENADO (SOBRE ÁRBOL PODADO)
    # ==========================================
    print(f"\n3. Calculando layout jerárquico y ordenado...")

    pos = None
    layout_name = ""

    # Intentar usar 'dot' para referencia, pero usaremos layout ordenado para el dibujo
    if layout_jerarquico_disponible:
        try:
            # Intentar obtener layout 'dot' (no se usa directamente para dibujo, sirve como referencia)
            pos_dot = nx_pydot.graphviz_layout(pruned_tree, prog='dot')
            # Calcular layout ordenado (por BFS) para dibujar de forma clara y ordenada
            pos = ordered_tree_layout(pruned_tree, root_node, x_spacing=2.5, y_spacing=2.5, order='bfs')
            layout_name = "Jerárquico (dot) + Ordenado (bfs)"
            print(f"  ✓ Layout 'dot' aplicado (referencia). Usando layout ordenado para dibujo.")
        except Exception as e:
            print(f"  ⚠️  Layout 'dot' falló. Usando layout ordenado (fallback).")
            pos = ordered_tree_layout(pruned_tree, root_node)
            layout_name = "Ordenado (fallback)"
    else:
        print(f"  ⚠️  nx_pydot no disponible. Usando layout ordenado.")
        pos = ordered_tree_layout(pruned_tree, root_node)
        layout_name = "Ordenado"

    # Ajuste opcional: escalar posiciones horizontales para evitar solapamientos en niveles con muchas hojas
    # (aumentar x_spacing arriba si es necesario)

    # ==========================================
    # 4. VISUALIZAR ÁRBOL PODADO
    # ==========================================
    print(f"\n4. Generando visualización (podada)...")

    fig, ax = plt.subplots(1, 1, figsize=(18, 14))
    # Título actualizado para reflejar la poda
    fig.suptitle(f"ÁRBOL PODADO (Nivel <= {LEVEL_THRESHOLD}) EN '{root_node}' - MST PRIM\nPartición: {partition_name.upper()}",
                 fontsize=16, fontweight='bold')

    # Colorear nodo raíz
    node_colors = ['#ff0000' if node == root_node else '#90EE90' for node in pruned_tree.nodes()] # <--- USA ÁRBOL PODADO

    # Dibujar árbol dirigido (podado)
    nx.draw(
        pruned_tree, # <--- USA ÁRBOL PODADO
        pos,
        ax=ax,
        with_labels=True,
        node_size=3500,
        node_color=node_colors,
        edge_color='#333333',
        width=2.5,
        font_size=12,
        font_weight='bold',
        arrows=True,
        arrowsize=30,
        arrowstyle='-|>',
        edgecolors='black',
        linewidths=2.5,
        connectionstyle='arc3,rad=0.1'
    )

    ax.set_title(f"Layout: {layout_name} | Raíz: '{root_node}'",
                fontsize=13, fontweight='bold', pad=15)
    ax.axis('off')

    plt.tight_layout()

    # Guardar figura
    filename = f"paso_{numero_paso:02d}_{partition_name}_arbol_enraizado_PODADO.png" # <--- Nombre actualizado
    plt.savefig(filename, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"  ✓ Visualización guardada: '{filename}'")
    plt.close(fig)  # Cerrar para no mostrar

    # ==========================================
    # 5. EJECUTAR BFS Y DFS (SOBRE ÁRBOL PODADO)
    # ==========================================
    print(f"\n5. Ejecutando recorridos BFS y DFS (sobre árbol podado)...")

    bfs_result = custom_bfs(pruned_tree, root_node) # <--- USA ÁRBOL PODADO
    dfs_result = custom_dfs(pruned_tree, root_node) # <--- USA ÁRBOL PODADO

    print(f"\n  BFS (Amplitud):")
    print(f"      {' → '.join(bfs_result)}")
    print(f"      Nodos visitados: {len(bfs_result)}")

    print(f"\n  DFS (Profundidad):")
    print(f"      {' → '.join(dfs_result)}")
    print(f"      Nodos visitados: {len(dfs_result)}")

    # ==========================================
    # 6. ESTADÍSTICAS DEL ÁRBOL (PODADO)
    # ==========================================
    heights = nx.shortest_path_length(pruned_tree, source=root_node) # <--- USA ÁRBOL PODADO
    altura = max(heights.values()) if heights else 0
    profundidad_promedio = sum(heights.values()) / len(heights) if heights else 0

    print(f"\n6. Estadísticas del árbol (podado):") # <--- Título actualizado
    print(f"  • Altura (profundidad máxima): {altura}")
    print(f"  • Profundidad promedio: {profundidad_promedio:.2f}")
    # Usar pruned_tree para el cálculo
    branch_factor = (pruned_tree.number_of_edges() / max(1, pruned_tree.number_of_nodes() - 1)) if pruned_tree.number_of_nodes() > 1 else 0
    print(f"  • Factor de ramificación: {branch_factor:.2f}")

    # ==========================================
    # 7. GUARDAR RESULTADOS (DEL ÁRBOL PODADO)
    # ==========================================
    resultados_arboles[partition_name] = {
        'arbol_enraizado': pruned_tree, # <--- Guarda el árbol podado
        'posiciones': pos,
        'bfs': bfs_result,
        'dfs': dfs_result,
        'nodos_totales': pruned_tree.number_of_nodes(), # <--- Stats del podado
        'aristas_totales': pruned_tree.number_of_edges(), # <--- Stats del podado
        'altura': altura,
        'profundidad_promedio': profundidad_promedio,
        'layout': layout_name
    }

    print(f"\n{'─'*90}")
    print(f"✓ Partición '{partition_name.upper()}' procesada (podada)")
    print(f"{'─'*90}\n")

    return resultados_arboles[partition_name]

# ========================================
# EJECUTAR PARA TODAS LAS PARTICIONES
# ========================================

print("\nObteniendo MST Prim de resultados previos...\n")

# Cargar datos y particiones
df = pd.read_csv('../data/dataset_ejemplo.csv')
particiones = {}
particion_names = ['B2C', 'W2C', 'B4C', 'W4C', 'B8C', 'W8C', 'B16C', 'W16C']
for name in particion_names:
    path = f'../data/{name}.csv'
    if os.path.exists(path):
        particiones[name] = pd.read_csv(path)

# Importar GraphAlgorithms
import sys
sys.path.append('.')
from analisis_correlacion_pearson import GraphAlgorithms

# Simular resultados_pearson (cargar desde CSVs si existen)
resultados_pearson = {}
for name in ['df_original'] + list(particiones.keys()):
    aristas_path = f'../results/pearson_aristas_{name}.csv'
    if os.path.exists(aristas_path):
        aristas_df = pd.read_csv(aristas_path)
        aristas = aristas_df.to_dict('records')
        resultados_pearson[name] = {'aristas': aristas}

particiones_lista = [
    ('df_original', df),
] + [(name, df_part) for name, df_part in particiones.items()]

paso = 0
for nombre, partition_df in particiones_lista:
    paso += 1
    try:
        # Obtener columnas numéricas
        numerical_cols = []
        if not partition_df.empty:
            for col in partition_df.columns:
                # Usar pd.to_numeric para ser más robusto
                if pd.to_numeric(partition_df[col], errors='coerce').notna().any():
                     numerical_cols.append(col)

        # Obtener aristas del resultado previo (si existe)
        if nombre in resultados_pearson:
            aristas = resultados_pearson[nombre]['aristas']

            # Recrear grafo
            G = nx.Graph()
            G.add_nodes_from(numerical_cols)
            if aristas:
                for arista in aristas:
                    r = arista['pearson_r']
                    weight = 1.0 - r
                    G.add_edge(arista['var1'], arista['var2'], weight=weight, correlation=r)

            # Calcular MST Prim
            if G.number_of_edges() > 0:
                # Usar nuestra implementación de Prim (del script anterior)
                mst_prim, _ = GraphAlgorithms.prim_mst(G)
                # Llamar a la función (modificada)
                crear_arbol_enraizado_particion(nombre, mst_prim, paso)
            else:
                print(f"\n{'='*90}")
                print(f"PASO {paso}: PARTICIÓN '{nombre.upper()}'")
                print(f"{'='*90}\n")
                print(f"⚠️  Sin aristas en {nombre}. Saltando.\n")
        else:
            print(f"\n{'='*90}")
            print(f"PASO {paso}: PARTICIÓN '{nombre.upper()}'")
            print(f"{'='*90}\n")
            print(f"⚠️  {nombre} no encontrado en resultados_pearson. Saltando.\n")

    except Exception as e:
        print(f"\n⚠️  Error en '{nombre}': {str(e)}")
        traceback.print_exc()
        print(f"  Continuando...\n")

# ========================================
# RESUMEN COMPARATIVO
# ========================================

if resultados_arboles:
    print("\n" + "="*90)
    print(f"RESUMEN COMPARATIVO DE ÁRBOLES ENRAIZADOS (PODADOS A NIVEL <= {LEVEL_THRESHOLD})")
    print("="*90)

    for partition_name, data in resultados_arboles.items():
        print(f"\n{partition_name.upper()}:")
        print(f"  Nodos: {data['nodos_totales']} | Aristas: {data['aristas_totales']} | Altura: {data['altura']}")
        print(f"  BFS: {' → '.join(data['bfs'])}")
        print(f"  DFS: {' → '.join(data['dfs'])}")

    # ========================================
    # TABLA COMPARATIVA
    # ========================================

    print("\n" + "="*90)
    print(f"TABLA COMPARATIVA DE ÁRBOLES ENRAIZADOS (PODADOS A NIVEL <= {LEVEL_THRESHOLD})")
    print("="*90 + "\n")

    tabla_arboles = []
    for partition_name, data in resultados_arboles.items():
        tabla_arboles.append({
            'Partición': partition_name,
            'Nodos': data['nodos_totales'],
            'Aristas': data['aristas_totales'],
            'Altura': data['altura'],
            'Prof_Promedio': f"{data['profundidad_promedio']:.2f}",
            'Layout': data['layout']
        })

    df_arboles = pd.DataFrame(tabla_arboles)
    print(df_arboles.to_string(index=False))

    # Guardar tabla
    df_arboles.to_csv('../results/resumen_arboles_enraizados_podados.csv', index=False)
    print("\n✓ Tabla guardada: '../results/resumen_arboles_enraizados_podados.csv'")

    print("\n" + "="*90)
    print("✓ GENERACIÓN DE ÁRBOLES PODADOS COMPLETADA")
    print("="*90)
else:
    print("\nNo se generaron resultados de árboles enraizados.")