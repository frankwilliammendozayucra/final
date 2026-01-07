# ========================================
# ANÁLISIS DE CORRELACIÓN DE PEARSON
# ========================================
# Script para calcular correlaciones de Pearson con threshold configurable.
# Incluye mapas de calor y exportación de resultados.

import math
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import traceback
import heapq

print("\n" + "="*120)
print("ANÁLISIS DE CORRELACIÓN DE PEARSON CON THRESHOLD FIJO (Y MATRIZ GRÁFICA)")
print("="*120)

# ⭐ PARÁMETRO CONFIGURABLE POR EL USUARIO ⭐
THRESHOLD = 0.6  # ← CAMBIA ESTE VALOR AQUÍ

print(f"\n📊 CONFIGURACIÓN:")
print(f"   Threshold: {THRESHOLD}")
print(f"   Solo correlaciones POSITIVAS r ≥ {THRESHOLD}\n")

# Cargar dataset y particiones (asumiendo que existen o se crean en scripts previos)
df = pd.read_csv('../data/dataset_ejemplo.csv')

# Crear particiones (reutilizando lógica de particiones_completas.py)
from algoritmos_ordenamiento import SortingAlgorithms

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
    sort_method = getattr(sorter, sorting_algorithm)

    if sort_method is None:
        raise ValueError(f"Algoritmo '{sorting_algorithm}' no disponible en SortingAlgorithms.")

    ascending_order = (partition_type == "worst")

    df_sorted = sort_method(df, criterion, ascending=ascending_order)

    partition_size = int(len(df) * percentage)

    df_result = df_sorted.iloc[:partition_size].copy()

    return df_result

selected_algorithm = 'quick_sort'
B2C = df_partition(df, 'y', 0.50, "best", selected_algorithm)
W2C = df_partition(df, 'y', 0.50, "worst", selected_algorithm)
B4C = df_partition(df, 'y', 0.25, "best", selected_algorithm)
W4C = df_partition(df, 'y', 0.25, "worst", selected_algorithm)
B8C = df_partition(df, 'y', 0.125, "best", selected_algorithm)
W8C = df_partition(df, 'y', 0.125, "worst", selected_algorithm)
B16C = df_partition(df, 'y', 0.0625, "best", selected_algorithm)
W16C = df_partition(df, 'y', 0.0625, "worst", selected_algorithm)

# Diccionario para almacenar resultados
resultados_pearson = {}

# Particiones a analizar
particiones_analizar = {
    'df_original': df,
    'B2C': B2C,
    'W2C': W2C,
    'B4C': B4C,
    'W4C': W4C,
    'B8C': B8C,
    'W8C': W8C,
    'B16C': B16C,
    'W16C': W16C
}

# Función para generar el mapa de calor
def plot_correlation_heatmap(correlation_df, partition_name, threshold=None):
    if correlation_df.empty:
        print(f"  No se puede generar mapa de calor para {partition_name}: DataFrame de correlación vacío.")
        return

    plt.figure(figsize=(len(correlation_df.columns)*0.8, len(correlation_df.index)*0.8))

    title = f'Mapa de Calor de Correlación de Pearson - {partition_name}'
    if threshold is not None:
        title += f' (Threshold r ≥ {threshold})'

    sns.heatmap(
        correlation_df,
        annot=True,
        cmap='coolwarm',
        fmt=".2f",
        linewidths=.5,
        vmin=-1, vmax=1
    )
    plt.title(title, fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    filename = f'../results/pearson_heatmap_{partition_name}.png'
    plt.savefig(filename)
    plt.close()
    print(f"  ✅ Mapa de calor guardado: {filename}")

# LOOP PRINCIPAL - ANALIZAR CADA PARTICIÓN
for nombre_particion, partition_df in particiones_analizar.items():

    n_datos = len(partition_df)

    print(f"\n{'='*120}")
    print(f"PARTICIÓN: {nombre_particion.upper()} ({n_datos} filas)")
    print(f"{'='*120}\n")

    # 1. Obtener columnas numéricas
    numerical_cols = []
    if n_datos == 0:
        print("1. DataFrame vacío (0 filas).")
        resultados_pearson[nombre_particion] = {'n_datos': 0, 'threshold': THRESHOLD, 'aristas': [], 'num_aristas': 0, 'matriz_completa': pd.DataFrame()}
        continue

    for col in partition_df.columns:
        try:
            if pd.to_numeric(partition_df[col], errors='coerce').notna().any():
                numerical_cols.append(col)
        except (ValueError, TypeError, IndexError):
            pass

    if not numerical_cols:
        print("1. No se encontraron columnas numéricas o el DataFrame es demasiado pequeño.")
        resultados_pearson[nombre_particion] = {'n_datos': n_datos, 'threshold': THRESHOLD, 'aristas': [], 'num_aristas': 0, 'matriz_completa': pd.DataFrame()}
        continue

    print(f"1. Variables numéricas: {numerical_cols}")

    # Crear la matriz de correlación
    corr_matrix_df = pd.DataFrame(index=numerical_cols, columns=numerical_cols, dtype=float)

    # 2. Calcular correlaciones de Pearson
    print(f"\n2. Calculando correlaciones de Pearson...\n")

    aristas_significativas = []

    for i in range(len(numerical_cols)):
        col1_name = numerical_cols[i]
        corr_matrix_df.loc[col1_name, col1_name] = 1.0

        for j in range(i + 1, len(numerical_cols)):
            col2_name = numerical_cols[j]

            series1 = pd.to_numeric(partition_df[col1_name], errors='coerce').dropna()
            series2 = pd.to_numeric(partition_df[col2_name], errors='coerce').dropna()

            common_index = series1.index.intersection(series2.index)

            col1_vals = series1.loc[common_index].tolist()
            col2_vals = series2.loc[common_index].tolist()

            n_valid_datos = len(col1_vals)

            r = 0.0
            if n_valid_datos > 1:
                media_x = sum(col1_vals) / n_valid_datos
                media_y = sum(col2_vals) / n_valid_datos

                suma_xy = sum((col1_vals[k] - media_x) * (col2_vals[k] - media_y) for k in range(n_valid_datos))
                suma_x2 = sum((col1_vals[k] - media_x) ** 2 for k in range(n_valid_datos))
                suma_y2 = sum((col2_vals[k] - media_y) ** 2 for k in range(n_valid_datos))

                denom = math.sqrt(suma_x2 * suma_y2)
                if denom != 0:
                    r = suma_xy / denom

            corr_matrix_df.loc[col1_name, col2_name] = r
            corr_matrix_df.loc[col2_name, col1_name] = r

            if r >= THRESHOLD:
                aristas_significativas.append({
                    'var1': col1_name,
                    'var2': col2_name,
                    'r_pearson': r
                })

    # 2b. MOSTRAR LA MATRIZ COMPLETA
    if not corr_matrix_df.empty:
        print(f"2b. Matriz de Correlación de Pearson COMPLETA:\n")
        print(corr_matrix_df.to_string(float_format="%.6f"))
        print("\n" + "-"*50 + "\n")

        print("  Generando mapa de calor...")
        plot_correlation_heatmap(corr_matrix_df, nombre_particion)
    else:
        print("2b. No se pudo generar una matriz de correlación.")

    aristas_significativas.sort(key=lambda x: x['r_pearson'], reverse=True)

    num_aristas = len(aristas_significativas)
    print(f"3. Aristas encontradas (r ≥ {THRESHOLD}): {num_aristas}\n")

    if len(aristas_significativas) > 0:
        r_max = max([a['r_pearson'] for a in aristas_significativas])
        r_min = min([a['r_pearson'] for a in aristas_significativas])
        r_prom = sum([a['r_pearson'] for a in aristas_significativas]) / len(aristas_significativas)

        print(f"4. Estadísticas (filtradas):")
        print(f"   Correlación máxima: {r_max:.6f}")
        print(f"   Correlación mínima: {r_min:.6f}")
        print(f"   Correlación promedio: {r_prom:.6f}\n")

        num_mostrar = min(15, len(aristas_significativas))
        print(f"5. Top {num_mostrar} correlaciones (filtradas):\n")
        print(f"   {'Nº':<4} {'Variable 1':<12} {'Variable 2':<12} {'Pearson r':<15}")
        print(f"   {'-'*50}")

        for idx, arista in enumerate(aristas_significativas[:num_mostrar], 1):
            print(f"   {idx:<4} {arista['var1']:<12} {arista['var2']:<12} {arista['r_pearson']:>+.8f}")

        if len(aristas_significativas) > 15:
            print(f"\n   ... y {len(aristas_significativas) - 15} más")
    else:
        print(f" 4. ⚠️   Sin correlaciones positivas por encima del threshold ({THRESHOLD})\n")

    resultados_pearson[nombre_particion] = {
        'n_datos': n_datos,
        'threshold': THRESHOLD,
        'aristas': aristas_significativas,
        'num_aristas': num_aristas,
        'matriz_completa': corr_matrix_df
    }

# TABLA COMPARATIVA GENERAL
print(f"\n\n{'='*120}")
print("TABLA COMPARATIVA: RESULTADOS POR PARTICIÓN")
print(f"{'='*120}\n")

print(f"Threshold aplicado: {THRESHOLD}\n")

print(f"{'Partición':<15} {'Filas':<10} {'Aristas':<12} {'r_Máx':<15} {'r_Mín':<15} {'r_Prom':<15} {'Ratio':<12}")
print("-"*120)

for nombre, datos in resultados_pearson.items():
    n = datos['n_datos']
    num_aristas = datos['num_aristas']

    if len(datos['aristas']) > 0:
        r_max = max([a['r_pearson'] for a in datos['aristas']])
        r_min = min([a['r_pearson'] for a in datos['aristas']])
        r_prom = sum([a['r_pearson'] for a in datos['aristas']]) / len(datos['aristas'])
        ratio = num_aristas / n if n > 0 else 0
    else:
        r_max = 0
        r_min = 0
        r_prom = 0
        ratio = 0

    print(f"{nombre:<15} {n:<10} {num_aristas:<12} {r_max:>+.8f}    "
          f"{r_min:>+.8f}    {r_prom:>+.8f}    {ratio:.6f}")

print("="*120)

# GRÁFICO COMPARATIVO ASCII
print(f"\n\n{'='*120}")
print("VISUALIZACIÓN: NÚMERO DE ARISTAS POR PARTICIÓN")
print(f"{'='*120}\n")

max_aristas = max([datos['num_aristas'] for datos in resultados_pearson.values()] + [0])

for nombre, datos in resultados_pearson.items():
    num_aristas = datos['num_aristas']

    if max_aristas > 0:
        barra_len = int((num_aristas / max_aristas) * 50)
    else:
        barra_len = 0

    barra = "█" * barra_len + "░" * (50 - barra_len)

    print(f"{nombre:<15} │{barra}│ {num_aristas:>3} aristas")

print("\n" + "="*120)

# ANÁLISIS POR RANGO DE CORRELACIÓN
print(f"\n{'='*120}")
print("DISTRIBUCIÓN POR RANGOS DE CORRELACIÓN (Sobre aristas filtradas)")
print(f"{'='*120}\n")

rangos = {
    f'Muy fuerte (r ≥ 0.9)': (0.9, 1.0),
    f'Fuerte (0.8 ≤ r < 0.9)': (0.8, 0.9),
    f'Moderada (0.7 ≤ r < 0.8)': (0.7, 0.8),
    f'Débil ({THRESHOLD} ≤ r < 0.7)': (THRESHOLD, 0.7),
}

for nombre, datos in resultados_pearson.items():
    print(f"\n{nombre.upper()}:")

    if len(datos['aristas']) > 0:
        total_aristas = len(datos['aristas'])
        rangos[f'Muy fuerte (r ≥ 0.9)'] = (0.9, 1.01)

        for rango_nombre, (min_r, max_r) in rangos.items():
            count = sum(1 for a in datos['aristas'] if min_r <= a['r_pearson'] < max_r or (rango_nombre.startswith('Muy') and a['r_pearson'] == 1.0))
            porcent = (count / total_aristas * 100) if total_aristas > 0 else 0

            barra_len_rango = int(porcent / 5)
            barra = "█" * barra_len_rango

            print(f"   {rango_nombre:<35} {barra:<20} {count:>3} ({porcent:.1f}%)")
    else:
        print(f"   Sin aristas")

print("\n" + "="*120)

# EXPORTAR RESULTADOS
print(f"\n{'='*120}")
print("GUARDANDO RESULTADOS")
print(f"{'='*120}\n")

# Guardar tabla resumida
with open('../results/pearson_resumen.csv', 'w', encoding='utf-8') as f:
    f.write(f"threshold,{THRESHOLD}\n")
    f.write("Partición,Filas,Aristas,r_Max,r_Min,r_Promedio\n")

    for nombre, datos in resultados_pearson.items():
        n = datos['n_datos']
        num_aristas = datos['num_aristas']

        if len(datos['aristas']) > 0:
            r_max = max([a['r_pearson'] for a in datos['aristas']])
            r_min = min([a['r_pearson'] for a in datos['aristas']])
            r_prom = sum([a['r_pearson'] for a in datos['aristas']]) / len(datos['aristas'])
        else:
            r_max = 0
            r_min = 0
            r_prom = 0

        f.write(f"{nombre},{n},{num_aristas},{r_max:.8f},{r_min:.8f},{r_prom:.8f}\n")

print("✅ Resumen guardado: ../results/pearson_resumen.csv")

# Guardar detalles de aristas por partición
for nombre, datos in resultados_pearson.items():
    filename_aristas = f"../results/pearson_aristas_{nombre}.csv"
    with open(filename_aristas, 'w', encoding='utf-8') as f:
        f.write("var1,var2,pearson_r\n")
        for arista in datos['aristas']:
            f.write(f"{arista['var1']},{arista['var2']},{arista['r_pearson']:.8f}\n")

    if len(datos['aristas']) > 0:
        print(f"✅ Aristas guardadas: {filename_aristas} ({len(datos['aristas'])} correlaciones)")
    else:
        print(f"⚠️   Sin aristas para: {nombre} (archivo vacío creado)")

    # Guardar la matriz de correlación completa
    filename_matriz = f"../results/pearson_matriz_completa_{nombre}.csv"
    if not datos['matriz_completa'].empty:
        datos['matriz_completa'].to_csv(filename_matriz, encoding='utf-8', float_format='%.8f')
        print(f"✅ Matriz completa guardada: {filename_matriz}")
    else:
        print(f"⚠️   No hay matriz completa para guardar para: {nombre}")

print(f"\n{'='*120}")
print("ANÁLISIS COMPLETADO")
print(f"{'='*120}\n")

# ========================================
# ALGORITMOS DE GRAFOS
# ========================================

class GraphAlgorithms:
    """Contiene implementaciones para MST (Kruskal y Prim)."""

    class DSU:
        """Helper class for Disjoint Set Union (DSU), used by Kruskal."""
        def __init__(self, nodes):
            self.parent = {node: node for node in nodes}

        def find(self, i):
            if self.parent[i] == i:
                return i
            self.parent[i] = self.find(self.parent[i])
            return self.parent[i]

        def union(self, x, y):
            root_x = self.find(x)
            root_y = self.find(y)
            if root_x != root_y:
                self.parent[root_x] = root_y
                return True
            return False

    @staticmethod
    def kruskal_mst(graph):
        """Calcula el MST usando Kruskal."""
        mst = nx.Graph()
        mst.add_nodes_from(graph.nodes())
        total_weight = 0

        edges = sorted(graph.edges(data=True), key=lambda t: t[2].get('weight', 1))

        dsu = GraphAlgorithms.DSU(graph.nodes())

        for u, v, data in edges:
            if dsu.union(u, v):
                mst.add_edge(u, v, **data)
                total_weight += data.get('weight', 1)

        return mst, total_weight

    @staticmethod
    def prim_mst(graph, start_node=None):
        """Calcula el MST usando Prim."""
        mst = nx.Graph()
        total_weight = 0

        if not graph.nodes():
            return mst, total_weight

        if start_node is None or start_node not in graph:
            valid_start_node = None
            for node in graph.nodes():
                if graph.degree(node) > 0:
                    valid_start_node = node
                    break
            if valid_start_node is None:
                if not graph.nodes():
                    return mst, total_weight
                valid_start_node = list(graph.nodes())[0]
            start_node = valid_start_node

        if start_node not in graph:
             return mst, total_weight

        mst.add_node(start_node)
        priority_queue = []

        for neighbor in graph.neighbors(start_node):
            weight = graph[start_node][neighbor].get('weight', 1)
            heapq.heappush(priority_queue, (weight, start_node, neighbor))

        while priority_queue and mst.number_of_nodes() < graph.number_of_nodes():
            weight, u, v = heapq.heappop(priority_queue)

            if v not in mst.nodes():
                mst.add_node(v)

                edge_data = graph[u][v].copy()
                edge_data.pop('weight', None)

                mst.add_edge(u, v, weight=weight, **edge_data)
                total_weight += weight

                for neighbor in graph.neighbors(v):
                    if neighbor not in mst.nodes():
                        neighbor_weight = graph[v][neighbor].get('weight', 1)
                        heapq.heappush(priority_queue, (neighbor_weight, v, neighbor))

        for node in graph.nodes():
            if node not in mst.nodes():
                mst.add_node(node)

        return mst, total_weight

# ========================================
# FUNCIÓN DE VISUALIZACIÓN CORREGIDA
# ========================================

def visualize_graphs_and_mst(aristas_filtradas, numerical_cols, partition_name, threshold):
    """
    Visualiza Grafo Original, MST Kruskal y MST Prim en configuración 1x3.
    VERSIÓN CORREGIDA: Sin zorder en draw_networkx_nodes()
    """
    print(f"\n--- Generando Grafos y MSTs para: {partition_name} ---")

    G = nx.Graph()
    G.add_nodes_from(numerical_cols)

    if not aristas_filtradas:
        print(f"  ⚠️  Sin aristas significativas (r >= {threshold}). El grafo no tendrá aristas.")
    else:
        for arista in aristas_filtradas:
            r = arista['r_pearson']
            weight = 1.0 - r
            G.add_edge(arista['var1'], arista['var2'], weight=weight, correlation=r)

    # Calcular MSTs
    mst_kruskal, peso_k = GraphAlgorithms.kruskal_mst(G)
    mst_prim, peso_p = GraphAlgorithms.prim_mst(G)

    print(f"  ✓ Grafo Original: {G.number_of_nodes()} nodos, {G.number_of_edges()} aristas")
    print(f"  ✓ MST Kruskal: {mst_kruskal.number_of_nodes()} nodos, {mst_kruskal.number_of_edges()} aristas, Peso: {peso_k:.4f}")
    print(f"  ✓ MST Prim: {mst_prim.number_of_nodes()} nodos, {mst_prim.number_of_edges()} aristas, Peso: {peso_p:.4f}")

    # Crear figura 1x3
    fig, axes = plt.subplots(1, 3, figsize=(22, 8))
    fig.suptitle(f"Análisis de Grafo: {partition_name} (r >= {threshold})", fontsize=16, fontweight='bold')

    # Calcular layout
    k_layout = 1.5 / math.sqrt(max(1, G.number_of_nodes()))
    pos = nx.spring_layout(G, seed=42, k=k_layout, iterations=50)

    # Constantes de estilo
    node_size = 2000
    node_alpha = 0.85
    font_size = 10

    # ==========================================
    # 1. GRAFO ORIGINAL
    # ==========================================
    ax1 = axes[0]

    # Dibujar aristas con grosor dinámico
    if G.number_of_edges() > 0:
        edge_correlations = [d['correlation'] for u, v, d in G.edges(data=True)]
        max_width = 4.0
        min_width = 0.5
        range_denominator = (1.0 - threshold) if (1.0 - threshold) > 0 else 1.0
        edge_widths = [
            min_width + (corr - threshold) * (max_width - min_width) / range_denominator
            for corr in edge_correlations
        ]

        nx.draw_networkx_edges(
            G, pos, ax=ax1,
            edgelist=G.edges(),
            width=edge_widths,
            edge_color=edge_correlations,
            edge_cmap=plt.cm.Greens,
            edge_vmin=threshold, edge_vmax=1.0,
            alpha=0.7
        )

        # Etiquetas de aristas
        edge_labels = {
            (u, v): f"{d['correlation']:.2f}"
            for u, v, d in G.edges(data=True)
        }
        nx.draw_networkx_edge_labels(
            G, pos, ax=ax1,
            edge_labels=edge_labels,
            font_size=font_size - 2,
            font_color='black'
        )

    # Dibujar nodos (SIN zorder)
    nx.draw_networkx_nodes(
        G, pos, ax=ax1, node_color='skyblue',
        node_size=node_size, alpha=node_alpha,
        edgecolors='black', linewidths=2
    )

    # Etiquetas de nodos
    nx.draw_networkx_labels(
        G, pos, ax=ax1,
        font_size=font_size, font_weight='bold'
    )

    ax1.set_title(f"Grafo de Correlación Original\n({G.number_of_edges()} aristas)",
                  fontsize=12, fontweight='bold', pad=10)
    ax1.axis('off')

    # ==========================================
    # 2. MST KRUSKAL
    # ==========================================
    ax2 = axes[1]

    # Dibujar aristas
    nx.draw_networkx_edges(
        mst_kruskal, pos, ax=ax2,
        width=3, alpha=0.8, edge_color='darkgreen'
    )

    # Dibujar nodos (SIN zorder)
    nx.draw_networkx_nodes(
        mst_kruskal, pos, ax=ax2, node_color='lightgreen',
        node_size=node_size, alpha=node_alpha,
        edgecolors='darkgreen', linewidths=2
    )

    # Etiquetas
    nx.draw_networkx_labels(
        mst_kruskal, pos, ax=ax2,
        font_size=font_size, font_weight='bold'
    )

    ax2.set_title(f"Árbol Expansión Mínima (Kruskal)\n({mst_kruskal.number_of_edges()} aristas)",
                  fontsize=12, fontweight='bold', pad=10)
    ax2.axis('off')

    # ==========================================
    # 3. MST PRIM
    # ==========================================
    ax3 = axes[2]

    # Dibujar aristas
    nx.draw_networkx_edges(
        mst_prim, pos, ax=ax3,
        width=3, alpha=0.8, edge_color='darkred'
    )

    # Dibujar nodos (SIN zorder)
    nx.draw_networkx_nodes(
        mst_prim, pos, ax=ax3, node_color='lightcoral',
        node_size=node_size, alpha=node_alpha,
        edgecolors='darkred', linewidths=2
    )

    # Etiquetas
    nx.draw_networkx_labels(
        mst_prim, pos, ax=ax3,
        font_size=font_size, font_weight='bold'
    )

    ax3.set_title(f"Árbol Expansión Mínima (Prim)\n({mst_prim.number_of_edges()} aristas)",
                  fontsize=12, fontweight='bold', pad=10)
    ax3.axis('off')

    plt.tight_layout()
    plt.savefig(f'../results/grafos_mst_{partition_name}.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Gráficos para {partition_name} guardados en ../results/grafos_mst_{partition_name}.png\n")

# ========================================
# FUNCIÓN PRINCIPAL
# ========================================

def mostrar_grafos_de_resultados():
    """Genera visualizaciones para todas las particiones."""
    print("\n" + "="*90)
    print("VISUALIZACIÓN DE GRAFOS: ORIGINAL, KRUSKAL Y PRIM")
    print("="*90)

    try:
        if not resultados_pearson:
             print("\n⚠️ ADVERTENCIA: 'resultados_pearson' está vacía.")
             return

        print(f"\n✓ Se encontraron {len(resultados_pearson)} particiones.\n")

        # Iterar sobre resultados
        for partition_name, data in resultados_pearson.items():

            aristas = data.get('aristas', [])
            threshold = data.get('threshold', 0.01)

            # Obtener columnas numéricas
            partition_df = particiones_analizar.get(partition_name)
            if partition_df is None:
                print(f"⚠️  Saltando {partition_name}: No se encuentra en particiones_analizar")
                continue

            numerical_cols = []
            for col in partition_df.columns:
                try:
                    _ = float(partition_df[col].iloc[0])
                    numerical_cols.append(col)
                except:
                    pass

            # Visualizar
            visualize_graphs_and_mst(aristas, numerical_cols, partition_name, threshold)

        print("="*90)
        print("✅ VISUALIZACIÓN DE GRAFOS COMPLETADA")
        print("="*90)

    except Exception as e:
        print(f"\n❌ Error inesperado: {e}")
        traceback.print_exc()

# Ejecutar visualización de grafos
mostrar_grafos_de_resultados()