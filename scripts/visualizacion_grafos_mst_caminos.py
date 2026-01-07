import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import traceback
import os

print("\n" + "="*100)
print("VISUALIZACIÓN: GRAFO ORIGINAL + MST KRUSKAL + MST PRIM CON CAMINOS MÁS LARGOS RESALTADOS")
print("="*100)

# Verificar disponibilidad de pydot/graphviz
layout_jerarquico_disponible = False
try:
    from networkx.drawing import nx_pydot
    layout_jerarquico_disponible = True
    print("✓ Se encontró 'nx_pydot'. Usando layout jerárquico 'dot'.\n")
except ImportError:
    print("⚠️  'nx_pydot' no disponible. Usando layout de resorte.\n")

# =================================================================
# FUNCIÓN AUXILIAR PARA ENCONTRAR EL CAMINO MÁS LARGO (DIÁMETRO)
# =================================================================

def encontrar_camino_mas_largo(T):
    """
    Encuentra el camino más largo (diámetro) entre dos hojas en un árbol.
    Devuelve el camino (lista de nodos) y su longitud (número de aristas).
    """
    if T.number_of_edges() == 0:
        return [], 0

    # Encontrar todas las hojas (nodos con grado 1)
    leaves = [node for node in T.nodes() if T.degree(node) == 1]

    # Caso especial: un grafo de 2 nodos y 1 arista (ambos son hojas)
    if len(leaves) < 2 and T.number_of_nodes() == 2:
        leaves = list(T.nodes())
    elif len(leaves) < 2:
        return [], 0

    longest_path = []
    max_len = -1

    # Encontrar el camino más largo entre todos los pares de hojas
    for i in range(len(leaves)):
        for j in range(i + 1, len(leaves)):
            try:
                path = nx.shortest_path(T, source=leaves[i], target=leaves[j])
                path_len = len(path) - 1  # Longitud en aristas

                if path_len > max_len:
                    max_len = path_len
                    longest_path = path
            except nx.NetworkXNoPath:
                continue

    return longest_path, max_len


# ========================================
# FUNCIÓN PRINCIPAL DE VISUALIZACIÓN
# ========================================

def visualizar_grafos_mst_particion(partition_name, aristas_filtradas, numerical_cols, numero_paso, threshold):
    """
    Visualiza para una partición:
    1. Grafo original de correlaciones
    2. MST Kruskal con camino resaltado
    3. MST Prim con camino resaltado
    """

    print(f"\n{'='*100}")
    print(f"PASO {numero_paso}: PARTICIÓN '{partition_name.upper()}' (threshold={threshold})")
    print(f"{'='*100}\n")

    # ==========================================
    # 1. CREAR GRAFO ORIGINAL
    # ==========================================
    print(f"1. Creando grafo de correlaciones...")

    G_temp = nx.Graph()
    G_temp.add_nodes_from(numerical_cols)

    if not aristas_filtradas:
        print(f"   ⚠️  Sin aristas significativas. Saltando.")
        return

    for arista in aristas_filtradas:
        r = arista['r_pearson']
        weight = 1.0 - r  # Para MST (busca peso mínimo)
        G_temp.add_edge(arista['var1'], arista['var2'], weight=weight, correlation=r)

    print(f"   ✓ Grafo: {G_temp.number_of_nodes()} nodos, {G_temp.number_of_edges()} aristas")

    # ==========================================
    # 2. CALCULAR MST KRUSKAL Y PRIM
    # ==========================================
    print(f"\n2. Calculando MST Kruskal y Prim...")

    if G_temp.number_of_edges() == 0:
        print(f"   ✗ Sin aristas. Saltando.")
        return

    # MST Kruskal
    mst_kruskal_temp = nx.minimum_spanning_tree(G_temp, weight='weight', algorithm='kruskal')
    peso_kruskal = sum(data['weight'] for u, v, data in mst_kruskal_temp.edges(data=True))

    # MST Prim
    mst_prim_temp = nx.minimum_spanning_tree(G_temp, weight='weight', algorithm='prim')
    peso_prim = sum(data['weight'] for u, v, data in mst_prim_temp.edges(data=True))

    print(f"   ✓ Kruskal: {mst_kruskal_temp.number_of_nodes()} nodos, {mst_kruskal_temp.number_of_edges()} aristas, Peso: {peso_kruskal:.4f}")
    print(f"   ✓ Prim: {mst_prim_temp.number_of_nodes()} nodos, {mst_prim_temp.number_of_edges()} aristas, Peso: {peso_prim:.4f}")

    # ==========================================
    # 3. ENCONTRAR CAMINOS MÁS LARGOS
    # ==========================================
    print(f"\n3. Encontrando caminos más largos (diámetro)...")

    path_k, len_k = encontrar_camino_mas_largo(mst_kruskal_temp)
    print(f"   ✓ Kruskal - Camino más largo ({len_k} aristas): {' → '.join(path_k)}")

    path_p, len_p = encontrar_camino_mas_largo(mst_prim_temp)
    print(f"   ✓ Prim - Camino más largo ({len_p} aristas): {' → '.join(path_p)}")

    # ==========================================
    # 4. CALCULAR LAYOUTS
    # ==========================================
    print(f"\n4. Calculando layouts...")

    pos_original = None
    pos_kruskal = None
    pos_prim = None
    layout_name = "Spring"

    if layout_jerarquico_disponible:
        try:
            pos_original = nx_pydot.graphviz_layout(G_temp, prog='neato')
            pos_kruskal = nx_pydot.graphviz_layout(mst_kruskal_temp, prog='dot')
            pos_prim = nx_pydot.graphviz_layout(mst_prim_temp, prog='dot')
            layout_name = "Jerárquico (dot)"
            print(f"   ✓ Layout jerárquico aplicado")
        except Exception as e:
            print(f"   ⚠️  Layout 'dot' falló. Usando Spring Layout.")
            pos_original = nx.spring_layout(G_temp, seed=42, k=2, iterations=200)
            pos_kruskal = nx.spring_layout(mst_kruskal_temp, seed=42, k=2, iterations=200)
            pos_prim = nx.spring_layout(mst_prim_temp, seed=42, k=2, iterations=200)
            layout_name = "Spring (Fallback)"
    else:
        print(f"   ⚠️  nx_pydot no disponible. Usando Spring Layout.")
        pos_original = nx.spring_layout(G_temp, seed=42, k=2, iterations=200)
        pos_kruskal = nx.spring_layout(mst_kruskal_temp, seed=42, k=2, iterations=200)
        pos_prim = nx.spring_layout(mst_prim_temp, seed=42, k=2, iterations=200)
        layout_name = "Spring (Fallback)"

    # ==========================================
    # 5. CREAR VISUALIZACIÓN 1x3
    # ==========================================
    print(f"\n5. Generando visualización...")

    fig, axes = plt.subplots(1, 3, figsize=(26, 10))
    fig.suptitle(f"PARTICIÓN: {partition_name.upper()} | Filas: {len(aristas_filtradas)} | Threshold: {threshold} | Layout: {layout_name}",
                 fontsize=17, fontweight='bold', y=0.98)

    # --- GRAFO 1: ORIGINAL ---
    ax1 = axes[0]

    if G_temp.number_of_edges() > 0:
        edge_widths_original = [abs(G_temp[u][v].get('weight', 1)) * 6 for u, v in G_temp.edges()]
        edge_colors_original = ['#0066cc' if G_temp[u][v].get('correlation', 1) > 0 else '#ff3333'
                                for u, v in G_temp.edges()]

        nx.draw_networkx_edges(G_temp, pos_original, ax=ax1, width=edge_widths_original,
                               edge_color=edge_colors_original, alpha=0.6)

    nx.draw_networkx_nodes(G_temp, pos_original, ax=ax1, node_size=2500, node_color='lightblue',
                           edgecolors='navy', linewidths=2.5)
    nx.draw_networkx_labels(G_temp, pos_original, ax=ax1, font_size=11, font_weight='bold')

    ax1.set_title(f"GRAFO ORIGINAL\nNodos: {G_temp.number_of_nodes()} | Aristas: {G_temp.number_of_edges()}",
                  fontsize=13, fontweight='bold', pad=10)
    ax1.axis('off')

    # --- GRAFO 2: MST KRUSKAL CON CAMINO RESALTADO ---
    ax2 = axes[1]

    path_nodes_k = set(path_k)
    other_nodes_k = set(mst_kruskal_temp.nodes()) - path_nodes_k

    path_edges_k = list(zip(path_k[:-1], path_k[1:]))
    other_edges_k = set(mst_kruskal_temp.edges())
    for u, v in path_edges_k:
        other_edges_k.discard((u, v))
        other_edges_k.discard((v, u))

    # Dibujar aristas normales
    if other_edges_k:
        nx.draw_networkx_edges(mst_kruskal_temp, pos_kruskal, ax=ax2, edgelist=other_edges_k,
                               width=3, edge_color='#00aa00', alpha=0.6)

    # Dibujar nodos normales
    if other_nodes_k:
        nx.draw_networkx_nodes(mst_kruskal_temp, pos_kruskal, ax=ax2, nodelist=other_nodes_k,
                               node_size=2500, node_color='lightgreen',
                               edgecolors='darkgreen', linewidths=2.5)

    # Dibujar aristas del camino
    if path_edges_k:
        nx.draw_networkx_edges(mst_kruskal_temp, pos_kruskal, ax=ax2, edgelist=path_edges_k,
                               width=7, edge_color='red', alpha=1.0, style='solid')

    # Dibujar nodos del camino
    if path_nodes_k:
        nx.draw_networkx_nodes(mst_kruskal_temp, pos_kruskal, ax=ax2, nodelist=path_nodes_k,
                               node_size=2800, node_color='gold',
                               edgecolors='black', linewidths=3)

    nx.draw_networkx_labels(mst_kruskal_temp, pos_kruskal, ax=ax2, font_size=11, font_weight='bold')

    ax2.set_title(f"MST KRUSKAL (Camino máximo: {len_k})\nNodos: {mst_kruskal_temp.number_of_nodes()} | Aristas: {mst_kruskal_temp.number_of_edges()}",
                  fontsize=13, fontweight='bold', pad=10)
    ax2.axis('off')

    # --- GRAFO 3: MST PRIM CON CAMINO RESALTADO ---
    ax3 = axes[2]

    path_nodes_p = set(path_p)
    other_nodes_p = set(mst_prim_temp.nodes()) - path_nodes_p

    path_edges_p = list(zip(path_p[:-1], path_p[1:]))
    other_edges_p = set(mst_prim_temp.edges())
    for u, v in path_edges_p:
        other_edges_p.discard((u, v))
        other_edges_p.discard((v, u))

    # Dibujar aristas normales
    if other_edges_p:
        nx.draw_networkx_edges(mst_prim_temp, pos_prim, ax=ax3, edgelist=other_edges_p,
                               width=3, edge_color='#cc0066', alpha=0.6)

    # Dibujar nodos normales
    if other_nodes_p:
        nx.draw_networkx_nodes(mst_prim_temp, pos_prim, ax=ax3, nodelist=other_nodes_p,
                               node_size=2500, node_color='lightcoral',
                               edgecolors='darkred', linewidths=2.5)

    # Dibujar aristas del camino
    if path_edges_p:
        nx.draw_networkx_edges(mst_prim_temp, pos_prim, ax=ax3, edgelist=path_edges_p,
                               width=7, edge_color='red', alpha=1.0, style='solid')

    # Dibujar nodos del camino
    if path_nodes_p:
        nx.draw_networkx_nodes(mst_prim_temp, pos_prim, ax=ax3, nodelist=path_nodes_p,
                               node_size=2800, node_color='gold',
                               edgecolors='black', linewidths=3)

    nx.draw_networkx_labels(mst_prim_temp, pos_prim, ax=ax3, font_size=11, font_weight='bold')

    ax3.set_title(f"MST PRIM (Camino máximo: {len_p})\nNodos: {mst_prim_temp.number_of_nodes()} | Aristas: {mst_prim_temp.number_of_edges()}",
                  fontsize=13, fontweight='bold', pad=10)
    ax3.axis('off')

    plt.tight_layout()

    filename = f"paso_{numero_paso:02d}_{partition_name}_grafos_mst_camino_resaltado.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"   ✓ Visualización guardada: '{filename}'")
    plt.show()

    # ==========================================
    # 6. ESTADÍSTICAS
    # ==========================================
    print(f"\n6. Estadísticas:")
    print(f"   {'─'*90}")

    degree_seq = [G_temp.degree(n) for n in G_temp.nodes()]
    print(f"   GRAFO ORIGINAL:")
    print(f"       • Densidad: {nx.density(G_temp):.4f}")
    print(f"       • Grado Promedio: {sum(degree_seq)/len(degree_seq) if degree_seq else 0:.2f}")
    print(f"       • Grado Máximo: {max(degree_seq) if degree_seq else 0}")

    print(f"\n   MST KRUSKAL:")
    print(f"       • Peso Total: {peso_kruskal:.4f}")
    print(f"       • Camino más largo: {len_k} aristas")

    print(f"\n   MST PRIM:")
    print(f"       • Peso Total: {peso_prim:.4f}")
    print(f"       • Camino más largo: {len_p} aristas")

    print(f"   {'─'*90}\n")

    return {
        'kruskal_camino_len': len_k,
        'prim_camino_len': len_p,
        'peso_kruskal': peso_kruskal,
        'peso_prim': peso_prim
    }


# ========================================
# EJECUTAR VISUALIZACIÓN
# ========================================

print("\nObtienendo datos del análisis previo...\n")

# Intentar cargar datos desde archivos CSV generados por el análisis de correlación
resultados_visualizacion = {}
paso = 0

# Lista de particiones a analizar (incluyendo df_original)
particiones_lista = ['df_original'] + list(particiones.keys()) if 'particiones' in globals() else ['df_original']

for partition_name in particiones_lista:
    paso += 1

    try:
        # Cargar aristas desde archivo CSV
        aristas_file = f'../results/pearson_aristas_{partition_name}.csv'
        if os.path.exists(aristas_file):
            aristas_df = pd.read_csv(aristas_file)
            aristas = []
            for _, row in aristas_df.iterrows():
                aristas.append({
                    'var1': row['var1'],
                    'var2': row['var2'],
                    'r_pearson': row['pearson_r']
                })

            # Usar threshold por defecto
            threshold = 0.6

            # Obtener columnas numéricas
            if partition_name == 'df_original':
                partition_df = df_original if 'df_original' in globals() else pd.read_csv('../data/dataset_ejemplo.csv')
            else:
                partition_df = particiones[partition_name] if 'particiones' in globals() and partition_name in particiones else pd.read_csv(f'../data/{partition_name}.csv')

            numerical_cols = []
            for col in partition_df.columns:
                try:
                    _ = float(partition_df[col].iloc[0])
                    numerical_cols.append(col)
                except:
                    pass

            # Visualizar
            resultado = visualizar_grafos_mst_particion(partition_name, aristas, numerical_cols, paso, threshold)
            if resultado:
                resultados_visualizacion[partition_name] = resultado

        else:
            print(f"⚠️  Archivo de aristas no encontrado: {aristas_file}")

    except Exception as e:
        print(f"\n⚠️  Error en '{partition_name}': {str(e)}")
        traceback.print_exc()
        print(f"   Continuando...\n")

    # ========================================
    # TABLA RESUMIDA
    # ========================================

    if resultados_visualizacion:
        print("\n" + "="*100)
        print("TABLA RESUMIDA: CAMINOS MÁS LARGOS Y PESOS MST")
        print("="*100 + "\n")

        tabla_resumen = []
        for partition_name, resultado in resultados_visualizacion.items():
            tabla_resumen.append({
                'Partición': partition_name,
                'Camino_Kruskal': resultado['kruskal_camino_len'],
                'Camino_Prim': resultado['prim_camino_len'],
                'Peso_Kruskal': f"{resultado['peso_kruskal']:.4f}",
                'Peso_Prim': f"{resultado['peso_prim']:.4f}"
            })

        df_resumen = pd.DataFrame(tabla_resumen)
        print(df_resumen.to_string(index=False))

        df_resumen.to_csv('resumen_caminos_mst.csv', index=False)
        print("\n✓ Tabla guardada: 'resumen_caminos_mst.csv'")

print("\n" + "="*100)
print("✓ VISUALIZACIÓN COMPLETADA")
print("="*100)