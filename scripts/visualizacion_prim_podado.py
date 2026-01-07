print("\n" + "="*100)
print("VISUALIZACIÓN: MST PRIM (COMPLETO + CAMINO) vs MST PRIM (PODADO)")
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

    leaves = [node for node in T.nodes() if T.degree(node) == 1]

    if len(leaves) < 2 and T.number_of_nodes() == 2:
        leaves = list(T.nodes())
    elif len(leaves) < 2:
        return [], 0

    longest_path = []
    max_len = -1

    for i in range(len(leaves)):
        for j in range(i + 1, len(leaves)):
            try:
                path = nx.shortest_path(T, source=leaves[i], target=leaves[j])
                path_len = len(path) - 1

                if path_len > max_len:
                    max_len = path_len
                    longest_path = path
            except nx.NetworkXNoPath:
                continue

    return longest_path, max_len


# =================================================================
# FUNCIÓN AUXILIAR PARA PODAR EL ÁRBOL
# =================================================================

def podar_arbol_por_la_mitad(T_original, nodo_raiz='y'):
    """
    Encuentra el camino más largo (diámetro), corta el árbol por la mitad
    de ese camino, y devuelve el sub-árbol que contiene el 'nodo_raiz'.
    """
    T = T_original.copy()

    if T.number_of_nodes() < 2 or T.number_of_edges() == 0:
        return T

    if nodo_raiz not in T.nodes():
        print(f"   ⚠️  Advertencia (Poda): Nodo raíz '{nodo_raiz}' no encontrado. Devolviendo MST completo.")
        return T

    # 1. Encontrar el camino más largo
    longest_path, path_len = encontrar_camino_mas_largo(T)

    if not longest_path:
        print(f"   ⚠️  Advertencia (Poda): No se encontró ningún camino. Devolviendo MST completo.")
        return T

    if path_len <= 1:
        print(f"   ℹ️  Info (Poda): Camino más largo ({path_len}) demasiado corto para cortar.")
        try:
            componente_y = nx.node_connected_component(T, nodo_raiz)
            return T.subgraph(componente_y).copy()
        except Exception:
            G_y = nx.Graph()
            G_y.add_node(nodo_raiz)
            return G_y

    # 2. Encontrar la arista a cortar (Lógica Par/Impar)
    target_edge_count = (path_len + (path_len % 2)) // 2

    node_index_a = target_edge_count - 1
    node_index_b = target_edge_count

    u, v = longest_path[node_index_a], longest_path[node_index_b]

    # 3. Cortar el árbol
    if T.has_edge(u, v):
        print(f"   ✂️  Poda: Camino más largo encontrado ({path_len} aristas).")
        print(f"   ✂️  Poda: Cortando arista #{target_edge_count}: ({u}, {v})")
        T.remove_edge(u, v)
    else:
        print(f"   ⚠️  Advertencia (Poda): La arista a cortar ({u}, {v}) no existe. Devolviendo MST completo.")
        return T

    # 4. Quedarse con el sub-árbol que contiene 'y'
    try:
        for component in nx.connected_components(T):
            if nodo_raiz in component:
                print(f"   ✓ Poda: Conservando componente con '{nodo_raiz}' ({len(component)} nodos).")
                return T.subgraph(component).copy()

        print(f"   ⚠️  Advertencia (Poda): Nodo '{nodo_raiz}' quedó aislado. Devolviendo solo el nodo.")
        G_y = nx.Graph()
        G_y.add_node(nodo_raiz)
        return G_y

    except Exception as e:
        print(f"   ⚠️  Error durante la selección de componente: {e}. Devolviendo MST completo.")
        return T_original


# ========================================
# FUNCIÓN PRINCIPAL DE VISUALIZACIÓN
# ========================================

def visualizar_prim_podado_particion(partition_name, aristas_filtradas, numerical_cols, numero_paso, threshold):
    """
    Visualiza para una partición:
    1. Grafo original
    2. MST Prim (completo con camino resaltado)
    3. MST Prim (podado)
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
    # 2. CALCULAR MST PRIM
    # ==========================================
    print(f"\n2. Calculando MST Prim...")

    if G_temp.number_of_edges() == 0:
        print(f"   ✗ Sin aristas. Saltando.")
        return

    mst_prim_temp = nx.minimum_spanning_tree(G_temp, weight='weight', algorithm='prim')
    peso_prim = sum(data['weight'] for u, v, data in mst_prim_temp.edges(data=True))

    print(f"   ✓ Prim: {mst_prim_temp.number_of_nodes()} nodos, {mst_prim_temp.number_of_edges()} aristas, Peso: {peso_prim:.4f}")

    # ==========================================
    # 3. ENCONTRAR CAMINO MÁS LARGO
    # ==========================================
    print(f"\n3. Encontrando camino más largo...")

    path_p, len_p = encontrar_camino_mas_largo(mst_prim_temp)
    print(f"   ✓ Camino más largo ({len_p} aristas): {' → '.join(path_p)}")

    # ==========================================
    # 4. PODAR EL ÁRBOL
    # ==========================================
    print(f"\n4. Podando MST Prim...")

    mst_prim_podado = podar_arbol_por_la_mitad(mst_prim_temp, nodo_raiz='y')

    # ==========================================
    # 5. CALCULAR LAYOUTS
    # ==========================================
    print(f"\n5. Calculando layouts...")

    pos_original = None
    pos_prim = None
    pos_prim_podado = None
    layout_name = "Spring"

    if layout_jerarquico_disponible:
        try:
            pos_original = nx_pydot.graphviz_layout(G_temp, prog='neato')
            pos_prim = nx_pydot.graphviz_layout(mst_prim_temp, prog='dot')
            pos_prim_podado = nx_pydot.graphviz_layout(mst_prim_podado, prog='dot')
            layout_name = "Jerárquico (dot)"
            print(f"   ✓ Layout jerárquico aplicado")
        except Exception as e:
            print(f"   ⚠️  Layout 'dot' falló. Usando Spring Layout.")
            pos_original = nx.spring_layout(G_temp, seed=42, k=2, iterations=200)
            pos_prim = nx.spring_layout(mst_prim_temp, seed=42, k=2, iterations=200)
            pos_prim_podado = nx.spring_layout(mst_prim_podado, seed=42, k=2, iterations=200)
            layout_name = "Spring (Fallback)"
    else:
        print(f"   ⚠️  nx_pydot no disponible. Usando Spring Layout.")
        pos_original = nx.spring_layout(G_temp, seed=42, k=2, iterations=200)
        pos_prim = nx.spring_layout(mst_prim_temp, seed=42, k=2, iterations=200)
        pos_prim_podado = nx.spring_layout(mst_prim_podado, seed=42, k=2, iterations=200)
        layout_name = "Spring (Fallback)"

    # ==========================================
    # 6. CREAR VISUALIZACIÓN 1x3
    # ==========================================
    print(f"\n6. Generando visualización...")

    fig, axes = plt.subplots(1, 3, figsize=(26, 10))
    fig.suptitle(f"PARTICIÓN: {partition_name.upper()} | Threshold: {threshold} | Layout: {layout_name}",
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

    # --- GRAFO 2: MST PRIM (COMPLETO CON CAMINO RESALTADO) ---
    ax2 = axes[1]

    path_nodes_p = set(path_p)
    other_nodes_p = set(mst_prim_temp.nodes()) - path_nodes_p

    path_edges_p = list(zip(path_p[:-1], path_p[1:]))
    other_edges_p = set(mst_prim_temp.edges())
    for u, v in path_edges_p:
        other_edges_p.discard((u, v))
        other_edges_p.discard((v, u))

    # Dibujar aristas normales
    if other_edges_p:
        nx.draw_networkx_edges(mst_prim_temp, pos_prim, ax=ax2, edgelist=other_edges_p,
                               width=3, edge_color='#cc0066', alpha=0.6)

    # Dibujar nodos normales
    if other_nodes_p:
        nx.draw_networkx_nodes(mst_prim_temp, pos_prim, ax=ax2, nodelist=other_nodes_p,
                               node_size=2500, node_color='lightcoral',
                               edgecolors='darkred', linewidths=2.5)

    # Dibujar aristas del camino
    if path_edges_p:
        nx.draw_networkx_edges(mst_prim_temp, pos_prim, ax=ax2, edgelist=path_edges_p,
                               width=7, edge_color='red', alpha=1.0, style='solid')

    # Dibujar nodos del camino
    if path_nodes_p:
        nx.draw_networkx_nodes(mst_prim_temp, pos_prim, ax=ax2, nodelist=path_nodes_p,
                               node_size=2800, node_color='gold',
                               edgecolors='black', linewidths=3)

    nx.draw_networkx_labels(mst_prim_temp, pos_prim, ax=ax2, font_size=11, font_weight='bold')

    ax2.set_title(f"MST PRIM (Camino máximo: {len_p})\nNodos: {mst_prim_temp.number_of_nodes()} | Aristas: {mst_prim_temp.number_of_edges()}",
                  fontsize=13, fontweight='bold', pad=10)
    ax2.axis('off')

    # --- GRAFO 3: MST PRIM (PODADO) ---
    ax3 = axes[2]

    if mst_prim_podado.number_of_edges() > 0:
        edge_widths_podado = [abs(mst_prim_podado[u][v].get('weight', 1)) * 6 for u, v in mst_prim_podado.edges()]
        edge_colors_podado = ['#00aa00' if mst_prim_podado[u][v].get('correlation', 1) > 0 else '#ff6600'
                               for u, v in mst_prim_podado.edges()]
        nx.draw_networkx_edges(mst_prim_podado, pos_prim_podado, ax=ax3, width=edge_widths_podado,
                               edge_color=edge_colors_podado, alpha=0.85, style='solid')

    nx.draw_networkx_nodes(mst_prim_podado, pos_prim_podado, ax=ax3, node_size=2500, node_color='lightgreen',
                           edgecolors='darkgreen', linewidths=2.5)
    nx.draw_networkx_labels(mst_prim_podado, pos_prim_podado, ax=ax3, font_size=11, font_weight='bold')

    ax3.set_title(f"MST PRIM (PODADO)\nNodos: {mst_prim_podado.number_of_nodes()} | Aristas: {mst_prim_podado.number_of_edges()}",
                  fontsize=13, fontweight='bold', pad=10)
    ax3.axis('off')

    plt.tight_layout()

    filename = f"paso_{numero_paso:02d}_{partition_name}_prim_poda_comparacion.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"   ✓ Visualización guardada: '{filename}'")
    plt.show()

    # ==========================================
    # 7. ESTADÍSTICAS
    # ==========================================
    print(f"\n7. Estadísticas:")
    print(f"   {'─'*90}")

    degree_seq = [G_temp.degree(n) for n in G_temp.nodes()]
    print(f"   GRAFO ORIGINAL:")
    print(f"       • Densidad: {nx.density(G_temp):.4f}")
    print(f"       • Grado Promedio: {sum(degree_seq)/len(degree_seq) if degree_seq else 0:.2f}")

    print(f"\n   MST PRIM (Original):")
    print(f"       • Peso Total: {peso_prim:.4f}")
    print(f"       • Camino más largo: {len_p} aristas")

    print(f"\n   MST PRIM (Podado):")
    print(f"       • Nodos restantes: {mst_prim_podado.number_of_nodes()}")
    print(f"       • Aristas restantes: {mst_prim_podado.number_of_edges()}")

    print(f"   {'─'*90}\n")

    return {
        'camino_len': len_p,
        'peso_prim': peso_prim,
        'nodos_podado': mst_prim_podado.number_of_nodes(),
        'aristas_podado': mst_prim_podado.number_of_edges()
    }


# ========================================
# EJECUTAR VISUALIZACIÓN
# ========================================

print("\nObtienendo datos del análisis previo...\n")

if 'resultados_pearson' not in globals():
    print("⚠️  ERROR: 'resultados_pearson' no se encontró.")
    print("Asegúrate de ejecutar primero el análisis de correlación.")
else:
    resultados_poda = {}
    paso = 0

    for partition_name, data in resultados_pearson.items():
        paso += 1

        try:
            aristas = data.get('aristas', [])
            threshold = data.get('threshold', THRESHOLD if 'THRESHOLD' in globals() else 0.75)

            # Obtener columnas numéricas
            if partition_name in particiones_analizar:
                partition_df = particiones_analizar[partition_name]
                numerical_cols = []
                for col in partition_df.columns:
                    try:
                        _ = float(partition_df[col].iloc[0])
                        numerical_cols.append(col)
                    except:
                        pass

                # Visualizar
                resultado = visualizar_prim_podado_particion(partition_name, aristas, numerical_cols, paso, threshold)
                if resultado:
                    resultados_poda[partition_name] = resultado

        except Exception as e:
            print(f"\n⚠️  Error en '{partition_name}': {str(e)}")
            traceback.print_exc()
            print(f"   Continuando...\n")

    # ========================================
    # TABLA RESUMIDA
    # ========================================

    if resultados_poda:
        print("\n" + "="*100)
        print("TABLA RESUMIDA: COMPARACIÓN PRIM ORIGINAL vs PODADO")
        print("="*100 + "\n")

        tabla_resumen = []
        for partition_name, resultado in resultados_poda.items():
            tabla_resumen.append({
                'Partición': partition_name,
                'Camino_Largo': resultado['camino_len'],
                'Peso_Prim': f"{resultado['peso_prim']:.4f}",
                'Nodos_Podado': resultado['nodos_podado'],
                'Aristas_Podado': resultado['aristas_podado']
            })

        df_resumen = pd.DataFrame(tabla_resumen)
        print(df_resumen.to_string(index=False))

        df_resumen.to_csv('resumen_poda_prim.csv', index=False)
        print("\n✓ Tabla guardada: 'resumen_poda_prim.csv'")

    print("\n" + "="*100)
    print("✓ VISUALIZACIÓN PODA COMPLETADA")


# =================================================================
# ETAPA 4: COMPARACIÓN GRÁFICA DE ÁRBOLES PODADOS (DIFERENCIAS)
# Reutiliza árboles ya calculados por ETAPA 3 cuando estén disponibles
# =================================================================
print("\n" + "="*80)
print("ETAPA 4: IDENTIFICACIÓN DE DIFERENCIAS EN ÁRBOLES PODADOS")
print("="*80 + "\n")

if 'mst_prim_trees' not in globals() or 'pruned_mst_trees' not in globals():
    print("⚠️  ADVERTENCIA: Datos de ETAPA 3 no encontrados.")
    print("   Se generarán los árboles podados localmente.")
    print("   (Asegúrate de ejecutar ETAPA 3 primero para máxima eficiencia)\n")

    mst_prim_trees = globals().get('mst_prim_trees', {})
    pruned_mst_trees = globals().get('pruned_mst_trees', {})
    USAR_DATOS_ETAPA3 = False
else:
    print("✓ Se encontraron datos de ETAPA 3.")
    print(f"✓ Árboles MST disponibles: {list(mst_prim_trees.keys())}")
    print(f"✓ Árboles podados disponibles: {list(pruned_mst_trees.keys())}\n")
    USAR_DATOS_ETAPA3 = True


def encontrar_camino_mas_largo(T):
    if T.number_of_edges() == 0:
        return [], 0

    leaves = [node for node in T.nodes() if T.degree(node) == 1]
    if len(leaves) < 2:
        if T.number_of_nodes() == 1:
            return list(T.nodes()), 0
        elif T.number_of_nodes() == 2:
            return list(T.nodes()), 1
        else:
            return [], 0

    longest_path = []
    max_len = -1

    for i in range(len(leaves)):
        for j in range(i + 1, len(leaves)):
            try:
                path = nx.shortest_path(T, source=leaves[i], target=leaves[j])
                path_len = len(path) - 1
                if path_len > max_len:
                    max_len = path_len
                    longest_path = path
            except nx.NetworkXNoPath:
                continue

    return longest_path, max_len


def podar_arbol_por_la_mitad(T_original, nodo_raiz_fijo='y'):
    T = T_original.copy()

    if T.number_of_nodes() < 2 or T.number_of_edges() == 0:
        return T

    if nodo_raiz_fijo not in T.nodes():
        if T.number_of_nodes() > 0:
            return T.subgraph([list(T.nodes())[0]]).copy()
        return T

    longest_path, path_len = encontrar_camino_mas_largo(T)

    if not longest_path or path_len <= 1:
        try:
            component = nx.node_connected_component(T, nodo_raiz_fijo)
            return T.subgraph(component).copy()
        except Exception:
            G_y = nx.Graph()
            G_y.add_node(nodo_raiz_fijo)
            return G_y

    target_edge_count = (path_len + (path_len % 2)) // 2
    node_index_a = target_edge_count - 1
    node_index_b = target_edge_count

    if node_index_b >= len(longest_path):
        return T_original.copy()

    u, v = longest_path[node_index_a], longest_path[node_index_b]

    if T.has_edge(u, v):
        T.remove_edge(u, v)

    try:
        for component in nx.connected_components(T):
            if nodo_raiz_fijo in component:
                return T.subgraph(component).copy()
        G_y = nx.Graph()
        G_y.add_node(nodo_raiz_fijo)
        return G_y
    except Exception:
        return T_original.copy()


def obtener_estructura_arbol(T, root):
    estructura = {}
    if T.number_of_nodes() == 0:
        return estructura

    try:
        for nodo in T.nodes():
            try:
                nivel = nx.shortest_path_length(T, root, nodo)
                estructura[nodo] = nivel
            except nx.NetworkXNoPath:
                estructura[nodo] = None
    except Exception as e:
        print(f"   ⚠️  Error obtener estructura: {e}")

    return estructura


def comparar_estructuras(struct_b, struct_w, nombre_b, nombre_w):
    diferencias = []

    nodos_comunes = set(struct_b.keys()) & set(struct_w.keys())
    solo_en_b = set(struct_b.keys()) - set(struct_w.keys())
    solo_en_w = set(struct_w.keys()) - set(struct_b.keys())

    for nodo in nodos_comunes:
        nivel_b = struct_b[nodo]
        nivel_w = struct_w[nodo]

        if nivel_b != nivel_w:
            diferencias.append({
                'Tipo': 'Diferente Nivel',
                'Nodo': nodo,
                f'Nivel en {nombre_b}': nivel_b,
                f'Nivel en {nombre_w}': nivel_w,
                'Diferencia': abs(nivel_b - nivel_w) if (nivel_b is not None and nivel_w is not None) else 'N/A'
            })

    for nodo in solo_en_b:
        diferencias.append({
            'Tipo': f'Solo en {nombre_b}',
            'Nodo': nodo,
            f'Nivel en {nombre_b}': struct_b[nodo],
            f'Nivel en {nombre_w}': '-',
            'Diferencia': 'Ausente'
        })

    for nodo in solo_en_w:
        diferencias.append({
            'Tipo': f'Solo en {nombre_w}',
            'Nodo': nodo,
            f'Nivel en {nombre_b}': '-',
            f'Nivel en {nombre_w}': struct_w[nodo],
            'Diferencia': 'Ausente'
        })

    if not diferencias:
        return None

    df = pd.DataFrame(diferencias)
    return df.sort_values('Nodo').reset_index(drop=True)


def visualizar_comparacion(T_b, T_w, root_b, root_w, nombre_b, nombre_w, df_diferencias):
    print(f"\n   🎨 Generando visualización comparativa...")

    nodos_b = set(T_b.nodes())
    nodos_w = set(T_w.nodes())

    if df_diferencias is not None:
        nodos_diferentes = set(df_diferencias['Nodo'].unique())
    else:
        nodos_diferentes = set()

    comunes = nodos_b.intersection(nodos_w) - nodos_diferentes
    solo_en_b = nodos_b - nodos_w
    solo_en_w = nodos_w - nodos_b

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 12))
    fig.suptitle(f"Comparación: {nombre_b} vs {nombre_w} | Diferencias Destacadas",
                 fontsize=16, fontweight='bold')

    pos_b = nx.spring_layout(T_b, seed=42, k=2, iterations=50) if T_b.number_of_nodes() > 0 else {}
    pos_w = nx.spring_layout(T_w, seed=42, k=2, iterations=50) if T_w.number_of_nodes() > 0 else {}

    ax1.set_title(f"{nombre_b} ({len(nodos_b)} nodos)", fontsize=14, fontweight='bold')
    if pos_b:
        if comunes:
            nx.draw_networkx_nodes(T_b, pos_b, ax=ax1, nodelist=comunes, node_size=2000,
                                  node_color='lightgray', edgecolors='black', linewidths=1, label='Igual posición')

        nodos_diff_b = nodos_diferentes & nodos_b
        if nodos_diff_b:
            nx.draw_networkx_nodes(T_b, pos_b, ax=ax1, nodelist=nodos_diff_b, node_size=2800,
                                  node_color='#FF6B6B', edgecolors='darkred', linewidths=3, label='Diferencia detectada')

        if solo_en_b:
            nx.draw_networkx_nodes(T_b, pos_b, ax=ax1, nodelist=solo_en_b, node_size=2800,
                                  node_color='#FFD93D', edgecolors='#FF9F1C', linewidths=3, label=f'Solo en {nombre_b}')

        if root_b and root_b in T_b.nodes():
            nx.draw_networkx_nodes(T_b, pos_b, ax=ax1, nodelist=[root_b], node_size=3200,
                                  node_color='gold', edgecolors='black', linewidths=3, label='Raíz')

        nx.draw_networkx_edges(T_b, pos_b, ax=ax1, edge_color='gray', width=2, alpha=0.6)
        nx.draw_networkx_labels(T_b, pos_b, ax=ax1, font_size=9, font_weight='bold')
        ax1.legend(loc='upper left', fontsize=10)
    ax1.axis('off')

    ax2.set_title(f"{nombre_w} ({len(nodos_w)} nodos)", fontsize=14, fontweight='bold')
    if pos_w:
        if comunes:
            nx.draw_networkx_nodes(T_w, pos_w, ax=ax2, nodelist=comunes, node_size=2000,
                                  node_color='lightgray', edgecolors='black', linewidths=1, label='Igual posición')

        nodos_diff_w = nodos_diferentes & nodos_w
        if nodos_diff_w:
            nx.draw_networkx_nodes(T_w, pos_w, ax=ax2, nodelist=nodos_diff_w, node_size=2800,
                                  node_color='#FF6B6B', edgecolors='darkred', linewidths=3, label='Diferencia detectada')

        if solo_en_w:
            nx.draw_networkx_nodes(T_w, pos_w, ax=ax2, nodelist=solo_en_w, node_size=2800,
                                  node_color='#4ECDC4', edgecolors='#1A7F7E', linewidths=3, label=f'Solo en {nombre_w}')

        if root_w and root_w in T_w.nodes():
            nx.draw_networkx_nodes(T_w, pos_w, ax=ax2, nodelist=[root_w], node_size=3200,
                                  node_color='gold', edgecolors='black', linewidths=3, label='Raíz')

        nx.draw_networkx_edges(T_w, pos_w, ax=ax2, edge_color='gray', width=2, alpha=0.6)
        nx.draw_networkx_labels(T_w, pos_w, ax=ax2, font_size=9, font_weight='bold')
        ax2.legend(loc='upper left', fontsize=10)
    ax2.axis('off')

    plt.tight_layout()
    filename = f"comparacion_{nombre_b}_vs_{nombre_w}_diferencias.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"   ✓ Visualización guardada: '{filename}'")
    plt.show()


print("Preparando árboles podados para comparación...\n")

pares_a_comparar = [('B2C', 'W2C'), ('B4C', 'W4C'), ('B8C', 'W8C'), ('B16C', 'W16C')]
todas_las_diferencias = {}

if not USAR_DATOS_ETAPA3:
    print("⚠️  Generando árboles localmente (fallback)...\n")

    if 'particiones_analizar' not in globals():
        print("❌ ERROR: 'particiones_analizar' no disponible. Abortando.")
    else:
        for nombre_b, nombre_w in pares_a_comparar:
            if nombre_b not in pruned_mst_trees and nombre_b in particiones_analizar:
                print(f"   Generando árbol podado para {nombre_b}...")
                df_temp = particiones_analizar[nombre_b]
                corr_matrix = df_temp.corr()
                G_temp = nx.Graph()
                for i in range(len(corr_matrix.columns)):
                    for j in range(i+1, len(corr_matrix.columns)):
                        if abs(corr_matrix.iloc[i, j]) > 0.01:
                            G_temp.add_edge(corr_matrix.columns[i], corr_matrix.columns[j],
                                          weight=1.0 - abs(corr_matrix.iloc[i, j]))

                if G_temp.number_of_edges() > 0:
                    mst_temp = nx.minimum_spanning_tree(G_temp, weight='weight')
                    pruned_mst_trees[nombre_b] = podar_arbol_por_la_mitad(mst_temp, nodo_raiz_fijo='y')

            if nombre_w not in pruned_mst_trees and nombre_w in particiones_analizar:
                print(f"   Generando árbol podado para {nombre_w}...")
                df_temp = particiones_analizar[nombre_w]
                corr_matrix = df_temp.corr()
                G_temp = nx.Graph()
                for i in range(len(corr_matrix.columns)):
                    for j in range(i+1, len(corr_matrix.columns)):
                        if abs(corr_matrix.iloc[i, j]) > 0.01:
                            G_temp.add_edge(corr_matrix.columns[i], corr_matrix.columns[j],
                                          weight=1.0 - abs(corr_matrix.iloc[i, j]))

                if G_temp.number_of_edges() > 0:
                    mst_temp = nx.minimum_spanning_tree(G_temp, weight='weight')
                    pruned_mst_trees[nombre_w] = podar_arbol_por_la_mitad(mst_temp, nodo_raiz_fijo='y')


print("\n" + "="*80)
print("COMPARANDO PARES DE ÁRBOLES PODADOS")
print("="*80 + "\n")

for nombre_b, nombre_w in pares_a_comparar:
    print(f"\n{'='*80}")
    print(f"COMPARANDO: {nombre_b} vs {nombre_w}")
    print(f"{'='*80}\n")

    try:
        if nombre_b not in pruned_mst_trees or nombre_w not in pruned_mst_trees:
            print(f"   ⚠️  Árboles no disponibles para esta comparación. Saltando...")
            continue

        T_b = pruned_mst_trees[nombre_b]
        T_w = pruned_mst_trees[nombre_w]

        print(f"   ✓ Árboles cargados desde ETAPA 3")

        print(f"\n   {nombre_b}: {T_b.number_of_nodes()} nodos, {T_b.number_of_edges()} aristas")
        print(f"   {nombre_w}: {T_w.number_of_nodes()} nodos, {T_w.number_of_edges()} aristas")

        root_b = nx.center(T_b)[0] if T_b.number_of_nodes() > 0 else None
        root_w = nx.center(T_w)[0] if T_w.number_of_nodes() > 0 else None
        print(f"\n   Raíz {nombre_b}: '{root_b}'")
        print(f"   Raíz {nombre_w}: '{root_w}'")

        print(f"\n   📊 Analizando estructuras...")
        struct_b = obtener_estructura_arbol(T_b, root_b)
        struct_w = obtener_estructura_arbol(T_w, root_w)

        df_diferencias = comparar_estructuras(struct_b, struct_w, nombre_b, nombre_w)

        if df_diferencias is None or len(df_diferencias) == 0:
            print(f"\n   ✓ NO HAY DIFERENCIAS - Ambos árboles tienen la misma estructura.")
        else:
            print(f"\n   ⚠️  DIFERENCIAS ENCONTRADAS: {len(df_diferencias)}")
            print("\n" + "="*80)
            print(df_diferencias.to_string(index=False))
            print("="*80)
            todas_las_diferencias[f"{nombre_b} vs {nombre_w}"] = df_diferencias

        visualizar_comparacion(T_b, T_w, root_b, root_w, nombre_b, nombre_w, df_diferencias)

    except Exception as e:
        print(f"   ❌ ERROR: {e}")
        traceback.print_exc()


print("\n\n" + "="*80)
print("📋 RESUMEN DE DIFERENCIAS")
print("="*80)

if todas_las_diferencias:
    for comparacion, df_diff in todas_las_diferencias.items():
        print(f"\n{comparacion}: ({len(df_diff)} diferencias)")
        print("-" * 80)
        print(df_diff.to_string(index=False))
else:
    print("\n✓ SIN DIFERENCIAS EN NINGUNA COMPARACIÓN")

if todas_las_diferencias:
    df_todas = pd.concat([df_diff.assign(Comparacion=comp)
                          for comp, df_diff in todas_las_diferencias.items()],
                         ignore_index=True)
    df_todas.to_csv('resumen_diferencias_arboles.csv', index=False)
    print(f"\n✓ Resumen guardado: 'resumen_diferencias_arboles.csv'")

print("\n" + "="*80)
print("✓ ANÁLISIS DE DIFERENCIAS FINALIZADO")
print("="*80)
