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
