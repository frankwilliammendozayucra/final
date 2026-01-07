import os
import itertools
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

print("\n=== SCRIPT 8: UNIÓN DE RECORRIDOS POR PARES ===\n")


def cargar_resultados():
    try:
        import arboles_enraizados as ae
        resultados = getattr(ae, 'resultados_arboles', None)
        if resultados:
            print("✓ 'resultados_arboles' cargado desde arboles_enraizados.py")
            return resultados
        else:
            print("✗ 'resultados_arboles' vacío en arboles_enraizados.py")
            return None
    except Exception as e:
        print(f"✗ No se pudo importar 'arboles_enraizados': {e}")
        return None


def edge_set(G):
    return set(tuple(sorted((u, v))) for u, v in G.edges())


def asegurar_dir(path):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def visualizar_union(graph, filename, title=None):
    plt.figure(figsize=(14, 10))
    pos = nx.spring_layout(graph, seed=42)
    nx.draw(graph, pos, with_labels=True, node_size=600, node_color='#90EE90', edge_color='#333333', width=2)
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()


def main():
    resultados = cargar_resultados()
    if not resultados:
        print("No hay resultados de árboles para procesar. Ejecuta 'arboles_enraizados.py' primero.")
        return

    rows = []
    pairs = list(itertools.combinations(sorted(resultados.keys()), 2))
    print(f"Procesando {len(pairs)} pares...\n")

    for i, (a, b) in enumerate(pairs, start=1):
        print(f"[{i}/{len(pairs)}] Unión: {a} + {b}")
        da = resultados[a]
        db = resultados[b]

        bfs_union = list(set(da.get('bfs', [])) | set(db.get('bfs', [])))
        dfs_union = list(set(da.get('dfs', [])) | set(db.get('dfs', [])))

        G_union = nx.compose(da['arbol_enraizado'], db['arbol_enraizado'])

        nodes_union = G_union.number_of_nodes()
        edges_union = G_union.number_of_edges()

        common_nodes = len(set(da['arbol_enraizado'].nodes()) & set(db['arbol_enraizado'].nodes()))
        common_edges = len(edge_set(da['arbol_enraizado']) & edge_set(db['arbol_enraizado']))

        filename = f"paso_08_union_{a}_{b}.png"
        asegurar_dir(os.path.join('..', 'results', filename))
        visualizar_union(G_union, os.path.join('..', 'results', filename), title=f"Unión {a} + {b}")

        rows.append({
            'particion_a': a,
            'particion_b': b,
            'nodos_union': nodes_union,
            'aristas_union': edges_union,
            'nodos_comunes': common_nodes,
            'aristas_comunes': common_edges,
            'union_bfs_count': len(bfs_union),
            'union_dfs_count': len(dfs_union),
            'imagen': filename
        })

    # Unión global (todas las particiones)
    print("\nGenerando unión global de todas las particiones...")
    all_graphs = [data['arbol_enraizado'] for data in resultados.values()]
    if all_graphs:
        G_all = nx.compose_all(all_graphs)
        filename_all = 'paso_08_union_global.png'
        visualizar_union(G_all, os.path.join('..', 'results', filename_all), title='Unión global de particiones')
        rows.append({
            'particion_a': 'GLOBAL',
            'particion_b': 'GLOBAL',
            'nodos_union': G_all.number_of_nodes(),
            'aristas_union': G_all.number_of_edges(),
            'nodos_comunes': '',
            'aristas_comunes': '',
            'union_bfs_count': '',
            'union_dfs_count': '',
            'imagen': filename_all
        })

    df = pd.DataFrame(rows)
    asegurar_dir('../results/union_recorridos_pares.csv')
    df.to_csv('../results/union_recorridos_pares.csv', index=False)
    print("\n✓ CSV guardado en '../results/union_recorridos_pares.csv'")
    print("✓ Imágenes guardadas en '../results/'")


if __name__ == '__main__':
    main()
