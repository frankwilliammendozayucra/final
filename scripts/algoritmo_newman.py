# ========================================
# ALGORITMO DE NEWMAN - SOLO IMPRIME RESULTADOS
# ========================================

import networkx as nx


class NewmanAlgorithm:
    """Detección de comunidades - Algoritmo de Newman"""

    def __init__(self, grafo_nx, nombre_particion=""):
        self.grafo = grafo_nx
        self.nombre = nombre_particion
        self.nodos = list(grafo_nx.nodes())
        self.n = len(self.nodos)
        self.nodo_a_idx = {nodo: idx for idx, nodo in enumerate(self.nodos)}
        self.idx_a_nodo = {idx: nodo for nodo, idx in self.nodo_a_idx.items()}

    def _inicializar_modularidad(self):
        """Calcula matriz de modularidad"""
        grados = {nodo: 0.0 for nodo in self.nodos}
        peso_total = 0.0

        for u, v, data in self.grafo.edges(data=True):
            peso = abs(data.get('weight', 1.0))
            peso_total += peso
            grados[u] += peso
            grados[v] += peso

        if peso_total == 0:
            return None, 0.0

        B = {}
        for u, v, data in self.grafo.edges(data=True):
            peso = abs(data.get('weight', 1.0))
            i, j = self.nodo_a_idx[u], self.nodo_a_idx[v]
            valor_b = peso - (grados[u] * grados[v]) / (2.0 * peso_total)
            B[(i, j)] = valor_b
            B[(j, i)] = valor_b

        return B, peso_total

    def _obtener_modularidad(self, B, i, j=None):
        if j is None:
            return B.get((i, i), 0.0)
        return B.get((i, j), 0.0) + B.get((j, i), 0.0)

    def _calcular_Q_total(self, B, activos):
        Q = 0.0
        for i in activos:
            for j in activos:
                if i != j:
                    Q += self._obtener_modularidad(B, i, j)
                else:
                    Q += B.get((i, i), 0.0)
        return Q / 2.0

    def ejecutar(self):
        """Ejecuta el algoritmo y retorna comunidades, Q_max, historial"""

        if self.n == 0:
            return [], 0.0, []

        B, peso_total = self._inicializar_modularidad()

        if B is None or not B:
            return [[nodo] for nodo in self.nodos], 0.0, []

        # Comunidades iniciales
        comunidades = {i: {self.idx_a_nodo[i]} for i in range(self.n)}
        a = [B.get((i, i), 0.0) for i in range(self.n)]
        activos = set(range(self.n))

        Q_inicial = self._calcular_Q_total(B, activos)
        Q_mejor = Q_inicial
        partition_mejor = [comunidades[i].copy() for i in activos]

        historial = []
        paso = 0

        while len(activos) > 1:
            paso += 1
            mejor_delta_Q = float('-inf')
            mejor_par = None

            activos_lista = sorted(list(activos))

            for idx_i in range(len(activos_lista)):
                for idx_j in range(idx_i + 1, len(activos_lista)):
                    i, j = activos_lista[idx_i], activos_lista[idx_j]
                    delta_Q = 2.0 * (self._obtener_modularidad(B, i, j) - a[i] * a[j])

                    if delta_Q > mejor_delta_Q:
                        mejor_delta_Q = delta_Q
                        mejor_par = (i, j)

            if mejor_par is None:
                break

            i, j = mejor_par
            Q_actual = self._calcular_Q_total(B, activos)
            Q_nuevo = Q_actual + mejor_delta_Q

            # Información de comunidades
            comm_i = sorted(list(comunidades[i]))
            comm_j = sorted(list(comunidades[j]))
            comm_i_str = "{" + ", ".join(comm_i) + "}"
            comm_j_str = "{" + ", ".join(comm_j) + "}"

            historial.append({
                'Paso': paso,
                'Comunidad_A': comm_i_str,
                'Comunidad_B': comm_j_str,
                'Delta_Q': mejor_delta_Q,
                'Q_anterior': Q_actual,
                'Q_nuevo': Q_nuevo,
                'Num_Comunidades': len(activos) - 1
            })

            # Fusionar
            comunidades[i] = comunidades[i].union(comunidades[j])
            del comunidades[j]

            # Actualizar B
            for col in range(self.n):
                B[(i, col)] = B.get((i, col), 0.0) + B.get((j, col), 0.0)
                B[(col, i)] = B.get((col, i), 0.0) + B.get((col, j), 0.0)

            nuevo_diagonal = B.get((i, i), 0.0) + B.get((j, j), 0.0) + B.get((j, i), 0.0)
            B[(i, i)] = nuevo_diagonal

            for col in range(self.n):
                B.pop((j, col), None)
                B.pop((col, j), None)

            a[i] = a[i] + a[j]
            a[j] = 0.0
            activos.discard(j)

            if Q_nuevo > Q_mejor:
                Q_mejor = Q_nuevo
                partition_mejor = [comunidades[k].copy() for k in activos]

        return partition_mejor, Q_mejor, historial


# ========================================
# FUNCIÓN PRINCIPAL - IMPRIME RESULTADOS
# ========================================

def ejecutar_newman_y_mostrar(df_particion, nombre_particion, threshold=0.01):
    """
    Ejecuta Newman en una partición e IMPRIME todos los resultados
    """

    print(f"\n{'='*130}")
    print(f"ALGORITMO DE NEWMAN - {nombre_particion.upper()}")
    print(f"{'='*130}")

    # Crear grafo desde correlaciones
    G = nx.Graph()

    # Agregar nodos (todas las columnas numéricas)
    cols_numericas = []
    for col in df_particion.columns:
        try:
            float(df_particion[col].iloc[0])
            cols_numericas.append(col)
        except:
            pass

    G.add_nodes_from(cols_numericas)

    # Calcular correlaciones y agregar aristas
    def calcular_correlacion(col1, col2):
        n = len(col1)
        media_x = sum(col1) / n
        media_y = sum(col2) / n
        suma_xy = sum((col1[i] - media_x) * (col2[i] - media_y) for i in range(n))
        suma_x2 = sum((col1[i] - media_x) ** 2 for i in range(n))
        suma_y2 = sum((col2[i] - media_y) ** 2 for i in range(n))
        if suma_x2 == 0 or suma_y2 == 0:
            return 0.0
        return suma_xy / (suma_x2 * suma_y2) ** 0.5

    aristas_agregadas = 0
    for i in range(len(cols_numericas)):
        for j in range(i + 1, len(cols_numericas)):
            col1_vals = [float(df_particion[cols_numericas[i]].iloc[k]) for k in range(len(df_particion))]
            col2_vals = [float(df_particion[cols_numericas[j]].iloc[k]) for k in range(len(df_particion))]
            corr = abs(calcular_correlacion(col1_vals, col2_vals))

            if corr >= threshold:
                G.add_edge(cols_numericas[i], cols_numericas[j], weight=corr)
                aristas_agregadas += 1

    print(f"\n📊 INFORMACIÓN DEL GRAFO:")
    print(f"   • Nodos:       {G.number_of_nodes()}")
    print(f"   • Aristas:     {G.number_of_edges()}")
    print(f"   • Threshold:   {threshold}")

    if G.number_of_nodes() == 0:
        print(f"\n   ⚠️  Grafo vacío. Sin análisis posible.")
        return None

    # Ejecutar Newman
    print(f"\n{'─'*130}")
    print(f"EJECUTANDO ALGORITMO DE NEWMAN...")
    print(f"{'─'*130}\n")

    algoritmo = NewmanAlgorithm(G, nombre_particion=nombre_particion)
    comunidades, Q_max, historial = algoritmo.ejecutar()

    # IMPRIMIR RESULTADOS INICIALES
    print(f"PASO 0: ESTADO INICIAL")
    print(f"   • Modularidad inicial (Q):  {0.0:.8f}")
    print(f"   • Comunidades:              {G.number_of_nodes()}")
    print(f"   • Configuración:            Cada nodo es su propia comunidad\n")

    # IMPRIMIR CADA PASO
    for paso_info in historial:
        print(f"PASO {paso_info['Paso']}: FUSIÓN")
        print(f"   • Fusionando: {paso_info['Comunidad_A']} + {paso_info['Comunidad_B']}")
        print(f"   • ΔQ:         {paso_info['Delta_Q']:+.8f}")
        print(f"   • Q anterior: {paso_info['Q_anterior']:.8f}")
        print(f"   • Q nuevo:    {paso_info['Q_nuevo']:.8f}")
        print(f"   • Comunidades activas: {paso_info['Num_Comunidades']}\n")

    # RESULTADOS FINALES
    print(f"{'='*130}")
    print(f"✅ RESULTADOS FINALES")
    print(f"{'='*130}")
    print(f"   • Q Máximo:                 {Q_max:.8f}")
    print(f"   • Comunidades detectadas:   {len(comunidades)}")
    print(f"   • Pasos ejecutados:         {len(historial)}")
    print(f"\n📋 COMUNIDADES DETECTADAS:")

    for idx, comunidad in enumerate(comunidades, 1):
        nodos_sorted = sorted(list(comunidad))
        print(f"   [{idx}] {nodos_sorted} ({len(nodos_sorted)} variable(s))")

    print(f"\n{'='*130}\n")

    return {
        'comunidades': comunidades,
        'Q_max': Q_max,
        'historial': historial,
        'num_comunidades': len(comunidades),
        'nodos': G.number_of_nodes(),
        'aristas': G.number_of_edges()
    }


# ========================================
# EJECUTAR EN TODAS LAS PARTICIONES
# ========================================

def ejecutar_newman_todas_particiones(particiones_lista):
    """
    Ejecuta Newman en todas las particiones y IMPRIME resultados

    particiones_lista: lista de tuplas [(nombre, dataframe), ...]
    """

    print("\n\n" + "🔷"*65)
    print("APLICANDO ALGORITMO DE NEWMAN A TODAS LAS PARTICIONES")
    print("🔷"*65)

    resultados_todos = {}

    for nombre, df_part in particiones_lista:
        resultado = ejecutar_newman_y_mostrar(df_part, nombre, threshold=0.01)
        if resultado:
            resultados_todos[nombre] = resultado

    # TABLA COMPARATIVA FINAL
    print("\n" + "="*130)
    print("TABLA COMPARATIVA: RESULTADOS POR PARTICIÓN")
    print("="*130)
    print(f"{'Partición':<15} {'Nodos':<10} {'Aristas':<10} {'Comunidades':<15} {'Q_máximo':<18} {'Pasos':<10}")
    print("─"*130)

    for nombre, resultado in resultados_todos.items():
        print(f"{nombre:<15} {resultado.get('nodos', '?'):<10} {resultado.get('aristas', '?'):<10} "
              f"{resultado['num_comunidades']:<15} {resultado['Q_max']:<18.8f} "
              f"{len(resultado['historial']):<10}")

    print("="*130)

    # ANÁLISIS: MEJOR Y PEOR Q
    print("\n" + "="*130)
    print("ANÁLISIS: MEJOR Y PEOR MODULARIDAD")
    print("="*130)

    if resultados_todos:
        mejor_nombre = max(resultados_todos, key=lambda x: resultados_todos[x]['Q_max'])
        peor_nombre = min(resultados_todos, key=lambda x: resultados_todos[x]['Q_max'])

        mejor = resultados_todos[mejor_nombre]
        peor = resultados_todos[peor_nombre]

        print(f"\n🥇 MEJOR Q: {mejor_nombre.upper()}")
        print(f"   • Q_máximo:         {mejor['Q_max']:.8f}")
        print(f"   • Comunidades:      {mejor['num_comunidades']}")
        print(f"   • Pasos:            {len(mejor['historial'])}")

        print(f"\n🥈 PEOR Q: {peor_nombre.upper()}")
        print(f"   • Q_máximo:         {peor['Q_max']:.8f}")
        print(f"   • Comunidades:      {peor['num_comunidades']}")
        print(f"   • Pasos:            {len(peor['historial'])}")

        print(f"\n📊 DIFERENCIA: {mejor['Q_max'] - peor['Q_max']:+.8f}")

    print("="*130 + "\n")

    return resultados_todos