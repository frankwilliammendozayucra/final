import pandas as pd
import os

print("\n" + "="*80)
print("PASO FINAL: GUARDANDO TODOS LOS RESULTADOS")
print("="*80)

# Cargar datos (asumiendo que ya existen)
df = pd.read_csv('../data/dataset_ejemplo.csv')

# Cargar particiones desde archivos CSV
particiones = {}
particion_names = ['B2C', 'W2C', 'B4C', 'W4C', 'B8C', 'W8C', 'B16C', 'W16C']
for name in particion_names:
    path = f'../data/{name}.csv'
    if os.path.exists(path):
        particiones[name] = pd.read_csv(path)

# Cargar estadísticas si existe
if os.path.exists('../results/estadisticas_particiones.csv'):
    df_stats = pd.read_csv('../results/estadisticas_particiones.csv')
else:
    print("⚠️  No se encontró 'estadisticas_particiones.csv'. Creando estadísticas básicas...")
    # Crear estadísticas básicas
    stats_data = []
    for name, partition_df in [('df_original', df)] + list(particiones.items()):
        stats_data.append({
            'Partición': name,
            'Filas': len(partition_df),
            'Columnas': len(partition_df.columns)
        })
    df_stats = pd.DataFrame(stats_data)

# 1. Guardar el DataFrame original
df.to_csv('../results/df_original.csv', index=False)
print("DataFrame original guardado en '../results/df_original.csv'")

# 2. Guardar todas las particiones
for nombre, partition_df in particiones.items():
    filename = f'../results/{nombre}_partition.csv'
    partition_df.to_csv(filename, index=False)
    print(f"Partición guardada en '{filename}'")

# 3. Guardar las estadísticas
df_stats.to_csv('../results/estadisticas_particiones.csv', index=False)
print("Tabla de estadísticas guardada en '../results/estadisticas_particiones.csv'")

print("\n✓ Todos los archivos guardados exitosamente.")