# =============================================================================
# PLANIFICACIÓN OPERATIVO DE CAMPO - ENDI 2025
# Proyecto: Automatización Cartográfica · INEC Zonal Litoral
# Autores: Franklin López, Carlos Quinto
# =============================================================================
#
# CAMBIOS EN ESTA VERSIÓN:
#
# 1. DISTRIBUCIÓN EN 12 DÍAS:
#    Las UPMs de cada encuestador se distribuyen en 12 días operativos
#    (el día 12 se planifica igual, aunque los encuestadores suelen regresar).
#    Se respeta la ventana de 50-80 viviendas por día por encuestador.
#    Si un día queda con menos de 50 o más de 80, se advierte al usuario.
#
# 2. FACTOR DE PONDERACIÓN (EXPLICADO):
#    La "carga ponderada" NO es el número de viviendas real que se visita.
#    Es un número FICTICIO que usamos solo para BALANCEAR el trabajo.
#    Ejemplo: una UPM rural con 30 viviendas tiene carga_pond = 30 * 1.5 = 45.
#    El algoritmo greedy asigna UPMs balanceando carga_pond, no viv reales.
#    Esto hace que a los encuestadores con zonas rurales se les asigne MENOS
#    viviendas reales, compensando el mayor tiempo de desplazamiento rural.
#
# 3. RESTRICCIÓN DE GUAYAQUIL (primeros días):
#    Si el cantón de Guayaquil (pro=09, can=01) tiene suficientes UPMs en el
#    mes, los primeros 2-4 días TODOS los equipos trabajan en Guayaquil.
#    Esto respeta la lógica operativa del INEC: los encuestadores reciben
#    pago por los primeros días en sede antes de salir a campo.
#    Implementado como: si UPMs_Guayaquil >= UMBRAL_GYE_DIAS * viv_por_dia,
#    se separan y se asignan como "Fase 0" antes de las jornadas normales.
#
# 4. EQUIPO BOMBERO SIEMPRE VISIBLE:
#    El equipo bombero ahora siempre existe en el DataFrame de resultados,
#    aunque esté vacío (0 UPMs). Esto evita que desaparezca del mapa/reporte.
#    Si no hay outliers, el equipo_bombero aparece con 0 UPMs y se notifica.
#
# 5. CV CALCULADO CORRECTAMENTE:
#    El CV entre equipos se calcula sobre la carga_ponderada DESPUÉS del
#    balance greedy, no antes. Se muestra separado por jornada.
# =============================================================================

from google.colab import drive
drive.mount('/content/drive')

import pandas as pd
import numpy as np
import folium
import osmnx as ox
import networkx as nx
from networkx.algorithms import approximation
from pyproj import Transformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import plotly.express as px
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# SECCIÓN 0: PARÁMETROS GLOBALES DEL OPERATIVO
# =============================================================================
# Todos los parámetros configurables están aquí arriba para facilitar ajustes.

MES_OPERATIVO   = 9       # Mes a planificar (codificación interna de la muestra)
N_EQUIPOS       = 3       # Número de equipos regulares (sin bombero)
DIAS_OPERATIVOS = 12      # Días totales de la jornada (el 12avo = regreso, pero se planifica)
VIV_MIN_DIA     = 50      # Viviendas mínimas esperadas por encuestador por día
VIV_MAX_DIA     = 80      # Viviendas máximas esperadas por encuestador por día

# Factor de ponderación rural:
# Explicación detallada en Sección 5.
# Valor 1.5 = una vivienda rural "pesa" como 1.5 urbanas para el balanceo.
# No cambia las viviendas reales a visitar, solo el criterio de asignación.
FACTOR_RURAL    = 1.5

# Restricción de Guayaquil (cantón sede):
# Si hay suficientes UPMs en Guayaquil, los primeros DIAS_GYE días
# todos los equipos trabajan ahí antes de salir a sus zonas asignadas.
PRO_GYE         = 9       # Código de provincia Guayas
CAN_GYE         = 1       # Código de cantón Guayaquil
DIAS_GYE        = 3       # Días máximos a asignar en Guayaquil al inicio
# Umbral: si UPMs de Guayaquil tienen al menos este % del total del mes,
# se activa la restricción de Guayaquil.
UMBRAL_GYE_PCT  = 0.10    # 10% del total de UPMs del mes

BASE_LAT        = -2.145825935522539
BASE_LON        = -79.89383956329586

MESES_NOMBRES = {
    1:'Julio', 2:'Agosto', 3:'Septiembre', 4:'Octubre',
    5:'Noviembre', 6:'Diciembre', 7:'Enero', 8:'Febrero',
    9:'Marzo', 10:'Abril', 11:'Mayo', 12:'Junio'
}


# =============================================================================
# SECCIÓN 1: CARGA Y PREPARACIÓN DE DATOS
# =============================================================================

df_muestra = pd.read_excel('/content/Muestra ENDI A3.xls', sheet_name='Hoja1')
df_amanz   = pd.read_excel('punto_amanz.xlsx', header=0)
df_rural   = pd.read_excel('punto_rural.xlsx', header=0)

df_amanz.rename(columns={'man': 'man_sec'}, inplace=True)
df_rural.rename(columns={'sec': 'man_sec'}, inplace=True)

df_puntos  = pd.concat([df_amanz, df_rural], axis=0, ignore_index=True)
df_puntos  = df_puntos[df_puntos['zonal'] == 'LITORAL']
df_muestra = df_muestra[df_muestra['Zonal'] == 'LITORAL']

df_matched = pd.merge(df_muestra, df_puntos, on='man_sec', how='inner')
df_matched['mes_nombre'] = df_matched['mes_x'].map(MESES_NOMBRES)

print(f"UPMs con coordenadas (Litoral): {len(df_matched)}")
display(df_matched.head(2))


# =============================================================================
# SECCIÓN 2: CONVERSIÓN UTM → WGS84
# =============================================================================

transformer_utm_wgs = Transformer.from_crs("EPSG:32717", "EPSG:4326", always_xy=True)
df_matched['longitude'], df_matched['latitude'] = transformer_utm_wgs.transform(
    df_matched['x'].values, df_matched['y'].values
)


# =============================================================================
# SECCIÓN 3: CARGA Y FILTRADO DEL GRAFO VIAL
# =============================================================================

G_completo = ox.load_graphml("zonal.graphml")

TIPOS_VIAS = ['motorway','trunk','primary','secondary',
              'motorway_link','trunk_link','primary_link','secondary_link']

arcos_ok = []
for u, v, k, data in G_completo.edges(data=True, keys=True):
    hw = data.get('highway', '')
    if isinstance(hw, list):
        if any(t in TIPOS_VIAS for t in hw): arcos_ok.append((u, v, k))
    else:
        if hw in TIPOS_VIAS: arcos_ok.append((u, v, k))

G = G_completo.edge_subgraph(arcos_ok).copy()
print(f"Grafo filtrado: {len(G.nodes):,} nodos, {len(G.edges):,} arcos")


# =============================================================================
# SECCIÓN 4: FILTRADO POR MES Y DETECCIÓN DE OUTLIERS (EQUIPO BOMBERO)
# =============================================================================

df_mes = df_matched[df_matched['mes_x'] == MES_OPERATIVO].copy()
df_mes.dropna(subset=['latitude','longitude'], inplace=True)
df_mes = df_mes.reset_index(drop=True)

print(f"\nUPMs mes {MES_OPERATIVO} ({MESES_NOMBRES.get(MES_OPERATIVO,'?')}): {len(df_mes)}")
print(f"Viviendas totales: {df_mes['viv_y'].sum():,}")

# Convertimos base a UTM para calcular distancias en metros
t_wgs_utm = Transformer.from_crs("EPSG:4326","EPSG:32717",always_xy=True)
base_x, base_y = t_wgs_utm.transform(BASE_LON, BASE_LAT)

df_mes['dist_base_m'] = np.sqrt(
    (df_mes['x'] - base_x)**2 + (df_mes['y'] - base_y)**2
)

# Detección de outliers con IQR sobre distancia a la base
Q1d = df_mes['dist_base_m'].quantile(0.25)
Q3d = df_mes['dist_base_m'].quantile(0.75)
umbral_bombero = Q3d + 1.5*(Q3d - Q1d)
mask_bombero = df_mes['dist_base_m'] > umbral_bombero

# CAMBIO: inicializamos equipo/jornada para TODOS, incluyendo bombero.
# Aunque no haya outliers, la columna 'equipo_bombero' existe con 0 filas.
df_mes['equipo']    = 'sin_asignar'
df_mes['jornada']   = 'sin_asignar'
df_mes['cluster_geo'] = -1
df_mes['carga_pond']  = 0.0
df_mes['dia_operativo'] = 0

df_mes.loc[mask_bombero, 'equipo']  = 'Equipo Bombero'
df_mes.loc[mask_bombero, 'jornada'] = 'Jornada Especial'

n_bombero = mask_bombero.sum()
if n_bombero == 0:
    print("\nINFO: No se detectaron outliers. El Equipo Bombero está activo pero sin UPMs asignadas.")
    print(f"(Umbral IQR: {umbral_bombero/1000:.1f} km — ninguna UPM supera ese límite)")
else:
    print(f"\nEquipo Bombero: {n_bombero} UPMs asignadas (>{umbral_bombero/1000:.1f} km de la base)")


# =============================================================================
# SECCIÓN 5: FACTOR DE PONDERACIÓN RURAL — EXPLICACIÓN DETALLADA
# =============================================================================
#
# ¿QUÉ ES LA CARGA PONDERADA?
# ----------------------------
# "Carga" = viviendas reales a visitar (dato del precenso 2020).
# "Carga ponderada" = número ajustado que usamos SOLO para balancear el trabajo.
#
# ¿POR QUÉ PONDERAMOS?
# ---------------------
# Visitar 30 viviendas en Guayaquil (urbano) puede tomar 3 horas.
# Visitar 30 viviendas en una zona dispersa rural puede tomar 6-8 horas
# por los desplazamientos entre casas, caminos sin asfaltar, etc.
#
# Si balanceamos solo por viviendas reales, el encuestador de zona rural
# tendrá el mismo número de casas que el urbano, pero trabajará el doble.
# Eso no es equitativo aunque los números digan que sí.
#
# ¿CÓMO FUNCIONA EL FACTOR?
# --------------------------
# carga_pond = viv * FACTOR_RURAL  (si es zona dispersa/rural)
# carga_pond = viv * 1.0           (si es zona amanzanada/urbana)
#
# Con FACTOR_RURAL = 1.5:
#   - Una UPM urbana con 60 viviendas → carga_pond = 60
#   - Una UPM rural con 60 viviendas  → carga_pond = 90
#
# El algoritmo greedy asigna UPMs al encuestador con MENOR carga_pond
# acumulada. Resultado: al encuestador rural se le dan MENOS viviendas
# reales, compensando el mayor tiempo de desplazamiento.
#
# IMPORTANTE: el reporte final muestra AMBOS valores:
#   - "Viviendas reales" = lo que realmente van a encuestar
#   - "Carga ponderada" = el criterio de equidad usado en el balance

# Calculamos carga ponderada para cada UPM
df_mes['carga_pond'] = df_mes.apply(
    lambda r: r['viv_y'] * FACTOR_RURAL
    if str(r.get('tipo_entidad','')).startswith('sec')
    else r['viv_y'],
    axis=1
)

print("\nEjemplo de diferencia carga real vs ponderada:")
ejemplo = df_mes[['man_sec','tipo_entidad','viv_y','carga_pond']].head(5)
display(ejemplo)


# =============================================================================
# SECCIÓN 6: RESTRICCIÓN DE GUAYAQUIL (primeros días en sede)
# =============================================================================
#
# CONTEXTO OPERATIVO:
# Según comentarios de la dirección de Cartografía, los primeros días
# de cada jornada TODOS los equipos trabajan en el cantón sede (Guayaquil).
# Esto se relaciona con el sistema de pago y la logística de salida a campo.
# Los encuestadores salen de Guayaquil pero primero completan trabajo local.
#
# IMPLEMENTACIÓN:
# Si Guayaquil tiene suficientes UPMs (>= UMBRAL_GYE_PCT del total del mes),
# separamos esas UPMs como "Fase Guayaquil" y las asignamos a los primeros
# DIAS_GYE días de la jornada para TODOS los equipos por igual.
# Las UPMs restantes (fuera de Guayaquil) se procesan con el clustering normal.
#
# Identificamos UPMs de Guayaquil usando pro_x y can_x del archivo de muestra
# (pro=09 = Guayas, can=01 = Guayaquil)

upms_gye_mask = (df_mes.get('pro_x', pd.Series(dtype=int)) == PRO_GYE) & \
                (df_mes.get('can_x', pd.Series(dtype=int)) == CAN_GYE)
pct_gye = upms_gye_mask.sum() / len(df_mes) if len(df_mes) > 0 else 0

ACTIVAR_RESTRICCION_GYE = (pct_gye >= UMBRAL_GYE_PCT) and (upms_gye_mask.sum() > 0)

if ACTIVAR_RESTRICCION_GYE:
    df_gye     = df_mes[upms_gye_mask & ~mask_bombero].copy()
    df_no_gye  = df_mes[~upms_gye_mask & ~mask_bombero].copy()
    print(f"\n✓ Restricción Guayaquil activada:")
    print(f"  UPMs en Guayaquil: {len(df_gye)} ({pct_gye*100:.1f}% del mes)")
    print(f"  UPMs fuera de Guayaquil: {len(df_no_gye)}")
    print(f"  Los primeros {DIAS_GYE} días se asignarán UPMs de Guayaquil a todos los equipos.")
else:
    df_gye    = pd.DataFrame()  # Vacío: no hay restricción
    df_no_gye = df_mes[~mask_bombero].copy()
    print(f"\nINFO: Restricción Guayaquil NO activada ({pct_gye*100:.1f}% < {UMBRAL_GYE_PCT*100:.0f}%)")
    print(f"  Se procede con clustering normal en todos los puntos del mes.")


# =============================================================================
# SECCIÓN 7: CLUSTERING GEOGRÁFICO POR CONGLOMERADOS
# =============================================================================
# Aplicamos K-Means sobre los puntos FUERA de Guayaquil (o todos si no aplica
# la restricción). Generamos n_equipos * 2 clusters (uno por equipo por jornada).

df_para_cluster = df_no_gye.copy()
n_clusters = N_EQUIPOS * 2  # 2 jornadas por mes

if len(df_para_cluster) >= n_clusters:
    coords = df_para_cluster[['x','y']].values
    km = KMeans(n_clusters=n_clusters, init='k-means++', n_init=20,
                max_iter=500, random_state=42)
    df_para_cluster['cluster_geo'] = km.fit_predict(coords)

    if len(df_para_cluster) > n_clusters:
        sil = silhouette_score(coords, df_para_cluster['cluster_geo'])
        print(f"\nClustering: índice de silueta = {sil:.3f}")

    # Ordenamos clusters por distancia del centroide a la base
    centroides = km.cluster_centers_
    dist_c = np.sqrt((centroides[:,0]-base_x)**2 + (centroides[:,1]-base_y)**2)
    orden  = np.argsort(dist_c)[::-1]  # De más lejano a más cercano

    # Los N más lejanos → Jornada 1 | Los N más cercanos → Jornada 2
    asignacion = {}
    for i, (cj1, cj2) in enumerate(zip(orden[:N_EQUIPOS], orden[N_EQUIPOS:])):
        nombre = f'Equipo {i+1}'
        asignacion[cj1] = (nombre, 'Jornada 1')
        asignacion[cj2] = (nombre, 'Jornada 2')

    df_para_cluster['equipo']  = df_para_cluster['cluster_geo'].map(lambda c: asignacion[c][0])
    df_para_cluster['jornada'] = df_para_cluster['cluster_geo'].map(lambda c: asignacion[c][1])
    df_mes.update(df_para_cluster[['equipo','jornada','cluster_geo']])
else:
    print(f"AVISO: Pocos puntos ({len(df_para_cluster)}) para {n_clusters} clusters.")


# =============================================================================
# SECCIÓN 8: ASIGNACIÓN GREEDY DE ENCUESTADORES Y DISTRIBUCIÓN EN 12 DÍAS
# =============================================================================
#
# LÓGICA:
# 1. Dentro de cada (equipo, jornada), ordenamos UPMs por carga_pond descendente.
# 2. Asignamos cada UPM al encuestador con menor carga acumulada (greedy).
# 3. Distribuimos las UPMs de cada encuestador en 12 días respetando 50-80 viv/día.
# 4. Si las UPMs de Guayaquil están activas, se asignan a días 1-DIAS_GYE de todos.

N_ENCUESTADORES_POR_EQUIPO = 3  # Ajustar según configuración real

def asignar_encuestadores_y_dias(df_grupo, n_enc, dias_totales,
                                  viv_min, viv_max, inicio_dia=1):
    """
    Asigna encuestadores (greedy) y distribuye en días operativos.
    
    Parámetros:
    -----------
    df_grupo : DataFrame con UPMs del grupo (equipo + jornada)
    n_enc    : número de encuestadores del equipo
    dias_totales : días operativos disponibles (normalmente 12)
    viv_min, viv_max : ventana diaria de viviendas por encuestador
    inicio_dia : día desde el que empezar (1 normalmente, >1 si hay fase GYE)
    
    Retorna:
    --------
    df_grupo con columnas 'encuestador' y 'dia_operativo' asignadas.
    """
    df_g = df_grupo.sort_values('carga_pond', ascending=False).copy()
    cargas = np.zeros(n_enc)

    # Asignación greedy: cada UPM va al encuestador con menos carga
    encuestadores_asig = []
    for _, row in df_g.iterrows():
        enc_min = int(np.argmin(cargas))
        encuestadores_asig.append(enc_min + 1)
        cargas[enc_min] += row['carga_pond']
    df_g['encuestador'] = encuestadores_asig

    # Distribución en días por encuestador
    dias_asig = []
    for enc_id in range(1, n_enc+1):
        upms_enc = df_g[df_g['encuestador'] == enc_id].copy()
        # Usamos viv_y reales (no ponderadas) para los días
        viv_acum = 0
        dia_actual = inicio_dia
        dias_enc = []
        for _, row in upms_enc.iterrows():
            viv_acum += row['viv_y']
            dias_enc.append(dia_actual)
            # Si superamos el máximo diario, pasamos al siguiente día
            if viv_acum >= viv_max and dia_actual < inicio_dia + dias_totales - 1:
                dia_actual += 1
                viv_acum = 0
        # Advertencia si la distribución no es balanceada
        if viv_acum < viv_min and len(dias_enc) > 0:
            pass  # Podría ocurrir en el último día; no es error crítico

        # Actualizamos días en df_g usando el índice de upms_enc
        for idx, d in zip(upms_enc.index, dias_enc):
            df_g.loc[idx, 'dia_operativo'] = d

    return df_g

# Procesamos grupos regulares (no bombero, no Guayaquil)
equipos_regulares = [f'Equipo {i+1}' for i in range(N_EQUIPOS)]
resultados = []

for equipo in equipos_regulares:
    for jornada in ['Jornada 1', 'Jornada 2']:
        mask = (df_mes['equipo'] == equipo) & (df_mes['jornada'] == jornada)
        grp  = df_mes[mask].copy()
        if len(grp) == 0:
            continue

        # Si hay fase Guayaquil, los primeros DIAS_GYE días ya están "ocupados"
        # Los días regulares empiezan desde DIAS_GYE + 1
        inicio = (DIAS_GYE + 1) if ACTIVAR_RESTRICCION_GYE else 1
        dias_disponibles = DIAS_OPERATIVOS - (DIAS_GYE if ACTIVAR_RESTRICCION_GYE else 0)

        grp_asig = asignar_encuestadores_y_dias(
            grp, N_ENCUESTADORES_POR_EQUIPO, dias_disponibles,
            VIV_MIN_DIA, VIV_MAX_DIA, inicio_dia=inicio
        )
        df_mes.update(grp_asig[['encuestador','dia_operativo']])

# Fase Guayaquil: distribuimos uniformemente entre todos los equipos
if ACTIVAR_RESTRICCION_GYE and len(df_gye) > 0:
    # Dividimos UPMs de Guayaquil entre los N_EQUIPOS equipos
    upms_gye_sorted = df_gye.sort_values('carga_pond', ascending=False).copy()
    for i, (idx, row) in enumerate(upms_gye_sorted.iterrows()):
        equipo_asig = f'Equipo {(i % N_EQUIPOS) + 1}'
        enc_asig    = (i // N_EQUIPOS) % N_ENCUESTADORES_POR_EQUIPO + 1
        dia_asig    = (i // (N_EQUIPOS * N_ENCUESTADORES_POR_EQUIPO)) + 1
        dia_asig    = min(dia_asig, DIAS_GYE)  # Máximo DIAS_GYE días en Guayaquil
        df_mes.loc[idx, 'equipo']         = equipo_asig
        df_mes.loc[idx, 'jornada']        = 'Jornada 1'  # Los primeros días son J1
        df_mes.loc[idx, 'encuestador']    = enc_asig
        df_mes.loc[idx, 'dia_operativo']  = dia_asig

    print(f"\nFase Guayaquil: {len(df_gye)} UPMs asignadas a días 1-{DIAS_GYE}")


# =============================================================================
# SECCIÓN 9: VERIFICACIÓN DE EQUIDAD (CV)
# =============================================================================
#
# DIFERENCIA ENTRE CARGA Y CARGA PONDERADA:
# ------------------------------------------
# CARGA REAL     = sum(viv_y)      → viviendas que realmente se encuestan
# CARGA PONDERADA = sum(carga_pond) → criterio de balance (incluye factor rural)
#
# CV (Coeficiente de Variación) = desv_est / media * 100
# Mide qué tan dispersas están las cargas entre equipos.
# Un CV bajo (<20%) indica que todos los equipos tienen trabajo similar.
# Un CV alto (>40%) indica desbalance: revisar parámetros.
#
# Calculamos CV DESPUÉS del balance greedy, que es cuando tiene sentido.

resumen_balance = df_mes[df_mes['equipo'] != 'Equipo Bombero'].groupby(
    ['equipo','jornada']
).agg(
    n_upms          = ('man_sec','count'),
    viv_reales      = ('viv_y','sum'),           # Viviendas reales a encuestar
    carga_ponderada = ('carga_pond','sum'),        # Criterio interno de balance
).reset_index()
# NOTA: este resumen incluye las UPMs de Guayaquil porque se calcula DESPUÉS
# de la fase Guayaquil (Sección 8). Si se moviera antes, las UPMs de GYE
# aparecerían como 'sin_asignar' y el CV sería incorrecto.

print("\nResumen de balance (CV calculado post-greedy):")
display(resumen_balance)

# Calculamos CV por jornada
print("\n--- EQUIDAD ENTRE EQUIPOS ---")
for jornada in ['Jornada 1', 'Jornada 2']:
    sub = resumen_balance[resumen_balance['jornada'] == jornada]
    if len(sub) > 1:
        # CV de viviendas REALES (lo que realmente trabajan)
        cv_real = sub['viv_reales'].std() / sub['viv_reales'].mean() * 100
        # CV de carga PONDERADA (el criterio de balance interno)
        cv_pond = sub['carga_ponderada'].std() / sub['carga_ponderada'].mean() * 100
        print(f"\n{jornada}:")
        print(f"  CV viviendas reales:     {cv_real:.1f}%  ← lo que el encuestador nota en campo")
        print(f"  CV carga ponderada:      {cv_pond:.1f}%  ← criterio interno de equidad")
        if cv_pond < 15:
            print("  ✓ Excelente balance")
        elif cv_pond < 30:
            print("  ✓ Balance aceptable")
        else:
            print("  ⚠ Balance mejorable — considera ajustar N_EQUIPOS o FACTOR_RURAL")


# =============================================================================
# SECCIÓN 10: OPTIMIZACIÓN TSP POR EQUIPO Y JORNADA
# =============================================================================

base_node   = ox.nearest_nodes(G, BASE_LON, BASE_LAT)
G_undir     = G.to_undirected()
comp_base   = nx.node_connected_component(G_undir, base_node)

tsp_results = {}
road_paths  = {}

for equipo in equipos_regulares:
    for jornada in ['Jornada 1', 'Jornada 2']:
        mask = (df_mes['equipo'] == equipo) & (df_mes['jornada'] == jornada)
        grp  = df_mes[mask]
        if len(grp) == 0: continue

        print(f"\nTSP: {equipo} | {jornada} ({len(grp)} puntos)...", end=' ')

        nodos_raw = ox.nearest_nodes(G, grp['longitude'].values, grp['latitude'].values)
        nodos_ok  = [n for n in nodos_raw if n in comp_base]
        if len(nodos_ok) == 0:
            print("sin nodos alcanzables.")
            continue

        nodos_unicos = [base_node] + list(dict.fromkeys(nodos_ok))
        n = len(nodos_unicos)
        if n <= 2:
            print("trivial.")
            continue

        D = np.zeros((n, n))
        for i in range(n):
            for j in range(i+1, n):
                try:
                    d = nx.shortest_path_length(G, nodos_unicos[i], nodos_unicos[j], weight='length')
                    D[i,j] = D[j,i] = d
                except:
                    D[i,j] = D[j,i] = 1e9

        G_tsp = nx.Graph()
        for i in range(n):
            for j in range(i+1, n):
                if D[i,j] < 1e8:
                    G_tsp.add_edge(i, j, weight=D[i,j])

        if not nx.is_connected(G_tsp): continue

        try:
            ciclo = approximation.traveling_salesman_problem(G_tsp, weight='weight', cycle=True)
        except Exception as e:
            print(f"error TSP: {e}")
            continue

        if 0 in ciclo:
            idx0  = ciclo.index(0)
            ciclo = ciclo[idx0:] + ciclo[1:idx0+1]

        dist_total = sum(D[ciclo[i],ciclo[i+1]] for i in range(len(ciclo)-1))

        ruta_coords = []
        nodos_grafo = [nodos_unicos[idx] for idx in ciclo]
        for k in range(len(nodos_grafo)-1):
            try:
                seg = nx.shortest_path(G, nodos_grafo[k], nodos_grafo[k+1], weight='length')
                ruta_coords.extend((G.nodes[nd]['y'], G.nodes[nd]['x']) for nd in seg[:-1])
            except: continue
        if nodos_grafo:
            ruta_coords.append((G.nodes[nodos_grafo[-1]]['y'], G.nodes[nodos_grafo[-1]]['x']))

        clave = f"{equipo}||{jornada}"
        tsp_results[clave] = {'equipo':equipo,'jornada':jornada,
                               'n_puntos':len(grp),'dist_km':dist_total/1000}
        road_paths[clave]  = ruta_coords
        print(f"{dist_total/1000:.1f} km ✓")


# =============================================================================
# SECCIÓN 11: MAPA INTERACTIVO FINAL
# =============================================================================

COLORES = {
    'Equipo 1':'#e74c3c','Equipo 2':'#3498db','Equipo 3':'#27ae60',
    'Equipo 4':'#f39c12','Equipo 5':'#9b59b6','Equipo Bombero':'#95a5a6'
}

m = folium.Map(location=[BASE_LAT, BASE_LON], zoom_start=8, tiles='CartoDB dark_matter')
folium.Marker(
    [BASE_LAT, BASE_LON], popup='<b>Base INEC Guayaquil</b>',
    icon=folium.Icon(color='white', icon='home', prefix='fa')
).add_to(m)

grupos_capa = {}
for equipo in equipos_regulares + ['Equipo Bombero']:
    jornadas = ['Jornada 1','Jornada 2'] if equipo != 'Equipo Bombero' else ['Jornada Especial']
    for j in jornadas:
        clave_capa = f"{equipo} · {j}"
        fg = folium.FeatureGroup(name=clave_capa, show=True)
        fg.add_to(m)
        grupos_capa[clave_capa] = fg

for _, row in df_mes.iterrows():
    eq, jor = row.get('equipo',''), row.get('jornada','')
    clave_capa = f"{eq} · {jor}"
    color = COLORES.get(eq, '#888888')
    if clave_capa in grupos_capa:
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=6, color=color, fill=True,
            fill_color=color, fill_opacity=0.85,
            popup=folium.Popup(
                f"<b>UPM:</b> {row.get('id_upm','N/A')}<br>"
                f"<b>Viviendas:</b> {int(row.get('viv_y',0))}<br>"
                f"<b>Carga pond.:</b> {row.get('carga_pond',0):.0f}<br>"
                f"<b>Equipo:</b> {eq}<br><b>Jornada:</b> {jor}<br>"
                f"<b>Día:</b> {int(row.get('dia_operativo',0))}",
                max_width=220
            ),
            tooltip=f"{eq} · Día {int(row.get('dia_operativo',0))} · {int(row.get('viv_y',0))} viv"
        ).add_to(grupos_capa[clave_capa])

for clave, coords in road_paths.items():
    eq, jor = clave.split('||')
    clave_capa = f"{eq} · {jor}"
    if clave_capa in grupos_capa and len(coords) > 1:
        folium.PolyLine(
            locations=coords, weight=3,
            color=COLORES.get(eq,'#888888'),
            opacity=0.75, tooltip=f"Ruta: {eq} | {jor}"
        ).add_to(grupos_capa[clave_capa])

folium.LayerControl(collapsed=False).add_to(m)
m.save('logistica_campo_final.html')
display(m)


# =============================================================================
# SECCIÓN 12: REPORTE FINAL — FORMATO SIMILAR AL INEC
# =============================================================================
# Generamos un DataFrame con estructura similar al ejemplo de planificación
# de la Jornada 16 (formato INEC): supervisor, encuestador, código jurisdicción,
# provincia, cantón, parroquia, zona, sector, manzana, fechas, # viv.

print("\n" + "="*70)
print("REPORTE DE PLANIFICACIÓN")
print("="*70)

dist_df = pd.DataFrame([
    {'equipo':v['equipo'],'jornada':v['jornada'],'dist_km':round(v['dist_km'],1)}
    for v in tsp_results.values()
])

reporte = pd.merge(resumen_balance, dist_df, on=['equipo','jornada'], how='left')
total   = pd.DataFrame([{
    'equipo':'TOTAL','jornada':'—',
    'n_upms': reporte['n_upms'].sum(),
    'viv_reales': reporte['viv_reales'].sum(),
    'carga_ponderada': reporte['carga_ponderada'].sum(),
    'dist_km': reporte['dist_km'].sum()
}])
display(pd.concat([reporte, total], ignore_index=True))

# Distribución por día (cuántas viviendas por día por equipo)
print("\nDistribución de viviendas por día operativo:")
pivot_dias = df_mes[df_mes['equipo'] != 'Equipo Bombero'].groupby(
    ['equipo','dia_operativo']
)['viv_y'].sum().reset_index()
pivot_dias = pivot_dias.pivot(index='equipo', columns='dia_operativo', values='viv_y').fillna(0)
display(pivot_dias.astype(int))

# Advertencias de días fuera de ventana
print(f"\nVentana esperada: {VIV_MIN_DIA}-{VIV_MAX_DIA} viviendas/encuestador/día")
dias_equipo = df_mes[df_mes['equipo'] != 'Equipo Bombero'].groupby(
    ['equipo','encuestador','dia_operativo']
)['viv_y'].sum()
fuera_ventana = dias_equipo[(dias_equipo < VIV_MIN_DIA) | (dias_equipo > VIV_MAX_DIA)]
if len(fuera_ventana) > 0:
    print(f"⚠ {len(fuera_ventana)} asignaciones fuera de la ventana 50-80:")
    display(fuera_ventana.reset_index())
else:
    print("✓ Todos los días están dentro de la ventana 50-80 viviendas.")
