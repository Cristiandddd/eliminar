#!/usr/bin/env python3
"""
Análisis topológico de redes - VERSIÓN OPTIMIZADA SIN NETWORKX
Implementación directa con NumPy y SciPy para matrices grandes (N > 2000)
"""
import json
import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components, shortest_path
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
from tqdm import tqdm
import gc
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import time

# ============================================================
# CONFIGURACIÓN
# ============================================================

UMBRALES = [0.20, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29, 0.30]
N_WORKERS = min(16, mp.cpu_count())

# ============================================================
# FUNCIONES DE MÉTRICAS SIN NETWORKX
# ============================================================

def calcular_grados(adj_matrix):
    """
    Calcula el grado de cada nodo.
    Para matriz dispersa o densa.
    """
    if sparse.issparse(adj_matrix):
        return np.asarray(adj_matrix.sum(axis=1)).flatten()
    else:
        return adj_matrix.sum(axis=1)


def calcular_clustering_local(adj_matrix, grados):
    """
    Calcula el coeficiente de clustering local para cada nodo.
    C_i = 2 * T_i / (k_i * (k_i - 1))
    donde T_i es el número de triángulos en el nodo i.
    
    Optimizado para matrices dispersas.
    """
    N = adj_matrix.shape[0]
    
    if sparse.issparse(adj_matrix):
        # Convertir a CSR para operaciones eficientes
        A = adj_matrix.tocsr()
        # A^2 da caminos de longitud 2
        A2 = A.dot(A)
        # Triángulos: (A^3)_ii / 2 = número de triángulos en nodo i
        # Pero A^3 es costoso. Usamos: triangulos_i = sum_j (A_ij * A2_ij) / 2
        # Esto cuenta los triángulos directamente
        triangulos = np.asarray(A.multiply(A2).sum(axis=1)).flatten() / 2.0
    else:
        A = adj_matrix
        A2 = A @ A
        triangulos = np.diag(A @ A2) / 2.0
    
    # Clustering local
    clustering = np.zeros(N)
    mask = grados > 1
    denominador = grados[mask] * (grados[mask] - 1) / 2.0
    clustering[mask] = triangulos[mask] / denominador
    
    return clustering


def calcular_clustering_promedio(adj_matrix):
    """
    Calcula el coeficiente de clustering promedio de la red.
    """
    grados = calcular_grados(adj_matrix)
    clustering = calcular_clustering_local(adj_matrix, grados)
    
    # Promedio solo de nodos con grado > 1
    mask = grados > 1
    if np.sum(mask) == 0:
        return 0.0
    
    return float(np.mean(clustering[mask]))


def calcular_transitividad(adj_matrix):
    """
    Calcula la transitividad global de la red.
    T = 3 * número_triángulos / número_tripletas_conectadas
    
    Más eficiente que clustering promedio para grafos grandes.
    """
    if sparse.issparse(adj_matrix):
        A = adj_matrix.tocsr()
        A2 = A.dot(A)
        # Número total de triángulos = trace(A^3) / 6
        # trace(A^3) = sum de A * A2 en diagonal
        num_triangulos = A.multiply(A2).sum() / 2.0  # Cada triángulo contado 3 veces en trace
    else:
        A = adj_matrix
        A2 = A @ A
        num_triangulos = np.trace(A @ A2) / 2.0
    
    # Número de tripletas conectadas = sum_i k_i * (k_i - 1) / 2
    grados = calcular_grados(adj_matrix)
    num_tripletas = np.sum(grados * (grados - 1)) / 2.0
    
    if num_tripletas == 0:
        return 0.0
    
    return float(3.0 * num_triangulos / num_tripletas)


def encontrar_componentes(adj_matrix):
    """
    Encuentra componentes conectados usando scipy.
    Retorna: (n_componentes, etiquetas, tamaños)
    """
    if not sparse.issparse(adj_matrix):
        adj_matrix = csr_matrix(adj_matrix)
    
    n_componentes, etiquetas = connected_components(
        adj_matrix, 
        directed=False, 
        return_labels=True
    )
    
    # Calcular tamaños de cada componente
    tamaños = np.bincount(etiquetas)
    
    return n_componentes, etiquetas, tamaños


def calcular_camino_promedio_componente(adj_matrix, nodos_componente):
    """
    Calcula el camino promedio dentro de un componente.
    Usa BFS para matrices dispersas (más eficiente que Floyd-Warshall).
    """
    n = len(nodos_componente)
    if n <= 1:
        return 0.0
    
    # Extraer submatriz del componente
    if sparse.issparse(adj_matrix):
        submatriz = adj_matrix[nodos_componente, :][:, nodos_componente]
    else:
        submatriz = adj_matrix[np.ix_(nodos_componente, nodos_componente)]
    
    # Usar shortest_path de scipy (Dijkstra por defecto para dispersas)
    # Para grafos no ponderados, BFS es equivalente
    if not sparse.issparse(submatriz):
        submatriz = csr_matrix(submatriz)
    
    # Calcular distancias
    distancias = shortest_path(
        submatriz, 
        method='D',  # Dijkstra (más eficiente para dispersas)
        directed=False,
        unweighted=True
    )
    
    # Promedio excluyendo diagonal e infinitos
    mask = ~np.isinf(distancias) & (distancias > 0)
    if np.sum(mask) == 0:
        return float('inf')
    
    return float(np.mean(distancias[mask]))


def calcular_camino_promedio(adj_matrix, etiquetas_componentes, tamaños_componentes):
    """
    Calcula el camino promedio de la componente gigante.
    """
    N = adj_matrix.shape[0]
    
    # Encontrar componente gigante
    idx_gigante = np.argmax(tamaños_componentes)
    tamaño_gigante = tamaños_componentes[idx_gigante]
    
    if tamaño_gigante <= 1:
        return float('inf'), 0.0
    
    # Nodos del componente gigante
    nodos_gigante = np.where(etiquetas_componentes == idx_gigante)[0]
    
    # Fracción del componente gigante
    fraccion = tamaño_gigante / N
    
    # Para componentes muy grandes (>5000), muestrear
    if tamaño_gigante > 5000:
        # Muestreo: calcular desde 500 nodos aleatorios
        n_muestra = min(500, tamaño_gigante)
        nodos_muestra = np.random.choice(nodos_gigante, n_muestra, replace=False)
        
        if not sparse.issparse(adj_matrix):
            adj_matrix = csr_matrix(adj_matrix)
        
        # Calcular distancias desde nodos de muestra
        distancias_totales = []
        for nodo in nodos_muestra:
            dist = shortest_path(
                adj_matrix,
                method='D',
                directed=False,
                unweighted=True,
                indices=nodo
            )
            dist_gigante = dist[nodos_gigante]
            dist_validas = dist_gigante[(dist_gigante > 0) & (~np.isinf(dist_gigante))]
            distancias_totales.extend(dist_validas)
        
        if len(distancias_totales) == 0:
            return float('inf'), fraccion
        
        return float(np.mean(distancias_totales)), fraccion
    else:
        # Componente pequeño: calcular exacto
        camino = calcular_camino_promedio_componente(adj_matrix, nodos_gigante)
        return camino, fraccion


def calcular_metricas_topologicas(adj_matrix, N):
    """
    Calcula todas las métricas topológicas de una red.
    Versión optimizada sin NetworkX.
    """
    metricas = {'nodos': N}
    
    # Asegurar que es matriz dispersa CSR
    if not sparse.issparse(adj_matrix):
        adj_matrix = csr_matrix(adj_matrix)
    else:
        adj_matrix = adj_matrix.tocsr()
    
    # Número de aristas
    n_aristas = adj_matrix.nnz // 2  # Dividir por 2 porque es simétrica
    metricas['aristas'] = n_aristas
    
    # Densidad
    metricas['densidad'] = 2 * n_aristas / (N * (N - 1)) if N > 1 else 0
    
    if n_aristas == 0:
        metricas['clustering'] = 0
        metricas['transitividad'] = 0
        metricas['camino'] = float('inf')
        metricas['fraccion_giant'] = 0
        metricas['grado_medio'] = 0
        metricas['n_componentes'] = N
        return metricas
    
    # Grados
    grados = calcular_grados(adj_matrix)
    metricas['grado_medio'] = float(np.mean(grados))
    
    # Clustering y transitividad (solo si hay suficientes aristas)
    if n_aristas > N // 2:
        metricas['clustering'] = calcular_clustering_promedio(adj_matrix)
        metricas['transitividad'] = calcular_transitividad(adj_matrix)
    else:
        metricas['clustering'] = 0
        metricas['transitividad'] = 0
    
    # Componentes conectados
    n_comp, etiquetas, tamaños = encontrar_componentes(adj_matrix)
    metricas['n_componentes'] = n_comp
    
    # Camino promedio y fracción gigante
    camino, fraccion = calcular_camino_promedio(adj_matrix, etiquetas, tamaños)
    metricas['camino'] = camino
    metricas['fraccion_giant'] = fraccion
    
    return metricas


# ============================================================
# FUNCIONES DE PROCESAMIENTO
# ============================================================

def clasificar_por_rango(estadisticas):
    """Clasifica distribución por rango (max - min)."""
    rango = estadisticas['max'] - estadisticas['min']
    return 'bimodal' if rango > 0.8 else 'unimodal'


def construir_matriz_adyacencia_sparse(valores, filas, columnas, N, umbral):
    """
    Construye matriz de adyacencia dispersa aplicando umbral.
    """
    mask = valores > umbral
    if not np.any(mask):
        return None, 0
    
    f = filas[mask]
    c = columnas[mask]
    v = np.ones(len(f), dtype=np.float32)
    
    # Crear matriz simétrica
    filas_full = np.concatenate([f, c])
    cols_full = np.concatenate([c, f])
    vals_full = np.concatenate([v, v])
    
    matriz = csr_matrix((vals_full, (filas_full, cols_full)), shape=(N, N))
    
    return matriz, len(f)


def procesar_una_matriz(args):
    """
    Procesa UNA matriz para TODOS los umbrales.
    Versión optimizada sin NetworkX.
    """
    ruta, N, umbrales, stats = args
    
    try:
        # Cargar matriz UNA SOLA VEZ
        datos = np.load(ruta)
        valores = datos['valores']
        
        # Índices del triángulo superior (calcular una sola vez)
        filas, columnas = np.triu_indices(N, k=1)
        
        resultados_umbrales = {}
        
        # Procesar TODOS los umbrales con la misma matriz cargada
        for umbral in umbrales:
            # Construir matriz dispersa
            adj_matrix, n_aristas = construir_matriz_adyacencia_sparse(
                valores, filas, columnas, N, umbral
            )
            
            if adj_matrix is None:
                resultados_umbrales[umbral] = {
                    'nodos': N,
                    'aristas': 0,
                    'densidad': 0,
                    'clustering': 0,
                    'transitividad': 0,
                    'camino': float('inf'),
                    'fraccion_giant': 0,
                    'grado_medio': 0,
                    'n_componentes': N
                }
                continue
            
            # Calcular métricas con implementación optimizada
            metricas = calcular_metricas_topologicas(adj_matrix, N)
            resultados_umbrales[umbral] = metricas
            
            # Liberar memoria
            del adj_matrix
        
        del valores, datos
        gc.collect()
        
        return {
            'tipo': clasificar_por_rango(stats),
            'resultados': resultados_umbrales
        }
    
    except Exception as e:
        print(f"    Error procesando {ruta}: {e}")
        import traceback
        traceback.print_exc()
        return None


def analizar_topologia_paralelo():
    """Analiza topología en PARALELO."""
    
    with open('resultados_consolidados.json') as f:
        data = json.load(f)
    
    # Preparar todas las tareas
    tareas = []
    alphas_por_idx = []
    
    for alpha_str, alpha_data in data.items():
        alpha = float(alpha_str)
        matrices_info = alpha_data['matrices']
        estadisticas = alpha_data['estadisticas']
        
        for info, stats in zip(matrices_info, estadisticas):
            tareas.append((
                info['ruta'],
                info['N'],
                UMBRALES,
                stats
            ))
            alphas_por_idx.append(alpha)
    
    print(f"\n{'='*70}")
    print(f"ANÁLISIS TOPOLÓGICO OPTIMIZADO (SIN NETWORKX)")
    print(f"{'='*70}")
    print(f"Total matrices: {len(tareas)}")
    print(f"Workers: {N_WORKERS}")
    print(f"Umbrales: {UMBRALES}")
    
    # Procesar en paralelo
    resultados_raw = {'bimodal': {}, 'unimodal': {}}
    
    with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
        futures = {executor.submit(procesar_una_matriz, t): i for i, t in enumerate(tareas)}
        
        for future in tqdm(as_completed(futures), total=len(futures), 
                          desc="Procesando matrices"):
            idx = futures[future]
            try:
                res = future.result()
                if res:
                    alpha = alphas_por_idx[idx]
                    tipo = res['tipo']
                    
                    if alpha not in resultados_raw[tipo]:
                        resultados_raw[tipo][alpha] = {u: [] for u in UMBRALES}
                    
                    for umbral, metricas in res['resultados'].items():
                        resultados_raw[tipo][alpha][umbral].append(metricas)
            
            except Exception as e:
                print(f"Error en tarea {idx}: {e}")
    
    # Calcular promedios
    promedios = {'bimodal': {}, 'unimodal': {}}
    
    for tipo in ['bimodal', 'unimodal']:
        for alpha in resultados_raw[tipo]:
            promedios[tipo][alpha] = {}
            for umbral in UMBRALES:
                lista = resultados_raw[tipo][alpha][umbral]
                if lista:
                    # Filtrar caminos infinitos para el promedio
                    caminos_finitos = [m['camino'] for m in lista if m['camino'] != float('inf')]
                    camino_media = float(np.mean(caminos_finitos)) if caminos_finitos else float('inf')
                    
                    promedios[tipo][alpha][umbral] = {
                        'n_analizadas': len(lista),
                        'densidad_media': float(np.mean([m['densidad'] for m in lista])),
                        'clustering_media': float(np.mean([m['clustering'] for m in lista])),
                        'transitividad_media': float(np.mean([m['transitividad'] for m in lista])),
                        'camino_media': camino_media,
                        'fraccion_giant_media': float(np.mean([m['fraccion_giant'] for m in lista])),
                        'grado_medio': float(np.mean([m['grado_medio'] for m in lista])),
                        'n_componentes_media': float(np.mean([m['n_componentes'] for m in lista])),
                    }
    
    return promedios


def graficar_topologia_por_tipo(resultados):
    """Genera gráficas separadas por tipo."""
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    
    umbral_ref = 0.24
    
    for tipo, color, marker in [('bimodal', 'red', 's'), ('unimodal', 'blue', 'o')]:
        if not resultados[tipo]:
            continue
            
        alphas = sorted(resultados[tipo].keys())
        
        # Filtrar alphas que tienen el umbral de referencia
        alphas_validos = [a for a in alphas if umbral_ref in resultados[tipo][a]]
        
        if not alphas_validos:
            continue
        
        # Densidad
        ax = axes[0, 0]
        densidades = [resultados[tipo][a][umbral_ref]['densidad_media'] for a in alphas_validos]
        ax.plot(alphas_validos, densidades, marker=marker, color=color, 
               label=tipo, linewidth=2, markersize=8)
        
        # Clustering
        ax = axes[0, 1]
        clustering = [resultados[tipo][a][umbral_ref]['clustering_media'] for a in alphas_validos]
        ax.plot(alphas_validos, clustering, marker=marker, color=color,
               label=tipo, linewidth=2, markersize=8)
        
        # Camino
        ax = axes[0, 2]
        caminos = [resultados[tipo][a][umbral_ref]['camino_media'] for a in alphas_validos]
        # Filtrar infinitos para graficar
        caminos_plot = [c if c != float('inf') else np.nan for c in caminos]
        ax.plot(alphas_validos, caminos_plot, marker=marker, color=color,
               label=tipo, linewidth=2, markersize=8)
        
        # Fracción giant
        ax = axes[1, 0]
        giant = [resultados[tipo][a][umbral_ref]['fraccion_giant_media'] for a in alphas_validos]
        ax.plot(alphas_validos, giant, marker=marker, color=color,
               label=tipo, linewidth=2, markersize=8)
        
        # Grado medio
        ax = axes[1, 1]
        grado = [resultados[tipo][a][umbral_ref]['grado_medio'] for a in alphas_validos]
        ax.plot(alphas_validos, grado, marker=marker, color=color,
               label=tipo, linewidth=2, markersize=8)
        
        # Número de componentes
        ax = axes[1, 2]
        componentes = [resultados[tipo][a][umbral_ref]['n_componentes_media'] for a in alphas_validos]
        ax.plot(alphas_validos, componentes, marker=marker, color=color,
               label=tipo, linewidth=2, markersize=8)
    
    # Configurar ejes
    titulos = ['Densidad', 'Clustering', 'Camino promedio', 
               'Fracción componente gigante', 'Grado medio', 'Nº componentes']
    
    for i, ax in enumerate(axes.flat):
        ax.set_xscale('log')
        ax.set_xlabel('α')
        ax.set_title(titulos[i])
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('topologia_por_tipo_optimizado.png', dpi=300)
    plt.show()
    print("Gráfica guardada: topologia_por_tipo_optimizado.png")


if __name__ == "__main__":
    start = time.time()
    
    resultados = analizar_topologia_paralelo()
    
    with open('analisis_topologico_por_tipo_optimizado.json', 'w') as f:
        json.dump(resultados, f, indent=2)
    
    print(f"\nAnálisis topológico completado")
    print(f"Tiempo total: {(time.time()-start)/60:.1f} minutos")
    
    graficar_topologia_por_tipo(resultados)
