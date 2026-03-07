#!/usr/bin/env python3
"""
Análisis topológico de redes - VERSIÓN PARALELIZADA Y OPTIMIZADA
"""
import json
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm
import gc
import os
from scipy import sparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import time

# ============================================================
# CONFIGURACIÓN
# ============================================================

UMBRALES = [0.20, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29, 0.30]
N_WORKERS = 16

def clasificar_por_rango(estadisticas):
    """Clasifica distribución por rango (max - min)."""
    rango = estadisticas['max'] - estadisticas['min']
    return 'bimodal' if rango > 0.8 else 'unimodal'

def procesar_una_matriz(args):
    """
    Procesa UNA matriz para TODOS los umbrales.
    Esto evita cargar la matriz múltiples veces.
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
            mask = valores > umbral
            if not np.any(mask):
                # Si no hay aristas, métricas vacías
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
            
            filas_mask = filas[mask]
            columnas_mask = columnas[mask]
            valores_mask = valores[mask]
            
            n_aristas = len(valores_mask)
            
            # Crear grafo directamente (más rápido que matriz dispersa)
            G = nx.Graph()
            G.add_nodes_from(range(N))
            edges = [(filas_mask[i], columnas_mask[i], {'weight': valores_mask[i]}) 
                    for i in range(n_aristas)]
            G.add_edges_from(edges)
            
            # Añadir aristas simétricas
            edges_sym = [(columnas_mask[i], filas_mask[i], {'weight': valores_mask[i]}) 
                        for i in range(n_aristas)]
            G.add_edges_from(edges_sym)
            
            # Calcular métricas
            metricas = {
                'nodos': N,
                'aristas': n_aristas,
                'densidad': 2 * n_aristas / (N * (N - 1)) if N > 1 else 0,
            }
            
            # Clustering (solo si hay suficientes aristas)
            if n_aristas > N:  # Mínimo para tener algún triángulo
                try:
                    metricas['clustering'] = nx.average_clustering(G)
                except:
                    metricas['clustering'] = 0
                try:
                    metricas['transitividad'] = nx.transitivity(G)
                except:
                    metricas['transitividad'] = 0
            else:
                metricas['clustering'] = 0
                metricas['transitividad'] = 0
            
            # Grado
            grados = [d for n, d in G.degree()]
            metricas['grado_medio'] = float(np.mean(grados))
            
            # Componentes
            metricas['n_componentes'] = nx.number_connected_components(G)
            
            # Componente gigante y camino
            if nx.is_connected(G):
                metricas['fraccion_giant'] = 1.0
                try:
                    metricas['camino'] = nx.average_shortest_path_length(G)
                except:
                    metricas['camino'] = float('inf')
            else:
                componentes = sorted(nx.connected_components(G), key=len, reverse=True)
                if componentes:
                    G_giant = G.subgraph(componentes[0])
                    metricas['fraccion_giant'] = len(componentes[0]) / N
                    try:
                        metricas['camino'] = nx.average_shortest_path_length(G_giant)
                    except:
                        metricas['camino'] = float('inf')
                else:
                    metricas['fraccion_giant'] = 0
                    metricas['camino'] = float('inf')
            
            resultados_umbrales[umbral] = metricas
        
        return {
            'tipo': clasificar_por_rango(stats),
            'resultados': resultados_umbrales
        }
    
    except Exception as e:
        print(f"    Error procesando {ruta}: {e}")
        return None

def analizar_topologia_paralelo():
    """Analiza topología en PARALELO."""
    
    with open('resultados_consolidados.json') as f:
        data = json.load(f)
    
    # Preparar todas las tareas
    tareas = []
    for alpha_str, alpha_data in data.items():
        alpha = float(alpha_str)
        matrices_info = alpha_data['matrices']
        estadisticas = alpha_data['estadisticas']
        
        for info, stats in zip(matrices_info, estadisticas):
            tareas.append((
                info['ruta'],
                info['N'],
                UMBRALES,
                stats,
                alpha  # Guardamos alpha para organizar después
            ))
    
    print(f"\n{'='*70}")
    print(f"ANÁLISIS TOPOLÓGICO PARALELIZADO")
    print(f"{'='*70}")
    print(f"Total matrices: {len(tareas)}")
    print(f"Workers: {N_WORKERS}")
    print(f"Umbrales: {UMBRALES}")
    
    # Procesar en paralelo
    resultados_raw = {'bimodal': {}, 'unimodal': {}}
    
    with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
        futures = [executor.submit(procesar_una_matriz, t[:4]) for t in tareas]
        
        for i, future in enumerate(tqdm(as_completed(futures), total=len(futures), 
                                       desc="Procesando matrices")):
            try:
                res = future.result()
                if res:
                    alpha = tareas[i][4]  # Recuperamos alpha
                    tipo = res['tipo']
                    
                    if alpha not in resultados_raw[tipo]:
                        resultados_raw[tipo][alpha] = {u: [] for u in UMBRALES}
                    
                    for umbral, metricas in res['resultados'].items():
                        resultados_raw[tipo][alpha][umbral].append(metricas)
            
            except Exception as e:
                print(f"Error en tarea {i}: {e}")
    
    # Calcular promedios
    promedios = {'bimodal': {}, 'unimodal': {}}
    
    for tipo in ['bimodal', 'unimodal']:
        for alpha in resultados_raw[tipo]:
            promedios[tipo][alpha] = {}
            for umbral in UMBRALES:
                lista = resultados_raw[tipo][alpha][umbral]
                if lista:
                    promedios[tipo][alpha][umbral] = {
                        'n_analizadas': len(lista),
                        'densidad_media': float(np.mean([m['densidad'] for m in lista])),
                        'clustering_media': float(np.mean([m['clustering'] for m in lista])),
                        'transitividad_media': float(np.mean([m['transitividad'] for m in lista])),
                        'camino_media': float(np.mean([m['camino'] for m in lista if m['camino'] != float('inf')])),
                        'fraccion_giant_media': float(np.mean([m['fraccion_giant'] for m in lista])),
                        'grado_medio': float(np.mean([m['grado_medio'] for m in lista])),
                        'n_componentes_media': float(np.mean([m['n_componentes'] for m in lista])),
                    }
    
    return promedios

def graficar_topologia_por_tipo(resultados):
    """Genera gráficas separadas por tipo."""
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    
    for tipo, color, marker in [('bimodal', 'red', 's'), ('unimodal', 'blue', 'o')]:
        if not resultados[tipo]:
            continue
            
        alphas = sorted(resultados[tipo].keys())
        
        # Densidad
        ax = axes[0, 0]
        densidades = [resultados[tipo][a][0.24]['densidad_media'] for a in alphas if 0.24 in resultados[tipo][a]]
        if densidades:
            ax.plot(alphas[:len(densidades)], densidades, marker=marker, color=color, 
                   label=tipo, linewidth=2, markersize=8)
        
        # Clustering
        ax = axes[0, 1]
        clustering = [resultados[tipo][a][0.24]['clustering_media'] for a in alphas if 0.24 in resultados[tipo][a]]
        if clustering:
            ax.plot(alphas[:len(clustering)], clustering, marker=marker, color=color,
                   label=tipo, linewidth=2, markersize=8)
        
        # Camino
        ax = axes[0, 2]
        caminos = [resultados[tipo][a][0.24]['camino_media'] for a in alphas if 0.24 in resultados[tipo][a]]
        if caminos:
            ax.plot(alphas[:len(caminos)], caminos, marker=marker, color=color,
                   label=tipo, linewidth=2, markersize=8)
        
        # Fracción giant
        ax = axes[1, 0]
        giant = [resultados[tipo][a][0.24]['fraccion_giant_media'] for a in alphas if 0.24 in resultados[tipo][a]]
        if giant:
            ax.plot(alphas[:len(giant)], giant, marker=marker, color=color,
                   label=tipo, linewidth=2, markersize=8)
        
        # Grado medio
        ax = axes[1, 1]
        grado = [resultados[tipo][a][0.24]['grado_medio'] for a in alphas if 0.24 in resultados[tipo][a]]
        if grado:
            ax.plot(alphas[:len(grado)], grado, marker=marker, color=color,
                   label=tipo, linewidth=2, markersize=8)
    
    # Configurar ejes
    titulos = ['Densidad', 'Clustering', 'Camino', 
               'Fracción Giant', 'Grado medio', 'Componentes']
    
    for i, ax in enumerate(axes.flat):
        ax.set_xscale('log')
        ax.set_xlabel('α')
        ax.set_title(titulos[i] if i < 5 else '')
        if i < 5:
            ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Sexto subplot: número de componentes
    ax = axes[1, 2]
    for tipo, color, marker in [('bimodal', 'red', 's'), ('unimodal', 'blue', 'o')]:
        if resultados[tipo]:
            alphas = sorted(resultados[tipo].keys())
            componentes = [resultados[tipo][a][0.24]['n_componentes_media'] for a in alphas if 0.24 in resultados[tipo][a]]
            if componentes:
                ax.plot(alphas[:len(componentes)], componentes, marker=marker, color=color,
                       label=tipo, linewidth=2, markersize=8)
    ax.set_xscale('log')
    ax.set_xlabel('α')
    ax.set_title('Nº componentes')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('topologia_por_tipo.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    start = time.time()
    
    resultados = analizar_topologia_paralelo()
    
    with open('analisis_topologico_por_tipo.json', 'w') as f:
        json.dump(resultados, f, indent=2)
    
    print(f"\n✅ Análisis topológico por tipo completado")
    print(f"   Tiempo total: {(time.time()-start)/60:.1f} minutos")
    
    graficar_topologia_por_tipo(resultados)