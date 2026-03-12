#!/usr/bin/env python3
"""
Cálculo de modularidad Q para redes de co-victoria del MG.
Adaptado para trabajar con tus archivos NPZ y resultados_consolidados.json
"""

import numpy as np
import networkx as nx
import json
import os
import glob
from collections import defaultdict

try:
    import community as community_louvain
    USE_LOUVAIN_PACKAGE = True
except ImportError:
    USE_LOUVAIN_PACKAGE = False
    print("Instalando python-louvain para mejor rendimiento:")
    print("pip install python-louvain")

def cargar_matriz_desde_npz(archivo_npz):
    """Carga matriz de co-victoria desde archivo NPZ (triángulo superior)."""
    datos = np.load(archivo_npz)
    valores = datos['valores']
    N = datos['N']
    
    # Reconstruir matriz completa desde triángulo superior
    W = np.zeros((N, N))
    iu = np.triu_indices(N, k=1)
    W[iu] = valores
    W = W + W.T
    
    return W

def detectar_comunidades_y_modularidad(W, theta=0.24):
    """
    Detecta comunidades en la red de co-victoria y calcula modularidad.
    
    Args:
        W: matriz de pesos de co-victoria (NxN)
        theta: umbral de binarización
    
    Returns:
        Q: modularidad
        n_comunidades: número de comunidades detectadas
    """
    N = W.shape[0]
    
    # Binarizar la red
    A = (W > theta).astype(int)
    np.fill_diagonal(A, 0)
    
    # Crear grafo
    G = nx.from_numpy_array(A)
    
    # Verificar que el grafo tiene aristas
    n_aristas = G.number_of_edges()
    if n_aristas == 0:
        print(f"  Advertencia: Red vacía para theta={theta}")
        return 0.0, N
    
    # Detectar comunidades con Louvain
    if USE_LOUVAIN_PACKAGE:
        comunidades = community_louvain.best_partition(G)
        Q = community_louvain.modularity(comunidades, G)
    else:
        from networkx.algorithms.community import louvain_communities
        comunidades_lista = louvain_communities(G, seed=42)
        Q = nx.community.modularity(G, comunidades_lista)
    
    n_comunidades = len(set(comunidades.values())) if USE_LOUVAIN_PACKAGE else len(comunidades_lista)
    
    return Q, n_comunidades

def analizar_modularidad_desde_consolidado(archivo_consolidado="resultados_consolidados.json", 
                                           theta=0.24):
    """
    Analiza modularidad usando el archivo resultados_consolidados.json
    """
    # Cargar consolidado
    with open(archivo_consolidado, 'r') as f:
        data = json.load(f)
    
    resultados = []
    
    print("\n" + "="*70)
    print("ANÁLISIS DE MODULARIDAD Q")
    print("="*70)
    
    for alpha_str, alpha_data in data.items():
        alpha = float(alpha_str)
        print(f"\n📊 α = {alpha}")
        
        matrices_info = alpha_data.get('matrices', [])
        if not matrices_info:
            print(f"  No hay matrices para α={alpha}")
            continue
        
        print(f"  {len(matrices_info)} matrices disponibles")
        
        Qs = []
        n_comms = []
        
        for i, info in enumerate(matrices_info):
            try:
                W = cargar_matriz_desde_npz(info['ruta'])
                Q, n_comm = detectar_comunidades_y_modularidad(W, theta)
                Qs.append(Q)
                n_comms.append(n_comm)
                print(f"    Matriz {i+1}: Q={Q:.4f}, comunidades={n_comm}")
            except Exception as e:
                print(f"    Error en matriz {i+1}: {e}")
        
        if Qs:
            resultados.append({
                'alpha': alpha,
                'Q_media': float(np.mean(Qs)),
                'Q_std': float(np.std(Qs)),
                'n_comunidades_media': float(np.mean(n_comms)),
                'n_comunidades_std': float(np.std(n_comms)),
                'n_muestras': len(Qs)
            })
            
            print(f"\n  ✅ α={alpha:.3f}: Q = {np.mean(Qs):.4f} ± {np.std(Qs):.4f}")
    
    return resultados

def graficar_modularidad(resultados):
    """Genera gráfica de Q vs α."""
    import matplotlib.pyplot as plt
    
    alphas = [r['alpha'] for r in resultados]
    Qs = [r['Q_media'] for r in resultados]
    errores = [r['Q_std'] for r in resultados]
    
    plt.figure(figsize=(10, 6))
    plt.errorbar(alphas, Qs, yerr=errores, fmt='o-', capsize=5, linewidth=2)
    plt.axhline(y=0.1, color='gray', linestyle='--', alpha=0.5, label='Q=0.1 (límite de estructura)')
    plt.xlabel('α')
    plt.ylabel('Modularidad Q')
    plt.title('Modularidad de redes de co-victoria (θ=0.24)')
    plt.xscale('log')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('modularidad_vs_alpha.png', dpi=300)
    plt.show()
    print("Gráfica guardada: modularidad_vs_alpha.png")

if __name__ == "__main__":
    # Ejecutar análisis
    resultados = analizar_modularidad_desde_consolidado()
    
    # Guardar resultados
    with open('modularidad_resultados.json', 'w') as f:
        json.dump(resultados, f, indent=2)
    
    print("\n✅ Análisis completado")
    print("📁 Resultados guardados en: modularidad_resultados.json")
    
    # Graficar
    if resultados:
        graficar_modularidad(resultados)