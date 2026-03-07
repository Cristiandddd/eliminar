#!/usr/bin/env python3
"""
Análisis de redes - VERSIÓN SIMPLIFICADA (CON JSON)
"""
import json
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm
import gc
import os
from scipy import sparse

# ============================================================
# CONFIGURACIÓN
# ============================================================

UMBRALES = [0.20, 0.22, 0.24, 0.26, 0.28, 0.30]

def clasificar_por_rango(estadisticas):
    """
    Clasifica distribución por rango (max - min).
    """
    rango = estadisticas['max'] - estadisticas['min']
    return 'bimodal' if rango > 0.9 else 'unimodal'

def cargar_y_procesar(ruta, N, umbral):
    """Carga matriz y binariza según umbral."""
    try:
        datos = np.load(ruta)
        valores = datos['valores']
        
        # Índices del triángulo superior
        filas, columnas = np.triu_indices(N, k=1)
        
        # Filtrar por umbral
        mask = valores > umbral
        return len(valores[mask])  # Número de aristas
    except Exception as e:
        print(f"    Error cargando {ruta}: {e}")
        return 0

def analizar_todo():
    """Analiza todas las redes."""
    
    # Cargar datos consolidados
    if not os.path.exists('resultados_consolidados.json'):
        print("❌ No se encuentra resultados_consolidados.json")
        return None
    
    with open('resultados_consolidados.json') as f:
        data = json.load(f)
    
    resultados = {
        'bimodal': {},
        'unimodal': {}
    }
    
    print("\n" + "="*70)
    print("ANÁLISIS DE REDES")
    print("="*70)
    
    for alpha_str, alpha_data in data.items():
        alpha = float(alpha_str)
        print(f"\n📊 α = {alpha}")
        
        matrices = alpha_data['matrices']
        estadisticas = alpha_data['estadisticas']
        
        if not matrices:
            print(f"  No hay matrices para α={alpha}")
            continue
        
        print(f"  {len(matrices)} matrices disponibles")
        
        # Clasificar CADA realización
        for idx, (mat_info, stats) in enumerate(zip(matrices, estadisticas)):
            tipo = clasificar_por_rango(stats)
            print(f"  Realización {idx+1}: {tipo}")
            
            # Inicializar estructuras
            if alpha not in resultados[tipo]:
                resultados[tipo][alpha] = {u: [] for u in UMBRALES}
            
            # Evaluar todos los umbrales
            for umbral in UMBRALES:
                n_aristas = cargar_y_procesar(mat_info['ruta'], mat_info['N'], umbral)
                resultados[tipo][alpha][umbral].append(n_aristas)
    
    # Calcular promedios
    promedios = {'bimodal': {}, 'unimodal': {}}
    
    for tipo in ['bimodal', 'unimodal']:
        for alpha in resultados[tipo]:
            promedios[tipo][alpha] = {}
            N = data[str(alpha)]['n_agentes']
            
            for umbral in UMBRALES:
                aristas = resultados[tipo][alpha][umbral]
                if aristas:
                    densidades = [2*a/(N*(N-1)) for a in aristas]
                    promedios[tipo][alpha][umbral] = {
                        'n': len(aristas),
                        'densidad_media': float(np.mean(densidades)),
                        'densidad_std': float(np.std(densidades)),
                        'aristas_media': float(np.mean(aristas)),
                        'aristas_std': float(np.std(aristas))
                    }
    
    return promedios

def graficar_resultados(resultados):
    """Genera gráficas."""
    
    if not resultados or (not resultados['bimodal'] and not resultados['unimodal']):
        print("❌ No hay datos para graficar")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Fracción de realizaciones bimodales por α
    ax = axes[0, 0]
    alphas_bimodal = sorted(resultados['bimodal'].keys())
    alphas_unimodal = sorted(resultados['unimodal'].keys())
    alphas_totales = sorted(set(alphas_bimodal) | set(alphas_unimodal))
    
    fracciones = []
    for a in alphas_totales:
        n_bimodal = resultados['bimodal'].get(a, {}).get(0.5, {}).get('n', 0)
        n_unimodal = resultados['unimodal'].get(a, {}).get(0.5, {}).get('n', 0)
        total = n_bimodal + n_unimodal
        fracciones.append(n_bimodal / total if total > 0 else 0)
    
    if fracciones:
        ax.plot(alphas_totales, fracciones, 'o-', color='purple', linewidth=2, markersize=8)
        ax.set_xscale('log')
        ax.set_xlabel('α')
        ax.set_ylabel('Fracción bimodal')
        ax.set_title('Probabilidad de estado bimodal')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([-0.05, 1.05])
    
    # 2. Densidad vs α para diferentes umbrales (bimodal)
    ax = axes[0, 1]
    if resultados['bimodal']:
        for umbral in [0.30, 0.50, 0.70]:
            alphas = sorted(resultados['bimodal'].keys())
            densidades = []
            alphas_validos = []
            for a in alphas:
                if umbral in resultados['bimodal'][a]:
                    densidades.append(resultados['bimodal'][a][umbral]['densidad_media'])
                    alphas_validos.append(a)
            if densidades:
                ax.plot(alphas_validos, densidades, 'o-', label=f'θ={umbral}', linewidth=2)
        
        ax.set_xscale('log')
        ax.set_xlabel('α')
        ax.set_ylabel('Densidad')
        ax.set_title('Redes BIMODALES')
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'Sin datos bimodales', ha='center', va='center')
    
    # 3. Densidad vs α (unimodal)
    ax = axes[1, 0]
    if resultados['unimodal']:
        for umbral in [0.25, 0.30, 0.35]:
            alphas = sorted(resultados['unimodal'].keys())
            densidades = []
            alphas_validos = []
            for a in alphas:
                if umbral in resultados['unimodal'][a]:
                    densidades.append(resultados['unimodal'][a][umbral]['densidad_media'])
                    alphas_validos.append(a)
            if densidades:
                ax.plot(alphas_validos, densidades, 's-', label=f'θ={umbral}', linewidth=2)
        
        ax.set_xscale('log')
        ax.set_xlabel('α')
        ax.set_ylabel('Densidad')
        ax.set_title('Redes UNIMODALES')
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'Sin datos unimodales', ha='center', va='center')
    
    # 4. Comparación directa (umbral 0.5)
    ax = axes[1, 1]
    alphas_comunes = sorted(set(resultados['bimodal'].keys()) & set(resultados['unimodal'].keys()))
    
    if alphas_comunes:
        x = range(len(alphas_comunes))
        dens_bimodal = []
        dens_unimodal = []
        
        for a in alphas_comunes:
            if 0.5 in resultados['bimodal'][a]:
                dens_bimodal.append(resultados['bimodal'][a][0.5]['densidad_media'])
            else:
                dens_bimodal.append(0)
            
            if 0.5 in resultados['unimodal'][a]:
                dens_unimodal.append(resultados['unimodal'][a][0.5]['densidad_media'])
            else:
                dens_unimodal.append(0)
        
        ax.bar([i-0.2 for i in x], dens_bimodal, width=0.4, label='Bimodal', alpha=0.7, color='red')
        ax.bar([i+0.2 for i in x], dens_unimodal, width=0.4, label='Unimodal', alpha=0.7, color='blue')
        ax.set_xticks(x)
        ax.set_xticklabels([f'{a:.2f}' for a in alphas_comunes])
        ax.set_xlabel('α')
        ax.set_ylabel('Densidad (θ=0.5)')
        ax.set_title('Comparación directa')
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No hay α comunes', ha='center', va='center')
    
    plt.tight_layout()
    plt.savefig('analisis_redes.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    resultados = analizar_todo()
    
    if resultados:
        # Guardar JSON
        with open('analisis_redes.json', 'w') as f:
            json.dump(resultados, f, indent=2)
        
        print("\n✅ JSON guardado: analisis_redes.json")
        
        # Generar gráfica
        graficar_resultados(resultados)
        
        print("\n✅ Análisis completado")
        print("   📊 Gráfica: analisis_redes.png")
        print("   📁 Datos: analisis_redes.json")
    else:
        print("\n❌ No se pudo completar el análisis")