#!/usr/bin/env python3
"""
Análisis de métricas globales: volatilidad, Lempel-Ziv y entropía
"""
import json
import numpy as np
import matplotlib.pyplot as plt
import glob
import os

def recopilar_metricas_globales():
    """
    Recopila volatilidad, LZ y entropía de todas las realizaciones.
    """
    # Cargar datos consolidados (para tener la lista de archivos)
    with open('resultados_consolidados.json') as f:
        data = json.load(f)
    
    resultados = {}
    
    print("\n" + "="*70)
    print("ANÁLISIS DE MÉTRICAS GLOBALES")
    print("="*70)
    
    for alpha_str, alpha_data in data.items():
        alpha = float(alpha_str)
        print(f"\n📊 α = {alpha}")
        
        n_realizaciones = alpha_data['n_realizaciones']
        print(f"  {n_realizaciones} realizaciones")
        
        # Inicializar listas para este α
        volatilidades = []
        lempel_ziv_medias = []
        entropias_medias = []
        
        # Para cada realización, buscar sus archivos
        for idx in range(n_realizaciones):
            # La información de rutas no está en consolidados,
            # necesitamos buscarlas directamente
            carpeta = f"resultados_{alpha}"
            archivos = glob.glob(f"{carpeta}/volatilidad_*.json")
            
            if idx < len(archivos):
                with open(archivos[idx]) as f:
                    vol_data = json.load(f)
                volatilidades.append(vol_data['resultados']['sigma2_N'])
            
            # Lempel-Ziv
            archivos_lz = glob.glob(f"{carpeta}/lempel_ziv_*.json")
            if idx < len(archivos_lz):
                with open(archivos_lz[idx]) as f:
                    lz_data = json.load(f)
                # Promedio sobre agentes
                lz_vals = [a['lempel_ziv'] for a in lz_data['agentes']]
                lempel_ziv_medias.append(np.mean(lz_vals))
            
            # Entropía
            archivos_ent = glob.glob(f"{carpeta}/entropia_*.json")
            if idx < len(archivos_ent):
                with open(archivos_ent[idx]) as f:
                    ent_data = json.load(f)
                ent_vals = [a['h_estimado'] for a in ent_data['agentes'] if 'h_estimado' in a]
                if ent_vals:
                    entropias_medias.append(np.mean(ent_vals))
        
        resultados[alpha] = {
            'n_realizaciones': n_realizaciones,
            'volatilidad': {
                'media': float(np.mean(volatilidades)) if volatilidades else 0,
                'std': float(np.std(volatilidades)) if volatilidades else 0,
                'valores': volatilidades
            },
            'lempel_ziv': {
                'media': float(np.mean(lempel_ziv_medias)) if lempel_ziv_medias else 0,
                'std': float(np.std(lempel_ziv_medias)) if lempel_ziv_medias else 0,
                'valores': lempel_ziv_medias
            },
            'entropia': {
                'media': float(np.mean(entropias_medias)) if entropias_medias else 0,
                'std': float(np.std(entropias_medias)) if entropias_medias else 0,
                'valores': entropias_medias
            }
        }
        
        print(f"  Volatilidad σ²/N: {resultados[alpha]['volatilidad']['media']:.6f} ± {resultados[alpha]['volatilidad']['std']:.6f}")
        print(f"  Lempel-Ziv: {resultados[alpha]['lempel_ziv']['media']:.4f} ± {resultados[alpha]['lempel_ziv']['std']:.4f}")
        print(f"  Entropía h: {resultados[alpha]['entropia']['media']:.4f} ± {resultados[alpha]['entropia']['std']:.4f}")
    
    return resultados

def graficar_metricas(resultados):
    """Genera gráficas comparativas de todas las métricas."""
    
    alphas = sorted(resultados.keys())
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Volatilidad (la métrica clásica del MG)
    ax = axes[0, 0]
    volatilidades = [resultados[a]['volatilidad']['media'] for a in alphas]
    errores_vol = [resultados[a]['volatilidad']['std'] for a in alphas]
    
    ax.errorbar(alphas, volatilidades, yerr=errores_vol, fmt='o-', 
                capsize=5, linewidth=2, markersize=8, color='red')
    ax.set_xscale('log')
    ax.set_xlabel('α')
    ax.set_ylabel('σ²/N')
    ax.set_title('Volatilidad del sistema')
    ax.grid(True, alpha=0.3)
    
    # 2. Complejidad Lempel-Ziv
    ax = axes[0, 1]
    lz_medias = [resultados[a]['lempel_ziv']['media'] for a in alphas]
    errores_lz = [resultados[a]['lempel_ziv']['std'] for a in alphas]
    
    ax.errorbar(alphas, lz_medias, yerr=errores_lz, fmt='s-', 
                capsize=5, linewidth=2, markersize=8, color='blue')
    ax.set_xscale('log')
    ax.set_xlabel('α')
    ax.set_ylabel('C_LZ')
    ax.set_title('Complejidad de Lempel-Ziv (promedio por agente)')
    ax.grid(True, alpha=0.3)
    
    # 3. Densidad de entropía
    ax = axes[1, 0]
    ent_medias = [resultados[a]['entropia']['media'] for a in alphas]
    errores_ent = [resultados[a]['entropia']['std'] for a in alphas]
    
    ax.errorbar(alphas, ent_medias, yerr=errores_ent, fmt='^-', 
                capsize=5, linewidth=2, markersize=8, color='green')
    ax.set_xscale('log')
    ax.set_xlabel('α')
    ax.set_ylabel('h (entropy rate)')
    ax.set_title('Densidad de entropía')
    ax.grid(True, alpha=0.3)
    
    # 4. Correlación entre métricas
    ax = axes[1, 1]
    scatter = ax.scatter(volatilidades, lz_medias, c=alphas, cmap='viridis', 
                        s=100, alpha=0.8)
    for i, alpha in enumerate(alphas):
        ax.annotate(f'{alpha:.2f}', (volatilidades[i], lz_medias[i]), 
                   fontsize=9, ha='center')
    ax.set_xlabel('Volatilidad σ²/N')
    ax.set_ylabel('Complejidad LZ')
    ax.set_title('Volatilidad vs Complejidad')
    plt.colorbar(scatter, ax=ax, label='α')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('metricas_globales.png', dpi=300)
    plt.show()

def tabla_latex(resultados):
    """Genera tabla en formato LaTeX."""
    
    print("\n" + "="*70)
    print("TABLA LaTeX PARA LA TESIS")
    print("="*70)
    
    print("\n\\begin{table}[h]")
    print("\\centering")
    print("\\caption{Métricas globales del Minority Game}")
    print("\\begin{tabular}{|c|c|c|c|}")
    print("\\hline")
    print("α & Volatilidad σ²/N & Lempel-Ziv & Entropía h \\\\ \\hline")
    
    for alpha in sorted(resultados.keys()):
        vol = resultados[alpha]['volatilidad']
        lz = resultados[alpha]['lempel_ziv']
        ent = resultados[alpha]['entropia']
        
        print(f"{alpha:.4f} & "
              f"{vol['media']:.6f} ± {vol['std']:.6f} & "
              f"{lz['media']:.4f} ± {lz['std']:.4f} & "
              f"{ent['media']:.4f} ± {ent['std']:.4f} \\\\ \\hline")
    
    print("\\end{tabular}")
    print("\\end{table}")

if __name__ == "__main__":
    resultados = recopilar_metricas_globales()
    
    # Guardar resultados
    with open('metricas_globales.json', 'w') as f:
        resultados_json = {}
        for alpha, data in resultados.items():
            resultados_json[str(alpha)] = {
                'volatilidad': data['volatilidad'],
                'lempel_ziv': data['lempel_ziv'],
                'entropia': data['entropia']
            }
        json.dump(resultados_json, f, indent=2)
    
    print(f"\n✅ Resultados guardados en: metricas_globales.json")
    
    # Graficar
    graficar_metricas(resultados)
    
    # Generar tabla LaTeX
    tabla_latex(resultados)