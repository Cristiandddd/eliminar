#!/usr/bin/env python3
"""
Análisis de distribuciones de co-victoria
"""
import json
import numpy as np
import matplotlib.pyplot as plt

def analizar_distribuciones():
    """Analiza distribuciones de co-victoria para cada α"""
    
    # Cargar datos consolidados
    with open('resultados_consolidados.json') as f:
        data = json.load(f)
    
    alphas = sorted([float(a) for a in data.keys()])
    
    resultados = {}
    
    print("\n" + "="*60)
    print("ANÁLISIS DE DISTRIBUCIONES")
    print("="*60)
    
    for alpha in alphas:
        alpha_str = str(alpha)
        realizaciones = data[alpha_str]['estadisticas']
        
        print(f"\n📊 α = {alpha} ({len(realizaciones)} realizaciones)")
        
        medias = [r['media'] for r in realizaciones]
        desviaciones = [r['desviacion'] for r in realizaciones]
        mins = [r['min'] for r in realizaciones]
        maxs = [r['max'] for r in realizaciones]
        
        resultados[alpha] = {
            'n_realizaciones': len(realizaciones),
            'media_global': float(np.mean(medias)),
            'media_std': float(np.std(medias)),
            'desviacion_promedio': float(np.mean(desviaciones)),
            'min_promedio': float(np.mean(mins)),
            'max_promedio': float(np.mean(maxs)),
            'rango_promedio': float(np.mean(maxs) - np.mean(mins)),
            'teorico_independiente': 0.25
        }
        
        print(f"  Media: {resultados[alpha]['media_global']:.4f} ± {resultados[alpha]['media_std']:.4f}")
        print(f"  Rango: [{resultados[alpha]['min_promedio']:.4f}, {resultados[alpha]['max_promedio']:.4f}]")
        print(f"  Desviación típica: {resultados[alpha]['desviacion_promedio']:.4f}")
    
    # Guardar resultados
    with open('analisis_distribuciones.json', 'w') as f:
        json.dump({str(k): v for k, v in resultados.items()}, f, indent=2)
    
    print(f"\n✅ Resultados guardados en: analisis_distribuciones.json")
    
    # Graficar
    graficar_resultados(resultados)
    
    return resultados

def graficar_resultados(resultados):
    """Genera gráficas de las distribuciones."""
    
    alphas = sorted(resultados.keys())
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Media vs α
    ax = axes[0, 0]
    medias = [resultados[a]['media_global'] for a in alphas]
    errores = [resultados[a]['media_std'] for a in alphas]
    
    ax.errorbar(alphas, medias, yerr=errores, fmt='o-', capsize=5, linewidth=2, markersize=8)
    ax.axhline(y=0.25, color='r', linestyle='--', label='Independencia (0.25)')
    ax.set_xscale('log')
    ax.set_xlabel('α')
    ax.set_ylabel('Media de W_ij')
    ax.set_title('Media de co-victoria vs α')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Rango vs α
    ax = axes[0, 1]
    rangos = [resultados[a]['rango_promedio'] for a in alphas]
    
    ax.plot(alphas, rangos, 's-', linewidth=2, markersize=8, color='green')
    ax.set_xscale('log')
    ax.set_xlabel('α')
    ax.set_ylabel('Rango (max - min)')
    ax.set_title('Ancho de la distribución vs α')
    ax.grid(True, alpha=0.3)
    
    # 3. Desviación vs α
    ax = axes[1, 0]
    desviaciones = [resultados[a]['desviacion_promedio'] for a in alphas]
    
    ax.plot(alphas, desviaciones, '^-', linewidth=2, markersize=8, color='purple')
    ax.set_xscale('log')
    ax.set_xlabel('α')
    ax.set_ylabel('Desviación típica')
    ax.set_title('Dispersión de W_ij vs α')
    ax.grid(True, alpha=0.3)
    
    # 4. Comparación con teórico
    ax = axes[1, 1]
    diferencia = [resultados[a]['media_global'] - 0.25 for a in alphas]
    colores = ['red' if d > 0 else 'blue' for d in diferencia]
    
    ax.bar(range(len(alphas)), diferencia, color=colores, tick_label=[f'{a:.2f}' for a in alphas])
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax.set_xlabel('α')
    ax.set_ylabel('Diferencia con 0.25')
    ax.set_title('Desviación de la independencia')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('analisis_distribuciones.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    resultados = analizar_distribuciones()
    print("\n✅ Análisis completado")
    print("   📊 Gráfica: analisis_distribuciones.png")
    print("   📁 Datos: analisis_distribuciones.json")