#!/usr/bin/env python3
"""
Graficador de resultados del análisis informacional
Genera gráficas individuales con valores promedio y desviaciones estándar
Línea vertical discontinua en el α donde la volatilidad es mínima
"""

import json
import glob
import numpy as np
import matplotlib.pyplot as plt
import os
from collections import defaultdict

def cargar_todos_resultados(directorio="resultados_analisis_informacional"):
    """
    Carga todos los archivos JSON generados por analisis_informacional.py
    y los agrupa por valor de alpha.
    """
    archivos = glob.glob(os.path.join(directorio, "analisis_*.json"))
    
    if not archivos:
        print(f"❌ No se encontraron archivos en {directorio}")
        return None
    
    print(f"📁 Cargando {len(archivos)} archivos...")
    
    # Agrupar por alpha
    resultados_por_alpha = defaultdict(list)
    
    for archivo in archivos:
        with open(archivo, 'r') as f:
            data = json.load(f)
        
        alpha = data['metadata']['alpha']
        resultados_por_alpha[alpha].append(data)
    
    print(f"✅ {len(resultados_por_alpha)} valores de alpha encontrados")
    return resultados_por_alpha

def calcular_estadisticas(resultados_por_alpha):
    """
    Calcula medias y desviaciones estándar para cada métrica y cada alpha.
    """
    estadisticas = {}
    
    for alpha, realizaciones in resultados_por_alpha.items():
        n = len(realizaciones)
        
        # Inicializar acumuladores
        volatilidades = []
        eficiencias = []
        predictibilidades = []
        mi_victorias = []
        mi_acciones = []
        mi_estado = []
        
        for r in realizaciones:
            volatilidades.append(r['volatilidad']['sigma2_N'])
            eficiencias.append(r['volatilidad']['eficiencia'])
            predictibilidades.append(r['predictibilidad']['H_N'])
            mi_victorias.append(r['mi_victorias']['MI_media'])
            mi_acciones.append(r['mi_acciones']['MI_media'])
            mi_estado.append(r['mi_accion_estado']['MI_media'])
        
        estadisticas[alpha] = {
            'n': n,
            'volatilidad': {
                'media': float(np.mean(volatilidades)),
                'std': float(np.std(volatilidades))
            },
            'eficiencia': {
                'media': float(np.mean(eficiencias)),
                'std': float(np.std(eficiencias))
            },
            'predictibilidad': {
                'media': float(np.mean(predictibilidades)),
                'std': float(np.std(predictibilidades))
            },
            'mi_victorias': {
                'media': float(np.mean(mi_victorias)),
                'std': float(np.std(mi_victorias))
            },
            'mi_acciones': {
                'media': float(np.mean(mi_acciones)),
                'std': float(np.std(mi_acciones))
            },
            'mi_estado': {
                'media': float(np.mean(mi_estado)),
                'std': float(np.std(mi_estado))
            }
        }
    
    return estadisticas

def encontrar_alpha_min_volatilidad(estadisticas):
    """
    Encuentra el valor de alpha donde la volatilidad es mínima.
    """
    alphas = sorted(estadisticas.keys())
    volatilidades = [estadisticas[a]['volatilidad']['media'] for a in alphas]
    
    idx_min = np.argmin(volatilidades)
    alpha_min = alphas[idx_min]
    
    print(f"\n📊 α de mínima volatilidad: {alpha_min:.4f}")
    return alpha_min

def graficar_volatilidad(estadisticas, alpha_min):
    """Gráfica de volatilidad σ²/N vs α"""
    alphas = sorted(estadisticas.keys())
    valores = [estadisticas[a]['volatilidad']['media'] for a in alphas]
    errores = [estadisticas[a]['volatilidad']['std'] for a in alphas]
    
    plt.figure(figsize=(10, 6))
    plt.errorbar(alphas, valores, yerr=errores, fmt='o-', capsize=5, 
                 linewidth=2, markersize=8, color='red', ecolor='gray')
    plt.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Aleatorio (σ²/N=1)')
    plt.axvline(x=alpha_min, color='red', linestyle='--', linewidth=2, alpha=0.7, 
                label=f'α* = {alpha_min:.3f}')
    plt.xlabel('α')
    plt.ylabel('σ²/N')
    plt.title('Volatilidad del sistema')
    plt.xscale('log')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('volatilidad_vs_alpha.png', dpi=300)
    plt.show()
    print("✅ Gráfica guardada: volatilidad_vs_alpha.png")

def graficar_predictibilidad(estadisticas, alpha_min):
    """Gráfica de predictibilidad H/N vs α"""
    alphas = sorted(estadisticas.keys())
    valores = [estadisticas[a]['predictibilidad']['media'] for a in alphas]
    errores = [estadisticas[a]['predictibilidad']['std'] for a in alphas]
    
    plt.figure(figsize=(10, 6))
    plt.errorbar(alphas, valores, yerr=errores, fmt='s-', capsize=5,
                 linewidth=2, markersize=8, color='blue', ecolor='gray')
    plt.axhline(y=0.25, color='gray', linestyle='--', alpha=0.5, label='Límite teórico (0.25)')
    plt.axvline(x=alpha_min, color='red', linestyle='--', linewidth=2, alpha=0.7, 
                label=f'α* = {alpha_min:.3f}')
    plt.xlabel('α')
    plt.ylabel('H/N')
    plt.title('Predictibilidad del sistema')
    plt.xscale('log')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('predictibilidad_vs_alpha.png', dpi=300)
    plt.show()
    print("✅ Gráfica guardada: predictibilidad_vs_alpha.png")

def graficar_mi_victorias(estadisticas, alpha_min):
    """Gráfica de MI entre victorias vs α"""
    alphas = sorted(estadisticas.keys())
    valores = [estadisticas[a]['mi_victorias']['media'] for a in alphas]
    errores = [estadisticas[a]['mi_victorias']['std'] for a in alphas]
    
    plt.figure(figsize=(10, 6))
    plt.errorbar(alphas, valores, yerr=errores, fmt='^-', capsize=5,
                 linewidth=2, markersize=8, color='green', ecolor='gray')
    plt.axvline(x=alpha_min, color='red', linestyle='--', linewidth=2, alpha=0.7, 
                label=f'α* = {alpha_min:.3f}')
    plt.xlabel('α')
    plt.ylabel('MI (bits)')
    plt.title('Información mutua entre victorias de agentes')
    plt.xscale('log')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('mi_victorias_vs_alpha.png', dpi=300)
    plt.show()
    print("✅ Gráfica guardada: mi_victorias_vs_alpha.png")

def graficar_mi_acciones(estadisticas, alpha_min):
    """Gráfica de MI entre acciones vs α"""
    alphas = sorted(estadisticas.keys())
    valores = [estadisticas[a]['mi_acciones']['media'] for a in alphas]
    errores = [estadisticas[a]['mi_acciones']['std'] for a in alphas]
    
    plt.figure(figsize=(10, 6))
    plt.errorbar(alphas, valores, yerr=errores, fmt='D-', capsize=5,
                 linewidth=2, markersize=8, color='purple', ecolor='gray')
    plt.axvline(x=alpha_min, color='red', linestyle='--', linewidth=2, alpha=0.7, 
                label=f'α* = {alpha_min:.3f}')
    plt.xlabel('α')
    plt.ylabel('MI (bits)')
    plt.title('Información mutua entre acciones de agentes')
    plt.xscale('log')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('mi_acciones_vs_alpha.png', dpi=300)
    plt.show()
    print("✅ Gráfica guardada: mi_acciones_vs_alpha.png")

def graficar_mi_estado(estadisticas, alpha_min):
    """Gráfica de MI acción-estado vs α"""
    alphas = sorted(estadisticas.keys())
    valores = [estadisticas[a]['mi_estado']['media'] for a in alphas]
    errores = [estadisticas[a]['mi_estado']['std'] for a in alphas]
    
    plt.figure(figsize=(10, 6))
    plt.errorbar(alphas, valores, yerr=errores, fmt='v-', capsize=5,
                 linewidth=2, markersize=8, color='orange', ecolor='gray')
    plt.axvline(x=alpha_min, color='red', linestyle='--', linewidth=2, alpha=0.7, 
                label=f'α* = {alpha_min:.3f}')
    plt.xlabel('α')
    plt.ylabel('MI (bits)')
    plt.title('Información mutua acción-estado')
    plt.xscale('log')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('mi_estado_vs_alpha.png', dpi=300)
    plt.show()
    print("✅ Gráfica guardada: mi_estado_vs_alpha.png")

def graficar_todas_juntas(estadisticas, alpha_min):
    """Gráfica comparativa con todas las métricas"""
    alphas = sorted(estadisticas.keys())
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. Volatilidad
    ax = axes[0, 0]
    valores = [estadisticas[a]['volatilidad']['media'] for a in alphas]
    errores = [estadisticas[a]['volatilidad']['std'] for a in alphas]
    ax.errorbar(alphas, valores, yerr=errores, fmt='o-', capsize=3, color='red')
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=alpha_min, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax.set_xscale('log')
    ax.set_xlabel('α')
    ax.set_ylabel('σ²/N')
    ax.set_title('Volatilidad')
    ax.grid(True, alpha=0.3)
    
    # 2. Predictibilidad
    ax = axes[0, 1]
    valores = [estadisticas[a]['predictibilidad']['media'] for a in alphas]
    errores = [estadisticas[a]['predictibilidad']['std'] for a in alphas]
    ax.errorbar(alphas, valores, yerr=errores, fmt='s-', capsize=3, color='blue')
    ax.axhline(y=0.25, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=alpha_min, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax.set_xscale('log')
    ax.set_xlabel('α')
    ax.set_ylabel('H/N')
    ax.set_title('Predictibilidad')
    ax.grid(True, alpha=0.3)
    
    # 3. MI victorias
    ax = axes[0, 2]
    valores = [estadisticas[a]['mi_victorias']['media'] for a in alphas]
    errores = [estadisticas[a]['mi_victorias']['std'] for a in alphas]
    ax.errorbar(alphas, valores, yerr=errores, fmt='^-', capsize=3, color='green')
    ax.axvline(x=alpha_min, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax.set_xscale('log')
    ax.set_xlabel('α')
    ax.set_ylabel('MI (bits)')
    ax.set_title('MI entre victorias')
    ax.grid(True, alpha=0.3)
    
    # 4. MI acciones
    ax = axes[1, 0]
    valores = [estadisticas[a]['mi_acciones']['media'] for a in alphas]
    errores = [estadisticas[a]['mi_acciones']['std'] for a in alphas]
    ax.errorbar(alphas, valores, yerr=errores, fmt='D-', capsize=3, color='purple')
    ax.axvline(x=alpha_min, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax.set_xscale('log')
    ax.set_xlabel('α')
    ax.set_ylabel('MI (bits)')
    ax.set_title('MI entre acciones')
    ax.grid(True, alpha=0.3)
    
    # 5. MI acción-estado
    ax = axes[1, 1]
    valores = [estadisticas[a]['mi_estado']['media'] for a in alphas]
    errores = [estadisticas[a]['mi_estado']['std'] for a in alphas]
    ax.errorbar(alphas, valores, yerr=errores, fmt='v-', capsize=3, color='orange')
    ax.axvline(x=alpha_min, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax.set_xscale('log')
    ax.set_xlabel('α')
    ax.set_ylabel('MI (bits)')
    ax.set_title('MI acción-estado')
    ax.grid(True, alpha=0.3)
    
    # 6. Eficiencia
    ax = axes[1, 2]
    valores = [estadisticas[a]['eficiencia']['media'] for a in alphas]
    errores = [estadisticas[a]['eficiencia']['std'] for a in alphas]
    ax.errorbar(alphas, valores, yerr=errores, fmt='p-', capsize=3, color='brown')
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=alpha_min, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax.set_xscale('log')
    ax.set_xlabel('α')
    ax.set_ylabel('Eficiencia')
    ax.set_title('Eficiencia del mercado')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('todas_metricas.png', dpi=300)
    plt.show()
    print("✅ Gráfica guardada: todas_metricas.png")

def generar_tabla_latex(estadisticas):
    """Genera tabla en formato LaTeX con los resultados"""
    alphas = sorted(estadisticas.keys())
    
    print("\n" + "="*80)
    print("TABLA LaTeX PARA LA TESIS")
    print("="*80)
    print("\n\\begin{table}[h]")
    print("\\centering")
    print("\\caption{Resumen de métricas informacionales}")
    print("\\begin{tabular}{|c|c|c|c|c|c|}")
    print("\\hline")
    print("α & σ²/N & H/N & MI victorias & MI acciones & MI estado \\\\ \\hline")
    
    for a in alphas:
        v = estadisticas[a]
        print(f"{a:.4f} & "
              f"{v['volatilidad']['media']:.4f}±{v['volatilidad']['std']:.4f} & "
              f"{v['predictibilidad']['media']:.4f}±{v['predictibilidad']['std']:.4f} & "
              f"{v['mi_victorias']['media']:.4f}±{v['mi_victorias']['std']:.4f} & "
              f"{v['mi_acciones']['media']:.4f}±{v['mi_acciones']['std']:.4f} & "
              f"{v['mi_estado']['media']:.4f}±{v['mi_estado']['std']:.4f} \\\\ \\hline")
    
    print("\\end{tabular}")
    print("\\end{table}")

def main():
    """Función principal"""
    print("\n" + "="*70)
    print("GRÁFICADOR DE RESULTADOS INFORMACIONALES")
    print("="*70)
    
    # Cargar datos
    resultados = cargar_todos_resultados()
    if not resultados:
        return
    
    # Calcular estadísticas
    print("\n📊 Calculando estadísticas...")
    estadisticas = calcular_estadisticas(resultados)
    
    # Encontrar α de mínima volatilidad
    alpha_min = encontrar_alpha_min_volatilidad(estadisticas)
    
    # Guardar estadísticas
    with open('estadisticas_informacionales.json', 'w') as f:
        # Convertir keys a string para JSON
        stats_json = {str(k): v for k, v in estadisticas.items()}
        json.dump(stats_json, f, indent=2)
    print("✅ Estadísticas guardadas en: estadisticas_informacionales.json")
    
    # Generar gráficas individuales
    print("\n📈 Generando gráficas...")
    graficar_volatilidad(estadisticas, alpha_min)
    graficar_predictibilidad(estadisticas, alpha_min)
    graficar_mi_victorias(estadisticas, alpha_min)
    graficar_mi_acciones(estadisticas, alpha_min)
    graficar_mi_estado(estadisticas, alpha_min)
    graficar_todas_juntas(estadisticas, alpha_min)
    
    # Generar tabla LaTeX
    generar_tabla_latex(estadisticas)
    
    print("\n" + "="*70)
    print("✅ PROCESO COMPLETADO")
    print("📁 Archivos generados:")
    print("   - estadisticas_informacionales.json")
    print("   - volatilidad_vs_alpha.png")
    print("   - predictibilidad_vs_alpha.png")
    print("   - mi_victorias_vs_alpha.png")
    print("   - mi_acciones_vs_alpha.png")
    print("   - mi_estado_vs_alpha.png")
    print("   - todas_metricas.png")
    print("="*70)

if __name__ == "__main__":
    main()