#!/usr/bin/env python3
"""
Cálculo de predictibilidad H (corregido) del Minority Game
Basado en la definición de la literatura: H = (1/P) Σ_μ ⟨A|μ⟩²
"""

import os
import json
import glob
import numpy as np
from datetime import datetime
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt
from collections import defaultdict

def cargar_datos_completos(archivo_path):
    """Carga TODOS los datos del archivo."""
    print(f"Cargando datos de {archivo_path}...")
    with open(archivo_path, 'r') as f:
        datos = json.load(f)
    
    secuencias = []
    alpha = None
    
    for item in tqdm(datos, desc="Cargando agentes"):
        if isinstance(item, list) and len(item) == 2:
            sec, a = item
            secuencias.append(np.array(sec, dtype=np.int8))
            if alpha is None:
                alpha = float(a)
    
    return secuencias, alpha


def detectar_formato_datos(acciones):
    """Detecta si los datos están en formato {0,1} o {-1,1}."""
    valores_unicos = np.unique(acciones)
    if -1 in valores_unicos:
        return 'binario_pm1'
    return 'binario_01'


def convertir_a_01(acciones, formato):
    """Convierte acciones a formato {0,1} si es necesario."""
    if formato == 'binario_pm1':
        return ((acciones + 1) // 2).astype(np.int8)
    return acciones


def calcular_predictibilidad_corregida(archivo_path):
    """
    Calcula la predictibilidad H según la definición de la literatura:
    H = (1/2^M) Σ_μ ⟨A|μ⟩²
    donde ⟨A|μ⟩ es la asistencia promedio condicionada al historial μ.
    """
    # Cargar datos
    secuencias, alpha = cargar_datos_completos(archivo_path)
    
    N = len(secuencias)
    T = len(secuencias[0])
    M = 9  # Memoria del juego
    
    # Convertir a matriz
    acciones = np.array(secuencias, dtype=np.int8)
    formato = detectar_formato_datos(acciones)
    acciones_01 = convertir_a_01(acciones, formato)
    
    # Asistencia por ronda
    asistencia = np.sum(acciones_01, axis=0, dtype=np.float64)
    
    # Acción ganadora (minoritaria) por ronda
    umbral = N / 2.0
    accion_ganadora = (asistencia < umbral).astype(np.int8)
    
    P = 2**M  # Número de historiales posibles
    
    # Acumular sumas para cada historial
    suma_A = {}  # Σ A(t) para cada historial (SIN normalizar)
    count = {}   # Número de veces que aparece cada historial
    
    for t in range(M, T):
        # Construir índice del historial
        historial = accion_ganadora[t-M:t]
        idx = 0
        for bit in historial:
            idx = (idx << 1) | bit
        
        # Asistencia CRUDA en esta ronda (sin dividir por N)
        A_cruda = asistencia[t]
        
        suma_A[idx] = suma_A.get(idx, 0.0) + A_cruda
        count[idx] = count.get(idx, 0) + 1
    
    # Calcular H
    H = 0.0
    for idx in suma_A:
        media_condicional = suma_A[idx] / count[idx]  # ⟨A|μ⟩
        H += media_condicional ** 2
    
    H = H / P  # Normalizar por número de historiales posibles
    
    return {
        'alpha': alpha,
        'H': float(H),
        'N': N,
        'T': T,
        'n_historiales_observados': len(count),
        'total_historiales': P
    }


def procesar_todos_archivos(directorio="."):
    """
    Procesa todos los archivos *_transformado.json en el directorio.
    """
    archivos = glob.glob(os.path.join(directorio, "*_transformado.json"))
    
    if not archivos:
        print(f"❌ No se encontraron archivos *_transformado.json en {directorio}")
        return None
    
    print(f"📁 Encontrados {len(archivos)} archivos")
    
    # Agrupar resultados por alpha
    resultados_por_alpha = defaultdict(list)
    
    for archivo in tqdm(archivos, desc="Procesando archivos"):
        try:
            resultado = calcular_predictibilidad_corregida(archivo)
            alpha = resultado['alpha']
            resultados_por_alpha[alpha].append(resultado['H'])
            print(f"  {os.path.basename(archivo)}: α={alpha:.4f}, H={resultado['H']:.4f}")
        except Exception as e:
            print(f"  Error en {archivo}: {e}")
    
    return resultados_por_alpha


def calcular_estadisticas(resultados_por_alpha):
    """
    Calcula media y desviación estándar de H para cada alpha.
    """
    estadisticas = []
    
    for alpha, valores_h in resultados_por_alpha.items():
        estadisticas.append({
            'alpha': alpha,
            'H_media': float(np.mean(valores_h)),
            'H_std': float(np.std(valores_h)),
            'n_muestras': len(valores_h)
        })
    
    # Ordenar por alpha
    estadisticas.sort(key=lambda x: x['alpha'])
    
    return estadisticas


def graficar_H(estadisticas):
    """
    Genera gráfica de H promedio vs alpha con barras de error.
    """
    alphas = [e['alpha'] for e in estadisticas]
    h_medias = [e['H_media'] for e in estadisticas]
    h_stds = [e['H_std'] for e in estadisticas]
    
    plt.figure(figsize=(10, 6))
    plt.errorbar(alphas, h_medias, yerr=h_stds, fmt='o-', 
                 capsize=5, linewidth=2, markersize=8, 
                 color='blue', ecolor='gray', elinewidth=1)
    
    plt.xlabel('α')
    plt.ylabel('H')
    plt.title('Predictibilidad H del Minority Game')
    plt.xscale('log')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('predictibilidad_H_vs_alpha.png', dpi=300)
    plt.show()
    
    print("✅ Gráfica guardada: predictibilidad_H_vs_alpha.png")


def main():
    parser = argparse.ArgumentParser(description='Cálculo de predictibilidad H (corregido)')
    parser.add_argument('--dir', type=str, default='.', 
                       help='Directorio con archivos *_transformado.json')
    parser.add_argument('--single', type=str, default=None,
                       help='Procesar un solo archivo específico')
    
    args = parser.parse_args()
    
    if args.single:
        # Procesar un solo archivo
        resultado = calcular_predictibilidad_corregida(args.single)
        print("\n" + "="*50)
        print(f"Resultado para {os.path.basename(args.single)}")
        print("="*50)
        print(f"α = {resultado['alpha']:.4f}")
        print(f"H = {resultado['H']:.4f}")
        print(f"N = {resultado['N']}")
        print(f"T = {resultado['T']}")
        print(f"Historiales observados: {resultado['n_historiales_observados']}/{resultado['total_historiales']}")
        
        # Guardar resultado individual
        output_file = f"predictibilidad_{os.path.basename(args.single).replace('.json', '')}.json"
        with open(output_file, 'w') as f:
            json.dump(resultado, f, indent=2)
        print(f"\n✅ Resultado guardado en: {output_file}")
        
    else:
        # Procesar todos los archivos
        resultados = procesar_todos_archivos(args.dir)
        
        if not resultados:
            return
        
        estadisticas = calcular_estadisticas(resultados)
        
        # Guardar estadísticas
        with open('predictibilidad_estadisticas.json', 'w') as f:
            json.dump(estadisticas, f, indent=2)
        
        print("\n" + "="*50)
        print("ESTADÍSTICAS POR ALPHA")
        print("="*50)
        for e in estadisticas:
            print(f"α = {e['alpha']:.4f}: H = {e['H_media']:.4f} ± {e['H_std']:.4f} (n={e['n_muestras']})")
        
        # Graficar
        graficar_H(estadisticas)


if __name__ == "__main__":
    main()