#!/usr/bin/env python3
"""
Calcula la predictibilidad H/N² exactamente como se define en el seminario de Kocevar (2010).

Definicion (Eq. 3.8 del seminario):
    H = (1/2^M) * sum_{mu=1}^{2^M} <A|mu>^2

En la Figura 3.6 del seminario, H está en el rango [0,1], lo que implica que grafican H/N².

Con acciones {-1, +1}:
- A(t) = sum_i a_i(t) donde a_i in {-1, +1}
- <A> = 0 (por simetria)
- A puede variar entre -N y +N
- H puede ser hasta N², por eso normalizamos por N² para obtener valores [0,1]
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
import glob
import re
import sys


def historial_a_indice(historial, M):
    """
    Convierte una secuencia de M bits a un indice entero [0, 2^M - 1].
    historial: lista/array de los ultimos M resultados ganadores (0 o 1)
    """
    idx = 0
    for i, bit in enumerate(historial):
        idx += int(bit) * (2 ** (M - 1 - i))
    return idx


def calcular_resultado_ganador(acciones_t, N, formato='01'):
    """
    Determina cual fue la accion ganadora (minoritaria) en el tiempo t.
    
    acciones_t: array de acciones de todos los agentes en tiempo t
    N: numero de agentes
    formato: '01' si acciones son {0,1}, 'pm1' si son {-1,+1}
    
    Retorna: 0 o 1 indicando la accion minoritaria (siempre en formato {0,1})
    """
    if formato == '01':
        suma = np.sum(acciones_t)
        # Si suma < N/2, la mayoria eligio 0, entonces 1 es minoritaria
        # Si suma > N/2, la mayoria eligio 1, entonces 0 es minoritaria
        return 1 if suma < N / 2 else 0
    else:  # pm1
        suma = np.sum(acciones_t)
        # Si suma < 0, la mayoria eligio -1, entonces +1 es minoritaria
        # Si suma > 0, la mayoria eligio +1, entonces -1 es minoritaria
        # Convertimos a 0/1: 0 para -1, 1 para +1
        return 1 if suma < 0 else 0


def calcular_asistencia_en_pm1(acciones_t, formato='01'):
    """
    Calcula la asistencia A(t) = sum_i a_i(t) con acciones en formato {-1, +1}.
    
    Si las acciones son {0,1}, las convierte automaticamente.
    """
    if formato == '01':
        # Convertir {0,1} -> {-1,+1}
        acciones_pm1 = 2 * acciones_t - 1
        return np.sum(acciones_pm1)
    else:
        return np.sum(acciones_t)


def calcular_predictibilidad(acciones, M=9, formato='01'):
    """
    Calcula H y H/N² segun la definicion del seminario.
    
    H = (1/2^M) * sum_{mu} <A|mu>^2
    
    donde <A|mu> es el promedio de A(t) (en formato {-1,+1}) condicionado a que el historial en t sea mu.
    El historial se construye con los resultados ganadores en formato {0,1}.
    
    Para comparar con la Figura 3.6 del seminario, usamos H_normalized = H / N²,
    que da valores en el rango [0,1].
    
    Parametros:
    -----------
    acciones : array (N, T)
        Matriz de acciones de N agentes durante T rondas
    M : int
        Longitud de memoria (brain size)
    formato : str
        '01' si acciones son {0,1}, 'pm1' si son {-1,+1}
    
    Retorna:
    --------
    H : float
        Predictibilidad cruda (puede ser del orden de N²)
    H_normalized : float
        Predictibilidad normalizada H/N² (tipicamente entre 0 y 1)
    """
    N, T = acciones.shape
    n_historiales = 2 ** M
    
    # Primero calculamos la secuencia de acciones ganadoras (para construir historiales)
    # y la secuencia de asistencias A(t) en formato {-1,+1}
    
    resultados_ganadores = np.zeros(T, dtype=int)  # Siempre en {0,1}
    asistencias_pm1 = np.zeros(T, dtype=float)    # En formato {-1,+1}
    
    for t in range(T):
        acciones_t = acciones[:, t]
        resultados_ganadores[t] = calcular_resultado_ganador(acciones_t, N, formato)
        asistencias_pm1[t] = calcular_asistencia_en_pm1(acciones_t, formato)
    
    # Ahora acumulamos <A|mu> para cada historial mu
    suma_A_por_historial = defaultdict(float)
    count_por_historial = defaultdict(int)
    
    # Empezamos desde t = M para tener M bits de historia
    for t in range(M, T):
        # El historial en tiempo t son los M resultados ganadores anteriores
        historial = resultados_ganadores[t-M:t]
        mu = historial_a_indice(historial, M)
        
        # Acumulamos la asistencia (en formato {-1,+1}) para este historial
        suma_A_por_historial[mu] += asistencias_pm1[t]
        count_por_historial[mu] += 1
    
    # Calculamos H = (1/2^M) * sum_{mu} <A|mu>^2
    H = 0.0
    for mu in range(n_historiales):
        if count_por_historial[mu] > 0:
            media_A_dado_mu = suma_A_por_historial[mu] / count_por_historial[mu]
            H += media_A_dado_mu ** 2
        # Si count = 0, ese historial nunca ocurrio, contribuye 0
    
    H = H / n_historiales  # Normalizar por 2^M
    H_normalized = H / N  
    
    return H, H_normalized


def cargar_acciones_de_json(filepath):
    """
    Carga las acciones de un archivo JSON transformado.
    Formato esperado: [ [acciones_agente1, alpha], [acciones_agente2, alpha], ... ]
    """
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    # Formato: lista de [secuencia, alpha]
    if isinstance(data, list) and len(data) > 0:
        # Extraer secuencias y alpha
        secuencias = []
        alpha = None
        for item in data:
            if isinstance(item, list) and len(item) == 2:
                sec, a = item
                secuencias.append(sec)
                if alpha is None:
                    alpha = float(a)
        
        if secuencias:
            acciones = np.array(secuencias, dtype=np.int8)
            N = acciones.shape[0]
            return acciones, alpha, N, 9  # M=9 por defecto
    
    return None, None, None, None


def extraer_alpha_de_nombre(nombre_archivo):
    """Extrae el valor de alpha del nombre del archivo."""
    # Buscar patron alpha_X.XXX o similar
    match = re.search(r'([0-9]+\.[0-9]+)', nombre_archivo)
    if match:
        return float(match.group(1))
    return None


def procesar_directorio(directorio, patron='*_transformado.json', M=9):
    """
    Procesa todos los archivos JSON en un directorio y calcula H/N² para cada uno.
    
    Retorna:
    --------
    resultados : dict
        {alpha: [lista de H/N² para ese alpha]}
    """
    resultados = defaultdict(list)
    archivos = glob.glob(str(Path(directorio) / patron))
    
    print(f"Encontrados {len(archivos)} archivos")
    
    for filepath in archivos:
        try:
            acciones, alpha, N, M_archivo = cargar_acciones_de_json(filepath)
            
            if acciones is None or alpha is None:
                # Intentar extraer alpha del nombre
                alpha = extraer_alpha_de_nombre(str(filepath))
                if alpha is None:
                    print(f"  Saltando {Path(filepath).name}: no se pudo determinar alpha")
                    continue
            
            # Usar M del archivo si esta disponible, sino usar el proporcionado
            M_usar = M_archivo if M_archivo else M
            
            H, H_normalized = calcular_predictibilidad(acciones, M=M_usar, formato='01')
            
            resultados[alpha].append(H_normalized)
            print(f"  {Path(filepath).name}: alpha={alpha:.4f}, N={N}, H/N²={H_normalized:.6f}")
            
        except Exception as e:
            print(f"  Error procesando {filepath}: {e}")
    
    return resultados


def graficar_HN2_vs_alpha(resultados, output_path='H_vs_alpha.png'):
    """
    Genera grafico de H/N² vs alpha con barras de error.
    """
    # Ordenar por alpha
    alphas = sorted(resultados.keys())
    
    medias = []
    stds = []
    
    for alpha in alphas:
        valores = resultados[alpha]
        medias.append(np.mean(valores))
        stds.append(np.std(valores) if len(valores) > 1 else 0)
    
    alphas = np.array(alphas)
    medias = np.array(medias)
    stds = np.array(stds)
    
    # Crear figura
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Graficar con barras de error
    ax.errorbar(alphas, medias, yerr=stds, fmt='o-', capsize=3, 
                color='blue', markersize=6, linewidth=1.5,
                label=r'$H/N$ (simulacion)')
    
    # Linea vertical en alpha_c
    ax.axvline(x=0.34, color='red', linestyle='--', linewidth=1.5, 
               label=r'$\alpha_c \approx 0.34$')
    
    # Regiones
    ax.axvspan(0, 0.34, alpha=0.1, color='lightblue', label='Fase simetrica')
    ax.axvspan(0.34, max(alphas)*1.1, alpha=0.1, color='lightcoral', label='Fase asimetrica')
    
    # Configurar escala logaritmica en x
    ax.set_xscale('log')
    
    # Labels y titulo
    ax.set_xlabel(r'$\alpha = 2^M / N$', fontsize=14)
    ax.set_ylabel(r'$H/N$', fontsize=14)
    ax.set_title(r'Predictibilidad $H/N$ vs $\alpha$ (comparacion con Figura 3.6)', fontsize=16)
    
    # Limites
    ax.set_xlim(min(alphas) * 0.8, max(alphas) * 1.2)
    ax.set_ylim(0, max(medias + stds) * 1.1 if len(medias) > 0 else 0.6)
    
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nGrafico guardado en: {output_path}")
    
    return alphas, medias, stds


def main():
    """
    Uso: python calcular_predictibilidad_HN2.py [directorio_jsons] [M]
    """
    if len(sys.argv) < 2:
        directorio = "."
    else:
        directorio = sys.argv[1]
    
    M = int(sys.argv[2]) if len(sys.argv) > 2 else 9
    
    print(f"Procesando directorio: {directorio}")
    print(f"Usando M = {M}")
    
    resultados = procesar_directorio(directorio, M=M)
    
    if len(resultados) == 0:
        print("No se encontraron resultados validos")
        return
    
    # Graficar
    alphas, medias, stds = graficar_HN2_vs_alpha(resultados)
    
    # Guardar resultados en JSON
    resultados_json = {}
    for alpha in resultados:
        resultados_json[str(alpha)] = {
            "valores": resultados[alpha],
            "media": float(np.mean(resultados[alpha])),
            "std": float(np.std(resultados[alpha])),
            "n": len(resultados[alpha])
        }
    
    with open('predictibilidad_HN2_resultados.json', 'w') as f:
        json.dump(resultados_json, f, indent=2)
    
    print("\nResultados guardados en: predictibilidad_HN2_resultados.json")
    
    # Imprimir resumen
    print("\n=== RESUMEN ===")
    print(f"{'alpha':>10} {'H/N² (media)':>15} {'std':>10} {'n_muestras':>12}")
    print("-" * 50)
    for i, alpha in enumerate(alphas):
        n = len(resultados[alpha])
        print(f"{alpha:>10.4f} {medias[i]:>15.6f} {stds[i]:>10.6f} {n:>12}")


if __name__ == '__main__':
    main()
