#!/usr/bin/env python3
"""
Análisis Informacional del Minority Game - VERSIÓN COMPLETA PARALELIZADA
========================================================================
Calcula TODOS los pares sin límites de muestreo con paralelización en TODOS los cálculos.
"""

import os
import json
import numpy as np
from collections import Counter
from datetime import datetime
from tqdm import tqdm
import argparse
import gc
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import time

# ============================================================
# CONFIGURACIÓN
# ============================================================

CONFIG = {
    "M": 9,
    "PRECISION": "float64",
    "N_WORKERS": 100,
}

# ============================================================
# FUNCIONES DE CARGA
# ============================================================

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
    
    print(f"Cargados {len(secuencias)} agentes, alpha={alpha}")
    print(f"Longitud de secuencias: {len(secuencias[0]) if secuencias else 0}")
    
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


# ============================================================
# FUNCIONES DE ENTROPÍA E INFORMACIÓN MUTUA
# ============================================================

def entropia(secuencia):
    """
    Calcula entropía de Shannon H(X) = -Σ p(x) log₂ p(x).
    Para secuencias binarias, el máximo es 1 bit.
    """
    valores, counts = np.unique(secuencia, return_counts=True)
    probs = counts / len(secuencia)
    # Evitar log(0)
    probs = probs[probs > 0]
    return -np.sum(probs * np.log2(probs))


def entropia_conjunta(seq1, seq2):
    """
    Calcula entropía conjunta H(X,Y).
    Para dos secuencias binarias, el máximo es 2 bits.
    """
    # Combinar en pares únicos
    n = len(seq1)
    pares = seq1.astype(np.int32) * 2 + seq2.astype(np.int32)
    _, counts = np.unique(pares, return_counts=True)
    probs = counts / n
    probs = probs[probs > 0]
    return -np.sum(probs * np.log2(probs))


def informacion_mutua(seq1, seq2):
    """
    Calcula información mutua I(X;Y) = H(X) + H(Y) - H(X,Y).
    
    Para secuencias binarias:
    - Si son idénticas: I = H(X) = H(Y) ≤ 1 bit
    - Si son independientes: I = 0
    - Rango típico: [0, 1] bits
    """
    H_X = entropia(seq1)
    H_Y = entropia(seq2)
    H_XY = entropia_conjunta(seq1, seq2)
    
    MI = H_X + H_Y - H_XY
    # Por errores numéricos puede ser ligeramente negativo
    return max(0.0, float(MI))


def informacion_mutua_normalizada(seq1, seq2):
    """
    Información mutua normalizada: NMI = I(X;Y) / min(H(X), H(Y)).
    Rango: [0, 1], donde 1 = dependencia perfecta.
    """
    H_X = entropia(seq1)
    H_Y = entropia(seq2)
    
    if H_X == 0 or H_Y == 0:
        return 0.0
    
    MI = informacion_mutua(seq1, seq2)
    return MI / min(H_X, H_Y)


# ============================================================
# FUNCIÓN 1: VOLATILIDAD σ²/N (NO PARALELIZABLE - RÁPIDA)
# ============================================================

def calcular_volatilidad(acciones_01, N):
    """
    Calcula volatilidad σ²/N del sistema.
    
    En formato {0,1}:
    - A(t) = Σᵢ aᵢ(t) = número de agentes que eligieron 1
    - σ² = Var[A(t)]
    - Para agentes aleatorios: σ²/N = 1/4 (pero normalizamos a 1)
    
    En la literatura del MG se usa la normalización donde aleatorio = 1.
    """
    T = acciones_01.shape[1]
    
    # Asistencia por ronda (número de agentes que eligieron 1)
    asistencia = np.sum(acciones_01, axis=0, dtype=np.float64)
    
    # Centrar respecto a N/2 para obtener la fluctuación
    fluctuacion = asistencia - N / 2.0
    
    # Varianza de la fluctuación
    sigma2 = np.var(fluctuacion, ddof=1)
    
    # Normalizar: para aleatorio σ² = N/4, así que σ²/N = 1/4
    # Pero queremos que aleatorio = 1, así que multiplicamos por 4
    sigma2_N = sigma2 / (N / 4.0)  # Equivale a 4*σ²/N
    
    # Eficiencia: 1 - σ²/N (si es < 0, el sistema es peor que aleatorio)
    eficiencia = 1.0 - sigma2_N if sigma2_N <= 1 else -(sigma2_N - 1)
    
    return {
        "sigma2_N": float(sigma2_N),
        "sigma2_raw": float(sigma2),
        "A_media": float(np.mean(asistencia)),
        "A_std": float(np.std(asistencia)),
        "eficiencia": float(eficiencia),
        "N": N,
        "T": T,
    }


# ============================================================
# FUNCIÓN 2: PREDICTIBILIDAD H/N (NO PARALELIZABLE - RÁPIDA)
# ============================================================

def calcular_predictibilidad(acciones_01, N, M=9):
    """
    Calcula la predictibilidad H/N del sistema.
    
    H = (1/P) Σ_μ <A|μ>²
    
    donde <A|μ> es la asistencia promedio condicionada al historial μ,
    y P = 2^M es el número de historiales posibles.
    """
    T = acciones_01.shape[1]
    
    # Asistencia por ronda
    asistencia = np.sum(acciones_01, axis=0, dtype=np.float64)
    
    # Acción ganadora (minoritaria) por ronda
    umbral = N / 2.0
    accion_ganadora = (asistencia < umbral).astype(np.int8)
    
    P = 2**M  # Número de historiales posibles
    
    # Acumular sumas para cada historial
    suma_A_normalizada = {}  # Σ (A(t)/N) para cada historial
    count = {}  # Número de veces que aparece cada historial
    
    for t in range(M, T):
        # Construir índice del historial
        historial = accion_ganadora[t-M:t]
        idx = 0
        for bit in historial:
            idx = (idx << 1) | bit
        
        # Asistencia normalizada en esta ronda
        A_norm = asistencia[t] / N
        
        suma_A_normalizada[idx] = suma_A_normalizada.get(idx, 0.0) + A_norm
        count[idx] = count.get(idx, 0) + 1
    
    # Calcular H/N
    H_N = 0.0
    for idx in suma_A_normalizada:
        # Media condicional normalizada: <A/N | μ>
        media_norm = suma_A_normalizada[idx] / count[idx]
        # Sumar el cuadrado
        H_N += media_norm ** 2
    
    # Promediar sobre todos los historiales posibles (no solo los observados)
    H_N = H_N / P
    
    return {
        "H_N": float(H_N),
        "n_historiales_observados": len(count),
        "total_historiales_posibles": P,
        "fraccion_observados": len(count) / P,
    }


# ============================================================
# FUNCIÓN 3: CALCULAR VICTORIAS (NO PARALELIZABLE - RÁPIDA)
# ============================================================

def calcular_victorias(acciones_01, N):
    """
    Calcula matriz de victorias: victoria[i,t] = 1 si agente i ganó en ronda t.
    
    Un agente gana si su acción coincide con la acción minoritaria.
    """
    T = acciones_01.shape[1]
    
    # Asistencia (número que eligió 1)
    asistencia = np.sum(acciones_01, axis=0)
    
    # Acción minoritaria por ronda: 1 si menos de la mitad eligió 1, else 0
    umbral = N / 2.0
    accion_minoritaria = (asistencia < umbral).astype(np.int8)
    
    # Victoria: agente i ganó si su acción == acción minoritaria
    victorias = (acciones_01 == accion_minoritaria).astype(np.int8)
    
    return victorias, accion_minoritaria


# ============================================================
# FUNCIÓN 4: INFORMACIÓN MUTUA ENTRE VICTORIAS (PARALELIZADA)
# ============================================================

def calcular_mi_par_victorias(args):
    """
    Calcula MI para un par de índices (i,j) con i < j.
    Args: (i, j, victoria_i, victoria_j)
    """
    i, j, victoria_i, victoria_j = args
    mi = informacion_mutua(victoria_i, victoria_j)
    nmi = informacion_mutua_normalizada(victoria_i, victoria_j)
    return {
        'i': i,
        'j': j,
        'mi': mi,
        'nmi': nmi
    }


def calcular_mi_entre_victorias_paralelo(victorias, n_workers=12):
    """
    Calcula MI entre secuencias de victoria de TODOS los pares de agentes en paralelo.
    """
    N, T = victorias.shape
    total_pares = N * (N - 1) // 2
    
    print(f"  Calculando MI entre victorias en paralelo ({N} agentes, {total_pares} pares, {n_workers} workers)...")
    
    # Preparar argumentos para cada par (i,j) con i < j
    args_list = []
    for i in range(N):
        for j in range(i + 1, N):
            args_list.append((i, j, victorias[i], victorias[j]))
    
    valores_mi = []
    valores_nmi = []
    
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = [executor.submit(calcular_mi_par_victorias, args) for args in args_list]
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="MI victorias paralelo"):
            try:
                res = future.result()
                valores_mi.append(res['mi'])
                valores_nmi.append(res['nmi'])
            except Exception as e:
                print(f"  Error en par: {e}")
    
    return {
        "MI_media": float(np.mean(valores_mi)),
        "MI_std": float(np.std(valores_mi)),
        "MI_min": float(np.min(valores_mi)),
        "MI_max": float(np.max(valores_mi)),
        "NMI_media": float(np.mean(valores_nmi)),
        "NMI_std": float(np.std(valores_nmi)),
        "n_pares_calculados": len(valores_mi),
        "total_pares": total_pares,
    }


# ============================================================
# FUNCIÓN 5: INFORMACIÓN MUTUA ENTRE ACCIONES (PARALELIZADA)
# ============================================================

def calcular_mi_par_acciones(args):
    """
    Calcula MI para un par de índices (i,j) con i < j.
    Args: (i, j, acciones_i, acciones_j)
    """
    i, j, acciones_i, acciones_j = args
    mi = informacion_mutua(acciones_i, acciones_j)
    nmi = informacion_mutua_normalizada(acciones_i, acciones_j)
    return {
        'i': i,
        'j': j,
        'mi': mi,
        'nmi': nmi
    }


def calcular_mi_entre_acciones_paralelo(acciones_01, n_workers=12):
    """
    Calcula MI entre secuencias de acciones de TODOS los pares de agentes en paralelo.
    """
    N, T = acciones_01.shape
    total_pares = N * (N - 1) // 2
    
    print(f"  Calculando MI entre acciones en paralelo ({N} agentes, {total_pares} pares, {n_workers} workers)...")
    
    # Preparar argumentos para cada par (i,j) con i < j
    args_list = []
    for i in range(N):
        for j in range(i + 1, N):
            args_list.append((i, j, acciones_01[i], acciones_01[j]))
    
    valores_mi = []
    valores_nmi = []
    
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = [executor.submit(calcular_mi_par_acciones, args) for args in args_list]
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="MI acciones paralelo"):
            try:
                res = future.result()
                valores_mi.append(res['mi'])
                valores_nmi.append(res['nmi'])
            except Exception as e:
                print(f"  Error en par: {e}")
    
    return {
        "MI_media": float(np.mean(valores_mi)),
        "MI_std": float(np.std(valores_mi)),
        "MI_min": float(np.min(valores_mi)),
        "MI_max": float(np.max(valores_mi)),
        "NMI_media": float(np.mean(valores_nmi)),
        "NMI_std": float(np.std(valores_nmi)),
        "n_pares_calculados": len(valores_mi),
        "total_pares": total_pares,
    }


# ============================================================
# FUNCIÓN 6: MI ACCIÓN-ESTADO (PARALELIZADA)
# ============================================================

def calcular_mi_accion_estado_par(args):
    """
    Calcula MI acción-estado para un agente i.
    Args: (i, estados, acciones_futuras_i)
    """
    i, estados, acciones_futuras_i = args
    mi = informacion_mutua(estados, acciones_futuras_i)
    return {'i': i, 'mi': mi}


def calcular_mi_accion_estado_paralelo(acciones_01, accion_minoritaria, M=9, n_workers=12):
    """
    Calcula MI entre la acción del agente en t+1 y el estado (historial) en t.
    Versión paralelizada.
    """
    N, T = acciones_01.shape
    
    print(f"  Calculando MI acción-estado en paralelo para {N} agentes con {n_workers} workers...")
    
    # Pre-calcular estados (historiales)
    estados = np.zeros(T - M, dtype=np.int32)
    for t in range(M, T):
        idx = 0
        for s in range(M):
            idx = (idx << 1) | accion_minoritaria[t - M + s]
        estados[t - M] = idx
    
    # Preparar argumentos para cada agente
    args_list = []
    for i in range(N):
        acciones_futuras = acciones_01[i, M:]
        args_list.append((i, estados, acciones_futuras))
    
    mi_por_agente = []
    
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = [executor.submit(calcular_mi_accion_estado_par, args) for args in args_list]
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="MI acción-estado paralelo"):
            try:
                res = future.result()
                mi_por_agente.append(res['mi'])
            except Exception as e:
                print(f"  Error en agente: {e}")
    
    return {
        "MI_media": float(np.mean(mi_por_agente)),
        "MI_std": float(np.std(mi_por_agente)),
        "MI_min": float(np.min(mi_por_agente)),
        "MI_max": float(np.max(mi_por_agente)),
    }


# ============================================================
# FUNCIÓN 7: ENTROPÍA DE TRANSFERENCIA (PARALELIZADA)
# ============================================================

def transfer_entropy_core(source, target, k=1, l=1):
    """
    Núcleo del cálculo de entropía de transferencia.
    Separado para poder llamarlo desde procesos paralelos.
    """
    n = len(source)
    offset = max(k, l)
    
    if n <= offset + 1:
        return 0.0
    
    # Construir arrays
    n_samples = n - offset - 1
    
    x_futuro = target[offset + 1:offset + 1 + n_samples]
    
    # Codificar pasados como enteros
    x_pasado = np.zeros(n_samples, dtype=np.int32)
    y_pasado = np.zeros(n_samples, dtype=np.int32)
    
    for i in range(n_samples):
        t = offset + i
        # x_pasado: últimos k valores de target
        xp = 0
        for s in range(k):
            xp = (xp << 1) | target[t - s]
        x_pasado[i] = xp
        
        # y_pasado: últimos l valores de source
        yp = 0
        for s in range(l):
            yp = (yp << 1) | source[t - s]
        y_pasado[i] = yp
    
    # H(X' | X^k) usando H(X', X^k) - H(X^k)
    H_xf_xp = entropia_conjunta(x_futuro, x_pasado)
    H_xp = entropia(x_pasado)
    H_cond_sin_y = H_xf_xp - H_xp
    
    # H(X' | X^k, Y^l) usando entropía conjunta de 3 variables
    # Codificar (X', X^k, Y^l) como un solo entero
    max_xp = 2**k
    max_yp = 2**l
    conjunto = x_futuro * (max_xp * max_yp) + x_pasado * max_yp + y_pasado
    _, counts = np.unique(conjunto, return_counts=True)
    probs = counts / n_samples
    H_conjunto = -np.sum(probs * np.log2(probs + 1e-12))
    
    # H(X^k, Y^l)
    xp_yp = x_pasado * max_yp + y_pasado
    _, counts_xy = np.unique(xp_yp, return_counts=True)
    probs_xy = counts_xy / n_samples
    H_xp_yp = -np.sum(probs_xy * np.log2(probs_xy + 1e-12))
    
    H_cond_con_y = H_conjunto - H_xp_yp
    
    TE = H_cond_sin_y - H_cond_con_y
    return max(0.0, float(TE))


def transfer_entropy_par(args):
    """
    Versión paralelizable de transfer_entropy.
    Args: (i, j, victorias_i, victorias_j, acciones_i, acciones_j)
    """
    i, j, victorias_i, victorias_j, acciones_i, acciones_j = args
    
    te_v = transfer_entropy_core(victorias_j, victorias_i)
    te_a = transfer_entropy_core(acciones_j, acciones_i)
    
    return {
        'i': i,
        'j': j,
        'te_v': te_v,
        'te_a': te_a
    }


def calcular_transfer_entropy_paralelo(victorias, acciones_01, n_workers=12):
    """
    Calcula estadísticas de entropía de transferencia entre TODOS los pares de agentes
    utilizando paralelización.
    """
    N = victorias.shape[0]
    total_pares = N * (N - 1)
    
    print(f"  Calculando TE en paralelo ({N} agentes, {total_pares} pares dirigidos, {n_workers} workers)...")
    
    # Preparar argumentos para cada par (i,j) con i != j
    args_list = []
    for i in range(N):
        for j in range(N):
            if i != j:
                args_list.append((
                    i, j,
                    victorias[i], victorias[j],
                    acciones_01[i], acciones_01[j]
                ))
    
    te_victorias = []
    te_acciones = []
    
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = [executor.submit(transfer_entropy_par, args) for args in args_list]
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="TE paralelo"):
            try:
                res = future.result()
                te_victorias.append(res['te_v'])
                te_acciones.append(res['te_a'])
            except Exception as e:
                print(f"  Error en par ({res['i']},{res['j']}): {e}")
    
    return {
        "TE_victorias": {
            "media": float(np.mean(te_victorias)),
            "std": float(np.std(te_victorias)),
            "min": float(np.min(te_victorias)),
            "max": float(np.max(te_victorias)),
        },
        "TE_acciones": {
            "media": float(np.mean(te_acciones)),
            "std": float(np.std(te_acciones)),
            "min": float(np.min(te_acciones)),
            "max": float(np.max(te_acciones)),
        },
    }


# ============================================================
# FUNCIÓN PRINCIPAL
# ============================================================

def analisis_informacional(archivo_path, output_dir="resultados"):
    """
    Análisis informacional completo (sin muestreo) con TODO paralelizado.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Cargar datos
    secuencias, alpha = cargar_datos_completos(archivo_path)
    nombre_base = os.path.basename(archivo_path).replace('.json', '')
    
    N = len(secuencias)
    T = len(secuencias[0])
    M = CONFIG['M']
    
    print(f"\n{'='*70}")
    print(f"ANÁLISIS INFORMACIONAL COMPLETO (TODO PARALELIZADO)")
    print(f"{'='*70}")
    print(f"Archivo: {archivo_path}")
    print(f"α = {alpha}, N = {N}, T = {T}, M = {M}")
    print(f"Workers: {CONFIG['N_WORKERS']}")
    print(f"{'='*70}\n")
    
    # Convertir a matriz
    acciones = np.array(secuencias, dtype=np.int8)
    formato = detectar_formato_datos(acciones)
    print(f"Formato detectado: {formato}")
    acciones_01 = convertir_a_01(acciones, formato)
    
    del secuencias
    gc.collect()
    
    # 1. Volatilidad (rápida, no paralelizar)
    print("\n[1] VOLATILIDAD σ²/N")
    vol = calcular_volatilidad(acciones_01, N)
    print(f"    σ²/N = {vol['sigma2_N']:.6f}")
    print(f"    Eficiencia = {vol['eficiencia']:.4f}")
    
    # 2. Predictibilidad (rápida, no paralelizar)
    print("\n[2] PREDICTIBILIDAD H/N")
    pred = calcular_predictibilidad(acciones_01, N, M)
    print(f"    H/N = {pred['H_N']:.6f}")
    print(f"    Historiales observados: {pred['n_historiales_observados']}/{pred['total_historiales_posibles']}")
    
    # 3. Calcular victorias (rápida, no paralelizar)
    print("\n[3] CALCULANDO VICTORIAS...")
    victorias, accion_minoritaria = calcular_victorias(acciones_01, N)
    
    # Verificar distribución de victorias
    win_rates = np.mean(victorias, axis=1)
    print(f"    Win rate medio: {np.mean(win_rates):.4f} ± {np.std(win_rates):.4f}")
    
    # 4. MI entre victorias (PARALELIZADA)
    print("\n[4] MI ENTRE VICTORIAS (PARALELIZADA)")
    start = time.time()
    mi_victorias = calcular_mi_entre_victorias_paralelo(victorias, CONFIG['N_WORKERS'])
    elapsed = time.time() - start
    print(f"    MI media = {mi_victorias['MI_media']:.6f} ± {mi_victorias['MI_std']:.6f}")
    print(f"    NMI media = {mi_victorias['NMI_media']:.6f}")
    print(f"    Tiempo: {elapsed:.2f} segundos")
    
    # 5. MI entre acciones (PARALELIZADA)
    print("\n[5] MI ENTRE ACCIONES (PARALELIZADA)")
    start = time.time()
    mi_acciones = calcular_mi_entre_acciones_paralelo(acciones_01, CONFIG['N_WORKERS'])
    elapsed = time.time() - start
    print(f"    MI media = {mi_acciones['MI_media']:.6f} ± {mi_acciones['MI_std']:.6f}")
    print(f"    NMI media = {mi_acciones['NMI_media']:.6f}")
    print(f"    Tiempo: {elapsed:.2f} segundos")
    
    # 6. MI acción-estado (PARALELIZADA)
    print("\n[6] MI ACCIÓN-ESTADO (PARALELIZADA)")
    start = time.time()
    mi_estado = calcular_mi_accion_estado_paralelo(acciones_01, accion_minoritaria, M, CONFIG['N_WORKERS'])
    elapsed = time.time() - start
    print(f"    MI media = {mi_estado['MI_media']:.6f} ± {mi_estado['MI_std']:.6f}")
    print(f"    Tiempo: {elapsed:.2f} segundos")
    
    # 7. Entropía de transferencia (PARALELIZADA)
    print("\n[7] ENTROPÍA DE TRANSFERENCIA (PARALELIZADA)")
    start = time.time()
    te = calcular_transfer_entropy_paralelo(victorias, acciones_01, CONFIG['N_WORKERS'])
    elapsed = time.time() - start
    print(f"    TE victorias media = {te['TE_victorias']['media']:.6f}")
    print(f"    TE acciones media = {te['TE_acciones']['media']:.6f}")
    print(f"    Tiempo: {elapsed:.2f} segundos")
    
    # Guardar resultados
    resultados = {
        "metadata": {
            "archivo": archivo_path,
            "alpha": alpha,
            "N": N,
            "T": T,
            "M": M,
            "workers": CONFIG['N_WORKERS'],
            "fecha": datetime.now().isoformat(),
        },
        "volatilidad": vol,
        "predictibilidad": pred,
        "mi_victorias": mi_victorias,
        "mi_acciones": mi_acciones,
        "mi_accion_estado": mi_estado,
        "transfer_entropy": te,
    }
    
    output_file = os.path.join(output_dir, f"analisis_{nombre_base}.json")
    with open(output_file, 'w') as f:
        json.dump(resultados, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"ANÁLISIS COMPLETADO")
    print(f"Resultados guardados en: {output_file}")
    print(f"{'='*70}\n")
    
    return resultados


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Análisis Informacional del MG (Completo, Todo Paralelo)')
    parser.add_argument('--file', type=str, required=True, help='Archivo JSON')
    parser.add_argument('--output', type=str, default='resultados', help='Directorio de salida')
    parser.add_argument('--M', type=int, default=9, help='Memoria del juego')
    parser.add_argument('--workers', type=int, default=12, help='Número de workers')
    
    args = parser.parse_args()
    CONFIG['M'] = args.M
    CONFIG['N_WORKERS'] = args.workers
    
    analisis_informacional(args.file, args.output)
