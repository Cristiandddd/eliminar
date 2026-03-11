#!/usr/bin/env python3
"""
Análisis Informacional del Minority Game - VERSIÓN OPTIMIZADA
==============================================================
- 8 workers
- Muestreo 50% si pares > 500,000
- TE calculada sobre pares no dirigidos (i < j)
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
import random

# ============================================================
# CONFIGURACIÓN
# ============================================================

CONFIG = {
    "M": 9,
    "PRECISION": "float64",
    "N_WORKERS": 8,
    "UMBRAL_MUESTREO": 200000,  # Si pares > esto, muestrear 20%
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
    probs = probs[probs > 0]
    return -np.sum(probs * np.log2(probs))


def entropia_conjunta(seq1, seq2):
    """
    Calcula entropía conjunta H(X,Y).
    Para dos secuencias binarias, el máximo es 2 bits.
    """
    n = len(seq1)
    pares = seq1.astype(np.int32) * 2 + seq2.astype(np.int32)
    _, counts = np.unique(pares, return_counts=True)
    probs = counts / n
    probs = probs[probs > 0]
    return -np.sum(probs * np.log2(probs))


def informacion_mutua(seq1, seq2):
    """
    Calcula información mutua I(X;Y) = H(X) + H(Y) - H(X,Y).
    """
    H_X = entropia(seq1)
    H_Y = entropia(seq2)
    H_XY = entropia_conjunta(seq1, seq2)
    
    MI = H_X + H_Y - H_XY
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
# FUNCIÓN 1: VOLATILIDAD σ²/N
# ============================================================

def calcular_volatilidad(acciones_01, N):
    """
    Calcula volatilidad σ²/N del sistema.
    """
    T = acciones_01.shape[1]
    
    # Asistencia por ronda (número de agentes que eligieron 1)
    asistencia = np.sum(acciones_01, axis=0, dtype=np.float64)
    
    # Centrar respecto a N/2 para obtener la fluctuación
    fluctuacion = asistencia - N / 2.0
    
    # Varianza de la fluctuación
    sigma2 = np.var(fluctuacion, ddof=1)
    
    # Normalizar: para aleatorio σ²/N = 1/4, pero queremos aleatorio = 1
    sigma2_N = sigma2 / (N / 4.0)  # Equivale a 4*σ²/N
    
    # Eficiencia: 1 - σ²/N
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
# FUNCIÓN 2: PREDICTIBILIDAD H/N
# ============================================================

def calcular_predictibilidad(acciones_01, N, M=9):
    """
    Calcula la predictibilidad H/N del sistema.
    """
    T = acciones_01.shape[1]
    
    # Asistencia por ronda
    asistencia = np.sum(acciones_01, axis=0, dtype=np.float64)
    
    # Acción ganadora (minoritaria) por ronda
    umbral = N / 2.0
    accion_ganadora = (asistencia < umbral).astype(np.int8)
    
    P = 2**M  # Número de historiales posibles
    
    # Acumular sumas para cada historial
    suma_A_normalizada = {}
    count = {}
    
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
        media_norm = suma_A_normalizada[idx] / count[idx]
        H_N += media_norm ** 2
    
    H_N = H_N / P
    
    return {
        "H_N": float(H_N),
        "n_historiales_observados": len(count),
        "total_historiales_posibles": P,
        "fraccion_observados": len(count) / P,
    }


# ============================================================
# FUNCIÓN 3: CALCULAR VICTORIAS
# ============================================================

def calcular_victorias(acciones_01, N):
    """
    Calcula matriz de victorias: victoria[i,t] = 1 si agente i ganó en ronda t.
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
# FUNCIÓN 4: MI ENTRE VICTORIAS (PARALELIZADA CON MUESTREO)
# ============================================================

def calcular_mi_par_victorias(args):
    """Calcula MI para un par de índices (i,j) con i < j."""
    i, j, victoria_i, victoria_j = args
    mi = informacion_mutua(victoria_i, victoria_j)
    nmi = informacion_mutua_normalizada(victoria_i, victoria_j)
    return {'mi': mi, 'nmi': nmi}


def calcular_mi_entre_victorias_paralelo(victorias, n_workers=8):
    """
    Calcula MI entre secuencias de victoria con muestreo automático.
    """
    N = victorias.shape[0]
    total_pares = N * (N - 1) // 2
    
    print(f"  MI victorias: {N} agentes, {total_pares} pares totales")
    
    # Decidir si muestrear
    if total_pares > CONFIG['UMBRAL_MUESTREO']:
        fraccion = 1
        n_muestra = int(total_pares * fraccion)
        print(f"  Muestreando {n_muestra} pares ({fraccion*100:.0f}%)...")
        
        # Generar pares aleatorios únicos (i < j)
        pares = set()
        while len(pares) < n_muestra:
            i = random.randint(0, N-1)
            j = random.randint(0, N-1)
            if i != j:
                pares.add((min(i,j), max(i,j)))
        pares = list(pares)
    else:
        print(f"  Calculando todos los pares...")
        # Generar todos los pares
        pares = [(i, j) for i in range(N) for j in range(i+1, N)]
    
    # Preparar argumentos
    args_list = [(i, j, victorias[i], victorias[j]) for i, j in pares]
    
    valores_mi = []
    valores_nmi = []
    
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = [executor.submit(calcular_mi_par_victorias, args) for args in args_list]
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="MI victorias"):
            try:
                res = future.result()
                valores_mi.append(res['mi'])
                valores_nmi.append(res['nmi'])
            except Exception as e:
                print(f"  Error: {e}")
    
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
# FUNCIÓN 5: MI ENTRE ACCIONES (PARALELIZADA CON MUESTREO)
# ============================================================

def calcular_mi_par_acciones(args):
    """Calcula MI para un par de índices (i,j) con i < j."""
    i, j, acciones_i, acciones_j = args
    mi = informacion_mutua(acciones_i, acciones_j)
    nmi = informacion_mutua_normalizada(acciones_i, acciones_j)
    return {'mi': mi, 'nmi': nmi}


def calcular_mi_entre_acciones_paralelo(acciones_01, n_workers=8):
    """
    Calcula MI entre secuencias de acciones con muestreo automático.
    """
    N = acciones_01.shape[0]
    total_pares = N * (N - 1) // 2
    
    print(f"  MI acciones: {N} agentes, {total_pares} pares totales")
    
    if total_pares > CONFIG['UMBRAL_MUESTREO']:
        fraccion = 1
        n_muestra = int(total_pares * fraccion)
        print(f"  Muestreando {n_muestra} pares ({fraccion*100:.0f}%)...")
        
        pares = set()
        while len(pares) < n_muestra:
            i = random.randint(0, N-1)
            j = random.randint(0, N-1)
            if i != j:
                pares.add((min(i,j), max(i,j)))
        pares = list(pares)
    else:
        print(f"  Calculando todos los pares...")
        pares = [(i, j) for i in range(N) for j in range(i+1, N)]
    
    args_list = [(i, j, acciones_01[i], acciones_01[j]) for i, j in pares]
    
    valores_mi = []
    valores_nmi = []
    
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = [executor.submit(calcular_mi_par_acciones, args) for args in args_list]
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="MI acciones"):
            try:
                res = future.result()
                valores_mi.append(res['mi'])
                valores_nmi.append(res['nmi'])
            except Exception as e:
                print(f"  Error: {e}")
    
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
# FUNCIÓN 6: MI ACCIÓN-ESTADO
# ============================================================

def calcular_mi_accion_estado(acciones_01, accion_minoritaria, M=9):
    """
    Calcula MI entre la acción del agente en t+1 y el estado (historial) en t.
    """
    N, T = acciones_01.shape
    
    print(f"  MI acción-estado para {N} agentes...")
    
    # Pre-calcular estados (historiales)
    estados = np.zeros(T - M, dtype=np.int32)
    for t in range(M, T):
        idx = 0
        for s in range(M):
            idx = (idx << 1) | accion_minoritaria[t - M + s]
        estados[t - M] = idx
    
    mi_por_agente = []
    
    for i in tqdm(range(N), desc="MI acción-estado"):
        acciones_futuras = acciones_01[i, M:]
        mi = informacion_mutua(estados, acciones_futuras)
        mi_por_agente.append(mi)
    
    return {
        "MI_media": float(np.mean(mi_por_agente)),
        "MI_std": float(np.std(mi_por_agente)),
        "MI_min": float(np.min(mi_por_agente)),
        "MI_max": float(np.max(mi_por_agente)),
    }


# ============================================================
# FUNCIÓN 7: ENTROPÍA DE TRANSFERENCIA (PARALELIZADA, i<j)
# ============================================================

def transfer_entropy_core(source, target, k=1, l=1):
    """
    Núcleo del cálculo de entropía de transferencia T(source→target).
    """
    n = len(source)
    offset = max(k, l)
    
    if n <= offset + 1:
        return 0.0
    
    n_samples = n - offset - 1
    x_futuro = target[offset + 1:offset + 1 + n_samples]
    
    # Codificar pasados como enteros
    x_pasado = np.zeros(n_samples, dtype=np.int32)
    y_pasado = np.zeros(n_samples, dtype=np.int32)
    
    for i in range(n_samples):
        t = offset + i
        xp = 0
        for s in range(k):
            xp = (xp << 1) | target[t - s]
        x_pasado[i] = xp
        
        yp = 0
        for s in range(l):
            yp = (yp << 1) | source[t - s]
        y_pasado[i] = yp
    
    # H(X' | X^k)
    H_xf_xp = entropia_conjunta(x_futuro, x_pasado)
    H_xp = entropia(x_pasado)
    H_cond_sin_y = H_xf_xp - H_xp
    
    # H(X' | X^k, Y^l)
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


def transfer_entropy_par_no_dirigido(args):
    """
    Calcula TE en ambas direcciones para un par no dirigido (i,j) con i < j.
    Retorna T(i→j) y T(j→i).
    """
    i, j, victorias_i, victorias_j, acciones_i, acciones_j = args
    
    te_ij_v = transfer_entropy_core(victorias_j, victorias_i)  # j→i
    te_ji_v = transfer_entropy_core(victorias_i, victorias_j)  # i→j
    
    te_ij_a = transfer_entropy_core(acciones_j, acciones_i)
    te_ji_a = transfer_entropy_core(acciones_i, acciones_j)
    
    return {
        'te_ij_v': te_ij_v,
        'te_ji_v': te_ji_v,
        'te_ij_a': te_ij_a,
        'te_ji_a': te_ji_a,
    }


def calcular_transfer_entropy_paralelo(victorias, acciones_01, n_workers=8):
    """
    Calcula TE para pares no dirigidos (i < j) con muestreo automático.
    """
    N = victorias.shape[0]
    total_pares = N * (N - 1) // 2  # Pares no dirigidos
    
    print(f"  TE: {N} agentes, {total_pares} pares no dirigidos totales")
    
    if total_pares > CONFIG['UMBRAL_MUESTREO']:
        fraccion = 0.2
        n_muestra = int(total_pares * fraccion)
        print(f"  Muestreando {n_muestra} pares ({fraccion*100:.0f}%)...")
        
        pares = set()
        while len(pares) < n_muestra:
            i = random.randint(0, N-1)
            j = random.randint(0, N-1)
            if i != j:
                pares.add((min(i,j), max(i,j)))
        pares = list(pares)
    else:
        print(f"  Calculando todos los pares...")
        pares = [(i, j) for i in range(N) for j in range(i+1, N)]
    
    # Preparar argumentos
    args_list = []
    for i, j in pares:
        args_list.append((
            i, j,
            victorias[i], victorias[j],
            acciones_01[i], acciones_01[j]
        ))
    
    te_ij_v = []
    te_ji_v = []
    te_ij_a = []
    te_ji_a = []
    
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = [executor.submit(transfer_entropy_par_no_dirigido, args) for args in args_list]
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="TE"):
            try:
                res = future.result()
                te_ij_v.append(res['te_ij_v'])
                te_ji_v.append(res['te_ji_v'])
                te_ij_a.append(res['te_ij_a'])
                te_ji_a.append(res['te_ji_a'])
            except Exception as e:
                print(f"  Error: {e}")
    
    # Combinar todas las TE (ambas direcciones)
    todas_te_v = te_ij_v + te_ji_v
    todas_te_a = te_ij_a + te_ji_a
    
    return {
        "TE_victorias": {
            "media": float(np.mean(todas_te_v)),
            "std": float(np.std(todas_te_v)),
            "min": float(np.min(todas_te_v)),
            "max": float(np.max(todas_te_v)),
            "n_pares_dirigidos": len(todas_te_v),
        },
        "TE_acciones": {
            "media": float(np.mean(todas_te_a)),
            "std": float(np.std(todas_te_a)),
            "min": float(np.min(todas_te_a)),
            "max": float(np.max(todas_te_a)),
            "n_pares_dirigidos": len(todas_te_a),
        },
        "metadata": {
            "pares_no_dirigidos_muestreados": len(pares),
            "total_pares_no_dirigidos": total_pares,
            "fraccion_muestreo": len(pares) / total_pares if total_pares > 0 else 0,
        }
    }


# ============================================================
# FUNCIÓN PRINCIPAL
# ============================================================

def analisis_informacional(archivo_path, output_dir="resultados"):
    """
    Análisis informacional completo con 8 workers y muestreo adaptativo.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Cargar datos
    secuencias, alpha = cargar_datos_completos(archivo_path)
    nombre_base = os.path.basename(archivo_path).replace('.json', '')
    
    N = len(secuencias)
    T = len(secuencias[0])
    M = CONFIG['M']
    
    print(f"\n{'='*70}")
    print(f"ANÁLISIS INFORMACIONAL - 8 WORKERS")
    print(f"{'='*70}")
    print(f"Archivo: {archivo_path}")
    print(f"α = {alpha}, N = {N}, T = {T}, M = {M}")
    print(f"Workers: {CONFIG['N_WORKERS']}")
    print(f"Umbral muestreo: {CONFIG['UMBRAL_MUESTREO']} pares")
    print(f"{'='*70}\n")
    
    # Convertir a matriz
    acciones = np.array(secuencias, dtype=np.int8)
    formato = detectar_formato_datos(acciones)
    print(f"Formato detectado: {formato}")
    acciones_01 = convertir_a_01(acciones, formato)
    
    del secuencias
    gc.collect()
    
    # 1. Volatilidad
    print("\n[1] VOLATILIDAD σ²/N")
    vol = calcular_volatilidad(acciones_01, N)
    print(f"    σ²/N = {vol['sigma2_N']:.6f}")
    print(f"    Eficiencia = {vol['eficiencia']:.4f}")
    
    # 2. Predictibilidad
    print("\n[2] PREDICTIBILIDAD H/N")
    pred = calcular_predictibilidad(acciones_01, N, M)
    print(f"    H/N = {pred['H_N']:.6f}")
    print(f"    Historiales observados: {pred['n_historiales_observados']}/{pred['total_historiales_posibles']}")
    
    # 3. Calcular victorias
    print("\n[3] CALCULANDO VICTORIAS...")
    victorias, accion_minoritaria = calcular_victorias(acciones_01, N)
    
    win_rates = np.mean(victorias, axis=1)
    print(f"    Win rate medio: {np.mean(win_rates):.4f} ± {np.std(win_rates):.4f}")
    
    # 4. MI entre victorias
    print("\n[4] MI ENTRE VICTORIAS")
    start = time.time()
    mi_victorias = calcular_mi_entre_victorias_paralelo(victorias, CONFIG['N_WORKERS'])
    elapsed = time.time() - start
    print(f"    MI media = {mi_victorias['MI_media']:.6f} ± {mi_victorias['MI_std']:.6f}")
    print(f"    NMI media = {mi_victorias['NMI_media']:.6f}")
    print(f"    Tiempo: {elapsed:.2f} s")
    
    # 5. MI entre acciones
    print("\n[5] MI ENTRE ACCIONES")
    start = time.time()
    mi_acciones = calcular_mi_entre_acciones_paralelo(acciones_01, CONFIG['N_WORKERS'])
    elapsed = time.time() - start
    print(f"    MI media = {mi_acciones['MI_media']:.6f} ± {mi_acciones['MI_std']:.6f}")
    print(f"    NMI media = {mi_acciones['NMI_media']:.6f}")
    print(f"    Tiempo: {elapsed:.2f} s")
    
    # 6. MI acción-estado
    print("\n[6] MI ACCIÓN-ESTADO")
    start = time.time()
    mi_estado = calcular_mi_accion_estado(acciones_01, accion_minoritaria, M)
    elapsed = time.time() - start
    print(f"    MI media = {mi_estado['MI_media']:.6f} ± {mi_estado['MI_std']:.6f}")
    print(f"    Tiempo: {elapsed:.2f} s")
    
    # 7. Entropía de transferencia
    print("\n[7] ENTROPÍA DE TRANSFERENCIA")
    start = time.time()
    te = calcular_transfer_entropy_paralelo(victorias, acciones_01, CONFIG['N_WORKERS'])
    elapsed = time.time() - start
    print(f"    TE victorias media = {te['TE_victorias']['media']:.6f}")
    print(f"    TE acciones media = {te['TE_acciones']['media']:.6f}")
    print(f"    Tiempo: {elapsed:.2f} s")
    
    # Guardar resultados
    resultados = {
        "metadata": {
            "archivo": archivo_path,
            "alpha": alpha,
            "N": N,
            "T": T,
            "M": M,
            "workers": CONFIG['N_WORKERS'],
            "umbral_muestreo": CONFIG['UMBRAL_MUESTREO'],
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
    parser = argparse.ArgumentParser(description='Análisis Informacional del MG (8 workers)')
    parser.add_argument('--file', type=str, required=True, help='Archivo JSON')
    parser.add_argument('--output', type=str, default='resultados', help='Directorio de salida')
    parser.add_argument('--M', type=int, default=9, help='Memoria del juego')
    parser.add_argument('--workers', type=int, default=8, help='Número de workers')
    parser.add_argument('--umbral', type=int, default=500000, help='Umbral para muestreo')
    
    args = parser.parse_args()
    CONFIG['M'] = args.M
    CONFIG['N_WORKERS'] = args.workers
    CONFIG['UMBRAL_MUESTREO'] = args.umbral
    
    analisis_informacional(args.file, args.output)
