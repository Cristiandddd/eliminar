#!/usr/bin/env python3
"""
Análisis Avanzado del Minority Game - VERSIÓN OPTIMIZADA PARA GRANDES N
======================================================================
Calcula las 5 métricas sin almacenar matrices completas N×N cuando N es grande.
"""

import os
import json
import glob
import numpy as np
from collections import Counter
from datetime import datetime
from tqdm import tqdm
import argparse
import gc

# ============================================================
# CONFIGURACIÓN
# ============================================================

CONFIG = {
    "M": 9,
    "PRECISION": "float64",
    "BINS_HISTOGRAMA": 50,
    "UMBRAL_MUESTREO_MI": 1000,  # Si N > 1000, calcular MI solo sobre muestra
}

# ============================================================
# FUNCIONES BÁSICAS
# ============================================================

def cargar_datos_completos(archivo_path):
    """Carga TODOS los datos del archivo."""
    print(f"Cargando TODOS los datos de {archivo_path}...")
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
    """
    Detecta si los datos están en formato {0,1} o {-1,1}.
    """
    valores_unicos = np.unique(acciones)
    
    if -1 in valores_unicos:
        print(f"  Formato detectado: {{-1, +1}}")
        return 'binario_pm1'
    else:
        print(f"  Formato detectado: {{0, 1}}")
        return 'binario_01'


def convertir_a_01(acciones, formato):
    """Convierte acciones a formato {0,1} si es necesario."""
    if formato == 'binario_pm1':
        return ((acciones + 1) // 2).astype(np.int8)
    return acciones


def entropia(secuencia):
    """Calcula entropía de Shannon H(X) = -Σ p(x) log p(x)."""
    _, counts = np.unique(secuencia, return_counts=True)
    probs = counts / len(secuencia)
    return -np.sum(probs * np.log2(probs + 1e-12))


def entropia_conjunta(seq1, seq2):
    """Calcula entropía conjunta H(X,Y) de manera eficiente."""
    # Usar un solo array para los pares
    pares = np.vstack([seq1, seq2])
    # Convertir a entero único para faster unique
    pares_combined = pares[0] * (np.max(seq2) + 1) + pares[1]
    _, counts = np.unique(pares_combined, return_counts=True)
    probs = counts / len(seq1)
    return -np.sum(probs * np.log2(probs + 1e-12))


def informacion_mutua(seq1, seq2):
    """
    Calcula información mutua I(X;Y) = H(X) + H(Y) - H(X,Y).
    Versión optimizada.
    """
    H_X = entropia(seq1)
    H_Y = entropia(seq2)
    H_XY = entropia_conjunta(seq1, seq2)
    
    MI = H_X + H_Y - H_XY
    return float(MI)


# ============================================================
# FUNCIÓN 1: VOLATILIDAD σ²/N
# ============================================================

def calcular_volatilidad(secuencias):
    """
    Calcula volatilidad σ²/N del sistema.
    """
    N = len(secuencias)
    T = len(secuencias[0])
    
    print(f"\nCalculando volatilidad (N={N}, T={T})...")
    
    acciones = np.array(secuencias, dtype=np.float64)
    formato = detectar_formato_datos(acciones.astype(np.int8))
    
    # Asistencia por ronda
    asistencia = np.sum(acciones, axis=0)
    
    A_media = np.mean(asistencia)
    A_var = np.var(asistencia, ddof=1)
    sigma2_N = A_var / N
    
    if formato == 'binario_pm1':
        A_teorica = 0.0
        var_teorica = N
    else:
        A_teorica = N / 2.0
        var_teorica = N / 4.0
    
    eficiencia = 1.0 - (A_var / var_teorica) if var_teorica > 0 else 0
    
    resultados = {
        "N": N,
        "T": T,
        "sigma2_N": float(sigma2_N),
        "A_media": float(A_media),
        "A_var": float(A_var),
        "eficiencia": float(eficiencia),
    }
    
    print(f"  σ²/N: {sigma2_N:.6f}")
    print(f"  Eficiencia: {eficiencia:.4f}")
    
    return resultados


# ============================================================
# FUNCIÓN 2: PREDICTIBILIDAD H/N
# ============================================================

def calcular_predictibilidad(acciones, N, formato, M=9):
    """
    Calcula la predictibilidad H/N del sistema.
    Versión optimizada en memoria.
    """
    acciones_01 = convertir_a_01(acciones, formato)
    T = acciones.shape[1]
    
    # Determinar acción ganadora en cada ronda
    asistencia = np.sum(acciones_01, axis=0)
    umbral = N / 2.0
    
    accion_ganadora = np.zeros(T, dtype=np.int8)
    accion_ganadora[asistencia < umbral] = 1
    
    n_historiales = 2**M
    
    # Usar arrays dispersos (solo guardamos historiales que aparecen)
    suma_asistencia_por_historial = {}
    count_por_historial = {}
    
    for t in range(M, T):
        # Historial: últimas M acciones ganadoras
        historial_bits = accion_ganadora[t-M:t]
        # Convertir bits a entero
        idx = 0
        for bit in historial_bits:
            idx = (idx << 1) | bit
        
        suma_asistencia_por_historial[idx] = suma_asistencia_por_historial.get(idx, 0) + asistencia[t]
        count_por_historial[idx] = count_por_historial.get(idx, 0) + 1
    
    # Calcular H
    H = 0.0
    for idx in suma_asistencia_por_historial:
        media_condicional = suma_asistencia_por_historial[idx] / count_por_historial[idx]
        H += media_condicional ** 2
    
    H = H / n_historiales
    H_normalizada = H / N
    
    resultados = {
        "H": float(H),
        "H_N": float(H_normalizada),
        "n_historiales_observados": len(count_por_historial),
        "total_historiales_posibles": n_historiales,
        "fraccion_historiales_observados": len(count_por_historial) / n_historiales
    }
    
    print(f"  H/N: {H_normalizada:.6f}")
    print(f"  Historiales observados: {len(count_por_historial)}/{n_historiales}")
    
    return resultados


# ============================================================
# FUNCIÓN 3: INFORMACIÓN MUTUA ENTRE VICTORIAS (OPTIMIZADA)
# ============================================================

def calcular_mi_entre_victorias_optimizado(victorias, max_pares=10000):
    """
    Calcula estadísticas de información mutua entre victorias.
    Para N grande, evita crear matriz N×N completa.
    """
    N, T = victorias.shape
    
    print(f"\nCalculando MI entre victorias (N={N})...")
    
    if N * (N-1) // 2 <= max_pares or N <= 1000:
        # Calcular todos los pares
        print(f"  Calculando todos los pares ({N*(N-1)//2} pares)...")
        valores_mi = []
        
        for i in tqdm(range(N), desc="MI entre victorias"):
            for j in range(i+1, N):
                mi = informacion_mutua(victorias[i], victorias[j])
                valores_mi.append(mi)
        
        media = np.mean(valores_mi)
        std = np.std(valores_mi)
        minimo = np.min(valores_mi)
        maximo = np.max(valores_mi)
        
    else:
        # Calcular sobre muestra representativa
        print(f"  N grande: calculando sobre muestra de {max_pares} pares...")
        import random
        random.seed(42)
        
        valores_mi = []
        indices_agentes = list(range(N))
        
        for _ in tqdm(range(max_pares), desc="MI muestreada"):
            i, j = random.sample(indices_agentes, 2)
            mi = informacion_mutua(victorias[i], victorias[j])
            valores_mi.append(mi)
        
        media = np.mean(valores_mi)
        std = np.std(valores_mi)
        minimo = np.min(valores_mi)
        maximo = np.max(valores_mi)
        
        print(f"  Nota: MI calculada sobre muestra de {max_pares} pares")
    
    resultados = {
        "media": float(media),
        "std": float(std),
        "min": float(minimo),
        "max": float(maximo),
        "n_pares_calculados": len(valores_mi),
        "total_pares_posibles": N * (N-1) // 2,
    }
    
    print(f"  MI media: {media:.6f} ± {std:.6f}")
    
    return resultados


# ============================================================
# FUNCIÓN 4: INFORMACIÓN MUTUA ACCIÓN-ESTADO
# ============================================================

def calcular_mi_accion_estado_optimizado(acciones, accion_ganadora, M=9):
    """
    Calcula MI entre acciones de agentes y estado del sistema.
    Versión optimizada en memoria.
    """
    N, T = acciones.shape
    print(f"\nCalculando MI acción-estado para {N} agentes...")
    
    # Pre-calcular todos los estados (una sola vez)
    estados = np.zeros(T - M, dtype=np.int32)
    for t in range(M, T):
        historial_bits = accion_ganadora[t-M:t]
        idx = 0
        for bit in historial_bits:
            idx = (idx << 1) | bit
        estados[t-M] = idx
    
    mi_por_agente = []
    
    for i in tqdm(range(N), desc="MI acción-estado"):
        acciones_agente = acciones[i, M:]  # Acciones alineadas con estados
        
        # Calcular MI entre acciones del agente y estados
        # Usar versión optimizada para arrays pequeños
        mi = informacion_mutua(estados, acciones_agente)
        mi_por_agente.append(mi)
        
        # Liberar memoria cada 100 agentes
        if i % 100 == 0:
            gc.collect()
    
    resultados = {
        "media": float(np.mean(mi_por_agente)),
        "std": float(np.std(mi_por_agente)),
        "min": float(np.min(mi_por_agente)),
        "max": float(np.max(mi_por_agente)),
        "por_agente": [float(m) for m in mi_por_agente]
    }
    
    print(f"  MI acción-estado media: {resultados['media']:.6f} ± {resultados['std']:.6f}")
    
    return resultados


# ============================================================
# FUNCIÓN 5: ENTROPÍA DE TRANSFERENCIA (OPTIMIZADA)
# ============================================================

def calcular_transfer_entropy_optimizado(seq_source, seq_target, k=1, l=1):
    """
    Calcula entropía de transferencia T(Y→X) de manera eficiente.
    """
    n = len(seq_source)
    if n <= max(k, l) + 1:
        return 0.0
    
    # Crear arrays para estados pasados
    n_samples = n - max(k, l)
    x_futuro = np.zeros(n_samples, dtype=np.int8)
    x_pasado = np.zeros(n_samples, dtype=np.int32)
    y_pasado = np.zeros(n_samples, dtype=np.int32)
    
    idx = 0
    for t in range(max(k, l), n-1):
        x_futuro[idx] = seq_target[t+1]
        
        # Codificar x_pasado como entero
        xp = 0
        for s in range(k):
            xp = (xp << 1) | seq_target[t-s]
        x_pasado[idx] = xp
        
        # Codificar y_pasado como entero
        yp = 0
        for s in range(l):
            yp = (yp << 1) | seq_source[t-s]
        y_pasado[idx] = yp
        
        idx += 1
    
    # Calcular H(X' | X)
    # Usar unique con combinación de x_futuro y x_pasado
    x_comb = x_futuro * (np.max(x_pasado) + 1) + x_pasado
    _, counts_x = np.unique(x_comb, return_counts=True)
    probs_x = counts_x / n_samples
    
    # Calcular H(X' | X, Y)
    xy_comb = x_futuro * (np.max(x_pasado) + 1) * (np.max(y_pasado) + 1) + \
              x_pasado * (np.max(y_pasado) + 1) + y_pasado
    _, counts_xy = np.unique(xy_comb, return_counts=True)
    probs_xy = counts_xy / n_samples
    
    # Entropías
    H_cond_sin_y = -np.sum(probs_x * np.log2(probs_x + 1e-12))
    H_cond_con_y = -np.sum(probs_xy * np.log2(probs_xy + 1e-12))
    
    TE = H_cond_sin_y - H_cond_con_y
    return max(0, TE)


def calcular_entropia_transferencia_optimizado(victorias, acciones, max_pares=5000):
    """
    Calcula estadísticas de entropía de transferencia.
    Para N grande, evita matrices completas.
    """
    N, T = victorias.shape
    
    print(f"\nCalculando entropía de transferencia (N={N})...")
    
    if N * (N-1) <= max_pares or N <= 200:
        # Calcular todos los pares
        te_victorias_lista = []
        te_acciones_lista = []
        
        for i in tqdm(range(N), desc="TE entre agentes"):
            for j in range(N):
                if i == j:
                    continue
                te_v = calcular_transfer_entropy_optimizado(victorias[j], victorias[i])
                te_a = calcular_transfer_entropy_optimizado(acciones[j], acciones[i])
                te_victorias_lista.append(te_v)
                te_acciones_lista.append(te_a)
        
        stats_victorias = {
            "media": float(np.mean(te_victorias_lista)),
            "std": float(np.std(te_victorias_lista)),
            "min": float(np.min(te_victorias_lista)),
            "max": float(np.max(te_victorias_lista)),
        }
        
        stats_acciones = {
            "media": float(np.mean(te_acciones_lista)),
            "std": float(np.std(te_acciones_lista)),
            "min": float(np.min(te_acciones_lista)),
            "max": float(np.max(te_acciones_lista)),
        }
        
    else:
        # Calcular sobre muestra
        print(f"  N grande: calculando sobre muestra de {max_pares} pares...")
        import random
        random.seed(42)
        
        te_victorias_lista = []
        te_acciones_lista = []
        indices_agentes = list(range(N))
        
        for _ in tqdm(range(max_pares), desc="TE muestreada"):
            i, j = random.sample(indices_agentes, 2)
            te_v = calcular_transfer_entropy_optimizado(victorias[j], victorias[i])
            te_a = calcular_transfer_entropy_optimizado(acciones[j], acciones[i])
            te_victorias_lista.append(te_v)
            te_acciones_lista.append(te_a)
        
        stats_victorias = {
            "media": float(np.mean(te_victorias_lista)),
            "std": float(np.std(te_victorias_lista)),
            "min": float(np.min(te_victorias_lista)),
            "max": float(np.max(te_victorias_lista)),
            "nota": "Calculado sobre muestra"
        }
        
        stats_acciones = {
            "media": float(np.mean(te_acciones_lista)),
            "std": float(np.std(te_acciones_lista)),
            "min": float(np.min(te_acciones_lista)),
            "max": float(np.max(te_acciones_lista)),
            "nota": "Calculado sobre muestra"
        }
    
    print(f"  TE victorias media: {stats_victorias['media']:.6f} ± {stats_victorias['std']:.6f}")
    print(f"  TE acciones media: {stats_acciones['media']:.6f} ± {stats_acciones['std']:.6f}")
    
    return {
        "stats_victorias": stats_victorias,
        "stats_acciones": stats_acciones
    }


# ============================================================
# FUNCIÓN PRINCIPAL OPTIMIZADA
# ============================================================

def analisis_avanzado_optimizado(archivo_path, output_dir="resultados_avanzados"):
    """
    Versión optimizada del análisis avanzado.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Cargar datos
    secuencias, alpha = cargar_datos_completos(archivo_path)
    nombre_base = os.path.basename(archivo_path).replace('.json', '')
    
    N = len(secuencias)
    T = len(secuencias[0])
    
    print(f"\n{'='*70}")
    print(f"ANÁLISIS AVANZADO OPTIMIZADO")
    print(f"{'='*70}")
    print(f"Archivo: {archivo_path}")
    print(f"α = {alpha}, N = {N}, T = {T}")
    print(f"M = {CONFIG['M']}")
    print(f"{'='*70}\n")
    
    # Convertir a matriz NumPy
    acciones = np.array(secuencias, dtype=np.int8)
    formato = detectar_formato_datos(acciones)
    acciones_01 = convertir_a_01(acciones, formato)
    
    # Liberar memoria de secuencias originales
    del secuencias
    gc.collect()
    
    # 1. Volatilidad
    print("\n[1] VOLATILIDAD σ²/N")
    vol_stats = calcular_volatilidad([acciones_01[i] for i in range(N)])
    
    # 2. Predictibilidad
    print("\n[2] PREDICTIBILIDAD H/N")
    pred_stats = calcular_predictibilidad(acciones_01, N, 'binario_01', M=CONFIG['M'])
    
    # 3. Matriz de victorias
    print("\n[3] PREPARANDO MATRIZ DE VICTORIAS...")
    asistencia = np.sum(acciones_01, axis=0)
    umbral = N / 2.0
    
    victorias = np.zeros((N, T), dtype=np.int8)
    victorias[:, asistencia < umbral] = 1  # Vectorizado
    
    # 4. Información mutua entre victorias (optimizada)
    print("\n[4] INFORMACIÓN MUTUA ENTRE VICTORIAS")
    mi_victorias_stats = calcular_mi_entre_victorias_optimizado(victorias)
    
    # 5. Información mutua acción-estado
    print("\n[5] INFORMACIÓN MUTUA ACCIÓN-ESTADO")
    asistencia = np.sum(acciones_01, axis=0)
    accion_ganadora = np.zeros(T, dtype=np.int8)
    accion_ganadora[asistencia < umbral] = 1
    
    mi_accion_estado_stats = calcular_mi_accion_estado_optimizado(
        acciones_01, accion_ganadora, M=CONFIG['M']
    )
    
    # 6. Entropía de transferencia (optimizada)
    print("\n[6] ENTROPÍA DE TRANSFERENCIA")
    te_stats = calcular_entropia_transferencia_optimizado(victorias, acciones_01)
    
    # ============================================================
    # EXPORTAR RESULTADOS
    # ============================================================
    
    resultados = {
        "metadata": {
            "archivo": archivo_path,
            "alpha": alpha,
            "N": N,
            "T": T,
            "M": CONFIG["M"],
            "fecha": datetime.now().isoformat()
        },
        "volatilidad": vol_stats,
        "predictibilidad": pred_stats,
        "mi_victorias": mi_victorias_stats,
        "mi_accion_estado": mi_accion_estado_stats,
        "transferencia": te_stats
    }
    
    output_file = os.path.join(output_dir, f"analisis_avanzado_{nombre_base}.json")
    with open(output_file, 'w') as f:
        json.dump(resultados, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"✅ ANÁLISIS AVANZADO COMPLETADO")
    print(f"📁 Resultados guardados en: {output_file}")
    print(f"{'='*70}\n")
    
    return resultados


# ============================================================
# INTERFAZ DE LÍNEA DE COMANDOS
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Análisis Avanzado Optimizado del Minority Game')
    parser.add_argument('--file', type=str, required=True, help='Archivo JSON a procesar')
    parser.add_argument('--output', type=str, default='resultados_avanzados', 
                       help='Directorio de salida')
    parser.add_argument('--M', type=int, default=9, help='Memoria del juego (bits)')
    
    args = parser.parse_args()
    
    CONFIG["M"] = args.M
    
    analisis_avanzado_optimizado(args.file, args.output)
