#!/usr/bin/env python3
"""
Análisis Informacional del Minority Game - VERSIÓN CORREGIDA
=============================================================
Correcciones:
1. H/N: Normaliza ANTES de elevar al cuadrado
2. Victorias: Calcula correctamente si cada agente ganó (su acción = minoritaria)
3. MI entre acciones: Añadido
"""

import os
import json
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
    "MAX_PARES_MI": 10000,
    "MAX_PARES_TE": 5000,
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
# FUNCIÓN 1: VOLATILIDAD σ²/N (CORREGIDA)
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
# FUNCIÓN 2: PREDICTIBILIDAD H/N (CORREGIDA)
# ============================================================

def calcular_predictibilidad(acciones_01, N, M=9):
    """
    Calcula la predictibilidad H/N del sistema.
    
    H = (1/P) Σ_μ <A|μ>²
    
    donde <A|μ> es la asistencia promedio condicionada al historial μ,
    y P = 2^M es el número de historiales posibles.
    
    CORRECCIÓN: Normalizamos <A|μ> ANTES de elevar al cuadrado.
    
    H/N = (1/P) Σ_μ (<A|μ>/N)²
    
    Esto garantiza que H/N ∈ [0, 0.25] (el máximo es cuando todos
    los agentes eligen lo mismo para cada historial).
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
        # Centrar respecto a 0.5 (que es el valor esperado para aleatorio)
        media_centrada = media_norm - 0.5
        # Sumar el cuadrado
        H_N += media_centrada ** 2
    
    # Promediar sobre todos los historiales posibles (no solo los observados)
    H_N = H_N / P
    
    # Alternativamente, podemos calcular H sin centrar:
    # H_N_sin_centrar = sum((sum_A/count)**2 for sum_A, count in zip(...)) / P
    
    return {
        "H_N": float(H_N),
        "H_N_sin_centrar": float(sum((suma_A_normalizada[idx]/count[idx])**2 
                                      for idx in suma_A_normalizada) / P),
        "n_historiales_observados": len(count),
        "total_historiales_posibles": P,
        "fraccion_observados": len(count) / P,
    }


# ============================================================
# FUNCIÓN 3: CALCULAR VICTORIAS CORRECTAMENTE
# ============================================================

def calcular_victorias(acciones_01, N):
    """
    Calcula matriz de victorias: victoria[i,t] = 1 si agente i ganó en ronda t.
    
    Un agente gana si su acción coincide con la acción minoritaria.
    
    CORRECCIÓN: Antes se marcaba victoria=1 para TODOS cuando A < N/2.
    Ahora se marca victoria=1 solo para quienes eligieron la acción minoritaria.
    """
    T = acciones_01.shape[1]
    
    # Asistencia (número que eligió 1)
    asistencia = np.sum(acciones_01, axis=0)
    
    # Acción minoritaria por ronda: 1 si menos de la mitad eligió 1, else 0
    umbral = N / 2.0
    accion_minoritaria = (asistencia < umbral).astype(np.int8)
    
    # Victoria: agente i ganó si su acción == acción minoritaria
    # victorias[i,t] = 1 si acciones_01[i,t] == accion_minoritaria[t]
    victorias = (acciones_01 == accion_minoritaria).astype(np.int8)
    
    return victorias, accion_minoritaria


# ============================================================
# FUNCIÓN 4: INFORMACIÓN MUTUA ENTRE VICTORIAS (CORREGIDA)
# ============================================================

def calcular_mi_entre_victorias(victorias, max_pares=10000):
    """
    Calcula MI entre secuencias de victoria de pares de agentes.
    
    NOTA: La MI entre victorias debería ser pequeña si los agentes
    ganan de manera casi independiente. Valores cercanos a 1 indicarían
    que dos agentes siempre ganan/pierden juntos (comparten estrategia).
    """
    N, T = victorias.shape
    total_pares = N * (N - 1) // 2
    
    print(f"  Calculando MI entre victorias ({N} agentes, {total_pares} pares)...")
    
    if total_pares <= max_pares:
        # Calcular todos los pares
        valores_mi = []
        valores_nmi = []
        
        for i in tqdm(range(N), desc="MI victorias"):
            for j in range(i + 1, N):
                mi = informacion_mutua(victorias[i], victorias[j])
                nmi = informacion_mutua_normalizada(victorias[i], victorias[j])
                valores_mi.append(mi)
                valores_nmi.append(nmi)
        
        n_calculados = len(valores_mi)
        
    else:
        # Muestrear pares
        print(f"  N grande: muestreando {max_pares} pares...")
        import random
        random.seed(42)
        
        valores_mi = []
        valores_nmi = []
        
        for _ in tqdm(range(max_pares), desc="MI victorias (muestra)"):
            i, j = random.sample(range(N), 2)
            mi = informacion_mutua(victorias[i], victorias[j])
            nmi = informacion_mutua_normalizada(victorias[i], victorias[j])
            valores_mi.append(mi)
            valores_nmi.append(nmi)
        
        n_calculados = max_pares
    
    return {
        "MI_media": float(np.mean(valores_mi)),
        "MI_std": float(np.std(valores_mi)),
        "MI_min": float(np.min(valores_mi)),
        "MI_max": float(np.max(valores_mi)),
        "NMI_media": float(np.mean(valores_nmi)),
        "NMI_std": float(np.std(valores_nmi)),
        "n_pares_calculados": n_calculados,
        "total_pares": total_pares,
    }


# ============================================================
# FUNCIÓN 5: INFORMACIÓN MUTUA ENTRE ACCIONES (NUEVA)
# ============================================================

def calcular_mi_entre_acciones(acciones_01, max_pares=10000):
    """
    Calcula MI entre secuencias de acciones de pares de agentes.
    
    La MI entre acciones mide cuánta información da la acción de un
    agente sobre la acción de otro. En el MG:
    - Régimen saturado: Alta MI (agentes comparten estrategias)
    - Régimen diluido: Baja MI (acciones casi independientes)
    """
    N, T = acciones_01.shape
    total_pares = N * (N - 1) // 2
    
    print(f"  Calculando MI entre acciones ({N} agentes, {total_pares} pares)...")
    
    if total_pares <= max_pares:
        valores_mi = []
        valores_nmi = []
        
        for i in tqdm(range(N), desc="MI acciones"):
            for j in range(i + 1, N):
                mi = informacion_mutua(acciones_01[i], acciones_01[j])
                nmi = informacion_mutua_normalizada(acciones_01[i], acciones_01[j])
                valores_mi.append(mi)
                valores_nmi.append(nmi)
        
        n_calculados = len(valores_mi)
        
    else:
        print(f"  N grande: muestreando {max_pares} pares...")
        import random
        random.seed(42)
        
        valores_mi = []
        valores_nmi = []
        
        for _ in tqdm(range(max_pares), desc="MI acciones (muestra)"):
            i, j = random.sample(range(N), 2)
            mi = informacion_mutua(acciones_01[i], acciones_01[j])
            nmi = informacion_mutua_normalizada(acciones_01[i], acciones_01[j])
            valores_mi.append(mi)
            valores_nmi.append(nmi)
        
        n_calculados = max_pares
    
    return {
        "MI_media": float(np.mean(valores_mi)),
        "MI_std": float(np.std(valores_mi)),
        "MI_min": float(np.min(valores_mi)),
        "MI_max": float(np.max(valores_mi)),
        "NMI_media": float(np.mean(valores_nmi)),
        "NMI_std": float(np.std(valores_nmi)),
        "n_pares_calculados": n_calculados,
        "total_pares": total_pares,
    }


# ============================================================
# FUNCIÓN 6: MI ACCIÓN-ESTADO
# ============================================================

def calcular_mi_accion_estado(acciones_01, accion_minoritaria, M=9):
    """
    Calcula MI entre la acción del agente en t+1 y el estado (historial) en t.
    
    Esto mide cuánta información tiene el historial sobre la próxima acción
    del agente.
    """
    N, T = acciones_01.shape
    
    print(f"  Calculando MI acción-estado para {N} agentes...")
    
    # Pre-calcular estados (historiales)
    estados = np.zeros(T - M, dtype=np.int32)
    for t in range(M, T):
        idx = 0
        for s in range(M):
            idx = (idx << 1) | accion_minoritaria[t - M + s]
        estados[t - M] = idx
    
    mi_por_agente = []
    
    for i in tqdm(range(N), desc="MI acción-estado"):
        # Acción del agente en t+1 (alineada con estados en t)
        acciones_futuras = acciones_01[i, M:]
        
        # Calcular MI
        mi = informacion_mutua(estados, acciones_futuras)
        mi_por_agente.append(mi)
    
    return {
        "MI_media": float(np.mean(mi_por_agente)),
        "MI_std": float(np.std(mi_por_agente)),
        "MI_min": float(np.min(mi_por_agente)),
        "MI_max": float(np.max(mi_por_agente)),
    }


# ============================================================
# FUNCIÓN 7: ENTROPÍA DE TRANSFERENCIA (OPTIMIZADA)
# ============================================================

def transfer_entropy(source, target, k=1, l=1):
    """
    Entropía de transferencia T(Y→X) = H(X' | X^k) - H(X' | X^k, Y^l).
    
    Mide la información que Y aporta sobre el futuro de X, más allá
    de lo que X ya sabe sobre sí mismo.
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


def calcular_transfer_entropy(victorias, acciones_01, max_pares=5000):
    """
    Calcula estadísticas de entropía de transferencia entre agentes.
    """
    N = victorias.shape[0]
    total_pares = N * (N - 1)
    
    print(f"  Calculando TE ({N} agentes, {total_pares} pares dirigidos)...")
    
    if total_pares <= max_pares:
        te_victorias = []
        te_acciones = []
        
        for i in tqdm(range(N), desc="TE"):
            for j in range(N):
                if i != j:
                    te_v = transfer_entropy(victorias[j], victorias[i])
                    te_a = transfer_entropy(acciones_01[j], acciones_01[i])
                    te_victorias.append(te_v)
                    te_acciones.append(te_a)
        
    else:
        print(f"  N grande: muestreando {max_pares} pares...")
        import random
        random.seed(42)
        
        te_victorias = []
        te_acciones = []
        
        for _ in tqdm(range(max_pares), desc="TE (muestra)"):
            i, j = random.sample(range(N), 2)
            te_v = transfer_entropy(victorias[j], victorias[i])
            te_a = transfer_entropy(acciones_01[j], acciones_01[i])
            te_victorias.append(te_v)
            te_acciones.append(te_a)
    
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
    Análisis informacional completo y corregido.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Cargar datos
    secuencias, alpha = cargar_datos_completos(archivo_path)
    nombre_base = os.path.basename(archivo_path).replace('.json', '')
    
    N = len(secuencias)
    T = len(secuencias[0])
    M = CONFIG['M']
    
    print(f"\n{'='*70}")
    print(f"ANÁLISIS INFORMACIONAL CORREGIDO")
    print(f"{'='*70}")
    print(f"Archivo: {archivo_path}")
    print(f"α = {alpha}, N = {N}, T = {T}, M = {M}")
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
    
    # 2. Predictibilidad (CORREGIDA)
    print("\n[2] PREDICTIBILIDAD H/N")
    pred = calcular_predictibilidad(acciones_01, N, M)
    print(f"    H/N = {pred['H_N']:.6f}")
    print(f"    Historiales observados: {pred['n_historiales_observados']}/{pred['total_historiales_posibles']}")
    
    # 3. Calcular victorias (CORREGIDO)
    print("\n[3] CALCULANDO VICTORIAS...")
    victorias, accion_minoritaria = calcular_victorias(acciones_01, N)
    
    # Verificar distribución de victorias
    win_rates = np.mean(victorias, axis=1)
    print(f"    Win rate medio: {np.mean(win_rates):.4f} ± {np.std(win_rates):.4f}")
    
    # 4. MI entre victorias (CORREGIDA)
    print("\n[4] MI ENTRE VICTORIAS")
    mi_victorias = calcular_mi_entre_victorias(victorias, CONFIG['MAX_PARES_MI'])
    print(f"    MI media = {mi_victorias['MI_media']:.6f} ± {mi_victorias['MI_std']:.6f}")
    print(f"    NMI media = {mi_victorias['NMI_media']:.6f}")
    
    # 5. MI entre acciones (NUEVA)
    print("\n[5] MI ENTRE ACCIONES")
    mi_acciones = calcular_mi_entre_acciones(acciones_01, CONFIG['MAX_PARES_MI'])
    print(f"    MI media = {mi_acciones['MI_media']:.6f} ± {mi_acciones['MI_std']:.6f}")
    print(f"    NMI media = {mi_acciones['NMI_media']:.6f}")
    
    # 6. MI acción-estado
    print("\n[6] MI ACCIÓN-ESTADO")
    mi_estado = calcular_mi_accion_estado(acciones_01, accion_minoritaria, M)
    print(f"    MI media = {mi_estado['MI_media']:.6f} ± {mi_estado['MI_std']:.6f}")
    
    # 7. Entropía de transferencia
    print("\n[7] ENTROPÍA DE TRANSFERENCIA")
    te = calcular_transfer_entropy(victorias, acciones_01, CONFIG['MAX_PARES_TE'])
    print(f"    TE victorias media = {te['TE_victorias']['media']:.6f}")
    print(f"    TE acciones media = {te['TE_acciones']['media']:.6f}")
    
    # Guardar resultados
    resultados = {
        "metadata": {
            "archivo": archivo_path,
            "alpha": alpha,
            "N": N,
            "T": T,
            "M": M,
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
    parser = argparse.ArgumentParser(description='Análisis Informacional del MG (Corregido)')
    parser.add_argument('--file', type=str, required=True, help='Archivo JSON')
    parser.add_argument('--output', type=str, default='resultados', help='Directorio de salida')
    parser.add_argument('--M', type=int, default=9, help='Memoria del juego')
    
    args = parser.parse_args()
    CONFIG['M'] = args.M
    
    analisis_informacional(args.file, args.output)
