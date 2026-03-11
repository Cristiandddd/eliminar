#!/usr/bin/env python3
"""
Análisis Avanzado del Minority Game - Información Mutua, Predictibilidad y Transferencia
======================================================================================
Calcula:
- sigma²/N (volatilidad) - ya implementado
- H/N (predictibilidad)
- Información mutua entre victorias de agentes
- Información mutua entre acciones de agentes y estado del sistema (M=9)
- Entropía de transferencia entre secuencias de acciones y de victorias
"""

import os
import json
import glob
import numpy as np
from collections import Counter
from datetime import datetime
from tqdm import tqdm
import argparse

# ============================================================
# CONFIGURACIÓN
# ============================================================

CONFIG = {
    "M": 9,  # Memoria del juego (bits de historial)
    "PRECISION": "float64",
    "BINS_HISTOGRAMA": 50,
}

# ============================================================
# FUNCIONES BÁSICAS (desde analisis_histograma.py)
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


# ============================================================
# FUNCIÓN 1: VOLATILIDAD σ²/N (desde analisis_histograma.py)
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
    
    # Para formato {-1, +1}, la asistencia teórica es 0
    # Para formato {0, 1}, la asistencia teórica es N/2
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

def calcular_predictibilidad(acciones, formato, M=9):
    """
    Calcula la predictibilidad H/N del sistema.
    
    H = 1/2^M * sum_{μ=1}^{2^M} <A|μ>^2
    donde <A|μ> es el promedio de asistencia condicionada al historial μ.
    
    El historial μ son las últimas M acciones ganadoras.
    """
    N, T = acciones.shape
    acciones_01 = convertir_a_01(acciones, formato)
    
    # Primero necesitamos determinar la acción ganadora en cada ronda
    asistencia = np.sum(acciones_01, axis=0)
    
    if formato == 'binario_pm1':
        # Para formato {-1,+1}, la acción ganadora es -1 si A>0, +1 si A<0
        accion_ganadora = np.zeros(T, dtype=np.int8)
        accion_ganadora[asistencia > 0] = -1
        accion_ganadora[asistencia < 0] = 1
        # Convertir a 0/1 para facilitar
        accion_ganadora_01 = ((accion_ganadora + 1) // 2).astype(np.int8)
    else:
        # Para formato {0,1}, gana 0 si A > N/2, gana 1 si A < N/2
        umbral = N / 2.0
        accion_ganadora_01 = np.zeros(T, dtype=np.int8)
        accion_ganadora_01[asistencia < umbral] = 1  # Minoría es 1
        # Nota: cuando asistencia == umbral, hay empate, dejamos 0
    
    # Construir historiales de longitud M
    # Cada historial es un entero de 0 a 2^M - 1
    n_historiales = 2**M
    
    # Acumuladores para cada historial: suma de asistencia y contador
    suma_asistencia_por_historial = np.zeros(n_historiales, dtype=np.float64)
    count_por_historial = np.zeros(n_historiales, dtype=np.int32)
    
    # Recorrer las rondas (necesitamos M rondas anteriores)
    for t in range(M, T):
        # Historial: últimas M acciones ganadoras (t-M a t-1)
        historial_bits = accion_ganadora_01[t-M:t]
        # Convertir bits a entero (bit más significativo es el más antiguo)
        idx = 0
        for bit in historial_bits:
            idx = (idx << 1) | bit
        
        suma_asistencia_por_historial[idx] += asistencia[t]
        count_por_historial[idx] += 1
    
    # Calcular <A|μ> para cada historial con al menos una ocurrencia
    H = 0.0
    n_historiales_observados = 0
    
    for idx in range(n_historiales):
        if count_por_historial[idx] > 0:
            media_condicional = suma_asistencia_por_historial[idx] / count_por_historial[idx]
            H += media_condicional ** 2
            n_historiales_observados += 1
    
    # Normalizar por el número total de historiales posibles
    H = H / n_historiales
    H_normalizada = H / N  # H/N
    
    resultados = {
        "H": float(H),
        "H_N": float(H_normalizada),
        "n_historiales_observados": int(n_historiales_observados),
        "total_historiales_posibles": n_historiales,
        "fraccion_historiales_observados": n_historiales_observados / n_historiales
    }
    
    print(f"  H/N: {H_normalizada:.6f}")
    print(f"  Historiales observados: {n_historiales_observados}/{n_historiales}")
    
    return resultados


# ============================================================
# FUNCIÓN 3: INFORMACIÓN MUTUA (implementación manual)
# ============================================================

def entropia(secuencia):
    """Calcula entropía de Shannon H(X) = -Σ p(x) log p(x)."""
    _, counts = np.unique(secuencia, return_counts=True)
    probs = counts / len(secuencia)
    return -np.sum(probs * np.log2(probs + 1e-12))


def entropia_conjunta(seq1, seq2):
    """Calcula entropía conjunta H(X,Y)."""
    pares = np.column_stack((seq1, seq2))
    pares_tuples = [tuple(p) for p in pares]
    _, counts = np.unique(pares_tuples, return_counts=True)
    probs = counts / len(seq1)
    return -np.sum(probs * np.log2(probs + 1e-12))


def informacion_mutua(seq1, seq2):
    """
    Calcula información mutua I(X;Y) = H(X) + H(Y) - H(X,Y).
    Implementación manual sin bibliotecas externas.
    """
    H_X = entropia(seq1)
    H_Y = entropia(seq2)
    H_XY = entropia_conjunta(seq1, seq2)
    
    MI = H_X + H_Y - H_XY
    return float(MI)


def calcular_mi_entre_victorias(victorias):
    """
    Calcula la matriz de información mutua entre las victorias de todos los agentes.
    
    victorias: matriz N×T con 1 si el agente ganó, 0 si perdió
    """
    N, T = victorias.shape
    print(f"\nCalculando matriz de información mutua entre victorias ({N}×{N})...")
    
    mi_matrix = np.zeros((N, N))
    valores_triu = []
    
    for i in tqdm(range(N), desc="MI entre victorias"):
        for j in range(i+1, N):
            mi = informacion_mutua(victorias[i], victorias[j])
            mi_matrix[i, j] = mi
            mi_matrix[j, i] = mi
            valores_triu.append(mi)
    
    # Estadísticas
    if valores_triu:
        media = np.mean(valores_triu)
        std = np.std(valores_triu)
        print(f"  MI media entre victorias: {media:.6f} ± {std:.6f}")
    else:
        media, std = 0, 0
    
    return mi_matrix, {
        "media": float(media),
        "std": float(std),
        "min": float(np.min(valores_triu)) if valores_triu else 0,
        "max": float(np.max(valores_triu)) if valores_triu else 0,
    }


def calcular_mi_accion_estado(acciones, accion_ganadora, M=9):
    """
    Calcula la información mutua entre las acciones de cada agente y el estado del sistema.
    
    El estado del sistema son las últimas M acciones ganadoras.
    Para cada agente i, calculamos I(acción_i(t+1); estado_t)
    donde estado_t = [ganadora(t-M+1), ..., ganadora(t)]
    """
    N, T = acciones.shape
    print(f"\nCalculando MI entre acciones y estado del sistema (M={M})...")
    
    # Convertir acciones a 0/1 si es necesario
    acciones_01 = acciones.copy()  # Asumimos que ya están en 0/1
    
    mi_por_agente = []
    
    for i in tqdm(range(N), desc="MI acción-estado"):
        mi_vals = []
        
        # Para cada tiempo t donde tenemos historial completo
        for t in range(M, T-1):  # t+1 debe existir
            # Estado: últimas M acciones ganadoras
            estado_bits = accion_ganadora[t-M:t]
            # Convertir a entero
            estado = 0
            for bit in estado_bits:
                estado = (estado << 1) | bit
            
            # Acción del agente en t+1
            accion = acciones_01[i, t+1]
            
            # Acumular para histograma conjunto
            # (simplificado: usaremos bins discretos)
            mi_vals.append((estado, accion))
        
        # Calcular MI para este agente usando histograma 2D
        if mi_vals:
            estados, acciones_t = zip(*mi_vals)
            mi = informacion_mutua(np.array(estados), np.array(acciones_t))
            mi_por_agente.append(mi)
        else:
            mi_por_agente.append(0.0)
    
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
# FUNCIÓN 4: ENTROPÍA DE TRANSFERENCIA
# ============================================================

def entropia_condicional(seq_target, seq_cond, k=1, l=1):
    """
    Calcula H(target_{t+1} | target_t^{(k)}, source_t^{(l)})
    Versión simplificada con histogramas.
    
    Para TE(Y→X): target = X, source = Y
    """
    n = len(seq_target)
    if n <= max(k, l) + 1:
        return 0.0
    
    # Construir tuplas (x_{t+1}, x_t^{(k)}, y_t^{(l)})
    triplas = []
    for t in range(max(k, l), n-1):
        x_futuro = seq_target[t+1]
        x_pasado = tuple(seq_target[t-k+1:t+1]) if k > 0 else ()
        y_pasado = tuple(seq_cond[t-l+1:t+1]) if l > 0 else ()
        triplas.append((x_futuro, x_pasado, y_pasado))
    
    # Calcular frecuencias
    from collections import Counter
    counter_triplas = Counter(triplas)
    total = len(triplas)
    
    # Calcular entropía condicional H(X' | X, Y)
    H_cond = 0.0
    for (x_fut, x_pas, y_pas), count in counter_triplas.items():
        # Probabilidad conjunta p(x', x, y)
        p_joint = count / total
        
        # Probabilidad condicional p(x' | x, y) = p(x', x, y) / p(x, y)
        # Necesitamos p(x, y)
        p_xy = 0
        for (xf2, xp2, yp2), c2 in counter_triplas.items():
            if xp2 == x_pas and yp2 == y_pas:
                p_xy += c2
        p_xy /= total
        
        if p_xy > 0:
            p_cond = p_joint / p_xy
            if p_cond > 0:
                H_cond -= p_joint * np.log2(p_cond)
    
    return H_cond


def entropia_condicional_sin_y(seq_target, k=1):
    """
    Calcula H(target_{t+1} | target_t^{(k)})
    """
    n = len(seq_target)
    if n <= k:
        return 0.0
    
    # Construir pares (x_{t+1}, x_t^{(k)})
    pares = []
    for t in range(k, n-1):
        x_futuro = seq_target[t+1]
        x_pasado = tuple(seq_target[t-k+1:t+1])
        pares.append((x_futuro, x_pasado))
    
    from collections import Counter
    counter_pares = Counter(pares)
    total = len(pares)
    
    H_cond = 0.0
    for (x_fut, x_pas), count in counter_pares.items():
        p_joint = count / total
        
        # p(x_pas)
        p_x_pas = 0
        for (xf2, xp2), c2 in counter_pares.items():
            if xp2 == x_pas:
                p_x_pas += c2
        p_x_pas /= total
        
        if p_x_pas > 0:
            p_cond = p_joint / p_x_pas
            if p_cond > 0:
                H_cond -= p_joint * np.log2(p_cond)
    
    return H_cond


def transfer_entropy(seq_source, seq_target, k=1, l=1):
    """
    Calcula entropía de transferencia T(source → target)
    
    T(Y→X) = H(X_{t+1} | X_t^{(k)}) - H(X_{t+1} | X_t^{(k)}, Y_t^{(l)})
    """
    H_cond_sin_y = entropia_condicional_sin_y(seq_target, k)
    H_cond_con_y = entropia_condicional(seq_target, seq_source, k, l)
    
    TE = H_cond_sin_y - H_cond_con_y
    return max(0, TE)  # TE debería ser no negativa


def calcular_entropia_transferencia(victorias, acciones, M=9):
    """
    Calcula matrices de entropía de transferencia entre agentes.
    
    Calculamos dos tipos:
    1. TE entre victorias: ¿la victoria de Y ayuda a predecir la victoria de X?
    2. TE entre acciones: ¿la acción de Y ayuda a predecir la acción de X?
    """
    N, T = victorias.shape
    print(f"\nCalculando matrices de entropía de transferencia ({N}×{N})...")
    
    # Matrices para TE
    te_victorias = np.zeros((N, N))
    te_acciones = np.zeros((N, N))
    
    # Para cada par (i,j) con i != j
    for i in tqdm(range(N), desc="TE entre agentes"):
        for j in range(N):
            if i == j:
                continue
            
            # TE de victorias: victorias[j] → victorias[i]
            te_v = transfer_entropy(victorias[j], victorias[i], k=1, l=1)
            te_victorias[i, j] = te_v
            
            # TE de acciones: acciones[j] → acciones[i]
            te_a = transfer_entropy(acciones[j], acciones[i], k=1, l=1)
            te_acciones[i, j] = te_a
    
    # Estadísticas
    stats_victorias = {
        "media": float(np.mean(te_victorias)),
        "std": float(np.std(te_victorias)),
        "min": float(np.min(te_victorias)),
        "max": float(np.max(te_victorias)),
    }
    
    stats_acciones = {
        "media": float(np.mean(te_acciones)),
        "std": float(np.std(te_acciones)),
        "min": float(np.min(te_acciones)),
        "max": float(np.max(te_acciones)),
    }
    
    print(f"  TE victorias media: {stats_victorias['media']:.6f} ± {stats_victorias['std']:.6f}")
    print(f"  TE acciones media: {stats_acciones['media']:.6f} ± {stats_acciones['std']:.6f}")
    
    return {
        "matriz_victorias": te_victorias.tolist(),
        "matriz_acciones": te_acciones.tolist(),
        "stats_victorias": stats_victorias,
        "stats_acciones": stats_acciones
    }


# ============================================================
# FUNCIÓN PRINCIPAL QUE INTEGRA TODO
# ============================================================

def analisis_avanzado(archivo_path, output_dir="resultados_avanzados"):
    """
    Realiza el análisis completo con las 5 métricas.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Cargar datos
    secuencias, alpha = cargar_datos_completos(archivo_path)
    nombre_base = os.path.basename(archivo_path).replace('.json', '')
    
    N = len(secuencias)
    T = len(secuencias[0])
    
    print(f"\n{'='*70}")
    print(f"ANÁLISIS AVANZADO DEL MINORITY GAME")
    print(f"{'='*70}")
    print(f"Archivo: {archivo_path}")
    print(f"α = {alpha}, N = {N}, T = {T}")
    print(f"M = {CONFIG['M']} (memoria)")
    print(f"{'='*70}\n")
    
    # Convertir a matriz NumPy
    acciones = np.array(secuencias, dtype=np.int8)
    formato = detectar_formato_datos(acciones)
    
    # Convertir a formato {0,1} para facilitar
    acciones_01 = convertir_a_01(acciones, formato)
    
    # 1. Calcular volatilidad y obtener victorias
    print("\n[1] VOLATILIDAD σ²/N")
    vol_stats = calcular_volatilidad(secuencias)
    
    # 2. Calcular predictibilidad H/N
    print("\n[2] PREDICTIBILIDAD H/N")
    pred_stats = calcular_predictibilidad(acciones_01, 'binario_01', M=CONFIG['M'])
    
    # 3. Para MI y TE necesitamos la matriz de victorias
    print("\n[3] PREPARANDO MATRIZ DE VICTORIAS...")
    # Reutilizamos la función del script original
    from types import SimpleNamespace
    # Simulamos el entorno para usar la función existente
    temp_config = SimpleNamespace()
    temp_config.USAR_TODOS_AGENTES = True
    temp_config.USAR_TODAS_RONDAS = True
    
    # Versión simplificada de calcular_covictoria_vectorizado solo para victorias
    N, T_local = acciones_01.shape
    asistencia = np.sum(acciones_01, axis=0)
    umbral = N / 2.0
    
    victorias = np.zeros((N, T_local), dtype=np.int8)
    for t in range(T_local):
        if asistencia[t] > umbral:
            victorias[:, t] = (acciones_01[:, t] == 0).astype(np.int8)
        elif asistencia[t] < umbral:
            victorias[:, t] = (acciones_01[:, t] == 1).astype(np.int8)
        # Empates: victorias queda en 0
    
    print(f"  Matriz de victorias calculada: {victorias.shape}")
    
    # 4. Información mutua entre victorias
    print("\n[4] INFORMACIÓN MUTUA ENTRE VICTORIAS")
    mi_victorias_matrix, mi_victorias_stats = calcular_mi_entre_victorias(victorias)
    
    # 5. Información mutua acción-estado
    print("\n[5] INFORMACIÓN MUTUA ACCIÓN-ESTADO")
    # Necesitamos la acción ganadora
    asistencia = np.sum(acciones_01, axis=0)
    umbral = N / 2.0
    accion_ganadora = np.zeros(T_local, dtype=np.int8)
    accion_ganadora[asistencia < umbral] = 1
    
    mi_accion_estado_stats = calcular_mi_accion_estado(
        acciones_01, accion_ganadora, M=CONFIG['M']
    )
    
    # 6. Entropía de transferencia
    print("\n[6] ENTROPÍA DE TRANSFERENCIA")
    te_stats = calcular_entropia_transferencia(
        victorias, acciones_01, M=CONFIG['M']
    )
    
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
        "mi_victorias": {
            "estadisticas": mi_victorias_stats,
            "matriz": mi_victorias_matrix.tolist()
        },
        "mi_accion_estado": mi_accion_estado_stats,
        "transferencia": te_stats
    }
    
    # Guardar resultados completos
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
    parser = argparse.ArgumentParser(description='Análisis Avanzado del Minority Game')
    parser.add_argument('--file', type=str, required=True, help='Archivo JSON a procesar')
    parser.add_argument('--output', type=str, default='resultados_avanzados', 
                       help='Directorio de salida')
    parser.add_argument('--M', type=int, default=9, help='Memoria del juego (bits)')
    
    args = parser.parse_args()
    
    CONFIG["M"] = args.M
    
    analisis_avanzado(args.file, args.output)