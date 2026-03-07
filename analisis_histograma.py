#!/usr/bin/env python3
"""
Análisis de Grafos del Minority Game - VERSIÓN CORREGIDA
=========================================================
Detecta automáticamente el formato de datos ({0,1} o {-1,1})
y aplica la lógica correcta del Minority Game.
"""

import os
import json
import glob
import numpy as np
from collections import Counter
from datetime import datetime
from tqdm import tqdm
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ============================================================
# CONFIGURACIÓN
# ============================================================

CONFIG = {
    "USAR_TODOS_AGENTES": True,
    "USAR_TODAS_RONDAS": True,
    "BATCH_SIZE_MATRIZ": 2000,
    "K_MAX_ENTROPIA": 5,
    "PRECISION": "float64",
    "N_WORKERS": None,
    "BINS_HISTOGRAMA": 50,
    "GUARDAR_HISTOGRAMA_IMG": True,
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
    
    Returns:
        str: 'binario_01' o 'binario_pm1' (plus/minus 1)
    """
    valores_unicos = np.unique(acciones)
    
    if -1 in valores_unicos:
        print(f"  Formato detectado: {{-1, +1}}")
        return 'binario_pm1'
    else:
        print(f"  Formato detectado: {{0, 1}}")
        return 'binario_01'


# ============================================================
# 1. MATRIZ DE CO-VICTORIA - CORREGIDA
# ============================================================

def calcular_covictoria_vectorizado(acciones, formato):
    """
    Calcula matriz de co-victoria de forma vectorizada.
    
    En el Minority Game, gana el grupo MINORITARIO.
    
    Para formato {-1, +1}:
        - A = suma de acciones (entre -N y +N)
        - Si A > 0: mayoría eligió +1, gana -1
        - Si A < 0: mayoría eligió -1, gana +1
        - Si A == 0: empate
    
    Para formato {0, 1}:
        - A = suma de acciones (entre 0 y N)
        - Si A > N/2: mayoría eligió 1, gana 0
        - Si A < N/2: mayoría eligió 0, gana 1
        - Si A == N/2: empate
    """
    N, T = acciones.shape
    
    # Calcular asistencia por ronda
    asistencia = np.sum(acciones, axis=0, dtype=np.float64)
    
    if formato == 'binario_pm1':
        # Formato {-1, +1}
        # Gana -1 si A > 0, gana +1 si A < 0
        ganador_minus1 = asistencia > 0   # Rondas donde gana -1
        ganador_plus1 = asistencia < 0    # Rondas donde gana +1
        empates = asistencia == 0
        
        # Matriz de victorias: agente i ganó en ronda t
        # Un agente gana si eligió la acción minoritaria
        # Si A > 0 (gana -1): ganan los que eligieron -1
        # Si A < 0 (gana +1): ganan los que eligieron +1
        victorias = np.zeros((N, T), dtype=np.int8)
        
        for t in range(T):
            if asistencia[t] > 0:
                # Gana -1
                victorias[:, t] = (acciones[:, t] == -1).astype(np.int8)
            elif asistencia[t] < 0:
                # Gana +1
                victorias[:, t] = (acciones[:, t] == 1).astype(np.int8)
            # Si empate, nadie gana (victorias queda en 0)
        
        n_gana_minus1 = int(np.sum(ganador_minus1))
        n_gana_plus1 = int(np.sum(ganador_plus1))
        n_empates = int(np.sum(empates))
        
        print(f"  Rondas con ganador -1: {n_gana_minus1}")
        print(f"  Rondas con ganador +1: {n_gana_plus1}")
        print(f"  Rondas con empate: {n_empates}")
        
    else:
        # Formato {0, 1}
        umbral = N / 2.0
        ganador_0 = asistencia > umbral   # Mayoría eligió 1, gana 0
        ganador_1 = asistencia < umbral   # Mayoría eligió 0, gana 1
        empates = asistencia == umbral
        
        victorias = np.zeros((N, T), dtype=np.int8)
        
        for t in range(T):
            if asistencia[t] > umbral:
                # Gana 0
                victorias[:, t] = (acciones[:, t] == 0).astype(np.int8)
            elif asistencia[t] < umbral:
                # Gana 1
                victorias[:, t] = (acciones[:, t] == 1).astype(np.int8)
        
        n_gana_0 = int(np.sum(ganador_0))
        n_gana_1 = int(np.sum(ganador_1))
        n_empates = int(np.sum(empates))
        
        print(f"  Rondas con ganador 0: {n_gana_0}")
        print(f"  Rondas con ganador 1: {n_gana_1}")
        print(f"  Rondas con empate: {n_empates}")
    
    # Calcular matriz de co-victoria: W_ij = (victorias_i · victorias_j) / T
    # Usamos multiplicación de matrices para eficiencia
    print(f"  Calculando matriz de co-victoria {N}x{N}...")
    
    # victorias es NxT, queremos W = V @ V.T / T
    W = victorias.astype(np.float64) @ victorias.T / T
    
    # La diagonal son las victorias individuales, la ponemos a 0
    np.fill_diagonal(W, 0)
    
    return W, victorias


def calcular_histograma_covictoria(secuencias):
    """
    Calcula histograma de pesos de co-victoria.
    """
    N = len(secuencias)
    T = len(secuencias[0])
    
    print(f"\nCalculando co-victorias (vectorizado)...")
    print(f"  Agentes: {N}, Rondas: {T}")
    
    # Convertir a matriz NumPy
    acciones = np.array(secuencias, dtype=np.int8)
    
    # Detectar formato
    formato = detectar_formato_datos(acciones)
    
    # Calcular matriz de co-victoria
    W, victorias = calcular_covictoria_vectorizado(acciones, formato)
    
    # Extraer triángulo superior (sin diagonal)
    indices_triu = np.triu_indices(N, k=1)
    valores_triu = W[indices_triu]
    
    n_pares = len(valores_triu)
    print(f"  Pares calculados: {n_pares:,}")
    
    # Estadísticas
    media = np.mean(valores_triu)
    std = np.std(valores_triu)
    minimo = np.min(valores_triu)
    maximo = np.max(valores_triu)
    
    print(f"  Media: {media:.4f}")
    print(f"  Std: {std:.4f}")
    print(f"  Min: {minimo:.4f}, Max: {maximo:.4f}")
    
    # Valor teórico si fueran independientes
    # P(ambos ganan) = P(gana_i) * P(gana_j) ≈ 0.5 * 0.5 = 0.25
    teorico = 0.25
    print(f"  Teoría (indep.): {teorico}, Diferencia: {abs(media - teorico):.4f}")
    
    # Crear histograma
    bins = CONFIG["BINS_HISTOGRAMA"]
    conteos, limites = np.histogram(valores_triu, bins=bins, range=(0, 1))
    centros = (limites[:-1] + limites[1:]) / 2
    
    histograma = {
        'bins': limites.tolist(),
        'counts': conteos.tolist(),
        'bin_centers': centros.tolist()
    }
    
    estadisticas = {
        'n_agentes': N,
        'n_rondas': T,
        'n_pares': int(n_pares),
        'formato_datos': formato,
        'media': float(media),
        'desviacion': float(std),
        'varianza': float(std**2),
        'min': float(minimo),
        'max': float(maximo),
        'teorico_independiente': teorico,
        'p5': float(np.percentile(valores_triu, 5)),
        'p25': float(np.percentile(valores_triu, 25)),
        'p50': float(np.percentile(valores_triu, 50)),
        'p75': float(np.percentile(valores_triu, 75)),
        'p95': float(np.percentile(valores_triu, 95)),
    }
    
    return estadisticas, histograma, W


def generar_imagen_histograma(histograma, estadisticas, alpha, output_dir, nombre_base):
    """Genera y guarda imagen del histograma."""
    if not CONFIG["GUARDAR_HISTOGRAMA_IMG"] or len(histograma['counts']) == 0:
        return None
    
    try:
        plt.figure(figsize=(12, 8))
        
        bins = histograma['bins']
        counts = histograma['counts']
        
        plt.bar(histograma['bin_centers'], counts, 
                width=(bins[1]-bins[0])*0.8, 
                alpha=0.7, 
                edgecolor='black',
                color='steelblue')
        
        # Líneas de referencia
        plt.axvline(estadisticas['media'], color='red', linestyle='--', 
                   linewidth=2, label=f"Media: {estadisticas['media']:.4f}")
        plt.axvline(estadisticas['teorico_independiente'], color='green', linestyle=':', 
                   linewidth=2, label=f"Teórico indep.: {estadisticas['teorico_independiente']:.2f}")
        
        plt.title(f'Distribución de Co-Victorias (α={alpha})\n'
                 f'N={estadisticas["n_agentes"]}, T={estadisticas["n_rondas"]}, '
                 f'Formato: {estadisticas["formato_datos"]}',
                 fontsize=14)
        plt.xlabel('Peso de Co-Victoria (normalizado)', fontsize=12)
        plt.ylabel('Frecuencia', fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(alpha=0.3)
        
        # Texto con estadísticas
        stats_text = (f"Media: {estadisticas['media']:.4f} ± {estadisticas['desviacion']:.4f}\n"
                     f"Mediana: {estadisticas['p50']:.4f}\n"
                     f"Rango: [{estadisticas['min']:.4f}, {estadisticas['max']:.4f}]")
        
        plt.text(0.98, 0.98, stats_text, transform=plt.gca().transAxes,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                fontsize=10)
        
        img_path = os.path.join(output_dir, f"histograma_covictoria_{nombre_base}.png")
        plt.savefig(img_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  Imagen histograma: {os.path.basename(img_path)}")
        return img_path
        
    except Exception as e:
        print(f"  Error generando imagen: {e}")
        return None


# ============================================================
# 2. DENSIDAD DE ENTROPÍA
# ============================================================

def calcular_entropia_bloque(secuencia, k):
    """Calcula entropía de bloque H_k."""
    n = len(secuencia)
    if k > n or n == 0:
        return 0.0
    
    bloques = []
    for i in range(n - k + 1):
        bloque = tuple(secuencia[i:i+k])
        bloques.append(bloque)
    
    contador = Counter(bloques)
    total = len(bloques)
    
    entropia = 0.0
    for count in contador.values():
        p = count / total
        if p > 0:
            entropia -= p * np.log2(p)
    
    return float(entropia)


def calcular_densidad_entropia(secuencias, k_max=5):
    """Calcula densidad de entropía para todos los agentes."""
    N = len(secuencias)
    print(f"\nCalculando densidad de entropía (k_max={k_max})...")
    
    resultados = []
    h_valores = []
    
    for i, seq in enumerate(tqdm(secuencias, desc="Entropía")):
        H_values = []
        res = {"agente_id": i}
        
        for k in range(1, k_max + 1):
            H_k = calcular_entropia_bloque(seq, k)
            H_values.append(H_k)
            res[f"H_{k}"] = H_k
        
        if len(H_values) > 1:
            h = H_values[-1] - H_values[-2]  # Entropy rate
            res["h_estimado"] = h
            h_valores.append(h)
        
        resultados.append(res)
    
    resumen = {
        "n_agentes": N,
        "k_max": k_max,
        "h_media": float(np.mean(h_valores)) if h_valores else 0,
        "h_std": float(np.std(h_valores)) if h_valores else 0,
        "h_mediana": float(np.median(h_valores)) if h_valores else 0,
    }
    
    print(f"  h (entropy rate) media: {resumen['h_media']:.4f} ± {resumen['h_std']:.4f}")
    
    return resultados, resumen


# ============================================================
# 3. LEMPEL-ZIV (VERSIÓN EFICIENTE DEL PRIMER SCRIPT)
# ============================================================

def lempel_ziv_completo(secuencia):
    """
    Calcula complejidad de Lempel-Ziv normalizada.
    Algoritmo LZ76 clásico - VERSIÓN EFICIENTE
    """
    n = len(secuencia)
    
    if n == 0:
        return 0.0
    if n == 1:
        return 1.0
    
    S = list(secuencia)
    
    C = 1
    u = 1
    v = 1
    vmax = 1
    
    while u + v <= n:
        if S[u + v - 1] == S[v - 1]:
            v += 1
        else:
            if v > vmax:
                vmax = v
            u += 1
            if u == vmax:
                C += 1
                u += vmax
                v = 1
                vmax = 1
            else:
                v = 1
    
    if v != 1:
        C += 1
    
    return float((C * np.log2(n)) / n)


def calcular_lempel_ziv(secuencias):
    """Calcula Lempel-Ziv para todos los agentes (usa versión eficiente)."""
    N = len(secuencias)
    print(f"\nCalculando Lempel-Ziv (versión eficiente)...")
    
    resultados = []
    valores = []
    
    for i, seq in enumerate(tqdm(secuencias, desc="Lempel-Ziv")):
        clz = lempel_ziv_completo(seq)  # AHORA USA LA VERSIÓN RÁPIDA
        resultados.append({"agente_id": i, "lempel_ziv": clz})
        valores.append(clz)
    
    resumen = {
        "n_agentes": N,
        "clz_media": float(np.mean(valores)),
        "clz_std": float(np.std(valores)),
        "clz_mediana": float(np.median(valores)),
        "clz_min": float(np.min(valores)),
        "clz_max": float(np.max(valores)),
    }
    
    print(f"  C_LZ media: {resumen['clz_media']:.4f} ± {resumen['clz_std']:.4f}")
    
    return resultados, resumen


# ============================================================
# 4. VOLATILIDAD
# ============================================================

def calcular_volatilidad(secuencias):
    """
    Calcula volatilidad σ²/N del sistema.
    
    Volatilidad = Var(A) / N
    donde A = asistencia (suma de acciones por ronda)
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
        # Varianza teórica para N agentes aleatorios: N (cada uno ±1)
        var_teorica = N
    else:
        A_teorica = N / 2.0
        # Varianza teórica: N/4
        var_teorica = N / 4.0
    
    eficiencia = 1.0 - (A_var / var_teorica) if var_teorica > 0 else 0
    
    resultados = {
        "N": N,
        "T": T,
        "formato_datos": formato,
        "A_media": float(A_media),
        "A_teorica": float(A_teorica),
        "A_std": float(np.std(asistencia, ddof=1)),
        "A_var": float(A_var),
        "sigma2": float(A_var),
        "sigma2_N": float(sigma2_N),
        "var_teorica_aleatorio": float(var_teorica),
        "eficiencia": float(eficiencia),
    }
    
    print(f"  A media: {A_media:.2f} (teórica: {A_teorica})")
    print(f"  σ²/N: {sigma2_N:.4f}")
    print(f"  Eficiencia: {eficiencia:.4f}")
    
    return resultados, asistencia.tolist()


# ============================================================
# 5. EXPORTACIÓN DE MATRIZ DE ADYACENCIA
# ============================================================

def exportar_matriz_adyacencia(W, alpha, output_dir, nombre_base):
    """
    Exporta SOLO el triángulo superior de la matriz en formato NPZ comprimido.
    Reduce espacio ~50% y usa formato binario (mucho más rápido).
    """
    N = W.shape[0]
    
    # Obtener triángulo superior (sin diagonal)
    indices_triu = np.triu_indices(N, k=1)
    valores_triu = W[indices_triu]
    
    # Crear nombre del archivo
    archivo = os.path.join(output_dir, f"matriz_{nombre_base}.npz")
    
    # Verificar si existe y generar nombre único
    contador = 1
    while os.path.exists(archivo):
        archivo = os.path.join(output_dir, f"matriz_{nombre_base}_{contador}.npz")
        contador += 1
    
    # Guardar en formato numpy comprimido
    np.savez_compressed(archivo, 
                        valores=valores_triu,
                        N=N,
                        alpha=alpha)
    
    # Calcular tamaños para info
    tamaño_original = N * N * 8 / (1024*1024)  # MB (float64)
    tamaño_guardado = os.path.getsize(archivo) / (1024*1024)
    
    print(f"  Matriz exportada: {os.path.basename(archivo)}")
    print(f"    Original: {tamaño_original:.1f} MB → Guardado: {tamaño_guardado:.1f} MB")
    print(f"    Compresión: {tamaño_original/tamaño_guardado:.1f}x")
    
    return archivo


# ============================================================
# 6. MENÚ INTERACTIVO
# ============================================================

def obtener_nombre_unico(base_path):
    """Genera nombre único si el archivo ya existe."""
    if not os.path.exists(base_path):
        return base_path
    
    directorio = os.path.dirname(base_path)
    nombre = os.path.basename(base_path)
    nombre_sin_ext, ext = os.path.splitext(nombre)
    
    contador = 1
    while os.path.exists(base_path):
        nuevo_nombre = f"{nombre_sin_ext}({contador}){ext}"
        base_path = os.path.join(directorio, nuevo_nombre)
        contador += 1
    
    return base_path


def menu_principal():
    """Menú interactivo principal."""
    print("\n" + "="*60)
    print("ANÁLISIS DE GRAFOS - MINORITY GAME")
    print("="*60)
    
    # Buscar archivos JSON
    archivos = glob.glob("*.json") + glob.glob("**/*.json", recursive=True)
    archivos = [a for a in archivos if 'betting_history' in a.lower() or 'procesado' in a.lower()]
    
    if not archivos:
        print("No se encontraron archivos JSON de betting_history")
        return
    
    print("\nArchivos disponibles:")
    for i, archivo in enumerate(archivos, 1):
        print(f"  {i}. {archivo}")
    
    seleccion = input("\nSeleccione archivo(s) (ej: 1 o 1,3,5): ").strip()
    indices = [int(x.strip()) - 1 for x in seleccion.split(",")]
    archivos_seleccionados = [archivos[i] for i in indices if 0 <= i < len(archivos)]
    
    print("\n" + "-"*60)
    print("OPERACIONES DISPONIBLES:")
    print("  1. Generar matriz de adyacencia y histograma de co-victoria")
    print("  2. Calcular densidad de entropía")
    print("  3. Calcular entropía de Lempel-Ziv")
    print("  4. Calcular volatilidad")
    print("  5. Análisis completo (todo lo anterior)")
    print("-"*60)
    
    ops = input("Seleccione operación(es) (ej: 1,3 o 5): ").strip()
    operaciones = [int(x.strip()) for x in ops.split(",")]
    
    output_dir = "resultados_grafos"
    os.makedirs(output_dir, exist_ok=True)
    
    for archivo in archivos_seleccionados:
        print(f"\n{'='*60}")
        print(f"PROCESANDO: {archivo}")
        print("="*60)
        
        secuencias, alpha = cargar_datos_completos(archivo)
        nombre_base = os.path.basename(archivo).replace('.json', '')
        
        if 5 in operaciones:
            operaciones = [1, 2, 3, 4]
        
        for op in operaciones:
            if op == 1:
                print("\n[1] HISTOGRAMA DE CO-VICTORIAS")
                stats, hist, W = calcular_histograma_covictoria(secuencias)
                
                # Imagen
                img_path = generar_imagen_histograma(hist, stats, alpha, output_dir, nombre_base)
                
                # JSON con histograma
                json_path = obtener_nombre_unico(
                    os.path.join(output_dir, f"histograma_covictoria_{nombre_base}.json")
                )
                with open(json_path, 'w') as f:
                    json.dump({"estadisticas": stats, "histograma": hist, "alpha": alpha}, f, indent=2)
                print(f"  Histograma exportado: {os.path.basename(json_path)}")
                
                exportar_matriz_adyacencia(W, alpha, output_dir, nombre_base)
                
                print(f"\n  Media: {stats['media']:.4f} ± {stats['desviacion']:.4f}")
            
            elif op == 2:
                print("\n[2] DENSIDAD DE ENTROPÍA")
                res, resumen = calcular_densidad_entropia(secuencias)
                
                json_path = obtener_nombre_unico(
                    os.path.join(output_dir, f"entropia_{nombre_base}.json")
                )
                with open(json_path, 'w') as f:
                    json.dump({"resumen": resumen, "agentes": res, "alpha": alpha}, f, indent=2)
                print(f"  Exportado: {os.path.basename(json_path)}")
            
            elif op == 3:
                print("\n[3] LEMPEL-ZIV")
                res, resumen = calcular_lempel_ziv(secuencias)
                
                json_path = obtener_nombre_unico(
                    os.path.join(output_dir, f"lempel_ziv_{nombre_base}.json")
                )
                with open(json_path, 'w') as f:
                    json.dump({"resumen": resumen, "agentes": res, "alpha": alpha}, f, indent=2)
                print(f"  Exportado: {os.path.basename(json_path)}")
            
            elif op == 4:
                print("\n[4] VOLATILIDAD")
                res, serie = calcular_volatilidad(secuencias)
                
                json_path = obtener_nombre_unico(
                    os.path.join(output_dir, f"volatilidad_{nombre_base}.json")
                )
                with open(json_path, 'w') as f:
                    json.dump({"resultados": res, "serie_asistencia": serie, "alpha": alpha}, f, indent=2)
                print(f"  Exportado: {os.path.basename(json_path)}")
    
    print(f"\n{'='*60}")
    print(f"ANÁLISIS COMPLETADO")
    print(f"Resultados guardados en: {output_dir}/")
    print("="*60)


# ============================================================
# PARÁMETROS DE LÍNEA DE COMANDO (para compatibilidad con .bat)
# ============================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Análisis de Minority Game')
    parser.add_argument('--file', type=str, help='Archivo JSON a procesar')
    parser.add_argument('--workers', type=int, default=4, help='Número de workers')
    parser.add_argument('--output', type=str, default='resultados', help='Directorio de salida')
    parser.add_argument('--auto', action='store_true', help='Modo automático (sin preguntar)')
    
    args = parser.parse_args()
    
    if args.file and args.auto:
        # MODO AUTOMÁTICO - para el script .bat
        print(f"\n{'='*60}")
        print(f"MODO AUTOMÁTICO")
        print(f"Archivo: {args.file}")
        print(f"Workers: {args.workers}")
        print(f"Salida: {args.output}")
        print("="*60)
        
        # Procesar un solo archivo en modo automático
        secuencias, alpha = cargar_datos_completos(args.file)
        nombre_base = os.path.basename(args.file).replace('.json', '')
        output_dir = args.output
        os.makedirs(output_dir, exist_ok=True)
        
        # Ejecutar análisis completo (operaciones 1,2,3,4)
        print("\n[1] HISTOGRAMA DE CO-VICTORIAS")
        stats, hist, W = calcular_histograma_covictoria(secuencias)
        
        img_path = generar_imagen_histograma(hist, stats, alpha, output_dir, nombre_base)
        
        json_path = os.path.join(output_dir, f"histograma_covictoria_{nombre_base}.json")
        with open(json_path, 'w') as f:
            json.dump({"estadisticas": stats, "histograma": hist, "alpha": alpha}, f, indent=2)
        print(f"  Histograma exportado: {os.path.basename(json_path)}")
        
        exportar_matriz_adyacencia(W, alpha, output_dir, nombre_base)
        
        print("\n[3] LEMPEL-ZIV")
        res_lz, sum_lz = calcular_lempel_ziv(secuencias)
        json_path = os.path.join(output_dir, f"lempel_ziv_{nombre_base}.json")
        with open(json_path, 'w') as f:
            json.dump({"resumen": sum_lz, "agentes": res_lz, "alpha": alpha}, f, indent=2)
        print(f"  Exportado: {os.path.basename(json_path)}")
        
        print("\n[4] VOLATILIDAD")
        res_vol, serie = calcular_volatilidad(secuencias)
        json_path = os.path.join(output_dir, f"volatilidad_{nombre_base}.json")
        with open(json_path, 'w') as f:
            json.dump({"resultados": res_vol, "serie_asistencia": serie, "alpha": alpha}, f, indent=2)
        print(f"  Exportado: {os.path.basename(json_path)}")
        
        print(f"\n{'='*60}")
        print(f"✅ ANÁLISIS COMPLETADO: {nombre_base}")
        print(f"📁 Resultados en: {output_dir}/")
        print("="*60)
        
    else:
        # MODO INTERACTIVO - menú original
        menu_principal()