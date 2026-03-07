#!/usr/bin/env python3
"""
Recopilador optimizado para matrices triangulares en formato NPZ.
"""
import json
import glob
import numpy as np
import os
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import gc

def procesar_una_realizacion(args):
    """Procesa una realización (para paralelizar)."""
    archivo_hist, carpeta = args
    
    try:
        # Cargar estadísticas del histograma
        with open(archivo_hist, 'r') as f:
            hist_data = json.load(f)
        
        stats = hist_data['estadisticas']
        
        # Buscar archivo de matriz NPZ correspondiente
        nombre_base = os.path.basename(archivo_hist).replace('histograma_covictoria_', '').replace('.json', '')
        archivo_matriz = os.path.join(carpeta, f"matriz_{nombre_base}.npz")
        
        matriz_info = None
        if os.path.exists(archivo_matriz):
            # Solo guardamos la ruta, NO cargamos la matriz aún
            matriz_info = {
                'ruta': archivo_matriz,
                'N': stats['n_agentes']
            }
        
        return {
            'exito': True,
            'stats': stats,
            'matriz_info': matriz_info,
            'nombre': nombre_base
        }
    
    except Exception as e:
        return {
            'exito': False,
            'error': str(e),
            'archivo': archivo_hist
        }

def recopilar_resultados(max_workers=4):
    """
    Recopila todos los resultados usando paralelismo.
    NO carga las matrices en memoria, solo guarda las rutas.
    """
    print("\n" + "="*60)
    print("RECOPILADOR OPTIMIZADO DE RESULTADOS")
    print("="*60)
    
    # Buscar todas las carpetas de resultados
    carpetas = glob.glob("resultados_*")
    resultados_por_alpha = {}
    
    for carpeta in carpetas:
        # Extraer alpha del nombre
        alpha_str = carpeta.replace("resultados_", "")
        try:
            alpha = float(alpha_str)
        except ValueError:
            print(f"  Ignorando carpeta: {carpeta}")
            continue
        
        print(f"\n📁 Procesando α = {alpha}")
        
        # Buscar archivos de histograma
        archivos_hist = glob.glob(os.path.join(carpeta, "histograma_covictoria_*.json"))
        
        if not archivos_hist:
            print(f"  No se encontraron archivos de histograma")
            continue
        
        print(f"  {len(archivos_hist)} realizaciones encontradas")
        
        # Preparar argumentos para paralelización
        args_list = [(ah, carpeta) for ah in archivos_hist]
        
        # Procesar en paralelo
        realizaciones = []
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(procesar_una_realizacion, args) 
                      for args in args_list]
            
            for future in tqdm(as_completed(futures), total=len(futures),
                              desc=f"  Procesando"):
                realizaciones.append(future.result())
        
        # Separar exitosos y fallidos
        exitosos = [r for r in realizaciones if r['exito']]
        fallidos = [r for r in realizaciones if not r['exito']]
        
        print(f"  ✅ {len(exitosos)} exitosas")
        if fallidos:
            print(f"  ⚠️ {len(fallidos)} fallidas")
        
        if exitosos:
            resultados_por_alpha[alpha] = {
                'n_agentes': exitosos[0]['stats']['n_agentes'],
                'n_realizaciones': len(exitosos),
                'estadisticas': [r['stats'] for r in exitosos],
                'matrices': [r['matriz_info'] for r in exitosos if r['matriz_info']]
            }
    
    return resultados_por_alpha

def guardar_resultados(resultados):
    """Guarda los resultados en JSON."""
    # Convertir para serialización JSON
    resultados_json = {}
    for alpha, data in resultados.items():
        resultados_json[str(alpha)] = {
            'n_agentes': data['n_agentes'],
            'n_realizaciones': data['n_realizaciones'],
            'estadisticas': data['estadisticas'],
            'matrices': data['matrices']  # Solo rutas, no datos
        }
    
    with open('resultados_consolidados.json', 'w') as f:
        json.dump(resultados_json, f, indent=2)
    
    print(f"\n✅ Resultados guardados en 'resultados_consolidados.json'")
    
    # Mostrar resumen
    print("\n📊 RESUMEN:")
    for alpha in sorted(resultados.keys()):
        data = resultados[alpha]
        print(f"  α={alpha}: {data['n_realizaciones']} realizaciones, "
              f"N={data['n_agentes']} agentes, "
              f"{len(data['matrices'])} matrices")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=4, help='Número de workers')
    args = parser.parse_args()
    
    resultados = recopilar_resultados(max_workers=args.workers)
    guardar_resultados(resultados)