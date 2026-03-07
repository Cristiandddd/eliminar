import os
import json
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import sys
import glob

def transformar_vectorizado(historia, max_elementos):
    """Versión ultra rápida con NumPy."""
    # Tomar primeros elementos y convertir a NumPy
    arr = np.array(historia[:max_elementos], dtype=np.int8)
    # Transformación: -1→0, 1→1  (fórmula: (x + 1) // 2)
    return ((arr + 1) // 2).tolist()

def procesar_archivo_rapido(archivo_entrada, max_elementos=10000, workers=4):
    """Procesamiento rápido sin barra de progreso."""
    try:
        print(f"📁 Procesando {os.path.basename(archivo_entrada)}...")
        
        # Leer JSON
        with open(archivo_entrada, 'r') as f:
            datos = json.load(f)
        
        # Extraer datos
        alpha = datos['game_parameters']['alpha']
        agentes = datos['agents']
        n_agentes = len(agentes)
        
        print(f"  Alpha: {alpha}, Agentes: {n_agentes}")
        
        # Procesar todos los agentes en paralelo
        with ProcessPoolExecutor(max_workers=workers) as executor:
            # Preparar todas las tareas
            futures = []
            for agente in agentes:
                future = executor.submit(
                    transformar_vectorizado, 
                    agente['betting_history'], 
                    max_elementos
                )
                futures.append(future)
            
            # Recolectar resultados (en orden)
            historias = [f.result() for f in futures]
        
        # Construir estructura de salida
        datos_salida = [[historia, alpha] for historia in historias]
        
        # Nombre de archivo de salida en ../workspace
        nombre_base = os.path.splitext(os.path.basename(archivo_entrada))[0]
        archivo_salida = os.path.join("..", "workspace", f"{nombre_base}_transformado.json")
        
        # Guardar (sin indentación para ahorrar espacio)
        with open(archivo_salida, 'w') as f:
            json.dump(datos_salida, f)
        
        print(f"  ✅ Guardado: {archivo_salida}")
        return True
        
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False

def encontrar_archivos_a_procesar():
    """Encuentra todos los archivos JSON que NO son _transformado."""
    # Buscar todos los .json
    todos_json = glob.glob("*.json")
    
    # Filtrar excluyendo _transformado.json
    archivos_a_procesar = []
    for archivo in todos_json:
        nombre = archivo.lower()
        if not nombre.endswith('_transformado.json') and not nombre.endswith('_procesado.json'):
            # Verificar que tenga la estructura correcta
            try:
                with open(archivo, 'r') as f:
                    datos = json.load(f)
                if 'game_parameters' in datos and 'agents' in datos:
                    archivos_a_procesar.append(archivo)
                else:
                    print(f"  ⚠️  Ignorado (formato incorrecto): {archivo}")
            except:
                print(f"  ⚠️  Ignorado (JSON inválido): {archivo}")
    
    return archivos_a_procesar

def main_interactivo():
    """Modo interactivo simple."""
    print("\n" + "="*50)
    print("   TRANSFORMADOR RÁPIDO DE JSON")
    print("   {-1, 1} → {0, 1}")
    print("="*50)
    
    # Verificar que existe ../workspace
    if not os.path.exists("../workspace"):
        print("📂 Creando directorio ../workspace...")
        os.makedirs("../workspace", exist_ok=True)
    
    # Encontrar archivos
    archivos = encontrar_archivos_a_procesar()
    
    if not archivos:
        print("❌ No se encontraron archivos JSON válidos para procesar.")
        print("   Los archivos deben tener estructura {game_parameters, agents}")
        return
    
    # Mostrar archivos encontrados
    print(f"\n📁 Archivos encontrados: {len(archivos)}")
    for i, archivo in enumerate(archivos, 1):
        tamano = os.path.getsize(archivo) / (1024*1024)  # MB
        print(f"   {i}. {archivo} ({tamano:.1f} MB)")
    
    # Configurar workers
    n_cores = mp.cpu_count()
    print(f"\n💻 Núcleos de CPU disponibles: {n_cores}")
    
    workers_input = input(f"🔧 Número de workers [{n_cores}]: ").strip()
    if workers_input.isdigit():
        workers = min(int(workers_input), n_cores)
    else:
        workers = n_cores
    
    # Procesar todos los archivos
    print(f"\n⚡ Procesando {len(archivos)} archivos con {workers} workers...")
    print("-" * 50)
    
    exitosos = 0
    for archivo in archivos:
        if procesar_archivo_rapido(archivo, workers=workers):
            exitosos += 1
    
    # Resumen
    print("\n" + "="*50)
    print("   RESUMEN DEL PROCESAMIENTO")
    print("="*50)
    print(f"   ✅ Archivos procesados exitosamente: {exitosos}/{len(archivos)}")
    print(f"\n   📂 Los archivos transformados están en: ../workspace/")
    print(f"      (ruta absoluta: /workspace/)")

if __name__ == "__main__":
    # Si se pasan argumentos, modo CLI automático
    if len(sys.argv) > 1 and sys.argv[1].isdigit():
        # Formato: python transformador_json.py 4
        workers = int(sys.argv[1])
        
        # Verificar directorio
        os.makedirs("../workspace", exist_ok=True)
        
        archivos = encontrar_archivos_a_procesar()
        if archivos:
            print(f"🔧 Modo automático: {len(archivos)} archivos, workers={workers}")
            print("-" * 40)
            
            exitosos = 0
            for archivo in archivos:
                if procesar_archivo_rapido(archivo, workers=workers):
                    exitosos += 1
            
            print(f"\n✅ Procesados: {exitosos}/{len(archivos)} archivos")
            print(f"📂 Archivos guardados en: ../workspace/")
        else:
            print("❌ No hay archivos para procesar")
    else:
        # Modo interactivo
        main_interactivo()