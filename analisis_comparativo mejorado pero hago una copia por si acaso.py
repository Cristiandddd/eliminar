#!/usr/bin/env python3
"""
Compara TODOS los α en gráficas de densidad vs umbral
Guarda grid completo y cada gráfica por separado
"""
import json
import matplotlib.pyplot as plt
import numpy as np
import os

# Crear carpeta para gráficas individuales
os.makedirs('graficas_individuales', exist_ok=True)

# Cargar datos
with open('analisis_redes.json') as f:
    data = json.load(f)

# ------------------------------------------------------------------
# GRÁFICA 1: Grid completo (todos los α)
# ------------------------------------------------------------------
print("Generando grid completo...")

fig = plt.figure(figsize=(18, 10))

# Obtener todos los α ordenados
alphas = sorted(data['unimodal'].keys(), key=float)
colores = plt.cm.viridis(np.linspace(0, 1, len(alphas)))

# Subplot 1: Densidad vs Umbral (todos los α)
ax1 = plt.subplot(2, 3, 1)
for i, alpha_str in enumerate(alphas):
    umbrales = []
    densidades = []
    for u_str in sorted(data['unimodal'][alpha_str].keys(), key=float):
        umbrales.append(float(u_str))
        densidades.append(data['unimodal'][alpha_str][u_str]['densidad_media'])
    
    ax1.plot(umbrales, densidades, marker='o', color=colores[i], 
             linewidth=2, label=f'α={float(alpha_str):.2f}', markersize=4, alpha=0.8)

ax1.set_xlabel('Umbral θ')
ax1.set_ylabel('Densidad de red')
ax1.set_title('Distribución acumulativa de W_ij')
ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=6)
ax1.grid(True, alpha=0.3)

# Subplot 2: Densidad en θ=0.24 vs α
ax2 = plt.subplot(2, 3, 2)
alphas_num = []
densidades_24 = []
errores_24 = []

for alpha_str in alphas:
    alpha = float(alpha_str)
    if '0.24' in data['unimodal'][alpha_str]:
        alphas_num.append(alpha)
        densidades_24.append(data['unimodal'][alpha_str]['0.24']['densidad_media'])
        errores_24.append(data['unimodal'][alpha_str]['0.24'].get('densidad_std', 0))

if alphas_num:
    ax2.errorbar(alphas_num, densidades_24, yerr=errores_24, 
                 fmt='o-', linewidth=2, color='purple', 
                 markersize=8, capsize=5, label='Unimodal')

ax2.axvline(x=0.34, color='red', linestyle='--', alpha=0.5, label='α crítico')
ax2.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5, label='Densidad=0.5')
ax2.set_xlabel('α')
ax2.set_ylabel('Densidad en θ=0.24')
ax2.set_title('Evolución de la correlación con α')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_xscale('log')

# Subplot 3: Información de clasificación
ax3 = plt.subplot(2, 3, 3)
info_text = "CLASIFICACIÓN POR α:\n"
info_text += "-" * 20 + "\n"

for alpha_str in alphas:
    n_uni = data['unimodal'][alpha_str]['0.24'].get('n', 0) if '0.24' in data['unimodal'][alpha_str] else 0
    n_bi = 0
    if 'bimodal' in data and alpha_str in data['bimodal']:
        n_bi = data['bimodal'][alpha_str].get('0.24', {}).get('n', []) if '0.24' in data['bimodal'][alpha_str] else 0
    total = n_uni + n_bi
    pct_bi = (n_bi / total * 100) if total > 0 else 0
    info_text += f"α={float(alpha_str):.3f}: {n_uni} uni, {n_bi} bi ({pct_bi:.1f}% bimodal)\n"

ax3.text(0.5, 0.5, info_text, ha='center', va='center',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9),
         fontsize=9, family='monospace')
ax3.axis('off')

# Subplot 4: Histograma de α (distribución de valores)
ax4 = plt.subplot(2, 3, 4)
alphas_valores = [float(a) for a in alphas]
ax4.bar(range(len(alphas_valores)), alphas_valores, tick_label=[f'{a:.2f}' for a in alphas_valores])
ax4.set_xlabel('Índice')
ax4.set_ylabel('α')
ax4.set_title('Valores de α analizados')
ax4.tick_params(axis='x', rotation=45)

# Subplot 5: Boxplot de densidades (opcional)
ax5 = plt.subplot(2, 3, 5)
densidades_todos = []
for alpha_str in alphas:
    if '0.24' in data['unimodal'][alpha_str]:
        densidades_todos.append(data['unimodal'][alpha_str]['0.24']['densidad_media'])

if densidades_todos:
    ax5.boxplot(densidades_todos)
    ax5.set_xticklabels([f'{float(a):.2f}' for a in alphas])
    ax5.set_xlabel('α')
    ax5.set_ylabel('Densidad en θ=0.24')
    ax5.set_title('Distribución de densidades')
    ax5.tick_params(axis='x', rotation=45)

# Subplot 6: Reservado para bimodales si existen
ax6 = plt.subplot(2, 3, 6)
if 'bimodal' in data and data['bimodal']:
    alphas_bi = sorted(data['bimodal'].keys(), key=float)
    for alpha_str in alphas_bi:
        umbrales = []
        densidades = []
        for u_str in sorted(data['bimodal'][alpha_str].keys(), key=float):
            umbrales.append(float(u_str))
            densidades.append(data['bimodal'][alpha_str][u_str]['densidad_media'])
        ax6.plot(umbrales, densidades, 's-', label=f'α={float(alpha_str):.2f}')
    ax6.set_xlabel('Umbral θ')
    ax6.set_ylabel('Densidad')
    ax6.set_title('Redes BIMODALES')
    ax6.legend(fontsize=8)
    ax6.grid(True, alpha=0.3)
else:
    ax6.text(0.5, 0.5, 'No hay redes bimodales\nen estos datos', 
             ha='center', va='center', fontsize=12)
    ax6.set_title('Redes BIMODALES')
    ax6.axis('off')

plt.tight_layout()
plt.savefig('grid_completo_comparacion.png', dpi=300, bbox_inches='tight')
plt.show()
print("✅ Grid guardado: grid_completo_comparacion.png")

# ------------------------------------------------------------------
# GRÁFICA 2: Densidad vs Umbral (por separado)
# ------------------------------------------------------------------
print("\nGenerando gráficas individuales...")

plt.figure(figsize=(10, 6))
for alpha_str in alphas:
    plt.clf()  # Limpiar figura
    
    umbrales = []
    densidades = []
    errores = []
    
    for u_str in sorted(data['unimodal'][alpha_str].keys(), key=float):
        umbrales.append(float(u_str))
        densidades.append(data['unimodal'][alpha_str][u_str]['densidad_media'])
        errores.append(data['unimodal'][alpha_str][u_str].get('densidad_std', 0))
    
    plt.errorbar(umbrales, densidades, yerr=errores, 
                 fmt='o-', linewidth=2, color='blue', 
                 markersize=8, capsize=5, label='Unimodal')
    
    # Si hay bimodales para este α, agregarlos
    if 'bimodal' in data and alpha_str in data['bimodal']:
        umbrales_bi = []
        densidades_bi = []
        for u_str in sorted(data['bimodal'][alpha_str].keys(), key=float):
            umbrales_bi.append(float(u_str))
            densidades_bi.append(data['bimodal'][alpha_str][u_str]['densidad_media'])
        plt.plot(umbrales_bi, densidades_bi, 's-', linewidth=2, color='red', 
                label='Bimodal', markersize=8)
    
    plt.xlabel('Umbral θ')
    plt.ylabel('Densidad de red')
    plt.title(f'Distribución acumulativa para α = {float(alpha_str):.3f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim([-0.05, 1.05])
    
    nombre = f'graficas_individuales/densidad_vs_umbral_alpha_{float(alpha_str):.3f}.png'
    plt.savefig(nombre, dpi=300, bbox_inches='tight')
    print(f"  Guardado: {nombre}")

# ------------------------------------------------------------------
# GRÁFICA 3: Densidad en θ=0.24 vs α (por separado)
# ------------------------------------------------------------------
plt.figure(figsize=(10, 6))

alphas_num = []
densidades_24 = []
errores_24 = []

for alpha_str in alphas:
    if '0.24' in data['unimodal'][alpha_str]:
        alphas_num.append(float(alpha_str))
        densidades_24.append(data['unimodal'][alpha_str]['0.24']['densidad_media'])
        errores_24.append(data['unimodal'][alpha_str]['0.24'].get('densidad_std', 0))

if alphas_num:
    plt.errorbar(alphas_num, densidades_24, yerr=errores_24, 
                 fmt='o-', linewidth=2, color='purple', 
                 markersize=8, capsize=5)

plt.axvline(x=0.34, color='red', linestyle='--', alpha=0.5, label='α crítico (teórico)')
plt.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5, label='Densidad=0.5')
plt.xlabel('α')
plt.ylabel('Densidad en θ=0.24')
plt.title('Evolución de la correlación con α')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xscale('log')

plt.savefig('densidad_en_theta_024_vs_alpha.png', dpi=300, bbox_inches='tight')
print("\n✅ Gráfica guardada: densidad_en_theta_024_vs_alpha.png")

# ------------------------------------------------------------------
# GRÁFICA 4: Boxplot de densidades por α (por separado)
# ------------------------------------------------------------------
plt.figure(figsize=(12, 6))

datos_boxplot = []
etiquetas = []

for alpha_str in alphas:
    if '0.24' in data['unimodal'][alpha_str] and 'n' in data['unimodal'][alpha_str]['0.24']:
        # Necesitamos los valores individuales, no solo la media
        # Si no están disponibles, usamos la media con un marcador
        pass

# Versión simplificada: puntos con medias
for i, alpha_str in enumerate(alphas):
    if '0.24' in data['unimodal'][alpha_str]:
        plt.scatter(i, data['unimodal'][alpha_str]['0.24']['densidad_media'], 
                   color='blue', s=100, zorder=5)
        plt.errorbar(i, data['unimodal'][alpha_str]['0.24']['densidad_media'],
                    yerr=data['unimodal'][alpha_str]['0.24'].get('densidad_std', 0),
                    color='blue', capsize=5)

plt.xticks(range(len(alphas)), [f'{float(a):.2f}' for a in alphas], rotation=45)
plt.xlabel('α')
plt.ylabel('Densidad en θ=0.24')
plt.title('Densidades por α (con desviación estándar)')
plt.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('densidades_por_alpha.png', dpi=300, bbox_inches='tight')
print("✅ Gráfica guardada: densidades_por_alpha.png")

print("\n🎉 ¡Todas las gráficas generadas exitosamente!")
print("   📁 Grid completo: grid_completo_comparacion.png")
print("   📁 Individuales: graficas_individuales/")
print("   📁 Especiales: densidad_en_theta_024_vs_alpha.png, densidades_por_alpha.png")