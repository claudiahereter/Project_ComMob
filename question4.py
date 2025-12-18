import numpy as np
import matplotlib.pyplot as plt
from cellular_network_simulator import CellularNetworkSimulator

# =========================================
# PARÁMETROS DE LA SIMULACIÓN
# =========================================
NUM_SNAPSHOTS = 5000
PATHLOSS_EXP = 3.8
BANDWIDTH_MHz = 100  # Ancho de banda total
SNR_GAP_dB = 4  # SNR gap to capacity
REUSE_FACTORS = [1, 3, 9]

# Inicializar simulador (sin control de potencia, como especifica la pregunta)
sim = CellularNetworkSimulator(
    cell_radius=1.0,
    pathloss_exp=PATHLOSS_EXP,
    shadow_sigma_dB=8.0,
    seed=42
)

# Función para calcular el throughput (Shannon capacity)
def calculate_throughput(sir_linear, bandwidth_MHz, snr_gap_dB):
    """
    Calcula el throughput usando la fórmula de Shannon:
    C = B * log2(1 + SIR / SNR_gap)
    
    donde SNR_gap es el gap a capacidad (en lineal)
    """
    snr_gap_linear = 10 ** (snr_gap_dB / 10)
    # Throughput en Mbps
    throughput = bandwidth_MHz * np.log2(1 + sir_linear / snr_gap_linear)
    return throughput

# Diccionario para almacenar resultados
results = {}

print("=" * 60)
print("PREGUNTA 4: ANÁLISIS DE THROUGHPUT CON DIFERENTES FACTORES DE REUSO")
print("=" * 60)

# =========================================
# SIMULACIÓN PARA CADA FACTOR DE REUSO
# =========================================
for reuse in REUSE_FACTORS:
    reuse_mode = f"reuse{reuse}"
    
    # Ancho de banda disponible por sector (considerando el reuso)
    bw_per_sector = BANDWIDTH_MHz / reuse
    
    print(f"\n{'=' * 60}")
    print(f"Factor de Reuso: {reuse}")
    print(f"Ancho de banda por sector: {bw_per_sector:.2f} MHz")
    print(f"{'=' * 60}")
    
    # Ejecutar simulación Monte Carlo (sin control de potencia)
    sir_linear = sim.run_monte_carlo(
        num_snapshots=NUM_SNAPSHOTS,
        reuse_mode=reuse_mode
    ).ravel()
    
    # Convertir SIR a dB para análisis
    sir_dB = sim.linear_to_dB(sir_linear)
    
    # Calcular throughput para cada usuario
    # IMPORTANTE: El ancho de banda disponible depende del factor de reuso
    throughput = calculate_throughput(sir_linear, bw_per_sector, SNR_GAP_dB)
    
    # Estadísticas
    avg_throughput = np.mean(throughput)
    percentile_97 = np.percentile(throughput, 97)
    
    # Almacenar resultados
    results[reuse] = {
        'sir_linear': sir_linear,
        'sir_dB': sir_dB,
        'throughput': throughput,
        'bandwidth': bw_per_sector,
        'avg_throughput': avg_throughput,
        'percentile_97': percentile_97
    }
    
    print(f"Throughput promedio: {avg_throughput:.2f} Mbps")
    print(f"Throughput alcanzado por el 97% de usuarios: {percentile_97:.2f} Mbps")

# =========================================
# VISUALIZACIÓN: CDF DE THROUGHPUT (CORREGIDA)
# =========================================
plt.figure(figsize=(12, 7))

colors = {'1': 'blue', '3': 'orange', '9': 'green'}
for reuse in REUSE_FACTORS:
    throughput_sorted = np.sort(results[reuse]['throughput'])
    # ✅ CORRECCIÓN: Usar endpoint=True para que la CDF incluya correctamente [0,1]
    cdf = np.linspace(0, 1, len(throughput_sorted), endpoint=True)
    
    plt.plot(
        throughput_sorted, 
        cdf, 
        linewidth=2.5,
        label=f'Reuse {reuse} (BW = {results[reuse]["bandwidth"]:.1f} MHz)',
        color=colors[str(reuse)]
    )

# Línea de referencia para 97%
plt.axhline(0.97, color='red', linestyle='--', linewidth=1.5, 
            alpha=0.7, label='97% de usuarios')

plt.xlabel('Throughput [Mbps]', fontsize=12)
plt.ylabel('CDF', fontsize=12)
plt.title('CDF de Throughput para Factores de Reuso 1, 3 y 9\n' + 
          f'(ν = {PATHLOSS_EXP}, BW total = {BANDWIDTH_MHz} MHz, SNR gap = {SNR_GAP_dB} dB)',
          fontsize=13)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=10, loc='lower right')
plt.xlim(left=0)  # ✅ Asegurar que el eje X comienza en 0
plt.ylim(0, 1.05)  # ✅ Asegurar rango correcto de CDF
plt.tight_layout()
plt.savefig('Figure_4_throughput_cdf.png', dpi=300, bbox_inches='tight')
plt.show()

# =========================================
# TABLA RESUMEN DE RESULTADOS
# =========================================
print("\n" + "=" * 80)
print("TABLA RESUMEN DE RESULTADOS")
print("=" * 80)
print(f"{'Factor Reuso':<15} {'BW (MHz)':<12} {'Throughput Promedio (Mbps)':<30} {'Throughput 97% (Mbps)':<25}")
print("-" * 80)

for reuse in REUSE_FACTORS:
    r = results[reuse]
    print(f"{reuse:<15} {r['bandwidth']:<12.2f} {r['avg_throughput']:<30.2f} {r['percentile_97']:<25.2f}")

print("=" * 80)

# =========================================
# GRÁFICO DE BARRAS COMPARATIVO
# =========================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Gráfico 1: Throughput promedio
reuse_labels = [str(r) for r in REUSE_FACTORS]
avg_throughputs = [results[r]['avg_throughput'] for r in REUSE_FACTORS]
p97_throughputs = [results[r]['percentile_97'] for r in REUSE_FACTORS]

bars1 = ax1.bar(reuse_labels, avg_throughputs, 
                color=[colors[str(r)] for r in REUSE_FACTORS],
                alpha=0.7, edgecolor='black', linewidth=1.5)
ax1.set_xlabel('Factor de Reuso', fontsize=12)
ax1.set_ylabel('Throughput Promedio [Mbps]', fontsize=12)
ax1.set_title('Throughput Promedio por Factor de Reuso', fontsize=13)
ax1.grid(True, alpha=0.3, axis='y')

# Añadir valores sobre las barras
for bar, val in zip(bars1, avg_throughputs):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
             f'{val:.1f}',
             ha='center', va='bottom', fontsize=10, fontweight='bold')

# Gráfico 2: Throughput alcanzado por 97% de usuarios
bars2 = ax2.bar(reuse_labels, p97_throughputs,
                color=[colors[str(r)] for r in REUSE_FACTORS],
                alpha=0.7, edgecolor='black', linewidth=1.5)
ax2.set_xlabel('Factor de Reuso', fontsize=12)
ax2.set_ylabel('Throughput al 97% [Mbps]', fontsize=12)
ax2.set_title('Throughput Alcanzado por el 97% de Usuarios', fontsize=13)
ax2.grid(True, alpha=0.3, axis='y')

# Añadir valores sobre las barras
for bar, val in zip(bars2, p97_throughputs):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
             f'{val:.1f}',
             ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('Figure_4_throughput_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# =========================================
# ANÁLISIS ADICIONAL
# =========================================
print("\n" + "=" * 80)
print("ANÁLISIS DETALLADO")
print("=" * 80)

print("\n1. RELACIÓN ANCHO DE BANDA vs INTERFERENCIA:")
print("-" * 80)
for reuse in REUSE_FACTORS:
    r = results[reuse]
    avg_sir_dB = np.mean(r['sir_dB'])
    median_sir_dB = np.median(r['sir_dB'])
    print(f"\nReuso {reuse}:")
    print(f"  - Ancho de banda disponible: {r['bandwidth']:.2f} MHz ({r['bandwidth']/BANDWIDTH_MHz*100:.0f}%)")
    print(f"  - SIR promedio: {avg_sir_dB:.2f} dB")
    print(f"  - SIR mediana: {median_sir_dB:.2f} dB")
    print(f"  - Throughput promedio: {r['avg_throughput']:.2f} Mbps")

print("\n2. COMPARACIÓN DE RENDIMIENTO:")
print("-" * 80)
baseline = results[1]['avg_throughput']
for reuse in REUSE_FACTORS:
    improvement = (results[reuse]['avg_throughput'] / baseline - 1) * 100
    print(f"Reuso {reuse}: {improvement:+.1f}% respecto a Reuso 1")

print("\n" + "=" * 80)
print("CONCLUSIONES:")
print("=" * 80)
print("""
El análisis muestra el trade-off fundamental entre ancho de banda e interferencia:

- Reuso 1: Máximo ancho de banda (100 MHz) pero alta interferencia → menor SIR
- Reuso 3: Balance intermedio (33.3 MHz por sector) con interferencia moderada
- Reuso 9: Menor ancho de banda (11.1 MHz) pero mínima interferencia → mayor SIR

El throughput depende de ambos factores según la fórmula de Shannon:
C = BW × log₂(1 + SIR/SNR_gap)

Donde el ancho de banda efectivo es: BW_efectivo = BW_total / factor_reuso
""")

print("\n" + "=" * 80)
