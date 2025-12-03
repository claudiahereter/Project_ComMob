import numpy as np
import matplotlib.pyplot as plt
from cellular_network_simulator_2 import CellularNetworkSimulator

NUM_SNAPSHOTS = 5000
THRESHOLD_dB = -5
REUSE_MODE = "reuse3"

sim = CellularNetworkSimulator(cell_radius=1.0, pathloss_exp=3.8, shadow_sigma_dB=8.0, seed=123)

alpha_values = np.linspace(0, 1, 21)
percentages = []

print("Scanning power control exponent (alpha)...")

for alpha in alpha_values:
    sir_linear = sim.run_monte_carlo_power_control(NUM_SNAPSHOTS, reuse_mode=REUSE_MODE, alpha=alpha).ravel()
    sir_dB = sim.linear_to_dB(sir_linear)
    pct = 100 * np.mean(sir_dB >= THRESHOLD_dB)
    percentages.append(pct)
    print(f"  alpha = {alpha:.2f}: {pct:.2f}%")

optimal_idx = np.argmax(percentages)
optimal_alpha = alpha_values[optimal_idx]
optimal_percentage = percentages[optimal_idx]

print(f"\nOPTIMAL ALPHA = {optimal_alpha:.2f}")
print(f"Maximum percentage: {optimal_percentage:.2f}%")

plt.figure(figsize=(10, 6))
plt.plot(alpha_values, percentages, 'b-o', linewidth=2, markersize=6)
plt.axhline(97, color='gray', linestyle='--', alpha=0.7, label='97% target')
plt.scatter([optimal_alpha], [optimal_percentage], color='red', s=150, zorder=5, label=f'Optimal: a={optimal_alpha:.2f}')
plt.xlabel('Power Control Exponent (alpha)')
plt.ylabel('Percentage of Users with SIR >= -5 dB (%)')
plt.title('Optimization of Fractional Power Control (Reuse Factor 3)')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig('question2_optimization.png', dpi=300)
plt.show()

sir_optimal = sim.run_monte_carlo_power_control(NUM_SNAPSHOTS, reuse_mode=REUSE_MODE, alpha=optimal_alpha).ravel()
sir_optimal_dB = sim.linear_to_dB(sir_optimal)

sir_no_pc = sim.run_monte_carlo_power_control(NUM_SNAPSHOTS, reuse_mode=REUSE_MODE, alpha=0.0).ravel()
sir_no_pc_dB = sim.linear_to_dB(sir_no_pc)

plt.figure(figsize=(10, 6))

sorted_no_pc = np.sort(sir_no_pc_dB)
p_no_pc = np.linspace(0, 1, len(sorted_no_pc), endpoint=False)
plt.plot(sorted_no_pc, p_no_pc, 'b-', linewidth=2, label='alpha = 0 (no power control)')

sorted_opt = np.sort(sir_optimal_dB)
p_opt = np.linspace(0, 1, len(sorted_opt), endpoint=False)
plt.plot(sorted_opt, p_opt, 'r-', linewidth=2, label=f'alpha = {optimal_alpha:.2f} (optimal)')

plt.axvline(THRESHOLD_dB, color='gray', linestyle='--', label='Threshold = -5 dB')
plt.axhline(0.97, color='green', linestyle=':', alpha=0.7, label='97% probability')

plt.xlabel('SIR [dB]')
plt.ylabel('CDF')
plt.title('CDF of SIR with Fractional Power Control (Reuse 3)')
plt.grid(True, alpha=0.3)
plt.legend()
plt.xlim(-30, 40)
plt.tight_layout()
plt.savefig('question2_sir_cdf.png', dpi=300)
plt.show()

pct_no_pc = 100 * np.mean(sir_no_pc_dB >= THRESHOLD_dB)
pct_opt = 100 * np.mean(sir_optimal_dB >= THRESHOLD_dB)
print(f"\nWithout power control: {pct_no_pc:.2f}%")
print(f"With optimal alpha={optimal_alpha:.2f}: {pct_opt:.2f}%")
print(f"Improvement: +{pct_opt - pct_no_pc:.2f} pp")
