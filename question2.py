import numpy as np
import matplotlib.pyplot as plt
from cellular_network_simulator import CellularNetworkSimulator

# Simulation parameters
NUM_SNAPSHOTS = 2000
REUSE_MODE = "reuse3"        
SIR_THRESHOLD_DB = -5.0
ETAS = np.linspace(0.0, 1.0, 21)   # 0, 0.05, ..., 1.0

sim = CellularNetworkSimulator(
    cell_radius=1.0,
    pathloss_exp=3.8,
    shadow_sigma_dB=8.0,
    seed=42          #for reproducibility
)

# Find best eta
best_eta, best_cov, etas, coverages = sim.find_best_eta(
    num_snapshots=NUM_SNAPSHOTS,
    reuse_mode=REUSE_MODE,
    threshold_db=SIR_THRESHOLD_DB,
    etas=ETAS
)

# Print results
print("\nPOWER CONTROL OPTIMIZATION")
for e, c in zip(etas, coverages):
    print(f"eta = {e:4.2f}  -->  coverage = {c:6.2f} %")
print(f"\nBest eta = {best_eta:.2f}")
print(f"Max coverage = {best_cov:.2f} %\n")

# plot coverage vs eta
plt.figure(figsize=(10, 6))
plt.plot(etas, coverages, 'b-o', linewidth=2, markersize=6)
plt.axhline(97, color='gray', linestyle='--', alpha=0.7, label='97% target')
plt.scatter(
    [best_eta], [best_cov],
    color='red', s=150, zorder=5,
    label=f'Optimal: η={best_eta:.2f}'
)

plt.xlabel('Power Control Exponent (η)')
plt.ylabel('Percentage of Users with SIR ≥ -5 dB (%)')
plt.title('Optimization of Fractional Power Control (Reuse Factor 3)')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig('Figure_2_optimization.png', dpi=300)
plt.show()

#compute and plot CDFs for optimal eta and no power control
sir_optimal = sim.run_monte_carlo_power_control(
    NUM_SNAPSHOTS,
    reuse_mode=REUSE_MODE,
    eta=best_eta
).ravel()
sir_optimal_dB = sim.linear_to_dB(sir_optimal)

sir_no_pc = sim.run_monte_carlo_power_control(
    NUM_SNAPSHOTS,
    reuse_mode=REUSE_MODE,
    eta=0.0
).ravel()
sir_no_pc_dB = sim.linear_to_dB(sir_no_pc)

plt.figure(figsize=(10, 6))

sorted_no_pc = np.sort(sir_no_pc_dB)
p_no_pc = np.linspace(0, 1, len(sorted_no_pc), endpoint=False)
plt.plot(sorted_no_pc, p_no_pc, 'b-', linewidth=2,
         label='η = 0 (no power control)')

sorted_opt = np.sort(sir_optimal_dB)
p_opt = np.linspace(0, 1, len(sorted_opt), endpoint=False)
plt.plot(sorted_opt, p_opt, 'r-', linewidth=2,
         label=f'η = {best_eta:.2f} (optimal)')

plt.axvline(SIR_THRESHOLD_DB, color='gray', linestyle='--',
            label='Threshold = -5 dB')
plt.axhline(0.97, color='green', linestyle=':', alpha=0.7,
            label='97% probability')

plt.xlabel('SIR [dB]')
plt.ylabel('CDF')
plt.title('CDF of SIR with Fractional Power Control (Reuse 3)')
plt.grid(True, alpha=0.3)
plt.legend()
plt.xlim(-30, 40)
plt.tight_layout()
plt.savefig('Figure_2_sir_cdf.png', dpi=300)
plt.show()

# final summary of results
pct_no_pc = 100 * np.mean(sir_no_pc_dB >= SIR_THRESHOLD_DB)
pct_opt = 100 * np.mean(sir_optimal_dB >= SIR_THRESHOLD_DB)

print(f"\nWithout power control: {pct_no_pc:.2f}%")
print(f"With optimal η={best_eta:.2f}: {pct_opt:.2f}%")
print(f"Improvement: +{pct_opt - pct_no_pc:.2f} pp")

