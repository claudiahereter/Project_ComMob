import numpy as np
import matplotlib.pyplot as plt
from cellular_network_simulator import CellularNetworkSimulator


NUM_SNAPSHOTS = 5000        
THRESHOLD_dB = -5        

sim = CellularNetworkSimulator(cell_radius=1.0, pathloss_exp=3.8, shadow_sigma_dB=8.0, seed=123)


# run simulation for each reuse factor and convert to dB
sir_r1 = sim.linear_to_dB(sim.run_monte_carlo(NUM_SNAPSHOTS, reuse_mode="reuse1").ravel())
sir_r3 = sim.linear_to_dB(sim.run_monte_carlo(NUM_SNAPSHOTS, reuse_mode="reuse3").ravel())
sir_r9 = sim.linear_to_dB(sim.run_monte_carlo(NUM_SNAPSHOTS, reuse_mode="reuse9").ravel())

results = {
    "Reuse 1": sir_r1,
    "Reuse 3": sir_r3,
    "Reuse 9": sir_r9
}

#helper functions
def percentage_above_threshold(data_dB, threshold):
    return 100 * np.mean(data_dB >= threshold)

def percentile_value(data_dB, percentile):
    return np.percentile(data_dB, percentile)

#print results
for label, sir_data in results.items():
    pct = percentage_above_threshold(sir_data, THRESHOLD_dB)
    p97 = percentile_value(sir_data, 97)

    print(f"\n{label}:")
    print(f"percentage of users with SIR ≥ {THRESHOLD_dB} dB: {pct:.2f}%")
    print(f"SIR at 97% probability = {p97:.2f} dB")

#cdf plot
plt.figure(figsize=(10, 6))

for label, sir_data in results.items():
    sorted_data = np.sort(sir_data)
    p = np.linspace(0, 1, len(sorted_data), endpoint=False)
    plt.plot(sorted_data, p, label=label)

plt.axvline(THRESHOLD_dB, color='gray', linestyle='--', label='Threshold = -5 dB')
plt.xlabel("SIR [dB]")
plt.ylabel("CDF")
plt.title("CDF of SIR for Reuse 1, 3, and 9")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()