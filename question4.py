import numpy as np
import matplotlib.pyplot as plt
from cellular_network_simulator import CellularNetworkSimulator

#parameters
NUM_SNAPSHOTS = 5000
PATHLOSS_EXP = 3.8
BANDWIDTH_MHz = 100  
SNR_GAP_dB = 4 
REUSE_FACTORS = [1, 3, 9]

# initialize simulator without power control
sim = CellularNetworkSimulator(
    cell_radius=1.0, pathloss_exp=PATHLOSS_EXP, shadow_sigma_dB=8.0, seed=42)

results = {}

# Simulate for each reuse factor
for reuse in REUSE_FACTORS:
    reuse_mode = f"reuse{reuse}"

    # Bandwidth per sector (in MHz)
    bw_per_sector_MHz = BANDWIDTH_MHz / reuse

    print(f"\nReuse factor: {reuse}")
    print(f"Bandwidth per sector: {bw_per_sector_MHz:.2f} MHz")

    # SIR samples (linear)
    sir_linear = sim.run_monte_carlo(
        num_snapshots=NUM_SNAPSHOTS,
        reuse_mode=reuse_mode
    ).ravel()

    # Convert SNR gap to linear
    snr_gap_linear = 10 ** (SNR_GAP_dB / 10.0)

    # Throughput in Mbit/s: C = B(MHz) * log2(1 + SIR / Γ)
    throughput_Mbps = bw_per_sector_MHz * np.log2(1.0 + sir_linear / snr_gap_linear)

    # Statistics
    avg_throughput = np.mean(throughput_Mbps)          # Mbit/s
    p97_throughput = np.percentile(throughput_Mbps, 3) # “97% of users” Mbit/s

    results[reuse] = {
        "bandwidth": bw_per_sector_MHz,          # for your label
        "throughput": throughput_Mbps,           # for the CDF
        "avg_throughput": avg_throughput,
        "percentile_97": p97_throughput,
    }

    print(f"Average throughput: {avg_throughput:.2f} Mbps")
    print(f"Throughput achieved by 97% of users: {p97_throughput:.2f} Mbps")

# Throughput CDF plot
plt.figure(figsize=(12, 7))

colors = {'1': 'blue', '3': 'orange', '9': 'green'}
for reuse in REUSE_FACTORS:
    throughput_sorted = np.sort(results[reuse]['throughput'])
    cdf = np.linspace(0, 1, len(throughput_sorted), endpoint=True)
    plt.plot(
        throughput_sorted,
        cdf,
        linewidth=2.5,
        label=f'Reuse {reuse} (BW = {results[reuse]["bandwidth"]:.1f} MHz)',
        color=colors[str(reuse)]
    )

# Reference line for 97%
plt.axhline(0.97, color='red', linestyle='--', linewidth=1.5, 
            alpha=0.7, label='97% de usuarios')

plt.xlabel('Throughput [Mbps]', fontsize=12)
plt.ylabel('CDF', fontsize=12)
plt.title('Throughput CDF for Reuse Factors 1, 3, and 9\n' + 
          f'(ν = {PATHLOSS_EXP}, BW total = {BANDWIDTH_MHz} MHz, SNR gap = {SNR_GAP_dB} dB)',
          fontsize=13)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=10, loc='lower right')
plt.xlim(left=0)  
plt.ylim(0, 1.05)  
plt.tight_layout()
plt.savefig('Figure_4_throughput_cdf.png', dpi=300, bbox_inches='tight')
plt.show()



# Summary table of results
print("\n" + "=" * 80)
print("Table of Throughput Results")
print("=" * 80)
print(f"{'Reuse Factor':<15} {'BW (MHz)':<12} {'Average Throughput (Mbps)':<30} {'Throughput 97% (Mbps)':<25}")
print("-" * 80)

for reuse in REUSE_FACTORS:
    r = results[reuse]
    print(f"{reuse:<15} {r['bandwidth']:<12.2f} {r['avg_throughput']:<30.2f} {r['percentile_97']:<25.2f}")

