import numpy as np
import matplotlib.pyplot as plt
from cellular_network_simulator import CellularNetworkSimulator

# Parameters
NUM_SNAPSHOTS = 2000
REUSE_MODE = "reuse3"
SIR_THRESHOLD_DB = -5.0

ETAS = np.linspace(0.0, 1.0, 21)
PATHLOSS_EXPS = [3.0, 3.8, 4.5]

results = {}

# Loop over path-loss exponents
for nu in PATHLOSS_EXPS:
    print(f"\nPathloss exponent ν = {nu}")

    sim = CellularNetworkSimulator(
        cell_radius=1.0,
        pathloss_exp=nu,
        shadow_sigma_dB=8.0,
        seed=42
    )

    # Find optimal eta
    best_eta, best_cov, etas, coverages = sim.find_best_eta(
        num_snapshots=NUM_SNAPSHOTS,
        reuse_mode=REUSE_MODE,
        threshold_db=SIR_THRESHOLD_DB,
        etas=ETAS
    )

    print(f"Optimal η = {best_eta:.2f}")
    print(f"Coverage = {best_cov:.2f} %")

    # Run SIR simulation with optimal eta
    sir_lin = sim.run_monte_carlo_power_control(
        num_snapshots=NUM_SNAPSHOTS,
        reuse_mode=REUSE_MODE,
        eta=best_eta
    )

    sir_db = sim.linear_to_dB(sir_lin).reshape(-1)

    results[nu] = {
        "eta": best_eta,
        "sir_db": sir_db
    }


# Plot CDFs 
plt.figure(figsize=(10, 6))

for nu in PATHLOSS_EXPS:
    sir_db = np.sort(results[nu]["sir_db"])
    p = np.linspace(0, 1, len(sir_db), endpoint=False)
    plt.plot(sir_db, p, linewidth=2, label=f"ν = {nu}")

plt.axvline(SIR_THRESHOLD_DB, color="gray", linestyle="--", label="Threshold = -5 dB")
plt.axhline(0.97, color="green", linestyle=":", label="97% probability")

plt.xlabel("SIR [dB]")
plt.ylabel("CDF")
plt.title("CDF of SIR for Different Pathloss Exponents (Reuse 3)")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("Figure_3_sir_cdf_pathloss.png", dpi=300)
plt.show()
