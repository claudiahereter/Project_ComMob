import numpy as np
import matplotlib.pyplot as plt
from cellular_network_simulator import CellularNetworkSimulator

print("\n" + "="*70)
print("QUESTION 3: Effect of Path Loss Exponent")
print("="*70)

NUM_SNAPSHOTS = 5000
THRESHOLD_dB = -5
REUSE_MODE = "reuse3"

path_loss_exponents = [3.0, 3.8, 4.5]
sir_results = {}
optimal_alphas = {}

for nu in path_loss_exponents:
    print(f"\n--- Path Loss Exponent ν = {nu} ---")
    
    # Initialize simulator with this path loss exponent
    sim = CellularNetworkSimulator(cell_radius=1.0, pathloss_exp=nu, shadow_sigma_dB=8.0, seed=123)
    
    # Find optimal alpha for this nu
    alpha_values = np.linspace(0, 1, 21)
    percentages = []
    
    for alpha in alpha_values:
        sir_linear = sim.run_monte_carlo_power_control(NUM_SNAPSHOTS, reuse_mode=REUSE_MODE, alpha=alpha).ravel()
        sir_dB = sim.linear_to_dB(sir_linear)
        percentage = 100 * np.mean(sir_dB >= THRESHOLD_dB)
        percentages.append(percentage)
    
    # Find optimal alpha
    optimal_idx = np.argmax(percentages)
    alpha_opt = alpha_values[optimal_idx]
    optimal_alphas[nu] = alpha_opt
    
    print(f"Optimal power control exponent: α = {alpha_opt:.2f}")
    print(f"Maximum percentage: {percentages[optimal_idx]:.2f}%")
    
    # Simulate with optimal alpha
    sir_linear_opt = sim.run_monte_carlo_power_control(NUM_SNAPSHOTS, reuse_mode=REUSE_MODE, alpha=alpha_opt).ravel()
    sir_dB_opt = sim.linear_to_dB(sir_linear_opt)
    
    sir_results[f'ν={nu}, α={alpha_opt:.2f}'] = sir_dB_opt
    
    # Calculate statistics
    mean_sir = np.mean(sir_dB_opt)
    median_sir = np.median(sir_dB_opt)
    sir_97 = np.percentile(sir_dB_opt, 97)
    
    print(f"Mean SIR: {mean_sir:.2f} dB")
    print(f"Median SIR: {median_sir:.2f} dB")
    print(f"SIR at 97% probability: {sir_97:.2f} dB")

# Plot CDFs for all three path loss exponents
plt.figure(figsize=(10, 6))

for label, sir_data in sir_results.items():
    sorted_data = np.sort(sir_data)
    p = np.linspace(0, 1, len(sorted_data), endpoint=False)
    plt.plot(sorted_data, p, linewidth=2, label=label)

plt.axvline(THRESHOLD_dB, color='gray', linestyle='--', label='Threshold = -5 dB')
plt.xlabel('SIR [dB]', fontsize=12)
plt.ylabel('CDF', fontsize=12)
plt.title('SIR CDF for Different Path Loss Exponents (Optimal Power Control)', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=10)
plt.xlim(-30, 40)
plt.tight_layout()
plt.savefig('question3_sir_cdf.png', dpi=300)
plt.show()

# Discussion
print("\n" + "="*70)
print("DISCUSSION")
print("="*70)
print("\nEffect of path loss exponent in interference-limited conditions:")
print("\n1. LARGER exponent is PREFERABLE:")
print("   - Higher ν means signals attenuate faster with distance")
print("   - Interfering signals from distant cells become weaker")
print("   - Ratio of desired signal to interference improves")
print("   - Users achieve higher SIR values\n")

print("2. COMPARISON:")
for nu in path_loss_exponents:
    label = [k for k in sir_results.keys() if f'ν={nu}' in k][0]
    sir_data = sir_results[label]
    pct = 100 * np.mean(sir_data >= THRESHOLD_dB)
    mean = np.mean(sir_data)
    print(f"   ν = {nu}: Mean SIR = {mean:.2f} dB, P(SIR ≥ -5 dB) = {pct:.2f}%")

print("\n3. HOW SYSTEM DESIGNER CAN AFFECT THE EXPONENT:")
print("   a) Operating Frequency:")
print("      - Higher frequencies (mmWave) → larger exponent (faster attenuation)")
print("      - Lower frequencies (sub-6 GHz) → smaller exponent")
print("\n   b) Environment Type:")
print("      - Urban with many obstacles → larger exponent (ν ≈ 4.0-4.5)")
print("      - Suburban with fewer obstacles → medium exponent (ν ≈ 3.5-4.0)")
print("      - Rural open space → smaller exponent (ν ≈ 2.5-3.5)")
print("\n   c) Antenna Height:")
print("      - Lower antennas → larger exponent (more ground effects)")
print("      - Higher antennas → smaller exponent (more line-of-sight)")
print("\n   d) Propagation Model:")
print("      - 3GPP models, ITU-R models, etc. define different exponents")
print("      - Can be measured/calibrated through field tests")

print("\n" + "="*70)
print("SUMMARY: ν=4.5 is best for interference-limited uplink systems.")
print("="*70)