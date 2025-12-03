"""
Main script to run all simulations and answers the 4 questions:
1. SIR for different reuse factors
2. Optimal power control exponent
3. Effect of path loss exponent
4. Throughput analysis
"""

import numpy as np
import matplotlib.pyplot as plt
from cellular_network_simulator import CellularNetworkSimulator


def question_1():
    """
    Question 1: Analyze SIR for different reuse factors (1, 3, 9), without power control
    """

    # Initialize simulator
    sim = CellularNetworkSimulator(
        path_loss_exponent=3.8,
        shadow_fading_std_db=8.0,
        cell_radius=1.0,
        num_snapshots=5000
    )

    # Simulate for reuse factors 1, 3, and 9
    reuse_factors = [1, 3, 9]
    sir_results = {}

    for rf in reuse_factors:
        sir_db = sim.simulate_sir(reuse_factor=rf, power_control_exponent=0)
        sir_results[f'Reuse Factor {rf}'] = sir_db

        # Calculate percentage of users with SIR >= -5 dB
        percentage = np.sum(sir_db >= -5) / len(sir_db) * 100
        print(f"  Percentage of users with SIR >= -5 dB: {percentage:.2f}%")

        # Calculate SIR at 97th percentile
        sir_97 = np.percentile(sir_db, 3)  # 3rd percentile = worst 3%
        print(f"  SIR at 97% probability (3rd percentile): {sir_97:.2f} dB")

    # Plot CDFs
    sim.plot_cdf(
        sir_results,
        xlabel='SIR (dB)',
        ylabel='CDF',
        title='SIR CDF for Different Reuse Factors (No Power Control)',
        threshold=-5,
        save_filename='question1_sir_cdf.png'
    )

    return sir_results


def question_2():
    """
    Question 2: Find optimal power control exponent for reuse factor 3 to maximize percentage of users with SIR >= -5 dB
    """

    # Initialize simulator
    sim = CellularNetworkSimulator(
        path_loss_exponent=3.8,
        shadow_fading_std_db=8.0,
        cell_radius=1.0,
        num_snapshots=5000
    )

    # Test different power control exponents
    alpha_values = np.linspace(0, 1, 21)  # all values from 0 to 1 in steps of 0.05 => generates 21 values 
    percentages = []

    for alpha in alpha_values:
        sir_db = sim.simulate_sir(reuse_factor=3, power_control_exponent=alpha)
        percentage = np.sum(sir_db >= -5) / len(sir_db) * 100
        percentages.append(percentage)
        if alpha % 0.2 < 0.051:  # Print every 0.2
            print(f"  alpha = {alpha:.2f}: {percentage:.2f}% users with SIR >= -5 dB")

    # Find optimal alpha
    optimal_idx = np.argmax(percentages)
    optimal_alpha = alpha_values[optimal_idx]
    optimal_percentage = percentages[optimal_idx]

    print(f"\nOptimal power control exponent: {optimal_alpha:.2f}")
    print(f"Maximum percentage of users with SIR >= -5 dB: {optimal_percentage:.2f}%")

    # Plot percentage vs alpha
    plt.figure(figsize=(10, 6))
    plt.plot(alpha_values, percentages, 'b-', linewidth=2)
    plt.plot(optimal_alpha, optimal_percentage, 'ro', markersize=10, 
             label=f'Optimal: α={optimal_alpha:.2f}, {optimal_percentage:.2f}%')
    plt.xlabel('Power Control Exponent (α)', fontsize=12)
    plt.ylabel('Percentage of Users with SIR ≥ -5 dB (%)', fontsize=12)
    plt.title('Optimization of Power Control Exponent (Reuse Factor 3)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig('question2_optimization.png', dpi=300, bbox_inches='tight')
    print("Plot saved as question2_optimization.png")
    plt.show()

    # Simulate with optimal alpha and plot CDF
    print(f"\nSimulating with optimal alpha = {optimal_alpha:.2f}...")
    sir_db_optimal = sim.simulate_sir(reuse_factor=3, power_control_exponent=optimal_alpha)

    sir_results = {f'Reuse Factor 3, α={optimal_alpha:.2f}': sir_db_optimal}
    sim.plot_cdf(
        sir_results,
        xlabel='SIR (dB)',
        ylabel='CDF',
        title=f'SIR CDF with Optimal Power Control (α={optimal_alpha:.2f})',
        threshold=-5,
        save_filename='question2_sir_cdf.png'
    )

    return optimal_alpha, sir_db_optimal


def question_3(optimal_alpha):
    """
    Question 3: Analyze effect of path loss exponent on SIR
    Compare ν = 3.0, 3.8, and 4.5
    """
    print("\n" + "="*70)
    print("QUESTION 3: Effect of Path Loss Exponent")
    print("="*70)

    path_loss_exponents = [3.0, 3.8, 4.5]
    sir_results = {}
    optimal_alphas = {}

    for nu in path_loss_exponents:
        print(f"\n--- Path Loss Exponent ν = {nu} ---")

        # Initialize simulator with this path loss exponent
        sim = CellularNetworkSimulator(
            path_loss_exponent=nu,
            shadow_fading_std_db=8.0,
            cell_radius=1.0,
            num_snapshots=5000
        )

        # Find optimal alpha for this nu
        alpha_values = np.linspace(0, 1, 21)
        percentages = []

        for alpha in alpha_values:
            sir_db = sim.simulate_sir(reuse_factor=3, power_control_exponent=alpha)
            percentage = np.sum(sir_db >= -5) / len(sir_db) * 100
            percentages.append(percentage)

        # Find optimal alpha
        optimal_idx = np.argmax(percentages)
        alpha_opt = alpha_values[optimal_idx]
        optimal_alphas[nu] = alpha_opt

        print(f"Optimal power control exponent: {alpha_opt:.2f}")
        print(f"Maximum percentage: {percentages[optimal_idx]:.2f}%")

        # Simulate with optimal alpha
        sir_db_opt = sim.simulate_sir(reuse_factor=3, power_control_exponent=alpha_opt)
        sir_results[f'ν={nu}, α={alpha_opt:.2f}'] = sir_db_opt

        # Calculate statistics
        mean_sir = np.mean(sir_db_opt)
        median_sir = np.median(sir_db_opt)
        sir_97 = np.percentile(sir_db_opt, 3)
        print(f"Mean SIR: {mean_sir:.2f} dB")
        print(f"Median SIR: {median_sir:.2f} dB")
        print(f"SIR at 97% probability: {sir_97:.2f} dB")

    # Plot CDFs
    sim.plot_cdf(
        sir_results,
        xlabel='SIR (dB)',
        ylabel='CDF',
        title='SIR CDF for Different Path Loss Exponents (Optimal Power Control)',
        threshold=-5,
        save_filename='question3_sir_cdf.png'
    )

    print("\n--- Discussion ---")
    print("A larger path loss exponent is preferable in interference-limited conditions.")
    print("With higher ν, signals attenuate faster with distance, which means:")
    print("  - Interfering signals from distant cells are weaker")
    print("  - The ratio of desired signal to interference improves")
    print("  - Higher SIR values are achieved")
    print("\nThe system designer can affect the exponent by:")
    print("  - Operating frequency: higher frequencies have larger exponents")
    print("  - Environment: urban areas (more obstacles) have larger exponents")
    print("  - Antenna height: lower antennas increase the exponent")

    return sir_results, optimal_alphas


def question_4():
    """
    Question 4: Throughput analysis for different reuse factors
    Bandwidth = 100 MHz, SNR gap = 4 dB, no power control
    """
    print("\n" + "="*70)
    print("QUESTION 4: Throughput Analysis")
    print("="*70)

    # Initialize simulator
    sim = CellularNetworkSimulator(
        path_loss_exponent=3.8,
        shadow_fading_std_db=8.0,
        cell_radius=1.0,
        num_snapshots=5000
    )

    total_bandwidth = 100  # MHz
    snr_gap_db = 4.0
    reuse_factors = [1, 3, 9]
    throughput_results = {}

    print(f"\nTotal bandwidth: {total_bandwidth} MHz")
    print(f"SNR gap to capacity: {snr_gap_db} dB")

    for rf in reuse_factors:
        print(f"\n--- Reuse Factor {rf} ---")

        # Calculate available bandwidth per sector
        bandwidth_per_sector = total_bandwidth / rf
        print(f"Bandwidth per sector: {bandwidth_per_sector:.2f} MHz")

        # Simulate SIR
        sir_db = sim.simulate_sir(reuse_factor=rf, power_control_exponent=0)
        sir_linear = 10 ** (sir_db / 10)

        # Calculate throughput
        throughput = sim.calculate_throughput(sir_linear, bandwidth_per_sector, snr_gap_db)
        throughput_results[f'Reuse Factor {rf}'] = throughput

        # Calculate statistics
        mean_throughput = np.mean(throughput)
        throughput_97 = np.percentile(throughput, 3)  # Worst 3%

        print(f"Average bit rate: {mean_throughput:.2f} Mbps")
        print(f"Bit rate attained by 97% of users: {throughput_97:.2f} Mbps")

    # Plot CDFs
    sim.plot_cdf(
        throughput_results,
        xlabel='Throughput (Mbps)',
        ylabel='CDF',
        title='Throughput CDF for Different Reuse Factors',
        xlim=(0, max([np.max(t) for t in throughput_results.values()]) * 1.1),
        save_filename='question4_throughput_cdf.png'
    )

    return throughput_results


def main():
    """
    Main function to run all simulations
    """
    # We set a random seed for reproducibility 
    # Ensures that all simulations give exactly the same results each time
    np.random.seed(42)

    # Run all questions
    sir_q1 = question_1()
    optimal_alpha, sir_q2 = question_2()
    sir_q3, alphas_q3 = question_3(optimal_alpha)
    throughput_q4 = question_4()

    print("ALL SIMULATIONS COMPLETED SUCCESSFULLY")
    print("\nGenerated files:")
    print("  - question1_sir_cdf.png")
    print("  - question2_optimization.png")
    print("  - question2_sir_cdf.png")
    print("  - question3_sir_cdf.png")
    print("  - question4_throughput_cdf.png")

if __name__ == "__main__":
    main()
