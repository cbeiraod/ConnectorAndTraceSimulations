import time
import matplotlib.pyplot as plt
import gc
import statistics

from rf_sim.helper import FDTDMesher1D

def main():
    # Setup a mathematically challenging geometry
    fixed = [0.0, 10.0, 10.1, 20.0]
    max_res = 0.5
    ratio = 1.1

    max_iters = 5000  # Increased slightly to give the timer more substance
    trials = 10       # Increased to 10 for better statistical deviation bounds

    intervals = [1, 2, 5, 10, 25, 50, 100, 250, 500, 1000]
    diag_min_tpi = []
    diag_mean_tpi = []
    diag_stdev_tpi = []

    print(f"Benchmarking FDTDMesher1D Diagnostics Impact")
    print(f"Parameters: {max_iters} iterations | {trials} trials per config")
    print("-" * 65)

    def run_benchmark_trial(diagnostics_on: bool, interval: int = 50) -> float:
        """Runs a single timed trial with GC disabled to prevent random noise."""
        mesher = FDTDMesher1D(fixed, [], max_res=max_res, ratio=ratio)

        gc.disable() # Prevent Python from pausing to clean memory during the timer
        start_time = time.perf_counter()
        try:
            mesher.generate(
                "iterative_relaxation_jacobi",
                max_iterations=max_iters,
                diagnostics=diagnostics_on,
                diagnostic_interval=interval
            )
        except RuntimeError:
            pass
        end_time = time.perf_counter()
        gc.enable()  # Re-enable immediately after

        return end_time - start_time

    # ---------------------------------------------------------
    # 0. CPU Warm-up (Get the processor out of low-power idle states)
    # ---------------------------------------------------------
    print("Warming up CPU (Turbo Boost initialization)...")
    mesher = FDTDMesher1D(fixed, [], max_res=max_res, ratio=ratio)
    try:
        mesher.generate("iterative_relaxation_jacobi", max_iterations=2000, diagnostics=False)
    except RuntimeError:
        pass
    print("-" * 65)

    # ---------------------------------------------------------
    # 1. Measure Baseline (Diagnostics OFF)
    # ---------------------------------------------------------
    baseline_trial_times = [run_benchmark_trial(False) for _ in range(trials)]

    # Convert all trial times to microseconds per iteration
    baseline_tpi_list = [(t / max_iters) * 1e6 for t in baseline_trial_times]

    baseline_min = min(baseline_tpi_list)
    baseline_mean = statistics.mean(baseline_tpi_list)
    baseline_stdev = statistics.stdev(baseline_tpi_list)

    print(f"Baseline (Diagnostics OFF): Min={baseline_min:.2f} | Mean={baseline_mean:.2f} ± {baseline_stdev:.2f} µs/iter")

    # ---------------------------------------------------------
    # 2. Measure with Diagnostics ON at varying intervals
    # ---------------------------------------------------------
    for interval in intervals:
        interval_trial_times = [run_benchmark_trial(True, interval) for _ in range(trials)]

        tpi_list = [(t / max_iters) * 1e6 for t in interval_trial_times]

        best_tpi = min(tpi_list)
        mean_tpi = statistics.mean(tpi_list)
        stdev_tpi = statistics.stdev(tpi_list)

        diag_min_tpi.append(best_tpi)
        diag_mean_tpi.append(mean_tpi)
        diag_stdev_tpi.append(stdev_tpi)

        print(f"Interval {interval:<4d}:             Min={best_tpi:.2f} | Mean={mean_tpi:.2f} ± {stdev_tpi:.2f} µs/iter")

    # ---------------------------------------------------------
    # 3. Plotting the Results
    # ---------------------------------------------------------
    plt.figure(figsize=(10, 6))

    # Plot the interval data (Diagnostics ON)
    plt.plot(intervals, diag_min_tpi, marker='o', linestyle='-', color='#1f77b4', label='Diagnostics Min (Best Trial)')
    plt.plot(intervals, diag_mean_tpi, marker='s', linestyle='--', color='#ff7f0e', label='Diagnostics Mean')

    # Uncertainty band for Diagnostics ON
    lower_bound = [m - s for m, s in zip(diag_mean_tpi, diag_stdev_tpi)]
    upper_bound = [m + s for m, s in zip(diag_mean_tpi, diag_stdev_tpi)]
    plt.fill_between(intervals, lower_bound, upper_bound, color='#ff7f0e', alpha=0.2, label='Diagnostics ±1 StdDev')

    # Plot the baseline (Diagnostics OFF)
    plt.axhline(y=baseline_min, color='#d62728', linestyle='-', linewidth=2, label='Baseline Min')
    plt.axhline(y=baseline_mean, color='#d62728', linestyle=':', linewidth=2, label='Baseline Mean')
    plt.axhspan(baseline_mean - baseline_stdev, baseline_mean + baseline_stdev, color='#d62728', alpha=0.15, label='Baseline ±1 StdDev')

    plt.title('Performance Impact of FDTD Solver Telemetry', fontsize=14, pad=15)
    plt.xlabel('Diagnostic Interval (Iterations between samples)', fontsize=12)
    plt.ylabel('Time per Iteration (µs)', fontsize=12)

    # A logarithmic X-axis helps visualize the massive spread between interval=1 and interval=1000
    plt.xscale('log')

    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend(fontsize=12)
    plt.tight_layout()

    # Save and display
    plt.savefig("diagnostic_impact.png", dpi=300, bbox_inches='tight')
    print("-" * 65)
    print("Plot saved as 'diagnostic_impact.png'. Displaying plot...")
    plt.show()

if __name__ == '__main__':
    main()