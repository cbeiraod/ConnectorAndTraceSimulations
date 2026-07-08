import time
import matplotlib.pyplot as plt
import gc

from rf_sim.helper import FDTDMesher1D

def main():
    # Setup a mathematically challenging geometry
    fixed = [0.0, 10.0, 10.1, 20.0]
    max_res = 0.5
    ratio = 1.1

    max_iters = 5000  # Increased slightly to give the timer more substance
    trials = 5        # Number of times to repeat each test

    intervals = [1, 2, 5, 10, 25, 50, 100, 250, 500, 1000]
    diag_times_per_iter = []

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

    # We take the MINIMUM time. The OS/GC can only slow things down, never speed them up.
    best_baseline_time = min(baseline_trial_times)
    baseline_tpi = (best_baseline_time / max_iters) * 1e6

    print(f"Baseline (Diagnostics OFF): {best_baseline_time:.4f} s -> {baseline_tpi:.2f} µs/iter (Best of {trials})")

    # ---------------------------------------------------------
    # 2. Measure with Diagnostics ON at varying intervals
    # ---------------------------------------------------------
    for interval in intervals:
        interval_trial_times = [run_benchmark_trial(True, interval) for _ in range(trials)]

        best_run_time = min(interval_trial_times)
        tpi = (best_run_time / max_iters) * 1e6
        diag_times_per_iter.append(tpi)

        print(f"Interval {interval:<4d}:             {best_run_time:.4f} s -> {tpi:.2f} µs/iter (Best of {trials})")

    # ---------------------------------------------------------
    # 3. Plotting the Results
    # ---------------------------------------------------------
    plt.figure(figsize=(10, 6))

    # Plot the interval data
    plt.plot(intervals, diag_times_per_iter, marker='o', linestyle='-', color='#1f77b4', label='Diagnostics ON')

    # Plot the baseline as a red dotted line
    plt.axhline(y=baseline_tpi, color='#d62728', linestyle='--', linewidth=2, label='Baseline (Diagnostics OFF)')

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

    print(f"Baseline: {baseline_tpi}")
    print(f"Intervals: {intervals}")
    print(f"Times: {diag_times_per_iter}")

if __name__ == '__main__':
    main()