import time
import matplotlib.pyplot as plt

from rf_sim.helper import FDTDMesher1D

def main():
    # Setup a mathematically challenging geometry that requires a lot of iterations
    # to resolve (massive ratio shock in the middle of the domain).
    fixed = [0.0, 10.0, 10.1, 20.0]
    max_res = 0.5
    ratio = 1.1

    # We intentionally restrict max_iterations to a fixed number.
    # This guarantees the solver hits the cap and throws a RuntimeError,
    # ensuring every single run computes EXACTLY this many iterations for a fair comparison.
    max_iters = 3000

    intervals = [1, 2, 5, 10, 25, 50, 100, 250, 500, 1000]
    diag_times_per_iter = []

    print(f"Benchmarking FDTDMesher1D Diagnostics Impact (Fixed {max_iters} iterations)")
    print("-" * 65)

    # ---------------------------------------------------------
    # 1. Measure Baseline (Diagnostics OFF)
    # ---------------------------------------------------------
    mesher = FDTDMesher1D(fixed, [], max_res=max_res, ratio=ratio)

    start_time = time.perf_counter()
    try:
        mesher.generate("iterative_relaxation_jacobi", max_iterations=max_iters, diagnostics=False)
    except RuntimeError:
        pass  # Expected to hit the iteration cap
    end_time = time.perf_counter()

    baseline_time = end_time - start_time
    baseline_tpi = (baseline_time / max_iters) * 1e6  # Microseconds per iter
    print(f"Baseline (Diagnostics OFF): {baseline_time:.4f} s -> {baseline_tpi:.2f} µs/iter")

    # ---------------------------------------------------------
    # 2. Measure with Diagnostics ON at varying intervals
    # ---------------------------------------------------------
    for interval in intervals:
        mesher = FDTDMesher1D(fixed, [], max_res=max_res, ratio=ratio)

        start_time = time.perf_counter()
        try:
            mesher.generate(
                "iterative_relaxation_jacobi",
                max_iterations=max_iters,
                diagnostics=True,
                diagnostic_interval=interval
            )
        except RuntimeError:
            pass
        end_time = time.perf_counter()

        run_time = end_time - start_time
        tpi = (run_time / max_iters) * 1e6 # Microseconds per iter
        diag_times_per_iter.append(tpi)

        print(f"Interval {interval:<4d}:             {run_time:.4f} s -> {tpi:.2f} µs/iter")

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

    print(intervals)
    print(diag_times_per_iter)

if __name__ == '__main__':
    main()