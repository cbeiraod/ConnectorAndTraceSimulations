# FDTD Meshing Algorithms: Theoretical Philosophy and Heuristics

## 1. Algorithm Selection Guide

In numerical methods, the "No Free Lunch" theorem dictates that no single algorithm optimally solves every topology. However, the available algorithms can be grouped into tiers based on their mathematical robustness and intended application:

**The Most Robust Iterative Solver: `iterative_relaxation_redblack` (with Nesterov Momentum and Adjoint Scaling)**

* **Why:** This configuration provides an optimal balance of speed and stability. It utilizes the rapid shockwave propagation characteristic of Gauss-Seidel sweeps but eliminates the directional bias (left-to-right skew) because the independent checkerboard sets maintain perfect spatial symmetry. Furthermore, Nesterov momentum provides a "look-ahead" capability to prevent violent coordinate collisions, while Adjoint scaling dynamically increases local viscosity near extreme mesh gradients.
* **Parallelization:** Red-Black sweeps are independent by definition, making this algorithm embarrassingly parallel for potential future C++/CUDA acceleration.

**The Most Robust Deterministic Solver: `segment_graded`**

* **Why:** For applications that require a mathematically pure, perfectly graded cascade between fixed points without simulating physical kinematics (springs/dampers), this algorithm is highly reliable. It operates strictly in $O(N)$ time complexity and is immune to numerical ringing or timeout conditions.

## 2. Deciding Based on the Fixed Points

When analyzing a raw array of fixed points, the routing to an algorithm should be based on the geometric signatures of the domain:

* **Signature A: Symmetrical Devices (e.g., Dipole Antennas, Center-Fed Waveguides)**
  * *Requirement:* The mesh must be perfectly mirrored across its center.
  * *Choice:* `iterative_relaxation_jacobi`, `iterative_relaxation_symmetricgaussseidel`, `iterative_relaxation_redblack`, or `segment_graded`.
  * *Avoid:* Standard `gaussseidel` or `alternatinggaussseidel`. Even with stringent error tolerances ($10^{-9}$), unidirectional sweeps inherently break perfect floating-point symmetry.

* **Signature B: Extreme Topological Shocks (e.g., $0.1$ unit gap next to a $10.0$ unit gap)**
  * *Requirement:* The solver must smoothly handle orders-of-magnitude differences in adjacent cell sizes without infinite looping, potentially injecting new points to break impossible mathematical constraints.
  * *Choice:* Any iterative solver equipped with `damping_mode="adjoint"`.

* **Signature C: Performance-Critical Loops (e.g., inside automated optimization routines)**
  * *Requirement:* Execution speed and absolute convergence guarantees.
  * *Choice:* `segment_graded`.

## 3. The Philosophy of Kinematic Variants

The base algorithm only dictates the *spatial* routing of information (how cells communicate). The variants dictate the *temporal* (kinematic) physics defining how nodes move:

* **`first_order` (Gradient Descent):** The node is treated as having zero mass. It moves exactly proportional to the current local force. While incredibly slow to converge (suffering from Critical Slowing Down), it is unconditionally stable.
* **`momentum` (Mass-Spring-Damper):** The node possesses simulated mass and inertia. It coasts through flat regions of the energy landscape, typically reducing iteration counts by 90%. However, momentum can cause "ringing" (oscillation) when encountering stiff mathematical boundaries.
* **`nesterov`:** An advanced momentum variant where the node calculates its corrective force at its *projected future position*. This acts as a dampening mechanism, significantly reducing the ringing associated with standard momentum.
* **`leapfrog` (Symplectic Integration):** Utilizes a staggered-time discretization scheme that natively conserves Hamiltonian energy in the undamped state. By explicitly separating the evaluation of velocities and coordinates into distinct half-steps, it provides exceptional long-term numerical stability. This variant is exclusively compatible with simultaneous (Jacobi) sweeps and tightly couples spatial and temporal stepping (strictly requiring $\omega = 1.0$). It is highly effective for dynamic relaxation, though its damping parameter must be strictly bounded ($< 2.0$) to avoid non-physical velocity reversals.
* **Adjoint Scaling (Curvature-Dependent Properties):** In a standard simulation, a 100x ratio shock acts as a rigid wall, reflecting kinetic energy and destabilizing the solver. Adjoint modes treat the mesh as a non-linear fluid: where the cell ratio mismatch is high, local physical properties scale exponentially to safely freeze nodes before they break the simulation.
  * **`lr_mode="adjoint"`:** Exponentially reduces the local step size (effectively increasing the node's mass) near shocks. This is primarily useful for preventing nodes from violently overshooting boundaries in highly erratic meshes.
  * **`damping_mode="adjoint"`:** Exponentially increases local friction (viscosity) near shocks. This is highly effective at instantly killing high-frequency ringing and wave reflections caused by extreme size differentials, allowing the rest of the well-behaved mesh to coast efficiently.

## 4. Hyperparameter Heuristics and Valid Ranges

Tuning the iterative engine is mathematically analogous to tuning a physical shock absorber.

### `relaxation_factor` ($\alpha$)

* **What it is:** The inverse-mass of the node, dictating how far it moves given a unit of restorative force.
* **Valid Range:** `(0.0, 0.5]`
* **Heuristic:** A standard starting point is `0.2`. If the solver times out without achieving equilibrium or injecting points, the nodes are too sluggish—increase toward `0.4`. If the mesh explodes or vibrates violently, the system is too elastic—drop toward `0.1`.

### `damping` ($\beta$)

* **What it is:** The momentum retention factor.
* **Valid Range:** `[0.0, 1.0)` for standard updates, `[0.0, 2.0)` for `leapfrog`.
* **Heuristic (Standard):** A value of `0.0` devolves the system into a first-order solver. A value of `0.9` creates a highly elastic system that will ring for tens of thousands of iterations. The optimal zone for critical damping in FDTD meshes generally falls between `0.7` and `0.85`.
* **Heuristic (Leapfrog):** In the Leapfrog transformation, `0.0` yields an undamped system that conserves perfect Hamiltonian energy (never settling). As the value approaches `2.0`, the system approaches absolute critical damping, instantly freezing momentum.

### `omega` ($\omega$ - Successive Over-Relaxation)

* **What it is:** A Successive Over-Relaxation (SOR) modifier that scales the calculated coordinate displacement to artificially accelerate or decelerate convergence, primarily utilized in Gauss-Seidel sweeps.
* **Valid Range:** `[0.5, 1.99]`
* **Heuristic:**
  * **$\omega > 1.0$ (Over-relaxation):** Accelerates convergence by anticipating the gradient trend and overstepping, should only be used for `first_order` variants. Values between `1.2` and `1.5` will drastically reduce iteration counts in highly graded meshes. Values approaching `1.8` or higher risk chaotic instability.
  * **$\omega = 1.0$:** Standard operation; no modification to the calculated step.
  * **$\omega < 1.0$ (Under-relaxation):** Artificially reduces the step size. While this increases the number of iterations required to converge, it is highly useful for stabilizing volatile or chaotic mesh configurations that would otherwise diverge under standard step sizes.

### `lr_gamma` / `damping_gamma` ($\gamma$)

* **What it is:** The sensitivity of the Adjoint system, dictating how aggressively the mesh alters its physics when it detects a ratio shock.
* **Valid Range:** `[1.0, 10.0]`
* **Heuristic:** `5.0` is an excellent default. At $\gamma=5$, a 100% ratio error (e.g., cell $A$ is exactly twice the size of cell $B$) drops the local momentum retention by a factor of $e^{-5} \approx 0.006$, instantly neutralizing the local kinetic energy. A lower value like `2.0` provides a softer safety net, while `10.0` treats even minor ratio violations as concrete walls.

## 5. Standardized Tuning Profiles

When implementing fallback sequences (e.g., for automated `Optuna` loops), the following standardized hyperparameter configurations have been rigorously proven to target specific geometric edge cases.

### 5.1 Red-Black Standard Nesterov (The "Goldilocks" Planar Tune)

* **Target:** X/Y planar axes, symmetric multi-cavity resonators, mild topological shocks (e.g., < 100x).
* **Performance:** High stability. Often converges in \~200-600 iterations on standard meshes.
* **Configuration:**
  ```json
  {
      "algorithm": "iterative_relaxation_redblack",
      "update_type": "nesterov",
      "lr_mode": "uniform",
      "damping_mode": "adjoint",
      "damping": 0.8,
      "relaxation_factor": 0.2,
      "max_iterations": 50000
  }
  ```

### 5.2 Red-Black Aggressive Nesterov (High-Ratio Shock Resolver)

* **Target:** Z-Axis stackups, massive voids next to microscopic traces (e.g., 500x shocks).
* **Performance:** Softens the adjoint brakes ($\gamma=2.0$) and lowers node mass to allow nodes to glide smoothly across large distances without freezing prematurely.
* **Configuration:**
  ```json
  {
      "algorithm": "iterative_relaxation_redblack",
      "update_type": "nesterov",
      "lr_mode": "uniform",
      "damping_mode": "adjoint",
      "damping": 0.9,
      "relaxation_factor": 0.35,
      "damping_gamma": 2.0,
      "max_iterations": 50000
  }
  ```

### 5.3 Red-Black Rigid Boundary Nesterov (Deep Freeze)

* **Target:** Meshes containing mid-sized rigid constraints (like internal fixed PML components) that cause violent ringing, but where point injection is not necessarily the primary bottleneck.
* **Performance:** Unconditionally kills local step sizes and kinetic energy near boundaries to forcibly settle extreme oscillations.
* **Configuration:**
  ```json
  {
      "algorithm": "iterative_relaxation_redblack",
      "update_type": "nesterov",
      "lr_mode": "adjoint",
      "damping_mode": "adjoint",
      "max_iterations": 50000,
      "relaxation_factor": 0.3,
      "damping": 0.85,
      "lr_gamma": 4.0,
      "damping_gamma": 4.0
  }
  ```

### 5.4 SGS SSOR Planar Crawler (The Point-Injection & Sloshing Resolver)

* **Target:** Highly constrained boundaries next to massive voids that *strictly require point injection*.
* **Performance:** Strips away all inertia to prevent global momentum "sloshing", allowing the stagnation counter to reliably trigger. Uses true symmetric sweeps to prevent asymmetric error clustering. Maximizes gradient descent speed via SOR ($\omega=1.5$).
* **Configuration:**
  ```json
  {
      "algorithm": "iterative_relaxation_symmetricgaussseidel",
      "update_type": "first_order",
      "lr_mode": "uniform",
      "damping_mode": "uniform",
      "relaxation_factor": 0.25,
      "omega": 1.5,
      "max_iterations": 100000
  }
  ```

### 5.5 Red-Black Over-Constrained Momentum (Micro-Cavity Friction)

* **Target:** Highly over-constrained geometries with tight, rigid micro-cavities (often caused by random optimizer generation or excessive manual fixed points).
* **Performance:** Disables Nesterov's look-ahead (which causes pinball-like bouncing in tiny boxes) in favor of standard momentum, paired with heavy friction ($\beta=0.6$) and strong adjoint brakes.
* **Configuration:**
  ```json
  {
      "algorithm": "iterative_relaxation_redblack",
      "update_type": "momentum",
      "lr_mode": "uniform",
      "damping_mode": "adjoint",
      "damping": 0.6,
      "relaxation_factor": 0.25,
      "damping_gamma": 4.0,
      "max_iterations": 50000
  }
  ```

### 5.6 Red-Black First-Order Adjoint (The Slow Crawler Ultimate Failsafe)

* **Target:** The final failsafe prior to executing `segment_graded`.
* **Performance:** Unconditionally stable. Zero inertia, ensuring no ringing, paired with Adjoint learning rates to prevent boundary crossing. Requires massive iteration counts but will slowly brute-force virtually any geometry to equilibrium.
* **Configuration:**
  ```json
  {
      "algorithm": "iterative_relaxation_redblack",
      "update_type": "first_order",
      "lr_mode": "adjoint",
      "relaxation_factor": 0.2,
      "max_iterations": 200000
  }
  ```