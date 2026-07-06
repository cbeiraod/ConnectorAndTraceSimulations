from typing import List
import logging
import random
import math

logger = logging.getLogger(__name__)

class FDTDMesher1D:
    algorithms = [
        "advancing_front", "segment_uniform", "segment_graded", "global_grid_search",
        "iterative_relaxation", "iterative_relaxation_fast",
        "iterative_relaxation_momentum", "iterative_relaxation_fast_momentum",
        "iterative_relaxation_jacobi", "iterative_relaxation_gaussseidel",
        "iterative_relaxation_symmetricgaussseidel", "iterative_relaxation_redblack"
    ]

    def __init__(self, fixed_points: list[float], optional_points: list[float], max_res: float, ratio: float):
        """
        Initializes the mesher, cleans up the inputs (sorting, removing duplicates,
        filtering out-of-bounds optional points), and sets up the initial state.
        """

        self._max_res = max_res
        self._ratio = ratio
        self.fixed_points = fixed_points
        self._optional_points = sorted([p for p in set(optional_points)
                       if self._min_val < p < self._max_val and p not in self._fixed_points])
        self.mesh = self.fixed_points

    @property
    def max_res(self):
        return self._max_res
    @max_res.setter
    def max_res(self, value: float):
        self._max_res = value
        self._pressure_left, self._pressure_right = self._calculate_cell_ratio_pressures()

    @property
    def ratio(self):
        return self._ratio
    @ratio.setter
    def ratio(self, value: float):
        self._ratio = value
        self._pressure_left, self._pressure_right = self._calculate_cell_ratio_pressures()

    @property
    def optional_points(self):
        return self._optional_points

    @property
    def fixed_points(self):
        return self._fixed_points
    @fixed_points.setter
    def fixed_points(self, value: list[float]):
        if not value or not isinstance(value, list):
            raise ValueError("You must define the fixed points of the mesh as a list")

        fixed = sorted(list(set(value)))
        min_val, max_val = fixed[0], fixed[-1]

        if len(fixed) < 2:
            raise ValueError("At least two fixed points are required to define a domain.")

        self._fixed_points = fixed
        self._min_val = min_val
        self._max_val = max_val
        self.mesh = fixed

    @property
    def mesh(self):
        return self._mesh
    @mesh.setter
    def mesh(self, value: list[float]):
        self._mesh = value

        self.dx = self._get_cell_sizes()
        self._pressure_left, self._pressure_right = self._calculate_cell_ratio_pressures()

    def generate(self, algorithm: str = "advancing_front", **kwargs) -> list[float]:
        """
        The main orchestrator loop.
        """
        if algorithm not in self.algorithms:
            raise RuntimeError(f"Unknown algorithm: {algorithm}")

        if algorithm == "advancing_front":
            return self._advancing_front(**kwargs)
        elif algorithm == "segment_uniform":
            return self._segment_uniform(**kwargs)
        elif algorithm == "segment_graded":
            return self._segment_graded(**kwargs)
        elif algorithm == "global_grid_search":
            return self._global_grid_search(**kwargs)
        elif algorithm == "iterative_relaxation":
            return self._iterative_relaxation(**kwargs)
        elif algorithm == "iterative_relaxation_fast":
            return self._iterative_relaxation_fast(**kwargs)
        elif algorithm == "iterative_relaxation_momentum":
            return self._iterative_relaxation_momentum(**kwargs)
        elif algorithm == "iterative_relaxation_fast_momentum":
            return self._iterative_relaxation_fast_momentum(**kwargs)
        elif algorithm == "iterative_relaxation_jacobi":
            return self._iterative_relaxation_jacobi(**kwargs)
        elif algorithm == "iterative_relaxation_gaussseidel":
            return self._iterative_relaxation_gaussseidel(**kwargs)
        elif algorithm == "iterative_relaxation_alternatinggaussseidel":
            return self._iterative_relaxation_alternatinggaussseidel(**kwargs)
        elif algorithm == "iterative_relaxation_symmetricgaussseidel":
            return self._iterative_relaxation_symmetricgaussseidel(**kwargs)
        elif algorithm == "iterative_relaxation_redblack":
            return self._iterative_relaxation_redblack(**kwargs)
        else:
            return self.mesh

    def _iterative_relaxation(self, max_iterations: int = 20000, relaxation_factor: float = 0.2, snap_to_optional: bool = True, **kwargs) -> list[float]:
        """
        An iterative relaxation (spring) approach that guarantees all fixed points are included.
        1. Starts with the best base grid from `_global_grid_search`.
        2. Force-injects any missing fixed points.
        3. Applies a spring-like relaxation to movable points to diffuse ratio/max_res violations.
        """
        # 1. Get base grid from global_grid_search
        base_mesh = self._global_grid_search(**kwargs)

        # 2. Inject all fixed points (if not already cleanly snapped)
        combined_pts = base_mesh.copy()
        for fp in self.fixed_points:
            if not any(abs(fp - p) < 1e-9 for p in combined_pts):
                combined_pts.append(fp)

        combined_pts.sort()
        self.mesh = combined_pts

        # Tag rigid anchors (fixed points)
        is_fixed = [False] * len(combined_pts)
        for i, p in enumerate(combined_pts):
            if any(abs(p - fp) < 1e-9 for fp in self.fixed_points):
                is_fixed[i] = True

        # 3. Iterative relaxation (Spring Model)
        iters = 0
        stagnation_counter = 0

        while iters < max_iterations:
            iters += 1

            # Calculate current cell sizes
            self.dx = self._get_cell_sizes()

            # Calculate "shrink demand" (stress) for each cell
            shrink_demand = [0.0] * len(self.dx)

            # True mathematical demand
            for j in range(len(self.dx)):
                demand = 0.0

                # Stress from exceeding max_res
                if self.dx[j] > self.max_res:
                    demand += (self.dx[j] - self.max_res)

                # Stress from ratio violation with left neighbor
                if j > 0 and self.dx[j] > self.dx[j-1] * self.ratio:
                    demand += (self.dx[j] - self.dx[j-1] * self.ratio)

                # Stress from ratio violation with right neighbor
                if j < len(self.dx) - 1 and self.dx[j] > self.dx[j+1] * self.ratio:
                    demand += (self.dx[j] - self.dx[j+1] * self.ratio)
                shrink_demand[j] = demand

            max_demand = max(shrink_demand) if shrink_demand else 0.0

            # Smooth Mathematical Exit Condition: Clean exit once within test tolerance
            if max_demand <= 1e-9:
                break

            shifts = [0.0] * len(self.mesh)
            max_shift_applied = 0.0

            # Calculate net force and shift for each internal node
            for i in range(1, len(self.mesh) - 1):
                if is_fixed[i]:
                    continue  # Rigid anchor, cannot move

                # Net force: right cell pushing right (+) minus left cell pushing left (-)
                force = shrink_demand[i] - shrink_demand[i-1]

                if abs(force) > 1e-12:
                    raw_shift = force * relaxation_factor

                    # Prevent points from crossing or creating microscopic slivers
                    max_left = -0.4 * self.dx[i-1]
                    max_right = 0.4 * self.dx[i]

                    shift = max(max_left, min(max_right, raw_shift))
                    shifts[i] = shift
                    max_shift_applied = max(max_shift_applied, abs(shift))

            # Apply shifts simultaneously to prevent asymmetric bias
            for i in range(1, len(self.mesh) - 1):
                self.mesh[i] += shifts[i]

            # --- Topological Insertion (Stagnation Break) ---
            # Evaluate if the points are moving too slowly to resolve the current demand.
            # Using a proportional shift threshold prevents triggering stagnation
            # while the solver is just in a naturally slow asymptotic crawl.
            #if max_shift_applied < 1e-6 or max_shift_applied < max_demand * 0.01:
            if max_demand > 1e-4 and max_shift_applied < max_demand * 1e-4:
                stagnation_counter += 1
            else:
                stagnation_counter = 0

            if stagnation_counter > 50:
                # Find all cells identically tied for max stress (maintains perfect topological symmetry)
                tied_indices = [i for i, d in enumerate(shrink_demand) if max_demand - d < 1e-9]

                # Reverse iteration ensures inserting points doesn't offset downstream indices!
                for idx in reversed(tied_indices):
                    new_pt = (self.mesh[idx] + self.mesh[idx + 1]) / 2.0
                    self.mesh.insert(idx + 1, new_pt)
                    is_fixed.insert(idx + 1, False)
                    logger.debug(f"Iterative Relaxation: Splitting cell {idx}. Inserted pt: {new_pt:.4f}")

                stagnation_counter = 0

        if iters >= max_iterations:
            raise RuntimeError(f"iterative_relaxation failed to converge after {max_iterations} iterations.")
        logger.warn(f"It took {iters} iterations")

        # 4. Optional Pass: Snap relaxed points to optional geometry
        if snap_to_optional:
            for i in range(1, len(self.mesh) - 1):
                if is_fixed[i]:
                    continue

                pt_ll = self.mesh[i-2] if i >= 2 else None
                pt_l = self.mesh[i-1]
                pt_r = self.mesh[i+1]
                pt_rr = self.mesh[i+2] if i <= len(self.mesh) - 3 else None

                snapped = self._evaluate_optional_snap(
                    candidate_pt=self.mesh[i],
                    pt_ll=pt_ll,
                    pt_l=pt_l,
                    pt_r=pt_r,
                    pt_rr=pt_rr,
                    from_left=True,
                    from_right=True
                )
                if snapped != self.mesh[i]:
                    self.mesh[i] = snapped

        return self.mesh

    def _iterative_relaxation_fast(self, max_iterations: int = 20000, relaxation_factor: float = 0.3, snap_to_optional: bool = True, **kwargs) -> list[float]:
        """
        An optimized iterative relaxation approach using a Sequential Update Sweep (Symmetric Gauss-Seidel).
        Unlike simultaneous updates, updating points in place allows ratio shockwaves to diffuse
        across the entire domain in a single iteration. Sweeping bidirectionally preserves mesh symmetry.
        """
        base_mesh = self._global_grid_search(**kwargs)

        combined_pts = base_mesh.copy()
        for fp in self.fixed_points:
            if not any(abs(fp - p) < 1e-9 for p in combined_pts):
                combined_pts.append(fp)

        combined_pts.sort()
        self.mesh = combined_pts

        is_fixed = [False] * len(combined_pts)
        for i, p in enumerate(combined_pts):
            if any(abs(p - fp) < 1e-9 for fp in self.fixed_points):
                is_fixed[i] = True

        iters = 0
        stagnation_counter = 0

        while iters < max_iterations:
            iters += 1

            self.dx = self._get_cell_sizes()
            shrink_demand = [0.0] * len(self.dx)

            for j in range(len(self.dx)):
                demand = 0.0
                if self.dx[j] > self.max_res:
                    demand += (self.dx[j] - self.max_res)
                if j > 0 and self.dx[j] > self.dx[j-1] * self.ratio:
                    demand += (self.dx[j] - self.dx[j-1] * self.ratio)
                if j < len(self.dx) - 1 and self.dx[j] > self.dx[j+1] * self.ratio:
                    demand += (self.dx[j] - self.dx[j+1] * self.ratio)
                shrink_demand[j] = demand

            max_demand = max(shrink_demand) if shrink_demand else 0.0

            # Smooth Mathematical Exit Condition
            if max_demand <= 1e-9:
                break

            max_shift_applied = 0.0

            # Symmetric Gauss-Seidel: Alternate left-to-right and right-to-left sweeps
            sweep_indices = range(1, len(self.mesh) - 1) if iters % 2 == 1 else range(len(self.mesh) - 2, 0, -1)

            for i in sweep_indices:
                if is_fixed[i]: continue

                # Recalculate local cell sizes dynamically (since the left neighbor might have just moved!)
                dx_l = self.mesh[i] - self.mesh[i-1]
                dx_r = self.mesh[i+1] - self.mesh[i]

                # Evaluate demand on the left cell (i-1)
                demand_l = 0.0
                if dx_l > self.max_res:
                    demand_l += (dx_l - self.max_res)
                if i > 1:
                    dx_ll = self.mesh[i-1] - self.mesh[i-2]
                    if dx_l > dx_ll * self.ratio:
                        demand_l += (dx_l - dx_ll * self.ratio)
                if dx_l > dx_r * self.ratio:
                    demand_l += (dx_l - dx_r * self.ratio)

                # Evaluate demand on the right cell (i)
                demand_r = 0.0
                if dx_r > self.max_res:
                    demand_r += (dx_r - self.max_res)
                if dx_r > dx_l * self.ratio:
                    demand_r += (dx_r - dx_l * self.ratio)
                if i < len(self.mesh) - 2:
                    dx_rr = self.mesh[i+2] - self.mesh[i+1]
                    if dx_r > dx_rr * self.ratio:
                        demand_r += (dx_r - dx_rr * self.ratio)

                force = demand_r - demand_l

                if abs(force) > 1e-12:
                    raw_shift = force * relaxation_factor
                    max_left = -0.4 * dx_l
                    max_right = 0.4 * dx_r
                    shift = max(max_left, min(max_right, raw_shift))

                    if abs(shift) > 1e-12:
                        self.mesh[i] += shift
                        max_shift_applied = max(max_shift_applied, abs(shift))

            # --- Topological Insertion (Stagnation Break) ---
            #if max_shift_applied < 1e-6 or max_shift_applied < max_demand * 0.01:
            if max_demand > 1e-4 and max_shift_applied < max_demand * 1e-4:
                stagnation_counter += 1
            else:
                stagnation_counter = 0

            if stagnation_counter > 50:
                tied_indices = [i for i, d in enumerate(shrink_demand) if max_demand - d < 1e-9]

                for idx in reversed(tied_indices):
                    new_pt = (self.mesh[idx] + self.mesh[idx + 1]) / 2.0
                    self.mesh.insert(idx + 1, new_pt)
                    is_fixed.insert(idx + 1, False)
                    logger.debug(f"Iterative Relaxation Fast: Splitting cell {idx}. Inserted pt: {new_pt:.4f}")

                stagnation_counter = 0

        if iters >= max_iterations:
            raise RuntimeError(f"iterative_relaxation_fast failed to converge after {max_iterations} iterations.")
        logger.warn(f"It took {iters} iterations")

        # Optional Pass: Snap relaxed points to optional geometry
        if snap_to_optional:
            for i in range(1, len(self.mesh) - 1):
                if is_fixed[i]: continue

                pt_ll = self.mesh[i-2] if i >= 2 else None
                pt_l = self.mesh[i-1]
                pt_r = self.mesh[i+1]
                pt_rr = self.mesh[i+2] if i <= len(self.mesh) - 3 else None

                snapped = self._evaluate_optional_snap(
                    candidate_pt=self.mesh[i],
                    pt_ll=pt_ll, pt_l=pt_l, pt_r=pt_r, pt_rr=pt_rr,
                    from_left=True, from_right=True
                )
                if snapped != self.mesh[i]:
                    self.mesh[i] = snapped

        return self.mesh

    def _iterative_relaxation_momentum(self, max_iterations: int = 20000, relaxation_factor: float = 0.2, damping: float = 0.8, snap_to_optional: bool = True, **kwargs) -> list[float]:
        """
        A second-order wave equation solver (Mass-Spring-Damper).
        Maintains a velocity for each point to overcome 'Critical Slowing Down',
        allowing the mesh to rapidly coast into equilibrium.
        """
        base_mesh = self._global_grid_search(**kwargs)

        combined_pts = base_mesh.copy()
        for fp in self.fixed_points:
            if not any(abs(fp - p) < 1e-9 for p in combined_pts):
                combined_pts.append(fp)

        combined_pts.sort()
        self.mesh = combined_pts

        is_fixed = [False] * len(combined_pts)
        for i, p in enumerate(combined_pts):
            if any(abs(p - fp) < 1e-9 for fp in self.fixed_points):
                is_fixed[i] = True

        # Initialize momentum trackers
        velocities = [0.0] * len(self.mesh)

        iters = 0
        stagnation_counter = 0

        while iters < max_iterations:
            iters += 1

            self.dx = self._get_cell_sizes()
            shrink_demand = [0.0] * len(self.dx)

            for j in range(len(self.dx)):
                demand = 0.0
                if self.dx[j] > self.max_res:
                    demand += (self.dx[j] - self.max_res)
                if j > 0 and self.dx[j] > self.dx[j-1] * self.ratio:
                    demand += (self.dx[j] - self.dx[j-1] * self.ratio)
                if j < len(self.dx) - 1 and self.dx[j] > self.dx[j+1] * self.ratio:
                    demand += (self.dx[j] - self.dx[j+1] * self.ratio)
                shrink_demand[j] = demand

            max_demand = max(shrink_demand) if shrink_demand else 0.0

            if max_demand <= 1e-9:
                break

            shifts = [0.0] * len(self.mesh)
            max_shift_applied = 0.0

            for i in range(1, len(self.mesh) - 1):
                if is_fixed[i]:
                    continue

                force = shrink_demand[i] - shrink_demand[i-1]

                if abs(force) > 1e-12 or abs(velocities[i]) > 1e-12:
                    # Apply damping to existing momentum, add new acceleration
                    velocities[i] = (velocities[i] * damping) + (force * relaxation_factor)

                    max_left = -0.4 * self.dx[i-1]
                    max_right = 0.4 * self.dx[i]

                    # Restrict actual movement to avoid bounds
                    shift = max(max_left, min(max_right, velocities[i]))

                    # Anti-windup: If we hit a wall, kill the phantom velocity trying to push past it
                    velocities[i] = shift
                    shifts[i] = shift

                    max_shift_applied = max(max_shift_applied, abs(shift))

            for i in range(1, len(self.mesh) - 1):
                self.mesh[i] += shifts[i]

            # --- Topological Insertion (Stagnation Break) ---
            if max_demand > 1e-4 and max_shift_applied < max_demand * 1e-4:
                stagnation_counter += 1
            else:
                stagnation_counter = 0

            if stagnation_counter > 50:
                tied_indices = [i for i, d in enumerate(shrink_demand) if max_demand - d < 1e-9]

                for idx in reversed(tied_indices):
                    new_pt = (self.mesh[idx] + self.mesh[idx + 1]) / 2.0
                    self.mesh.insert(idx + 1, new_pt)
                    is_fixed.insert(idx + 1, False)
                    velocities.insert(idx + 1, 0.0) # Synchronize the momentum array!
                    logger.debug(f"Iterative Relaxation Momentum: Splitting cell {idx}. Inserted pt: {new_pt:.4f}")

                stagnation_counter = 0

        if iters >= max_iterations:
            raise RuntimeError(f"iterative_relaxation_momentum failed to converge after {max_iterations} iterations.")
        logger.warn(f"It took {iters} iterations")

        if snap_to_optional:
            for i in range(1, len(self.mesh) - 1):
                if is_fixed[i]:
                    continue

                pt_ll = self.mesh[i-2] if i >= 2 else None
                pt_l = self.mesh[i-1]
                pt_r = self.mesh[i+1]
                pt_rr = self.mesh[i+2] if i <= len(self.mesh) - 3 else None

                snapped = self._evaluate_optional_snap(
                    candidate_pt=self.mesh[i],
                    pt_ll=pt_ll,
                    pt_l=pt_l,
                    pt_r=pt_r,
                    pt_rr=pt_rr,
                    from_left=True,
                    from_right=True
                )
                if snapped != self.mesh[i]:
                    self.mesh[i] = snapped

        return self.mesh

    def _iterative_relaxation_fast_momentum(self, max_iterations: int = 20000, relaxation_factor: float = 0.3, damping: float = 0.8, snap_to_optional: bool = True, **kwargs) -> list[float]:
        """
        Combines the $O(1)$ shockwave propagation of Symmetric Gauss-Seidel with
        the asymptotic acceleration of Mass-Spring-Damper momentum for ultra-fast convergence.
        """
        base_mesh = self._global_grid_search(**kwargs)

        combined_pts = base_mesh.copy()
        for fp in self.fixed_points:
            if not any(abs(fp - p) < 1e-9 for p in combined_pts):
                combined_pts.append(fp)

        combined_pts.sort()
        self.mesh = combined_pts

        is_fixed = [False] * len(combined_pts)
        for i, p in enumerate(combined_pts):
            if any(abs(p - fp) < 1e-9 for fp in self.fixed_points):
                is_fixed[i] = True

        velocities = [0.0] * len(self.mesh)
        iters = 0
        stagnation_counter = 0

        while iters < max_iterations:
            iters += 1

            self.dx = self._get_cell_sizes()
            shrink_demand = [0.0] * len(self.dx)

            for j in range(len(self.dx)):
                demand = 0.0
                if self.dx[j] > self.max_res:
                    demand += (self.dx[j] - self.max_res)
                if j > 0 and self.dx[j] > self.dx[j-1] * self.ratio:
                    demand += (self.dx[j] - self.dx[j-1] * self.ratio)
                if j < len(self.dx) - 1 and self.dx[j] > self.dx[j+1] * self.ratio:
                    demand += (self.dx[j] - self.dx[j+1] * self.ratio)
                shrink_demand[j] = demand

            max_demand = max(shrink_demand) if shrink_demand else 0.0

            if max_demand <= 1e-9:
                break

            max_shift_applied = 0.0

            sweep_indices = range(1, len(self.mesh) - 1) if iters % 2 == 1 else range(len(self.mesh) - 2, 0, -1)

            for i in sweep_indices:
                if is_fixed[i]: continue

                dx_l = self.mesh[i] - self.mesh[i-1]
                dx_r = self.mesh[i+1] - self.mesh[i]

                demand_l = 0.0
                if dx_l > self.max_res:
                    demand_l += (dx_l - self.max_res)
                if i > 1:
                    dx_ll = self.mesh[i-1] - self.mesh[i-2]
                    if dx_l > dx_ll * self.ratio:
                        demand_l += (dx_l - dx_ll * self.ratio)
                if dx_l > dx_r * self.ratio:
                    demand_l += (dx_l - dx_r * self.ratio)

                demand_r = 0.0
                if dx_r > self.max_res:
                    demand_r += (dx_r - self.max_res)
                if dx_r > dx_l * self.ratio:
                    demand_r += (dx_r - dx_l * self.ratio)
                if i < len(self.mesh) - 2:
                    dx_rr = self.mesh[i+2] - self.mesh[i+1]
                    if dx_r > dx_rr * self.ratio:
                        demand_r += (dx_r - dx_rr * self.ratio)

                force = demand_r - demand_l

                if abs(force) > 1e-12 or abs(velocities[i]) > 1e-12:
                    velocities[i] = (velocities[i] * damping) + (force * relaxation_factor)

                    max_left = -0.4 * dx_l
                    max_right = 0.4 * dx_r
                    shift = max(max_left, min(max_right, velocities[i]))

                    velocities[i] = shift

                    if abs(shift) > 1e-12:
                        self.mesh[i] += shift
                        max_shift_applied = max(max_shift_applied, abs(shift))

            if max_demand > 1e-4 and max_shift_applied < max_demand * 1e-4:
                stagnation_counter += 1
            else:
                stagnation_counter = 0

            if stagnation_counter > 50:
                tied_indices = [i for i, d in enumerate(shrink_demand) if max_demand - d < 1e-9]

                for idx in reversed(tied_indices):
                    new_pt = (self.mesh[idx] + self.mesh[idx + 1]) / 2.0
                    self.mesh.insert(idx + 1, new_pt)
                    is_fixed.insert(idx + 1, False)
                    velocities.insert(idx + 1, 0.0) # Sync momentum
                    logger.debug(f"Iterative Relaxation Fast Momentum: Splitting cell {idx}. Inserted pt: {new_pt:.4f}")

                stagnation_counter = 0

        if iters >= max_iterations:
            raise RuntimeError(f"iterative_relaxation_fast_momentum failed to converge after {max_iterations} iterations.")
        logger.warn(f"It took {iters} iterations")

        if snap_to_optional:
            for i in range(1, len(self.mesh) - 1):
                if is_fixed[i]: continue

                pt_ll = self.mesh[i-2] if i >= 2 else None
                pt_l = self.mesh[i-1]
                pt_r = self.mesh[i+1]
                pt_rr = self.mesh[i+2] if i <= len(self.mesh) - 3 else None

                snapped = self._evaluate_optional_snap(
                    candidate_pt=self.mesh[i],
                    pt_ll=pt_ll, pt_l=pt_l, pt_r=pt_r, pt_rr=pt_rr,
                    from_left=True, from_right=True
                )
                if snapped != self.mesh[i]:
                    self.mesh[i] = snapped

        return self.mesh

    def _iterative_relaxation_jacobi(self, **kwargs) -> list[float]:
        """
        Performs simultaneous Jacobi-style physical grid updates.
        """
        return self._relax_grid_engine(sweep_strategy="jacobi", **kwargs)

    def _iterative_relaxation_gaussseidel(self, **kwargs) -> list[float]:
        """
        Performs sequential, single-direction (left-to-right) Gauss-Seidel physical grid updates.
        """
        return self._relax_grid_engine(sweep_strategy="gaussseidel", **kwargs)

    def _iterative_relaxation_alternatinggaussseidel(self, **kwargs) -> list[float]:
        """
        Performs sequential, alternating single-direction Gauss-Seidel physical grid updates.
        """
        return self._relax_grid_engine(sweep_strategy="alternatinggaussseidel", **kwargs)

    def _iterative_relaxation_symmetricgaussseidel(self, **kwargs) -> list[float]:
        """
        Performs bidirectional (Symmetric) Gauss-Seidel sequential updates.
        """
        return self._relax_grid_engine(sweep_strategy="symmetricgaussseidel", **kwargs)

    def _iterative_relaxation_redblack(self, **kwargs) -> list[float]:
        """
        Performs checkerboard-interleaved Red-Black parallel updates.
        """
        return self._relax_grid_engine(sweep_strategy="redblack", **kwargs)

    def _global_grid_search(self, min_test_cell_ratio: float = 0.6, **kwargs) -> list[float]:
        """
        A global grid search algorithm that evaluates uniform grids of varying resolutions.
        It searches from `min_cells` to `max_cells` to find the base grid that maximizes
        the number of 'compatible' fixed points.

        Compatibility is defined as being able to snap a uniform grid edge to a fixed point
        without violating the max_res or ratio constraints for the 4 surrounding cells
        (the cells immediately left and right of the edge, and their adjacent neighbors).
        """
        domain = self._max_val - self._min_val
        if domain <= 0:
            return self.mesh

        min_cells = max(1, math.ceil(domain / self.max_res))
        max_cells = max(min_cells, math.ceil(domain / (self.max_res * min_test_cell_ratio)))

        best_N = min_cells
        best_score = -1
        best_opt_score = -1
        best_mesh = []
        best_min_dist = float('inf')

        for N in range(min_cells, max_cells + 1):
            step = domain / N
            grid = [self._min_val + i * step for i in range(N + 1)]
            grid[0] = self._min_val
            grid[-1] = self._max_val

            test_mesh = grid.copy()
            score = 0
            opt_score = 0
            total_dist = 0.0

            # Loop over internal grid points sequentially
            for i in range(1, N):
                # Bounds: left is potentially already snapped, right is original grid
                left_bound = test_mesh[i-1]
                right_bound = grid[i+1]

                best_fp = None
                min_dist = float('inf')

                # Find the closest fixed point strictly within the surrounding cells
                for fp in self.fixed_points[1:-1]:
                    if left_bound < fp < right_bound:
                        dist = abs(grid[i] - fp)
                        if dist < min_dist:
                            min_dist = dist
                            best_fp = fp

                best_op = None
                min_op_dist = float('inf')

                # Find the closest optional point strictly within the surrounding cells
                for op in self.optional_points:
                    if left_bound < op < right_bound:
                        dist = abs(grid[i] - op)
                        if dist < min_op_dist:
                            min_op_dist = dist
                            best_op = op

                pt_ll = test_mesh[i-2] if i >= 2 else None
                pt_l = test_mesh[i-1]
                pt_r = right_bound
                pt_rr = grid[i+2] if i <= N - 2 else None

                original_val = test_mesh[i]
                substituted = False

                # 1. Attempt to snap to the closest fixed point
                if best_fp is not None and self._is_point_compatible(best_fp, pt_ll, pt_l, pt_r, pt_rr):
                    test_mesh[i] = best_fp
                    score += 1
                    total_dist += min_dist
                    substituted = True

                # 2. If no valid fixed point, attempt to snap to the closest optional point
                if not substituted and best_op is not None and self._is_point_compatible(best_op, pt_ll, pt_l, pt_r, pt_rr):
                    test_mesh[i] = best_op
                    opt_score += 1
                    substituted = True

                # 3. If neither can be safely snapped, revert to the uniform grid point
                if not substituted:
                    test_mesh[i] = original_val

            # Keep the grid that successfully snapped the most fixed points.
            # Tie-breakers:
            # 1. Highest number of successfully snapped optional points
            # 2. Minimum total snap distance (for fixed points only)
            is_better = False
            if score > best_score:
                is_better = True
            elif score == best_score:
                if opt_score > best_opt_score:
                    is_better = True
                elif opt_score == best_opt_score and total_dist < best_min_dist:
                    is_better = True

            if is_better:
                best_score = score
                best_opt_score = opt_score
                best_N = N
                best_mesh = test_mesh.copy()
                best_min_dist = total_dist

        # Fallback to segment uniform if no valid base grid exists without clashes
        if not best_mesh:
            return self._segment_uniform(**kwargs)

        self.mesh = best_mesh
        return self.mesh

    def _segment_graded(self, max_iterations: int = 2000, snap_to_optional: bool = True, **kwargs) -> list[float]:
        """
        A refinement of segment_uniform that enforces the global ratio constraint.
        It determines the required number of cells for each segment independently,
        then iteratively propagates ratio constraints across segment boundaries by
        increasing the cell counts of neighboring segments until the entire mesh
        satisfies the ratio.
        """
        S = len(self.fixed_points) - 1
        if S == 0:
            return self.mesh

        sizes = [self.fixed_points[i+1] - self.fixed_points[i] for i in range(S)]

        # Initial pass: satisfy max_res
        N = [max(1, math.ceil(size / self.max_res)) for size in sizes]

        changed = True
        iters = 0

        # Iteratively grade the segments to satisfy the ratio constraint
        while changed and iters < max_iterations:
            changed = False
            iters += 1

            # Forward pass
            for i in range(S - 1):
                dx_left = sizes[i] / N[i]
                dx_right = sizes[i+1] / N[i+1]

                if dx_left > dx_right * self.ratio + 1e-9:
                    N[i] = math.ceil(sizes[i] / (dx_right * self.ratio))
                    changed = True
                elif dx_right > dx_left * self.ratio + 1e-9:
                    N[i+1] = math.ceil(sizes[i+1] / (dx_left * self.ratio))
                    changed = True

            # Backward pass (accelerates convergence across the domain)
            for i in range(S - 2, -1, -1):
                dx_left = sizes[i] / N[i]
                dx_right = sizes[i+1] / N[i+1]

                if dx_right > dx_left * self.ratio + 1e-9:
                    N[i+1] = math.ceil(sizes[i+1] / (dx_left * self.ratio))
                    changed = True
                elif dx_left > dx_right * self.ratio + 1e-9:
                    N[i] = math.ceil(sizes[i] / (dx_right * self.ratio))
                    changed = True

        if iters >= max_iterations:
            raise RuntimeError(f"segment_graded failed to converge after {max_iterations} iterations.")

        # Temporarily ensure self.mesh only contains fixed points for _tesselate_mesh_cell reference
        self.mesh = self.fixed_points.copy()

        new_mesh = [self.fixed_points[0]]
        for i in range(S):
            inner_points = self._tesselate_mesh_cell(
                cell_index=i,
                N=N[i],
                snap_opt=False,
            )
            new_mesh.extend(inner_points)
            new_mesh.append(self.fixed_points[i+1])

        if snap_to_optional:
            for idx in range(len(new_mesh)):
                if idx == 0 or idx == len(new_mesh) - 1:
                    continue
                if new_mesh[idx] in self._fixed_points:
                    continue

                pt_ll = new_mesh[idx - 2] if idx > 1 else None
                pt_l  = new_mesh[idx - 1]
                pt_r  = new_mesh[idx + 1]
                pt_rr = new_mesh[idx + 2] if idx < len(new_mesh) - 2 else None

                snapped_pt = self._evaluate_optional_snap(
                    candidate_pt = new_mesh[idx],
                    pt_ll = pt_ll,
                    pt_l = pt_l,
                    pt_r = pt_r,
                    pt_rr = pt_rr,
                    from_left = True,
                    from_right = True
                )

                if snapped_pt != new_mesh[idx]:
                    new_mesh[idx] = snapped_pt

        self.mesh = new_mesh

        return self.mesh

    def _segment_uniform(self, snap_to_optional: bool = True, **kwargs) -> list[float]:
        """
        A robust, baseline segment-by-segment meshing algorithm.
        It isolates the domain between consecutive fixed points and applies
        a uniform subdivision to strictly satisfy max_res.
        Ignores the global ratio constraint, ensuring algorithmic stability.
        Accepts **kwargs to safely swallow unexpected parameters.
        """
        # Temporarily ensure self.mesh only contains fixed points for _tesselate_mesh_cell reference
        self.mesh = self.fixed_points.copy()

        new_mesh = [self.fixed_points[0]]
        for i in range(len(self.fixed_points) - 1):
            size = self.fixed_points[i+1] - self.fixed_points[i]
            N = math.ceil(size / self.max_res)

            inner_points = self._tesselate_mesh_cell(
                cell_index=i,
                N=N,
                snap_opt=snap_to_optional
            )
            new_mesh.extend(inner_points)
            new_mesh.append(self.fixed_points[i+1])

        self.mesh = new_mesh

        return self.mesh

    def _advancing_front(self, max_iterations: int = 2000, **kwargs) -> list[float]:
        """
        An advanced, iterative cascading algorithm (Advancing Front).
        It dynamically propagates ratio constraints across the grid by applying
        'forces' from smaller cells to larger neighbors, splitting them
        geometrically to maintain smooth transitions and minimize dispersion error.
        """
        loop_count = 0
        while True:
            loop_count += 1
            if loop_count > max_iterations:
                raise RuntimeError(
                    f"Mesh generation failed to converge after {max_iterations} iterations. "
                    f"Current mesh size: {len(self.mesh)}. "
                    "This is likely an algorithmic infinite loop edge case triggered by the optimizer."
                )

            target_idx = self._find_target_cell()

            if target_idx is None:
                # Mesh is fully valid!
                break

            # A cell is forced if there is any force applied from either side.
            f_l = self._pressure_left[target_idx]
            f_r = self._pressure_right[target_idx]

            if f_l == 0.0 and f_r == 0.0:
                new_points = self._split_unforced_cell(target_idx)
            else:
                new_points = self._split_forced_cell(target_idx)

            if not new_points:
                raise RuntimeError(
                    f"Splitting logic failed to return new points for cell index {target_idx}."
                    " Check if an unimplemented splitting case was reached."
                    f"\n - Fixed points: {self._fixed_points}"
                    f"\n - Mesh: {self.mesh}"
                )

            # Insert the new points into the mesh.
            # Assigning to self.mesh triggers the setter, updating dx and forces automatically!
            self.mesh = self.mesh[:target_idx + 1] + new_points + self.mesh[target_idx + 1:]

        return self.mesh

    # ===============================================
    # Unified Iterative Relaxation Solver Core Engine
    # ===============================================

    def _relax_grid_engine(
        self,
        sweep_strategy: str = "jacobi",
        update_type: str = "first_order",
        lr_mode: str = "uniform",
        damping_mode: str = "uniform",
        max_iterations: int = 20000,
        relaxation_factor: float = 0.2,
        omega: float = 1.0,
        damping: float = 0.8,
        stiff_gamma: float = 5.0,
        snap_to_optional: bool = True,
        **kwargs
    ) -> list[float]:
        r"""
        The unified physical solver engine integrating spatial sweeps with temporal updates.

        Spatial Sweeps (`sweep_strategy`):
        - "jacobi": Simultaneous updates across all nodes.
        - "gaussseidel": Sequential updates using immediate in-place neighbor states.
        - "alternatinggaussseidel": Same as "gaussseidel" but in alternating directions
        - "symmetricgaussseidel": Sequential forward followed by sequential backward sweeps.
        - "redblack": Alternating simultaneous passes on independent interleaving index sets.

        Temporal Updates (`update_type`):
        - "first_order": Update steps based strictly on raw local forces.
        - "momentum": Newton's Second Law with friction history tracking.
        - "nesterov": Accelerated momentum evaluated at a projected coordinate state.
        - "leapfrog": Symplectic staggered-time discretization conserving Hamiltonian energy, only for Jacobi `sweep_strategy`.

        Local Parameterizations (`damping_mode`):
        - "uniform": Global static learning rates and damping.
        - "adjoint": Curvature-dependent Stiffness-Adjoint scaling for damping or step sizes.
        """
        # 1. Initialize coordinate array and Tag Anchors
        base_mesh = self._global_grid_search(**kwargs)
        combined_pts = base_mesh.copy()
        for fp in self.fixed_points:
            if not any(abs(fp - p) < 1e-9 for p in combined_pts):
                combined_pts.append(fp)
        combined_pts.sort()
        self.mesh = combined_pts

        is_fixed = [False] * len(combined_pts)
        for i, p in enumerate(combined_pts):
            if any(abs(p - fp) < 1e-9 for fp in self.fixed_points):
                is_fixed[i] = True
        # Validation for Leapfrog Physics Requirements
        if update_type == "leapfrog":
            if sweep_strategy != "jacobi":
                raise RuntimeError(f"Leapfrog integration is strictly a simultaneous scheme and cannot be used with sequential sweep strategy: {sweep_strategy}")
            if abs(omega - 1.0) > 1e-9:
                raise ValueError("Leapfrog integration strictly couples spatial steps to temporal velocity. 'omega' must be exactly 1.0.")
            if damping >= 2.0:
                raise ValueError("For Leapfrog, base damping must be strictly less than 2.0 to maintain numerical stability.")
            if (damping_mode == "adjoint" or lr_mode == "adjoint") and not kwargs.get("suppress_leapfrog_warning", False):
                logger.warning("Leapfrog used with 'adjoint' scaling. Kinetic Energy Resets will be applied to maintain stability.")

        # Initialize velocity buffers for second-order methods
        velocities = [0.0] * len(self.mesh)

        # Leapfrog Kickoff Step (Explicitly half-step the velocities before the loop)
        if update_type == "leapfrog":
            initial_forces = self._calculate_node_spring_forces(self.mesh)
            lf_alphas = self._calculate_stiffness_adjoint_learning_rate(self.mesh, relaxation_factor, stiff_gamma) if lr_mode == "adjoint" else [relaxation_factor] * len(self.mesh)
            for i in range(1, len(self.mesh) - 1):
                if not is_fixed[i]:
                    velocities[i] = 0.5 * lf_alphas[i] * initial_forces[i]

        iters = 0
        stagnation_counter = 0

        while iters < max_iterations:
            iters += 1

            self.dx = self._get_cell_sizes()
            shrink_demand = [self._calculate_cell_demand(self.dx, j) for j in range(len(self.dx))]
            max_demand = max(shrink_demand) if shrink_demand else 0.0

            # Exit once mathematically inside tolerance
            if max_demand <= 1e-9:
                break

            max_shift_applied = 0.0

            # ----------------------------------------
            # Jacobi Sweep Strategy (Simultaneous)
            # ----------------------------------------
            if sweep_strategy == "jacobi":
                alphas = self._calculate_stiffness_adjoint_learning_rate(self.mesh, relaxation_factor, stiff_gamma) if lr_mode == "adjoint" else [relaxation_factor] * len(self.mesh)
                betas = self._calculate_stiffness_adjoint_damping(self.mesh, damping, stiff_gamma) if damping_mode == "adjoint" else [damping] * len(self.mesh)

                raw_shifts = [0.0] * len(self.mesh)

                if update_type == "nesterov":
                    # 1. Project entire mesh forward to the lookahead state
                    look_mesh = self.mesh.copy()
                    for i in range(1, len(self.mesh) - 1):
                        if is_fixed[i]: continue
                        look_shift = betas[i] * velocities[i]
                        look_shift_clipped = self._clip_local_elastic_step(self.mesh, i, look_shift)
                        look_mesh[i] += look_shift_clipped

                    # 2. Evaluate global forces at the lookahead state
                    look_forces = self._calculate_node_spring_forces(look_mesh)

                    # 3. Calculate Nesterov velocity and shift
                    for i in range(1, len(self.mesh) - 1):
                        if is_fixed[i]: continue
                        velocities[i] = betas[i] * velocities[i] + alphas[i] * look_forces[i]
                        raw_shifts[i] = velocities[i]
                else:
                    # Standard updates evaluate forces at the current state
                    forces = self._calculate_node_spring_forces(self.mesh)
                    for i in range(1, len(self.mesh) - 1):
                        if is_fixed[i]: continue

                        if update_type == "first_order":
                            raw_shifts[i] = alphas[i] * forces[i]
                        elif update_type == "momentum":
                            velocities[i] = betas[i] * velocities[i] + alphas[i] * forces[i]
                            raw_shifts[i] = velocities[i]
                        elif update_type == "leapfrog":
                            # True Damped Leapfrog (Dynamic Relaxation)
                            lf_beta = (2.0 - betas[i]) / (2.0 + betas[i])
                            lf_alpha = (2.0 * alphas[i]) / (2.0 + betas[i])
                            velocities[i] = lf_beta * velocities[i] + lf_alpha * forces[i]

                            # Adaptive Kinetic Energy Reset (Local velocity truncation)
                            if (damping_mode == "adjoint" or lr_mode == "adjoint") and forces[i] * velocities[i] < 0.0:
                                velocities[i] = 0.0

                            raw_shifts[i] = velocities[i]

                max_shift_applied = max(max_shift_applied, self._apply_global_kinematic_step(
                    self.mesh,
                    velocities,
                    raw_shifts,
                    is_fixed,
                    omega,
                    list(range(1, len(self.mesh) - 1)),
                    update_velocity=(update_type not in ("first_order", "leapfrog"))
                ))

            # ----------------------------------------
            # Gauss-Seidel Sweep Strategy (Sequential L2R or Alternating)
            # ----------------------------------------
            elif sweep_strategy in ["gaussseidel", "alternatinggaussseidel"]:
                is_alternating = (sweep_strategy == "alternatinggaussseidel")
                sweep_indices = range(len(self.mesh) - 2, 0, -1) if (is_alternating and iters % 2 == 0) else range(1, len(self.mesh) - 1)

                for i in sweep_indices:
                    if is_fixed[i]: continue

                    force = self._calculate_local_node_spring_force(self.mesh, i)
                    alpha_i, beta_i = self._get_local_coefficients(i, relaxation_factor, damping, lr_mode, damping_mode, stiff_gamma)

                    raw_shift = 0.0
                    if update_type == "first_order":
                        raw_shift = alpha_i * force
                    elif update_type == "momentum":
                        velocities[i] = beta_i * velocities[i] + alpha_i * force
                        raw_shift = velocities[i]
                    elif update_type == "nesterov":
                        look_shift = beta_i * velocities[i]
                        look_shift_clipped = self._clip_local_elastic_step(self.mesh, i, look_shift)
                        temp_mesh = self.mesh.copy()
                        temp_mesh[i] += look_shift_clipped
                        look_force = self._calculate_local_node_spring_force(temp_mesh, i)
                        velocities[i] = beta_i * velocities[i] + alpha_i * look_force
                        raw_shift = velocities[i]

                    shift_applied = self._apply_local_kinematic_step(
                        self.mesh,
                        velocities,
                        i,
                        raw_shift,
                        omega,
                        update_velocity=(update_type != "first_order")
                    )
                    max_shift_applied = max(max_shift_applied, shift_applied)

            # ----------------------------------------
            # Symmetric Gauss-Seidel Sweep Strategy (Bidirectional Sequential)
            # ----------------------------------------
            elif sweep_strategy == "symmetricgaussseidel":
                # Forward Sweep (Left-to-Right)
                for i in range(1, len(self.mesh) - 1):
                    if is_fixed[i]: continue

                    force = self._calculate_local_node_spring_force(self.mesh, i)
                    alpha_i, beta_i = self._get_local_coefficients(i, relaxation_factor, damping, lr_mode, damping_mode, stiff_gamma)

                    raw_shift = 0.0
                    if update_type == "first_order":
                        raw_shift = alpha_i * force
                    elif update_type == "momentum":
                        velocities[i] = beta_i * velocities[i] + alpha_i * force
                        raw_shift = velocities[i]
                    elif update_type == "nesterov":
                        look_shift = beta_i * velocities[i]
                        look_shift_clipped = self._clip_local_elastic_step(self.mesh, i, look_shift)
                        temp_mesh = self.mesh.copy()
                        temp_mesh[i] += look_shift_clipped
                        look_force = self._calculate_local_node_spring_force(temp_mesh, i)
                        velocities[i] = beta_i * velocities[i] + alpha_i * look_force
                        raw_shift = velocities[i]

                    shift_applied = self._apply_local_kinematic_step(
                        self.mesh,
                        velocities,
                        i,
                        raw_shift,
                        omega,
                        update_velocity=(update_type != "first_order")
                    )
                    max_shift_applied = max(max_shift_applied, shift_applied)

                # Backward Sweep (Right-to-Left)
                for i in range(len(self.mesh) - 2, 0, -1):
                    if is_fixed[i]: continue

                    force = self._calculate_local_node_spring_force(self.mesh, i)
                    alpha_i, beta_i = self._get_local_coefficients(i, relaxation_factor, damping, lr_mode, damping_mode, stiff_gamma)

                    raw_shift = 0.0
                    if update_type == "first_order":
                        raw_shift = alpha_i * force
                    elif update_type == "momentum":
                        velocities[i] = beta_i * velocities[i] + alpha_i * force
                        raw_shift = velocities[i]
                    elif update_type == "nesterov":
                        look_shift = beta_i * velocities[i]
                        look_shift_clipped = self._clip_local_elastic_step(self.mesh, i, look_shift)
                        temp_mesh = self.mesh.copy()
                        temp_mesh[i] += look_shift_clipped
                        look_force = self._calculate_local_node_spring_force(temp_mesh, i)
                        velocities[i] = beta_i * velocities[i] + alpha_i * look_force
                        raw_shift = velocities[i]

                    shift_applied = self._apply_local_kinematic_step(
                        self.mesh,
                        velocities,
                        i,
                        raw_shift,
                        omega,
                        update_velocity=(update_type != "first_order")
                    )
                    max_shift_applied = max(max_shift_applied, shift_applied)

            # ----------------------------------------
            # Red-Black Checkerboard Sweep Strategy
            # ----------------------------------------
            elif sweep_strategy == "redblack":
                # Divide index space into Red and Black subsets
                even_indices = list(range(2, len(self.mesh) - 1, 2))
                odd_indices = list(range(1, len(self.mesh) - 1, 2))

                alphas = self._calculate_stiffness_adjoint_learning_rate(self.mesh, relaxation_factor, stiff_gamma) if lr_mode == "adjoint" else [relaxation_factor] * len(self.mesh)
                betas = self._calculate_stiffness_adjoint_damping(self.mesh, damping, stiff_gamma) if damping_mode == "adjoint" else [damping] * len(self.mesh)

                # Step 1: Red Node Update
                raw_shifts = [0.0] * len(self.mesh)
                for i in even_indices:
                    if is_fixed[i]: continue

                    force = self._calculate_local_node_spring_force(self.mesh, i)

                    if update_type == "first_order":
                        raw_shifts[i] = alphas[i] * force
                    elif update_type == "momentum":
                        velocities[i] = betas[i] * velocities[i] + alphas[i] * force
                        raw_shifts[i] = velocities[i]
                    elif update_type == "nesterov":
                        look_shift = betas[i] * velocities[i]
                        look_shift_clipped = self._clip_local_elastic_step(self.mesh, i, look_shift)
                        temp_mesh = self.mesh.copy()
                        temp_mesh[i] += look_shift_clipped
                        look_force = self._calculate_local_node_spring_force(temp_mesh, i)
                        velocities[i] = betas[i] * velocities[i] + alphas[i] * look_force
                        raw_shifts[i] = velocities[i]

                max_shift_applied = max(max_shift_applied, self._apply_global_kinematic_step(
                    self.mesh,
                    velocities,
                    raw_shifts,
                    is_fixed,
                    omega,
                    even_indices,
                    update_velocity=(update_type != "first_order")
                ))

                # Step 2: Black Node Update
                # Recalculate stiffness/damping coefficients to reflect the updated Red nodes!
                alphas = self._calculate_stiffness_adjoint_learning_rate(self.mesh, relaxation_factor, stiff_gamma) if lr_mode == "adjoint" else [relaxation_factor] * len(self.mesh)
                betas = self._calculate_stiffness_adjoint_damping(self.mesh, damping, stiff_gamma) if damping_mode == "adjoint" else [damping] * len(self.mesh)

                raw_shifts = [0.0] * len(self.mesh)
                for i in odd_indices:
                    if is_fixed[i]: continue

                    force = self._calculate_local_node_spring_force(self.mesh, i)

                    if update_type == "first_order":
                        raw_shifts[i] = alphas[i] * force
                    elif update_type == "momentum":
                        velocities[i] = betas[i] * velocities[i] + alphas[i] * force
                        raw_shifts[i] = velocities[i]
                    elif update_type == "nesterov":
                        look_shift = betas[i] * velocities[i]
                        look_shift_clipped = self._clip_local_elastic_step(self.mesh, i, look_shift)
                        temp_mesh = self.mesh.copy()
                        temp_mesh[i] += look_shift_clipped
                        look_force = self._calculate_local_node_spring_force(temp_mesh, i)
                        velocities[i] = betas[i] * velocities[i] + alphas[i] * look_force
                        raw_shifts[i] = velocities[i]

                max_shift_applied = max(max_shift_applied, self._apply_global_kinematic_step(
                    self.mesh,
                    velocities,
                    raw_shifts,
                    is_fixed,
                    omega,
                    odd_indices,
                    update_velocity=(update_type != "first_order")
                ))
            else:
                raise RuntimeError(f"Unknown sweep strategy: {sweep_strategy}")


            # --- Topological Insertion (Stagnation Break) ---
            if max_demand > 1e-4 and max_shift_applied < max_demand * 1e-4:
                stagnation_counter += 1
            else:
                stagnation_counter = 0

            if stagnation_counter > 50:
                tied_indices = [i for i, d in enumerate(shrink_demand) if max_demand - d < 1e-9]

                for idx in reversed(tied_indices):
                    new_pt = (self.mesh[idx] + self.mesh[idx + 1]) / 2.0
                    self.mesh.insert(idx + 1, new_pt)
                    is_fixed.insert(idx + 1, False)
                    velocities.insert(idx + 1, 0.0) # Synchronize the momentum array!
                    logger.debug(f"Iterative Relaxation: Splitting cell {idx}. Inserted pt: {new_pt:.4f}")

                stagnation_counter = 0

        if iters >= max_iterations:
            raise RuntimeError(f"{sweep_strategy}_{update_type} failed to converge after {max_iterations} iterations.")
        logger.warn(f"It took {iters} iterations")

        # Optional Pass: Snap relaxed points to optional geometry
        if snap_to_optional:
            for i in range(1, len(self.mesh) - 1):
                if is_fixed[i]:
                    continue

                pt_ll = self.mesh[i-2] if i >= 2 else None
                pt_l = self.mesh[i-1]
                pt_r = self.mesh[i+1]
                pt_rr = self.mesh[i+2] if i <= len(self.mesh) - 3 else None

                snapped = self._evaluate_optional_snap(
                    candidate_pt=self.mesh[i],
                    pt_ll=pt_ll,
                    pt_l=pt_l,
                    pt_r=pt_r,
                    pt_rr=pt_rr,
                    from_left=True,
                    from_right=True
                )
                if snapped != self.mesh[i]:
                    self.mesh[i] = snapped

        return self.mesh

    # ============================
    # Stateless Physical Operators
    # ============================

    def _calculate_cell_demand(self, dx: list[float], idx: int) -> float:
        r"""
        Calculates the stress (shrink demand) for cell `idx` given cell sizes `dx`.

        Formulas for demand:
        - Stress from exceeding max_res:
          $$D_{\text{res}} = \max(0, dx[idx] - \text{max}\_res)$$
        - Stress from ratio violation with left neighbor:
          $$D_{\text{left}} = \max(0, dx[idx] - dx[idx-1] \cdot \text{ratio}) \quad \text{if } idx > 0$$
        - Stress from ratio violation with right neighbor:
          $$D_{\text{right}} = \max(0, dx[idx] - dx[idx+1] \cdot \text{ratio}) \quad \text{if } idx < M-1$$
        - Total demand:
          $$D = D_{\text{res}} + D_{\text{left}} + D_{\text{right}}$$
        """
        demand = 0.0
        if dx[idx] > self.max_res:
            demand += (dx[idx] - self.max_res)
        if idx > 0 and dx[idx] > dx[idx-1] * self.ratio:
            demand += (dx[idx] - dx[idx-1] * self.ratio)
        if idx < len(dx) - 1 and dx[idx] > dx[idx+1] * self.ratio:
            demand += (dx[idx] - dx[idx+1] * self.ratio)
        return demand

    def _calculate_node_spring_forces(self, mesh: list[float]) -> list[float]:
        """
        Computes the localized spatial spring forces acting on all nodes.
        $$F_i = D_i - D_{i-1}$$
        """
        dx = [mesh[i+1] - mesh[i] for i in range(len(mesh)-1)]
        demands = [self._calculate_cell_demand(dx, j) for j in range(len(dx))]
        forces = [0.0] * len(mesh)
        for i in range(1, len(mesh) - 1):
            forces[i] = demands[i] - demands[i-1]
        return forces

    def _calculate_local_node_spring_force(self, mesh: list[float], i: int) -> float:
        """
        Computes the localized demand forces acting on node `i` based on local coordinate states.
        $$F_i = D_i - D_{i-1}$$

        Optimized to extract a local cell-width window (max 4 elements) to prevent
        allocating a full-mesh cell-width array on every sequential single-node update.
        """
        # Determine active local cell index boundaries
        start_cell = i - 2 if i >= 2 else i - 1
        end_cell = i + 1 if i <= len(mesh) - 3 else i

        # Calculate only the relevant local cell sizes (at most 4 elements)
        local_dx = [mesh[k+1] - mesh[k] for k in range(start_cell, end_cell + 1)]

        # Map global cell indices to local relative offsets
        idx_l = (i - 1) - start_cell
        idx_r = i - start_cell

        # Compute localized demands
        demand_l = self._calculate_cell_demand(local_dx, idx_l)
        demand_r = self._calculate_cell_demand(local_dx, idx_r)

        return demand_r - demand_l

    def _clip_elastic_steps(self, mesh: list[float], proposed_shifts: list[float], is_fixed: list[bool], safety_factor: float = 0.4) -> list[float]:
        """
        Clips proposed simultaneous node displacements to prevent node crossing or mesh inversion.
        $$-safety\\_factor * dx_{i-1} \\le \\Delta x_i \\le safety\\_factor * dx_i$$
        """
        clipped = [0.0] * len(mesh)
        for i in range(1, len(mesh) - 1):
            if is_fixed[i]: continue
            dx_l = mesh[i] - mesh[i-1]
            dx_r = mesh[i+1] - mesh[i]
            max_left = -safety_factor * dx_l
            max_right = safety_factor * dx_r
            clipped[i] = max(max_left, min(max_right, proposed_shifts[i]))
        return clipped

    def _clip_local_elastic_step(self, mesh: list[float], i: int, proposed_shift: float, safety_factor: float = 0.4) -> float:
        """
        Clips the proposed coordinate update of a single node in-place to prevent cell collapse.
        """
        dx_l = mesh[i] - mesh[i-1]
        dx_r = mesh[i+1] - mesh[i]
        max_left = -safety_factor * dx_l
        max_right = safety_factor * dx_r
        return max(max_left, min(max_right, proposed_shift))

    def _apply_global_kinematic_step(self, mesh: list[float], velocities: list[float], raw_shifts: list[float], is_fixed: list[bool], omega: float, active_indices: list[int], update_velocity: bool = True, safety_factor: float = 0.4) -> float:
        """
        Atomically applies simultaneous coordinate shifts and synchronizes momentum.
        Clips the raw_shifts, optionally commits them to the velocity history array,
        and finally scales the displacement by omega for the coordinate update.
        """
        scaled_shifts = [shift * omega for shift in raw_shifts]
        clipped_shifts = self._clip_elastic_steps(mesh, scaled_shifts, is_fixed, safety_factor)

        max_shift = 0.0

        for i in active_indices:
            if is_fixed[i]: continue

            actual_shift = clipped_shifts[i]

            if update_velocity and abs(clipped_shifts[i] - scaled_shifts[i]) > 1e-12:
                # Sync velocity: divide back by omega to store the unscaled equivalent shift
                velocities[i] = actual_shift / omega if omega != 0 else 0.0

            mesh[i] += actual_shift
            max_shift = max(max_shift, abs(actual_shift))

        return max_shift

    def _apply_local_kinematic_step(self, mesh: list[float], velocities: list[float], i: int, raw_shift: float, omega: float, update_velocity: bool = True, safety_factor: float = 0.4) -> float:
        """
        Atomically applies a single sequential coordinate shift and synchronizes momentum.
        """
        scaled_shift = raw_shift * omega
        actual_shift = self._clip_local_elastic_step(mesh, i, scaled_shift, safety_factor)

        if update_velocity and abs(actual_shift - scaled_shift) > 1e-12:
            # Sync velocity: divide back by omega to store the unscaled equivalent shift
            velocities[i] = actual_shift / omega if omega != 0 else 0.0

        mesh[i] += actual_shift
        return abs(actual_shift)

    def _calculate_stiffness_adjoint_damping(self, mesh: list[float], base_damping: float, stiff_gamma: float) -> list[float]:
        """
        Computes localized Stiffness-Adjoint damping coefficients vector ($\beta_i$) for all nodes.
        $$\\beta_i = \\beta_{\\text{base}} \\cdot \\exp(-\\gamma \\cdot \\kappa_i^2)$$
        Where $\\kappa_i$ represents the local cell ratio mismatch:
        $$\\kappa_i = \\left|\\frac{dx_i}{dx_{i-1}} - 1.0\\right|$$
        """
        dx = [mesh[k+1] - mesh[k] for k in range(len(mesh)-1)]
        betas = [base_damping] * len(mesh)
        for i in range(1, len(mesh) - 1):
            dx_l = dx[i-1]
            dx_r = dx[i]
            kappa = abs((dx_r / dx_l) - 1.0) if dx_l > 1e-12 else 0.0
            betas[i] = base_damping * math.exp(-stiff_gamma * (kappa ** 2))
        return betas

    def _calculate_stiffness_adjoint_learning_rate(self, mesh: list[float], base_lr: float, stiff_gamma: float) -> list[float]:
        """
        Computes localized Stiffness-Adjoint step/learning-rate coefficients vector ($\\alpha_i$) for all nodes.
        $$\\alpha_i = \\alpha_{\\text{base}} \\cdot \\exp(-\\gamma \\cdot \\kappa_i^2)$$
        Where $\\kappa_i$ represents the local cell ratio mismatch:
        $$\\kappa_i = \\left|\\frac{dx_i}{dx_{i-1}} - 1.0\\right|$$
        """
        dx = [mesh[k+1] - mesh[k] for k in range(len(mesh)-1)]
        alphas = [base_lr] * len(mesh)
        for i in range(1, len(mesh) - 1):
            dx_l = dx[i-1]
            dx_r = dx[i]
            kappa = abs((dx_r / dx_l) - 1.0) if dx_l > 1e-12 else 0.0
            alphas[i] = base_lr * math.exp(-stiff_gamma * (kappa ** 2))
        return alphas

    def _get_local_coefficients(self, i: int, base_lr: float, base_damping: float, lr_mode: str, damping_mode: str, stiff_gamma: float) -> tuple[float, float]:
        """
        Computes local learning rate and damping parameters for sequential updates.
        """
        alpha = base_lr
        beta = base_damping

        if lr_mode == "adjoint" or damping_mode == "adjoint":
            dx_l = self.mesh[i] - self.mesh[i-1]
            dx_r = self.mesh[i+1] - self.mesh[i]
            kappa = abs((dx_r / dx_l) - 1.0) if dx_l > 1e-12 else 0.0
            scale = math.exp(-stiff_gamma * (kappa ** 2))

            if lr_mode == "adjoint":
                alpha = base_lr * scale
            if damping_mode == "adjoint":
                beta = base_damping * scale

        return alpha, beta

    # ========================
    # State Inspection Methods
    # ========================

    def _get_cell_sizes(self) -> list[float]:
        """Returns the sizes of all current cells in the mesh."""
        return [self.mesh[i+1] - self.mesh[i] for i in range(len(self.mesh)-1)]

    def _calculate_cell_ratio_pressures(self) -> tuple[list[float], list[float]]:
        """
        Evaluates the mesh and returns two lists (force_left, force_right).
        A force is > 0 if a cell is violating the ratio constraint from its neighbor.
        """
        num_cells = len(self.dx)
        force_left = [0.0] * num_cells
        force_right = [0.0] * num_cells

        # Calculate Forces
        for i in range(num_cells):
            # Force applied from the left neighbor onto cell i
            if i > 0 and self.dx[i-1] < self.max_res - 1e-9: # Account for float imprecision
                # Added 1e-9 tolerance to prevent microscopic float noise from registering as forces
                if self.dx[i] > self.dx[i-1] * self.ratio + 1e-9:
                    force_left[i] = 1.0 / self.dx[i-1]

            # Force applied from the right neighbor onto cell i
            if i < num_cells - 1 and self.dx[i+1] < self.max_res - 1e-9:
                # Added 1e-9 tolerance
                if self.dx[i] > self.dx[i+1] * self.ratio + 1e-9:
                    force_right[i] = 1.0 / self.dx[i+1]

        return force_left, force_right

    def _find_target_cell(self) -> int | None:
        """
        Determines which cell needs to be split next based on maximum force,
        or smallest over-sized cell. Returns None if the mesh is fully valid.
        """
        max_f_l = max(self._pressure_left) if self._pressure_left else 0.0
        max_f_r = max(self._pressure_right) if self._pressure_right else 0.0
        max_f = max(max_f_l, max_f_r)

        target_idx = -1

        if max_f > 0:
            # Find all cells tied for the maximum force
            tied_indices = []
            for i in range(len(self.dx)):
                if abs(self._pressure_left[i] - max_f) < 1e-9 or abs(self._pressure_right[i] - max_f) < 1e-9:
                    tied_indices.append(i)
            # Choose randomly among ties
            target_idx = random.choice(tied_indices)
        else:
            # No force applied, look for smallest cell larger than max_res
            candidates = [(i, d) for i, d in enumerate(self.dx) if d > self.max_res + 1e-9]
            if not candidates:
                # No forces AND no cells > max_res. The mesh is complete!
                return None

            # Find the minimum size among cells larger than max_res
            min_oversize = min(d for i, d in candidates)
            tied_candidates = [i for i, d in candidates if abs(d - min_oversize) < 1e-9]
            target_idx = random.choice(tied_candidates)

        return target_idx


    # =============================
    # Geometric & Splitting Methods
    # =============================

    def _is_point_compatible(self, candidate_pt: float, pt_ll: float | None, pt_l: float | None, pt_r: float | None, pt_rr: float | None, do_max_res: bool = True, do_ratio: bool = True) -> bool:
        """
        Evaluates if placing a point at `candidate_pt` satisfies max_res and ratio constraints
        relative to up to 4 surrounding points (defining up to 4 adjacent cells).
        Points can be None if the mesh is being built incrementally and those bounds don't exist yet.
        """
        dx1 = pt_l - pt_ll if (pt_l is not None and pt_ll is not None) else None
        dx2 = candidate_pt - pt_l if pt_l is not None else None
        dx3 = pt_r - candidate_pt if pt_r is not None else None
        dx4 = pt_rr - pt_r if (pt_r is not None and pt_rr is not None) else None

        compatible = True

        # Check max_res for immediate cells
        if do_max_res:
            if dx2 is not None and dx2 > self.max_res + 1e-9: compatible = False
            if dx3 is not None and dx3 > self.max_res + 1e-9: compatible = False

        # Check ratio cascading constraints across the 4 cells
        if do_ratio:
            if compatible and dx1 is not None and dx2 is not None:
                if dx1 > dx2 * self.ratio + 1e-9 or dx2 > dx1 * self.ratio + 1e-9: compatible = False
            if compatible and dx2 is not None and dx3 is not None:
                if dx2 > dx3 * self.ratio + 1e-9 or dx3 > dx2 * self.ratio + 1e-9: compatible = False
            if compatible and dx3 is not None and dx4 is not None:
                if dx3 > dx4 * self.ratio + 1e-9 or dx4 > dx3 * self.ratio + 1e-9: compatible = False

        return compatible

    def _evaluate_optional_snap(self, candidate_pt: float, pt_ll: float | None, pt_l: float, pt_r: float, pt_rr: float | None, from_left: bool = True, from_right: bool = False) -> float:
        """
        Checks if a candidate point can be safely moved to an optional point
        without violating ratio or max_res constraints.
        """
        logger.debug(f"FDTDMesher1D:_evaluate_optional_snap(candidate_pt={candidate_pt}, pt_ll={pt_ll}, pt_l={pt_l}, pt_r={pt_r}, pt_rr={pt_rr}, from_left={from_left}, from_right={from_right})")
        # Simple tolerance window for now (e.g., +/- 20% of the target step)
        best_opt = candidate_pt
        min_dist = pt_r - pt_l + 1e-9

        optional_points = [(p, abs(p-candidate_pt)) for p in self.optional_points if pt_l < p < pt_r]

        for opt, dist_to_candidate in optional_points:
            if dist_to_candidate <= min_dist:
                min_dist = dist_to_candidate

                if from_left and from_right:
                    if not self._is_point_compatible(
                        opt,
                        pt_ll, pt_l, pt_r, pt_rr
                        ):
                        continue
                elif from_left:
                    if not self._is_point_compatible(
                        opt,
                        pt_ll, pt_l, None, None
                        ):
                        continue
                elif from_right:
                    if not self._is_point_compatible(
                        opt,
                        None, None, pt_r, pt_rr
                        ):
                        continue
                #else:
                #    pass

                # TODO: Mathematical Lookahead Placeholder
                # In the future, we need to check if taking this optional point
                # leaves a remaining gap (next_pt - opt) that is mathematically
                # impossible to close without violating max_res or ratio in the
                # remaining steps. If it is impossible, we should `continue`.

                best_opt = opt

        logger.debug(f"FDTDMesher1D:_evaluate_optional_snap(...) -> best_opt={best_opt}")
        return best_opt

    def _tesselate_mesh_cell(self, cell_index: int, N: int, snap_opt: bool = True, graded_left: bool = False, graded_right: bool = False, include_left: bool = False, include_right: bool = False) -> list[float]:
        """
        Subdivides a specific mesh cell into N smaller cells.

        This method replaces the manual looping previously scattered across the meshing
        algorithms. It generates internal points for the cell and attempts to snap them
        to optional points while strictly respecting max_res and ratio constraints.

        Args:
            cell_index: The index of the cell in self.mesh to split.
            N: The number of sub-cells to create.
            snap_opt: Whether to attempt snapping internal points to optional points.
            graded_left: Placeholder for future geometric grading from the left boundary.
            graded_right: Placeholder for future geometric grading from the right boundary.
            include_left: Include the left cell for the left-left evaluation of the leftmost edge.
            include_rightt: Include the right cell for the right-right evaluation of the rightmost edge.

        Returns:
            A list of the new internal points (excluding the existing cell boundaries).
        """
        if N <= 1:
            return []

        X = self.mesh[cell_index]
        Y = self.mesh[cell_index + 1]

        if graded_left or graded_right:
            raise NotImplementedError("Graded tesselation logic is not yet implemented.")

        target_step = (Y - X) / N
        new_points = [X + k * target_step for k in range(1, N)]

        if snap_opt:
            for idx in range(len(new_points)):
                # Carefully resolve the 4 surrounding points for the constraint check
                # We pull from self.mesh to accurately evaluate edges outside this sub-cell

                # Left-Left (pt_ll):
                if idx > 1:
                    pt_ll = new_points[idx-2]
                elif idx == 1:
                    pt_ll = X
                else: # idx == 0
                    if graded_left or include_left:
                        pt_ll = self.mesh[cell_index-1] if cell_index > 0 else None
                    else:
                        pt_ll = None

                # Left (pt_l):
                pt_l = new_points[idx-1] if idx > 0 else X

                # Right (pt_r):
                pt_r = new_points[idx+1] if idx < len(new_points)-1 else Y

                # Right-Right (pt_rr):
                if idx < len(new_points)-2:
                    pt_rr = new_points[idx+2]
                elif idx == len(new_points)-2:
                    pt_rr = Y
                else: # idx == len(new_points)-1
                    if graded_right or include_right:
                        pt_rr = self.mesh[cell_index+2] if cell_index < len(self.mesh)-2 else None
                    else:
                        pt_rr = None

                snapped_pt = self._evaluate_optional_snap(
                    candidate_pt=new_points[idx],
                    pt_ll=pt_ll,
                    pt_l=pt_l,
                    pt_r=pt_r,
                    pt_rr=pt_rr,
                    from_left=True,
                    from_right=True,
                )

                if snapped_pt != new_points[idx]:
                    new_points[idx] = snapped_pt

        #self.mesh = self.mesh[:target_idx + 1] + new_points + self.mesh[target_idx + 1:]
        return new_points


    def _split_unforced_cell(self, cell_index: int) -> list[float]:
        """
        Case A: Splits a cell that has no forces acting on it but is larger than max_res.
        Returns a list of NEW points to be inserted inside the cell.
        """
        X = self.mesh[cell_index]
        Y = self.mesh[cell_index + 1]
        size = Y - X

        N = math.ceil(size / self.max_res)

        new_points = self._tesselate_mesh_cell(
            cell_index = cell_index,
            N = N,
            snap_opt = True,
            graded_left = False,
            graded_right = False
        )

        return new_points

    def _split_forced_cell(self, cell_index: int) -> list[float]:
        """
        Cases B, C, & D: Handles the geometric progression from one or both sides.
        Returns a list of NEW points to be inserted inside the cell.
        """
        X = self.mesh[cell_index]
        Y = self.mesh[cell_index + 1]

        prev_X = self.mesh[cell_index - 1] if cell_index > 0 else None
        prev_Y = self.mesh[cell_index + 1] if cell_index < len(self.dx) - 1 else None
        cur_X, cur_Y = X, Y
        cur_dl = self.dx[cell_index - 1] if cell_index > 0 else None
        cur_dr = self.dx[cell_index + 1] if cell_index < len(self.dx) - 1 else None

        pts_left = []
        pts_right = []

        # Internal safeguard loop for the advancing front
        for _ in range(1000):
            G = cur_Y - cur_X
            if G <= 1e-9:
                break

            # Re-evaluate forces internally based on the remaining gap
            fl = (1.0 / cur_dl) if (cur_dl is not None and cur_dl < self.max_res - 1e-9 and G > cur_dl * self.ratio + 1e-9) else 0.0
            fr = (1.0 / cur_dr) if (cur_dr is not None and cur_dr < self.max_res - 1e-9 and G > cur_dr * self.ratio + 1e-9) else 0.0

            if fl == 0.0 and fr == 0.0:
                # Forces exhausted, but gap remains. Apply Case A on the remainder.
                #N = math.ceil(G / self.max_res)
                #new_points = self._tesselate_mesh_cell(
                #    cell_index = cell_index,
                #    N = N,
                #    snap_opt = True,
                #    graded_left = False,
                #    graded_right = False
                #)

                N = math.ceil(G / self.max_res)
                if N > 1:
                    step = G / N
                    new_points = [cur_X + k * step for k in range(1, N)]

                    for idx in range(len(new_points)):
                        # Look back to pts_left or prev_X, and look ahead to pts_right or cur_Y
                        # Left-Left (pt_ll)
                        if idx > 1:
                            pt_ll = new_points[idx-2]
                        elif idx == 1:
                            pt_ll = cur_X
                        else:
                            pt_ll = pts_left[-1] if len(pts_left) > 0 else prev_X

                        # Left (pt_l)
                        pt_l = new_points[idx-1] if idx > 0 else cur_X

                        # Right (pt_r)
                        pt_r = new_points[idx+1] if idx < len(new_points)-1 else cur_Y

                        # Right-Right (pt_rr)
                        if idx < len(new_points)-2:
                            pt_rr = new_points[idx+2]
                        elif idx == len(new_points)-2:
                            pt_rr = cur_Y
                        else:
                            pt_rr = pts_right[0] if len(pts_right) > 0 else prev_Y

                        snapped_pt = self._evaluate_optional_snap(
                            candidate_pt=new_points[idx],
                            pt_ll=pt_ll,
                            pt_l=pt_l,
                            pt_r=pt_r,
                            pt_rr=pt_rr,
                            from_left = True,
                            from_right = True,
                        )

                        if snapped_pt != new_points[idx]:
                            new_points[idx] = snapped_pt

                    pts_left.extend(new_points)
                break

            if fl >= fr:
                # Case B/C: Grow from Left
                next_dl = min(cur_dl * self.ratio, self.max_res) if cur_dl else self.max_res

                # Case D: Rollback Check (Sliver prevention)
                if self._check_rollback_condition(G, next_dl):
                    # Step back one point to absorb the sliver!
                    if len(pts_left) > 0:
                        pts_left.pop() # Remove the last greedily placed point
                        cur_X = pts_left[-1] if len(pts_left) > 0 else X
                        G = cur_Y - cur_X

                    # Distribute this combined gap safely
                    N = math.ceil(G / next_dl)
                    if N <= 1: N = 2
                    step = G / N
                    for k in range(1, N):
                        pts_left.append(cur_X + k * step)
                    break
                else:
                    # Proceed with geometric step
                    candidate_pt = cur_X + next_dl
                    snapped_pt = self._evaluate_optional_snap(
                        candidate_pt=candidate_pt,
                        pt_ll=prev_X,
                        pt_l=cur_X,
                        pt_r=cur_Y,
                        pt_rr=None,
                        from_left = True,
                        from_right = False,
                    )
                    prev_X = cur_X
                    cur_X = snapped_pt

                    pts_left.append(cur_X)
                    cur_dl = next_dl

            else:
                # Case B/C: Grow from Right
                next_dr = min(cur_dr * self.ratio, self.max_res) if cur_dr else self.max_res

                # Case D: Rollback Check
                if self._check_rollback_condition(G, next_dr):
                    # Step back one point to absorb the sliver!
                    if len(pts_right) > 0:
                        pts_right.pop(0) # Remove the last greedily placed point (index 0 is front)
                        cur_Y = pts_right[0] if len(pts_right) > 0 else Y
                        G = cur_Y - cur_X

                    N = math.ceil(G / next_dr)
                    if N <= 1: N = 2
                    step = G / N
                    for k in range(1, N):
                        pts_right.insert(0, cur_Y - k * step)
                    break
                else:
                    candidate_pt = cur_Y - next_dr
                    snapped_pt = self._evaluate_optional_snap(
                        candidate_pt=candidate_pt,
                        pt_ll=None,
                        pt_l=cur_X,
                        pt_r=cur_Y,
                        pt_rr=prev_Y,
                        from_left = False,
                        from_right = True,
                    )
                    prev_Y = cur_Y
                    cur_Y = snapped_pt

                    pts_right.insert(0, cur_Y)
                    cur_dr = next_dr

        return pts_left + pts_right

    def _check_rollback_condition(self, remaining_gap: float, next_proposed_step: float) -> bool:
        """
        Evaluates if taking the next geometric step will leave an unsolvable sliver.
        """
        leftover = remaining_gap - next_proposed_step

        # If it fits perfectly (or overshoots by a rounding error), no rollback needed.
        if leftover <= 1e-9:
            return False

        # What is the smallest step we are allowed to take after this one?
        min_next_step = next_proposed_step / self.ratio

        # If the space left is smaller than the minimum allowed next step, it's an uncloseable sliver.
        if leftover < min_next_step - 1e-9:
            return True

        return False