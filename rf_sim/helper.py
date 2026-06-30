from typing import List
import random
import math


class FDTDMesher1D:
    algorithms = ["advancing_front", "segment_uniform"]

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
        self._force_left, self._force_right = self._calculate_forces()

    @property
    def ratio(self):
        return self._ratio
    @ratio.setter
    def ratio(self, value: float):
        self._ratio = value
        self._force_left, self._force_right = self._calculate_forces()

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

    @property
    def mesh(self):
        return self._mesh
    @mesh.setter
    def mesh(self, value: list[float]):
        self._mesh = value

        self.dx = self._get_cell_sizes()
        self._force_left, self._force_right = self._calculate_forces()

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
        #elif algorithm == "simple_uniform":
        #    return self._simple_uniform(**kwargs)
        else:
            return self.mesh

    def _segment_uniform(self, snap_to_optional: bool = True, **kwargs) -> list[float]:
        """
        A robust, baseline segment-by-segment meshing algorithm.
        It isolates the domain between consecutive fixed points and applies
        a uniform subdivision to strictly satisfy max_res.
        Ignores the global ratio constraint, ensuring algorithmic stability.
        Accepts **kwargs to safely swallow unexpected parameters.
        """
        new_mesh = []

        for i in range(len(self.fixed_points) - 1):
            X = self.fixed_points[i]
            Y = self.fixed_points[i+1]

            # Add the left boundary of the segment
            new_mesh.append(X)

            size = Y - X
            N = math.ceil(size / self.max_res)

            if N > 1:
                step = size / N
                for k in range(1, N):
                    candidate_pt = X + k * step

                    if snap_to_optional:
                        prev_bound = new_mesh[-1]
                        next_bound = candidate_pt + step if k < N - 1 else Y
                        snapped_pt = self._evaluate_optional_snap(
                            candidate_pt=candidate_pt,
                            prev_pt=prev_bound,
                            next_pt=next_bound,
                            target_step=step,
                            from_left=True,
                            from_right=True
                        )
                        new_mesh.append(snapped_pt)
                    else:
                        new_mesh.append(candidate_pt)

        # Append the final boundary point
        new_mesh.append(self.fixed_points[-1])

        return new_mesh

    def _simple_uniform(self, min_test_cell_ratio=0.6, **kwargs):
        domain = self._max_val - self._min_val

        min_cells = math.ceil(domain / self.max_res)
        max_cells = math.ceil(domain / (self.max_res * min_test_cell_ratio))

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
            f_l = self._force_left[target_idx]
            f_r = self._force_right[target_idx]

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

    # ========================
    # State Inspection Methods
    # ========================

    def _get_cell_sizes(self) -> list[float]:
        """Returns the sizes of all current cells in the mesh."""
        return [self.mesh[i+1] - self.mesh[i] for i in range(len(self.mesh)-1)]

    def _calculate_forces(self) -> tuple[list[float], list[float]]:
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
        max_f_l = max(self._force_left) if self._force_left else 0.0
        max_f_r = max(self._force_right) if self._force_right else 0.0
        max_f = max(max_f_l, max_f_r)

        target_idx = -1

        if max_f > 0:
            # Find all cells tied for the maximum force
            tied_indices = []
            for i in range(len(self.dx)):
                if abs(self._force_left[i] - max_f) < 1e-9 or abs(self._force_right[i] - max_f) < 1e-9:
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

    def _evaluate_optional_snap(self, candidate_pt: float, prev_pt: float, next_pt: float, target_step: float, from_left: bool = True, from_right: bool = False) -> float:
        """
        Checks if a candidate point can be safely moved to an optional point
        without violating ratio or max_res constraints.
        """
        # Simple tolerance window for now (e.g., +/- 20% of the target step)
        tolerance = 0.2 * target_step
        best_opt = candidate_pt
        min_dist = tolerance + 1e-9

        optional_points = [(p, abs(p-candidate_pt)) for p in self.optional_points if prev_pt < p < next_pt]

        for opt, dist_to_candidate in optional_points:
            if dist_to_candidate <= min_dist:
                dist_prev = opt - prev_pt
                dist_next = next_pt - opt

                if from_left:
                    # Check max_res constraint
                    if dist_prev > self.max_res + 1e-9:
                        continue

                    # Check ratio constraint relative to the ideal target_step
                    if dist_prev > target_step * self.ratio + 1e-9:
                        continue
                    if dist_prev < target_step / self.ratio - 1e-9:
                        continue
                if from_right:
                    # Check max_res constraint
                    if dist_next > self.max_res + 1e-9:
                        continue

                    # Check ratio constraint relative to the ideal target_step
                    if dist_next > target_step * self.ratio + 1e-9:
                        continue
                    if dist_next < target_step / self.ratio - 1e-9:
                        continue

                # TODO: Mathematical Lookahead Placeholder
                # In the future, we need to check if taking this optional point
                # leaves a remaining gap (next_pt - opt) that is mathematically
                # impossible to close without violating max_res or ratio in the
                # remaining steps. If it is impossible, we should `continue`.

                best_opt = opt
                min_dist = dist_to_candidate

        return best_opt

    def _split_unforced_cell(self, cell_index: int) -> list[float]:
        """
        Case A: Splits a cell that has no forces acting on it but is larger than max_res.
        Returns a list of NEW points to be inserted inside the cell.
        """
        X = self.mesh[cell_index]
        Y = self.mesh[cell_index + 1]
        size = Y - X

        N = math.ceil(size / self.max_res)
        if N <= 1:
            return []

        target_step = size / N
        new_points = []

        for k in range(1, N):
            candidate_pt = X + k * target_step
            prev_bound = new_points[-1] if new_points else X

            snapped_pt = self._evaluate_optional_snap(
                candidate_pt=candidate_pt,
                prev_pt=prev_bound,
                next_pt=Y,
                target_step=target_step
            )
            new_points.append(snapped_pt)

        return new_points

    def _split_forced_cell(self, cell_index: int) -> list[float]:
        """
        Cases B, C, & D: Handles the geometric progression from one or both sides.
        Returns a list of NEW points to be inserted inside the cell.
        """
        X = self.mesh[cell_index]
        Y = self.mesh[cell_index + 1]

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
                N = math.ceil(G / self.max_res)
                if N > 1:
                    step = G / N
                    for k in range(1, N):
                        candidate_pt = cur_X + k * step
                        prev_bound = pts_left[-1] if k==1 and pts_left else cur_X

                        snapped_pt = self._evaluate_optional_snap(
                            candidate_pt=candidate_pt,
                            prev_pt=prev_bound,
                            next_pt=cur_Y,
                            target_step=step
                        )
                        pts_left.append(snapped_pt)
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
                        prev_pt=cur_X,
                        next_pt=cur_Y,
                        target_step=next_dl
                    )
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
                        prev_pt=cur_X,
                        next_pt=cur_Y,
                        target_step=next_dr,
                        from_left=False,
                        from_right=True
                    )
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
