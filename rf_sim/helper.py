from typing import List
import random
import math


class FDTDMesher1D:
    def __init__(self, fixed_points: list[float], optional_points: list[float], max_res: float, ratio: float):
        """
        Initializes the mesher, cleans up the inputs (sorting, removing duplicates,
        filtering out-of-bounds optional points), and sets up the initial state.
        """
        # 1. Preprocess inputs
        if not fixed_points or not isinstance(fixed_points, list):
            raise ValueError("You must define the fixed points of the mesh as a list")

        fixed = sorted(list(set(fixed_points)))
        min_val, max_val = fixed[0], fixed[-1]

        if len(fixed) < 2:
            raise ValueError("At least two fixed points are required to define a domain.")

        self._fixed_points = fixed
        self._min_val = min_val
        self._max_val = max_val
        self._optional_points = sorted([p for p in set(optional_points)
                       if min_val < p < max_val and p not in fixed])

        self._max_res = max_res
        self._ratio = ratio

        self.mesh = fixed

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
    def mesh(self):
        return self._mesh
    @mesh.setter
    def mesh(self, value: list[float]):
        self._mesh = value
        self.dx = self._get_cell_sizes()
        self._force_left, self._force_right = self._calculate_forces()

    def generate(self, max_iterations: int = 2000) -> list[float]:
        """
        The main orchestrator loop.
        Calls the helper methods below until no forces exist and no cells > max_res.
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
                if self.dx[i] > self.dx[i-1] * self.ratio:
                    force_left[i] = 1.0 / self.dx[i-1]

            # Force applied from the right neighbor onto cell i
            if i < num_cells - 1 and self.dx[i+1] < self.max_res - 1e-9:
                if self.dx[i] > self.dx[i+1] * self.ratio:
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
                        prev_bound = pts_left[-1] if k==1 else cur_X

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
                    # Rollback! Apply modified Case A on the current gap using the force neighbor size
                    N = math.ceil(G / next_dl)
                    if N <= 1: N = 2 # Guarantee a split if we hit the sliver condition
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

# Fixed points: [0.0, 1.0, 6.1, 7.1]; max_res: 1.0; ratio: 1.1
# Mesh Attemp 1: [0.0, 0.5, 1.0, 1.425, 1.85, 2.2750000000000004, 2.7, 3.125, 3.55, 3.975, 4.4, 4.825, 5.25, 5.675, 6.1, 6.33375, 6.4506250000000005, 6.509062500000001, 6.5675, 6.600781250000001, 6.634062500000001, 6.667343750000001, 6.7006250000000005, 6.737234375000001, 6.757369531250001, 6.767437109375001, 6.772470898437501, 6.772836683368119, 6.773202468298739, 6.7736306175263294, 6.774019844096866, 6.7743736864337185, 6.774695361285402, 6.7749877929687505, 6.775253639953613, 6.775519486938475, 6.775709827087402, 6.775900167236328, 6.7760732037353515, 6.776246240234375, 6.7764035461425784, 6.776560852050782, 6.776718157958985, 6.776875463867188, 6.77703276977539, 6.777190075683594, 6.777347381591797, 6.777404758921814, 6.777457102462769, 6.777480894981385, 6.7775046875000005, 6.777526317062378, 6.777547946624757, 6.777569576187135, 6.777591205749513, 6.77761283531189, 6.777634464874268, 6.7776560944366455, 6.777677723999024, 6.777711021499634, 6.777744319000245, 6.777810914001465, 6.7778775090026855, 6.777944104003907, 6.778383520507813, 6.779262353515625, 6.78102001953125, 6.7845353515625, 6.791566015625, 6.80562734375, 6.819688671875, 6.83375, 6.8492174609375, 6.864684921875, 6.8987133359375, 6.93614459140625, 6.977318972421875, 7.022610791539062, 7.061305395769531, 7.1]
# Mesh Attemp 2: [0.0, 1.0, 1.85, 2.7, 3.55, 4.4, 5.25, 5.4625, 5.675, 5.887499999999999, 5.940624999999999, 5.9937499999999995, 6.046875, 6.0734375, 6.077607421874999, 6.081777343749999, 6.084248046874999, 6.086718749999999, 6.0889648437499995, 6.0912109375, 6.09345703125, 6.095703125, 6.0978515625, 6.1, 6.101953125, 6.10390625, 6.105859375, 6.1078125, 6.109765625, 6.11171875, 6.113671875, 6.115625, 6.117578125, 6.11953125, 6.121484375, 6.1234375, 6.125390625, 6.12734375, 6.129296875, 6.13125, 6.1333984374999995, 6.135546874999999, 6.1376953125, 6.13984375, 6.14220703125, 6.1445703125, 6.149296875, 6.155898437499999, 6.1625, 6.225, 6.2875, 6.35, 6.475, 6.6, 6.725, 6.85, 7.1]
# Mesh Attemp 3: [0.0, 0.25, 0.5, 0.75, 0.875, 0.9375, 0.96875, 0.984375, 1.0, 1.01328125, 1.0265625, 1.0398437500000002, 1.053125, 1.067734375, 1.0757695312499997, 1.0797871093749998, 1.0817958984374998, 1.08280029296875, 1.0830513916015625, 1.0831769409179688, 1.0833024902343749, 1.083428039550781, 1.0835535888671872, 1.0835849761962888, 1.0836163635253904, 1.0836320571899412, 1.083647750854492, 1.0836634445190427, 1.0836791381835935, 1.0836948318481443, 1.083710525512695, 1.0837262191772459, 1.0837419128417967, 1.0837497596740722, 1.0837576065063474, 1.0837654533386227, 1.0837733001708982, 1.0837811470031737, 1.0837831558412792, 1.0837851646793846, 1.0837871704277993, 1.083788993835449, 1.083790651478767, 1.0837923091220851, 1.0837946827888485, 1.0837968406677243, 1.0837988023757932, 1.083800764083862, 1.083802725791931, 1.0838046874999998, 1.083807427406311, 1.083810167312622, 1.0838129072189329, 1.083815647125244, 1.083826606750488, 1.083837566375732, 1.0838485260009763, 1.0838594856262205, 1.0838704452514647, 1.083892364501953, 1.083980041503906, 1.0841553955078123, 1.0845061035156247, 1.0852075195312498, 1.0866103515624999, 1.0880131835937499, 1.089416015625, 1.09502734375, 1.10625, 1.118594921875, 1.1321743359374998, 1.14577466796875, 1.1593749999999998, 1.2125, 1.265625, 1.31875, 1.425, 1.85, 2.2750000000000004, 2.7, 3.125, 3.55, 3.975, 4.4, 4.825, 5.25, 6.1, 7.1]
# Mesh Attemp 4: [0.0, 1.0, 1.85, 2.7, 3.55, 4.4, 5.25, 5.4625, 5.56875, 5.621874999999999, 5.675, 5.728125, 5.78125, 5.834375, 5.887499999999999, 5.940624999999999, 5.9937499999999995, 6.046875, 6.053515625, 6.06015625, 6.066796875, 6.0734375, 6.080078125, 6.086718749999999, 6.0933593749999995, 6.1, 6.10390625, 6.1078125, 6.11171875, 6.115625, 6.1177734374999995, 6.119921874999999, 6.122285156249999, 6.124648437499999, 6.1279492187499995, 6.13125, 6.1625, 6.225, 6.35, 6.475, 6.6, 6.725, 6.85, 7.1]
# Mesh Attemp 5: [0.0, 0.25, 0.5, 0.75, 0.8125, 0.875, 0.8864493349048828, 0.8971053435214843, 0.9067926240820312, 0.9155992427734375, 0.923605259765625, 0.93088345703125, 0.9375, 0.9435150390625, 0.949530078125, 0.95959765625, 0.96875, 0.9770703125, 0.9853906250000001, 1.0, 1.01328125, 1.0265625, 1.0398437500000002, 1.053125, 1.06640625, 1.0796875, 1.0929687499999998, 1.10625, 1.120859375, 1.13546875, 1.1515390625, 1.167609375, 1.1900546875, 1.2125, 1.2371898437499997, 1.2643486718749997, 1.2942233828124996, 1.3270855648437494, 1.3632339650781242, 1.394116982539062, 1.425, 1.85, 2.7, 3.55, 4.4, 5.25, 6.1, 7.1]


def generate_fdtd_mesh_1d(
    fixed_points: List[float],
    optional_points: List[float],
    max_res: float,
    ratio: float
) -> List[float]:
    """
    Generates a 1D FDTD mesh starting from fixed_points, snapping to optional_points
    where possible, and strictly enforcing max_res and cell grading ratios.
    Uses an analytical sub-mesh generator to avoid Zeno's paradox and infinite loops.
    """
    if not fixed_points:
        return []

    # 1. Sort and clean fixed points
    fixed = sorted(list(set(fixed_points)))
    if len(fixed) < 2:
        return fixed

    min_val, max_val = fixed[0], fixed[-1]

    # Filter optional points: only keep those inside the boundary
    optional = sorted([p for p in set(optional_points) if min_val < p < max_val and p not in fixed])
    mesh = list(fixed)

    def snap_to_optional(base_pt: float, max_safe_step: float, prev_dx: float, direction: int, ideal_step: float):
        """
        Attempts to snap an analytically generated point to an optional point,
        ensuring it strictly obeys the mathematical limits of the geometric progression.
        """
        best_opt = None
        best_diff = float('inf')

        for p_opt in optional:
            if p_opt <= min_val + 1e-9 or p_opt >= max_val - 1e-9:
                continue

            step_opt = abs(p_opt - base_pt)

            if direction == 1 and p_opt <= base_pt + 1e-9: continue
            if direction == -1 and p_opt >= base_pt - 1e-9: continue

            # Must not exceed absolute safety limits dictated by max_res and ratio
            if step_opt > max_safe_step + 1e-9: continue
            if prev_dx > step_opt * ratio + 1e-9: continue

            diff = abs(step_opt - ideal_step)

            # Tie-breaking logic: prefer the closest to ideal_step. If equidistant, prefer larger step.
            if diff < best_diff - 1e-9 or (abs(diff - best_diff) <= 1e-9 and step_opt > (abs(best_opt - base_pt) if best_opt else -1)):
                best_diff = diff
                best_opt = p_opt

        return best_opt, abs(best_opt - base_pt) if best_opt else -1.0

    def analytical_fill_gap(idx: int, step_L: float, step_R: float):
        """
        Replaces a single violating gap with a complete, mathematically perfect sub-mesh.
        """
        left_pt = mesh[idx]
        right_pt = mesh[idx+1]

        # Physical cells adjacent to this gap
        prev_sz_L = dx[idx-1] if idx > 0 else max_res
        prev_sz_R = dx[idx+1] if idx < len(dx)-1 else max_res

        new_pts_L = []
        new_pts_R = []

        curr_L = left_pt
        curr_R = right_pt
        curr_step_L = step_L
        curr_step_R = step_R

        u_opts = []
        last_step_dir = -1  # Used to ensure perfect symmetry in the middle

        while True:
            gap = curr_R - curr_L
            if gap <= 1e-9:
                break

            # 1. Try Uniform Fill
            N_ceil = math.ceil(gap / max_res) if max_res > 0 else 1
            N_floor = math.floor(gap / max_res) if max_res > 0 else 1
            if N_floor < 1: N_floor = 1

            u_ceil = gap / N_ceil
            u_floor = gap / N_floor

            fits_ceil = (u_ceil <= curr_step_L + 1e-9) and (u_ceil <= curr_step_R + 1e-9)
            fits_floor = (u_floor <= curr_step_L + 1e-9) and (u_floor <= curr_step_R + 1e-9) and (u_floor <= max_res + 1e-9)

            N_fill = 0
            if fits_floor:
                N_fill = N_floor
            elif fits_ceil:
                N_fill = N_ceil

            if N_fill > 0:
                rem_cells = N_fill
                while rem_cells > 1:
                    u_fill = (curr_R - curr_L) / rem_cells
                    ideal_pt = curr_L + u_fill

                    opt_pt, _ = snap_to_optional(curr_L, max_safe_step=curr_step_L, prev_dx=prev_sz_L, direction=1, ideal_step=u_fill)
                    if opt_pt is not None and opt_pt < curr_R - 1e-9:
                        curr_L = opt_pt
                        u_opts.append(opt_pt)
                    else:
                        curr_L = ideal_pt

                    new_pts_L.append(curr_L)
                    prev_sz_L = curr_L - (new_pts_L[-2] if len(new_pts_L)>1 else left_pt)
                    curr_step_L = min(prev_sz_L * ratio, max_res)
                    rem_cells -= 1
                break

            # 2. Geometric Grading (steps simultaneously inwards from the tightest constraint)
            step_from_left = False
            if curr_step_L < curr_step_R - 1e-9:
                step_from_left = True
            elif curr_step_R < curr_step_L - 1e-9:
                step_from_left = False
            else:
                # Perfectly symmetric limits -> alternate to maintain perfect spatial symmetry!
                step_from_left = (last_step_dir == -1)

            if step_from_left:
                last_step_dir = 1
                ideal_pt = curr_L + curr_step_L
                opt_pt, _ = snap_to_optional(curr_L, max_safe_step=curr_step_L, prev_dx=prev_sz_L, direction=1, ideal_step=curr_step_L)

                actual_pt = opt_pt if (opt_pt is not None and opt_pt < curr_R - 1e-9) else ideal_pt
                actual_step = actual_pt - curr_L

                # Sliver prevention fallback
                rem_gap = curr_R - actual_pt
                if rem_gap < actual_step / ratio - 1e-9 or rem_gap < curr_step_R / ratio - 1e-9:
                    actual_pt = curr_L + gap / 2.0
                    opt_pt, _ = snap_to_optional(curr_L, max_safe_step=gap, prev_dx=prev_sz_L, direction=1, ideal_step=gap / 2.0)
                    if opt_pt is not None and opt_pt < curr_R - 1e-9: actual_pt = opt_pt

                if opt_pt == actual_pt and opt_pt is not None: u_opts.append(opt_pt)

                new_pts_L.append(actual_pt)
                prev_sz_L = actual_pt - curr_L
                curr_L = actual_pt
                curr_step_L = min(prev_sz_L * ratio, max_res)
            else:
                last_step_dir = -1
                ideal_pt = curr_R - curr_step_R
                opt_pt, _ = snap_to_optional(curr_R, max_safe_step=curr_step_R, prev_dx=prev_sz_R, direction=-1, ideal_step=curr_step_R)

                actual_pt = opt_pt if (opt_pt is not None and opt_pt > curr_L + 1e-9) else ideal_pt
                actual_step = curr_R - actual_pt

                rem_gap = actual_pt - curr_L
                if rem_gap < actual_step / ratio - 1e-9 or rem_gap < curr_step_L / ratio - 1e-9:
                    actual_pt = curr_R - gap / 2.0
                    opt_pt, _ = snap_to_optional(curr_R, max_safe_step=gap, prev_dx=prev_sz_R, direction=-1, ideal_step=gap / 2.0)
                    if opt_pt is not None and opt_pt > curr_L + 1e-9: actual_pt = opt_pt

                if opt_pt == actual_pt and opt_pt is not None: u_opts.append(opt_pt)

                new_pts_R.append(actual_pt)
                prev_sz_R = curr_R - actual_pt
                curr_R = actual_pt
                curr_step_R = min(prev_sz_R * ratio, max_res)

        return new_pts_L + list(reversed(new_pts_R)), u_opts

    loop_count = 0
    max_loops = 2000

    while True:
        loop_count += 1
        if loop_count > max_loops:
            raise RuntimeError(
                f"Mesh generation failed to converge after {max_loops} iterations. "
                f"Current mesh size: {len(mesh)}. "
                "This is likely an algorithmic infinite loop edge case triggered by the optimizer."
            )

        dx = [mesh[i+1] - mesh[i] for i in range(len(mesh)-1)]
        forces = []

        # Identify all rule violations
        for i in range(len(dx)):
            L = dx[i]
            if i > 0 and L > dx[i-1] * ratio + 1e-9: forces.append((dx[i-1], i))
            if i < len(dx)-1 and L > dx[i+1] * ratio + 1e-9: forces.append((dx[i+1], i))
            if L > max_res + 1e-9: forces.append((0, i))

        if not forces:
            break

        # Process the gap(s) driven by the tightest constraint simultaneously
        min_force_val = min(f[0] for f in forces)
        active_cells = set(f[1] for f in forces if abs(f[0] - min_force_val) < 1e-9)

        new_points = []
        opts_to_remove = []

        for i in active_cells:
            step_L = dx[i-1] * ratio if i > 0 else max_res
            step_R = dx[i+1] * ratio if i < len(dx)-1 else max_res

            step_L = min(step_L, max_res)
            step_R = min(step_R, max_res)

            # Replaces the cell with a completely valid sub-mesh mathematically
            pts, u_opts = analytical_fill_gap(i, step_L, step_R)
            new_points.extend(pts)
            opts_to_remove.extend(u_opts)

        mesh.extend(new_points)
        mesh = sorted(list(set(mesh)))

        for opt in opts_to_remove:
            if opt in optional:
                optional.remove(opt)

    return mesh