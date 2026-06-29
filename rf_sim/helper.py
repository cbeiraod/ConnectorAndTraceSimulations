from typing import List
import math


class FDTDMesher1D:
    def __init__(self, fixed_points: list[float], optional_points: list[float], max_res: float, ratio: float):
        """
        Initializes the mesher, cleans up the inputs (sorting, removing duplicates,
        filtering out-of-bounds optional points), and sets up the initial state.
        """
        # 1. Preprocess inputs
        if not fixed_points:
            raise ValueError("You must define the fixed points of the mesh")

        fixed = sorted(list(set(fixed_points)))
        min_val, max_val = fixed[0], fixed[-1]

        if len(fixed) < 2:
            raise ValueError("At least two fixed points are required to define a domain.")

        self.fixed_points = fixed
        self.min_val = min_val
        self.max_val = max_val
        self.optional_points = sorted([p for p in set(optional_points)
                       if min_val < p < max_val and p not in fixed])
        self.max_res = max_res
        self.ratio = ratio

        self.mesh = fixed

        self.dx = [self.mesh[i+1] - self.mesh[i] for i in range(len(self.mesh)-1)]

    def generate(self, max_iterations: int = 2000) -> list[float]:
        """
        The main orchestrator loop.
        Calls the helper methods below until no forces exist and no cells > max_res.
        """
        pass

    # ========================
    # State Inspection Methods
    # ========================

    def _get_cell_sizes(self) -> list[float]:
        """Returns the sizes of all current cells in the mesh."""
        pass

    def _calculate_forces(self) -> tuple[list[float], list[float]]:
        """
        Evaluates the mesh and returns two lists (force_left, force_right).
        A force is > 0 if a cell is violating the ratio constraint from its neighbor.
        """
        pass

    def _find_target_cell(self, force_left: list[float], force_right: list[float]) -> int | None:
        """
        Determines which cell needs to be split next based on maximum force,
        or smallest over-sized cell. Returns None if the mesh is fully valid.
        """
        pass

    # =============================
    # Geometric & Splitting Methods
    # =============================

    def _evaluate_optional_snap(self, candidate_pt: float, prev_pt: float, next_pt: float, target_step: float) -> float:
        """
        Checks if a candidate point can be safely moved to an optional point
        without violating ratio or max_res constraints.
        """
        pass

    def _split_unforced_cell(self, cell_index: int) -> list[float]:
        """
        Case A: Splits a cell that has no forces acting on it but is larger than max_res.
        Returns a list of NEW points to be inserted inside the cell.
        """
        pass

    def _split_forced_cell(self, cell_index: int) -> list[float]:
        """
        Cases B, C, & D: Handles the geometric progression from one or both sides.
        Returns a list of NEW points to be inserted inside the cell.
        """
        pass

    def _check_rollback_condition(self, remaining_gap: float, next_proposed_step: float) -> bool:
        """
        Evaluates if taking the next geometric step will leave an unsolvable sliver.
        """
        pass

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