from typing import List
import math
import time

def generate_fdtd_mesh_1d(
    fixed_points: List[float],
    optional_points: List[float],
    max_res: float,
    ratio: float
) -> List[float]:
    """
    Generates a 1D FDTD mesh starting from fixed_points, snapping to optional_points
    where possible, and strictly enforcing max_res and cell grading ratios.
    """
    if not fixed_points:
        return []

    # 1. Sort and clean fixed points
    fixed = sorted(list(set(fixed_points)))
    if len(fixed) < 2:
        return fixed

    min_val, max_val = fixed[0], fixed[-1]

    # Filter optional points: only keep those inside the boundary and not already in fixed
    optional = sorted([p for p in set(optional_points) if min_val < p < max_val and p not in fixed])

    mesh = list(fixed)

    def snap_to_optional(base_pt: float, ideal_step: float, prev_dx: float, direction: int):
        """
        Attempts to find a valid optional point to snap to.
        direction: 1 (stepping right) or -1 (stepping left)
        """
        best_opt = None
        best_step = -1.0

        for p_opt in optional:
            step_opt = abs(p_opt - base_pt)

            # Condition 1: Must be in the correct direction
            if direction == 1 and p_opt <= base_pt + 1e-9: continue
            if direction == -1 and p_opt >= base_pt - 1e-9: continue

            # Condition 2: Must slightly reduce the size (step_opt <= ideal_step)
            if step_opt > ideal_step + 1e-9: continue

            # Condition 3: Must not violate the ratio backwards with the cell we just radiated from
            if prev_dx > step_opt * ratio + 1e-9: continue

            # Pick the optional point that gives the largest valid step (closest to ideal_step)
            # This minimizes unnecessary cell count inflation.
            if step_opt > best_step:
                best_step = step_opt
                best_opt = p_opt

        return best_opt, best_step

    loop_count = 0
    max_loops = 2000  # Safety valve: A 1D mesh should mathematically never need this many iterations

    while True:
        loop_count += 1
        if loop_count > max_loops:
            raise RuntimeError(
                f"Mesh generation failed to converge after {max_loops} iterations. "
                f"Current mesh size: {len(mesh)}. "
                "This is likely an algorithmic infinite loop edge case triggered by the optimizer."
            )

        #print(mesh)
        #time.sleep(1)
        # Calculate current cell sizes
        dx = [mesh[i+1] - mesh[i] for i in range(len(mesh)-1)]

        # forces list stores: (forcing_size, direction_of_action, cell_index)
        forces = []
        for i in range(len(dx)):
            L = dx[i]

            # Left side of cell violates ratio with previous cell
            if i > 0 and L > dx[i-1] * ratio + 1e-9:
                forces.append((dx[i-1], 'left', i))

            # Right side of cell violates ratio with next cell
            if i < len(dx)-1 and L > dx[i+1] * ratio + 1e-9:
                forces.append((dx[i+1], 'right', i))

            # Cell violates max_res
            if L > max_res + 1e-9:
                # To maintain symmetry, max_res acts like a force from the smaller neighbor
                if i > 0 and i < len(dx)-1:
                    if dx[i-1] < dx[i+1] - 1e-9:
                        forces.append((dx[i-1], 'left_max', i))
                    elif dx[i+1] < dx[i-1] - 1e-9:
                        forces.append((dx[i+1], 'right_max', i))
                    else:
                        forces.append((dx[i-1], 'left_max', i))
                        forces.append((dx[i+1], 'right_max', i))
                elif i > 0:
                    forces.append((dx[i-1], 'left_max', i))
                elif i < len(dx)-1:
                    forces.append((dx[i+1], 'right_max', i))
                else:
                    forces.append((0, 'center_max', i)) # Domain only has 1 massive cell

        if not forces:
            break # Mesh perfectly satisfies max_res and ratio!

        # Optimization: Emulate "radiating outwards from smallest cells"
        # by only processing the forces driven by the absolute minimum adjacent cell sizes.
        min_force_val = min(f[0] for f in forces)
        active_forces = [f for f in forces if abs(f[0] - min_force_val) < 1e-9]

        # Group active forces by the target cell they are breaking up
        cell_actions = {}
        for f_val, direction, i in active_forces:
            if i not in cell_actions:
                cell_actions[i] = []
            cell_actions[i].append(direction)

        new_points = []
        opts_to_remove = []

        def safe_split(idx: int, target_step_L: float, target_step_R: float = None):
            """
            Divides a gap into N equal segments so no segment exceeds the target step.
            This prevents backward ratio cascades that cause micro-segmentation loops.
            """
            L_cell = dx[idx]
            left_b = mesh[idx]
            right_b = mesh[idx+1]

            target = target_step_L
            if target_step_R is not None:
                target = min(target_step_L, target_step_R)

            N = math.ceil(L_cell / target)
            if N <= 1: N = 2 # Safety fallback

            pts = []
            u_opts = []
            ideal_segment = L_cell / N
            # Calculate a mathematically safe snap deviation to prevent internal ratio breakage
            safe_deviation_fraction = ((ratio - 1.0) / (ratio + 1.0)) * 0.9

            for k in range(1, N):
                ideal_pt = left_b + k * ideal_segment
                best_opt = None
                best_diff = ideal_segment * safe_deviation_fraction

                for p_opt in optional:
                    if left_b < p_opt < right_b:
                        diff = abs(p_opt - ideal_pt)
                        if diff < best_diff:
                            best_diff = diff
                            best_opt = p_opt

                if best_opt is not None:
                    pts.append(best_opt)
                    u_opts.append(best_opt)
                else:
                    pts.append(ideal_pt)

            return pts, u_opts

        # Execute splits
        for i, actions in cell_actions.items():
            L = dx[i]
            step_L, step_R = None, None

            # Calculate Ideal Steps
            if 'left' in actions or 'left_max' in actions:
                prev_dx = dx[i-1] if i > 0 else max_res
                step_L = min(prev_dx * ratio, max_res)

            if 'right' in actions or 'right_max' in actions:
                prev_dx = dx[i+1] if i < len(dx)-1 else max_res
                step_R = min(prev_dx * ratio, max_res)

            if 'center_max' in actions:
                step_L, step_R = max_res, max_res

            # Attempt to snap to optional points
            p_L, p_R = None, None
            final_step_L, final_step_R = 0.0, 0.0
            opt_pt_L, opt_pt_R = None, None

            if step_L is not None:
                prev_dx = dx[i-1] if i > 0 else step_L
                opt_pt_L, opt_step_L = snap_to_optional(mesh[i], step_L, prev_dx, 1)

                final_step_L = opt_step_L if opt_pt_L is not None else step_L
                p_L = opt_pt_L if opt_pt_L is not None else mesh[i] + step_L

            if step_R is not None:
                prev_dx = dx[i+1] if i < len(dx)-1 else step_R
                opt_pt_R, opt_step_R = snap_to_optional(mesh[i+1], step_R, prev_dx, -1)

                final_step_R = opt_step_R if opt_pt_R is not None else step_R
                p_R = opt_pt_R if opt_pt_R is not None else mesh[i+1] - step_R

            # Resolve insertions, preventing crossing and slivers
            final_new_pts = []
            used_opts_this_cell = []

            if p_L is not None and p_R is not None:
                # Coming from both sides (closing the gap)
                if p_L >= p_R - 1e-9:
                    # Steps cross, safely bisect to close the gap cleanly
                    pts, u_opts = safe_split(i, step_L, step_R)
                    final_new_pts.extend(pts)
                    used_opts_this_cell.extend(u_opts)
                else:
                    mid_gap = p_R - p_L
                    # Prevent creating a tiny middle sliver that would cascade points later
                    if mid_gap < final_step_L / ratio - 1e-9 or mid_gap < final_step_R / ratio - 1e-9:
                        pts, u_opts = safe_split(i, step_L, step_R)
                        final_new_pts.extend(pts)
                        used_opts_this_cell.extend(u_opts)
                    else:
                        final_new_pts.extend([p_L, p_R])
                        if opt_pt_L: used_opts_this_cell.append(opt_pt_L)
                        if opt_pt_R: used_opts_this_cell.append(opt_pt_R)

            elif p_L is not None:
                # Radiating left-to-right
                if L - final_step_L < final_step_L / ratio - 1e-9:
                    # Bisect instead to avoid sliver
                    pts, u_opts = safe_split(i, step_L)
                    final_new_pts.extend(pts)
                    used_opts_this_cell.extend(u_opts)
                else:
                    final_new_pts.append(p_L)
                    if opt_pt_L: used_opts_this_cell.append(opt_pt_L)

            elif p_R is not None:
                # Radiating right-to-left
                if L - final_step_R < final_step_R / ratio - 1e-9:
                    # Bisect instead to avoid sliver
                    pts, u_opts = safe_split(i, step_R)
                    final_new_pts.extend(pts)
                    used_opts_this_cell.extend(u_opts)
                else:
                    final_new_pts.append(p_R)
                    if opt_pt_R: used_opts_this_cell.append(opt_pt_R)

            new_points.extend(final_new_pts)
            opts_to_remove.extend(used_opts_this_cell)

        # Update Mesh & clean up optional points
        mesh.extend(new_points)
        mesh = sorted(list(set(mesh)))

        for opt in opts_to_remove:
            if opt in optional:
                optional.remove(opt)

    return mesh