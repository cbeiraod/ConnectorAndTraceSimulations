from typing import List
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

    def smart_bisect(idx: int):
        """
        Attempts to snap a forced bisection to a nearby optional point,
        provided the two resulting halves don't violate the ratio against each other.
        """
        L_cell = dx[idx]
        left_b = mesh[idx]
        right_b = mesh[idx+1]

        best_pt = left_b + L_cell / 2.0
        best_diff = L_cell
        best_opt = None

        for p_opt in optional:
            if left_b < p_opt < right_b:
                L1 = p_opt - left_b
                L2 = right_b - p_opt

                # Check if the split respects the ratio internally
                if L1 <= L2 * ratio + 1e-9 and L2 <= L1 * ratio + 1e-9:
                    diff = abs(p_opt - (left_b + L_cell / 2.0))
                    if diff < best_diff:
                        best_diff = diff
                        best_pt = p_opt
                        best_opt = p_opt

        return best_pt, best_opt

    while True:
        #print(mesh)
        #time.sleep(5)
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
            final_step_L, final_step_R = 0, 0
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
                    pt, u_opt = smart_bisect(i)
                    final_new_pts.append(pt)
                    if u_opt: used_opts_this_cell.append(u_opt)
                else:
                    mid_gap = p_R - p_L
                    # Prevent creating a tiny middle sliver that would cascade points later
                    if mid_gap < final_step_L / ratio - 1e-9 or mid_gap < final_step_R / ratio - 1e-9:
                        pt, u_opt = smart_bisect(i)
                        final_new_pts.append(pt)
                        if u_opt: used_opts_this_cell.append(u_opt)
                    else:
                        final_new_pts.extend([p_L, p_R])
                        if opt_pt_L: used_opts_this_cell.append(opt_pt_L)
                        if opt_pt_R: used_opts_this_cell.append(opt_pt_R)

            elif p_L is not None:
                # Radiating left-to-right
                if L - final_step_L < final_step_L / ratio - 1e-9:
                    # Bisect instead to avoid sliver
                    pt, u_opt = smart_bisect(i)
                    final_new_pts.append(pt)
                    if u_opt: used_opts_this_cell.append(u_opt)
                else:
                    final_new_pts.append(p_L)
                    if opt_pt_L: used_opts_this_cell.append(opt_pt_L)

            elif p_R is not None:
                # Radiating right-to-left
                if L - final_step_R < final_step_R / ratio - 1e-9:
                    # Bisect instead to avoid sliver
                    pt, u_opt = smart_bisect(i)
                    final_new_pts.append(pt)
                    if u_opt: used_opts_this_cell.append(u_opt)
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