import math
from rf_sim.helper import FDTDMesher1D

# =============================================================================
# CONFIGURATION
# Hardcode your parameters here to quickly test different meshing scenarios
# =============================================================================
FIXED_POINTS = [0.0, 10.0, 10.1]
OPTIONAL_POINTS = [5.0]
MAX_RES = 2.0
RATIO = 1.5
ALGORITHM = "segment_uniform"
ALGORITHM = "segment_graded"
ALGORITHM = "global_grid_search"
#ALGORITHM = "advancing_front"

# =============================================================================
# ANSI COLOR CODES (For terminal formatting)
# =============================================================================
C_RESET = "\033[0m"
C_FIXED = "\033[91m"     # Red
C_OPT = "\033[96m"       # Cyan
C_ADDED = "\033[94m"     # Blue
C_OK = "\033[92m"        # Green
C_WARN = "\033[93m"      # Yellow/Orange

def is_close(val, lst, tol=1e-9):
    """Helper to cleanly check if a point exists in a list (float safe)."""
    return any(math.isclose(val, x, abs_tol=tol) for x in lst)

def main():
    print("=" * 85)
    print(" 1D FDTD MESH VISUALIZATION")
    print("=" * 85)
    print(f" Algorithm:       {ALGORITHM}")
    print(f" Max Resolution:  {MAX_RES}")
    print(f" Target Ratio:    {RATIO}")
    print("-" * 85)

    # 1. Generate the Mesh
    mesher = FDTDMesher1D(FIXED_POINTS, OPTIONAL_POINTS, MAX_RES, RATIO)
    try:
        mesh = mesher.generate(algorithm=ALGORITHM)
    except Exception as e:
        print(f"{C_WARN}Mesher failed with error: {e}{C_RESET}")
        return

    # 2. Calculate cell sizes
    dx = [mesh[i+1] - mesh[i] for i in range(len(mesh)-1)]

    # 3. Render the timeline
    for i, pt in enumerate(mesh):
        # -- Print the Point --
        if is_close(pt, FIXED_POINTS):
            pt_label = f"{C_FIXED}FIXED{C_RESET}"
        elif is_close(pt, OPTIONAL_POINTS):
            pt_label = f"{C_OPT}OPTIONAL{C_RESET}"
        else:
            pt_label = f"{C_ADDED}ADDED{C_RESET}"

        print(f"[{i:4}]  {pt:10.4f}  ({pt_label})")

        # -- Print the Cell (dx) immediately below the point --
        if i < len(mesh) - 1:
            cell_size = dx[i]

            # Check Max Res constraint
            res_flag = f"{C_WARN}(! > max_res){C_RESET}" if cell_size > MAX_RES + 1e-9 else ""

            # Check Ratio with Previous Cell (Look-behind)
            ratio_prev_str = "N/A   "
            if i > 0:
                prev_size = dx[i-1]
                r_prev = cell_size / prev_size if prev_size > 0 else 0
                r_prev_actual = max(r_prev, 1/r_prev if r_prev > 0 else 0)

                color = C_WARN if r_prev_actual > RATIO + 1e-9 else C_OK
                ratio_prev_str = f"{color}{r_prev:5.2f}x{C_RESET}"

            # Check Ratio with Next Cell (Look-ahead)
            ratio_next_str = "N/A   "
            if i < len(dx) - 1:
                next_size = dx[i+1]
                r_next = cell_size / next_size if next_size > 0 else 0
                r_next_actual = max(r_next, 1/r_next if r_next > 0 else 0)

                color = C_WARN if r_next_actual > RATIO + 1e-9 else C_OK
                ratio_next_str = f"{color}{r_next:5.2f}x{C_RESET}"

            # Format the "link" between points
            print(f"        |   dx = {cell_size:<8.4f} {res_flag}")
            print(f"        |       (Ratio to prev cell: {ratio_prev_str} | Ratio to next cell: {ratio_next_str})")

    print("=" * 85)
    print(f" Summary: {len(mesh)} Points, {len(dx)} Cells")
    print("=" * 85)

if __name__ == '__main__':
    main()