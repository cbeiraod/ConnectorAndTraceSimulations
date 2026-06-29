import pytest
import random
import sys
from hypothesis import given, assume, settings, seed, example, strategies as st
from rf_sim.helper import FDTDMesher1D

class TestFDTDMesher1DState:

    def test_init_raises_error_on_invalid_fixed(self):
        """Test that we need at least two unique fixed points."""
        with pytest.raises(ValueError):
            FDTDMesher1D(None, [], 2.0, 1.5)

        with pytest.raises(ValueError):
            FDTDMesher1D(1.2, [], 2.0, 1.5)

        with pytest.raises(ValueError):
            FDTDMesher1D([], [], 2.0, 1.5)

        with pytest.raises(ValueError):
            FDTDMesher1D([5.0], [], 2.0, 1.5)

        with pytest.raises(ValueError):
            FDTDMesher1D([5.0, 5.0], [], 2.0, 1.5) # Only 1 unique point

    def test_init_cleans_inputs(self):
        """
        Test that __init__ correctly sorts, removes duplicates,
        and filters optional points outside the fixed boundaries or colliding with fixed points.
        """
        fixed = [10.0, 0.0, 10.0, 5.0, 0.0]  # Unordered, has duplicates
        optional = [-5.0, 2.0, 5.0, 8.0, 8.0, 15.0]  # Out of bounds, collisions, duplicates

        mesher = FDTDMesher1D(fixed, optional, max_res=2.0, ratio=1.5)

        # Expected fixed: sorted and unique
        assert mesher.mesh == [0.0, 5.0, 10.0]

        # Expected optional: strictly between 0.0 and 10.0, not 5.0, unique
        assert mesher.optional_points == [2.0, 8.0]
        assert mesher.max_res == 2.0
        assert mesher.ratio == 1.5

    def test_ignore_out_of_bounds_optional(self):
        """Ensures optional points outside the fixed domain are safely ignored."""
        fixed = [0.0, 1.0]
        optional = [-1.0, 0.5, 2.0] # -1.0 and 2.0 should be ignored
        mesher = FDTDMesher1D(fixed, optional, max_res=0.6, ratio=1.2)

        assert mesher.mesh == pytest.approx([0.0, 1.0])
        assert mesher.optional_points == pytest.approx([0.5])

    def test_get_cell_sizes(self):
        """Test calculation of adjacent point distances."""
        mesher = FDTDMesher1D([0.0, 10.0], [], 2.0, 1.5)
        mesher.mesh = [0.0, 2.5, 5.0, 9.0] # Artificially set mesh

        sizes = mesher._get_cell_sizes()

        # pytest.approx can handle list comparisons directly!
        assert sizes == pytest.approx([2.5, 2.5, 4.0])

    def test_calculate_forces_no_forces(self):
        """Test when ratio is respected everywhere or cells are > max_res."""
        mesher = FDTDMesher1D([0.0, 1.5, 4.0], [], max_res=2.0, ratio=2.0)
        # dx = [1.5, 2.5] -> 1.5 is < max_res, but 2.5 < 1.5 * ratio (3.0), so no force

        assert mesher._force_left == [0.0, 0.0]
        assert mesher._force_right == [0.0, 0.0]

    def test_calculate_forces_with_violations(self):
        """Test force calculations when ratio is violated."""
        mesher = FDTDMesher1D([0.0, 10.0], [], max_res=2.0, ratio=1.5)
        mesher.mesh = [0.0, 1.0, 6.0, 7.2]
        # Mesh sizes (dx):
        # Cell 0: 1.0 (< max_res)
        # Cell 1: 5.0 (violates ratio from Cell 0 and Cell 2)
        # Cell 2: 1.2 (< max_res)

        # Cell 1 feels force from Cell 0 on the left: 1.0 / 1.0 = 1.0
        assert mesher._force_left[1] == pytest.approx(1.0 / 1.0)
        assert mesher._force_left[0] == 0.0
        assert mesher._force_left[2] == 0.0

        # Cell 1 feels force from Cell 2 on the right: 1.0 / 1.2
        assert mesher._force_right[1] == pytest.approx(1.0 / 1.2)
        assert mesher._force_right[0] == 0.0
        assert mesher._force_right[2] == 0.0

    def test_find_target_cell_by_force(self, monkeypatch):
        """Test target selection when forces are active."""
        mesher = FDTDMesher1D([0.0, 1.0, 3.0, 8.0], [], 2.1, 2.0)

        # Simulate an environment where Cell 2 has the highest force
        mesher._force_left = [0.0, 0.5, 1.0] # Max force is 1.0 at index 2
        mesher._force_right = [0.0, 0.0, 0.8]

        # Setup mock to just return the first tied index using pytest's monkeypatch
        monkeypatch.setattr('random.choice', lambda seq: seq[0])

        target = mesher._find_target_cell()
        assert target == 2

    def test_find_target_cell_by_size(self, monkeypatch):
        """Test target selection when no forces, but oversize cells exist."""
        mesher = FDTDMesher1D([0.0, 3.0, 4.5, 8.5], [], max_res=2.0, ratio=1.5)

        # No forces, but Cell 0 (3.0) and Cell 2 (4.0) are > max_res
        # Cell 0 is the smallest oversized cell
        self._force_left = [0.0, 0.0, 0.0]
        self._force_right = [0.0, 0.0, 0.0]

        monkeypatch.setattr('random.choice', lambda seq: seq[0])

        target = mesher._find_target_cell()
        assert target == 0 # Index of the 3.0 cell

    def test_find_target_cell_complete(self):
        """Test when mesh is perfectly valid."""
        mesher = FDTDMesher1D([0.0, 1.5, 3.3, 4.5], [], max_res=2.0, ratio=1.5)

        # All cells <= max_res

        target = mesher._find_target_cell()
        assert target is None


class TestFDTDMesher1DSplitting:

    def test_evaluate_optional_snap_success(self):
        """Test snapping to a valid optional point."""
        mesher = FDTDMesher1D([0.0, 10.0], [1.05], max_res=2.0, ratio=1.5)
        # Candidate at 1.0. Target step is 1.0. Optional point at 1.05 is within tolerance.
        # Distance from prev_pt (0.0) is 1.05 <= 2.0 (max_res).
        # Ratio checks pass.
        pt = mesher._evaluate_optional_snap(candidate_pt=1.0, prev_pt=0.0, next_pt=2.0, target_step=1.0)
        assert pt == 1.05

    def test_evaluate_optional_snap_violation_max_res(self):
        """Test rejecting an optional point because it violates max_res."""
        mesher = FDTDMesher1D([0.0, 10.0], [2.1], max_res=2.0, ratio=1.5)
        # Candidate at 1.9. prev_pt is 0.0.
        # Optional point at 2.1 is within a tolerance of candidate (e.g. 0.2 * 1.9 = 0.38)
        # BUT 2.1 - 0.0 = 2.1 > max_res (2.0), so it should be rejected.
        pt = mesher._evaluate_optional_snap(candidate_pt=1.9, prev_pt=0.0, next_pt=3.8, target_step=1.9)
        assert pt == 1.9

    def test_evaluate_optional_snap_violation_ratio(self):
        """Test rejecting an optional point because it violates the ratio constraint."""
        # Set a strict ratio
        mesher = FDTDMesher1D([0.0, 10.0], [1.2], max_res=2.0, ratio=1.1)
        # Candidate is 1.0, target_step is 1.0. Optional is 1.2.
        # Distance to prev_pt (0.0) is 1.2.
        # 1.2 > 1.0 * 1.1 (violates ratio compared to ideal uniform target_step).
        pt = mesher._evaluate_optional_snap(candidate_pt=1.0, prev_pt=0.0, next_pt=2.0, target_step=1.0)
        assert pt == 1.0

    def test_split_unforced_cell_no_optional(self):
        """Test Case A: Evenly splitting an oversized cell with no forces."""
        mesher = FDTDMesher1D([0.0, 5.0], [], max_res=2.0, ratio=1.5)
        # Mesh has one cell size 5.0 at index 0.
        # N = ceil(5.0 / 2.0) = 3.
        # Step = 5.0 / 3 = 1.666...
        new_points = mesher._split_unforced_cell(0)
        assert new_points == pytest.approx([5.0 / 3.0, 10.0 / 3.0])

    def test_split_unforced_cell_with_optional(self):
        """Test Case A with an optional point to snap to."""
        mesher = FDTDMesher1D([0.0, 5.0], [1.7], max_res=2.0, ratio=1.5)
        # Ideal points: 1.666... and 3.333...
        # 1.666... should snap to 1.7. 3.333... stays the same.
        new_points = mesher._split_unforced_cell(0)
        assert new_points == pytest.approx([1.7, 10.0 / 3.0])

    def test_split_forced_cell_one_sided_left(self):
        """Test Case B: Geometric progression from the left side perfectly closing gap."""
        mesher = FDTDMesher1D([0.0, 1.0, 7.0], [], max_res=5.0, ratio=2.0)
        # Target cell is index 1 (1.0 to 7.0), size 6.0
        # Left neighbor size: 1.0. Right neighbor: None
        # Geometric steps from left: 2.0, then 4.0 (totaling 6.0 exactly)
        # We expect a new point to be placed at 1.0 + 2.0 = 3.0
        new_points = mesher._split_forced_cell(1)
        assert new_points == pytest.approx([3.0])

    def test_split_forced_cell_one_sided_right(self):
        """Test Case B: Geometric progression from the right side perfectly closing gap."""
        mesher = FDTDMesher1D([0.0, 6.0, 7.0], [], max_res=5.0, ratio=2.0)
        # Target cell is index 0 (0.0 to 6.0), size 6.0
        # Left neighbor: None. Right neighbor size: 1.0
        # Geometric steps stepping back from right: 2.0, then 4.0
        # We expect a new point to be placed at 7.0 - 1.0 (right point) - 2.0 = 4.0
        # Wait, the cell is 0.0 to 6.0. 6.0 - 2.0 = 4.0.
        new_points = mesher._split_forced_cell(0)
        assert new_points == pytest.approx([4.0])

    def test_split_forced_cell_two_sided(self):
        """Test Case C: Geometric progression from both sides perfectly meeting in the middle."""
        mesher = FDTDMesher1D([0.0, 1.0, 7.0, 8.0], [], max_res=5.0, ratio=2.0)
        # Target cell is index 1 (1.0 to 7.0), size 6.0.
        # Both left and right neighbors have size 1.0.
        # Left steps: 2.0 -> pt at 3.0.
        # Right steps: 2.0 -> pt at 5.0.
        # Remaining gap between 3.0 and 5.0 is exactly 2.0 (no slivers, fits perfectly).
        new_points = mesher._split_forced_cell(1)
        assert new_points == pytest.approx([3.0, 5.0])

    def test_split_forced_cell_two_sided_larger(self):
        """Test Case C: Geometric progression from both sides perfectly meeting in the middle."""
        mesher = FDTDMesher1D([0.0, 1.0, 9.0, 10.0], [], max_res=5.0, ratio=2.0)
        # Target cell is index 1 (1.0 to 9.0), size 8.0.
        # Both left and right neighbors have size 1.0.
        # Left steps: 2.0 -> pt at 3.0.
        # Right steps: 2.0 -> pt at 7.0.
        # Remaining gap between 3.0 and 7.0 is exactly 4.0 (no slivers, fits perfectly).
        new_points = mesher._split_forced_cell(1)
        assert new_points == pytest.approx([3.0, 7.0])

    def test_check_rollback_condition(self):
        """Test Case D: Identifying when the next step creates an unsolvable sliver."""
        mesher = FDTDMesher1D([0.0, 10.0], [], max_res=5.0, ratio=2.0)

        # Scenario 1: Gap is 3.0, next proposed step is 2.0.
        # Remaining space is 1.0. Next step can safely reduce by ratio down to (2.0 / 2.0) = 1.0.
        # 1.0 >= 1.0. So it's safe (barely). Not a sliver.
        assert mesher._check_rollback_condition(remaining_gap=3.0, next_proposed_step=2.0) is False

        # Scenario 2: Gap is 2.5, next proposed step is 2.0.
        # Remaining space is 0.5. Next step minimum allowed is (2.0 / 2.0) = 1.0.
        # 0.5 < 1.0, so this leaves a sliver! Rollback condition triggers.
        assert mesher._check_rollback_condition(remaining_gap=2.5, next_proposed_step=2.0) is True

        # Scenario 3: Gap is exactly the next step (2.0), no space left. Perfect fit.
        assert mesher._check_rollback_condition(remaining_gap=2.0, next_proposed_step=2.0) is False


def check_mesh_validity(mesh, fixed_points, max_res, ratio):
    """Helper function to assert a mesh respects all fundamental FDTD constraints."""
    assert mesh is not None, "Mesh generation failed (returned None)."

    # 1. All fixed points must be present
    for fp in fixed_points:
        assert fp in mesh, f"Fixed point {fp} is missing from the generated mesh."

    # 2. Points must be sorted monotonically
    assert mesh == sorted(mesh), "Mesh points are not strictly increasing."

    # Calculate final sizes
    dx = [mesh[i+1] - mesh[i] for i in range(len(mesh)-1)]

    # 3. max_res constraint
    for i, size in enumerate(dx):
        assert size <= max_res + 1e-9, f"Cell {i} size {size} exceeds max_res {max_res}."

    # 4. ratio constraint
    for i in range(len(dx) - 1):
        assert dx[i] <= dx[i+1] * ratio + 1e-9, \
            f"Ratio violation: {dx[i]} > {dx[i+1]} * {ratio} at cells {i} and {i+1}"
        assert dx[i+1] <= dx[i] * ratio + 1e-9, \
            f"Ratio violation: {dx[i+1]} > {dx[i]} * {ratio} at cells {i} and {i+1}"

@st.composite
def non_sliver_fixed_points(draw, min_gap=0.5, min_val=-100.0, max_val=100.0):
    """
    A custom strategy that guarantees points are sorted, unique,
    and never closer together than `min_gap` (no slivers).
    """
    # 1. Draw how many points we want (e.g., 2 to 10)
    num_points = draw(st.integers(min_value=2, max_value=10))

    # 2. Draw the starting point
    start = draw(st.floats(min_value=min_val, max_value=max_val - (num_points + 10) * min_gap))
    points = [start]

    # 3. Draw the sequential gaps, strictly bounding them to be >= min_gap
    # We use st.lists to draw exactly (num_points - 1) gaps.
    gaps = draw(st.lists(
        st.floats(min_value=min_gap, max_value=10.0),
        min_size=num_points - 1,
        max_size=num_points - 1
    ))

    # 4. Accumulate the gaps to create the final monotonic list of points
    current = start
    for gap in gaps:
        current += gap
        if current < max_val:
            points.append(current)
        else:
            break

    return points

class TestFDTDMesher1DIntegration:

    @pytest.mark.parametrize("fixed_steps", [
        [0.0, 1.0, 2.1],           # cell0: max_res, cell1: 1.1 max_res
        [0.0, 1.0, 2.1, 3.1],      # cell0: max_res, cell1: 1.1 max_res, cell2: max_res
        [0.0, 1.0, 2.9],           # cell0: max_res, cell1: 1.9 max_res
        [0.0, 1.0, 2.9, 3.9],      # cell0: max_res, cell1: 1.9 max_res, cell2: max_res
        [0.0, 1.0, 6.1],           # cell0: max_res, cell1: 5.1 max_res
        [0.0, 1.0, 6.1, 7.1],      # cell0: max_res, cell1: 5.1 max_res, cell2: max_res
        [0.0, 1.0, 6.9],           # cell0: max_res, cell1: 5.9 max_res
        [0.0, 1.0, 6.9, 7.9],      # cell0: max_res, cell1: 5.9 max_res, cell2: max_res
    ])
    def test_unforced_cell_edge_cases_respect_constraints(self, fixed_steps, max_res = 1.0, ratio = 1.1):
        """
        Tests the edge cases where unforced cells are just above integer multiples
        of max_res, forcing backward ratio corrections.
        """
        #max_res = 2.0
        #ratio = 1.5

        # Construct the fixed points based on the test case multipliers
        if max_res == 1.0:
            fixed = fixed_steps
        else:
            fixed = [0.0]
            current = 0.0
            for mult in fixed_steps:
                current += max_res * mult
                fixed.append(current)

        mesher = FDTDMesher1D(fixed, optional_points=[], max_res=max_res, ratio=ratio)

        for val in mesher._force_left:
            assert val == pytest.approx(0.0)
        for val in mesher._force_right:
            assert val == pytest.approx(0.0)

        final_mesh = mesher.generate()

        check_mesh_validity(final_mesh, fixed, max_res, ratio)

    def test_simple_bisection(self):
        """Tests the trivial case where the max_res is exactly half of the point spacing."""
        fixed = [-1.0, 1.0]
        mesher = FDTDMesher1D(fixed, optional_points=[], max_res=1.0, ratio=1.2)
        final_mesh = mesher.generate()

        assert final_mesh == pytest.approx([-1.0, 0.0, 1.0])
        check_mesh_validity(final_mesh, fixed, max_res=1.0, ratio=1.2)

    def test_basic_bisection(self):
        """Tests the first edge case: safely bisecting to avoid infinite loops."""
        fixed = [-1.0, 1.0]
        mesher = FDTDMesher1D(fixed, optional_points=[], max_res=0.9, ratio=1.2)
        final_mesh = mesher.generate()

        # Expected to safely split into 3 equal cells of 2/3
        assert final_mesh == pytest.approx([-1.0, -1/3, 1/3, 1.0])
        check_mesh_validity(final_mesh, fixed, max_res=0.9, ratio=1.2)

    def test_basic_bisection_with_optional_points(self):
        """Tests the first edge case: safely bisecting to avoid infinite loops."""
        fixed = [-1.0, 1.0]
        optional = [-0.3, 0.3]
        mesher = FDTDMesher1D(fixed, optional_points=optional, max_res=0.9, ratio=1.2)
        final_mesh = mesher.generate()

        # Expected to safely split into 3 equal cells of 2/3
        assert final_mesh == pytest.approx([-1.0, -0.3, 0.3, 1.0])
        check_mesh_validity(final_mesh, fixed, max_res=0.9, ratio=1.2)


    def test_complex_grading_cascade(self):
        """Tests a highly disparate domain to ensure the ratio cascades correctly without hanging."""
        fixed = [0.0, 10.0]
        # Small forced cell at the start will force a ratio cascade all the way to max_res
        fixed.insert(1, 0.1)

        mesher = FDTDMesher1D(fixed, optional_points=[], max_res=2.0, ratio=1.5)
        final_mesh = mesher.generate()

        check_mesh_validity(final_mesh, fixed, max_res=2.0, ratio=1.5)
        # Ensure it successfully expanded up to max_res
        assert any(pytest.approx(val, abs=1e-2) == 2.0 for val in mesher.dx), "Mesh failed to scale up to max_res"

    def test_symmetric_non_uniform_mesh(self):
        """Tests that a symmetric starting mesh results in a perfectly symmetric final mesh."""
        positive_half = [0.0, 4.0, 4.5, 5.0, 5.5, 6.0, 12.0]
        # Reconstruct the full symmetric domain
        fixed = sorted(list(set([-x for x in positive_half] + positive_half)))

        mesher = FDTDMesher1D(fixed, optional_points=[], max_res=1.0, ratio=1.2)
        final_mesh = mesher.generate()

        # 1. Must satisfy all mathematical FDTD requirements
        check_mesh_validity(final_mesh, fixed, max_res=1.0, ratio=1.2)

        # 2. Must be perfectly symmetric around 0
        for pt in final_mesh:
            assert any(pytest.approx(-pt, abs=1e-9) == m for m in final_mesh), f"Symmetry broken: {pt} exists but {-pt} does not"

    def test_multiple_optional_points(self):
        """Tests the selection logic when multiple optional points are available in a gap."""
        fixed = [0.0, 2.0]
        # Max res is 1.0, so ideal step is 1.0.
        # Optional points: 0.9 (invalid step > max_res), 1.05 (should be selected)
        # The algorithm should pick 1.05 because it's the closest (only) valid optional step to ideal_step.
        optional = [0.5, 0.9, 1.05]

        mesher = FDTDMesher1D(fixed, optional_points=optional, max_res=1.09, ratio=1.5)
        final_mesh = mesher.generate()

        check_mesh_validity(final_mesh, fixed, max_res=1.09, ratio=1.5)
        assert any(pytest.approx(1.05, abs=1e-9) == m for m in final_mesh), "Failed to snap to the optimal optional point (1.05)"

    def test_symmetric_non_uniform_example_mesh(self):
        """Tests that a symmetric starting mesh results in a perfectly symmetric final mesh."""
        positive_half = [0.25, 0.45, 0.4, 0.35, 0.3, 5.0, 0.8500000000000001]
        # Reconstruct the full symmetric domain
        fixed = sorted(list(set([-x for x in positive_half] + positive_half)))

        #raise RuntimeError(fixed)

        mesher = FDTDMesher1D(fixed, optional_points=[], max_res=1.7472369284948892, ratio=1.2)
        final_mesh = mesher.generate()

        # 1. Must satisfy all mathematical FDTD requirements
        check_mesh_validity(final_mesh, fixed, max_res=1.7472369284948892, ratio=1.2)

        # 2. Must be perfectly symmetric around 0
        for pt in final_mesh:
            assert any(pytest.approx(-pt, abs=1e-9) == m for m in final_mesh), f"Symmetry broken: {pt} exists but {-pt} does not"

    @settings(max_examples=5)
    # @seed(42)  # <-- Uncomment this to globally freeze the random seed for this test
    # @example(...) <-- When Hypothesis finds a bug, paste the output here to keep it forever!
    @example(
        # This should replace the test above... remove it later
        fixed_points=[-0.8500000000000001, -5.0, -0.3, -0.35, -0.4, -0.45, -0.25, 0.25, 0.45, 0.4, 0.35, 0.3, 5.0, 0.8500000000000001],
        optional_points=[],
        max_res=1.7472369284948892,
        ratio=1.2
    )
    @given(
        # Use our custom strategy to guarantee no initial slivers!
        fixed_points=non_sliver_fixed_points(min_gap=0.5),
        optional_points=st.lists(st.floats(min_value=0.0, max_value=100.0), max_size=5),
        max_res=st.floats(min_value=1.0, max_value=10.0),
        ratio=st.floats(min_value=1.1, max_value=2.0)
    )
    def test_fuzz_mesh_generation(self, fixed_points, optional_points, max_res, ratio):
        """
        Fuzz test the mesher with physically sound, non-sliver inputs.
        """
        domain_size = max(fixed_points) - min(fixed_points)
        assume(domain_size >= len(fixed_points) * max_res)

        mesher = FDTDMesher1D(fixed_points, optional_points, max_res, ratio)
        mesh = mesher.generate()

        assert len(mesh) >= len(fixed_points)
        check_mesh_validity(mesh, fixed_points, max_res=max_res, ratio=ratio)


def test_openems_native_mesher_fails_constraints():
    """
    Tests that the native OpenEMS SmoothMeshLines function fails to maintain
    the strict mathematical constraints (max_res, ratio) on a non-uniform starting mesh,
    proving the need for our custom helper.
    """
    # This safely skips the test if CSXCAD is not installed (e.g., local development),
    # but it will run perfectly inside the OpenEMS Docker container in CI.
    CSXCAD = pytest.importorskip("CSXCAD")

    positive_half = [0.25, 0.45, 0.4, 0.35, 0.3, 5.0, 0.8500000000000001]
    fixed = sorted(list(set([-x for x in positive_half] + positive_half)))

    # Run OpenEMS's native 1D smoother
    openems_mesh = CSXCAD.SmoothMeshLines.SmoothMeshLines(fixed, 1.7472369284948892, 1.2)
    openems_mesh = list(openems_mesh) # Ensure it is a standard Python list

    # We expect the native OpenEMS mesher to violate our strict validation checks
    # (usually a grading ratio violation when cascading across highly variable gaps).
    with pytest.raises(AssertionError):
        check_mesh_validity(openems_mesh, fixed, max_res=1.7472369284948892, ratio=1.2)
