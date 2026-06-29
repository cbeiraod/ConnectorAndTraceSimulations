import pytest
import random
import sys
from rf_sim.helper import generate_fdtd_mesh_1d
from rf_sim.helper import FDTDMesher1D

class TestFDTDMesher1DState:

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

    def test_init_raises_error_on_invalid_fixed(self):
        """Test that we need at least two unique fixed points."""
        with pytest.raises(ValueError):
            FDTDMesher1D([5.0], [], 2.0, 1.5)

        with pytest.raises(ValueError):
            FDTDMesher1D([5.0, 5.0], [], 2.0, 1.5) # Only 1 unique point

    def test_get_cell_sizes(self):
        """Test calculation of adjacent point distances."""
        mesher = FDTDMesher1D([0.0, 10.0], [], 2.0, 1.5)
        mesher.mesh = [0.0, 2.5, 5.0, 9.0] # Artificially set mesh

        sizes = mesher._get_cell_sizes()

        # pytest.approx can handle list comparisons directly!
        assert sizes == pytest.approx([2.5, 2.5, 4.0])

    def test_calculate_forces_no_forces(self):
        """Test when ratio is respected everywhere or cells are > max_res."""
        mesher = FDTDMesher1D([0.0, 10.0], [], max_res=2.0, ratio=2.0)
        # dx = [1.5, 2.5] -> 1.5 is < max_res, but 2.5 < 1.5 * ratio (3.0), so no force
        mesher.mesh = [0.0, 1.5, 4.0]

        f_left, f_right = mesher._calculate_forces()

        assert f_left == [0.0, 0.0]
        assert f_right == [0.0, 0.0]

    def test_calculate_forces_with_violations(self):
        """Test force calculations when ratio is violated."""
        mesher = FDTDMesher1D([0.0, 10.0], [], max_res=2.0, ratio=1.5)
        # Mesh sizes (dx):
        # Cell 0: 1.0 (< max_res)
        # Cell 1: 5.0 (violates ratio from Cell 0 and Cell 2)
        # Cell 2: 1.2 (< max_res)
        mesher.mesh = [0.0, 1.0, 6.0, 7.2]

        f_left, f_right = mesher._calculate_forces()

        # Cell 1 feels force from Cell 0 on the left: 1.0 / 1.0 = 1.0
        assert f_left[1] == pytest.approx(1.0 / 1.0)
        assert f_left[0] == 0.0
        assert f_left[2] == 0.0

        # Cell 1 feels force from Cell 2 on the right: 1.0 / 1.2
        assert f_right[1] == pytest.approx(1.0 / 1.2)
        assert f_right[0] == 0.0
        assert f_right[2] == 0.0

    def test_find_target_cell_by_force(self, monkeypatch):
        """Test target selection when forces are active."""
        mesher = FDTDMesher1D([0.0, 10.0], [], 2.0, 1.5)

        # Setup mock to just return the first tied index using pytest's monkeypatch
        monkeypatch.setattr('random.choice', lambda seq: seq[0])

        # Simulate an environment where Cell 2 has the highest force
        mesher._get_cell_sizes = lambda: [1.0, 2.0, 5.0]
        f_left = [0.0, 0.5, 1.0] # Max force is 1.0 at index 2
        f_right = [0.0, 0.0, 0.8]

        target = mesher._find_target_cell(f_left, f_right)
        assert target == 2

    def test_find_target_cell_by_size(self, monkeypatch):
        """Test target selection when no forces, but oversize cells exist."""
        mesher = FDTDMesher1D([0.0, 10.0], [], max_res=2.0, ratio=1.5)

        monkeypatch.setattr('random.choice', lambda seq: seq[0])

        # No forces, but Cell 0 (3.0) and Cell 2 (4.0) are > max_res
        # Cell 0 is the smallest oversized cell
        mesher._get_cell_sizes = lambda: [3.0, 1.5, 4.0]
        f_left = [0.0, 0.0, 0.0]
        f_right = [0.0, 0.0, 0.0]

        target = mesher._find_target_cell(f_left, f_right)
        assert target == 0 # Index of the 3.0 cell

    def test_find_target_cell_complete(self):
        """Test when mesh is perfectly valid."""
        mesher = FDTDMesher1D([0.0, 10.0], [], max_res=2.0, ratio=1.5)

        # All cells <= max_res
        mesher._get_cell_sizes = lambda: [1.5, 1.8, 1.2]
        f_left = [0.0, 0.0, 0.0]
        f_right = [0.0, 0.0, 0.0]

        target = mesher._find_target_cell(f_left, f_right)
        assert target is None


#def validate_mesh_constraints(mesh, fixed_points, max_res, ratio):
#    """
#    Helper function to validate that a generated mesh mathematically
#    satisfies all FDTD criteria. Use this in all future tests!
#    """
#    if not mesh:
#        assert not fixed_points
#        return
#
#    # 1. Mesh must be monotonically increasing (sorted) and have no duplicates
#    assert mesh == sorted(list(set(mesh))), "Mesh is not sorted or contains duplicates"
#
#    # 2. Mesh must contain all original fixed points
#    for pt in fixed_points:
#        # Use approx to handle minor floating point inaccuracies
#        assert any(pt == pytest.approx(m, abs=1e-9) for m in mesh), f"Fixed point {pt} is missing from mesh"
#
#    # 3. Check spacing constraints (max_res and ratio)
#    if len(mesh) > 1:
#        dx = [mesh[i+1] - mesh[i] for i in range(len(mesh)-1)]
#
#        for i, val in enumerate(dx):
#            # Check max_res
#            assert val <= max_res + 1e-9, f"Cell {i} size ({val}) exceeds max_res ({max_res})"
#
#            # Check grading ratio against neighbors
#            if i > 0:
#                # Compare with previous cell
#                assert val <= dx[i-1] * ratio + 1e-9, f"Ratio violation at cell {i}: {val} > {dx[i-1]} * {ratio}"
#                assert dx[i-1] <= val * ratio + 1e-9, f"Ratio violation at cell {i-1}: {dx[i-1]} > {val} * {ratio}"
#
#def test_empty_and_single_point():
#    assert generate_fdtd_mesh_1d([], [], 0.9, 1.2) == []
#    assert generate_fdtd_mesh_1d([1.0], [], 0.9, 1.2) == [1.0]
#
#def test_trivial_bisection():
#    """Tests the trivial case where the max_res is exactly half of the point spacing."""
#    fixed = [-1.0, 1.0]
#    mesh = generate_fdtd_mesh_1d(fixed, optional_points=[], max_res=1.0, ratio=1.2)
#
#    expected = [-1.0, 0.0, 1.0]
#    assert [pytest.approx(m, abs=1e-9) for m in mesh] == expected
#    validate_mesh_constraints(mesh, fixed, max_res=1.0, ratio=1.2)
#
#def test_basic_bisection():
#    """Tests the first edge case: safely bisecting to avoid infinite loops."""
#    fixed = [-1.0, 1.0]
#    mesh = generate_fdtd_mesh_1d(fixed, optional_points=[], max_res=0.9, ratio=1.2)
#
#    # Expected to safely split into 4 equal cells of 0.5
#    expected = [-1.0, -1/3, 1/3, 1.0]
#    assert [pytest.approx(m, abs=1e-9) for m in mesh] == expected
#    validate_mesh_constraints(mesh, fixed, max_res=0.9, ratio=1.2)
#
#def test_smart_bisection_optional_point():
#    """Tests the second edge case: pulling a bisection to a nearby optional point."""
#    fixed = [-1.0, 1.0]
#    optional = [-0.3, 0.3]
#    mesh = generate_fdtd_mesh_1d(fixed, optional, max_res=0.9, ratio=1.2)
#
#    # Should snap the 0.0 bisection to 0.05, then bisect the remaining halves
#    # Left half: [-1.0, 0.05] (size 1.05) -> split to 0.525
#    # Right half: [0.05, 1.0] (size 0.95) -> split to 0.475
#    expected = [-1.0, -0.3, 0.3, 1.0]
#
#    assert [pytest.approx(m, abs=1e-9) for m in mesh] == expected
#    assert -0.3 in [pytest.approx(m, abs=1e-9) for m in mesh]
#    assert 0.3 in [pytest.approx(m, abs=1e-9) for m in mesh]
#    validate_mesh_constraints(mesh, fixed, max_res=0.9, ratio=1.2)
#
#def test_ignore_out_of_bounds_optional():
#    """Ensures optional points outside the fixed domain are safely ignored."""
#    fixed = [0.0, 1.0]
#    optional = [-1.0, 0.5, 2.0] # -1.0 and 2.0 should be ignored
#    mesh = generate_fdtd_mesh_1d(fixed, optional, max_res=0.6, ratio=1.2)
#
#    assert mesh == [0.0, 0.5, 1.0]
#    validate_mesh_constraints(mesh, fixed, max_res=0.6, ratio=1.2)
#
#def test_complex_grading_cascade():
#    """Tests a highly disparate domain to ensure the ratio cascades correctly without hanging."""
#    fixed = [0.0, 10.0]
#    # Small forced cell at the start will force a ratio cascade all the way to max_res
#    fixed.insert(1, 0.1)
#
#    mesh = generate_fdtd_mesh_1d(fixed, optional_points=[], max_res=2.0, ratio=1.5)
#
#    validate_mesh_constraints(mesh, fixed, max_res=2.0, ratio=1.5)
#    # Ensure it successfully expanded up to max_res
#    dx = [mesh[i+1] - mesh[i] for i in range(len(mesh)-1)]
#    assert any(pytest.approx(val, abs=1e-2) == 2.0 for val in dx), "Mesh failed to scale up to max_res"
#
#def test_symmetric_non_uniform_mesh():
#    """Tests that a symmetric starting mesh results in a perfectly symmetric final mesh."""
#    positive_half = [0.0, 4.0, 4.5, 5.0, 5.5, 6.0, 12.0]
#    # Reconstruct the full symmetric domain
#    fixed = sorted(list(set([-x for x in positive_half] + positive_half)))
#
#    mesh = generate_fdtd_mesh_1d(fixed, optional_points=[], max_res=1.0, ratio=1.2)
#
#    # 1. Must satisfy all mathematical FDTD requirements
#    validate_mesh_constraints(mesh, fixed, max_res=1.0, ratio=1.2)
#
#    # 2. Must be perfectly symmetric around 0
#    for pt in mesh:
#        assert any(pytest.approx(-pt, abs=1e-9) == m for m in mesh), f"Symmetry broken: {pt} exists but {-pt} does not"
#
#def test_unordered_fixed_points():
#    """Tests that the function correctly handles unsorted fixed points."""
#    ordered_fixed = [-2.0, -1.0, 0.0, 1.0, 2.0]
#    unordered_fixed = [1.0, -2.0, 2.0, 0.0, -1.0]
#
#    mesh1 = generate_fdtd_mesh_1d(ordered_fixed, optional_points=[], max_res=0.5, ratio=1.2)
#    mesh2 = generate_fdtd_mesh_1d(unordered_fixed, optional_points=[], max_res=0.5, ratio=1.2)
#
#    assert mesh1 == mesh2
#    validate_mesh_constraints(mesh2, ordered_fixed, max_res=0.5, ratio=1.2)
#
#def test_multiple_optional_points():
#    """Tests the selection logic when multiple optional points are available in a gap."""
#    fixed = [0.0, 2.0]
#    # Max res is 1.0, so ideal step is 1.0.
#    # Optional points: 0.9 (step 0.9), 1.05 (invalid step > max_res)
#    # The algorithm should pick 0.9 because it's the largest valid step <= ideal_step.
#    optional = [0.5, 0.9, 1.05]
#
#    mesh = generate_fdtd_mesh_1d(fixed, optional, max_res=1.0, ratio=1.5)
#
#    validate_mesh_constraints(mesh, fixed, max_res=1.0, ratio=1.5)
#    assert any(pytest.approx(1.05, abs=1e-9) == m for m in mesh), "Failed to snap to the optimal optional point (1.05)"
#
#def test_symmetric_non_uniform_example_mesh():
#    """Tests that a symmetric starting mesh results in a perfectly symmetric final mesh."""
#    positive_half = [0.25, 0.45, 0.4, 0.35, 0.3, 5.0, 0.8500000000000001]
#    # Reconstruct the full symmetric domain
#    fixed = sorted(list(set([-x for x in positive_half] + positive_half)))
#
#    #raise RuntimeError(fixed)
#
#    mesh = generate_fdtd_mesh_1d(fixed, optional_points=[], max_res=1.7472369284948892, ratio=1.2)
#
#    # 1. Must satisfy all mathematical FDTD requirements
#    validate_mesh_constraints(mesh, fixed, max_res=1.7472369284948892, ratio=1.2)
#
#    # 2. Must be perfectly symmetric around 0
#    for pt in mesh:
#        assert any(pytest.approx(-pt, abs=1e-9) == m for m in mesh), f"Symmetry broken: {pt} exists but {-pt} does not"
#
#
#def test_openems_native_mesher_fails_constraints():
#    """
#    Tests that the native OpenEMS SmoothMeshLines function fails to maintain
#    the strict mathematical constraints (max_res, ratio) on a non-uniform starting mesh,
#    proving the need for our custom helper.
#    """
#    # This safely skips the test if CSXCAD is not installed (e.g., local development),
#    # but it will run perfectly inside the OpenEMS Docker container in CI.
#    CSXCAD = pytest.importorskip("CSXCAD")
#
#    positive_half = [0.25, 0.45, 0.4, 0.35, 0.3, 5.0, 0.8500000000000001]
#    fixed = sorted(list(set([-x for x in positive_half] + positive_half)))
#
#    # Run OpenEMS's native 1D smoother
#    openems_mesh = CSXCAD.SmoothMeshLines.SmoothMeshLines(fixed, 1.7472369284948892, 1.2)
#    openems_mesh = list(openems_mesh) # Ensure it is a standard Python list
#
#    # We expect the native OpenEMS mesher to violate our strict validation checks
#    # (usually a grading ratio violation when cascading across highly variable gaps).
#    with pytest.raises(AssertionError):
#        validate_mesh_constraints(openems_mesh, fixed, max_res=1.7472369284948892, ratio=1.2)
#
#def test_fuzz_mesh_generation():
#    """
#    Fuzz testing: Throws completely random FDTD geometries at the algorithm
#    hundreds of times to ensure it always converges and meets constraints without
#    triggering the RuntimeError circuit breaker.
#    """
#    # Use a random seed and report it so if it fails, it is reproducible for testing on the same random geometry,
#    # allowing to debug exactly what caused the failure.
#    seed = random.randint(1, sys.maxsize)
#    seed = 6178866328706153477
#    random.seed(seed)
#
#    for it in range(500):  # Run 500 completely random scenarios
#        num_fixed = random.randint(2, 15)
#        # Generate random fixed points spread across a wide domain
#        fixed = sorted(list(set([random.uniform(-50.0, 50.0) for _ in range(num_fixed)])))
#
#        # Sometimes provide no optional points, sometimes provide many
#        optional = [random.uniform(-50.0, 50.0) for _ in range(random.randint(0, 30))]
#
#        # Randomize the FDTD strict constraints
#        max_res = random.uniform(0.1, 5.0)
#        ratio = random.uniform(1.05, 1.8)  # Ratios from very strict (1.05) to very loose (1.8)
#
#        try:
#            # If the algorithm hits an infinite loop, our circuit breaker will raise a RuntimeError,
#            # causing the test to fail.
#            mesh = generate_fdtd_mesh_1d(fixed, optional, max_res, ratio)
#
#            # Validate that the resulting mesh mathematically satisfies all rules
#            validate_mesh_constraints(mesh, fixed, max_res, ratio)
#
#        except Exception as e:
#            # If it fails, print the exact parameters that caused the failure so you can debug it
#            pytest.fail(f"Fuzz test failed (iteration {it})!\nSeed: {seed}\nFixed: {fixed}\nOptional: {optional}\n"
#                        f"Max Res: {max_res}\nRatio: {ratio}\nError: {str(e)}")
#