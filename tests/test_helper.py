import pytest
from rf_sim.helper import generate_fdtd_mesh_1d

def validate_mesh_constraints(mesh, fixed_points, max_res, ratio):
    """
    Helper function to validate that a generated mesh mathematically
    satisfies all FDTD criteria. Use this in all future tests!
    """
    if not mesh:
        assert not fixed_points
        return

    # 1. Mesh must be monotonically increasing (sorted) and have no duplicates
    assert mesh == sorted(list(set(mesh))), "Mesh is not sorted or contains duplicates"

    # 2. Mesh must contain all original fixed points
    for pt in fixed_points:
        # Use approx to handle minor floating point inaccuracies
        assert any(pt == pytest.approx(m, abs=1e-9) for m in mesh), f"Fixed point {pt} is missing from mesh"

    # 3. Check spacing constraints (max_res and ratio)
    if len(mesh) > 1:
        dx = [mesh[i+1] - mesh[i] for i in range(len(mesh)-1)]

        for i, val in enumerate(dx):
            # Check max_res
            assert val <= max_res + 1e-9, f"Cell {i} size ({val}) exceeds max_res ({max_res})"

            # Check grading ratio against neighbors
            if i > 0:
                # Compare with previous cell
                assert val <= dx[i-1] * ratio + 1e-9, f"Ratio violation at cell {i}: {val} > {dx[i-1]} * {ratio}"
                assert dx[i-1] <= val * ratio + 1e-9, f"Ratio violation at cell {i-1}: {dx[i-1]} > {val} * {ratio}"

def test_empty_and_single_point():
    assert generate_fdtd_mesh_1d([], [], 0.9, 1.2) == []
    assert generate_fdtd_mesh_1d([1.0], [], 0.9, 1.2) == [1.0]

def test_trivial_bisection():
    """Tests the trivial case where the max_res is exactly half of the point spacing."""
    fixed = [-1.0, 1.0]
    mesh = generate_fdtd_mesh_1d(fixed, optional_points=[], max_res=1.0, ratio=1.2)

    expected = [-1.0, 0.0, 1.0]
    assert [pytest.approx(m, abs=1e-9) for m in mesh] == expected
    validate_mesh_constraints(mesh, fixed, max_res=1.0, ratio=1.2)

def test_basic_bisection():
    """Tests the first edge case: safely bisecting to avoid infinite loops."""
    fixed = [-1.0, 1.0]
    mesh = generate_fdtd_mesh_1d(fixed, optional_points=[], max_res=0.9, ratio=1.2)

    # Expected to safely split into 4 equal cells of 0.5
    expected = [-1.0, -0.5, 0.0, 0.5, 1.0]
    assert [pytest.approx(m, abs=1e-9) for m in mesh] == expected
    validate_mesh_constraints(mesh, fixed, max_res=0.9, ratio=1.2)

def test_smart_bisection_optional_point():
    """Tests the second edge case: pulling a bisection to a nearby optional point."""
    fixed = [-1.0, 1.0]
    optional = [0.05]
    mesh = generate_fdtd_mesh_1d(fixed, optional, max_res=0.9, ratio=1.2)

    # Should snap the 0.0 bisection to 0.05, then bisect the remaining halves
    # Left half: [-1.0, 0.05] (size 1.05) -> split to 0.525
    # Right half: [0.05, 1.0] (size 0.95) -> split to 0.475
    expected = [-1.0, -0.475, 0.05, 0.525, 1.0]

    assert [pytest.approx(m, abs=1e-9) for m in mesh] == expected
    assert 0.05 in [pytest.approx(m, abs=1e-9) for m in mesh]
    validate_mesh_constraints(mesh, fixed, max_res=0.9, ratio=1.2)

def test_ignore_out_of_bounds_optional():
    """Ensures optional points outside the fixed domain are safely ignored."""
    fixed = [0.0, 1.0]
    optional = [-1.0, 0.5, 2.0] # -1.0 and 2.0 should be ignored
    mesh = generate_fdtd_mesh_1d(fixed, optional, max_res=0.6, ratio=1.2)

    assert mesh == [0.0, 0.5, 1.0]
    validate_mesh_constraints(mesh, fixed, max_res=0.6, ratio=1.2)

def test_complex_grading_cascade():
    """Tests a highly disparate domain to ensure the ratio cascades correctly without hanging."""
    fixed = [0.0, 10.0]
    # Small forced cell at the start will force a ratio cascade all the way to max_res
    fixed.insert(1, 0.1)

    mesh = generate_fdtd_mesh_1d(fixed, optional_points=[], max_res=2.0, ratio=1.5)

    validate_mesh_constraints(mesh, fixed, max_res=2.0, ratio=1.5)
    # Ensure it successfully expanded up to max_res
    dx = [mesh[i+1] - mesh[i] for i in range(len(mesh)-1)]
    assert any(pytest.approx(val, abs=1e-2) == 2.0 for val in dx), "Mesh failed to scale up to max_res"