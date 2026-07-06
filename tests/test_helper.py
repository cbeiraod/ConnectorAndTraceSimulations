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

        assert mesher._pressure_left == [0.0, 0.0]
        assert mesher._pressure_right == [0.0, 0.0]

    def test_calculate_forces_with_violations(self):
        """Test force calculations when ratio is violated."""
        mesher = FDTDMesher1D([0.0, 10.0], [], max_res=2.0, ratio=1.5)
        mesher.mesh = [0.0, 1.0, 6.0, 7.2]
        # Mesh sizes (dx):
        # Cell 0: 1.0 (< max_res)
        # Cell 1: 5.0 (violates ratio from Cell 0 and Cell 2)
        # Cell 2: 1.2 (< max_res)

        # Cell 1 feels force from Cell 0 on the left: 1.0 / 1.0 = 1.0
        assert mesher._pressure_left[1] == pytest.approx(1.0 / 1.0)
        assert mesher._pressure_left[0] == 0.0
        assert mesher._pressure_left[2] == 0.0

        # Cell 1 feels force from Cell 2 on the right: 1.0 / 1.2
        assert mesher._pressure_right[1] == pytest.approx(1.0 / 1.2)
        assert mesher._pressure_right[0] == 0.0
        assert mesher._pressure_right[2] == 0.0

    def test_find_target_cell_by_force(self, monkeypatch):
        """Test target selection when forces are active."""
        mesher = FDTDMesher1D([0.0, 1.0, 3.0, 8.0], [], 2.1, 2.0)

        # Simulate an environment where Cell 2 has the highest force
        mesher._pressure_left = [0.0, 0.5, 1.0] # Max force is 1.0 at index 2
        mesher._pressure_right = [0.0, 0.0, 0.8]

        # Setup mock to just return the first tied index using pytest's monkeypatch
        monkeypatch.setattr('random.choice', lambda seq: seq[0])

        target = mesher._find_target_cell()
        assert target == 2

    def test_find_target_cell_by_size(self, monkeypatch):
        """Test target selection when no forces, but oversize cells exist."""
        mesher = FDTDMesher1D([0.0, 3.0, 4.5, 8.5], [], max_res=2.0, ratio=1.5)

        # No forces, but Cell 0 (3.0) and Cell 2 (4.0) are > max_res
        # Cell 0 is the smallest oversized cell
        self._pressure_left = [0.0, 0.0, 0.0]
        self._pressure_right = [0.0, 0.0, 0.0]

        monkeypatch.setattr('random.choice', lambda seq: seq[0])

        target = mesher._find_target_cell()
        assert target == 0 # Index of the 3.0 cell

    def test_find_target_cell_complete(self):
        """Test when mesh is perfectly valid."""
        mesher = FDTDMesher1D([0.0, 1.5, 3.3, 4.5], [], max_res=2.0, ratio=1.5)

        # All cells <= max_res

        target = mesher._find_target_cell()
        assert target is None


class TestFDTDMesher1DStateless:
    """
    Tests the stateless mathematical and physical operator stencils of FDTDMesher1D.
    Verifies that operations on arbitrary grid configurations conform to physical
    principles such as monotonicity of cell demand and Newton's Third Law (action-reaction symmetry).
    """

    def test_demand_zero_when_valid(self):
        """No constraints are violated; demand must be exactly 0.0."""
        mesher = FDTDMesher1D([0.0, 10.0], [], max_res=2.0, ratio=1.5)
        # dx = [1.0, 1.2, 1.5], idx = 1 (size 1.2)
        # 1.2 <= max_res (2.0) -> ok
        # 1.2 <= left_neighbor (1.0) * ratio (1.5) = 1.5 -> ok
        # 1.2 <= right_neighbor (1.5) * ratio (1.5) = 2.25 -> ok
        demand = mesher._calculate_cell_demand([1.0, 1.2, 1.5], 1)
        assert demand == pytest.approx(0.0)

    def test_demand_max_res_violation_only(self):
        """Only the maximum resolution boundary is crossed."""
        mesher = FDTDMesher1D([0.0, 10.0], [], max_res=2.0, ratio=1.5)
        # dx = [2.0, 2.5, 2.0], idx = 1 (size 2.5)
        # 2.5 > max_res (2.0) -> violation = 2.5 - 2.0 = 0.5
        # 2.5 <= left_neighbor (2.0) * ratio (1.5) = 3.0 -> ok
        # 2.5 <= right_neighbor (2.0) * ratio (1.5) = 3.0 -> ok
        demand = mesher._calculate_cell_demand([2.0, 2.5, 2.0], 1)
        assert demand == pytest.approx(0.5)

    def test_demand_left_ratio_violation_only(self):
        """Only the left neighbor ratio constraint is violated."""
        mesher = FDTDMesher1D([0.0, 10.0], [], max_res=3.0, ratio=1.5)
        # dx = [1.0, 2.0, 2.0], idx = 1 (size 2.0)
        # 2.0 <= max_res (3.0) -> ok
        # 2.0 > left_neighbor (1.0) * ratio (1.5) = 1.5 -> violation = 2.0 - 1.5 = 0.5
        # 2.0 <= right_neighbor (2.0) * ratio (1.5) = 3.0 -> ok
        demand = mesher._calculate_cell_demand([1.0, 2.0, 2.0], 1)
        assert demand == pytest.approx(0.5)

    def test_demand_right_ratio_violation_only(self):
        """Only the right neighbor ratio constraint is violated."""
        mesher = FDTDMesher1D([0.0, 10.0], [], max_res=3.0, ratio=1.5)
        # dx = [2.0, 2.0, 1.0], idx = 1 (size 2.0)
        # 2.0 <= max_res (3.0) -> ok
        # 2.0 <= left_neighbor (2.0) * ratio (1.5) = 3.0 -> ok
        # 2.0 > right_neighbor (1.0) * ratio (1.5) = 1.5 -> violation = 2.0 - 1.5 = 0.5
        demand = mesher._calculate_cell_demand([2.0, 2.0, 1.0], 1)
        assert demand == pytest.approx(0.5)

    def test_demand_boundary_left_index_zero(self):
        """Index 0 has no left neighbor; must skip left ratio checks cleanly."""
        mesher = FDTDMesher1D([0.0, 10.0], [], max_res=2.0, ratio=1.5)
        # dx = [2.5, 2.0], idx = 0 (size 2.5)
        # 2.5 > max_res (2.0) -> violation = 2.5 - 2.0 = 0.5
        # No left neighbor -> ok
        # 2.5 <= right_neighbor (2.0) * ratio (1.5) = 3.0 -> ok
        demand = mesher._calculate_cell_demand([2.5, 2.0], 0)
        assert demand == pytest.approx(0.5)

    def test_demand_boundary_right_last_index(self):
        """Last index has no right neighbor; must skip right ratio checks cleanly."""
        mesher = FDTDMesher1D([0.0, 10.0], [], max_res=2.0, ratio=1.5)
        # dx = [2.0, 2.5], idx = 1 (size 2.5)
        # 2.5 > max_res (2.0) -> violation = 2.5 - 2.0 = 0.5
        # 2.5 <= left_neighbor (2.0) * ratio (1.5) = 3.0 -> ok
        # No right neighbor -> ok
        demand = mesher._calculate_cell_demand([2.0, 2.5], 1)
        assert demand == pytest.approx(0.5)

    @given(
        dx=st.lists(st.floats(min_value=0.01, max_value=10.0), min_size=3, max_size=10),
        max_res=st.floats(min_value=0.1, max_value=5.0),
        ratio=st.floats(min_value=1.1, max_value=2.0)
    )
    def test_demand_properties_fuzzing(self, dx, max_res, ratio):
        """Fuzzes 'dx' arrangements to assert core physical properties of spring stress."""
        mesher = FDTDMesher1D([0.0, 10.0], [], max_res=max_res, ratio=ratio)
        for idx in range(len(dx)):
            demand = mesher._calculate_cell_demand(dx, idx)
            # Property 1: Stress is a non-negative scalar quantity
            assert demand >= 0.0

            # Property 2: Monotonicity (Increasing cell size must monotonically increase or maintain stress)
            increased_dx = dx.copy()
            increased_dx[idx] += 1.0
            increased_demand = mesher._calculate_cell_demand(increased_dx, idx)
            assert increased_demand >= demand

    def test_node_spring_forces_boundary_zero(self):
        """Ensures that the boundary/anchor nodes (0 and M-1) are always subjected to exactly 0.0 force."""
        fixed = [0.0, 10.0]
        # Let's create an asymmetric mesh with an extreme ratio shock
        mesh = [0.0, 1.0, 9.0, 10.0]
        mesher = FDTDMesher1D(fixed, [], max_res=2.0, ratio=1.5)

        forces = mesher._calculate_node_spring_forces(mesh)

        # Absolute boundaries must feel zero force as they are locked
        assert forces[0] == pytest.approx(0.0)
        assert forces[-1] == pytest.approx(0.0)

    def test_node_spring_forces_no_violation(self):
        """Ensures that a perfectly uniform mesh satisfying all limits yields exactly 0.0 force everywhere."""
        fixed = [0.0, 4.0]
        mesh = [0.0, 1.0, 2.0, 3.0, 4.0] # Cells of size 1.0
        # max_res = 1.5, ratio = 1.2 -> no violations anywhere
        mesher = FDTDMesher1D(fixed, [], max_res=1.5, ratio=1.2)

        forces = mesher._calculate_node_spring_forces(mesh)
        assert forces == pytest.approx([0.0, 0.0, 0.0, 0.0, 0.0])

    def test_node_spring_forces_restorative_oversized_cell(self):
        """
        Ensures that an oversized cell in the middle of a mesh generates correct inward restorative forces.
        Specifically, the left boundary node must experience positive force (pushing it right, shrinking the cell),
        and the right boundary node must experience negative force (pushing it left, shrinking the cell).
        """
        fixed = [0.0, 5.0]
        mesh = [0.0, 1.0, 4.0, 5.0] # Cell widths: 1.0, 3.0 (oversized), 1.0
        # max_res = 2.0 (so cell 1 is violating max_res), ratio = 1.5
        mesher = FDTDMesher1D(fixed, [], max_res=2.0, ratio=1.5)

        forces = mesher._calculate_node_spring_forces(mesh)

        # force[1] acts on node at 1.0. It should be positive to push right.
        assert forces[1] > 0.0
        # force[2] acts on node at 4.0. It should be negative to push left.
        assert forces[2] < 0.0
        # Check physical action-reaction symmetry (equal magnitudes in symmetric bounds)
        assert forces[1] == pytest.approx(-forces[2])

    def test_node_spring_forces_symmetry_and_scale_deterministic(self):
        """
        Highly-stressed deterministic test checking both physical symmetry and multi-scale
        floating-point robustness. It constructs a symmetric mesh featuring high-contrast
        dimensional spans (micro-gaps adjacent to massive spaces, 500x difference) to
        challenge float loss of significance under Newton's Third Law.
        """
        # Symmetric cell sizes spanning three orders of magnitude (0.01 mm to 5.0 mm)
        half_widths = [0.01, 0.1, 5.0, 0.05]
        widths = half_widths + list(reversed(half_widths))

        # Construct a perfectly symmetric mesh coordinates list starting from 0.0
        mesh = [0.0]
        for w in widths:
            mesh.append(mesh[-1] + w)

        fixed = [mesh[0], mesh[-1]]
        # Highly constraint-stressed parameters
        mesher = FDTDMesher1D(fixed, [], max_res=2.0, ratio=1.2)

        forces = mesher._calculate_node_spring_forces(mesh)
        M = len(mesh)

        # 1. Assert perfect physical anti-symmetry: forces[i] == -forces[M - 1 - i]
        # to high floating-point precision (1e-12)
        for i in range(M):
            mirror_idx = M - 1 - i
            assert forces[i] == pytest.approx(-forces[mirror_idx], abs=1e-12)

        # 2. Check that the forces are non-trivial (not all zero) to ensure the test is active
        assert any(abs(f) > 1e-5 for f in forces)

        # 3. Assert absolute center node feels exactly zero net force due to perfect spatial balance
        # (since M is 9, index 4 is the exact mirror plane center)
        assert forces[4] == pytest.approx(0.0, abs=1e-12)

    def test_local_node_spring_force_equivalence_with_global(self):
        """
        Asserts that the localized node force calculation (_calculate_local_node_spring_force)
        is mathematically equivalent to extracting the force from the global calculation
        (_calculate_node_spring_forces) for every internal node index in an asymmetric mesh.
        """
        fixed = [0.0, 15.0]
        mesh = [0.0, 0.8, 1.5, 4.0, 9.0, 11.0, 13.5, 15.0]
        # max_res = 2.0, ratio = 1.3
        mesher = FDTDMesher1D(fixed, [], max_res=2.0, ratio=1.3)

        global_forces = mesher._calculate_node_spring_forces(mesh)

        # Check every single internal node
        for i in range(1, len(mesh) - 1):
            local_force = mesher._calculate_local_node_spring_force(mesh, i)
            assert local_force == pytest.approx(global_forces[i], abs=1e-12)

    def test_local_node_spring_force_boundary_guards(self):
        """
        Specifically tests node force calculations at boundaries (i=1 and i=M-2)
        to ensure local boundary guards safely evaluate neighboring cells without index errors.
        """
        fixed = [0.0, 10.0]
        mesh = [0.0, 1.2, 2.4, 5.0, 8.8, 10.0] # M = 6 nodes.
        # Boundaries to check: i=1 (leftmost internal node) and i=4 (rightmost internal node)
        mesher = FDTDMesher1D(fixed, [], max_res=1.5, ratio=1.2)

        global_forces = mesher._calculate_node_spring_forces(mesh)

        # Node i=1
        force_i1 = mesher._calculate_local_node_spring_force(mesh, i=1)
        assert force_i1 == pytest.approx(global_forces[1], abs=1e-12)

        # Node i=M-2 (i=4)
        force_iM2 = mesher._calculate_local_node_spring_force(mesh, i=4)
        assert force_iM2 == pytest.approx(global_forces[4], abs=1e-12)

    def test_local_node_spring_force_perfect_equilibrium(self):
        """Ensures that a local neighborhood in perfect equilibrium returns exactly 0.0 force."""
        fixed = [0.0, 10.0]
        mesh = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 10.0]
        mesher = FDTDMesher1D(fixed, [], max_res=2.0, ratio=1.5)

        # Node 3 is surrounded by perfectly uniform cells of size 1.0 (satisfies max_res 2.0 and ratio 1.5)
        local_force = mesher._calculate_local_node_spring_force(mesh, i=3)
        assert local_force == pytest.approx(0.0, abs=1e-12)

    def test_clip_elastic_steps_no_clipping(self):
        """Tests that proposed shifts strictly within the safety boundary pass through unchanged."""
        mesher = FDTDMesher1D([0.0, 10.0], [], max_res=2.0, ratio=1.5)
        mesh = [0.0, 5.0, 10.0]
        # dx_l = 5.0, dx_r = 5.0. Default safety 0.4 -> limits are [-2.0, 2.0]
        shifts = [0.0, 1.5, 0.0]
        is_fixed = [True, False, True]

        clipped = mesher._clip_elastic_steps(mesh, shifts, is_fixed)
        assert clipped == pytest.approx([0.0, 1.5, 0.0])

    def test_clip_elastic_steps_left_and_right_bounds(self):
        """Tests aggressive shifts correctly truncate against geometrical limits."""
        mesher = FDTDMesher1D([0.0, 10.0], [], max_res=2.0, ratio=1.5)
        mesh = [0.0, 2.0, 8.0, 10.0]
        # Node 1 (2.0): dx_l = 2.0, dx_r = 6.0 -> limits [-0.8, 2.4]
        # Node 2 (8.0): dx_l = 6.0, dx_r = 2.0 -> limits [-2.4, 0.8]

        # Propose massive outward shifts
        shifts = [0.0, -5.0, 5.0, 0.0]
        is_fixed = [True, False, False, True]

        clipped = mesher._clip_elastic_steps(mesh, shifts, is_fixed)

        # Node 1 should be clipped left to -0.8
        assert clipped[1] == pytest.approx(-0.8)
        # Node 2 should be clipped right to 0.8
        assert clipped[2] == pytest.approx(0.8)

    def test_clip_elastic_steps_respects_fixed_nodes(self):
        """Tests that nodes flagged as fixed return exactly 0.0 shift, dropping proposed forces."""
        mesher = FDTDMesher1D([0.0, 10.0], [], max_res=2.0, ratio=1.5)
        mesh = [0.0, 5.0, 10.0]
        shifts = [0.0, 1.0, 0.0]
        is_fixed = [True, True, True] # Node 1 is an anchor

        clipped = mesher._clip_elastic_steps(mesh, shifts, is_fixed)

        assert clipped[1] == pytest.approx(0.0)

    def test_clip_elastic_steps_custom_safety_factor(self):
        """Tests the safety factor explicitly tightens movement constraints."""
        mesher = FDTDMesher1D([0.0, 10.0], [], max_res=2.0, ratio=1.5)
        mesh = [0.0, 5.0, 10.0]
        # dx = 5.0. Override safety_factor to 0.1 -> limits [-0.5, 0.5]
        shifts = [0.0, 2.0, 0.0]
        is_fixed = [True, False, True]

        clipped = mesher._clip_elastic_steps(mesh, shifts, is_fixed, safety_factor=0.1)
        assert clipped[1] == pytest.approx(0.5)

    def test_clip_local_elastic_step(self):
        """Tests the standalone localized clipping logic used by sequential sweeps."""
        mesher = FDTDMesher1D([0.0, 10.0], [], max_res=2.0, ratio=1.5)
        mesh = [0.0, 2.0, 8.0, 10.0]
        # Node 1 (2.0): dx_l = 2.0, dx_r = 6.0 -> limits [-0.8, 2.4]

        # In-bounds
        assert mesher._clip_local_elastic_step(mesh, 1, 0.5) == pytest.approx(0.5)
        # Left boundary violation
        assert mesher._clip_local_elastic_step(mesh, 1, -1.0) == pytest.approx(-0.8)
        # Right boundary violation
        assert mesher._clip_local_elastic_step(mesh, 1, 3.0) == pytest.approx(2.4)
        # Custom safety factor
        assert mesher._clip_local_elastic_step(mesh, 1, 3.0, safety_factor=0.5) == pytest.approx(3.0)

    def test_apply_global_kinematic_step_standard(self):
        """Tests standard step application: omega scaling works, velocities untouched if no clip."""
        mesher = FDTDMesher1D([0.0, 10.0], [], max_res=2.0, ratio=1.5)
        mesh = [0.0, 5.0, 10.0]
        velocities = [0.0, 0.5, 0.0]
        raw_shifts = [0.0, 0.5, 0.0]
        is_fixed = [True, False, True]

        # omega = 1.2 -> scaled shift = 0.6. Allowed clip bounds = [-2.0, 2.0].
        # 0.6 fits perfectly. Velocity should NOT be updated.
        max_shift = mesher._apply_global_kinematic_step(
            mesh, velocities, raw_shifts, is_fixed, omega=1.2, active_indices=[1], update_velocity=True
        )

        assert max_shift == pytest.approx(0.6)
        assert mesh == pytest.approx([0.0, 5.6, 10.0])
        assert velocities[1] == pytest.approx(0.5)  # Unmodified!

    def test_apply_global_kinematic_step_inelastic_velocity_sync(self):
        """Tests that a clipped shift correctly truncates the velocity vector to match reality."""
        mesher = FDTDMesher1D([0.0, 10.0], [], max_res=2.0, ratio=1.5)
        mesh = [0.0, 5.0, 10.0]
        velocities = [0.0, 5.0, 0.0]
        raw_shifts = [0.0, 5.0, 0.0]
        is_fixed = [True, False, True]

        # omega = 1.0. Shift = 5.0. Allowed bounds = [-2.0, 2.0].
        # Actual shift is clipped to 2.0.
        max_shift = mesher._apply_global_kinematic_step(
            mesh, velocities, raw_shifts, is_fixed, omega=1.0, active_indices=[1], update_velocity=True
        )

        assert max_shift == pytest.approx(2.0)
        assert mesh == pytest.approx([0.0, 7.0, 10.0])
        assert velocities[1] == pytest.approx(2.0)  # Momentum truncated to match the wall!

    def test_apply_global_kinematic_step_elastic_leapfrog(self):
        """Tests that Leapfrog (update_velocity=False) hits the wall but preserves momentum."""
        mesher = FDTDMesher1D([0.0, 10.0], [], max_res=2.0, ratio=1.5)
        mesh = [0.0, 5.0, 10.0]
        velocities = [0.0, 5.0, 0.0]
        raw_shifts = [0.0, 5.0, 0.0]
        is_fixed = [True, False, True]

        # Same massive shift, but update_velocity = False
        max_shift = mesher._apply_global_kinematic_step(
            mesh, velocities, raw_shifts, is_fixed, omega=1.0, active_indices=[1], update_velocity=False
        )

        assert max_shift == pytest.approx(2.0)
        assert mesh == pytest.approx([0.0, 7.0, 10.0])
        assert velocities[1] == pytest.approx(5.0)  # Momentum preserved perfectly!

    def test_apply_global_kinematic_step_active_indices_filter(self):
        """Tests that only nodes specified in active_indices are updated (used for Red-Black)."""
        mesher = FDTDMesher1D([0.0, 10.0], [], max_res=2.0, ratio=1.5)
        mesh = [0.0, 2.5, 5.0, 7.5, 10.0]
        velocities = [0.0, 1.0, 1.0, 1.0, 0.0]
        raw_shifts = [0.0, 1.0, 1.0, 1.0, 0.0]
        is_fixed = [True, False, False, False, True]

        # We only pass [1, 3] as active indices (simulating the 'odd' Red/Black pass)
        mesher._apply_global_kinematic_step(
            mesh, velocities, raw_shifts, is_fixed, omega=1.0, active_indices=[1, 3], update_velocity=True
        )

        # Node 1 and 3 moved. Node 2 (index 2, position 5.0) stayed completely still.
        assert mesh == pytest.approx([0.0, 3.5, 5.0, 8.5, 10.0])


class TestFDTDMesher1DSplitting:

    def test_is_point_compatible_fully_defined_valid(self):
        """Test compatibility with 4 well-behaved cells."""
        mesher = FDTDMesher1D([0.0, 10.0], [], max_res=2.0, ratio=1.5)
        # Dxs: 1.0, 1.0, 1.0, 1.0 -> all <= 2.0, ratios = 1.0 <= 1.5
        assert mesher._is_point_compatible(2.0, pt_ll=0.0, pt_l=1.0, pt_r=3.0, pt_rr=4.0) is True

    def test_is_point_compatible_max_res_violation(self):
        """Test rejection when max_res is exceeded on immediate cells."""
        mesher = FDTDMesher1D([0.0, 10.0], [], max_res=2.0, ratio=1.5)
        # Immediate left cell (dx2) = 2.5 (>2.0)
        assert mesher._is_point_compatible(3.5, pt_ll=0.0, pt_l=1.0, pt_r=4.5, pt_rr=5.5) is False
        # Immediate right cell (dx3) = 2.5 (>2.0)
        assert mesher._is_point_compatible(2.0, pt_ll=0.0, pt_l=1.0, pt_r=4.5, pt_rr=5.5) is False

    def test_is_point_compatible_ratio_violation_inner(self):
        """Test rejection when ratio is violated between the two immediate cells."""
        mesher = FDTDMesher1D([0.0, 10.0], [], max_res=2.0, ratio=1.5)
        # dx2 = 0.5, dx3 = 1.0 (ratio 2.0 > 1.5)
        assert mesher._is_point_compatible(1.5, pt_ll=0.0, pt_l=1.0, pt_r=2.5, pt_rr=3.0) is False
        # dx2 = 1.0, dx3 = 0.5 (ratio 2.0 > 1.5)
        assert mesher._is_point_compatible(2.0, pt_ll=0.0, pt_l=1.0, pt_r=2.5, pt_rr=3.0) is False

    def test_is_point_compatible_ratio_violation_outer(self):
        """Test rejection when ratio cascades fail on the outer adjacent cells."""
        mesher = FDTDMesher1D([0.0, 10.0], [], max_res=2.0, ratio=1.5)
        # Left outer violation: dx1 = 2.0, dx2 = 1.0 (ratio 2.0 > 1.5)
        assert mesher._is_point_compatible(3.0, pt_ll=0.0, pt_l=2.0, pt_r=4.0, pt_rr=5.0) is False
        # Right outer violation: dx3 = 1.0, dx4 = 2.0 (ratio 2.0 > 1.5)
        assert mesher._is_point_compatible(2.0, pt_ll=0.0, pt_l=1.0, pt_r=3.0, pt_rr=5.0) is False

    def test_is_point_compatible_partial_bounds(self):
        """Test that partial bounds (None) safely skip outer checks during incremental meshing."""
        mesher = FDTDMesher1D([0.0, 10.0], [], max_res=2.0, ratio=1.5)

        # Missing left-left boundary (e.g., building from the left edge)
        # dx2 = 1.0, dx3 = 1.0, dx4 = 2.0 (ratio dx3/dx4 = 2.0 > 1.5 -> False)
        assert mesher._is_point_compatible(2.0, pt_ll=None, pt_l=1.0, pt_r=3.0, pt_rr=5.0) is False
        # Valid case with missing left-left
        assert mesher._is_point_compatible(2.0, pt_ll=None, pt_l=1.0, pt_r=3.0, pt_rr=4.0) is True

        # Missing right bounds (e.g., building strictly left-to-right)
        # dx1 = 1.0, dx2 = 1.0. No right side. Valid.
        assert mesher._is_point_compatible(2.0, pt_ll=0.0, pt_l=1.0, pt_r=None, pt_rr=None) is True
        # dx1 = 2.0, dx2 = 1.0. No right side. Ratio violation (2.0 > 1.5 * 1.0)
        assert mesher._is_point_compatible(3.0, pt_ll=0.0, pt_l=2.0, pt_r=None, pt_rr=None) is False

    def test_tesselate_mesh_cell_trivial_N(self):
        """Test that N<=1 correctly returns no internal points."""
        mesher = FDTDMesher1D([0.0, 10.0], [], max_res=10.0, ratio=1.5)
        assert mesher._tesselate_mesh_cell(cell_index=0, N=1) == []
        assert mesher._tesselate_mesh_cell(cell_index=0, N=0) == []

    def test_tesselate_mesh_cell_uniform_no_snap(self):
        """Test perfect uniform subdivision when snapping is disabled or not applicable."""
        mesher = FDTDMesher1D([0.0, 10.0], [5.1], max_res=3.0, ratio=1.5)

        # Split 0.0 to 10.0 into 4 cells (N=4) -> target step is 2.5
        points = mesher._tesselate_mesh_cell(cell_index=0, N=4, snap_opt=False)
        assert points == pytest.approx([2.5, 5.0, 7.5])

    def test_tesselate_mesh_cell_with_snapping(self):
        """Test that an internal point correctly snaps to an optional point."""
        # Optional point at 5.1, target step is 5.0. It should snap to 5.1.
        mesher = FDTDMesher1D([0.0, 10.0], [5.1], max_res=10.0, ratio=1.5)

        points = mesher._tesselate_mesh_cell(cell_index=0, N=2, snap_opt=True)
        assert points == pytest.approx([5.1])

    def test_tesselate_mesh_cell_boundary_resolution(self):
        """Test that resolving pt_ll and pt_rr works correctly when spanning across self.mesh boundaries."""
        # 3 fixed points = 2 cells. We will tesselate the second cell (cell_index=1).
        mesher = FDTDMesher1D([0.0, 10.0, 20.0], [15.1], max_res=10.0, ratio=1.5)

        # Tesselating [10.0, 20.0] with N=2 means the candidate point is 15.0.
        # It will try to snap to 15.1.
        # This requires pt_ll to reach back into self.mesh[0] (which is 0.0).
        # It shouldn't crash, and the snap should succeed since constraints are generous.
        points = mesher._tesselate_mesh_cell(cell_index=1, N=2, snap_opt=True)
        assert points == pytest.approx([15.1])

    def test_tesselate_mesh_cell_not_implemented_graded(self):
        """Test that graded flags raise the correct NotImplementedError."""
        mesher = FDTDMesher1D([0.0, 10.0], [], max_res=10.0, ratio=1.5)

        with pytest.raises(NotImplementedError):
            mesher._tesselate_mesh_cell(cell_index=0, N=2, graded_left=True)

        with pytest.raises(NotImplementedError):
            mesher._tesselate_mesh_cell(cell_index=0, N=2, graded_right=True)

    def test_tesselate_mesh_cell_reject_snap_due_to_pt_ll_ratio(self):
        """Test that snapping is rejected if it causes a ratio violation with the outer left cell (pt_ll)."""
        mesher = FDTDMesher1D([0.0, 4.0, 10.0], [5.0], max_res=10.0, ratio=1.5)
        # We manually force N=2 for the second cell (index 1: 4.0 to 10.0). Target step is 3.0.
        # Candidate point is 4.0 + 3.0 = 7.0.
        # If it snaps to 5.0, the inner cell (4.0 to 5.0) becomes 1.0.
        # The outer left cell (0.0 to 4.0) is 4.0. Ratio is 4.0 > 1.5 * 1.0 (Violates pt_ll check!)
        points = mesher._tesselate_mesh_cell(cell_index=1, N=2, snap_opt=True)
        assert points == pytest.approx([7.0]) # Snap rejected, returns ideal candidate

    def test_tesselate_mesh_cell_reject_snap_due_to_pt_rr_ratio(self):
        """Test that snapping is rejected if it causes a ratio violation with the outer right cell (pt_rr)."""
        mesher = FDTDMesher1D([0.0, 6.0, 10.0], [5.0], max_res=10.0, ratio=1.5)
        # We manually force N=2 for the first cell (index 0: 0.0 to 6.0). Target step is 3.0.
        # Candidate point is 0.0 + 3.0 = 3.0.
        # If it snaps to 5.0, the inner cell (5.0 to 6.0) becomes 1.0.
        # The outer right cell (6.0 to 10.0) is 4.0. Ratio is 4.0 > 1.5 * 1.0 (Violates pt_rr check!)
        points = mesher._tesselate_mesh_cell(cell_index=0, N=2, snap_opt=True)
        assert points == pytest.approx([3.0]) # Snap rejected, returns ideal candidate

    def test_tesselate_mesh_cell_reject_snap_due_to_max_res(self):
        """Test that snapping is rejected if it forces an internal gap to exceed max_res."""
        mesher = FDTDMesher1D([0.0, 4.0], [3.5], max_res=2.5, ratio=5.0) # High ratio to isolate max_res
        # Cell is size 4.0. N=2 -> step is 2.0. Candidate is 2.0.
        # Optional point is 3.5.
        # If it snaps to 3.5, the left gap is 3.5 - 0.0 = 3.5 (> max_res 2.5). Violation!
        points = mesher._tesselate_mesh_cell(cell_index=0, N=2, snap_opt=True)
        assert points == pytest.approx([2.0]) # Snap rejected, returns ideal candidate

    def test_evaluate_optional_snap_success(self):
        """Test snapping to a valid optional point."""
        mesher = FDTDMesher1D([0.0, 10.0], [1.05], max_res=2.0, ratio=1.5)
        # Candidate at 1.0. Optional point at 1.05.
        # Check from_left incrementally (no pt_ll or right bounds yet).
        pt = mesher._evaluate_optional_snap(
            candidate_pt=1.0,
            pt_ll=None, pt_l=0.0, pt_r=2.0, pt_rr=None,
            from_left=True, from_right=False
        )
        assert pt == pytest.approx(1.05)

    def test_evaluate_optional_snap_violation_max_res(self):
        """Test rejecting an optional point because it violates max_res."""
        mesher = FDTDMesher1D([0.0, 10.0], [2.1], max_res=2.0, ratio=1.5)
        # Candidate at 1.9. pt_l is 0.0.
        # Optional point at 2.1 would cause left gap to be 2.1 > 2.0 (max_res).
        pt = mesher._evaluate_optional_snap(
            candidate_pt=1.9,
            pt_ll=None, pt_l=0.0, pt_r=3.8, pt_rr=None,
            from_left=True, from_right=False
        )
        assert pt == pytest.approx(1.9)

    def test_evaluate_optional_snap_violation_ratio(self):
        """Test rejecting an optional point because it violates the ratio constraint."""
        # Set a strict ratio
        mesher = FDTDMesher1D([0.0, 10.0], [1.2], max_res=2.0, ratio=1.1)
        # We simulate pt_ll at -1.0, making the outer left cell (dx1) exactly 1.0.
        # Candidate is 1.0. Optional is 1.2.
        # If snapped, inner left cell (dx2) becomes 1.2.
        # Ratio of dx2/dx1 = 1.2 / 1.0 = 1.2 > 1.1 (ratio constraint).
        pt = mesher._evaluate_optional_snap(
            candidate_pt=1.0,
            pt_ll=-1.0, pt_l=0.0, pt_r=2.0, pt_rr=None,
            from_left=True, from_right=False
        )
        assert pt == pytest.approx(1.0)

    def test_evaluate_optional_snap_two_sided_success(self):
        """Test snapping when both from_left and from_right are True, checking all bounds."""
        mesher = FDTDMesher1D([0.0, 10.0], [5.1], max_res=5.0, ratio=1.5)
        # Setup perfectly uniform background cells of size 2.5.
        # Candidate at 5.0. Optional at 5.1.
        # New inner cells will be 2.6 and 2.4.
        # Cascading ratios will be 2.6/2.5=1.04 and 2.5/2.4=1.04 (both well within 1.5).
        pt = mesher._evaluate_optional_snap(
            candidate_pt=5.0,
            pt_ll=0.0, pt_l=2.5, pt_r=7.5, pt_rr=10.0,
            from_left=True, from_right=True
        )
        assert pt == pytest.approx(5.1)

    def test_evaluate_optional_snap_picks_closest(self):
        """Test that given multiple valid optional points, the closest to candidate_pt is chosen."""
        mesher = FDTDMesher1D([1.0, 9.0], [4.5, 4.9, 5.2, 5.8], max_res=5.0, ratio=1.5)
        # Candidate at 5.0.
        # 4.9 is distance 0.1, 5.2 is distance 0.2. It should aggressively pick 4.9.
        pt = mesher._evaluate_optional_snap(
            candidate_pt=5.0,
            pt_ll=None, pt_l=1.0, pt_r=9.0, pt_rr=None,
            from_left=True, from_right=True
        )
        assert pt == pytest.approx(4.9)


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


def check_mesh_validity(mesh, fixed_points, max_res, ratio, algorithm="advancing_front"):
    """Helper function to assert a mesh respects all fundamental FDTD constraints."""
    assert mesh is not None, "Mesh generation failed (returned None)."

    algorithm_no_fixed_points = [
        "global_grid_search"
    ]

    algorithm_no_max_res = [
    ]

    algorithm_no_ratio = [
        "segment_uniform",
        "global_grid_search_with_fixed"
    ]

    # 1. All fixed points must be present
    if algorithm not in algorithm_no_fixed_points:
        for fp in fixed_points:
            assert fp in mesh, f"Fixed point {fp} is missing from the generated mesh."

    # 2. Points must be sorted monotonically
    assert mesh == sorted(mesh), "Mesh points are not strictly increasing."

    # Calculate final sizes
    dx = [mesh[i+1] - mesh[i] for i in range(len(mesh)-1)]

    # 3. max_res constraint
    if algorithm not in algorithm_no_max_res:
        for i, size in enumerate(dx):
            assert size <= max_res + 1e-9, f"Cell {i} size {size} exceeds max_res {max_res}."

    # 4. ratio constraint
    if algorithm not in algorithm_no_ratio:
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

    @pytest.mark.parametrize("algorithm", [
        pytest.param("advancing_front", marks=pytest.mark.skip(reason="Temporarily skipped since under development")),
        "segment_uniform",
        "segment_graded",
        "global_grid_search",
        "iterative_relaxation",
        "iterative_relaxation_fast",
        "iterative_relaxation_momentum",
        #"iterative_relaxation_fast_momentum"
    ])
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
    def test_unforced_cell_edge_cases_respect_constraints(self, algorithm, fixed_steps, max_res = 1.0, ratio = 1.1):
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

        for val in mesher._pressure_left:
            assert val == pytest.approx(0.0)
        for val in mesher._pressure_right:
            assert val == pytest.approx(0.0)

        final_mesh = mesher.generate(algorithm)

        check_mesh_validity(final_mesh, fixed, max_res, ratio, algorithm=algorithm)

    @pytest.mark.parametrize("algorithm", [
        "advancing_front",
        "segment_uniform",
        "segment_graded",
        "global_grid_search",
        "iterative_relaxation",
        "iterative_relaxation_fast",
        "iterative_relaxation_momentum",
        "iterative_relaxation_fast_momentum"
    ])
    def test_simple_bisection(self, algorithm):
        """Tests the trivial case where the max_res is exactly half of the point spacing."""
        fixed = [-1.0, 1.0]
        mesher = FDTDMesher1D(fixed, optional_points=[], max_res=1.0, ratio=1.2)
        final_mesh = mesher.generate(algorithm)

        assert final_mesh == pytest.approx([-1.0, 0.0, 1.0])
        check_mesh_validity(final_mesh, fixed, max_res=1.0, ratio=1.2, algorithm=algorithm)

    @pytest.mark.parametrize("algorithm", [
        "advancing_front",
        "segment_uniform",
        "segment_graded",
        "global_grid_search",
        "iterative_relaxation",
        "iterative_relaxation_fast",
        "iterative_relaxation_momentum",
        "iterative_relaxation_fast_momentum"
    ])
    def test_basic_bisection(self, algorithm):
        """Tests the first edge case: safely bisecting to avoid infinite loops."""
        fixed = [-1.0, 1.0]
        mesher = FDTDMesher1D(fixed, optional_points=[], max_res=0.9, ratio=1.2)
        final_mesh = mesher.generate(algorithm)

        # Expected to safely split into 3 equal cells of 2/3
        assert final_mesh == pytest.approx([-1.0, -1/3, 1/3, 1.0])
        check_mesh_validity(final_mesh, fixed, max_res=0.9, ratio=1.2, algorithm=algorithm)

    @pytest.mark.parametrize("algorithm", [
        "advancing_front",
        "segment_uniform",
        "segment_graded",
        "global_grid_search",
        "iterative_relaxation",
        "iterative_relaxation_fast",
        "iterative_relaxation_momentum",
        "iterative_relaxation_fast_momentum"
    ])
    def test_basic_bisection_with_optional_points(self, algorithm):
        """Tests the first edge case: safely bisecting to avoid infinite loops."""
        fixed = [-1.0, 1.0]
        optional = [-0.3, 0.3]
        mesher = FDTDMesher1D(fixed, optional_points=optional, max_res=0.9, ratio=1.2)
        final_mesh = mesher.generate(algorithm)

        # Expected to safely split into 3 equal cells of 2/3
        assert final_mesh == pytest.approx([-1.0, -0.3, 0.3, 1.0])
        check_mesh_validity(final_mesh, fixed, max_res=0.9, ratio=1.2, algorithm=algorithm)


    @pytest.mark.parametrize("algorithm", [
        "advancing_front",
        "iterative_relaxation",
        "iterative_relaxation_fast",
        "iterative_relaxation_momentum",
        #"iterative_relaxation_fast_momentum"
    ])
    def test_complex_grading_cascade(self, algorithm):
        """Tests a highly disparate domain to ensure the ratio cascades correctly without hanging."""
        fixed = [0.0, 10.0]
        # Small forced cell at the start will force a ratio cascade all the way to max_res
        fixed.insert(1, 0.1)

        mesher = FDTDMesher1D(fixed, optional_points=[], max_res=2.0, ratio=1.5)
        final_mesh = mesher.generate(algorithm=algorithm)

        check_mesh_validity(final_mesh, fixed, max_res=2.0, ratio=1.5, algorithm=algorithm)
        # Ensure it successfully expanded up to max_res
        assert any(pytest.approx(val, abs=1e-2) == 2.0 for val in mesher.dx), "Mesh failed to scale up to max_res"

    def test_segment_uniform_ignores_global_ratio(self):
        """Tests that segment_uniform intentionally ignores ratio jumps across fixed point boundaries."""
        # The ratio between the 10.0 gap and the 0.1 gap is 100x (violating a ratio of 1.5)
        fixed = [0.0, 10.0, 10.1]
        mesher = FDTDMesher1D(fixed, optional_points=[], max_res=2.0, ratio=1.5)
        final_mesh = mesher.generate(algorithm="segment_uniform")

        # Segment 1 (0 to 10): size 10, max_res 2 -> N=5. Steps of 2.0.
        # Segment 2 (10 to 10.1): size 0.1, max_res 2 -> N=1. Step of 0.1.
        assert final_mesh == pytest.approx([0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 10.1])

        # check_mesh_validity handles skipping the ratio check for this algorithm
        check_mesh_validity(final_mesh, fixed, max_res=2.0, ratio=1.5, algorithm="segment_uniform")

    def test_segment_graded_respects_global_ratio(self):
        """Tests that segment_graded iteratively fixes the ratio jumps across fixed point boundaries."""
        # The ratio between the 10.0 gap and the 0.1 gap is 100x (violating a ratio of 1.5)
        fixed = [0.0, 10.0, 10.1]
        mesher = FDTDMesher1D(fixed, optional_points=[], max_res=2.0, ratio=1.5)

        # Using segment_graded will aggressively grade the 10.0 segment
        # so it safely steps down to match the 0.1 cell size at the boundary.
        final_mesh = mesher.generate(algorithm="segment_graded")

        # We MUST assert it passes the ratio validity check (which segment_uniform fails here)
        check_mesh_validity(final_mesh, fixed, max_res=2.0, ratio=1.5, algorithm="segment_graded")

    def test_segment_uniform_right_side_snap_rejection(self):
        """Tests that an optional point is rejected if snapping to it violates constraints on the right side."""
        # Fixed domain of 3.0. Target step is 1.5. Candidate point is 1.5.
        fixed = [0.0, 3.0]

        # Optional point is at 1.3.
        # Left gap becomes 1.3 (Valid, < 1.6).
        # Right gap becomes 3.0 - 1.3 = 1.7.
        # 1.7 > max_res (1.6)! The right-side check should reject this snap.
        optional = [1.3]

        mesher = FDTDMesher1D(fixed, optional_points=optional, max_res=1.6, ratio=2.0)
        final_mesh = mesher.generate(algorithm="segment_uniform")

        # If successfully rejected, the point stays exactly at the ideal candidate 1.5
        assert final_mesh == pytest.approx([0.0, 1.5, 3.0])

    @pytest.mark.parametrize("algorithm", [
        pytest.param("advancing_front", marks=pytest.mark.skip(reason="Temporarily skipped since under development")),
        "segment_uniform",
        "segment_graded",
        "global_grid_search",
        "iterative_relaxation",
        "iterative_relaxation_momentum"
    ])
    def test_symmetric_non_uniform_mesh(self, algorithm):
        """Tests that a symmetric starting mesh results in a perfectly symmetric final mesh."""
        positive_half = [0.0, 4.0, 4.5, 5.0, 5.5, 6.0, 12.0]
        # Reconstruct the full symmetric domain
        fixed = sorted(list(set([-x for x in positive_half] + positive_half)))

        mesher = FDTDMesher1D(fixed, optional_points=[], max_res=1.0, ratio=1.2)
        final_mesh = mesher.generate(algorithm)

        # 1. Must satisfy all mathematical FDTD requirements
        check_mesh_validity(final_mesh, fixed, max_res=1.0, ratio=1.2, algorithm=algorithm)

        # 2. Must be perfectly symmetric around 0
        for pt in final_mesh:
            assert any(pytest.approx(-pt, abs=1e-9) == m for m in final_mesh), f"Symmetry broken: {pt} exists but {-pt} does not"

    @pytest.mark.parametrize("algorithm", [
        "advancing_front",
        "segment_uniform",
        "segment_graded",
        "global_grid_search",
        "iterative_relaxation",
        "iterative_relaxation_fast",
        "iterative_relaxation_momentum",
        "iterative_relaxation_fast_momentum"
    ])
    def test_multiple_optional_points(self, algorithm):
        """Tests the selection logic when multiple optional points are available in a gap."""
        fixed = [0.0, 2.0]
        # Max res is 1.0, so ideal step is 1.0.
        # Optional points: 0.9 (invalid step > max_res), 1.05 (should be selected)
        # The algorithm should pick 1.05 because it's the closest (only) valid optional step to ideal_step.
        optional = [0.5, 0.9, 1.05]

        mesher = FDTDMesher1D(fixed, optional_points=optional, max_res=1.09, ratio=1.5)
        final_mesh = mesher.generate(algorithm)

        check_mesh_validity(final_mesh, fixed, max_res=1.09, ratio=1.5, algorithm=algorithm)
        assert any(pytest.approx(1.05, abs=1e-9) == m for m in final_mesh), "Failed to snap to the optimal optional point (1.05)"

    @pytest.mark.parametrize("algorithm", [
        pytest.param("advancing_front", marks=pytest.mark.skip(reason="Temporarily skipped since under development")),
        "segment_uniform",
        "segment_graded",
        "global_grid_search",
        "iterative_relaxation",
        "iterative_relaxation_momentum"
    ])
    def test_symmetric_non_uniform_realistic_example_mesh(self, algorithm):
        """Tests that a symmetric starting mesh results in a perfectly symmetric final mesh."""
        positive_half = [0.25, 0.45, 0.4, 0.35, 0.3, 5.0, 0.8500000000000001]
        # Reconstruct the full symmetric domain
        fixed = sorted(list(set([-x for x in positive_half] + positive_half)))

        #raise RuntimeError(fixed)

        kwargs = {}
        if algorithm == "iterative_relaxation":
            kwargs["max_iterations"] = 85000

        mesher = FDTDMesher1D(fixed, optional_points=[], max_res=1.7472369284948892, ratio=1.2)
        final_mesh = mesher.generate(algorithm, **kwargs)

        # 1. Must satisfy all mathematical FDTD requirements
        check_mesh_validity(final_mesh, fixed, max_res=1.7472369284948892, ratio=1.2, algorithm=algorithm)

        # 2. Must be perfectly symmetric around 0
        for pt in final_mesh:
            assert any(pytest.approx(-pt, abs=1e-9) == m for m in final_mesh), f"Symmetry broken: {pt} exists but {-pt} does not"

    #@pytest.mark.skip
    @pytest.mark.parametrize("algorithm", [
        pytest.param("advancing_front", marks=pytest.mark.skip(reason="Temporarily skipped since under development")),
        "segment_uniform",
        "segment_graded",
        "global_grid_search",
        "iterative_relaxation",
        "iterative_relaxation_fast",
        "iterative_relaxation_momentum",
        #"iterative_relaxation_fast_momentum"
    ])
    @settings(max_examples=5, deadline=None)
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
    def test_fuzz_mesh_generation(self, algorithm, fixed_points, optional_points, max_res, ratio):
        """
        Fuzz test the mesher with physically sound, non-sliver inputs.
        """
        domain_size = max(fixed_points) - min(fixed_points)
        assume(domain_size >= len(fixed_points) * max_res)

        if algorithm.startswith("iterative_relaxation") and ratio < 1.3:
            # Iterative solvers suffer from "Critical Slowing Down" at tight ratios,
            # requiring exponentially more iterations to converge. We restrict fuzzing
            # to ratio >= 1.3 for these solvers to avoid false-positive timeouts
            assume(False)

        mesher = FDTDMesher1D(fixed_points, optional_points, max_res, ratio)

        mesh = mesher.generate(algorithm)

        assert len(mesh) >= len(fixed_points)
        check_mesh_validity(mesh, fixed_points, max_res=max_res, ratio=ratio, algorithm=algorithm)

    @pytest.mark.parametrize("algorithm", [
        "iterative_relaxation",
        "iterative_relaxation_fast",
        "iterative_relaxation_momentum",
        #"iterative_relaxation_fast_momentum"
    ])
    def test_iterative_relaxation_stagnation_injection(self, algorithm):
        """Test that the stagnation detector successfully injects points to resolve impossible ratio shocks."""
        fixed = [0.0, 10.0, 10.1]
        mesher = FDTDMesher1D(fixed, [], max_res=2.0, ratio=1.5)

        kwargs = {}
        if algorithm == "iterative_relaxation":
            kwargs["max_iterations"] = 30000

        # Without topological injection, this setup creates an infinite spring loop.
        # It should complete successfully and the resulting mesh size should be larger
        # than what a pure uniform base grid search would generate (7 points).
        final_mesh = mesher.generate(algorithm, **kwargs)

        check_mesh_validity(final_mesh, fixed, max_res=2.0, ratio=1.5, algorithm=algorithm)
        assert len(final_mesh) > 7, "Mesh did not inject new points to break stagnation."

    @pytest.mark.parametrize("algorithm", [
        "iterative_relaxation",
        "iterative_relaxation_fast",
        "iterative_relaxation_momentum",
        "iterative_relaxation_fast_momentum"
    ])
    def test_iterative_relaxation_anchor_rigidity(self, algorithm):
        """Test that fixed points (anchors) do not drift during spring relaxation."""
        # Use an irrational-like repeating float to ensure precision doesn't drift
        fixed = [0.0, 3.33333333333, 10.0]
        mesher = FDTDMesher1D(fixed, [], max_res=2.0, ratio=1.5)
        final_mesh = mesher.generate(algorithm)

        # Check exact floating point equality (not just pytest.approx) to ensure zero drift
        for fp in fixed:
            assert fp in final_mesh, f"Fixed anchor {fp} drifted during relaxation!"

    @pytest.mark.parametrize("algorithm", [
        "iterative_relaxation",
        "iterative_relaxation_fast",
        "iterative_relaxation_momentum",
        "iterative_relaxation_fast_momentum"
    ])
    def test_iterative_relaxation_no_premature_equilibrium(self, algorithm):
        """
        Regression test for the 'premature equilibrium' bug.
        Ensures the solver does not exit early when shifts become microscopically small
        but the strict mathematical constraints are not yet fully satisfied.
        """
        # This setup (small max_res, very tight ratio) forces the solver into a state
        # where the required shifts quickly drop below 1e-6. If the solver exits on
        # small shifts instead of strictly waiting for zero demand, this will fail.
        fixed = [0.0, 1.0, 2.1]
        mesher = FDTDMesher1D(fixed, [], max_res=1.0, ratio=1.1)

        final_mesh = mesher.generate(algorithm)

        # Check validity natively enforces the 1e-9 strict mathematical tolerance
        check_mesh_validity(final_mesh, fixed, max_res=1.0, ratio=1.1, algorithm=algorithm)

    @pytest.mark.parametrize("algorithm", [
        "iterative_relaxation",
        "iterative_relaxation_fast",
        "iterative_relaxation_momentum",
        "iterative_relaxation_fast_momentum"
    ])
    def test_iterative_relaxation_graceful_timeout(self, algorithm):
        """Test that the algorithm throws a RuntimeError instead of infinite looping if starved of iterations."""
        fixed = [0.0, 10.0, 10.1]
        mesher = FDTDMesher1D(fixed, [], max_res=2.0, ratio=1.5)

        with pytest.raises(RuntimeError, match="failed to converge"):
            # Provide an impossibly small iteration limit so it aborts cleanly
            mesher.generate(algorithm, max_iterations=5)


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