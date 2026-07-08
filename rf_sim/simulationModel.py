import os
import numpy as np
import logging
from CSXCAD import ContinuousStructure
from openEMS import openEMS
from .helper import FDTDMesher1D

logger = logging.getLogger(__name__)

custom_mesher_tunes = [
    {
        "name": "Red-Black Standard Nesterov", # (Mild Shocks / X & Y Axes)
        "kwargs": {
            "algorithm": "iterative_relaxation_redblack",
            "update_type": "nesterov",
            "lr_mode": "uniform",
            "damping_mode": "adjoint",
            "damping": 0.8,
            "relaxation_factor": 0.2,
            "max_iterations": 50000
        }
    },
    {
        "name": "Red-Black Aggressive Nesterov", # (High-Ratio Shocks / Z-Axis)
        "kwargs": {
            "algorithm": "iterative_relaxation_redblack",
            "update_type": "nesterov",
            "lr_mode": "uniform",
            "damping_mode": "adjoint",
            "damping": 0.9,
            "relaxation_factor": 0.35,
            "damping_gamma": 2.0,       # Softens the brakes to allow sliding across massive voids
            "max_iterations": 100000    # Gives it room to crawl if necessary
        }
    },
    {
        "name": "Red-Black First-Order Adjoint", # (The Slow Crawler)
        "kwargs": {
            "algorithm": "iterative_relaxation_redblack",
            "update_type": "first_order",
            "lr_mode": "adjoint",       # Exponentially freeze both LR and Damping
            "relaxation_factor": 0.2,
            # No momentum, so no ringing. Guaranteed stability, but needs huge iteration counts
            "max_iterations": 100000
        }
    }
]

class SimulationModel:
    """
    Abstract Base Class for OpenEMS simulations.
    Handles engine initialization, mesh state tracking, and execution boilerplate.
    """
    def __init__(self, unit=1e-3, f_0=2.0e9, f_max=4.0e9, nr_ts=50000, end_criteria=1e-4):
        self.unit = unit
        self.f_0 = f_0
        self.f_max = f_max

        # Engine State
        self.FDTD = openEMS(NrTS=nr_ts, EndCriteria=end_criteria)
        self.FDTD.SetGaussExcite(f_0, f_max)
        self.CSX = ContinuousStructure()
        self.FDTD.SetCSX(self.CSX)

        # Data State
        self.materials = {}
        self.mesh_lines = {'x': [], 'y': [], 'z': []}
        self.optional_mesh_lines = {'x': [], 'y': [], 'z': []}
        self.ports = []

        # Execution State Trackers
        self._boundaries_set = False
        self._materials_added = False
        self._geometry_built = False
        self._ports_setup = False
        self._mesh_built = False

        # Enforce units
        self.mesh = self.CSX.GetGrid()
        self.mesh.SetDeltaUnit(self.unit)

        #self._apply_wrapper_workaround()

    #def _apply_wrapper_workaround(self):
    #    """
    #    WORKAROUND for openEMS Python Wrapper Issue #113.
    #    The wrapper asserts the existence of a property explicitly named 'Excitation'.
    #    """
    #    self.CSX.AddExcitation('Excitation', exc_type=0, exc_val=[0, 0, 0])

    def set_boundary_conditions(self, cond_list):
        """E.g. ['MUR', 'MUR', 'MUR', 'MUR', 'PEC', 'MUR']"""
        self.FDTD.SetBoundaryCond(cond_list)
        self._boundaries_set = True

    def add_material(self, name, **kwargs):
        """Creates and stores a CSXCAD material."""
        if name == "Copper":
            self.materials[name] = self.CSX.AddMetal('Copper')
        elif name == "PEC":
            self.materials[name] = self.CSX.AddMetal('PEC')
        else:
            self.materials[name] = self.CSX.AddMaterial(name, **kwargs)

        self._materials_added = True
        return self.materials[name]

    def add_mesh_lines(self, axis, lines):
        """
        Stores critical coordinates to guarantee they are locked into the grid.
        """
        if isinstance(lines, (int, float)):
            lines = [lines]
        elif isinstance(lines, np.ndarray):
            lines = lines.tolist()
        self.mesh_lines[axis].extend(lines)

    def add_optional_mesh_lines(self, axis, lines):
        """
        Stores critical coordinates to guarantee they are locked into the grid.
        """
        if isinstance(lines, (int, float)):
            lines = [lines]
        elif isinstance(lines, np.ndarray):
            lines = lines.tolist()
        self.optional_mesh_lines[axis].extend(lines)

    def add_mesh_region(self, axis, start, stop, step):
        """
        Generates a uniform mesh region and locks the coordinates.
        Useful for forcing a high-resolution grid around complex structures (like connectors).
        """
        # np.arange with step/2 ensures the 'stop' coordinate is included if perfectly aligned
        region_lines = np.arange(start, stop + step/2, step).tolist()
        self.mesh_lines[axis].extend(region_lines)

    def build_graded_mesh(self, max_res_x=None, max_res_y=None, max_res_z=None, ratio=1.2, custom_mesher=True, mesher_preferred_tunes={}):
        """
        Applies stored exact lines to the grid.
        If max_res parameters are provided, smoothly fills the remaining space.
        """
        for axis in ['x', 'y', 'z']:
            if custom_mesher:
                max_res = max_res_x
                if axis == 'y':
                    max_res = max_res_y
                elif axis == 'z':
                    max_res = max_res_z

                preferred_tune = None
                if axis in mesher_preferred_tunes:
                    preferred_tune = mesher_preferred_tunes[axis]

                custom_mesh = self._run_custom_mesher_tune(axis, max_res, ratio, preferred_tune)
                if custom_mesh:
                    self.mesh.AddLine(axis, custom_mesh)
            else:
                unique_lines = np.unique(self.mesh_lines[axis]).tolist()
                if unique_lines:
                    self.mesh.AddLine(axis, unique_lines)

        if not custom_mesher:
            if max_res_x: self.mesh.SmoothMeshLines('x', max_res_x, ratio=ratio)
            if max_res_y: self.mesh.SmoothMeshLines('y', max_res_y, ratio=ratio)
            if max_res_z: self.mesh.SmoothMeshLines('z', max_res_z, ratio=ratio)

        self._mesh_built = True
        logger.debug(f"Applied mesh -> Smooth X:{max_res_x}, Y:{max_res_y}, Z:{max_res_z} mm")

    def _run_custom_mesher_tune(self, axis: str, max_res: float | None, ratio: float, preferred_tune: dict | None):
        if axis not in ['x', 'y', 'z']:
            raise RuntimeError("Trying to run custom mesher on an axis which doesn't exist")

        fixed_edges = np.unique(self.mesh_lines[axis]).tolist()
        optional_edges = np.unique(self.optional_mesh_lines[axis]).tolist()

        if not fixed_edges:
            return []

        if max_res is None:
            return fixed_edges

        mesher = FDTDMesher1D(fixed_edges, optional_edges, max_res, ratio)

        if preferred_tune is not None:
            try:
                kwargs = None
                for tune in custom_mesher_tunes:
                    if tune['name'] == preferred_tune["tune"]:
                        kwargs = tune["kwargs"]
                        if "override" in preferred_tune:
                            for key in preferred_tune["override"]:
                                kwargs[key] = preferred_tune["override"][key]
                        break
                if kwargs is not None:
                    mesh = mesher.generate(**kwargs)
                    logger.info(f"Completed Meshing with preferred tune: {preferred_tune}")
                    return mesh
            except RuntimeError as e:
                logger.debug(f"Meshing failed with preferred tune: {preferred_tune}. Trying tune list...")

        # Try each iterative solver in sequence
        for tune in custom_mesher_tunes:
            if preferred_tune is not None and tune['name'] == preferred_tune:
                continue

            try:
                mesh = mesher.generate(**tune["kwargs"])
                logger.info(f"Completed Meshing with tune: {tune['name']}")
                return mesh
            except RuntimeError as e:
                logger.debug(f"Meshing failed with {tune['name']}. Trying next tune...")
                continue

        # If all iterative solvers timed out, trigger the ultimate failsafe.
        # It operates in O(N) time and requires zero iterations.
        logger.warning("All iterative solvers failed! Falling back to segment_graded.")
        logger.warning(f"List of fixed edges: {fixed_edges}")
        mesh = mesher.generate(algorithm="segment_graded")
        logger.info(f"Completed Meshing with fallback: segment_graded")
        return mesh

    def run_simulation(self, sim_dir="Sim_Data", show_gui=False, cleanup=True):
        """Writes the XML, optionally shows GUI, and runs the FDTD engine."""
        # Enforce execution order to prevent C++ Engine crashes
        if not self._boundaries_set:
            raise RuntimeError("Simulation cannot start: Boundary conditions have not been set.")
        if not self._materials_added:
            raise RuntimeError("Simulation cannot start: No materials have been added.")
        if not self._geometry_built:
            raise RuntimeError("Simulation cannot start: Geometry has not been built. (Ensure self._geometry_built = True is set in your subclass)")
        if not self._ports_setup:
            raise RuntimeError("Simulation cannot start: Ports have not been configured. (Ensure self._ports_setup = True is set in your subclass)")
        if not self._mesh_built:
            raise RuntimeError("Simulation cannot start: The mesh has not been generated. Please call build_graded_mesh() first.")

        os.makedirs(sim_dir, exist_ok=True)
        csx_file = os.path.join(sim_dir, 'model.xml')
        self.CSX.Write2XML(csx_file)

        if show_gui:
            logger.info("Launching AppCSXCAD... Close the GUI to start the simulation.")
            os.system(f'AppCSXCAD "{csx_file}"')

        logger.info(f"--- Running OpenEMS Simulation in {sim_dir} ---")

        abs_sim_dir = os.path.abspath(sim_dir)

        original_cwd = os.getcwd()
        try:
            self.FDTD.Run(abs_sim_dir, cleanup=cleanup)
        finally:
            os.chdir(original_cwd)

    def preview_geometry(self, sim_dir="Sim_Data"):
        """Writes the XML and shows GUI."""
        # Enforce execution order to prevent C++ Engine crashes
        if not self._boundaries_set:
            raise RuntimeError("Simulation cannot start: Boundary conditions have not been set.")
        if not self._materials_added:
            raise RuntimeError("Simulation cannot start: No materials have been added.")
        if not self._geometry_built:
            raise RuntimeError("Simulation cannot start: Geometry has not been built. (Ensure self._geometry_built = True is set in your subclass)")
        if not self._mesh_built:
            raise RuntimeError("Simulation cannot start: The mesh has not been generated. Please call build_graded_mesh() first.")

        os.makedirs(sim_dir, exist_ok=True)
        csx_file = os.path.join(sim_dir, 'model.xml')
        self.CSX.Write2XML(csx_file)

        logger.info("Launching AppCSXCAD...")
        os.system(f'AppCSXCAD "{csx_file}"')

    def calc_all_ports(self, sim_dir="Sim_Data", f_min=100e6, f_max=1e9, f_steps=100, ref_impedance=50.0):
        """
        Calculates port data for all registered ports over the given frequency array.
        Returns a dictionary mapping port numbers to their raw complex data arrays.
        Subclasses should use this raw data to calculate specific S-parameters or Z_in.
        """
        freqs = np.linspace(f_min, f_max, f_steps)

        port_data = {}

        abs_sim_dir = os.path.abspath(sim_dir)

        for idx, port in enumerate(self.ports):
            # CalcPort populates the port object with complex numpy arrays
            port.CalcPort(abs_sim_dir, freqs, ref_impedance=ref_impedance)

            p_nr = getattr(port, 'port_nr', idx + 1)

            # Note: port.port_nr is typically 1-indexed in OpenEMS
            port_data[p_nr] = {
                "uf_inc": port.uf_inc, # Incident Voltage
                "uf_ref": port.uf_ref, # Reflected Voltage
                "uf_tot": port.uf_tot, # Total Voltage
                "if_tot": port.if_tot, # Total Current
                "p_inc": port.P_inc,   # Incident Power
                "p_acc": port.P_acc    # Accepted Power
            }
        return freqs, port_data