# Simulation is a direct copy, but adapted to my class implementation, from the tutorial at https://docs.openems.de/python/openEMS/Tutorials/MSL_NotchFilter.html

import os
import numpy as np
import logging
import matplotlib.pyplot as plt
from .simulationModel import SimulationModel

from openEMS.physical_constants import C0

logger = logging.getLogger(__name__)

class MicrostripNotchModel(SimulationModel):
    """
    Constructs and simulates a Microstrip Notch Filter model as defined in the tutorial: https://docs.openems.de/python/openEMS/Tutorials/MSL_NotchFilter.html.
    """
    def __init__(self, msl_length=50000, msl_width=600,
    substrate_thickness=256, substrate_epr=3.66, stub_length = 12e3,
    unit=1e-6, f_max = 7e9, **kwargs):

        super().__init__(unit=unit, f_0=f_max/2, f_max=f_max, **kwargs)

        self.set_boundary_conditions(['PML_8', 'PML_8', 'MUR', 'MUR', 'PEC', 'MUR'])

        self.resolution = C0/(f_max*np.sqrt(substrate_epr))/unit/50 # resolution of lambda/50
        self.third_mesh = np.array([2*self.resolution/3, -self.resolution/3])/4

        self.add_mesh_lines('x', -msl_width/2-self.third_mesh)
        self.add_mesh_lines('x', [0])
        self.add_mesh_lines('x', msl_width/2+self.third_mesh)
        self.add_mesh_lines('y', -msl_width/2-self.third_mesh)
        self.add_mesh_lines('y', [0])
        self.add_mesh_lines('y', msl_width/2+self.third_mesh)
        self.build_graded_mesh(max_res_x=self.resolution/4, max_res_y=self.resolution/4, custom_mesher=False)

        self.add_mesh_lines('x', [-msl_length, msl_length])
        self.add_mesh_lines('y', [-15*msl_width, 15*msl_width+stub_length])
        self.add_mesh_lines('y', (msl_width/2+stub_length)+self.third_mesh)
        self.add_mesh_lines('z', np.linspace(0,substrate_thickness,5))
        self.add_mesh_lines('z', 3000)
        self.build_graded_mesh(max_res_x=self.resolution, max_res_y=self.resolution, max_res_z=self.resolution, custom_mesher=False)


        # Define Materials
        self.substrate = self.add_material( 'RO4350B', epsilon=substrate_epr)
        self.pec = self.add_material( 'PEC' )


        # Define substrate
        start = [-msl_length, -15*msl_width, 0]
        stop  = [+msl_length, +15*msl_width+stub_length, substrate_thickness]
        self.substrate.AddBox(start, stop )


        # Define ports (and straight component of microstrip)
        self.ports = [None, None]
        portstart = [ -msl_length, -msl_width/2, substrate_thickness]
        portstop  = [ 0,  msl_width/2, 0]
        self.ports[0] = self.FDTD.AddMSLPort( 1,  self.pec, portstart, portstop, 'x', 'z', excite=-1, FeedShift=10*self.resolution, MeasPlaneShift=msl_length/3, priority=10)

        portstart = [msl_length, -msl_width/2, substrate_thickness]
        portstop  = [0         ,  msl_width/2, 0]
        self.ports[1] = self.FDTD.AddMSLPort( 2, self.pec, portstart, portstop, 'x', 'z', MeasPlaneShift=msl_length/3, priority=10 )

        self._ports_setup = True


        # Define Filter stub
        start = [-msl_width/2,  msl_width/2, substrate_thickness]
        stop  = [ msl_width/2,  msl_width/2+stub_length, substrate_thickness]
        self.pec.AddBox(start, stop, priority=10 )

        self._geometry_built = True

    def calculate_s_params(self, sim_dir, f_min=1e6, f_steps=1601, show_gui=False):
        f, port_data = self.calc_all_ports(sim_dir=sim_dir, f_min=f_min, f_max=self.f_max, f_steps=f_steps)

        s11 = port_data[1]["uf_ref"] / port_data[1]["uf_inc"]
        s21 = port_data[2]["uf_ref"] / port_data[1]["uf_inc"]

        fig, (ax1) = plt.subplots(1, 1, figsize=(8, 8))

        ax1.plot(f/1e9,20*np.log10(np.abs(s11)),'k-',linewidth=2 , label='$S_{11}$')
        ax1.grid()
        ax1.plot(f/1e9,20*np.log10(np.abs(s21)),'r--',linewidth=2 , label='$S_{21}$')
        ax1.legend()
        ax1.set_ylabel('S-Parameter (dB)')
        ax1.set_xlabel('frequency (GHz)')

        plt.tight_layout()

        if show_gui:
            plt.show()

        plot_file = os.path.join(sim_dir, 'sParam.png')
        plt.savefig(plot_file)
