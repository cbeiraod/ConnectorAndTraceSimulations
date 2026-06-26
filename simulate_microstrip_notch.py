import argparse
import logging
import numpy as np
import matplotlib.pyplot as plt

from rf_sim import STACKUPS, setup_logging, MicrostripNotchModel

def main():
    parser = argparse.ArgumentParser(description="Run a single Microstrip Notch Simulation.")
    parser.add_argument("--microstrip_length", type=float, default=50000, help="Length of the signal trace (um)")
    parser.add_argument("--microstrip_width", type=float, default=600, help="Width of the signal trace (um)")
    parser.add_argument("--substrate_thickness", type=float, default=256, help="Thickness of the substrate (um)")
    parser.add_argument("--substrate_permitivity", type=float, default=3.66, help="Permitivity of the substrate")
    parser.add_argument("--stub_length", type=float, default=12e3, help="Length of the notch stub (um)")
    parser.add_argument("--max_frequency", type=float, default=7e9, help="Maximum frequency (Hz)")
    parser.add_argument("--gui", action="store_true", help="Launch AppCSXCAD before simulating")
    parser.add_argument("--debug", action="store_true", help="Print debugging info")

    args = parser.parse_args()

    setup_logging(debug_mode=args.debug)

    logger = logging.getLogger(__name__)

    logger.info(f"Simulating Microstrip Notch: Length {args.microstrip_length}um,  Width {args.microstrip_width}um")

    model = MicrostripNotchModel(
        msl_length=args.microstrip_length,
        msl_width=args.microstrip_width,
        substrate_thickness=args.substrate_thickness,
        substrate_epr=args.substrate_permitivity,
        stub_length=args.stub_length,
        unit=1e-6,
        f_max=args.max_frequency,
    )
    model.run_simulation(sim_dir="Sim_MicrostripNotch", show_gui=args.gui, cleanup=True)

    model.calculate_s_params(sim_dir="Sim_MicrostripNotch", f_min=1e6, f_steps=1601, show_gui=args.gui)

if __name__ == "__main__":
    main()