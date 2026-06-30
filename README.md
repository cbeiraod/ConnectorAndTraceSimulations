# Trace and Connector Simulation

Simulating traces and connectors for PCB design using OpenEMS.

## Environment Setup (macOS / Linux)

Because OpenEMS requires complex C++ math libraries, standard `pip` cannot install it natively. We recommend using Docker with the custom `cristovao/openems-env` (or `ghcr.io/cbeiraod/openems-docker-env`) container, which comes pre-configured with OpenEMS. You can then install the additional Python data science libraries via the `requirements.txt` file.

Alternatively, you may install the C++ engine directly on your bare-metal machine. This takes a bit more setup, but allows you to use the `AppCSXCAD` GUI viewer to visually inspect your 3D geometry.

I personally ran into an issue on my local installation on macOS where I could not run the simulations. Instead of running `python` normally, I had to pass the flag `python -O` to ignore asserts. I could never quite figure out which assert was failing, since it is internal to the OpenEMS engine and the eror reporting gives no useful information.

### Dev

If you are actually developing code... possibly modifying helper functions and so on, you may want to install the `requirements-dev.txt` instead and run tests with: `python -m pytest`.
If you want to run the `visualize_mesh.py` script (which allows to preview the results of the meshing algorithms), please run it with `python -m scripts.visualize_mesh` from the root directory.


---

### Path A: The Docker Method (Recommended - 100% CI Parity)
This method runs the simulation locally using the exact same Linux image the CI pipeline uses, guaranteeing environment parity. The only drawback is the lack of the `AppCSXCAD` GUI viewer.

1. **Run the OpenEMS Docker container** (this automatically downloads the image and maps your current directory):
   ```bash
   docker run -it -v $(pwd):/opt/openems_sim cristovao/openems-env bash
   ```
2. **Inside the container, install the Python data stack:**
   ```bash
   pip3 install -r requirements.txt
   ```
3. **Run your simulation:**
   ```bash
   python3 simulate_microstrip_notch.py
   ```

---

### Path B: Bare-Metal Installation (Required for the AppCSXCAD GUI Viewer)
This method installs the C++ engine directly on your machine so you can visually inspect your 3D meshes.

#### macOS Setup

1. **Install the OpenEMS C++ Engine via Homebrew:**
   ```bash
   brew tap vinn-ie/openems
   brew trust vinn-ie/openems  # Required on some Macs to allow the custom tap
   brew install openems
   ```
2. **Set up a standard Python virtual environment:**
   ```bash
   python3.11 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```
3. **Link the Homebrew OpenEMS Python bindings to your virtual environment:**
   ```bash
   CSXCAD_ROOT=$(brew --prefix csxcad) pip install $(brew --prefix csxcad)/share/CSXCAD/python
   OPENEMS_ROOT=$(brew --prefix openems) CSXCAD_ROOT=$(brew --prefix csxcad) pip install $(brew --prefix openems)/share/openEMS/python
   ```

**Troubleshooting macOS Python Bindings:** On some machines, Homebrew fails to build the Python integration files. If the link commands above fail, you can manually fetch and compile the missing bindings:
1. **Compile the CSXCAD Bindings:**
   *Robustness tip: The `git checkout` tag below should match the version installed by Homebrew. You can check your installed version by running `brew info csxcad`. It seems brew may bump some version strings, so you may need to experiment with a previous version*
   ```bash
   cd /tmp
   git clone https://github.com/thliebig/CSXCAD.git
   cd CSXCAD
   git checkout v0.6.3  # Update this tag to match your Homebrew version if it bumps in the future

   # Force the compiler to see the Homebrew directories
   export CPATH="$(brew --prefix csxcad)/include"
   export LIBRARY_PATH="$(brew --prefix csxcad)/lib"

   CSXCAD_ROOT=$(brew --prefix csxcad) pip install --no-build-isolation ./python
   ```

2. **Compile the OpenEMS Bindings:**
   *Robustness tip: Ensure this git tag matches the version from `brew info openems`.*
   ```bash
   cd /tmp
   git clone https://github.com/thliebig/openEMS.git
   cd openEMS
   git checkout v0.0.36 # Update this tag to match your Homebrew version if it bumps in the future

   # Expand compiler vision to include OpenEMS, CSXCAD, and the main Homebrew directory (for Boost)
   export CPATH="$(brew --prefix openems)/include:$(brew --prefix csxcad)/include:$(brew --prefix)/include"
   export LIBRARY_PATH="$(brew --prefix openems)/lib:$(brew --prefix csxcad)/lib:$(brew --prefix)/lib"

   # Force the compiler to use C++14 to prevent syntax errors with modern Boost libraries
   export CXXFLAGS="-std=c++14"
   export CFLAGS="-std=c++14"

   OPENEMS_INSTALL_PATH=$(brew --prefix openems) CSXCAD_INSTALL_PATH=$(brew --prefix csxcad) pip install --no-build-isolation ./python
   ```

3. **Verify the Installation:**
   Return to your main project directory (do not run this from inside the `/tmp` source folders, or Python will fail to find the compiled `.so` modules):
   ```bash
   cd /path/to/your/project
   python -c "import CSXCAD; import openEMS; print('macOS local bindings successful')"
   ```

#### Linux Setup (Ubuntu/Debian) - UNTESTED at the moment

Because the official Ubuntu package managers are outdated, OpenEMS must be compiled from source on Linux.

1. **Install the required C++ compilers and dependencies:**
   ```bash
   sudo apt-get update
   sudo apt-get install -y build-essential cmake git libhdf5-dev libvtk9-dev \
   libboost-all-dev libcgal-dev libtinyxml-dev qtbase5-dev libvtk9-qt-dev \
   python3-numpy python3-matplotlib cython3 python3-h5py python3-pip \
   python3-dev python3-setuptools python-is-python3
   ```
2. **Clone the repository and compile:**
   ```bash
   git clone --recursive [https://github.com/thliebig/openEMS-Project.git](https://github.com/thliebig/openEMS-Project.git) ~/openEMS-Project
   cd ~/openEMS-Project
   ./update_openEMS.sh ~/opt/openEMS --python
   ```
3. **Set up your Python virtual environment and link the bindings:**
   ```bash
   cd path/to/your/simulation/repo
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt

   # Link the bindings you just compiled
   pip install ~/opt/openEMS/share/CSXCAD/python
   pip install ~/opt/openEMS/share/openEMS/python
   ```
   *(Note: Whenever you want to use the OpenEMS binaries outside the virtual environment, ensure `~/opt/openEMS/bin` is added to your system `$PATH`).*
