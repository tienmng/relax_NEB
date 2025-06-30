#!/usr/bin/env python
"""
Complete Control NEB Workflow with Parallel Multistage Relaxations
==================================================================
This workflow provides complete control over every step with options for:
- Parallel multistage structure relaxation (both initial and final simultaneously)
- Multistage NEB calculation (coarse -> fine)
- Full constraint control
- Automatic convergence checking to avoid restarts
"""

import os
import logging
import time
from pymatgen.core import Structure
from pymatgen.io.vasp.inputs import Poscar
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# =============================================================================
# STEP 0: CONFIGURATION
# =============================================================================
print("="*60)
print("COMPLETE CONTROL NEB WORKFLOW - PARALLEL RELAXATIONS")
print("="*60)

# Define your system-specific paths and settings
potcar_path = "/nfs/home/6/nguyenm/sensor/POTCAR-files/potpaw_PBE_54"
#potcar_path = "potcars/"
potcar_mapping = {
    "O": "O_s", "La": "La", "Ni": "Ni_pv", "V": "V_sv",
    "Fe": "Fe_pv", "Co": "Co_pv", "Mn": "Mn_pv"
}

ldau_settings = {
    'LDAU': True,
    "La": {"L": 0, "U": 0}, "Ni": {"L": 2, "U": 7}, "V": {"L": 2, "U": 3}, "Ti": {"L": 2,"U": 14.5},
    "Fe": {"L": 2, "U": 5}, "Co": {"L": 2, "U": 3.5}, "Mn": {"L": 2, "U": 4}, "Nb": {"L": 2,"U": 5},
    "O": {"L": 0, "U": 0}
}

# Input files (change these to your actual files)
init_file = "structure_00_start_relaxed.POSCAR"
final_file = "structure_01_end_relaxed.POSCAR"
moving_o_idx = 51

# Calculation parameters
n_images = 5
nodes = 5
ntasks_per_node = 128
walltime = "48:00:00"

print(f"Input files: {init_file}, {final_file}")
print(f"NEB images: {n_images}")
print(f"Job resources: {nodes} nodes, {ntasks_per_node} tasks/node")
print()

# =============================================================================
# IMPORT COMPONENTS
# =============================================================================
print("Initializing components...")
import sys
module_dir = os.path.abspath(os.path.join(os.getcwd(), '/nfs/home/6/nguyenm/pymatgen-packages/relax_NEB'))
if module_dir not in sys.path:
    sys.path.insert(0, module_dir)
    print(f"Added {module_dir} to Python path")
from file_manager import FileManager
from vasp_inputs import VASPInputGenerator
from structure_analyzer import StructureAnalyzer
from slurm_manager import SLURMManager
from multistage_relaxation import MultiStageStructureRelaxer
from neb_path_generation_with_constraint_control import NEBPathGenerator
from energy_analyzer import NEBEnergyAnalyzer
from neb_multistage import MultiStageNEB

# Initialize components
file_manager = FileManager(potcar_path, potcar_mapping)
analyzer = StructureAnalyzer()
input_gen = VASPInputGenerator()
slurm_manager = SLURMManager()
structure_relaxer = MultiStageStructureRelaxer(file_manager, input_gen, slurm_manager, auto_cleanup=True)
path_generator = NEBPathGenerator()
energy_analyzer = NEBEnergyAnalyzer()

from neb_path_visualizer import NEBPathVisualizer

# Create visualizer
visualizer = NEBPathVisualizer()

# Generate all visualizations for your NEB calculation
visualizer.create_all_visualizations(
    neb_dir="./multistage_neb/neb_stage2",  # Your NEB directory
    n_images=n_images,                             # Number of intermediate images
    moving_atom_idx=moving_o_idx,                     # Index of moving oxygen
    output_dir="./visualizations"           # Where to save figures
)
