# relax_NEB
pymatgen+VASP to run NEB for Oxygen migration pathways

neb_MULTISTAGE  : STAGE 3 POTIM 0.2 STAGE 4 POTIM 0.1, if stage 3 didn't converge and ask for increase POTIM, STAGE 4 POTIM 0.2
neb_workflow_from_migration_analysis.py take initial and final structures of path_*** generated from Oxygen_workflow_improved.py with O_moving_idx from the report and number of images to be created

neb_visualizing.py after neb_workflow_from_migration_analysis.py finishes.
use analyze_multistage_neb_energies.py to analyze completed multistage_neb folders
run ts_freq_simple.py
Needs to integrate ts_freq_simple.py into the workflow.

