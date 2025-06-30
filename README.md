# relax_NEB
pymatgen+VASP to run NEB for Oxygen migration pathways
neb_workflow_from_migration_analysis.py take initial and final structures of path_*** generated from Oxygen_workflow_improved.py with O_moving_idx from the report and number of images to be created

neb_visualizing.py after neb_workflow_from_migration_analysis.py finishes.

Needs to integrate ts_freq_example.py into the workflow.
currently: python ts_freq_example.py will run freq by looking at the ./multistage_neb/...report.txt
