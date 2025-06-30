# relax_NEB
pymatgen+VASP to run NEB for Oxygen migration pathways
neb_workflow_from_migration_analysis.py take initial and final structures of path_*** generated from Oxygen_workflow_improved.py with O_moving_idx from the report and number of images to be created

neb_visualizing.py after neb_workflow_from_migration_analysis.py finishes.

Needs to integrate ts_freq_example.py into the workflow.
currently: python ts_freq_example.py will run freq by looking at the ./multistage_neb/...report.txt
2025-06-30 01:33:22,741 - INFO -   Ni: 22
2025-06-30 01:33:22,741 - INFO -   O: 61
2025-06-30 01:33:22,742 - INFO - Setting up frequency calculation in ./ts_frequency
2025-06-30 01:33:22,742 - ERROR - Error setting up frequency calculation: len(values)=0 must equal sites in structure=119
2025-06-30 01:33:22,742 - ERROR - Failed to set up frequency calculation

Frequency calculation prepared but not submitted.
Input files created in: ./ts_frequency
