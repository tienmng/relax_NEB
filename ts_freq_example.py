#!/usr/bin/env python
"""
Example script for using the Transition State Frequency Calculator
==================================================================
This script demonstrates how to use the TS frequency calculator module
either as a standalone tool or integrated into your workflow.
"""

import os
import sys
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Add the module directory to path
module_dir = os.path.abspath(os.path.join(os.getcwd(), '/nfs/home/6/nguyenm/pymatgen-packages/relax_NEB'))
if module_dir not in sys.path:
    sys.path.insert(0, module_dir)
    print(f"Added {module_dir} to Python path")


# Import the frequency calculator
from ts_frequency_calc import run_ts_frequency_analysis

# =============================================================================
# CONFIGURATION
# =============================================================================

# Paths from your NEB calculation
ENERGY_PROFILE_FILE = "./multistage_neb/final_energy_profile.dat"
NEB_DIR = "./multistage_neb/neb_stage3"  # Use the final NEB stage directory

# Or for single-stage NEB:
# ENERGY_PROFILE_FILE = "./vasp_analysis/final_energy_profile.dat"
# NEB_DIR = "./neb_calculation"

# System-specific settings (match your NEB calculation)
MOVING_ATOM_IDX = 51  # Index of the moving oxygen atom
POTCAR_PATH = "/nfs/home/6/nguyenm/sensor/POTCAR-files/potpaw_PBE_54"
POTCAR_MAPPING = {
    "O": "O_s",
    "La": "La",
    "Ni": "Ni_pv",
    "V": "V_sv",
    "Fe": "Fe_pv",
    "Co": "Co_pv",
    "Mn": "Mn_pv"
}

LDAU_SETTINGS =  {
    'LDAU': True,
    "La": {"L": 0, "U": 0}, "Ni": {"L": 2, "U": 7}, "V": {"L": 2, "U": 3}, "Ti": {"L": 2,"U": 14.5},
    "Fe": {"L": 2, "U": 5}, "Co": {"L": 2, "U": 3.5}, "Mn": {"L": 2, "U": 4}, "Nb": {"L": 2,"U": 5},
    "O": {"L": 0, "U": 0}
}
# Frequency calculation settings
FREEZE_RADIUS = 3.5  # Freeze atoms beyond 5 Å from moving atom
OUTPUT_DIR = "./ts_frequency"

# Job settings
SUBMIT_JOB = True  # Set to False to just prepare files
MONITOR_JOB = True  # Set to False to submit without monitoring

# =============================================================================
# MAIN WORKFLOW
# =============================================================================

def main():
    """Run the transition state frequency analysis."""
    
    print("="*70)
    print("TRANSITION STATE FREQUENCY ANALYSIS")
    print("="*70)
    print()
    
    # Check if required files exist
    if not os.path.exists(ENERGY_PROFILE_FILE):
        print(f"ERROR: Energy profile file not found: {ENERGY_PROFILE_FILE}")
        print("Make sure your NEB calculation has completed successfully.")
        return
    
    if not os.path.exists(NEB_DIR):
        print(f"ERROR: NEB directory not found: {NEB_DIR}")
        return
    
    print(f"Energy profile: {ENERGY_PROFILE_FILE}")
    print(f"NEB directory: {NEB_DIR}")
    print(f"Moving atom index: {MOVING_ATOM_IDX}")
    print(f"Freeze radius: {FREEZE_RADIUS} Å")
    print(f"Output directory: {OUTPUT_DIR}")
    print()
    
    # Run the frequency analysis
    print("Starting frequency analysis...")
    
    try:
        results = run_ts_frequency_analysis(
            energy_profile_file=ENERGY_PROFILE_FILE,
            neb_dir=NEB_DIR,
            moving_atom_idx=MOVING_ATOM_IDX,
            output_dir=OUTPUT_DIR,
            potcar_path=POTCAR_PATH,
            potcar_mapping=POTCAR_MAPPING,
            ldau_settings=LDAU_SETTINGS,
            freeze_radius=FREEZE_RADIUS,
            submit=SUBMIT_JOB,
            monitor=MONITOR_JOB
        )
        
        # Process results
        if results:
            print("\n" + "="*70)
            print("RESULTS SUMMARY")
            print("="*70)
            
            # Transition state info
            if 'ts_info' in results:
                ts_info = results['ts_info']
                print(f"\nTransition State:")
                print(f"  Image index: {ts_info.get('index', 'N/A')}")
                print(f"  Energy: {ts_info.get('energy', 'N/A'):.6f} eV")
                print(f"  Barrier height: {ts_info.get('relative_energy', 'N/A'):.3f} eV")
            
            # Frequency results
            if 'all_frequencies' in results:
                print(f"\nFrequency Analysis:")
                print(f"  Total modes: {len(results['all_frequencies'])}")
                print(f"  Imaginary frequencies: {results.get('n_imaginary', 0)}")
                print(f"  Real frequencies: {results.get('n_real', 0)}")
                
                # Validation
                if results.get('is_transition_state', False):
                    print("\n✓ VALID TRANSITION STATE")
                    if results.get('imaginary_frequencies'):
                        print(f"  Imaginary frequency: {results['imaginary_frequencies'][0]:.2f} cm⁻¹")
                else:
                    print("\n✗ NOT A VALID TRANSITION STATE")
                    n_imag = results.get('n_imaginary', 0)
                    if n_imag == 0:
                        print("  No imaginary frequencies - structure is a minimum")
                    else:
                        print(f"  {n_imag} imaginary frequencies - higher-order saddle point")
            
            # Output files
            print(f"\nOutput Files:")
            print(f"  Calculation directory: {OUTPUT_DIR}")
            if results.get('plot_path'):
                print(f"  Frequency spectrum: {results['plot_path']}")
            if results.get('report_path'):
                print(f"  Analysis report: {results['report_path']}")
            
            print("\n" + "="*70)
            
        elif SUBMIT_JOB and not MONITOR_JOB:
            print("\nFrequency calculation submitted.")
            print("Check job status and run analysis later.")
            print(f"\nTo analyze completed calculation:")
            print(f"  python analyze_completed_frequency.py {OUTPUT_DIR}")
            
        else:
            print("\nFrequency calculation prepared but not submitted.")
            print(f"Input files created in: {OUTPUT_DIR}")
            
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()


def analyze_completed_calculation(calc_dir):
    """
    Analyze a completed frequency calculation.
    
    This function can be used to analyze results after the job completes,
    especially if you submitted without monitoring.
    """
    print(f"Analyzing completed calculation in {calc_dir}")
    
    # Import required modules
    from file_manager import FileManager
    from vasp_inputs import VASPInputGenerator
    from slurm_manager import SLURMManager
    from structure_analyzer import StructureAnalyzer
    from ts_frequency_calc import TransitionStateFrequencyCalculator
    
    # Initialize components
    file_manager = FileManager()
    input_gen = VASPInputGenerator()
    slurm_manager = SLURMManager()
    struct_analyzer = StructureAnalyzer()
    
    # Create frequency calculator
    freq_calc = TransitionStateFrequencyCalculator(
        file_manager, input_gen, slurm_manager, struct_analyzer
    )
    
    # Analyze frequencies
    results = freq_calc.analyze_frequencies(calc_dir)
    
    if results:
        # Create plots
        plot_path = freq_calc.plot_frequency_spectrum(
            results, os.path.join(calc_dir, "frequency_spectrum.png")
        )
        
        # Read TS info from frequency_info.txt if available
        ts_info = {}
        info_file = os.path.join(calc_dir, "frequency_info.txt")
        if os.path.exists(info_file):
            with open(info_file, 'r') as f:
                for line in f:
                    if "Moving atom index:" in line:
                        ts_info['moving_atom_idx'] = int(line.split(":")[-1].strip())
        
        # Generate report
        report_path = freq_calc.generate_report(results, calc_dir, ts_info)
        
        # Print summary
        print("\nAnalysis Results:")
        print(f"  Imaginary frequencies: {results.get('n_imaginary', 0)}")
        print(f"  Is valid TS: {results.get('is_transition_state', False)}")
        print(f"  Plot saved: {plot_path}")
        print(f"  Report saved: {report_path}")
        
        return results
    else:
        print("Failed to analyze frequencies")
        return None


if __name__ == "__main__":
    # Check if analyzing completed calculation
    if len(sys.argv) > 1:
        # Usage: python example_ts_frequency.py <calc_dir>
        calc_dir = sys.argv[1]
        analyze_completed_calculation(calc_dir)
    else:
        # Run full workflow
        main()
