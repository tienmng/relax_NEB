#!/usr/bin/env python
"""
Simple Transition State Frequency Calculator
===========================================
This script performs frequency calculations on a structure (typically a transition state)
from a CONTCAR file. It creates a new folder for the frequency calculation and
documents the source of the structure.

Usage:
    python ts_freq_simple.py [options]
    
Options:
    --input-dir PATH      Directory containing CONTCAR (default: current directory)
    --input-file NAME     Input file name (default: CONTCAR)
    --output-dir PATH     Output directory for frequency calc (default: ./freq_calc)
    --moving-atom INDEX   Index of moving atom (0-based, required)
    --freeze-radius FLOAT Freeze atoms beyond this radius in Å (default: 5.0)
    --submit              Submit the job to SLURM
    --monitor             Monitor job until completion
"""

import os
import sys
import argparse
import logging
from datetime import datetime
from pymatgen.core import Structure
from pymatgen.io.vasp.inputs import Incar, Kpoints, Poscar
import shutil

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add module directory to path if needed
module_dir = os.path.abspath(os.path.join(os.getcwd(), '/nfs/home/6/nguyenm/pymatgen-packages/relax_NEB'))
if module_dir not in sys.path:
    sys.path.insert(0, module_dir)

# Default VASP settings
DEFAULT_POTCAR_PATH = "/nfs/home/6/nguyenm/sensor/POTCAR-files/potpaw_PBE_54"
DEFAULT_POTCAR_MAPPING = {
    "O": "O_s",
    "La": "La",
    "Ni": "Ni_pv",
    "V": "V_sv",
    "Fe": "Fe_pv",
    "Co": "Co_pv",
    "Mn": "Mn_pv",
    "Ti": "Ti_pv",
    "Nb": "Nb_pv"
}

DEFAULT_LDAU_SETTINGS = {
    'LDAU': True,
    "La": {"L": 0, "U": 0},
    "Ni": {"L": 2, "U": 7},
    "V": {"L": 2, "U": 3},
    "Ti": {"L": 2, "U": 14.5},
    "Fe": {"L": 2, "U": 5},
    "Co": {"L": 2, "U": 3.5},
    "Mn": {"L": 2, "U": 4},
    "Nb": {"L": 2, "U": 5},
    "O": {"L": 0, "U": 0}
}


class SimpleFrequencyCalculator:
    """Simple frequency calculator for a single structure."""
    
    def __init__(self, potcar_path=None, potcar_mapping=None, ldau_settings=None):
        self.potcar_path = potcar_path or DEFAULT_POTCAR_PATH
        self.potcar_mapping = potcar_mapping or DEFAULT_POTCAR_MAPPING
        self.ldau_settings = ldau_settings or DEFAULT_LDAU_SETTINGS
        
    def setup_frequency_calculation(self, structure_file, output_dir, moving_atom_idx=None, 
                                  freeze_radius=2.2, source_info=None):
        """
        Set up frequency calculation for a structure.
        
        Args:
            structure_file: Path to structure file (CONTCAR/POSCAR)
            output_dir: Directory for frequency calculation
            moving_atom_idx: Index of moving atom (optional)
            freeze_radius: Radius for freezing atoms (Å)
            source_info: Information about where the structure came from
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Setting up frequency calculation in {output_dir}")
        
        # Read structure
        logger.info(f"Reading structure from {structure_file}")
        structure = Structure.from_file(structure_file)
        logger.info(f"Structure: {structure.composition}, {len(structure)} atoms")
        
        # Determine atoms to freeze if moving atom is specified
        selective_dynamics = None
        if moving_atom_idx is not None:
            logger.info(f"Setting up constraints with moving atom {moving_atom_idx}, freeze radius {freeze_radius} Å")
            selective_dynamics = self._get_selective_dynamics(structure, moving_atom_idx, freeze_radius)
        
        # Write POSCAR with selective dynamics
        poscar = Poscar(structure, selective_dynamics=selective_dynamics)
        poscar_path = os.path.join(output_dir, "POSCAR")
        poscar.write_file(poscar_path)
        
        # Create INCAR for frequency calculation
        incar = self._create_frequency_incar(structure)
        incar_path = os.path.join(output_dir, "INCAR")
        incar.write_file(incar_path)
        
        # Create KPOINTS
        kpoints = Kpoints.automatic_density(structure, 40)  # ~0.3 Å spacing
        kpoints_path = os.path.join(output_dir, "KPOINTS")
        kpoints.write_file(kpoints_path)
        
        # Create POTCAR
        self._create_potcar(structure, output_dir)
        
        # Write source information
        self._write_source_info(output_dir, structure_file, source_info, 
                              moving_atom_idx, freeze_radius)
        
        # Create job script
        self._create_job_script(output_dir)
        
        logger.info("Frequency calculation setup complete")
        return True
        
    def _get_selective_dynamics(self, structure, moving_atom_idx, freeze_radius):
        """Determine which atoms to freeze based on distance from moving atom."""
        moving_atom_pos = structure[moving_atom_idx].coords
        selective_dynamics = []
        
        n_frozen = 0
        for i, site in enumerate(structure):
            # Calculate distance considering periodic boundaries
            dist = structure.get_distance(i, moving_atom_idx)
            
            if dist > freeze_radius:
                selective_dynamics.append([False, False, False])  # Freeze
                n_frozen += 1
            else:
                selective_dynamics.append([True, True, True])     # Allow to move
        
        logger.info(f"Freezing {n_frozen} atoms (> {freeze_radius} Å from atom {moving_atom_idx})")
        logger.info(f"Allowing {len(structure) - n_frozen} atoms to move")
        
        return selective_dynamics
    
    def _create_frequency_incar(self, structure):
        """Create INCAR for frequency calculation."""
        incar = Incar()
        
        # Frequency calculation settings
        incar['IBRION'] = 5          # Finite differences
        incar['POTIM'] = 0.015       # Displacement in Angstrom
        incar['NFREE'] = 2           # Central differences
        
        # Electronic settings (high accuracy)
        incar['EDIFF'] = 1E-7        # Very tight convergence
        incar['PREC'] = 'Accurate'
        incar['ENCUT'] = 600
        incar['LREAL'] = False       # Exact projectors
        incar['ALGO'] = 'Fast'
        incar['NELM'] = 200
        
        # Other settings
        incar['ISMEAR'] = 0
        incar['SIGMA'] = 0.05
        incar['LASPH'] = True
        incar['LMAXMIX'] = 6
        incar['NCORE'] = 32
        incar['LSCALAPACK'] = True
        incar['LSCALU'] = False
        
        # Output settings
        incar['LWAVE'] = False
        incar['LCHARG'] = False
        
        # Apply LDAU if needed
        if self.ldau_settings.get('LDAU'):
            elements = [str(site.specie) for site in structure]
            unique_elements = list(dict.fromkeys(elements))
            
            ldaul = []
            ldauu = []
            ldauj = []
            
            for elem in unique_elements:
                if elem in self.ldau_settings:
                    ldaul.append(self.ldau_settings[elem].get('L', 0))
                    ldauu.append(self.ldau_settings[elem].get('U', 0))
                    ldauj.append(self.ldau_settings[elem].get('J', 0))
                else:
                    ldaul.append(0)
                    ldauu.append(0)
                    ldauj.append(0)
            
            incar['LDAU'] = True
            incar['LDAUTYPE'] = 2
            incar['LDAUL'] = ldaul
            incar['LDAUU'] = ldauu
            incar['LDAUJ'] = ldauj
            incar['LDAUPRINT'] = 2
        
        # Apply magnetic moments if transition metals present
        mag_elements = {'Ni': 5.0, 'Fe': 5.0, 'Co': 5.0, 'Mn': 5.0, 'V': 5.0}
        magmoms = []
        for site in structure:
            elem = str(site.specie)
            magmoms.append(mag_elements.get(elem, 0.6))
        
        if any(m > 0.6 for m in magmoms):
            incar['MAGMOM'] = magmoms
            incar['ISPIN'] = 2
        
        return incar
    
    def _create_potcar(self, structure, output_dir):
        """Create POTCAR file."""
        elements = [str(site.specie) for site in structure]
        unique_elements = list(dict.fromkeys(elements))
        
        potcar_path = os.path.join(output_dir, "POTCAR")
        with open(potcar_path, 'w') as f:
            for elem in unique_elements:
                potcar_type = self.potcar_mapping.get(elem, elem)
                elem_potcar = os.path.join(self.potcar_path, potcar_type, "POTCAR")
                
                if not os.path.exists(elem_potcar):
                    logger.warning(f"POTCAR not found for {elem} at {elem_potcar}")
                    continue
                    
                with open(elem_potcar, 'r') as pf:
                    f.write(pf.read())
        
        logger.info(f"Created POTCAR for elements: {', '.join(unique_elements)}")
    
    def _write_source_info(self, output_dir, structure_file, source_info, 
                          moving_atom_idx, freeze_radius):
        """Write information about the source of the structure."""
        info_file = os.path.join(output_dir, "calculation_info.txt")
        
        with open(info_file, 'w') as f:
            f.write("="*70 + "\n")
            f.write("FREQUENCY CALCULATION INFORMATION\n")
            f.write("="*70 + "\n\n")
            
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("SOURCE STRUCTURE\n")
            f.write("-"*30 + "\n")
            f.write(f"Structure file: {os.path.abspath(structure_file)}\n")
            f.write(f"Parent directory: {os.path.dirname(os.path.abspath(structure_file))}\n")
            
            if source_info:
                f.write(f"\nAdditional info:\n{source_info}\n")
            
            f.write("\nCALCULATION SETTINGS\n")
            f.write("-"*30 + "\n")
            f.write(f"Calculation type: Vibrational frequency analysis\n")
            f.write(f"Method: Finite differences (IBRION=5)\n")
            f.write(f"Displacement: 0.015 Å\n")
            
            if moving_atom_idx is not None:
                f.write(f"\nCONSTRAINTS\n")
                f.write("-"*30 + "\n")
                f.write(f"Moving atom index: {moving_atom_idx}\n")
                f.write(f"Freeze radius: {freeze_radius} Å\n")
                f.write(f"Atoms beyond {freeze_radius} Å from atom {moving_atom_idx} are frozen\n")
            else:
                f.write(f"\nNo constraints applied - all atoms free to move\n")
            
            f.write("\n" + "="*70 + "\n")
    
    def _create_job_script(self, output_dir):
        """Create SLURM job submission script."""
        script = """#!/bin/bash
#SBATCH --job-name=freq_calc
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=128
#SBATCH --time=48:00:00
#SBATCH --partition=bigmem
#SBATCH --qos=normal


# Load modules
module load vasp/6.4.2/standard_vtst199
# Set environment
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# Run VASP
cd $SLURM_SUBMIT_DIR
srun vasp > vasp.out

# Check if calculation completed
if grep -q "General timing and accounting informations for this job:" OUTCAR; then
    echo "Frequency calculation completed successfully"
    
    # Extract frequencies
    echo "" >> calculation_info.txt
    echo "FREQUENCY RESULTS" >> calculation_info.txt
    echo "================" >> calculation_info.txt
    grep "cm-1" OUTCAR >> calculation_info.txt
else
    echo "Frequency calculation may have failed - check vasp.out"
fi
"""
        
        script_path = os.path.join(output_dir, "job_freq.sh")
        with open(script_path, 'w') as f:
            f.write(script)
        
        os.chmod(script_path, 0o755)
        logger.info(f"Created job script: {script_path}")
        
    def analyze_frequencies(self, calc_dir):
        """Analyze completed frequency calculation."""
        outcar_path = os.path.join(calc_dir, "OUTCAR")
        
        if not os.path.exists(outcar_path):
            logger.error(f"OUTCAR not found in {calc_dir}")
            return None
            
        frequencies = []
        with open(outcar_path, 'r') as f:
            for line in f:
                if "cm-1" in line and "THz" in line:
                    parts = line.split()
                    if len(parts) >= 8:
                        freq_cm = float(parts[7])
                        frequencies.append(freq_cm)
        
        # Analyze
        imaginary = [f for f in frequencies if f < 0]
        real = [f for f in frequencies if f >= 0]
        
        results = {
            'total_frequencies': len(frequencies),
            'n_imaginary': len(imaginary),
            'n_real': len(real),
            'imaginary_frequencies': sorted(imaginary),
            'real_frequencies': sorted(real),
            'is_transition_state': len(imaginary) == 1
        }
        
        # Print summary
        print("\nFrequency Analysis Results:")
        print(f"  Total modes: {results['total_frequencies']}")
        print(f"  Imaginary frequencies: {results['n_imaginary']}")
        print(f"  Real frequencies: {results['n_real']}")
        
        if results['is_transition_state']:
            print(f"\n✓ Valid transition state (1 imaginary frequency: {imaginary[0]:.2f} cm⁻¹)")
        else:
            print(f"\n✗ Not a valid transition state ({results['n_imaginary']} imaginary frequencies)")
            
        return results


def main(args=None):
    """Main function to run frequency calculation."""
    parser = argparse.ArgumentParser(description='Simple frequency calculator for transition states')
    parser.add_argument('--input-dir', default='.', help='Directory containing structure file')
    parser.add_argument('--input-file', default='CONTCAR', help='Input structure file name')
    parser.add_argument('--output-dir', default='./freq_calc', help='Output directory')
    parser.add_argument('--moving-atom', type=int, help='Index of moving atom (0-based)')
    parser.add_argument('--freeze-radius', type=float, default=2.2, help='Freeze radius in Angstrom')
    parser.add_argument('--submit', action='store_true', help='Submit job to SLURM')
    parser.add_argument('--monitor', action='store_true', help='Monitor job completion')
    parser.add_argument('--analyze', help='Analyze completed calculation in given directory')
    parser.add_argument('--source-info', help='Additional info about structure source')
    
    args = parser.parse_args(args)
    
    # Initialize calculator
    calc = SimpleFrequencyCalculator()
    
    # If analyzing existing calculation
    if args.analyze:
        logger.info(f"Analyzing calculation in {args.analyze}")
        calc.analyze_frequencies(args.analyze)
        return
    
    # Setup new calculation
    structure_file = os.path.join(args.input_dir, args.input_file)
    
    if not os.path.exists(structure_file):
        logger.error(f"Structure file not found: {structure_file}")
        sys.exit(1)
    
    # Prepare source info
    source_info = args.source_info
    if not source_info:
        # Try to infer from directory structure
        parent_dir = os.path.basename(os.path.abspath(args.input_dir))
        if 'neb' in parent_dir.lower():
            source_info = "Structure from NEB calculation"
        elif any(x in parent_dir for x in ['01', '02', '03', '04', '05']):
            source_info = f"Structure from NEB image {parent_dir}"
    
    # Setup calculation
    success = calc.setup_frequency_calculation(
        structure_file=structure_file,
        output_dir=args.output_dir,
        moving_atom_idx=args.moving_atom,
        freeze_radius=args.freeze_radius,
        source_info=source_info
    )
    
    if not success:
        logger.error("Failed to setup calculation")
        sys.exit(1)
    
    print(f"\nFrequency calculation prepared in: {args.output_dir}")
    print(f"Source structure: {structure_file}")
    
    if args.moving_atom is not None:
        print(f"Moving atom: {args.moving_atom} (atoms > {args.freeze_radius} Å away are frozen)")
    
    # Submit if requested
    if args.submit:
        os.chdir(args.output_dir)
        os.system('sbatch job_freq.sh')
        print(f"\nJob submitted from {args.output_dir}")
        print("Check job status with: squeue -u $USER")
        
        if args.monitor:
            print("\nMonitoring job... (this may take a while)")
            # Simple monitoring - you can enhance this
            import time
            while True:
                time.sleep(60)
                if os.path.exists('OUTCAR') and 'General timing' in open('OUTCAR').read():
                    print("Job completed!")
                    calc.analyze_frequencies(args.output_dir)
                    break
    else:
        print("\nTo submit the job:")
        print(f"  cd {args.output_dir}")
        print("  sbatch job_freq.sh")
        print("\nTo analyze after completion:")
        print(f"  python {sys.argv[0]} --analyze {args.output_dir}")


if __name__ == "__main__":
    custom_args = [
        '--input-dir', 'multistage_neb/neb_stage3/03',
        '--output-dir', 'freq_calc_stage3',
        '--moving-atom', '5',
        '--freeze-radius', '2.5',
        '--submit'
    ]
    main(args=custom_args)

