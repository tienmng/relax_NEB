#!/usr/bin/env python
"""
Transition State Frequency Calculation Module
=============================================
Performs frequency calculations on transition state structures from NEB
to verify the presence of exactly one imaginary frequency.
"""

import os
import sys
import logging
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from pymatgen.core import Structure
from pymatgen.io.vasp.inputs import Poscar, Incar, Kpoints
from pymatgen.io.vasp.outputs import Outcar, Vasprun

# Set up logging
logger = logging.getLogger(__name__)

class TransitionStateFrequencyCalculator:
    """
    Handles frequency calculations for transition state verification.
    """
    
    def __init__(self, file_manager, input_generator, slurm_manager, structure_analyzer):
        """
        Initialize the frequency calculator.
        
        Args:
            file_manager: FileManager instance for file operations
            input_generator: VASPInputGenerator for creating input files
            slurm_manager: SLURMManager for job submission
            structure_analyzer: StructureAnalyzer for distance calculations
        """
        self.file_manager = file_manager
        self.input_generator = input_generator
        self.slurm_manager = slurm_manager
        self.structure_analyzer = structure_analyzer
        
        # Default parameters
        self.freeze_radius = 5.0  # Angstrom
        self.displacement = 0.01  # Angstrom for finite differences
        
    def identify_transition_state(self, energy_profile_file: str) -> Tuple[int, float, float]:
        """
        Identify the transition state from the energy profile.
        
        Args:
            energy_profile_file: Path to final_energy_profile.dat
            
        Returns:
            Tuple of (ts_image_index, ts_energy, ts_distance)
        """
        logger.info(f"Reading energy profile from {energy_profile_file}")
        
        try:
            # Read the energy profile data
            data = np.loadtxt(energy_profile_file, skiprows=1)  # Skip header
            
            if len(data.shape) == 1:
                # Single row of data
                distances = [data[0]]
                energies = [data[1]]
                relative_energies = [data[2]]
            else:
                distances = data[:, 0]
                energies = data[:, 1]
                relative_energies = data[:, 2]
            
            # Find the maximum energy excluding endpoints
            # Assume first and last are endpoints
            interior_indices = list(range(1, len(energies) - 1))
            
            if not interior_indices:
                raise ValueError("Not enough images to identify transition state")
            
            # Find max energy among interior images
            max_energy = -np.inf
            ts_index = None
            
            for idx in interior_indices:
                if relative_energies[idx] > max_energy:
                    max_energy = relative_energies[idx]
                    ts_index = idx
            
            if ts_index is None:
                raise ValueError("Could not identify transition state")
            
            ts_energy = energies[ts_index]
            ts_distance = distances[ts_index]
            
            logger.info(f"Identified transition state:")
            logger.info(f"  Image index: {ts_index}")
            logger.info(f"  Energy: {ts_energy:.6f} eV")
            logger.info(f"  Relative energy: {max_energy:.6f} eV")
            logger.info(f"  Distance along path: {ts_distance:.3f} Å")
            
            return ts_index, ts_energy, ts_distance
            
        except Exception as e:
            logger.error(f"Error identifying transition state: {e}")
            raise
    
    def get_ts_structure_from_neb(self, neb_dir: str, ts_index: int) -> Structure:
        """
        Extract the transition state structure from NEB calculation.
        
        Args:
            neb_dir: Directory containing NEB calculation
            ts_index: Index of the transition state image
            
        Returns:
            Transition state Structure object
        """
        # Format image directory name
        img_dir = os.path.join(neb_dir, f"{ts_index:02d}")
        contcar_path = os.path.join(img_dir, "CONTCAR")
        poscar_path = os.path.join(img_dir, "POSCAR")
        
        # Try CONTCAR first (optimized structure)
        if os.path.exists(contcar_path) and os.path.getsize(contcar_path) > 0:
            logger.info(f"Reading TS structure from {contcar_path}")
            self.file_manager.clean_contcar_elements(contcar_path)
            structure = Structure.from_file(contcar_path)
        elif os.path.exists(poscar_path):
            logger.info(f"Reading TS structure from {poscar_path}")
            structure = Structure.from_file(poscar_path)
        else:
            raise FileNotFoundError(f"No structure file found for image {ts_index:02d}")
        
        return structure
    
    def identify_atoms_to_freeze(self, structure: Structure, moving_atom_idx: int,
                                freeze_radius: float = None) -> List[int]:
        """
        Identify atoms to freeze based on distance from moving atom.
        
        Args:
            structure: Structure object
            moving_atom_idx: Index of the moving atom
            freeze_radius: Radius beyond which atoms are frozen (Å)
            
        Returns:
            List of atom indices to freeze
        """
        if freeze_radius is None:
            freeze_radius = self.freeze_radius
        
        logger.info(f"Identifying atoms to freeze (radius > {freeze_radius:.1f} Å from atom {moving_atom_idx})")
        
        # Get position of moving atom
        moving_atom_pos = structure[moving_atom_idx].coords
        
        atoms_to_freeze = []
        atoms_to_relax = []
        
        # Check each atom
        for i, site in enumerate(structure):
            # Calculate distance using PBC
            distance = self.structure_analyzer.get_pbc_distance(
                structure, site.coords, moving_atom_pos
            )
            
            if distance > freeze_radius:
                atoms_to_freeze.append(i)
            else:
                atoms_to_relax.append(i)
        
        logger.info(f"Atoms to freeze: {len(atoms_to_freeze)}")
        logger.info(f"Atoms to relax: {len(atoms_to_relax)}")
        
        # Log some details
        elements_to_freeze = {}
        for idx in atoms_to_freeze:
            elem = str(structure[idx].specie)
            elements_to_freeze[elem] = elements_to_freeze.get(elem, 0) + 1
        
        logger.info("Frozen atoms by element:")
        for elem, count in sorted(elements_to_freeze.items()):
            logger.info(f"  {elem}: {count}")
        
        return atoms_to_freeze
    
    def create_frequency_incar(self, structure: Structure, ldau_settings: Optional[Dict] = None,
                              ediff: float = 1E-8, potim: float = 0.015) -> Incar:
        """
        Create INCAR for frequency calculation.
        
        Args:
            structure: Structure for the calculation
            ldau_settings: LDAU parameters
            ediff: Electronic convergence (tighter for frequencies)
            potim: Displacement for finite differences
            
        Returns:
            Incar object
        """
        incar = Incar()
        
        # Frequency calculation settings
        incar['IBRION'] = 5          # Finite differences
        incar['POTIM'] = potim       # Displacement in Angstrom
        incar['NFREE'] = 2           # Number of displacements (2 = central differences)
        
        # Electronic structure settings (high accuracy)
        incar['EDIFF'] = ediff       # Very tight convergence
        incar['PREC'] = 'Accurate'   # High precision
        incar['ENCUT'] = 600         # High cutoff
        incar['LREAL'] = 'False'     # Exact projectors for accuracy
        incar['ALGO'] = 'Normal'     # Reliable algorithm
        incar['NELM'] = 200          # Allow more electronic steps
        
        # Other settings
        incar['ISMEAR'] = 0
        incar['SIGMA'] = 0.05
        incar['LASPH'] = True
        incar['LMAXMIX'] = 6
        incar['NCORE'] = 32
        incar['LSCALAPACK'] = True
        incar['LSCALU'] = False
        
        # Output settings
        incar['LWAVE'] = False       # Don't need wavefunction
        incar['LCHARG'] = False      # Don't need charge density
        
        # Apply LDAU settings if provided
        if ldau_settings:
            elements = [str(site.specie) for site in structure]
            unique_elements = list(dict.fromkeys(elements))
            incar = self.input_generator._apply_ldau_settings(incar, unique_elements, ldau_settings)
        
        # Apply magnetic moments
        incar = self.input_generator._apply_magmom(structure, incar)
        
        return incar
    
    def setup_frequency_calculation(self, structure: Structure, calc_dir: str,
                                   moving_atom_idx: int, atoms_to_freeze: List[int],
                                   ldau_settings: Optional[Dict] = None,
                                   kspacing: float = 0.3) -> bool:
        """
        Set up frequency calculation directory with all input files.
        
        Args:
            structure: Transition state structure
            calc_dir: Directory for frequency calculation
            moving_atom_idx: Index of moving atom
            atoms_to_freeze: List of atom indices to freeze
            ldau_settings: LDAU parameters
            kspacing: K-point spacing
            
        Returns:
            True if successful
        """
        try:
            # Create calculation directory
            os.makedirs(calc_dir, exist_ok=True)
            logger.info(f"Setting up frequency calculation in {calc_dir}")
            
            # Write POSCAR with selective dynamics
            self._write_poscar_with_constraints(
                structure, os.path.join(calc_dir, "POSCAR"), 
                atoms_to_freeze
            )
            
            # Create INCAR
            incar = self.create_frequency_incar(structure, ldau_settings)
            incar.write_file(os.path.join(calc_dir, "INCAR"))
            
            # Create KPOINTS
            kpoints = self.input_generator.generate_kpoints(structure, kspacing)
            kpoints.write_file(os.path.join(calc_dir, "KPOINTS"))
            
            # Create POTCAR
            elements = [str(site.specie) for site in structure]
            unique_elements = list(dict.fromkeys(elements))
            self.file_manager.create_potcar(calc_dir, unique_elements)
            
            # Write info file
            info_file = os.path.join(calc_dir, "frequency_info.txt")
            with open(info_file, 'w') as f:
                f.write("Transition State Frequency Calculation\n")
                f.write("=====================================\n\n")
                f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Moving atom index: {moving_atom_idx}\n")
                f.write(f"Freeze radius: {self.freeze_radius:.1f} Å\n")
                f.write(f"Total atoms: {len(structure)}\n")
                f.write(f"Frozen atoms: {len(atoms_to_freeze)}\n")
                f.write(f"Free atoms: {len(structure) - len(atoms_to_freeze)}\n")
                f.write(f"Displacement (POTIM): {incar['POTIM']} Å\n")
                f.write(f"Electronic convergence: {incar['EDIFF']}\n")
            
            logger.info("Frequency calculation setup complete")
            return True
            
        except Exception as e:
            logger.error(f"Error setting up frequency calculation: {e}")
            return False
    
    def _write_poscar_with_constraints(self, structure: Structure, filepath: str,
                                      atoms_to_freeze: List[int]):
        """
        Write POSCAR with selective dynamics for frozen atoms.
        
        Args:
            structure: Structure object
            filepath: Output file path
            atoms_to_freeze: List of atom indices to freeze
        """
        # Create a Poscar object
        poscar = Poscar(structure)
        
        # Enable selective dynamics
        poscar.selective_dynamics = []
        
        # Set up selective dynamics flags
        for i in range(len(structure)):
            if i in atoms_to_freeze:
                # Freeze this atom (False False False)
                poscar.selective_dynamics.append([False, False, False])
            else:
                # Allow this atom to move (True True True)
                poscar.selective_dynamics.append([True, True, True])
        
        # Write the file
        poscar.write_file(filepath)
        logger.info(f"Wrote POSCAR with selective dynamics to {filepath}")
    
    def run_frequency_calculation(self, calc_dir: str, job_name: str = "ts_freq",
                                 nodes: int = 2, walltime: str = "24:00:00",
                                 submit: bool = True, monitor: bool = True) -> Optional[str]:
        """
        Submit and optionally monitor frequency calculation.
        
        Args:
            calc_dir: Calculation directory
            job_name: SLURM job name
            nodes: Number of nodes
            walltime: Wall time limit
            submit: Whether to submit the job
            monitor: Whether to monitor the job
            
        Returns:
            Job ID if submitted, None otherwise
        """
        # Create job script
        script_path = self.slurm_manager.create_vasp_job_script(
            job_dir=calc_dir,
            job_name=job_name,
            nodes=nodes,
            ntasks_per_node=128,
            walltime=walltime,
            auto_restart=False  # Frequency calculations shouldn't need restart
        )
        
        if not submit:
            logger.info(f"Job script created at {script_path} (not submitted)")
            return None
        
        # Submit job
        job_id = self.slurm_manager.submit_job(script_path, calc_dir)
        
        if not job_id:
            logger.error("Failed to submit frequency calculation")
            return None
        
        logger.info(f"Frequency calculation submitted with job ID: {job_id}")
        
        if monitor:
            # Monitor job completion
            success = self.slurm_manager.monitor_job(job_id, quiet=True)
            
            if success:
                logger.info("Frequency calculation completed successfully")
                return job_id
            else:
                logger.error("Frequency calculation failed")
                return None
        
        return job_id
    
    def analyze_frequencies(self, calc_dir: str) -> Dict:
        """
        Analyze frequency calculation results.
        
        Args:
            calc_dir: Directory containing frequency calculation
            
        Returns:
            Dictionary with frequency analysis results
        """
        outcar_path = os.path.join(calc_dir, "OUTCAR")
        
        if not os.path.exists(outcar_path):
            logger.error(f"OUTCAR not found in {calc_dir}")
            return {}
        
        try:
            # Read frequencies from OUTCAR
            frequencies = []
            with open(outcar_path, 'r') as f:
                lines = f.readlines()
            
            # Find frequency section
            for i, line in enumerate(lines):
                if "THz" in line and "2PiTHz" in line and "cm-1" in line and "meV" in line:
                    # This is a frequency line
                    parts = line.split()
                    if len(parts) >= 8:
                        freq_cm = float(parts[7])  # Frequency in cm^-1
                        frequencies.append(freq_cm)
            
            logger.info(f"Found {len(frequencies)} frequencies")
            
            # Separate real and imaginary frequencies
            imaginary_freqs = [f for f in frequencies if f < 0]
            real_freqs = [f for f in frequencies if f >= 0]
            
            # Sort frequencies
            imaginary_freqs.sort()
            real_freqs.sort()
            
            # Analyze results
            n_imaginary = len(imaginary_freqs)
            is_transition_state = (n_imaginary == 1)
            
            results = {
                'all_frequencies': frequencies,
                'imaginary_frequencies': imaginary_freqs,
                'real_frequencies': real_freqs,
                'n_imaginary': n_imaginary,
                'n_real': len(real_freqs),
                'is_transition_state': is_transition_state,
                'lowest_real_freq': real_freqs[0] if real_freqs else None,
                'highest_imaginary_freq': imaginary_freqs[-1] if imaginary_freqs else None
            }
            
            # Log results
            logger.info(f"Frequency analysis results:")
            logger.info(f"  Total frequencies: {len(frequencies)}")
            logger.info(f"  Imaginary frequencies: {n_imaginary}")
            logger.info(f"  Real frequencies: {len(real_freqs)}")
            
            if imaginary_freqs:
                logger.info(f"  Imaginary frequencies (cm⁻¹): {imaginary_freqs}")
            
            if is_transition_state:
                logger.info("  ✓ Structure is a valid transition state (1 imaginary frequency)")
            else:
                logger.warning(f"  ✗ Structure is NOT a valid transition state ({n_imaginary} imaginary frequencies)")
            
            return results
            
        except Exception as e:
            logger.error(f"Error analyzing frequencies: {e}")
            return {}
    
    def plot_frequency_spectrum(self, results: Dict, output_path: str) -> Optional[str]:
        """
        Create a plot of the frequency spectrum.
        
        Args:
            results: Frequency analysis results
            output_path: Path for output plot
            
        Returns:
            Path to created plot or None
        """
        try:
            frequencies = results.get('all_frequencies', [])
            if not frequencies:
                logger.warning("No frequencies to plot")
                return None
            
            # Create figure
            plt.figure(figsize=(10, 6))
            
            # Separate imaginary and real frequencies
            imag_freqs = [f for f in frequencies if f < 0]
            real_freqs = [f for f in frequencies if f >= 0]
            
            # Plot frequencies
            if imag_freqs:
                plt.scatter(range(len(imag_freqs)), imag_freqs, 
                           color='red', s=100, label='Imaginary', marker='o')
            
            if real_freqs:
                plt.scatter(range(len(imag_freqs), len(imag_freqs) + len(real_freqs)), 
                           real_freqs, color='blue', s=50, label='Real', marker='o')
            
            # Add horizontal line at zero
            plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            
            # Labels and formatting
            plt.xlabel('Mode Index')
            plt.ylabel('Frequency (cm⁻¹)')
            plt.title('Vibrational Frequency Spectrum of Transition State')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Add text annotation
            n_imag = len(imag_freqs)
            if n_imag == 1:
                plt.text(0.02, 0.98, f'✓ Valid TS: {n_imag} imaginary frequency',
                        transform=plt.gca().transAxes, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
            else:
                plt.text(0.02, 0.98, f'✗ Invalid TS: {n_imag} imaginary frequencies',
                        transform=plt.gca().transAxes, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.5))
            
            # Save plot
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Frequency spectrum plot saved to {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error creating frequency plot: {e}")
            return None
    
    def generate_report(self, results: Dict, calc_dir: str, ts_info: Dict) -> str:
        """
        Generate a comprehensive report of the frequency calculation.
        
        Args:
            results: Frequency analysis results
            calc_dir: Calculation directory
            ts_info: Transition state information (index, energy, etc.)
            
        Returns:
            Path to report file
        """
        report_path = os.path.join(calc_dir, "frequency_analysis_report.txt")
        
        try:
            with open(report_path, 'w') as f:
                # Header
                f.write("="*70 + "\n")
                f.write("TRANSITION STATE FREQUENCY ANALYSIS REPORT\n")
                f.write("="*70 + "\n\n")
                
                # Timestamp
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                # Transition state information
                f.write("TRANSITION STATE INFORMATION\n")
                f.write("-"*30 + "\n")
                f.write(f"Image index: {ts_info.get('index', 'N/A')}\n")
                f.write(f"Energy: {ts_info.get('energy', 'N/A'):.6f} eV\n")
                f.write(f"Relative energy: {ts_info.get('relative_energy', 'N/A'):.3f} eV\n")
                f.write(f"Distance along path: {ts_info.get('distance', 'N/A'):.3f} Å\n\n")
                
                # Frequency analysis results
                f.write("FREQUENCY ANALYSIS RESULTS\n")
                f.write("-"*30 + "\n")
                f.write(f"Total vibrational modes: {len(results.get('all_frequencies', []))}\n")
                f.write(f"Imaginary frequencies: {results.get('n_imaginary', 0)}\n")
                f.write(f"Real frequencies: {results.get('n_real', 0)}\n\n")
                
                # Transition state validation
                f.write("TRANSITION STATE VALIDATION\n")
                f.write("-"*30 + "\n")
                if results.get('is_transition_state', False):
                    f.write("✓ VALID TRANSITION STATE\n")
                    f.write("  The structure has exactly one imaginary frequency.\n")
                    f.write(f"  Imaginary frequency: {results['imaginary_frequencies'][0]:.2f} cm⁻¹\n")
                else:
                    f.write("✗ INVALID TRANSITION STATE\n")
                    n_imag = results.get('n_imaginary', 0)
                    if n_imag == 0:
                        f.write("  No imaginary frequencies found - structure is a minimum.\n")
                    else:
                        f.write(f"  Found {n_imag} imaginary frequencies - structure is a higher-order saddle point.\n")
                f.write("\n")
                
                # Frequency details
                f.write("FREQUENCY DETAILS\n")
                f.write("-"*30 + "\n")
                
                # Imaginary frequencies
                imag_freqs = results.get('imaginary_frequencies', [])
                if imag_freqs:
                    f.write("Imaginary frequencies (cm⁻¹):\n")
                    for i, freq in enumerate(imag_freqs):
                        f.write(f"  {i+1}. {freq:.2f}\n")
                    f.write("\n")
                
                # Low-lying real frequencies
                real_freqs = results.get('real_frequencies', [])
                if real_freqs:
                    f.write("Lowest 10 real frequencies (cm⁻¹):\n")
                    for i, freq in enumerate(real_freqs[:10]):
                        f.write(f"  {i+1}. {freq:.2f}\n")
                    f.write("\n")
                
                # Calculation details
                f.write("CALCULATION DETAILS\n")
                f.write("-"*30 + "\n")
                
                # Read from frequency_info.txt if it exists
                info_file = os.path.join(calc_dir, "frequency_info.txt")
                if os.path.exists(info_file):
                    with open(info_file, 'r') as info:
                        for line in info:
                            if any(key in line for key in ['Moving atom', 'Freeze radius', 
                                                           'Total atoms', 'Frozen atoms', 
                                                           'Free atoms', 'Displacement']):
                                f.write(line)
                
                f.write("\n")
                
                # Recommendations
                f.write("RECOMMENDATIONS\n")
                f.write("-"*30 + "\n")
                
                if results.get('is_transition_state', False):
                    f.write("1. The transition state is correctly identified.\n")
                    f.write("2. You may proceed with reaction rate calculations.\n")
                    f.write("3. Consider visualizing the imaginary mode to confirm the reaction coordinate.\n")
                else:
                    n_imag = results.get('n_imaginary', 0)
                    if n_imag == 0:
                        f.write("1. The structure is a minimum, not a transition state.\n")
                        f.write("2. Check your NEB path - the highest energy image may not be fully converged.\n")
                        f.write("3. Consider tightening NEB convergence criteria.\n")
                    elif n_imag > 1:
                        f.write("1. The structure is a higher-order saddle point.\n")
                        f.write("2. This may indicate an incorrect reaction pathway.\n")
                        f.write("3. Consider using a different initial guess or reaction coordinate.\n")
                
                f.write("\n" + "="*70 + "\n")
            
            logger.info(f"Report generated: {report_path}")
            return report_path
            
        except Exception as e:
            logger.error(f"Error generating report: {e}")
            return None


def run_ts_frequency_analysis(energy_profile_file: str, neb_dir: str, 
                            moving_atom_idx: int, output_dir: str = "./ts_frequency",
                            potcar_path: str = None, potcar_mapping: Dict = None,
                            ldau_settings: Dict = None, freeze_radius: float = 5.0,
                            submit: bool = True, monitor: bool = True) -> Dict:
    """
    Complete workflow for transition state frequency analysis.
    
    Args:
        energy_profile_file: Path to final_energy_profile.dat
        neb_dir: Directory containing NEB calculation
        moving_atom_idx: Index of the moving atom
        output_dir: Directory for frequency calculation
        potcar_path: Path to POTCAR files
        potcar_mapping: Element to POTCAR mapping
        ldau_settings: LDAU parameters
        freeze_radius: Radius for freezing atoms (Å)
        submit: Whether to submit the job
        monitor: Whether to monitor the job
        
    Returns:
        Dictionary with analysis results
    """
    # Import required modules
    from file_manager import FileManager
    from vasp_inputs import VASPInputGenerator
    from slurm_manager import SLURMManager
    from structure_analyzer import StructureAnalyzer
    
    # Initialize components
    file_manager = FileManager(potcar_path, potcar_mapping)
    input_gen = VASPInputGenerator()
    slurm_manager = SLURMManager()
    struct_analyzer = StructureAnalyzer()
    
    # Create frequency calculator
    freq_calc = TransitionStateFrequencyCalculator(
        file_manager, input_gen, slurm_manager, struct_analyzer
    )
    freq_calc.freeze_radius = freeze_radius
    
    logger.info("Starting transition state frequency analysis")
    
    # Step 1: Identify transition state
    ts_index, ts_energy, ts_distance = freq_calc.identify_transition_state(energy_profile_file)
    ts_info = {
        'index': ts_index,
        'energy': ts_energy,
        'distance': ts_distance,
        'relative_energy': None  # Will be filled from energy profile
    }
    
    # Get relative energy
    try:
        data = np.loadtxt(energy_profile_file, skiprows=1)
        if len(data.shape) == 1:
            ts_info['relative_energy'] = data[2] if ts_index == 0 else None
        else:
            ts_info['relative_energy'] = data[ts_index, 2]
    except:
        pass
    
    # Step 2: Get transition state structure
    ts_structure = freq_calc.get_ts_structure_from_neb(neb_dir, ts_index)
    
    # Step 3: Identify atoms to freeze
    atoms_to_freeze = freq_calc.identify_atoms_to_freeze(ts_structure, moving_atom_idx, freeze_radius)
    
    # Step 4: Set up frequency calculation
    success = freq_calc.setup_frequency_calculation(
        ts_structure, output_dir, moving_atom_idx, atoms_to_freeze, 
        ldau_settings, kspacing=0.3
    )
    
    if not success:
        logger.error("Failed to set up frequency calculation")
        return {}
    
    # Step 5: Run calculation (if requested)
    if submit:
        job_id = freq_calc.run_frequency_calculation(
            output_dir, job_name="ts_freq", nodes=2, 
            walltime="24:00:00", submit=True, monitor=monitor
        )
        
        if not job_id and monitor:
            logger.error("Frequency calculation failed")
            return {}
    
    # Step 6: Analyze results (if calculation completed)
    results = {}
    if not submit or monitor:
        results = freq_calc.analyze_frequencies(output_dir)
        
        # Step 7: Generate plots and report
        if results:
            # Create frequency spectrum plot
            plot_path = freq_calc.plot_frequency_spectrum(
                results, os.path.join(output_dir, "frequency_spectrum.png")
            )
            results['plot_path'] = plot_path
            
            # Generate comprehensive report
            report_path = freq_calc.generate_report(results, output_dir, ts_info)
            results['report_path'] = report_path
            
            # Add ts_info to results
            results['ts_info'] = ts_info
    
    return results


# Example standalone usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Example usage after NEB calculation
    print("Transition State Frequency Calculator")
    print("=====================================")
    print()
    print("This module performs frequency calculations on transition states")
    print("identified from NEB calculations to verify they have exactly one")
    print("imaginary frequency.")
    print()
    print("Usage example:")
    print()
    print("from ts_frequency_calculator import run_ts_frequency_analysis")
    print()
    print("# After your NEB calculation completes:")
    print("results = run_ts_frequency_analysis(")
    print("    energy_profile_file='./multistage_neb/final_energy_profile.dat',")
    print("    neb_dir='./multistage_neb/neb_stage3',  # Final NEB stage")
    print("    moving_atom_idx=51,  # Your moving atom index")
    print("    output_dir='./ts_frequency',")
    print("    potcar_path='/path/to/potcars',")
    print("    potcar_mapping={'O': 'O_s', 'Ni': 'Ni_pv', ...},")
    print("    ldau_settings={...},  # Your LDAU settings")
    print("    freeze_radius=5.0,  # Freeze atoms > 5 Å from moving atom")
    print("    submit=True,")
    print("    monitor=True")
    print(")")
    print()
    print("if results.get('is_transition_state'):")
    print("    print('Valid transition state confirmed!')")
    print("    print(f\"Imaginary frequency: {results['imaginary_frequencies'][0]} cm⁻¹\")")
    print("else:")
    print("    print('Not a valid transition state')")
    print("    print(f\"Number of imaginary frequencies: {results['n_imaginary']}\")")
    print()
    print("The module will:")
    print("1. Identify the TS from the energy profile")
    print("2. Extract the TS structure from NEB")
    print("3. Set up a frequency calculation with appropriate constraints")
    print("4. Submit and monitor the VASP job")
    print("5. Analyze the frequencies")
    print("6. Generate plots and a comprehensive report")
    print()
    print("Output files:")
    print("- frequency_info.txt: Calculation setup details")
    print("- frequency_spectrum.png: Visual frequency spectrum")
    print("- frequency_analysis_report.txt: Comprehensive report")