#!/usr/bin/env python
"""
Convenience functions for common NEB workflow tasks.
"""

import os
from typing import Dict, Optional, Union
from pymatgen.core import Structure

from neb_file_manager import FileManager
from neb_vasp_inputs import VASPInputGenerator
from neb_slurm_manager import SLURMManager
from neb_structure_relaxation import StructureRelaxer

def run_relax(structure: Structure, 
              relax_dir: str = "./relax",
              potcar_path: Optional[str] = None,
              potcar_mapping: Optional[Dict[str, str]] = None,
              ldau_settings: Optional[Dict] = None,
              kspacing: float = 0.3,
              max_iterations: int = 200,
              ediffg: float = -0.01,
              submit: bool = True,
              monitor: bool = True,
              auto_cleanup: bool = True,
              quiet: bool = True) -> Union[Dict, str, None]:
    """
    Convenience function to run structure relaxation.
    
    Args:
        structure: Structure to relax
        relax_dir: Directory for relaxation
        potcar_path: Path to POTCAR files
        potcar_mapping: Element to POTCAR mapping
        ldau_settings: LDAU parameters
        kspacing: K-point spacing
        max_iterations: Maximum ionic steps
        ediffg: Force convergence criterion
        submit: Whether to submit job
        monitor: Whether to monitor job
        auto_cleanup: Whether to clean up files
        quiet: Use quiet monitoring
        
    Returns:
        Dict with 'structure' and 'energy' if successful,
        Job ID if submitted but not monitored,
        None if failed
    """
    # Initialize components
    file_manager = FileManager(potcar_path, potcar_mapping)
    input_gen = VASPInputGenerator()
    slurm_manager = SLURMManager()
    
    # Create relaxer
    relaxer = StructureRelaxer(file_manager, input_gen, slurm_manager, auto_cleanup)
    
    # Modify input generator to use custom k-spacing
    original_generate_kpoints = input_gen.generate_kpoints
    def custom_generate_kpoints(struct, spacing=None):
        return original_generate_kpoints(struct, spacing or kspacing)
    input_gen.generate_kpoints = custom_generate_kpoints
    
    # Run relaxation
    return relaxer.relax_structure(
        structure=structure,
        relax_dir=relax_dir,
        ldau_settings=ldau_settings,
        max_iterations=max_iterations,
        ediffg=ediffg,
        submit=submit,
        monitor=monitor,
        quiet_monitoring=quiet
    )

def identify_and_move_atom(defect_struct: Structure, perfect_struct: Structure,
                          element_to_move: str = "O") -> Structure:
    """
    Convenience function to identify vacancy and create final structure.
    
    Args:
        defect_struct: Structure with vacancy
        perfect_struct: Perfect structure
        element_to_move: Element to move to vacancy (default: "O")
        
    Returns:
        Final structure with atom moved to vacancy
    """
    from neb_structure_analysis import StructureAnalyzer
    
    analyzer = StructureAnalyzer()
    
    # Identify vacancy
    vacancy_pos, vacancy_idx, vacancy_element = analyzer.identify_vacancy(
        defect_struct, perfect_struct
    )
    
    if vacancy_pos is None:
        raise ValueError("Could not identify vacancy position")
    
    # Find nearest atom of specified element
    nearest_atoms = analyzer.find_nearest_atoms(
        defect_struct, vacancy_pos, element=element_to_move, n_nearest=1
    )
    
    if not nearest_atoms:
        raise ValueError(f"No {element_to_move} atoms found near vacancy")
    
    moving_atom_idx, _ = nearest_atoms[0]
    
    # Create final structure
    final_struct = defect_struct.copy()
    vacancy_frac = final_struct.lattice.get_fractional_coords(vacancy_pos)
    final_struct[moving_atom_idx] = (final_struct[moving_atom_idx].species, vacancy_frac)
    
    return final_struct

def clean_and_load_structures(defect_file: str, perfect_file: str,
                             potcar_path: Optional[str] = None,
                             potcar_mapping: Optional[Dict[str, str]] = None) -> tuple:
    """
    Convenience function to clean and load structures.
    
    Args:
        defect_file: Path to defect POSCAR
        perfect_file: Path to perfect POSCAR
        potcar_path: Path to POTCAR files
        potcar_mapping: Element to POTCAR mapping
        
    Returns:
        Tuple of (defect_structure, perfect_structure, cleaned_defect_file, cleaned_perfect_file)
    """
    file_manager = FileManager(potcar_path, potcar_mapping)
    
    # Create cleaned copies
    import shutil
    import os
    
    base_dir = os.path.dirname(defect_file)
    clean_defect = os.path.join(base_dir, "POSCAR-defect-clean")
    clean_perfect = os.path.join(base_dir, "POSCAR-perfect-clean")
    
    shutil.copy2(defect_file, clean_defect)
    shutil.copy2(perfect_file, clean_perfect)
    
    # Clean files
    file_manager.clean_contcar_elements(clean_defect)
    file_manager.clean_contcar_elements(clean_perfect)
    
    # Load structures
    defect_struct = Structure.from_file(clean_defect)
    perfect_struct = Structure.from_file(clean_perfect)
    
    return defect_struct, perfect_struct, clean_defect, clean_perfect

# Example usage with convenience functions
def example_with_convenience_functions():
    """Example using the convenience functions."""
    
    # Define settings
    potcar_path = "/nfs/home/6/nguyenm/sensor/POTCAR-files/potpaw_PBE_54"
    potcar_mapping = {
        "O": "O_s", "La": "La", "Ni": "Ni_pv", "V": "V_sv",
        "Fe": "Fe_pv", "Co": "Co_pv", "Mn": "Mn_pv"
    }
    
    ldau_settings = {
        'LDAU': True,
        "La": {"L": 0, "U": 0}, "Ni": {"L": 2, "U": 7}, "V": {"L": 2, "U": 3},
        "Fe": {"L": 2, "U": 5}, "Co": {"L": 2, "U": 3.5}, "Mn": {"L": 2, "U": 4},
        "O": {"L": 0, "U": 0}
    }
    
    # Clean and load structures
    defect_struct, perfect_struct, clean_defect, clean_perfect = clean_and_load_structures(
        "POSCAR-defect", "POSCAR-perfect", potcar_path, potcar_mapping
    )
    
    # Run relaxation of initial structure (your original approach but working)
    print("Running initial structure relaxation...")
    relaxed_result = run_relax(
        structure=defect_struct,
        relax_dir="./relax_initial",
        potcar_path=potcar_path,
        potcar_mapping=potcar_mapping,
        ldau_settings=ldau_settings,
        kspacing=0.3,  # Custom k-spacing as you wanted
        submit=True,
        monitor=True,
        quiet=True
    )
    
    if isinstance(relaxed_result, dict):
        relaxed_init = relaxed_result['structure']
        print(f"Initial relaxation complete: E = {relaxed_result['energy']} eV")
        
        # Create final structure
        final_struct = identify_and_move_atom(defect_struct, perfect_struct, "O")
        
        # Run final relaxation
        print("Running final structure relaxation...")
        relaxed_final_result = run_relax(
            structure=final_struct,
            relax_dir="./relax_final",
            potcar_path=potcar_path,
            potcar_mapping=potcar_mapping,
            ldau_settings=ldau_settings,
            kspacing=0.3,
            submit=True,
            monitor=True,
            quiet=True
        )
        
        if isinstance(relaxed_final_result, dict):
            relaxed_final = relaxed_final_result['structure']
            print(f"Final relaxation complete: E = {relaxed_final_result['energy']} eV")
            
            # Now you can use these relaxed structures for NEB setup
            # ... continue with NEB setup using relaxed_init and relaxed_final
    else:
        print("Relaxation failed or returned job ID")

if __name__ == "__main__":
    example_with_convenience_functions()
