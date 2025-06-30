#!/usr/bin/env python
"""
Modified path generation that allows turning off constraints after ASE interpolation.
"""

import numpy as np
import matplotlib.pyplot as plt
import logging
from typing import List, Tuple, Optional, Dict
from ase.io import read, write
from ase.mep import NEB
from ase.optimize import MDMin
from ase.constraints import FixAtoms
from ase.calculators.lj import LennardJones
from pymatgen.core import Structure
from pymatgen.io.vasp.inputs import Poscar

logger = logging.getLogger(__name__)

class NEBPathGenerator:
    """Generates and analyzes NEB reaction paths using ASE with optional constraint removal."""
    
    def __init__(self):
        """Initialize NEB path generator."""
        # Use non-interactive backend for matplotlib
        import matplotlib
        matplotlib.use('Agg')
    
    def generate_neb_path(self, initial_structure: Structure, final_structure: Structure,
                         n_images: int = 5, moving_atom_idx: int = None,
                         cutoff_distance: float = 6.0, 
                         remove_constraints_after_ase: bool = False) -> Tuple[List, int]:
        """
        Generate NEB path between initial and final structures using ASE.
        
        Args:
            initial_structure: Initial state structure
            final_structure: Final state structure
            n_images: Number of intermediate images
            moving_atom_idx: Index of the moving atom (if known)
            cutoff_distance: Distance cutoff for fixing atoms
            remove_constraints_after_ase: If True, remove all constraints after ASE interpolation
            
        Returns:
            Tuple of (list of ASE atoms objects, moving_atom_index)
        """
        # Write temporary POSCAR files for ASE
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(suffix='.vasp', delete=False) as init_file:
            init_poscar = Poscar(initial_structure)
            init_poscar.write_file(init_file.name)
            initial = read(init_file.name)
            os.unlink(init_file.name)
            
        with tempfile.NamedTemporaryFile(suffix='.vasp', delete=False) as final_file:
            final_poscar = Poscar(final_structure)
            final_poscar.write_file(final_file.name)
            final_atoms = read(final_file.name)
            os.unlink(final_file.name)
        
        # Determine moving atom if not provided
        if moving_atom_idx is None:
            moving_atom_idx = self._find_moving_atom(initial, final_atoms)
            if moving_atom_idx is None:
                logger.error("Could not determine moving atom")
                return None, None
        
        logger.info(f"Using atom index {moving_atom_idx} as moving atom")
        
        # Create images
        configs = [initial.copy() for i in range(n_images + 2)]
        configs[-1] = final_atoms.copy()
        
        # Set up constraints (will be used for ASE interpolation)
        constraint = self._setup_constraints(initial, final_atoms, moving_atom_idx, cutoff_distance)
        constraint_ase = FixAtoms(mask=[atom.index != moving_atom_idx for atom in initial])
        
        # Set calculator and constraints for all configs
        for config in configs:
            config.calc = LennardJones()
            if constraint:
                config.set_constraint(constraint_ase)
        
        # Create NEB and interpolate
        try:
            band = NEB(configs)
            band.interpolate(mic=True, apply_constraint=True)
            
            # Quick optimization with LJ potential
            relax = MDMin(band)
            relax.run(fmax=100)
            
            logger.info(f"Generated NEB path with {len(configs)} images")
            
            # REMOVE CONSTRAINTS AFTER ASE INTERPOLATION IF REQUESTED
            if remove_constraints_after_ase:
                logger.info("Removing all constraints after ASE interpolation")
                for config in configs:
                    config.set_constraint([])  # Remove all constraints
                
                # Log constraint removal
                logger.info("All atomic constraints removed - VASP will relax all atoms freely")
            else:
                # Log which atoms are constrained
                if constraint and hasattr(constraint, 'mask'):
                    num_fixed = sum(constraint.mask)
                    num_free = len(constraint.mask) - num_fixed
                    logger.info(f"Keeping constraints: {num_fixed} fixed atoms, {num_free} free atoms")
                    for config in configs:
                        config.set_constraint(constraint)
            
            return configs, moving_atom_idx
            
        except Exception as e:
            logger.error(f"Error generating NEB path: {e}")
            return None, None
    
    def _find_moving_atom(self, initial, final, threshold: float = 0.1) -> Optional[int]:
        """
        Find the atom that moves the most between initial and final states.
        
        Args:
            initial: Initial ASE atoms object
            final: Final ASE atoms object
            threshold: Minimum displacement to consider as moving
            
        Returns:
            Index of moving atom or None
        """
        if len(initial) != len(final):
            logger.error("Structures have different numbers of atoms")
            return None
        
        max_displacement = 0.0
        moving_atom_idx = None
        
        # Get cell and PBC info
        cell = initial.get_cell()
        pbc = initial.get_pbc()
        
        for i in range(len(initial)):
            # Calculate displacement with minimum image convention
            disp_vector = self._get_mic_vector(
                initial[i].position, final[i].position, cell, pbc
            )
            displacement = np.linalg.norm(disp_vector)
            
            if displacement > max_displacement and displacement > threshold:
                max_displacement = displacement
                moving_atom_idx = i
        
        if moving_atom_idx is not None:
            logger.info(f"Found moving atom: index {moving_atom_idx}, displacement: {max_displacement:.3f} Å")
        else:
            logger.warning(f"No atom displaced more than {threshold} Å")
        
        return moving_atom_idx
    
    def _setup_constraints(self, initial, final, moving_atom_idx: int, 
                          cutoff_distance: float) -> Optional[FixAtoms]:
        """
        Set up constraints for NEB calculation.
        
        Args:
            initial: Initial ASE atoms object
            final: Final ASE atoms object
            moving_atom_idx: Index of moving atom
            cutoff_distance: Distance cutoff for fixing atoms
            
        Returns:
            FixAtoms constraint or None
        """
        # Get initial and final positions of moving atom
        initial_pos = initial[moving_atom_idx].position
        final_pos = final[moving_atom_idx].position
        
        # Get cell and PBC information
        cell = initial.get_cell()
        pbc = initial.get_pbc()
        
        # Create mask for atoms to fix
        mask = []
        for i, atom in enumerate(initial):
            if i == moving_atom_idx:
                # Don't fix the moving atom
                mask.append(False)
            else:
                # Calculate distances with minimum image convention
                dist_to_initial = self._get_mic_distance(atom.position, initial_pos, cell, pbc)
                dist_to_final = self._get_mic_distance(atom.position, final_pos, cell, pbc)
                
                # Fix the atom if it's far from both positions
                if dist_to_initial > cutoff_distance and dist_to_final > cutoff_distance:
                    mask.append(True)
                else:
                    mask.append(False)
        
        # Log constraint info
        num_fixed = sum(mask)
        num_free = len(mask) - num_fixed
        logger.info(f"ASE Constraints setup: {num_fixed} fixed atoms, {num_free} free atoms")
        
        if num_fixed > 0:
            return FixAtoms(mask=mask)
        else:
            logger.warning("No atoms will be fixed")
            return None
    
    def _get_mic_vector(self, pos1: np.ndarray, pos2: np.ndarray, 
                       cell: np.ndarray, pbc: List[bool]) -> np.ndarray:
        """Calculate minimum image vector between two positions."""
        vector = pos2 - pos1
        fractional = np.linalg.solve(cell.T, vector)
        
        for i in range(3):
            if pbc[i]:
                while fractional[i] > 0.5:
                    fractional[i] -= 1.0
                while fractional[i] < -0.5:
                    fractional[i] += 1.0
        
        return np.dot(cell.T, fractional)
    
    def _get_mic_distance(self, pos1: np.ndarray, pos2: np.ndarray, 
                         cell: np.ndarray, pbc: List[bool]) -> float:
        """Calculate minimum image distance between two positions."""
        vector = self._get_mic_vector(pos1, pos2, cell, pbc)
        return np.linalg.norm(vector)
    
    def analyze_ase_path(self, configs: List, moving_atom_idx: int, 
                        output_dir: str = ".") -> Dict:
        """
        Analyze the ASE-generated NEB path and create energy profile.
        
        Args:
            configs: List of ASE atoms objects
            moving_atom_idx: Index of moving atom
            output_dir: Directory to save analysis files
            
        Returns:
            Dictionary with analysis results
        """
        if not configs:
            logger.error("No configurations provided for analysis")
            return {}
        
        # Calculate energies and distances
        energies = []
        cumulative_distances = [0.0]
        
        # Get energy of first image for reference
        e0 = configs[0].get_potential_energy()
        energies.append(0.0)  # First image relative to itself
        
        # Calculate energies and cumulative distances
        for i in range(1, len(configs)):
            # Get energy relative to first image
            energy = configs[i].get_potential_energy() - e0
            energies.append(energy)
            
            # Calculate distance from previous image
            prev_pos = configs[i-1][moving_atom_idx].position
            curr_pos = configs[i][moving_atom_idx].position
            
            # Get minimum image vector
            cell = configs[i].get_cell()
            pbc = configs[i].get_pbc()
            vec = self._get_mic_vector(prev_pos, curr_pos, cell, pbc)
            distance = np.linalg.norm(vec)
            
            cumulative_distances.append(cumulative_distances[-1] + distance)
        
        # Find barrier and reaction energy
        max_energy = max(energies)
        max_idx = energies.index(max_energy)
        final_energy = energies[-1]
        
        # Create plot
        plt.figure(figsize=(10, 6))
        plt.plot(cumulative_distances, energies, 'o-', linewidth=2)
        
        # Mark important points
        plt.plot(cumulative_distances[0], energies[0], 'o', color='green', 
                markersize=10, label='Initial state')
        plt.plot(cumulative_distances[-1], energies[-1], 'o', color='red', 
                markersize=10, label='Final state')
        plt.plot(cumulative_distances[max_idx], energies[max_idx], 'o', 
                color='purple', markersize=10, label='Transition state')
        
        # Add annotations
        plt.annotate(f'Barrier: {max_energy:.2f} eV',
                    xy=(cumulative_distances[max_idx], energies[max_idx]),
                    xytext=(cumulative_distances[max_idx], energies[max_idx] + max_energy/10),
                    ha='center', va='bottom')
        
        plt.annotate(f'ΔE: {final_energy:.2f} eV',
                    xy=(cumulative_distances[-1], energies[-1]),
                    xytext=(cumulative_distances[-1] - 0.1, energies[-1] + max_energy/10),
                    ha='right', va='bottom')
        
        plt.xlabel('Cumulative distance along reaction path (Å)')
        plt.ylabel('Energy relative to initial state (eV)')
        plt.title('ASE NEB Energy Profile (Pre-VASP)')
        plt.grid(True)
        plt.legend(loc='best')
        
        # Save plot
        import os
        plot_path = os.path.join(output_dir, 'ase_energy_profile.png')
        plt.savefig(plot_path, dpi=300)
        plt.close()
        
        # Save data
        data_path = os.path.join(output_dir, 'ase_energy_profile.dat')
        with open(data_path, 'w') as f:
            f.write("# Distance(Å)  Energy(eV)  RelEnergy(eV)\n")
            for i in range(len(configs)):
                abs_energy = configs[i].get_potential_energy()
                f.write(f"{cumulative_distances[i]:.6f}  {abs_energy:.6f}  {energies[i]:.6f}\n")
        
        logger.info(f"ASE analysis complete: barrier={max_energy:.3f} eV, "
                   f"reaction energy={final_energy:.3f} eV")
        logger.info(f"Plot saved to {plot_path}, data saved to {data_path}")
        
        return {
            "plot_file": plot_path,
            "data_file": data_path,
            "barrier": max_energy,
            "reaction_energy": final_energy,
            "energies": energies,
            "distances": cumulative_distances
        }
    
    def write_neb_structures(self, configs: List, neb_dir: str, n_images: int,
                           write_constraint_info: bool = True) -> bool:
        """
        Write NEB structures to POSCAR files.
        
        Args:
            configs: List of ASE atoms objects
            neb_dir: NEB calculation directory
            n_images: Number of NEB images
            write_constraint_info: Whether to write constraint information to a file
            
        Returns:
            True if successful
        """
        try:
            # Create image directories
            import os
            for i in range(n_images + 2):
                img_dir = os.path.join(neb_dir, f"{i:02d}" if i < n_images+1 else f"{n_images+1:02d}")
                os.makedirs(img_dir, exist_ok=True)
            
            # Write structures
            constraint_info = []
            for i in range(len(configs)):
                img_dir = os.path.join(neb_dir, f"{i:02d}" if i < n_images+1 else f"{n_images+1:02d}")
                poscar_path = os.path.join(img_dir, "POSCAR")
                write(poscar_path, configs[i], format='vasp', direct=True)
                logger.debug(f"Wrote structure to {poscar_path}")
                
                # Check constraint status for this image
                if hasattr(configs[i], 'constraints') and configs[i].constraints:
                    constraint_info.append(f"Image {i:02d}: {len(configs[i].constraints)} constraints")
                else:
                    constraint_info.append(f"Image {i:02d}: No constraints")
            
            # Write constraint information to file
            if write_constraint_info:
                constraint_file = os.path.join(neb_dir, "constraint_info.txt")
                with open(constraint_file, 'w') as f:
                    f.write("NEB Constraint Information\n")
                    f.write("=========================\n\n")
                    for info in constraint_info:
                        f.write(info + "\n")
                    
                    # Add summary
                    has_constraints = any("No constraints" not in info for info in constraint_info)
                    f.write(f"\nSummary: {'Constraints applied' if has_constraints else 'No constraints'}\n")
                logger.info(f"Constraint information written to {constraint_file}")
            
            logger.info(f"Successfully wrote {len(configs)} NEB structures")
            return True
            
        except Exception as e:
            logger.error(f"Error writing NEB structures: {e}")
            return False
    
    def create_neb_visualization_files(self, configs: List, neb_dir: str, n_images: int,
                                    moving_atom_idx: int, output_subdir: str = "visualization") -> bool:
        """
        Create visualization files with frozen atoms as C and moving atom as S.
        
        Args:
            configs: List of ASE atoms objects
            neb_dir: NEB calculation directory  
            n_images: Number of NEB images
            moving_atom_idx: Index of moving atom
            output_subdir: Subdirectory name for visualization files
            
        Returns:
            True if successful
        """
        try:
            import os
            
            # Create visualization directory
            vis_dir = os.path.join(neb_dir, output_subdir)
            os.makedirs(vis_dir, exist_ok=True)
            
            logger.info(f"Creating visualization files in {vis_dir}")
            logger.info("Frozen atoms → Carbon (C), Moving atom → Sulfur (S)")
            
            for i in range(len(configs)):
                # Create visualization structure
                vis_config = configs[i].copy()
                symbols = list(vis_config.get_chemical_symbols())
                
                # Get constraint information
                constraint_mask = None
                if hasattr(configs[i], 'constraints') and configs[i].constraints:
                    constraint = configs[i].constraints[0]
                    if hasattr(constraint, 'mask'):
                        constraint_mask = constraint.mask
                
                # Modify symbols for visualization
                for j in range(len(symbols)):
                    if j == moving_atom_idx:
                        symbols[j] = 'S'  # Moving atom as Sulfur
                    elif constraint_mask is not None and constraint_mask[j]:
                        symbols[j] = 'C'  # Frozen atoms as Carbon
                
                vis_config.set_chemical_symbols(symbols)
                
                # Write visualization file
                img_name = f"image_{i:02d}_vis.vasp"
                vis_path = os.path.join(vis_dir, img_name)
                write(vis_path, vis_config, format='vasp', direct=True)
                
                logger.debug(f"Created visualization file: {vis_path}")
            
            # Create a README file explaining the visualization
            readme_path = os.path.join(vis_dir, "README_visualization.txt")
            with open(readme_path, 'w') as f:
                f.write("NEB Path Visualization Files\n")
                f.write("===========================\n\n")
                f.write("These files are for VISUALIZATION ONLY - do not use for calculations!\n\n")
                f.write("Atom types in visualization:\n")
                f.write("- Carbon (C): Frozen/constrained atoms\n")
                f.write("- Sulfur (S): Moving atom\n\n")
                f.write(f"Moving atom index: {moving_atom_idx}\n")
                f.write(f"Total images: {len(configs)}\n")
            
            logger.info(f"Created {len(configs)} visualization files with C/S atoms")
            logger.info(f"README written to {readme_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error creating visualization files: {e}")
            return False
