#!/usr/bin/env python
"""
Structure analysis utilities for NEB calculations.
Handles vacancy identification, atomic distances, and structure analysis.
"""

import numpy as np
import logging
from typing import Tuple, Optional, List
from pymatgen.core import Structure

logger = logging.getLogger(__name__)

class StructureAnalyzer:
    """Analyzes crystal structures for NEB calculations."""
    
    def __init__(self):
        """Initialize structure analyzer."""
        pass
    
    def identify_vacancy(self, defect_structure: Structure, 
                        original_structure: Structure, 
                        threshold: float = 0.5) -> Tuple[Optional[np.ndarray], Optional[int], Optional[str]]:
        """
        Identify vacancy position by comparing defect and original structures.
        
        Args:
            defect_structure: Structure with vacancy
            original_structure: Perfect structure
            threshold: Distance threshold in Angstroms for site matching
            
        Returns:
            Tuple of (vacancy_position, vacancy_index, element)
        """
        # Check if structures have different number of sites
        if len(original_structure) <= len(defect_structure):
            logger.warning("Original structure doesn't have more atoms than defect structure")
            return None, None, None
        
        # Use Cartesian coordinates for comparison
        defect_coords = defect_structure.cart_coords
        original_coords = original_structure.cart_coords
        
        # Find missing site
        missing_sites = []
        for i, site in enumerate(original_structure):
            found = False
            for j, defect_site in enumerate(defect_structure):
                dist = np.linalg.norm(defect_coords[j] - original_coords[i])
                if dist < threshold:
                    found = True
                    break
            
            if not found:
                missing_sites.append((i, site.coords, site.species_string))
        
        if not missing_sites:
            logger.warning("No vacancy found by comparison")
            return None, None, None
        
        for site in missing_sites:
            vacancy_idx, vacancy_pos, element = site
            logger.info(f"Found {element} vacancy at position {vacancy_pos}")
        
        return missing_sites[0][1], missing_sites[0][0], missing_sites[0][2]
    
    def find_nearest_atoms(self, structure: Structure, target_position: np.ndarray,
                          element: str = None, exclude_indices: List[int] = None,
                          n_nearest: int = 1) -> List[Tuple[int, float]]:
        """
        Find nearest atoms to a target position.
        
        Args:
            structure: Crystal structure
            target_position: Target position in Cartesian coordinates
            element: Only consider atoms of this element (optional)
            exclude_indices: List of atom indices to exclude
            n_nearest: Number of nearest atoms to return
            
        Returns:
            List of tuples (atom_index, distance)
        """
        exclude_indices = exclude_indices or []
        candidates = []
        
        for i, site in enumerate(structure):
            if i in exclude_indices:
                continue
            
            if element and str(site.specie) != element:
                continue
            
            # Calculate distance considering PBC
            dist = self.get_pbc_distance(structure, site.coords, target_position)
            candidates.append((i, dist))
        
        # Sort by distance and return n_nearest
        candidates.sort(key=lambda x: x[1])
        return candidates[:n_nearest]
    
    def get_pbc_distance(self, structure: Structure, pos1: np.ndarray, 
                        pos2: np.ndarray) -> float:
        """
        Calculate distance between two positions considering periodic boundary conditions.
        
        Args:
            structure: Crystal structure for lattice information
            pos1: First position (Cartesian)
            pos2: Second position (Cartesian)
            
        Returns:
            float: Minimum distance considering PBC
        """
        # Get the cell matrix and PBC settings
        cell = structure.lattice.matrix
        pbc = [True, True, True]  # Assuming all directions are periodic
        
        # Use the existing get_mic_vector function
        mic_vector = self.get_mic_vector(pos1, pos2, cell, pbc)
        
        # Return the norm of the vector
        return np.linalg.norm(mic_vector)
    
    def get_mic_vector(self, pos1: np.ndarray, pos2: np.ndarray, 
                      cell: np.ndarray, pbc: List[bool]) -> np.ndarray:
        """
        Calculate minimum image convention vector from pos1 to pos2.
        
        Args:
            pos1: Starting position
            pos2: Ending position
            cell: Unit cell matrix
            pbc: Periodic boundary conditions
            
        Returns:
            np.ndarray: Minimum image vector
        """
        # Calculate direct vector
        vector = pos2 - pos1
        
        # Convert to fractional coordinates
        fractional = np.linalg.solve(cell.T, vector)
        
        # Apply minimum image convention in fractional coordinates
        for i in range(3):
            if pbc[i]:
                # Wrap fractional coordinate to [-0.5, 0.5]
                while fractional[i] > 0.5:
                    fractional[i] -= 1.0
                while fractional[i] < -0.5:
                    fractional[i] += 1.0
        
        # Convert back to Cartesian coordinates
        return np.dot(cell.T, fractional)
    
    def get_mic_distance(self, pos1: np.ndarray, pos2: np.ndarray, 
                        cell: np.ndarray, pbc: List[bool]) -> float:
        """
        Calculate minimum image distance with PBC.
        
        Args:
            pos1: First position
            pos2: Second position
            cell: Unit cell matrix
            pbc: Periodic boundary conditions
            
        Returns:
            float: Minimum image distance
        """
        vector = self.get_mic_vector(pos1, pos2, cell, pbc)
        return np.linalg.norm(vector)
    
    def analyze_path_distances(self, structures: List[Structure], 
                              moving_atom_idx: int) -> List[float]:
        """
        Calculate cumulative distances along a reaction path.
        
        Args:
            structures: List of structures along the path
            moving_atom_idx: Index of the moving atom
            
        Returns:
            List of cumulative distances
        """
        if not structures:
            return []
        
        cumulative_distances = [0.0]
        
        for i in range(1, len(structures)):
            prev_pos = structures[i-1][moving_atom_idx].coords
            curr_pos = structures[i][moving_atom_idx].coords
            
            # Get distance considering PBC
            dist = self.get_pbc_distance(structures[i], prev_pos, curr_pos)
            cumulative_distances.append(cumulative_distances[-1] + dist)
        
        return cumulative_distances
    
    def determine_moving_atom(self, initial_structure: Structure, 
                             final_structure: Structure,
                             threshold: float = 0.1) -> Optional[int]:
        """
        Determine which atom moves the most between initial and final structures.
        
        Args:
            initial_structure: Initial structure
            final_structure: Final structure
            threshold: Minimum displacement to consider as moving
            
        Returns:
            Optional[int]: Index of moving atom, or None if not found
        """
        if len(initial_structure) != len(final_structure):
            logger.error("Structures have different numbers of atoms")
            return None
        
        max_displacement = 0.0
        moving_atom_idx = None
        
        for i in range(len(initial_structure)):
            init_pos = initial_structure[i].coords
            final_pos = final_structure[i].coords
            
            # Calculate displacement considering PBC
            displacement = self.get_pbc_distance(initial_structure, init_pos, final_pos)
            
            if displacement > max_displacement and displacement > threshold:
                max_displacement = displacement
                moving_atom_idx = i
        
        if moving_atom_idx is not None:
            logger.info(f"Moving atom: index {moving_atom_idx}, displacement: {max_displacement:.3f} Å")
        else:
            logger.warning(f"No atom displaced more than {threshold} Å")
        
        return moving_atom_idx
    
    def analyze_neb_convergence(self, neb_dir: str, n_images: int) -> dict:
        """
        Analyze convergence of NEB calculation from OUTCAR files.
        
        Args:
            neb_dir: NEB calculation directory
            n_images: Number of NEB images
            
        Returns:
            Dictionary with convergence information
        """
        convergence_info = {
            "converged": False,
            "forces": [],
            "max_force": None,
            "converged_images": 0
        }
        
        try:
            from pymatgen.io.vasp.outputs import Outcar
            
            max_forces = []
            converged_count = 0
            
            # Check each image
            for i in range(n_images + 2):
                img_dir = os.path.join(neb_dir, f"{i:02d}" if i < n_images+1 else f"{n_images+1:02d}")
                outcar_path = os.path.join(img_dir, "OUTCAR")
                
                if os.path.exists(outcar_path):
                    try:
                        outcar = Outcar(outcar_path)
                        if hasattr(outcar, 'final_energy'):
                            # Check for convergence in OUTCAR
                            with open(outcar_path, 'r') as f:
                                content = f.read()
                                if "reached required accuracy" in content:
                                    converged_count += 1
                            
                            # Get maximum force if available
                            if hasattr(outcar, 'forces') and outcar.forces:
                                max_force = np.max(np.linalg.norm(outcar.forces[-1], axis=1))
                                max_forces.append(max_force)
                                
                    except Exception as e:
                        logger.warning(f"Error reading OUTCAR for image {i}: {e}")
            
            convergence_info["forces"] = max_forces
            convergence_info["max_force"] = max(max_forces) if max_forces else None
            convergence_info["converged_images"] = converged_count
            convergence_info["converged"] = converged_count == (n_images + 2)
            
        except Exception as e:
            logger.error(f"Error analyzing convergence: {e}")
        
        return convergence_info
