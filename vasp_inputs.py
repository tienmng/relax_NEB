#!/usr/bin/env python
"""
VASP input generation utilities for NEB calculations.
Handles INCAR, KPOINTS, and magnetic moment settings.
"""

import numpy as np
import logging
from typing import Dict, List, Optional
from pymatgen.core import Structure
from pymatgen.io.vasp.inputs import Incar, Kpoints

logger = logging.getLogger(__name__)

class VASPInputGenerator:
    """Generates VASP input files for NEB calculations."""
    
    def __init__(self):
        """Initialize VASP input generator."""
        pass
    
    def create_neb_incar(self, structure: Structure, n_images: int = 5,
                        ldau_settings: Optional[Dict] = None,
                        ediffg: float = -0.01, nsw: int = 200) -> Incar:
        """
        Create INCAR file for NEB calculation.
        
        Args:
            structure: Structure for magnetic moment settings
            n_images: Number of NEB images
            ldau_settings: LDAU parameters
            ediffg: Force convergence criteria
            nsw: Maximum ionic steps
            
        Returns:
            Incar object
        """
        incar = Incar()
        
        # NEB-specific settings
        incar['IMAGES'] = n_images
        incar['LCLIMB'] = True       # Use climbing image NEB
        incar['ICHAIN'] = 0
        incar['SPRING'] = -5
        
        # Optimizer settings
        incar['IBRION'] = 3          
        incar['POTIM'] = 0
        incar['IOPT'] = 0
        incar['EDIFFG'] = ediffg     # Force convergence criteria 
        incar['NSW'] = nsw           # Maximum ionic steps
        
        # Basic DFT settings
        incar['ISIF'] = 2
        incar['EDIFF'] = 1E-6        # Energy convergence criteria
        incar['NELM'] = 199          # Maximum electronic steps per ionic step
        incar['ALGO'] = 'Fast'       # Electronic minimization algorithm
        incar['PREC'] = 'Normal'     # Precision
        incar['LSCALAPACK'] = False  # Turn off ScaLAPACK for better performance with many cores
        incar['NCORE'] = 16
        incar['ISMEAR'] = 0
        incar['SIGMA'] = 0.05
        incar['LASPH'] = True
        incar['LREAL'] = 'Auto'
        incar['LMAXMIX'] = 6
        incar['ENCUT'] = 600
        
        # Apply LDAU settings if provided
        if ldau_settings:
            elements = [str(site.specie) for site in structure]
            unique_elements = list(dict.fromkeys(elements))  # Get unique elements while preserving order
            incar = self._apply_ldau_settings(incar, unique_elements, ldau_settings)
        
        # Apply magnetic moments
        incar = self._apply_magmom(structure, incar)
        
        return incar
    
    def create_relax_incar(self, structure: Structure, max_iterations: int = 200,
                          ediffg: float = -0.01, ldau_settings: Optional[Dict] = None,
                          restart: bool = False) -> Incar:
        """
        Create INCAR file for structure relaxation.
        
        Args:
            structure: Structure for magnetic moment settings
            max_iterations: Maximum number of ionic steps
            ediffg: Force convergence criteria
            ldau_settings: LDAU parameters
            restart: Whether this is a restart calculation
            
        Returns:
            Incar object
        """
        incar = Incar()
        
        # Relaxation settings
        incar['IBRION'] = 2          # Use conjugate gradient algorithm
        incar['NSW'] = max_iterations # Maximum ionic steps
        incar['ISIF'] = 2            # Relax ions only, fixed cell
        incar['EDIFFG'] = ediffg     # Force convergence criteria
        incar['EDIFF'] = 1E-6        # Energy convergence criteria
        incar['NELM'] = 199          # Maximum electronic steps
        incar['ALGO'] = 'Fast'       # Electronic minimization algorithm
        incar['PREC'] = 'Accurate'   # Precision
        incar['ISMEAR'] = 0
        incar['SIGMA'] = 0.05
        incar['LMAXMIX'] = 6
        incar['LASPH'] = True
        incar['ENCUT'] = 600
        incar['NCORE'] = 32
        
        # Apply LDAU settings if provided
        if ldau_settings:
            elements = [str(site.specie) for site in structure]
            unique_elements = list(dict.fromkeys(elements))  # Get unique elements while preserving order
            incar = self._apply_ldau_settings(incar, unique_elements, ldau_settings)
        
        # Apply magnetic moments
        incar = self._apply_magmom(structure, incar)
        
        # If this is a restart, add restart flags
        if restart:
            incar['ISTART'] = 1      # Read WAVECAR if exists
            incar['ICHARG'] = 1      # Read CHGCAR if exists
            logger.info("Added restart flags to INCAR")
        
        return incar
    
    def _apply_magmom(self, structure: Structure, incar: Incar) -> Incar:
        """
        Apply magnetic moment settings to INCAR.
        
        Args:
            structure: Crystal structure
            incar: INCAR object to modify
            
        Returns:
            Modified INCAR object
        """
        # Initialize the MAGMOM list
        magmom = []
        
        # Element-specific magnetic moments
        mag_values = {
    # Transition metals
    'Ti': 0.6, 'V': 5.0, 'Cr': 5.0, 'Mn': 4.0,
    'Fe': 5.0, 'Co': 3.5, 'Ni': 5.0, 'Cu': 0.6,
    # Rare earth
    'La': 0.6, 'Ce': 0.6, 'Nd': 0.6,
    # Main group
    'O': 0.6, 'N': 0.6, 'C': 0.6,
    # Alkali/Alkaline earth
    'Li': 0.0, 'Na': 0.0, 'K': 0.0,
    'Mg': 0.0, 'Ca': 0.0, 'Sr': 0.0,
    }

        
        # Iterate through sites in the structure
        for site in structure:
            element = site.specie.symbol
            # Use default value of 0.0 if element not in mag_values
            moment = mag_values.get(element, 0.0)
            magmom.append(moment)
        # Check if any magnetic moments are non-zero
        if any(abs(m) > 0.001 for m in magmom):
            incar['ISPIN'] = 2    # Enable spin polarization
            incar['MAGMOM'] = magmom   # Set magnetic moments
        else:
            incar['ISPIN'] = 1    # Disable spin polarization
        
        return incar
    
    def _apply_ldau_settings(self, incar: Incar, elements: List[str], 
                           ldau_settings: Dict) -> Incar:
        """
        Apply LDAU settings to INCAR.
        
        Args:
            incar: INCAR object to modify
            elements: List of unique elements
            ldau_settings: Dictionary with LDAU parameters
            
        Returns:
            Modified INCAR object
        """
        if not ldau_settings or not ldau_settings.get('LDAU', False):
            return incar
        
        # Set LDAU = .TRUE.
        incar['LDAU'] = True
        incar['LDAUTYPE'] = 2  # Standard LDAU+J type
        
        # Create LDAUL, LDAUU, and LDAUJ lists
        ldaul = []
        ldauu = []
        ldauj = []
        
        for element in elements:
            if element in ldau_settings:
                ldaul.append(ldau_settings[element]["L"])
                ldauu.append(ldau_settings[element]["U"])
                ldauj.append(ldau_settings[element].get("J", 0.0))  # Default J value
            else:
                ldaul.append(0)
                ldauu.append(0)
                ldauj.append(0)
        
        incar['LDAUL'] = ldaul
        incar['LDAUU'] = ldauu
        incar['LDAUJ'] = ldauj
        incar['LMAXMIX'] = 6  # Good for f-orbitals
        
        return incar
    
    def generate_kpoints(self, structure: Structure, kspacing: float = 0.3) -> Kpoints:
        """
        Generate k-points mesh from k-point spacing.
        
        Args:
            structure: Crystal structure
            kspacing: K-point spacing in 1/Angstrom
            
        Returns:
            Kpoints object
        """
        # Get reciprocal lattice vectors
        recip_lattice = structure.lattice.reciprocal_lattice
        
        # Calculate number of k-points along each reciprocal lattice vector
        kpts = [max(1, int(np.ceil(np.linalg.norm(v) / kspacing))) for v in recip_lattice.matrix]
        
        # Create Kpoints object
        kpoints = Kpoints(
            comment=f"Automatic mesh with kspacing={kspacing}",
            style="Gamma",
            num_kpts=0,
            kpts=[kpts],
            kpts_shift=(0, 0, 0)
        )
        
        logger.info(f"Generated k-point mesh: {kpts[0]}x{kpts[1]}x{kpts[2]} with spacing {kspacing}")
        
        return kpoints
    
    def modify_incar_for_restart(self, incar: Incar, restart_count: int = 1) -> Incar:
        """
        Modify INCAR for restart calculations.
        
        Args:
            incar: Original INCAR object
            restart_count: Number of previous restarts
            
        Returns:
            Modified INCAR object
        """
        # Set restart flags
        incar['ISTART'] = 1  # Read WAVECAR if exists
        incar['ICHARG'] = 1  # Read CHGCAR if exists
        
        # Increase NSW for restarts to ensure enough steps
        if 'NSW' in incar:
            current_nsw = incar['NSW']
            if restart_count <= 2:  # Only increase NSW on the first two restarts
                incar['NSW'] = int(current_nsw * 1.5)
                logger.info(f"Increased NSW from {current_nsw} to {incar['NSW']}")
            elif restart_count > 2 and current_nsw < 500:  # Cap at a reasonable maximum
                incar['NSW'] = 500
                logger.info(f"Set NSW to maximum of 500 after multiple restarts")
        
        return incar
