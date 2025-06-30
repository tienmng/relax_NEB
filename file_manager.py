#!/usr/bin/env python
"""
File management utilities for NEB calculations.
Handles POTCAR creation, CONTCAR cleaning, and file operations.
"""

import os
import warnings
import logging
from typing import Dict, List, Optional, Tuple
import fnmatch

# Suppress encoding warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
os.environ['PYTHONIOENCODING'] = 'utf-8'

logger = logging.getLogger(__name__)

class FileManager:
    """Handles file operations for NEB calculations."""
    
    def __init__(self, potcar_path: Optional[str] = None, 
                 potcar_mapping: Optional[Dict[str, str]] = None):
        """
        Initialize file manager.
        
        Args:
            potcar_path: Path to VASP pseudopotentials directory
            potcar_mapping: Dictionary mapping elements to POTCAR variants
        """
        self.potcar_path = potcar_path
        self.potcar_mapping = potcar_mapping or {}
        
        # Set environment variable for POTCAR
        if potcar_path:
            os.environ["VASP_PSP_DIR"] = potcar_path
            logger.info(f"Set VASP_PSP_DIR to {potcar_path}")
    
    def clean_contcar_elements(self, contcar_path: str) -> bool:
        """
        Clean up CONTCAR file by removing POTCAR-style element descriptors.
        Removes both slash paths (/potpaw_PBE_54/...) and underscore variants (_s, _pv, etc.)
        
        Args:
            contcar_path: Path to CONTCAR file to clean
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            with open(contcar_path, 'r') as f:
                lines = f.readlines()
            
            # Check if we have at least 6 lines (minimum for a valid POSCAR/CONTCAR)
            if len(lines) < 6:
                logger.error(f"CONTCAR file {contcar_path} has too few lines")
                return False
            
            # Element line is typically line 5 (index 4) or 6 (index 5)
            element_line_idx = None
            
            # Check line 5 (index 4) - common location
            if len(lines) > 4 and ('/' in lines[4] or '_' in lines[4]):
                element_line_idx = 4
            # Check line 6 (index 5) - alternative location
            elif len(lines) > 5 and ('/' in lines[5] or '_' in lines[5]):
                element_line_idx = 5
            else:
                # No POTCAR-style elements found, file might be fine as is
                return True
            
            # Clean the element line
            element_line = lines[element_line_idx]
            elements = element_line.strip().split()
            
            # Clean each element
            cleaned_elements = []
            for elem in elements:
                cleaned_elem = elem
                
                # First, remove everything after '/' (full POTCAR paths)
                if '/' in cleaned_elem:
                    cleaned_elem = cleaned_elem.split('/')[0]
                
                # Then, remove everything after '_' (underscore variants)
                if '_' in cleaned_elem:
                    cleaned_elem = cleaned_elem.split('_')[0]
                    
                cleaned_elements.append(cleaned_elem)
            
            # Replace the line with cleaned elements
            lines[element_line_idx] = '  '.join(cleaned_elements) + '\n'
            
            # Write back to file
            with open(contcar_path, 'w') as f:
                f.writelines(lines)
                
            logger.info(f"Cleaned CONTCAR elements: {' '.join(cleaned_elements)}")
            return True
            
        except Exception as e:
            logger.error(f"Error cleaning CONTCAR file {contcar_path}: {e}")
            return False
    
    def create_potcar(self, calc_dir: str, elements: List[str]) -> bool:
        """
        Create POTCAR file for the calculation.
        
        Args:
            calc_dir: Directory where POTCAR should be created
            elements: List of element symbols
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.potcar_path or not self.potcar_mapping:
            logger.warning("POTCAR path or mapping not provided")
            return False
            
        try:
            with open(os.path.join(calc_dir, "POTCAR"), 'w') as potcar_file:
                for element in elements:
                    potcar_symbol = self.potcar_mapping.get(element, element)
                    potcar_path_for_element = os.path.join(self.potcar_path, potcar_symbol, "POTCAR")
                    
                    if not os.path.exists(potcar_path_for_element):
                        raise FileNotFoundError(f"POTCAR not found for {potcar_symbol} at {potcar_path_for_element}")
                    
                    with open(potcar_path_for_element, 'r') as element_potcar:
                        potcar_file.write(element_potcar.read())
            
            logger.info(f"Created POTCAR for elements: {elements}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating POTCAR: {e}")
            return False
    
    def cleanup_calculation(self, calc_dir: str, keep_essential: bool = True, 
                          cleanup_wave: bool = True, cleanup_chg: bool = True) -> Tuple[int, List[str]]:
        """
        Clean up unnecessary files after calculation completes.
        
        Args:
            calc_dir: Directory containing the calculation
            keep_essential: Keep essential files (POSCAR, CONTCAR, OUTCAR, etc.)
            cleanup_wave: Remove WAVECAR files
            cleanup_chg: Remove CHG/CHGCAR files
            
        Returns:
            Tuple of (total_size_saved, files_removed)
        """
        # Define file patterns to remove
        cleanup_patterns = []
        
        # Large files that are often not needed
        if cleanup_wave:
            cleanup_patterns.extend(['WAVECAR', 'WAVECAR.*'])
        if cleanup_chg:
            cleanup_patterns.extend(['CHG', 'CHGCAR', 'CHG.*', 'CHGCAR.*'])
        
        # Other large files
        cleanup_patterns.extend([
            'PROCAR',      # Can be large for band structure calculations
            'LOCPOT',      # Large, only needed for specific analyses
            'ELFCAR',      # Electron localization function
            'vasprun.xml', # Can be large, but might want to keep for pymatgen
            'EIGENVAL',    # Usually not needed if vasprun.xml is kept
            'DOSCAR',      # Can be regenerated if needed
            'PCDAT',       # Pair correlation function
            'XDATCAR',     # Trajectory file, large for MD/NEB
        ])
        
        # Essential files to always keep
        essential_files = [
            'INCAR',
            'POSCAR',
            'CONTCAR',
            'KPOINTS',
            'POTCAR',
            'OUTCAR',
            'OSZICAR',
            'energy',      # If you write energy to file
            '*.log',       # Log files
            'job.sh',      # Job script
            'vasp.out',    # VASP output
            'slurm-*.out', # SLURM output files
        ]
        
        # For NEB calculations, also keep
        neb_essential = [
            'NEBEF.dat',   # NEB energy profile
            'neb*.dat',    # Other NEB data files
        ]
        
        # Count space saved
        total_size_saved = 0
        files_removed = []
        
        # Function to check if file matches essential patterns
        def is_essential(filename):
            if not keep_essential:
                return False
            for pattern in essential_files + neb_essential:
                if fnmatch.fnmatch(filename, pattern):
                    return True
            return False
        
        # Walk through directory
        for root, dirs, files in os.walk(calc_dir):
            for file in files:
                filepath = os.path.join(root, file)
                
                # Skip if it's an essential file
                if is_essential(file):
                    continue
                
                # Check if file matches cleanup patterns
                for pattern in cleanup_patterns:
                    if fnmatch.fnmatch(file, pattern):
                        try:
                            # Get file size before deletion
                            file_size = os.path.getsize(filepath)
                            # Remove file
                            os.remove(filepath)
                            total_size_saved += file_size
                            files_removed.append(filepath)
                            logger.debug(f"Removed {filepath} ({file_size/1024/1024:.1f} MB)")
                        except Exception as e:
                            logger.warning(f"Could not remove {filepath}: {e}")
                        break
        
        logger.info(f"Cleanup complete: removed {len(files_removed)} files, saved {total_size_saved/1024/1024/1024:.2f} GB")
        return total_size_saved, files_removed
    
    def backup_files(self, src_dir: str, backup_dir: str, 
                    file_patterns: List[str] = None) -> bool:
        """
        Backup specific files from source to backup directory.
        
        Args:
            src_dir: Source directory
            backup_dir: Backup directory
            file_patterns: List of file patterns to backup (default: important files)
            
        Returns:
            bool: True if successful
        """
        if file_patterns is None:
            file_patterns = ["INCAR", "KPOINTS", "POTCAR", "POSCAR", "CONTCAR", 
                           "OUTCAR", "OSZICAR", "vasprun.xml", "NEBEF.dat", "job.sh"]
        
        try:
            os.makedirs(backup_dir, exist_ok=True)
            
            for pattern in file_patterns:
                src_file = os.path.join(src_dir, pattern)
                if os.path.exists(src_file):
                    import shutil
                    shutil.copy2(src_file, os.path.join(backup_dir, pattern))
                    logger.debug(f"Backed up {pattern}")
            
            logger.info(f"Backup complete to {backup_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Error during backup: {e}")
            return False
