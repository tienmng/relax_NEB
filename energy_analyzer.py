#!/usr/bin/env python
"""
Energy analysis utilities for NEB calculations.
Handles energy extraction from VASP output and plotting.
"""

import numpy as np
import matplotlib.pyplot as plt
import logging
import os
import shutil
import tempfile
from typing import List, Dict, Optional, Tuple
from pymatgen.io.vasp.outputs import Outcar

logger = logging.getLogger(__name__)

class NEBEnergyAnalyzer:
    """Analyzes energies from completed NEB calculations."""
    
    def __init__(self):
        """Initialize energy analyzer."""
        # Use non-interactive backend for matplotlib
        import matplotlib
        matplotlib.use('Agg')
    
    def extract_vasp_energies(self, neb_dir: str, n_images: int) -> Tuple[List[float], bool]:
        """
        Extract energies from VASP OUTCAR files.
        
        Args:
            neb_dir: NEB calculation directory
            n_images: Number of NEB images
            
        Returns:
            Tuple of (energies list, success boolean)
        """
        energies = []
        success = True
        
        # Read energies from each image
        for i in range(n_images + 2):  # +2 for initial and final
            img_dir = os.path.join(neb_dir, f"{i:02d}" if i < n_images+1 else f"{n_images+1:02d}")
            outcar_path = os.path.join(img_dir, "OUTCAR")
            
            if os.path.exists(outcar_path):
                try:
                    outcar = Outcar(outcar_path)
                    if hasattr(outcar, "final_energy"):
                        energies.append(outcar.final_energy)
                        logger.debug(f"Read energy from image {i}: {outcar.final_energy}")
                    else:
                        logger.warning(f"No final_energy found in {outcar_path}")
                        energies.append(None)
                        success = False
                except Exception as e:
                    logger.error(f"Error reading OUTCAR file {outcar_path}: {e}")
                    energies.append(None)
                    success = False
            else:
                logger.warning(f"OUTCAR file not found at {outcar_path}")
                energies.append(None)
                success = False
        
        return energies, success
    
    def extract_nebef_energies(self, neb_dir: str) -> Tuple[List[float], List[float], bool]:
        """
        Extract energies from NEBEF.dat file.
        
        Args:
            neb_dir: NEB calculation directory
            
        Returns:
            Tuple of (distances, energies, success)
        """
        nebef_path = os.path.join(neb_dir, "NEBEF.dat")
        
        if not os.path.exists(nebef_path):
            logger.warning(f"NEBEF.dat not found at {nebef_path}")
            return [], [], False
        
        try:
            # Read data from NEBEF.dat
            data = np.loadtxt(nebef_path)
            distances = data[:, 0]
            energies = data[:, 1]
            
            logger.info(f"Successfully read {len(energies)} energy points from NEBEF.dat")
            return distances.tolist(), energies.tolist(), True
            
        except Exception as e:
            logger.error(f"Error reading NEBEF.dat: {e}")
            return [], [], False
    
    def interpolate_missing_energies(self, energies: List[Optional[float]]) -> List[float]:
        """
        Interpolate missing energy values.
        
        Args:
            energies: List with some None values
            
        Returns:
            List with interpolated values
        """
        if not energies:
            return []
        
        # Find valid energy indices
        valid_indices = [i for i, e in enumerate(energies) if e is not None]
        
        if not valid_indices:
            logger.error("No valid energies found for interpolation")
            return []
        
        # Create a copy to work with
        interpolated = energies.copy()
        
        # Interpolate missing values
        for i in range(len(energies)):
            if energies[i] is None:
                # Find nearest valid energies before and after
                before_indices = [idx for idx in valid_indices if idx < i]
                after_indices = [idx for idx in valid_indices if idx > i]
                
                before_idx = max(before_indices) if before_indices else None
                after_idx = min(after_indices) if after_indices else None
                
                if before_idx is not None and after_idx is not None:
                    # Linear interpolation
                    weight = (i - before_idx) / (after_idx - before_idx)
                    interpolated[i] = energies[before_idx] * (1 - weight) + energies[after_idx] * weight
                    logger.info(f"Interpolated energy for image {i}: {interpolated[i]}")
                elif before_idx is not None:
                    # Use previous value
                    interpolated[i] = energies[before_idx]
                    logger.info(f"Used energy from image {before_idx} for image {i}")
                elif after_idx is not None:
                    # Use next value
                    interpolated[i] = energies[after_idx]
                    logger.info(f"Used energy from image {after_idx} for image {i}")
        
        return interpolated
    
    def calculate_reaction_path_distances(self, structures_dir: str, n_images: int,
                                        moving_atom_idx: int, file_manager=None) -> List[float]:
        """
        Calculate cumulative distances along reaction path from structures.
        
        Args:
            structures_dir: Directory containing POSCAR/CONTCAR files
            n_images: Number of NEB images
            moving_atom_idx: Index of moving atom
            file_manager: FileManager instance for cleaning CONTCAR files
            
        Returns:
            List of cumulative distances
        """
        from ase.io import read
        
        cumulative_distances = [0.0]
        configs = []
        
        # Read all structures
        for i in range(n_images + 2):
            img_dir = os.path.join(structures_dir, f"{i:02d}" if i < n_images+1 else f"{n_images+1:02d}")
            
            # Try CONTCAR first, then POSCAR
            structure_loaded = False
            
            for filename in ["CONTCAR", "POSCAR"]:
                structure_path = os.path.join(img_dir, filename)
                if os.path.exists(structure_path) and os.path.getsize(structure_path) > 0:
                    try:
                        if filename == "CONTCAR" and file_manager is not None:
                            # Clean CONTCAR before reading
                            logger.debug(f"Cleaning CONTCAR for image {i:02d}")
                            
                            # Create temporary cleaned file
                            temp_dir = tempfile.mkdtemp()
                            temp_contcar = os.path.join(temp_dir, "CONTCAR_clean")
                            shutil.copy(structure_path, temp_contcar)
                            
                            # Clean the temporary file
                            if file_manager.clean_contcar_elements(temp_contcar):
                                configs.append(read(temp_contcar))
                                structure_loaded = True
                                logger.debug(f"Successfully read cleaned CONTCAR for image {i:02d}")
                            else:
                                logger.warning(f"Failed to clean CONTCAR for image {i:02d}")
                            
                            # Clean up temporary file
                            try:
                                shutil.rmtree(temp_dir)
                            except:
                                pass
                        else:
                            # Read POSCAR directly or CONTCAR without cleaning
                            configs.append(read(structure_path))
                            structure_loaded = True
                            logger.debug(f"Successfully read {filename} for image {i:02d}")
                        
                        if structure_loaded:
                            break
                            
                    except Exception as e:
                        logger.warning(f"Error reading {structure_path}: {e}")
                        continue
            
            if not structure_loaded:
                logger.error(f"No valid structure found for image {i}")
                return []
        
        # Calculate cumulative distances
        for i in range(1, len(configs)):
            prev_pos = configs[i-1][moving_atom_idx].position
            curr_pos = configs[i][moving_atom_idx].position
            
            # Calculate minimum image vector
            cell = configs[i].get_cell()
            pbc = configs[i].get_pbc()
            vec = self._get_mic_vector(prev_pos, curr_pos, cell, pbc)
            distance = np.linalg.norm(vec)
            
            cumulative_distances.append(cumulative_distances[-1] + distance)
        
        logger.info(f"Calculated distances for {len(cumulative_distances)} images")
        return cumulative_distances
    
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
    
    def analyze_energy_profile(self, energies: List[float], distances: List[float] = None,
                             initial_energy: float = None) -> Dict:
        """
        Analyze energy profile and extract key quantities.
        
        Args:
            energies: List of energies
            distances: List of distances (optional)
            initial_energy: Reference energy for initial state
            
        Returns:
            Dictionary with analysis results
        """
        if not energies:
            return {}
        
        # Use first energy as reference if not provided
        if initial_energy is None:
            initial_energy = energies[0]
        
        # Calculate relative energies
        rel_energies = [e - initial_energy for e in energies]
        
        # Find barrier (maximum energy)
        max_energy = max(rel_energies)
        max_idx = rel_energies.index(max_energy)
        
        # Final state energy
        final_energy = rel_energies[-1]
        
        # Create analysis results
        results = {
            "barrier": max_energy,
            "barrier_image": max_idx,
            "reaction_energy": final_energy,
            "is_endothermic": final_energy > 0,
            "energies": energies,
            "relative_energies": rel_energies,
            "initial_energy": initial_energy
        }
        
        if distances:
            results["distances"] = distances
            results["barrier_distance"] = distances[max_idx] if max_idx < len(distances) else None
        
        logger.info(f"Energy analysis: barrier={max_energy:.3f} eV, "
                   f"reaction energy={final_energy:.3f} eV")
        
        return results
    
    def plot_energy_profile(self, energies: List[float], distances: List[float] = None,
                           title: str = "NEB Energy Profile", output_path: str = None,
                           initial_energy: float = None) -> str:
        """
        Create energy profile plot.
        
        Args:
            energies: List of energies
            distances: List of distances (optional, uses image indices if not provided)
            title: Plot title
            output_path: Output file path
            initial_energy: Reference energy for initial state
            
        Returns:
            Path to saved plot
        """
        if not energies:
            logger.error("No energies provided for plotting")
            return None
        
        # Prepare x-axis
        if distances is None:
            x_values = list(range(len(energies)))
            x_label = "Image Number"
        else:
            x_values = distances
            x_label = "Distance along reaction path (Å)"
        
        # Calculate relative energies
        if initial_energy is None:
            initial_energy = energies[0]
        rel_energies = [e - initial_energy for e in energies]
        
        # Find important points
        max_energy = max(rel_energies)
        max_idx = rel_energies.index(max_energy)
        final_energy = rel_energies[-1]
        
        # Create plot
        plt.figure(figsize=(10, 6))
        plt.plot(x_values, rel_energies, 'o-', linewidth=2, markersize=8)
        
        # Mark important points
        plt.plot(x_values[0], rel_energies[0], 'o', color='green', 
                markersize=12, label='Initial state')
        plt.plot(x_values[-1], rel_energies[-1], 'o', color='red', 
                markersize=12, label='Final state')
        
        if max_idx > 0 and max_idx < len(x_values) - 1:
            plt.plot(x_values[max_idx], rel_energies[max_idx], 'o', 
                    color='purple', markersize=12, label='Transition state')
        
        # Add annotations
        if len(x_values) > 1:
           # plt.annotate(f'Barrier: {max_energy:.2f} eV',
            #            xy=(x_values[max_idx], rel_energies[max_idx]),
             #           xytext=(x_values[max_idx], rel_energies[max_idx] + abs(max_energy)/10),
              #          ha='center', va='bottom', fontsize=12,
               #         arrowprops=dict(arrowstyle='->', color='purple'))
            
            plt.annotate(f'ΔE: {final_energy:.2f} eV',
                        xy=(x_values[-1], rel_energies[-1]),
                        xytext=(x_values[-1] * 0.8, rel_energies[-1] + abs(max_energy)/10),
                        ha='center', va='bottom', fontsize=12,
                        arrowprops=dict(arrowstyle='->', color='red'))
        
        # Format plot
        plt.xlabel(x_label, fontsize=14)
        plt.ylabel('Energy relative to initial state (eV)', fontsize=14)
        plt.title(title, fontsize=16)
        plt.grid(True, alpha=0.7, linestyle='--')
        plt.legend(loc='best', fontsize=12)
        
        # Set reasonable y-axis limits
        y_range = max(rel_energies) - min(rel_energies)
        y_margin = 0.1 * y_range if y_range > 0 else 0.1
        plt.ylim(min(rel_energies) - y_margin, max(rel_energies) + y_margin)
        
        # Save plot
        if output_path is None:
            output_path = "neb_energy_profile.png"
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Energy profile plot saved to {output_path}")
        return output_path
    
    def save_energy_data(self, energies: List[float], distances: List[float] = None,
                        output_path: str = None, initial_energy: float = None) -> str:
        """
        Save energy data to text file.
        
        Args:
            energies: List of energies
            distances: List of distances (optional)
            output_path: Output file path
            initial_energy: Reference energy for initial state
            
        Returns:
            Path to saved data file
        """
        if output_path is None:
            output_path = "neb_energy_profile.dat"
        
        # Calculate relative energies
        if initial_energy is None:
            initial_energy = energies[0]
        rel_energies = [e - initial_energy for e in energies]
        
        # Write data
        with open(output_path, 'w') as f:
            if distances:
                f.write("# Distance(Å)  Energy(eV)  RelEnergy(eV)\n")
                for i in range(len(energies)):
                    dist = distances[i] if i < len(distances) else 0.0
                    f.write(f"{dist:.6f}  {energies[i]:.6f}  {rel_energies[i]:.6f}\n")
            else:
                f.write("# Image  Energy(eV)  RelEnergy(eV)\n")
                for i in range(len(energies)):
                    f.write(f"{i}  {energies[i]:.6f}  {rel_energies[i]:.6f}\n")
        
        logger.info(f"Energy data saved to {output_path}")
        return output_path
    
    def compare_ase_vasp_profiles(self, ase_results: Dict, vasp_results: Dict,
                                 output_dir: str = ".") -> str:
        """
        Create comparison plot between ASE and VASP energy profiles.
        
        Args:
            ase_results: Results from ASE analysis
            vasp_results: Results from VASP analysis
            output_dir: Output directory
            
        Returns:
            Path to comparison plot
        """
        plt.figure(figsize=(12, 8))
        
        # Plot ASE results
        if "distances" in ase_results and "relative_energies" in ase_results:
            plt.subplot(2, 1, 1)
            plt.plot(ase_results["distances"], ase_results["relative_energies"], 
                    'o-', linewidth=2, label='ASE (LJ potential)', color='blue')
            plt.xlabel('Distance along path (Å)')
            plt.ylabel('Energy (eV)')
            plt.title('ASE Energy Profile')
            plt.grid(True, alpha=0.7)
            plt.legend()
        
        # Plot VASP results
        if "distances" in vasp_results and "relative_energies" in vasp_results:
            plt.subplot(2, 1, 2)
            plt.plot(vasp_results["distances"], vasp_results["relative_energies"], 
                    'o-', linewidth=2, label='VASP (DFT)', color='red')
            plt.xlabel('Distance along path (Å)')
            plt.ylabel('Energy (eV)')
            plt.title('VASP Energy Profile')
            plt.grid(True, alpha=0.7)
            plt.legend()
        
        plt.tight_layout()
        
        # Save comparison plot
        output_path = os.path.join(output_dir, "ase_vasp_comparison.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"ASE-VASP comparison plot saved to {output_path}")
        return output_path
