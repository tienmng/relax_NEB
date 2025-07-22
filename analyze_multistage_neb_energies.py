#!/usr/bin/env python
"""
Analyze energies along migration distance for oxygen in multistage NEB calculations.
Works with incomplete calculations and extracts whatever data is available.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from pymatgen.core import Structure
from pymatgen.io.vasp.outputs import Outcar, Vasprun
from pymatgen.io.vasp.inputs import Poscar
import argparse
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class NEBEnergyExtractor:
    """Extract energies and distances from NEB calculations."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def get_pbc_distance(self, structure, pos1, pos2):
        """Calculate minimum distance considering periodic boundary conditions."""
        lattice = structure.lattice
        
        # Convert positions to fractional coordinates
        frac1 = lattice.get_fractional_coords(pos1)
        frac2 = lattice.get_fractional_coords(pos2)
        
        # Check all periodic images
        min_dist = float('inf')
        for i in [-1, 0, 1]:
            for j in [-1, 0, 1]:
                for k in [-1, 0, 1]:
                    # Shift frac2 by the periodic image
                    shifted_frac2 = frac2 + np.array([i, j, k])
                    # Convert back to Cartesian
                    shifted_pos2 = lattice.get_cartesian_coords(shifted_frac2)
                    # Calculate distance
                    dist = np.linalg.norm(pos1 - shifted_pos2)
                    min_dist = min(min_dist, dist)
        
        return min_dist
    
    def extract_energy_from_outcar(self, outcar_path):
        """Extract energy from OUTCAR file."""
        if not os.path.exists(outcar_path):
            return None
        
        try:
            # Try to use pymatgen first
            outcar = Outcar(outcar_path)
            if hasattr(outcar, 'final_energy') and outcar.final_energy is not None:
                return outcar.final_energy
            
            # Fallback: parse OUTCAR manually for last energy
            with open(outcar_path, 'r') as f:
                lines = f.readlines()
            
            # Search for energy patterns from the end
            for line in reversed(lines):
                if "energy  without entropy=" in line:
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if part == "energy(sigma->0)" and i+1 < len(parts):
                            try:
                                return float(parts[i+1])
                            except ValueError:
                                continue
                elif "free  energy   TOTEN" in line:
                    parts = line.split()
                    if len(parts) >= 5:
                        try:
                            return float(parts[4])
                        except ValueError:
                            continue
            
        except Exception as e:
            self.logger.warning(f"Error reading {outcar_path}: {e}")
        
        return None
    
    def extract_structure_from_poscar(self, poscar_path):
        """Extract structure from POSCAR file."""
        if not os.path.exists(poscar_path):
            return None
        
        try:
            # Clean POSCAR if needed (remove element line if corrupted)
            temp_poscar = poscar_path + ".tmp"
            with open(poscar_path, 'r') as f:
                lines = f.readlines()
            
            # Check if line 5 looks like element names
            if len(lines) > 5:
                try:
                    # If line 5 can be parsed as integers, it's already clean
                    [int(x) for x in lines[5].split()]
                except ValueError:
                    # Line 5 has element names, remove it
                    lines = lines[:5] + lines[6:]
                    with open(temp_poscar, 'w') as f:
                        f.writelines(lines)
                    poscar_path = temp_poscar
            
            structure = Poscar.from_file(poscar_path).structure
            
            # Clean up temp file if created
            if os.path.exists(temp_poscar):
                os.remove(temp_poscar)
            
            return structure
            
        except Exception as e:
            self.logger.warning(f"Error reading {poscar_path}: {e}")
            return None
    
    def extract_image_data(self, image_dir, moving_o_idx, reference_structure=None):
        """Extract energy and structure data from a single NEB image directory."""
        result = {
            'energy': None,
            'structure': None,
            'moving_o_position': None,
            'distance_from_ref': None
        }
        
        # Try to get energy
        outcar_path = os.path.join(image_dir, "OUTCAR")
        result['energy'] = self.extract_energy_from_outcar(outcar_path)
        
        # Try to get structure (prefer CONTCAR, fallback to POSCAR)
        for struct_file in ["CONTCAR", "POSCAR"]:
            struct_path = os.path.join(image_dir, struct_file)
            structure = self.extract_structure_from_poscar(struct_path)
            if structure is not None:
                result['structure'] = structure
                
                # Get moving oxygen position
                if moving_o_idx < len(structure):
                    result['moving_o_position'] = structure[moving_o_idx].coords
                    
                    # Calculate distance from reference if provided
                    if reference_structure is not None and moving_o_idx < len(reference_structure):
                        ref_pos = reference_structure[moving_o_idx].coords
                        result['distance_from_ref'] = self.get_pbc_distance(
                            structure, ref_pos, result['moving_o_position']
                        )
                break
        
        return result
    
    def extract_neb_stage_data(self, stage_dir, n_images, moving_o_idx):
        """Extract data from a single NEB stage."""
        stage_data = {
            'images': {},
            'has_data': False
        }
        
        # Get initial structure as reference
        initial_struct = None
        init_dir = os.path.join(stage_dir, "00")
        if os.path.exists(init_dir):
            init_data = self.extract_image_data(init_dir, moving_o_idx)
            if init_data['structure'] is not None:
                initial_struct = init_data['structure']
        
        # Extract data for all images (including endpoints)
        for i in range(n_images + 2):
            image_dir = os.path.join(stage_dir, f"{i:02d}")
            if os.path.exists(image_dir):
                self.logger.info(f"  Extracting image {i:02d}...")
                image_data = self.extract_image_data(image_dir, moving_o_idx, initial_struct)
                
                if image_data['energy'] is not None or image_data['structure'] is not None:
                    stage_data['images'][i] = image_data
                    stage_data['has_data'] = True
        
        return stage_data
    
    def extract_relaxation_energy(self, relax_dir):
        """Extract energy from a relaxation calculation."""
        if not os.path.exists(relax_dir):
            return None
        
        # Direct calculation in relax_dir
        outcar_path = os.path.join(relax_dir, "OUTCAR")
        energy = self.extract_energy_from_outcar(outcar_path)
        
        if energy is not None:
            return energy
        
        # Check for staged relaxations
        stage_dirs = [d for d in os.listdir(relax_dir) if d.startswith('stage') and os.path.isdir(os.path.join(relax_dir, d))]
        if stage_dirs:
            # Get the highest numbered stage
            stage_numbers = []
            for stage_dir in stage_dirs:
                try:
                    stage_num = int(stage_dir.replace('stage', ''))
                    stage_numbers.append(stage_num)
                except:
                    continue
            
            if stage_numbers:
                final_stage = f"stage{max(stage_numbers)}"
                final_stage_path = os.path.join(relax_dir, final_stage)
                outcar_path = os.path.join(final_stage_path, "OUTCAR")
                return self.extract_energy_from_outcar(outcar_path)
        
        return None

def save_stage_data(stage_data, stage_name, output_dir):
    """Save energy profile data for a single stage."""
    # Create analysis directory for this stage
    # Extract stage number from stage_name (e.g., 'neb_stage1' -> 'stage1')
    stage_num = stage_name.replace('neb_', '')  # Remove 'neb_' prefix
    analysis_dir = os.path.join(output_dir, f"analysis_{stage_num}")
    os.makedirs(analysis_dir, exist_ok=True)
    
    # Prepare data for saving
    data_lines = ["# Image  Distance(Å)  Energy(eV)  Energy-E0(eV)\n"]
    
    # Get reference energy (first image)
    ref_energy = None
    if 0 in stage_data['images'] and stage_data['images'][0]['energy'] is not None:
        ref_energy = stage_data['images'][0]['energy']
    
    # Collect and sort data
    for img_idx in sorted(stage_data['images'].keys()):
        img_data = stage_data['images'][img_idx]
        
        distance = img_data['distance_from_ref'] if img_data['distance_from_ref'] is not None else 0.0
        energy = img_data['energy']
        
        if energy is not None:
            rel_energy = energy - ref_energy if ref_energy is not None else 0.0
            data_lines.append(f"{img_idx:3d}  {distance:12.6f}  {energy:15.8f}  {rel_energy:12.6f}\n")
    
    # Save data file (use stage_num for filename)
    dat_file = os.path.join(analysis_dir, f"{stage_num}_energy_profile.dat")
    with open(dat_file, 'w') as f:
        f.writelines(data_lines)
    
    # Create individual plot for this stage (use stage_num for filename)
    plot_file = os.path.join(analysis_dir, f"{stage_num}_energy_profile.png")
    plot_single_stage(stage_data, stage_name, plot_file, ref_energy)
    
    return dat_file, plot_file

def plot_single_stage(stage_data, stage_name, output_file, ref_energy=None):
    """Create plot for a single NEB stage."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Collect data
    distances = []
    energies = []
    rel_energies = []
    image_indices = []
    
    for img_idx, img_data in sorted(stage_data['images'].items()):
        if img_data['energy'] is not None:
            image_indices.append(img_idx)
            energies.append(img_data['energy'])
            
            if ref_energy is not None:
                rel_energies.append(img_data['energy'] - ref_energy)
            else:
                rel_energies.append(img_data['energy'])
            
            if img_data['distance_from_ref'] is not None:
                distances.append(img_data['distance_from_ref'])
            else:
                distances.append(0.0)
    
    # Plot relative energy vs image index
    ax1.plot(image_indices, rel_energies, 'bo-', linewidth=2, markersize=8)
    ax1.set_xlabel('Image Index')
    ax1.set_ylabel('Relative Energy (eV)')
    ax1.set_title(f'{stage_name} - Energy Profile by Image')
    ax1.grid(True, alpha=0.3)
    
    # Add barrier annotation
    if rel_energies:
        max_rel_e = max(rel_energies)
        max_idx = image_indices[rel_energies.index(max_rel_e)]
        ax1.annotate(f'Barrier: {max_rel_e:.3f} eV', 
                    xy=(max_idx, max_rel_e),
                    xytext=(max_idx, max_rel_e + 0.1),
                    ha='center',
                    arrowprops=dict(arrowstyle='->', color='red'))
    
    # Plot relative energy vs distance
    if distances and rel_energies:
        ax2.plot(distances, rel_energies, 'ro-', linewidth=2, markersize=8)
        ax2.set_xlabel('Migration Distance (Å)')
        ax2.set_ylabel('Relative Energy (eV)')
        ax2.set_title(f'{stage_name} - Energy Profile by Distance')
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

def plot_neb_energies(all_stage_data, output_file="neb_energy_analysis.png"):
    """Plot energies vs migration distance for all stages."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(all_stage_data)))
    
    for idx, (stage_name, stage_data) in enumerate(all_stage_data.items()):
        if not stage_data['has_data']:
            continue
        
        # Collect data points
        distances = []
        energies = []
        image_indices = []
        
        for img_idx, img_data in sorted(stage_data['images'].items()):
            if img_data['distance_from_ref'] is not None:
                distances.append(img_data['distance_from_ref'])
            else:
                distances.append(np.nan)
            
            if img_data['energy'] is not None:
                energies.append(img_data['energy'])
            else:
                energies.append(np.nan)
            
            image_indices.append(img_idx)
        
        # Get reference energy for relative plotting
        ref_energy = None
        if 0 in stage_data['images'] and stage_data['images'][0]['energy'] is not None:
            ref_energy = stage_data['images'][0]['energy']
        
        # Plot relative energy vs image index
        valid_e = [(i, e) for i, e in zip(image_indices, energies) if not np.isnan(e)]
        if valid_e and ref_energy is not None:
            indices, energies_valid = zip(*valid_e)
            rel_energies_valid = [e - ref_energy for e in energies_valid]
            ax1.plot(indices, rel_energies_valid, 'o-', label=stage_name, color=colors[idx])
        
        # Plot relative energy vs migration distance
        valid_d = [(d, e) for d, e in zip(distances, energies) if not np.isnan(d) and not np.isnan(e)]
        if valid_d and ref_energy is not None:
            dists, energies_valid = zip(*valid_d)
            rel_energies_valid = [e - ref_energy for e in energies_valid]
            ax2.plot(dists, rel_energies_valid, 'o-', label=stage_name, color=colors[idx])
    
    ax1.set_xlabel('Image Index')
    ax1.set_ylabel('Relative Energy (eV)')
    ax1.set_title('NEB Energy Profile by Image (All Stages)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.set_xlabel('Migration Distance (Å)')
    ax2.set_ylabel('Relative Energy (eV)')
    ax2.set_title('NEB Energy Profile by Distance (All Stages)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_file

def save_final_summary(all_stage_data, output_dir, final_stage='neb_stage4'):
    """Save final energy profile data combining all stages or using the final stage."""
    # Determine which data to use for final profile
    if final_stage in all_stage_data and all_stage_data[final_stage]['has_data']:
        # Use the specified final stage data
        final_data = all_stage_data[final_stage]
        source = final_stage
    else:
        # Use the last available stage
        available_stages = [s for s in all_stage_data if all_stage_data[s]['has_data']]
        if not available_stages:
            return None, None
        
        # Sort stages by number
        sorted_stages = sorted(available_stages, 
                             key=lambda x: int(x.replace('neb_stage', '')) if 'neb_stage' in x else 0)
        final_stage = sorted_stages[-1]
        final_data = all_stage_data[final_stage]
        source = final_stage
    
    # Save final data
    data_lines = [f"# Final NEB Energy Profile (from {source})\n"]
    data_lines.append("# Image  Distance(Å)  Energy(eV)  Energy-E0(eV)\n")
    
    # Get reference energy
    ref_energy = None
    if 0 in final_data['images'] and final_data['images'][0]['energy'] is not None:
        ref_energy = final_data['images'][0]['energy']
    
    # Collect data
    for img_idx in sorted(final_data['images'].keys()):
        img_data = final_data['images'][img_idx]
        
        distance = img_data['distance_from_ref'] if img_data['distance_from_ref'] is not None else 0.0
        energy = img_data['energy']
        
        if energy is not None:
            rel_energy = energy - ref_energy if ref_energy is not None else 0.0
            data_lines.append(f"{img_idx:3d}  {distance:12.6f}  {energy:15.8f}  {rel_energy:12.6f}\n")
    
    # Save final data file
    final_dat = os.path.join(output_dir, "final_energy_profile.dat")
    with open(final_dat, 'w') as f:
        f.writelines(data_lines)
    
    # Create final plot
    final_plot = os.path.join(output_dir, "final_energy_profile.png")
    plot_single_stage(final_data, f"Final Profile ({source})", final_plot, ref_energy)
    
    return final_dat, final_plot

def run_neb_analysis(neb_dir, n_images, moving_o_idx, 
                    initial_relax_dir=None, final_relax_dir=None,
                    plot_results=True):
    """
    Main analysis function for multistage NEB calculations.
    
    Args:
        neb_dir: Path to multistage NEB directory
        n_images: Number of intermediate NEB images
        moving_o_idx: Index of the moving oxygen atom
        initial_relax_dir: Optional path to initial relaxation for endpoint energy
        final_relax_dir: Optional path to final relaxation for endpoint energy
        plot_results: Whether to generate plots
    
    Returns:
        Dictionary with analysis results
    """
    extractor = NEBEnergyExtractor()
    results = {
        'stages': {},
        'relaxation_energies': {
            'initial': None,
            'final': None
        },
        'summary': {}
    }
    
    # Extract relaxation energies if provided
    if initial_relax_dir:
        logging.info(f"Extracting initial relaxation energy from {initial_relax_dir}")
        results['relaxation_energies']['initial'] = extractor.extract_relaxation_energy(initial_relax_dir)
        if results['relaxation_energies']['initial']:
            logging.info(f"  Initial energy: {results['relaxation_energies']['initial']:.6f} eV")
    
    if final_relax_dir:
        logging.info(f"Extracting final relaxation energy from {final_relax_dir}")
        results['relaxation_energies']['final'] = extractor.extract_relaxation_energy(final_relax_dir)
        if results['relaxation_energies']['final']:
            logging.info(f"  Final energy: {results['relaxation_energies']['final']:.6f} eV")
    
    # Check for stage directories
    stage_dirs = []
    for stage_name in ['neb_stage1', 'neb_stage2', 'neb_stage3', 'neb_stage4']:
        stage_path = os.path.join(neb_dir, stage_name)
        if os.path.exists(stage_path):
            stage_dirs.append((stage_name, stage_path))
    
    if not stage_dirs:
        logging.error(f"No NEB stage directories found in {neb_dir}")
        return results
    
    # Extract data from each stage
    logging.info(f"Found {len(stage_dirs)} NEB stages")
    
    for stage_name, stage_path in stage_dirs:
        logging.info(f"\nAnalyzing {stage_name}...")
        stage_data = extractor.extract_neb_stage_data(stage_path, n_images, moving_o_idx)
        
        if stage_data['has_data']:
            results['stages'][stage_name] = stage_data
            
            # Replace endpoint energies with relaxation energies if available
            if results['relaxation_energies']['initial'] is not None and 0 in stage_data['images']:
                stage_data['images'][0]['energy'] = results['relaxation_energies']['initial']
                
            if results['relaxation_energies']['final'] is not None and (n_images + 1) in stage_data['images']:
                stage_data['images'][n_images + 1]['energy'] = results['relaxation_energies']['final']
            
            # Calculate summary statistics
            energies = [img['energy'] for img in stage_data['images'].values() if img['energy'] is not None]
            if energies:
                min_e = min(energies)
                max_e = max(energies)
                barrier = max_e - min_e
                
                logging.info(f"  Found {len(energies)} images with energies")
                logging.info(f"  Energy range: {min_e:.6f} to {max_e:.6f} eV")
                logging.info(f"  Barrier estimate: {barrier:.3f} eV")
                
                results['summary'][stage_name] = {
                    'n_images_with_energy': len(energies),
                    'min_energy': min_e,
                    'max_energy': max_e,
                    'barrier_estimate': barrier
                }
    
    # Generate plots if requested
    if plot_results and results['stages']:
        # Save individual stage data and plots
        logging.info("\nSaving individual stage analyses...")
        for stage_name, stage_data in results['stages'].items():
            if stage_data['has_data']:
                dat_file, plot_file = save_stage_data(stage_data, stage_name, neb_dir)
                logging.info(f"  {stage_name}: {os.path.basename(dat_file)}, {os.path.basename(plot_file)}")
        
        # Save final summary
        logging.info("\nSaving final energy profile...")
        final_dat, final_plot = save_final_summary(results['stages'], neb_dir)
        if final_dat and final_plot:
            logging.info(f"  Final profile: {os.path.basename(final_dat)}, {os.path.basename(final_plot)}")
        
        # Also create the comparison plot
        plot_file = os.path.join(os.path.dirname(neb_dir), "neb_energy_analysis.png")
        plot_path = plot_neb_energies(results['stages'], plot_file)
        logging.info(f"\nComparison plot saved to: {plot_path}")
        results['plot_file'] = plot_path
    
    return results


# Main execution
if __name__ == "__main__":

    class Args:
        """Simple class to hold arguments"""
        pass
    
    args = Args()
    args.neb_dir = './multistage_neb'
    args.n_images = 5
    args.moving_o_idx = 58
    args.initial_relax_dir = './relax_initial/stage1'
    args.final_relax_dir = './relax_final/stage1'
    args.no_plot = False


    # Run analysis with error handling
    try:
        results = run_neb_analysis(
            neb_dir=args.neb_dir,
            n_images=args.n_images,
            moving_o_idx=args.moving_o_idx,
            initial_relax_dir=args.initial_relax_dir,
            final_relax_dir=args.final_relax_dir,
            plot_results=not args.no_plot
        )
        
        # Print summary
        print("\n" + "="*60)
        print("NEB ENERGY ANALYSIS SUMMARY")
        print("="*60)
        
        if results['relaxation_energies']['initial'] is not None:
            print(f"Initial relaxation energy: {results['relaxation_energies']['initial']:.6f} eV")
        if results['relaxation_energies']['final'] is not None:
            print(f"Final relaxation energy: {results['relaxation_energies']['final']:.6f} eV")
        
        print(f"\nStages analyzed: {len(results['stages'])}")
        
        for stage_name, summary in results['summary'].items():
            print(f"\n{stage_name}:")
            print(f"  Images with energy: {summary['n_images_with_energy']}")
            print(f"  Barrier estimate: {summary['barrier_estimate']:.3f} eV")
        
        if 'plot_file' in results:
            print(f"\nPlot saved to: {results['plot_file']}")
            
    except Exception as e:
        logging.error(f"Analysis failed: {e}")
        import traceback
        traceback.print_exc()
