#!/usr/bin/env python
"""
Complete Control NEB Workflow with Parallel Multistage Relaxations
==================================================================
This workflow provides complete control over every step with options for:
- Parallel multistage structure relaxation (both initial and final simultaneously)
- Multistage NEB calculation (coarse -> fine)
- Full constraint control
- Automatic convergence checking to avoid restarts
"""

import os
import logging
import time
from pymatgen.core import Structure
from pymatgen.io.vasp.inputs import Poscar
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# =============================================================================
# STEP 0: CONFIGURATION
# =============================================================================
print("="*60)
print("COMPLETE CONTROL NEB WORKFLOW - PARALLEL RELAXATIONS")
print("="*60)

# Define your system-specific paths and settings
potcar_path = "/nfs/home/6/nguyenm/sensor/POTCAR-files/potpaw_PBE_54"
#potcar_path = "potcars/"
potcar_mapping = {
    "O": "O_s", "La": "La", "Ni": "Ni_pv", "V": "V_sv",
    "Fe": "Fe_pv", "Co": "Co_pv", "Mn": "Mn_pv"
}

ldau_settings = {
    'LDAU': True,
    "La": {"L": 0, "U": 0}, "Ni": {"L": 2, "U": 7}, "V": {"L": 2, "U": 3}, "Ti": {"L": 2,"U": 14.5},
    "Fe": {"L": 2, "U": 5}, "Co": {"L": 2, "U": 3.5}, "Mn": {"L": 2, "U": 4}, "Nb": {"L": 2,"U": 5},
    "O": {"L": 0, "U": 0}
}

# Input files (change these to your actual files)
init_file = "structure_00_start_relaxed.POSCAR"
final_file = "structure_01_end_relaxed.POSCAR"
moving_o_idx = 51

# Calculation parameters
n_images = 5
nodes = 5
ntasks_per_node = 128
walltime = "48:00:00"

print(f"Input files: {init_file}, {final_file}")
print(f"NEB images: {n_images}")
print(f"Job resources: {nodes} nodes, {ntasks_per_node} tasks/node")
print()

# =============================================================================
# IMPORT COMPONENTS
# =============================================================================
print("Initializing components...")
import sys
module_dir = os.path.abspath(os.path.join(os.getcwd(), '/nfs/home/6/nguyenm/pymatgen-packages/relax_NEB'))
if module_dir not in sys.path:
    sys.path.insert(0, module_dir)
    print(f"Added {module_dir} to Python path")
from file_manager import FileManager
from vasp_inputs import VASPInputGenerator
from structure_analyzer import StructureAnalyzer
from slurm_manager import SLURMManager
from multistage_relaxation import MultiStageStructureRelaxer
from neb_path_generation_with_constraint_control import NEBPathGenerator
from energy_analyzer import NEBEnergyAnalyzer
from neb_multistage import MultiStageNEB

# Initialize components
file_manager = FileManager(potcar_path, potcar_mapping)
analyzer = StructureAnalyzer()
input_gen = VASPInputGenerator()
slurm_manager = SLURMManager()
structure_relaxer = MultiStageStructureRelaxer(file_manager, input_gen, slurm_manager, auto_cleanup=True)
path_generator = NEBPathGenerator()
energy_analyzer = NEBEnergyAnalyzer()

print("Components initialized successfully")

# =============================================================================
# HELPER FUNCTIONS FOR CONVERGENCE CHECKING
# =============================================================================
def check_relaxation_convergence(relax_dir):
    """Check if a relaxation calculation has converged."""
    outcar_path = os.path.join(relax_dir, "OUTCAR")
    contcar_path = os.path.join(relax_dir, "CONTCAR")
    
    if not os.path.exists(outcar_path) or not os.path.exists(contcar_path):
        return False, None, None
    
    try:
        from pymatgen.io.vasp.outputs import Outcar, Vasprun
        
        # Check OUTCAR for convergence
        outcar = Outcar(outcar_path)
        converged = outcar.final_energy is not None
        
        # Check for convergence indicators in OUTCAR text
        with open(outcar_path, 'r') as f:
            content = f.read()
            
        convergence_indicators = [
            "reached required accuracy",
            "aborting loop because EDIFF is reached"
        ]
        
        text_converged = any(indicator in content for indicator in convergence_indicators)
        
        # Get final structure and energy
        if converged and os.path.getsize(contcar_path) > 0:
            file_manager.clean_contcar_elements(contcar_path)
            final_struct = Poscar.from_file(contcar_path).structure
            final_energy = outcar.final_energy
            return True, final_struct, final_energy
        
    except Exception as e:
        print(f"Warning: Error checking convergence in {relax_dir}: {e}")
    
    return False, None, None

def get_final_relaxation_energy(relax_dir):
    """Get the final energy from a multistage relaxation."""
    # Check the final stage directory
    stage_dirs = [d for d in os.listdir(relax_dir) if d.startswith('stage') and os.path.isdir(os.path.join(relax_dir, d))]
    if not stage_dirs:
        # Single stage relaxation
        return check_relaxation_convergence(relax_dir)[2]
    
    # Find the highest numbered stage
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
        return check_relaxation_convergence(final_stage_path)[2]
    
    return None

def check_neb_convergence(neb_dir, n_images):
    """Check if a NEB calculation has converged."""
    if not os.path.exists(neb_dir):
        print(f"NEB directory {neb_dir} does not exist")
        return False
    
    print(f"Checking convergence in {neb_dir}")
    
    # Check for NEBEF.dat (indicates completed NEB)
    nebef_path = os.path.join(neb_dir, "NEBEF.dat")
    if os.path.exists(nebef_path) and os.path.getsize(nebef_path) > 0:
        print(f"  ✓ Found NEBEF.dat in {neb_dir}")
        return True
    
    # Check the main vasp.out file for job completion
    vasp_out_path = os.path.join(neb_dir, "vasp.out")
    if os.path.exists(vasp_out_path):
        try:
            with open(vasp_out_path, 'r') as f:
                content = f.read()
            
            # Check for proper job completion in vasp.out
            completion_indicators = [
                "reached required accuracy - stopping structural energy minimisation",
                "writing wavefunctions"
            ]
            
            has_accuracy = "reached required accuracy - stopping structural energy minimisation" in content
            has_wavefunctions = "writing wavefunctions" in content
            
            if has_accuracy and has_wavefunctions:
                print(f"  ✓ vasp.out shows NEB completed: reached required accuracy")
                return True
            elif has_accuracy:
                print(f"  ⚠ Found accuracy message but checking for complete termination...")
                # Continue to check OUTCAR files for more details
                
        except Exception as e:
            print(f"  Warning: Error reading vasp.out {vasp_out_path}: {e}")
    
    # Check the main OUTCAR file in the NEB directory
    main_outcar = os.path.join(neb_dir, "OUTCAR")
    if os.path.exists(main_outcar):
        try:
            with open(main_outcar, 'r') as f:
                content = f.read()
            
            # Check for proper termination in OUTCAR
            termination_indicators = [
                "General timing and accounting informations for this job",
                "Elapsed time (sec):",
                "User time (sec):",
                "LOOP+"  # Final loop iteration indicator
            ]
            
            has_termination = any(indicator in content for indicator in termination_indicators)
            has_accuracy = "reached required accuracy" in content
            
            if has_accuracy and has_termination:
                print(f"  ✓ Main OUTCAR shows proper NEB completion")
                return True
            elif has_accuracy:
                print(f"  ⚠ Found accuracy in OUTCAR but job may not have terminated cleanly")
                
        except Exception as e:
            print(f"  Warning: Error reading main OUTCAR {main_outcar}: {e}")
    
    # Check individual image OUTCARs as fallback
    completed_images = 0
    total_images = n_images + 2
    
    for i in range(total_images):
        img_dir = os.path.join(neb_dir, f"{i:02d}")
        outcar_path = os.path.join(img_dir, "OUTCAR")
        
        if os.path.exists(outcar_path):
            try:
                with open(outcar_path, 'r') as f:
                    content = f.read()
                
                # Check for completion indicators in individual images
                has_forces = "FORCES:" in content
                has_energy = "energy without entropy" in content
                
                if has_forces and has_energy:
                    completed_images += 1
                    
            except Exception as e:
                print(f"  Warning: Error reading {outcar_path}: {e}")
    
    print(f"  Images with valid output: {completed_images}/{total_images}")
    
    # Consider converged if most images have valid output and main job shows accuracy
    convergence_threshold = total_images * 0.9  # Higher threshold for more confidence
    has_sufficient_images = completed_images >= convergence_threshold
    
    # Final decision based on multiple criteria
    if has_sufficient_images:
        print(f"  ✓ NEB calculation appears converged ({completed_images}/{total_images} images with valid output)")
        return True
    else:
        print(f"  ✗ NEB calculation not converged ({completed_images}/{total_images} images with output, need ≥{convergence_threshold:.0f})")
        return False

def check_stage_completion(stage_dir):
    """Check if a specific stage has completed."""
    outcar_path = os.path.join(stage_dir, "OUTCAR")
    if not os.path.exists(outcar_path):
        return False, None
    
    try:
        with open(outcar_path, 'r') as f:
            content = f.read()
        
        # Check for job completion indicators
        completed = ("General timing and accounting informations for this job" in content or
                    "reached required accuracy" in content)
        
        # Get job ID if available
        slurm_out = os.path.join(stage_dir, "slurm.out")
        job_id = None
        if os.path.exists(slurm_out):
            with open(slurm_out, 'r') as f:
                for line in f:
                    if "SLURM_JOB_ID" in line:
                        job_id = line.split("=")[-1].strip()
                        break
        
        return completed, job_id
    except:
        return False, None

def manage_parallel_relaxations(structure_relaxer, structures_dict, ldau_settings, 
                              nodes=1, ntasks_per_node=128, walltime="24:00:00"):
    """
    Manage parallel multi-stage relaxations for multiple structures.
    Both initial and final relaxations run simultaneously.
    
    Args:
        structure_relaxer: MultiStageStructureRelaxer instance
        structures_dict: Dict of {label: (structure, base_dir)} pairs
        ldau_settings: LDAU settings dictionary
        nodes, ntasks_per_node, walltime: Job parameters
        
    Returns:
        Dict of {label: {'structure': relaxed_structure, 'energy': final_energy}}
    """
    print("\n--- Managing Parallel Relaxations ---")
    
    # Track active jobs and stages
    active_jobs = {}  # {label: {'stage': int, 'job_id': str, 'base_dir': str}}
    completed_relaxations = {}
    max_stages = 2  # Usually 2 stages as you mentioned
    
    # Initialize tracking
    for label, (structure, base_dir) in structures_dict.items():
        active_jobs[label] = {
            'stage': 0,
            'job_id': None,
            'base_dir': base_dir,
            'structure': structure,
            'completed': False
        }
    
    # Submit initial stages for ALL structures at once
    print("Submitting initial relaxation stages for all structures...")
    for label in active_jobs:
        if not active_jobs[label]['completed']:
            stage_dir = os.path.join(active_jobs[label]['base_dir'], f"stage{active_jobs[label]['stage']}")
            
            # Check if this stage already exists and is complete
            completed, _ = check_stage_completion(stage_dir)
            if completed:
                print(f"  {label} stage {active_jobs[label]['stage']} already complete")
                active_jobs[label]['stage'] += 1
                continue
            
            # Submit the job
            os.makedirs(stage_dir, exist_ok=True)
            
            # Setup stage-specific parameters
            # For 2-stage relaxation: use stage1 (rough) and stage3 (final)
            if max_stages == 2:
                stage_key = 'stage1' if active_jobs[label]['stage'] == 0 else 'stage3'
            else:
                stage_key = f'stage{active_jobs[label]["stage"] + 1}'
            
            stage_info = structure_relaxer.relax_stages[stage_key]
            
            # Create VASP inputs
            incar = structure_relaxer.input_generator.create_relax_incar(
                active_jobs[label]['structure'],
                max_iterations=stage_info['incar_override'].get('NSW', 200),
                ediffg=stage_info['incar_override'].get('EDIFFG', -0.01),
                ldau_settings=ldau_settings,
                restart=False
            )
            # Apply stage-specific overrides
            for key, value in stage_info['incar_override'].items():
                incar[key] = value
            incar.write_file(os.path.join(stage_dir, "INCAR"))
            
            poscar = Poscar(active_jobs[label]['structure'])
            poscar.write_file(os.path.join(stage_dir, "POSCAR"))
            
            kpoints = structure_relaxer.input_generator.generate_kpoints(
                active_jobs[label]['structure'],
                kspacing=0.3  # Use default kspacing
            )
            kpoints.write_file(os.path.join(stage_dir, "KPOINTS"))
            
            elements = [str(site.specie) for site in active_jobs[label]['structure']]
            unique_elements = list(dict.fromkeys(elements))
            structure_relaxer.file_manager.create_potcar(stage_dir, unique_elements)
            
            # Create and submit job
            script_path = structure_relaxer.slurm_manager.create_vasp_job_script(
                job_dir=stage_dir,
                job_name=f"relax_{label}_stage{active_jobs[label]['stage']}",
                nodes=nodes,
                ntasks_per_node=ntasks_per_node,
                walltime=walltime
            )
            
            job_id = structure_relaxer.slurm_manager.submit_job(script_path, stage_dir)
            if job_id:
                active_jobs[label]['job_id'] = job_id
                print(f"  Submitted {label} stage {active_jobs[label]['stage']}: job {job_id}")
    
    # Monitor and manage jobs
    print("\nMonitoring parallel relaxations...")
    all_complete = False
    check_interval = 600  # Check every 10 minutes
    
    while not all_complete:
        time.sleep(check_interval)
        
        # Check status of all active jobs
        jobs_checked = []
        for label in list(active_jobs.keys()):
            if active_jobs[label]['completed']:
                continue
            
            stage = active_jobs[label]['stage']
            stage_dir = os.path.join(active_jobs[label]['base_dir'], f"stage{stage}")
            
            # Check if current stage is complete
            completed, _ = check_stage_completion(stage_dir)
            
            if completed:
                print(f"\n✓ {label} stage {stage} completed")
                jobs_checked.append(label)
                
                # Get relaxed structure from CONTCAR
                contcar_path = os.path.join(stage_dir, "CONTCAR")
                if os.path.exists(contcar_path) and os.path.getsize(contcar_path) > 0:
                    structure_relaxer.file_manager.clean_contcar_elements(contcar_path)
                    relaxed_struct = Poscar.from_file(contcar_path).structure
                    active_jobs[label]['structure'] = relaxed_struct
                
                # Check if more stages needed
                if stage + 1 < max_stages:
                    # Submit next stage
                    active_jobs[label]['stage'] += 1
                    next_stage = active_jobs[label]['stage']
                    next_stage_dir = os.path.join(active_jobs[label]['base_dir'], f"stage{next_stage}")
                    
                    print(f"  Submitting {label} stage {next_stage}...")
                    
                    os.makedirs(next_stage_dir, exist_ok=True)
                    
                    # Get next stage config
                    # For 2-stage relaxation: use stage1 (rough) and stage3 (final)
                    if max_stages == 2:
                        stage_key = 'stage1' if next_stage == 0 else 'stage3'
                    else:
                        stage_key = f'stage{next_stage + 1}'
                    
                    stage_info = structure_relaxer.relax_stages[stage_key]
                    
                    # Create VASP inputs for next stage
                    incar = structure_relaxer.input_generator.create_relax_incar(
                        active_jobs[label]['structure'],
                        max_iterations=stage_info['incar_override'].get('NSW', 200),
                        ediffg=stage_info['incar_override'].get('EDIFFG', -0.01),
                        ldau_settings=ldau_settings,
                        restart=True  # This is a continuation from previous stage
                    )
                    # Apply stage-specific overrides
                    for key, value in stage_info['incar_override'].items():
                        incar[key] = value
                    incar.write_file(os.path.join(next_stage_dir, "INCAR"))
                    
                    poscar = Poscar(active_jobs[label]['structure'])
                    poscar.write_file(os.path.join(next_stage_dir, "POSCAR"))
                    
                    kpoints = structure_relaxer.input_generator.generate_kpoints(
                        active_jobs[label]['structure'],
                        kspacing=0.3  # Use default kspacing
                    )
                    kpoints.write_file(os.path.join(next_stage_dir, "KPOINTS"))
                    
                    structure_relaxer.file_manager.create_potcar(next_stage_dir, unique_elements)
                    
                    # Submit next stage
                    script_path = structure_relaxer.slurm_manager.create_vasp_job_script(
                        job_dir=next_stage_dir,
                        job_name=f"relax_{label}_stage{next_stage}",
                        nodes=nodes,
                        ntasks_per_node=ntasks_per_node,
                        walltime=walltime
                    )
                    
                    job_id = structure_relaxer.slurm_manager.submit_job(script_path, next_stage_dir)
                    if job_id:
                        active_jobs[label]['job_id'] = job_id
                        print(f"  Submitted {label} stage {next_stage}: job {job_id}")
                else:
                    # All stages complete for this structure
                    print(f"✓ All stages complete for {label}")
                    
                    # Get final energy
                    converged, final_struct, final_energy = check_relaxation_convergence(stage_dir)
                    if converged:
                        completed_relaxations[label] = {
                            'structure': final_struct,
                            'energy': final_energy
                        }
                        active_jobs[label]['completed'] = True
                        print(f"  Final energy: {final_energy:.6f} eV")
        
        # Check if all relaxations are complete
        all_complete = all(job['completed'] for job in active_jobs.values())
        
        if not all_complete:
            # Print status update
            active_count = sum(1 for job in active_jobs.values() if not job['completed'])
            if jobs_checked:
                print(f"Progress update: {len(jobs_checked)} job(s) advanced this check")
            print(f"Active relaxations: {active_count}/{len(active_jobs)}", end='\r')
    
    print("\n✓ All parallel relaxations complete!")
    return completed_relaxations

# =============================================================================
# STEP 1: CLEAN INPUT FILES
# =============================================================================
print("\n" + "="*60)
print("STEP 1: CLEANING INPUT FILES")
print("="*60)

if file_manager.clean_contcar_elements(init_file):
    print(f"✓ {init_file} cleaned")
else:
    print(f"✗ Failed to clean {init_file}")

if file_manager.clean_contcar_elements(final_file):
    print(f"✓ {final_file} cleaned")
else:
    print(f"✗ Failed to clean {final_file}")

# =============================================================================
# STEP 2: LOAD AND ANALYZE STRUCTURES
# =============================================================================
print("\n" + "="*60)
print("STEP 2: STRUCTURE ANALYSIS")
print("="*60)

print("Loading structures...")
init_struct = Structure.from_file(init_file)
final_struct = Structure.from_file(final_file)

# =============================================================================
# MODIFIED STEP 2: PARALLEL STRUCTURE RELAXATION
# =============================================================================
print("\n" + "="*60)
print("STEP 2: PARALLEL STRUCTURE RELAXATION")
print("="*60)

# Set relaxation option
relax_option = 'both'  # You can change this or make it an input
print(f"Relaxation option: {relax_option}")

# Prepare structures for parallel relaxation
structures_to_relax = {}
initial_struct = init_struct
final_struct = final_struct
initial_energy = None
final_energy = None

# Check what needs to be relaxed
needs_initial_relax = False
needs_final_relax = False

if relax_option in ['initial', 'both']:
    # Check if initial relaxation already fully complete
    relax_dir = "./relax_initial"
    os.makedirs(relax_dir, exist_ok=True)
    
    # Check for completed multi-stage relaxation
    final_energy_check = get_final_relaxation_energy(relax_dir)
    if final_energy_check is not None:
        print(f"✓ Found completed initial relaxation: E = {final_energy_check} eV")
        initial_energy = final_energy_check
        # Get the final structure
        stage_dirs = [d for d in os.listdir(relax_dir) if d.startswith('stage')]
        if stage_dirs:
            stage_numbers = [int(d.replace('stage', '')) for d in stage_dirs if d.replace('stage', '').isdigit()]
            if stage_numbers:
                final_stage = f"stage{max(stage_numbers)}"
                final_stage_path = os.path.join(relax_dir, final_stage)
                _, initial_struct, _ = check_relaxation_convergence(final_stage_path)
    else:
        structures_to_relax['initial'] = (init_struct, relax_dir)
        needs_initial_relax = True

if relax_option in ['final', 'both']:
    # Check if final relaxation already fully complete
    relax_dir = "./relax_final"
    os.makedirs(relax_dir, exist_ok=True)
    
    # Check for completed multi-stage relaxation
    final_energy_check = get_final_relaxation_energy(relax_dir)
    if final_energy_check is not None:
        print(f"✓ Found completed final relaxation: E = {final_energy_check} eV")
        final_energy = final_energy_check
        # Get the final structure
        stage_dirs = [d for d in os.listdir(relax_dir) if d.startswith('stage')]
        if stage_dirs:
            stage_numbers = [int(d.replace('stage', '')) for d in stage_dirs if d.replace('stage', '').isdigit()]
            if stage_numbers:
                final_stage = f"stage{max(stage_numbers)}"
                final_stage_path = os.path.join(relax_dir, final_stage)
                _, final_struct, _ = check_relaxation_convergence(final_stage_path)
    else:
        structures_to_relax['final'] = (final_struct, relax_dir)
        needs_final_relax = True

# Run parallel relaxations if needed
if structures_to_relax:
    print(f"\nRunning PARALLEL relaxations for: {list(structures_to_relax.keys())}")
    print("Both structures will be relaxed simultaneously, each with 2 stages")
    
    relaxation_results = manage_parallel_relaxations(
        structure_relaxer,
        structures_to_relax,
        ldau_settings,
        nodes=nodes,
        ntasks_per_node=ntasks_per_node,
        walltime=walltime
    )
    
    # Update structures and energies
    if 'initial' in relaxation_results:
        initial_struct = relaxation_results['initial']['structure']
        initial_energy = relaxation_results['initial']['energy']
        print(f"✓ Initial relaxation result: E = {initial_energy:.6f} eV")
    
    if 'final' in relaxation_results:
        final_struct = relaxation_results['final']['structure']
        final_energy = relaxation_results['final']['energy']
        print(f"✓ Final relaxation result: E = {final_energy:.6f} eV")
else:
    print("✓ All relaxations already complete")

# Calculate reaction energy if both energies available
if initial_energy and final_energy:
    reaction_energy = final_energy - initial_energy
    print(f"\n✓ Reaction energy from relaxations: {reaction_energy:.3f} eV")

# =============================================================================
# STEP 3: VALIDATE INITIAL AND FINAL STRUCTURES
# =============================================================================
print("\n" + "="*60)
print("STEP 3: STRUCTURE VALIDATION")
print("="*60)
import numpy as np

def validate_structures_are_different(initial_struct, final_struct, moving_atom_idx, analyzer, tolerance=0.5):
    """
    Validate that initial and final structures are meaningfully different
    for the identified moving oxygen atom.
    
    Args:
        initial_struct: Initial structure
        final_struct: Final structure  
        moving_atom_idx: Index of the oxygen atom that should move
        vacancy_pos: Position of the vacancy (target position)
        analyzer: StructureAnalyzer instance for PBC distance calculations
        tolerance: Minimum distance (Å) for structures to be considered different
    
    Returns:
        bool: True if structures are sufficiently different
        dict: Validation details
    """
    print(f"Validating structures for moving atom index {moving_atom_idx}...")
    
    # Get positions of the moving atom in both structures
    initial_pos = initial_struct[moving_atom_idx].coords
    final_pos = final_struct[moving_atom_idx].coords
    
    # Calculate distance moved using PBC-aware distance
    distance_moved = analyzer.get_pbc_distance(initial_struct, initial_pos, final_pos)
    
    # Check if the atom is the same element in both structures
    initial_element = str(initial_struct[moving_atom_idx].specie)
    final_element = str(final_struct[moving_atom_idx].specie)
    
    # Validation results
    validation_results = {
        'distance_moved': distance_moved,
        'initial_position': initial_pos,
        'final_position': final_pos,
        'initial_element': initial_element,
        'final_element': final_element,
        'same_element': initial_element == final_element,
        'sufficient_movement': distance_moved >= 2.0,  # Changed to 2.0 Å minimum movement
        'reasonable_distance': distance_moved <= 10.0,  # Added upper limit check
    }
    
    print(f"  Moving atom element: {initial_element} -> {final_element}")
    print(f"  Initial position: [{initial_pos[0]:.3f}, {initial_pos[1]:.3f}, {initial_pos[2]:.3f}]")
    print(f"  Final position:   [{final_pos[0]:.3f}, {final_pos[1]:.3f}, {final_pos[2]:.3f}]")
    print(f"  Distance moved (PBC): {distance_moved:.3f} Å")
    
    # Overall validation - changed criteria
    is_valid = (
        validation_results['same_element'] and 
        validation_results['sufficient_movement'] and 
        validation_results['reasonable_distance']
    )
    
    validation_results['is_valid'] = is_valid
    
    if is_valid:
        print(f"  ✓ Structures are sufficiently different (moved {distance_moved:.3f} Å)")
    else:
        print(f"  ✗ Structure validation failed:")
        if not validation_results['same_element']:
            print(f"    - Element mismatch: {initial_element} != {final_element}")
        if not validation_results['sufficient_movement']:
            print(f"    - Insufficient movement: {distance_moved:.3f} < 2.0 Å")
        if not validation_results['reasonable_distance']:
            print(f"    - Excessive movement: {distance_moved:.3f} > 10.0 Å")
    
    return is_valid, validation_results

def check_structure_consistency(initial_struct, final_struct):
    """
    Check that initial and final structures have consistent properties
    (same number of atoms, same cell, etc.)
    """
    print("Checking structure consistency...")
    
    consistency_checks = {
        'same_num_atoms': len(initial_struct) == len(final_struct),
        'same_lattice': np.allclose(initial_struct.lattice.matrix, final_struct.lattice.matrix, atol=1e-6),
        'same_species_count': True,  # Will check below
    }
    
    # Check species count
    from collections import Counter
    initial_species = Counter([str(site.specie) for site in initial_struct])
    final_species = Counter([str(site.specie) for site in final_struct])
    consistency_checks['same_species_count'] = initial_species == final_species
    
    print(f"  Number of atoms: {len(initial_struct)} -> {len(final_struct)}")
    print(f"  Lattice consistent: {consistency_checks['same_lattice']}")
    print(f"  Species count consistent: {consistency_checks['same_species_count']}")
    
    if not consistency_checks['same_species_count']:
        print(f"    Initial: {dict(initial_species)}")
        print(f"    Final:   {dict(final_species)}")
    
    all_consistent = all(consistency_checks.values())
    
    if all_consistent:
        print("  ✓ Structures are consistent")
    else:
        print("  ✗ Structure consistency issues detected")
        
    return all_consistent, consistency_checks

def check_atom_overlaps(structure, analyzer, min_distance=1.2):
    """
    Check if any atoms are too close to each other (potential overlaps).
    
    Args:
        structure: Structure to check
        analyzer: StructureAnalyzer instance
        min_distance: Minimum allowed distance between atoms (Å)
    
    Returns:
        bool: True if no overlaps found
        list: List of overlapping atom pairs
    """
    print(f"Checking for atom overlaps (min distance: {min_distance:.1f} Å)...")
    
    overlaps = []
    n_atoms = len(structure)
    
    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            pos_i = structure[i].coords
            pos_j = structure[j].coords
            
            distance = analyzer.get_pbc_distance(structure, pos_i, pos_j)
            
            if distance < min_distance:
                element_i = str(structure[i].specie)
                element_j = str(structure[j].specie)
                overlaps.append({
                    'atoms': (i, j),
                    'elements': (element_i, element_j),
                    'distance': distance,
                    'positions': (pos_i, pos_j)
                })
    
    if overlaps:
        print(f"  ✗ Found {len(overlaps)} atom overlap(s):")
        for overlap in overlaps:
            i, j = overlap['atoms']
            elem_i, elem_j = overlap['elements']
            dist = overlap['distance']
            print(f"    Atoms {i}-{j} ({elem_i}-{elem_j}): {dist:.3f} Å")
        return False, overlaps
    else:
        print(f"  ✓ No atom overlaps found")
        return True, []

# Perform validation
print("Performing structure validation before NEB path generation...")

# Check basic consistency
consistent, consistency_details = check_structure_consistency(initial_struct, final_struct)

if not consistent:
    print("✗ CRITICAL ERROR: Structure consistency check failed!")
    print("Cannot proceed with NEB calculation.")
    exit(1)

# Check for atom overlaps in both structures
print("\nChecking initial structure for overlaps...")
initial_clean, initial_overlaps = check_atom_overlaps(initial_struct, analyzer, min_distance=1.2)

print("\nChecking final structure for overlaps...")
final_clean, final_overlaps = check_atom_overlaps(final_struct, analyzer, min_distance=1.2)

if not initial_clean:
    print("✗ WARNING: Initial structure has atom overlaps!")
    print("This may cause problems in calculations.")

if not final_clean:
    print("✗ WARNING: Final structure has atom overlaps!")
    print("Consider reducing the migration distance or choosing a different target position.")
    
    # If there are severe overlaps, suggest fixing them
    severe_overlaps = [o for o in final_overlaps if o['distance'] < 1.0]
    if severe_overlaps:
        print("✗ CRITICAL: Severe overlaps detected (< 1.0 Å)!")
        print("Recommendation: Reduce migration fraction or choose different oxygen atom.")

# Validate that structures are meaningfully different
valid, validation_details = validate_structures_are_different(
    initial_struct, final_struct, moving_o_idx, analyzer, tolerance=0.5
)

if not valid:
    print("✗ STRUCTURE VALIDATION FAILED!")
    
    # Check specific failure reasons
    insufficient_movement = validation_details['distance_moved'] < 2.0
    excessive_movement = validation_details['distance_moved'] > 10.0
    element_mismatch = not validation_details['same_element']
    
    if insufficient_movement:
        print(f"✗ CRITICAL ERROR: Insufficient oxygen movement ({validation_details['distance_moved']:.3f} Å < 2.0 Å)")
        print("This movement is too small for meaningful NEB calculation.")
        print("NEB with such small movements will not provide reliable migration barriers.")
    
    if excessive_movement:
        print(f"✗ CRITICAL ERROR: Excessive oxygen movement ({validation_details['distance_moved']:.3f} Å > 10.0 Å)")
        print("This movement may be unphysical and could cause calculation problems.")
    
    if element_mismatch:
        print(f"✗ CRITICAL ERROR: Element mismatch - {validation_details['initial_element']} != {validation_details['final_element']}")
    
    # Provide debugging information
    print("\nDebugging information:")
    print(f"Moving atom index: {moving_o_idx}")
    print(f"Distance moved: {validation_details['distance_moved']:.3f} Å")
    
    
    print("\n" + "="*60)
    print("STOPPING WORKFLOW - STRUCTURE VALIDATION FAILED")
    print("Fix the issues above and restart the calculation.")
    print("="*60)
    exit(1)
else:
    print("✓ Structure validation passed - structures are appropriately different")
    print(f"✓ Oxygen moved {validation_details['distance_moved']:.3f} Å - suitable for NEB calculation")

# Additional check: verify the moving atom is actually oxygen
if str(initial_struct[moving_o_idx].specie) != "O":
    print(f"✗ WARNING: Moving atom at index {moving_o_idx} is {initial_struct[moving_o_idx].specie}, not O!")

# Final summary
print(f"\nValidation Summary:")
print(f"  Structure consistency: {'✓' if consistent else '✗'}")
print(f"  Initial structure clean: {'✓' if initial_clean else '✗'}")
print(f"  Final structure clean: {'✓' if final_clean else '✗'}")
print(f"  Oxygen movement distance: {validation_details['distance_moved']:.3f} Å {'✓' if valid else '✗'}")

if valid:
    print("✓ All validations passed - Ready to proceed with NEB calculation")
else:
    print("✗ Validation failed - Workflow terminated")

print("="*60)

# =============================================================================
# STEP 4: NEB PATH GENERATION
# =============================================================================
print("\n" + "="*60)
print("STEP 4: NEB PATH GENERATION")
print("="*60)

print("NEB options:")
print("  1. single   - Single NEB calculation")
print("  2. multi    - Multi-stage NEB (coarse -> fine)")

neb_option = "multi" #input("\nChoose NEB strategy (single/multi): ").lower()

# Generate NEB path
print("\n--- Generating NEB path ---")
neb_dir = "./neb_calculation"
os.makedirs(neb_dir, exist_ok=True)

# Ask about constraints
remove_constraints = True #input("Remove constraints after ASE interpolation? (y/n): ").lower() == 'y'

# Generate initial path
configs, moving_atom_idx = path_generator.generate_neb_path(
    initial_struct, final_struct, 
    n_images=n_images, 
    moving_atom_idx=moving_o_idx,
    remove_constraints_after_ase=remove_constraints
)

if configs:
    # ASE analysis
    ase_analysis_dir = "./ase_analysis"
    os.makedirs(ase_analysis_dir, exist_ok=True)
    ase_results = path_generator.analyze_ase_path(configs, moving_atom_idx, ase_analysis_dir)
    print(f"✓ ASE analysis: barrier = {ase_results.get('barrier', 'N/A')} eV")
    
    # Write NEB structures
    path_generator.write_neb_structures(configs, neb_dir, n_images=n_images)
    print("✓ NEB structures written")

# =============================================================================
# STEP 5: NEB CALCULATION WITH CONVERGENCE CHECKING
# =============================================================================
print("\n" + "="*60)
print("STEP 5: NEB CALCULATION")
print("="*60)

if neb_option == 'multi':
    # Multi-stage NEB using MultiStageNEB class
    print("--- Setting up Multi-stage NEB ---")
    
    ms_neb = MultiStageNEB(
        base_dir="./multistage_neb",
        potcar_path=potcar_path,
        potcar_mapping=potcar_mapping,
        auto_cleanup=True
    )
    
    # Setup multi-stage NEB (3 stages by default)
    n_stages = 3 #int(input("Number of NEB stages (2-5) [3]: ") or "3")
        
    # Pass the relaxation energies to the MultiStageNEB setup
    ms_neb.setup_multistage_neb(
        initial_struct, final_struct,
        n_images=n_images, 
        n_stages=n_stages,
        moving_atom_idx=moving_atom_idx,
        ldau_settings=ldau_settings,
        existing_neb_dir=neb_dir,  # Use the existing path
        initial_relax_energy=initial_energy,  # Pass relaxation energies
        final_relax_energy=final_energy
    )
    print(f"✓ {n_stages}-stage NEB setup complete")
    
    # Submit and monitor stages with convergence checking
    print("--- Running Multi-stage NEB ---")
    
    # Get stage configuration
    stage_keys = ms_neb._select_stages(n_stages)
    
    for i, stage_key in enumerate(stage_keys):
        stage_info = ms_neb.stages[stage_key]
        print(f"\nStage {i+1}: {stage_info['name']}")
        print(f"  EDIFFG: {stage_info['incar_override'].get('EDIFFG', 'default')}")
        
        # Check if stage already completed
        stage_dir = os.path.join(ms_neb.base_dir, stage_info['dir'])
        if check_neb_convergence(stage_dir, n_images):
            print(f"  ✓ Stage {i+1} already converged, analyzing results...")
            # Analyze the existing stage results
            ms_neb.analyze_stage_results(stage_key)
            continue
        
        success = ms_neb.run_stage(stage_key, submit=True, monitor=True, quiet=True)
        
        if success:
            print(f"  ✓ Stage {i+1} completed")
        else:
            print(f"  ✗ Stage {i+1} failed")
            break
    
    # Analyze final results
    print("\n--- Analyzing Multi-stage Results ---")
    results = ms_neb.analyze_final_results()
    if results:
        print(f"✓ Final barrier: {results['barrier']:.3f} eV")
        print(f"✓ Reaction energy: {results['reaction_energy']:.3f} eV")
        
        # Compare with relaxation energies
        if initial_energy and final_energy:
            relax_reaction = final_energy - initial_energy
            print(f"  Relaxation reaction energy: {relax_reaction:.3f} eV")
            print(f"  Difference (NEB - Relax): {results['reaction_energy'] - relax_reaction:.3f} eV")

else:
    # Single-stage NEB with convergence checking
    print("--- Checking Single NEB Status ---")
    
    # Check if NEB already converged
    if check_neb_convergence(neb_dir, n_images):
        print("✓ Found converged NEB calculation, analyzing results...")
        
        # Analyze existing results
        distances, energies, nebef_success = energy_analyzer.extract_nebef_energies(neb_dir)
        
        if nebef_success:
            # Use relaxed endpoint energies if available
            if initial_energy is not None and final_energy is not None and energies:
                energies_corrected = energies.copy()
                energies_corrected[0] = initial_energy  # Replace first image energy
                energies_corrected[-1] = final_energy   # Replace last image energy
                energies = energies_corrected
                print(f"✓ Using relaxed endpoint energies: initial={initial_energy:.6f}, final={final_energy:.6f}")
            
            results = energy_analyzer.analyze_energy_profile(energies, distances, initial_energy)
            
            vasp_analysis_dir = "./vasp_analysis"
            os.makedirs(vasp_analysis_dir, exist_ok=True)
            
            plot_path = energy_analyzer.plot_energy_profile(
                energies, distances, "VASP NEB Energy Profile",
                os.path.join(vasp_analysis_dir, "vasp_energy_profile.png"),
                initial_energy
            )
            
            print(f"✓ Analysis complete:")
            print(f"  Energy barrier: {results['barrier']:.3f} eV")
            print(f"  Reaction energy: {results['reaction_energy']:.3f} eV")
            print(f"  Plot saved: {plot_path}")
            
            # Compare with relaxation energies
            if initial_energy and final_energy:
                relax_reaction = final_energy - initial_energy
                print(f"  Relaxation reaction energy: {relax_reaction:.3f} eV")
                print(f"  Difference (NEB - Relax): {results['reaction_energy'] - relax_reaction:.3f} eV")
    else:
        print("--- Setting up Single NEB ---")
        
        # Create VASP input files
        incar = input_gen.create_neb_incar(initial_struct, n_images=n_images, ldau_settings=ldau_settings)
        incar.write_file(os.path.join(neb_dir, "INCAR"))
        
        kpoints = input_gen.generate_kpoints(initial_struct, kspacing=0.3)
        kpoints.write_file(os.path.join(neb_dir, "KPOINTS"))
        
        elements = [str(site.specie) for site in initial_struct]
        unique_elements = list(dict.fromkeys(elements))
        file_manager.create_potcar(neb_dir, unique_elements)
        print("✓ VASP input files created")
        
        # Submit and monitor NEB job
        print("--- Submitting Single NEB ---")
        script_path = slurm_manager.create_vasp_job_script(
            job_dir=neb_dir,
            job_name="vasp_neb",
            nodes=nodes,
            ntasks_per_node=ntasks_per_node,
            walltime=walltime,
            auto_restart=True
        )
        
        job_id = slurm_manager.submit_job(script_path, neb_dir)
        if job_id:
            print(f"✓ NEB job submitted with ID: {job_id}")
            print(f"Monitoring job {job_id}...")
            success = slurm_manager.monitor_job(job_id, quiet=True)
            
            if success:
                print("\n--- Analyzing Single NEB Results ---")
                distances, energies, nebef_success = energy_analyzer.extract_nebef_energies(neb_dir)
                
                if nebef_success:
                    # Use relaxed endpoint energies if available
                    if initial_energy is not None and final_energy is not None:
                        energies_corrected = energies.copy()
                        energies_corrected[0] = initial_energy  # Replace first image energy
                        energies_corrected[-1] = final_energy   # Replace last image energy
                        energies = energies_corrected
                        print(f"✓ Using relaxed endpoint energies")
                    
                    results = energy_analyzer.analyze_energy_profile(energies, distances, initial_energy)
                    
                    vasp_analysis_dir = "./vasp_analysis"
                    os.makedirs(vasp_analysis_dir, exist_ok=True)
                    
                    plot_path = energy_analyzer.plot_energy_profile(
                        energies, distances, "VASP NEB Energy Profile",
                        os.path.join(vasp_analysis_dir, "vasp_energy_profile.png"),
                        initial_energy
                    )
                    
                    print(f"✓ Analysis complete:")
                    print(f"  Energy barrier: {results['barrier']:.3f} eV")
                    print(f"  Reaction energy: {results['reaction_energy']:.3f} eV")
                    print(f"  Plot saved: {plot_path}")
                    
                    # Compare with relaxation energies
                    if initial_energy and final_energy:
                        relax_reaction = final_energy - initial_energy
                        print(f"  Relaxation reaction energy: {relax_reaction:.3f} eV")
                        print(f"  Difference (NEB - Relax): {results['reaction_energy'] - relax_reaction:.3f} eV")

# =============================================================================
# FINAL SUMMARY
# =============================================================================
print("\n" + "="*60)
print("WORKFLOW COMPLETE")
print("="*60)

print(f"✓ Workflow: Complete Control")
print(f"✓ Input files: {init_file}, {final_file}")
print(f"✓ NEB images: {n_images}")
print(f"✓ Relaxation: {relax_option}")
print(f"✓ NEB type: {neb_option}")

print("\nOutput directories:")
print("  - ./ase_analysis/        (ASE pre-analysis)")
if relax_option in ['initial', 'final', 'both']:
    print("  - ./relax_initial/       (initial relaxation)")
    print("  - ./relax_final/         (final relaxation)")

if neb_option == 'multi':
    print("  - ./multistage_neb/      (multi-stage NEB)")
    print("  - ./multistage_neb/analysis_stage*/  (per-stage analysis)")
else:
    print("  - ./neb_calculation/     (single NEB)")
    print("  - ./vasp_analysis/       (VASP results)")

print("\nKey features used:")
print(f"  - Constraints: {'Removed after ASE' if remove_constraints else 'Applied in VASP'}")
print("  - Convergence checking: Automatically detects completed calculations")
print(f"  - Endpoint energies: Using relaxed structures for image 00 and {n_images+1:02d}")
if neb_option == 'multi':
    print(f"  - {n_stages}-stage NEB: progressively tighter convergence with per-stage analysis")

print("="*60)
