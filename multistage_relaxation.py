#!/usr/bin/env python
"""
Multi-stage structure relaxation utilities with 2 or 3 stages.
Each stage uses progressively tighter convergence criteria.
"""

import os
import json
import logging
import shutil
from datetime import datetime
from typing import Dict, Optional, Tuple, Union, List
from pymatgen.core import Structure
from pymatgen.io.vasp.inputs import Poscar, Incar
from pymatgen.io.vasp.outputs import Outcar, Poscar as PoscarOutput

logger = logging.getLogger(__name__)

class MultiStageStructureRelaxer:
    """Handles multi-stage structure relaxation calculations."""
    
    def __init__(self, file_manager, input_generator, slurm_manager, auto_cleanup: bool = True):
        """
        Initialize multi-stage structure relaxer.
        
        Args:
            file_manager: FileManager instance
            input_generator: VASPInputGenerator instance
            slurm_manager: SLURMManager instance
            auto_cleanup: Whether to automatically clean up files
        """
        self.file_manager = file_manager
        self.input_generator = input_generator
        self.slurm_manager = slurm_manager
        self.auto_cleanup = auto_cleanup
        
        # Define relaxation stage configurations
        self.relax_stages = self._define_relaxation_stages()
    
    def _define_relaxation_stages(self) -> Dict:
        """Define the relaxation stage configurations."""
        stages = {
            'stage1': {
                'name': 'Rough Relaxation',
                'description': 'Initial structure relaxation with loose convergence',
                'dir': 'relax_stage1',
                'incar_override': {
                    # Relaxation settings
                    'IBRION': 3,          # Use conjugate gradient
                    'POTIM' : 0.05,
                    'NSW': 50,           # Fewer steps for rough relaxation
                    'ISIF': 2,           # Relax ions only
                    'EDIFFG': -0.1,      # Loose force convergence
                    'EDIFF': 1E-4,       # Loose electronic convergence
                    'NELM': 50,         # Fewer electronic steps
                    'ALGO': 'Fast',  # Fast algorithm for rough relaxation
                    'PREC': 'Normal',    # Normal precision
                    'LSCALAPACK': True,
                    'LSCALU' : False,
                    'NCORE': 32,
                    'ISMEAR': 0,
                    'SIGMA': 0.05,
                    'LASPH': True,
                    'LREAL': 'Auto',
                    'LMAXMIX': 4,
                    'ENCUT': 520,        # Lower cutoff for speed
                    
                    # Large system parallelization
                    #'LPLANE': True,
                    #'NPAR': 1,           # Conservative for stage 1
                    #'NSIM': 4
                }
            },
            'stage2': {
                'name': 'Intermediate Relaxation',
                'description': 'Refined relaxation with intermediate convergence',
                'dir': 'relax_stage2',
                'incar_override': {
                    # Relaxation settings
                    'IBRION': 2,          # Use conjugate gradient
                    'POTIM' : 0.1,
                    'NSW': 150,          # More steps for intermediate
                    'ISIF': 2,           # Relax ions only
                    'EDIFFG': -0.05,     # Intermediate force convergence
                    'EDIFF': 1E-5,       # Better electronic convergence
                    'NELM': 150,         # More electronic steps
                    'ALGO': 'Fast',    # Normal algorithm
                    'PREC': 'Accurate',  # Higher precision for stage 2
                    'LSCALAPACK': True,
                    'LSCALU' : False,
                    'NCORE': 32,
                    'ISMEAR': 0,
                    'SIGMA': 0.05,
                    'LASPH': True,
                    'LREAL': 'Auto',
                    'LMAXMIX': 6,
                    'ENCUT': 520,        # Intermediate cutoff
                    
                    # Large system parallelization
                    #'LPLANE': True,
                    #'NPAR': 2,           # Can increase for stage 2
                    #'NSIM': 4
                }
            },
            'stage3': {
                'name': 'Final Tight Relaxation',
                'description': 'Final relaxation with tight convergence',
                'dir': 'relax_stage3',
                'incar_override': {
                    # Relaxation settings - match pymatgen_NEB_v6 defaults
                    'IBRION': 2,          # Use conjugate gradient (matches pymatgen_NEB_v6)
                    'POTIM' : 0.1,
                    'NSW': 200,          # Maximum ionic steps (matches pymatgen_NEB_v6)
                    'ISIF': 2,           # Relax ions only
                    'EDIFFG': -0.01,     # Tight force convergence (matches pymatgen_NEB_v6)
                    'EDIFF': 1E-6,       # Energy convergence (matches pymatgen_NEB_v6)
                    'NELM': 199,         # Maximum electronic steps (matches pymatgen_NEB_v6)
                    'ALGO': 'Fast',      # Fast algorithm (matches pymatgen_NEB_v6)
                    'PREC': 'Accurate',  # Accurate precision (matches pymatgen_NEB_v6)
                    'LSCALAPACK': True,
                    'LSCALU' : False,
                    'NCORE': 32,         # Matches pymatgen_NEB_v6
                    'ISMEAR': 0,
                    'SIGMA': 0.05,
                    'LASPH': True,
                    'LREAL': 'Auto',
                    'LMAXMIX': 6,
                    'ENCUT': 600,        # Full cutoff for final stage
                    
                    # Large system parallelization
                    #'LPLANE': True,
                    #'NPAR': 4,           # Most aggressive for final stage
                    #'NSIM': 4
                }
            }
        }
        return stages
    
    def run_multistage_relaxation(self, structure: Structure, base_dir: str,
                                 structure_name: str = "structure",
                                 n_stages: int = 2, 
                                 ldau_settings: Optional[Dict] = None,
                                 custom_kspacing: float = 0.3,
                                 submit: bool = True, monitor: bool = True,
                                 quiet_monitoring: bool = True) -> Union[Dict, str, None]:
        """
        Run multi-stage relaxation with 2 or 3 stages.
        
        Args:
            structure: Structure to relax
            base_dir: Base directory for relaxation stages
            structure_name: Name for logging and directories
            n_stages: Number of stages (2 or 3)
            ldau_settings: LDAU parameters
            custom_kspacing: K-point spacing
            submit: Whether to submit jobs
            monitor: Whether to monitor jobs
            quiet_monitoring: Use quiet monitoring
            
        Returns:
            Dict with final relaxed structure and energy, job ID, or None
        """
        if n_stages not in [2, 3]:
            raise ValueError("n_stages must be 2 or 3")
        
        # Select stages based on n_stages
        if n_stages == 2:
            stage_keys = ['stage1', 'stage3']  # Rough + Final
        else:
            stage_keys = ['stage1', 'stage2', 'stage3']  # All stages
        
        logger.info(f"Running {n_stages}-stage relaxation for {structure_name}")
        
        # Create base directory
        os.makedirs(base_dir, exist_ok=True)
        
        # Create status file for tracking
        status_file = os.path.join(base_dir, "multistage_relax_status.json")
        status = self._load_or_create_status(status_file, n_stages, stage_keys)
        
        # Run each stage in sequence
        current_structure = structure
        final_energy = None
        
        for i, stage_key in enumerate(stage_keys):
            stage_info = self.relax_stages[stage_key]
            stage_dir = os.path.join(base_dir, f"{structure_name}_{stage_info['dir']}")
            
            logger.info(f"Stage {i+1}/{n_stages}: {stage_info['name']}")
            
            # Check if stage already completed
            if stage_key in status['completed_stages']:
                logger.info(f"Stage {stage_key} already completed, loading result...")
                result = self._load_stage_result(stage_dir)
                if result:
                    current_structure = result['structure']
                    if result.get('energy'):
                        final_energy = result['energy']
                    continue
                else:
                    logger.warning(f"Could not load result for completed stage {stage_key}, re-running...")
            
            # Run the stage
            stage_result = self._run_single_relaxation_stage(
                current_structure, stage_dir, stage_info, 
                ldau_settings, custom_kspacing, 
                submit, monitor, quiet_monitoring,
                restart=(i > 0)  # Restart from previous stage if not first
            )
            
            # Handle stage result
            if isinstance(stage_result, dict):
                # Stage completed successfully
                current_structure = stage_result['structure']
                if stage_result.get('energy'):
                    final_energy = stage_result['energy']
                
                # Update status
                status['completed_stages'].append(stage_key)
                status['current_stage'] = None
                status[f'{stage_key}_energy'] = stage_result.get('energy')
                self._save_status(status_file, status)
                
                logger.info(f"Stage {i+1} completed: E = {stage_result.get('energy', 'N/A')} eV")
                
                # Cleanup previous stage if not the last stage
                if self.auto_cleanup and i > 0:
                    prev_stage_key = stage_keys[i-1]
                    prev_stage_dir = os.path.join(base_dir, f"{structure_name}_{self.relax_stages[prev_stage_key]['dir']}")
                    logger.info(f"Cleaning up previous stage: {prev_stage_key}")
                    self.file_manager.cleanup_calculation(prev_stage_dir, keep_essential=True)
                
            elif isinstance(stage_result, str):
                # Got job ID but not monitoring
                logger.info(f"Stage {i+1} submitted with job ID: {stage_result}")
                
                # Update status
                status['current_stage'] = stage_key
                status[f'{stage_key}_job_id'] = stage_result
                self._save_status(status_file, status)
                
                if not monitor:
                    # Return job ID for external monitoring
                    return {
                        'job_id': stage_result,
                        'stage': stage_key,
                        'stage_dir': stage_dir,
                        'status_file': status_file
                    }
            else:
                # Stage failed
                logger.error(f"Stage {i+1} failed")
                status['failed_stage'] = stage_key
                self._save_status(status_file, status)
                return None
        
        # All stages completed successfully
        logger.info(f"{n_stages}-stage relaxation completed successfully")
        
        # Update final status
        status['completed'] = True
        status['final_structure_file'] = os.path.join(base_dir, "final_structure.vasp")
        status['final_energy'] = final_energy
        self._save_status(status_file, status)
        
        # Save final structure
        final_poscar = Poscar(current_structure)
        final_poscar.write_file(status['final_structure_file'])
        
        # Cleanup final stage if requested (but keep essential files)
        if self.auto_cleanup:
            final_stage_key = stage_keys[-1]
            final_stage_dir = os.path.join(base_dir, f"{structure_name}_{self.relax_stages[final_stage_key]['dir']}")
            logger.info(f"Cleaning up final stage files...")
            self.file_manager.cleanup_calculation(final_stage_dir, keep_essential=True)
        
        return {
            'structure': current_structure,
            'energy': final_energy,
            'n_stages': n_stages,
            'stages_completed': status['completed_stages'],
            'status_file': status_file,
            'final_structure_file': status['final_structure_file']
        }
    
    def _run_single_relaxation_stage(self, structure: Structure, stage_dir: str,
                                    stage_info: Dict, ldau_settings: Optional[Dict],
                                    kspacing: float, submit: bool, monitor: bool,
                                    quiet_monitoring: bool, restart: bool = False) -> Union[Dict, str, None]:
        """Run a single relaxation stage."""
        # Create stage directory
        os.makedirs(stage_dir, exist_ok=True)
        
        # Write POSCAR
        poscar = Poscar(structure)
        poscar.write_file(os.path.join(stage_dir, "POSCAR"))
        
        # Create stage-specific INCAR
        incar = self._create_stage_incar(structure, stage_info, ldau_settings, restart)
        incar.write_file(os.path.join(stage_dir, "INCAR"))
        
        # Create KPOINTS with custom spacing
        kpoints = self.input_generator.generate_kpoints(structure, kspacing)
        kpoints.write_file(os.path.join(stage_dir, "KPOINTS"))
        
        # Create POTCAR
        elements = [str(site.specie) for site in structure]
        unique_elements = list(dict.fromkeys(elements))
        self.file_manager.create_potcar(stage_dir, unique_elements)
        
        # Determine nodes based on stage (more nodes for later stages)
        if 'stage1' in stage_info['dir']:
            nodes = 2
        elif 'stage2' in stage_info['dir']:
            nodes = 3
        else:  # stage3
            nodes = 4
        
        # Create job script
        script_path = self.slurm_manager.create_vasp_job_script(
            job_dir=stage_dir,
            job_name=f"relax_{stage_info['dir']}",
            nodes=nodes,
            ntasks_per_node=128,
            walltime="48:00:00",
            auto_restart=False  # Let the multi-stage handle restarts
        )
        
        # Submit job
        if submit:
            job_id = self.slurm_manager.submit_job(script_path, stage_dir)
            
            if not job_id:
                logger.error(f"Failed to submit {stage_info['name']}")
                return None
            
            logger.info(f"{stage_info['name']} submitted with job ID: {job_id}")
            
            # Monitor job if requested
            if monitor:
                success = self.slurm_manager.monitor_job(job_id, quiet=quiet_monitoring)
                
                if success:
                    # Extract results
                    return self._extract_stage_results(stage_dir)
                else:
                    logger.error(f"{stage_info['name']} failed")
                    return None
            else:
                # Return job ID for external monitoring
                return job_id
        else:
            logger.info(f"{stage_info['name']} prepared but not submitted")
            return {'structure': structure, 'energy': None}
    
    def _create_stage_incar(self, structure: Structure, stage_info: Dict,
                           ldau_settings: Optional[Dict], restart: bool = False) -> Incar:
        """Create INCAR for a specific relaxation stage."""
        # Start with base relaxation INCAR
        incar = self.input_generator.create_relax_incar(
            structure, 
            max_iterations=stage_info['incar_override']['NSW'],
            ediffg=stage_info['incar_override']['EDIFFG'],
            ldau_settings=ldau_settings,
            restart=restart
        )
        
        # Apply stage-specific overrides
        for key, value in stage_info['incar_override'].items():
            incar[key] = value
        
        return incar
    
    def _extract_stage_results(self, stage_dir: str) -> Optional[Dict]:
        """Extract structure and energy from completed stage."""
        contcar_path = os.path.join(stage_dir, "CONTCAR")
        outcar_path = os.path.join(stage_dir, "OUTCAR")
        
        # Read structure
        if os.path.exists(contcar_path) and os.path.getsize(contcar_path) > 0:
            try:
                self.file_manager.clean_contcar_elements(contcar_path)
                structure = PoscarOutput.from_file(contcar_path).structure
            except Exception as e:
                logger.error(f"Error reading CONTCAR: {e}")
                return None
        else:
            logger.error(f"No valid CONTCAR found in {stage_dir}")
            return None
        
        # Read energy
        energy = None
        if os.path.exists(outcar_path):
            try:
                outcar = Outcar(outcar_path)
                energy = outcar.final_energy
            except Exception as e:
                logger.warning(f"Could not read energy from OUTCAR: {e}")
        
        return {
            'structure': structure,
            'energy': energy,
            'stage_dir': stage_dir
        }
    
    def _load_stage_result(self, stage_dir: str) -> Optional[Dict]:
        """Load results from a previously completed stage."""
        return self._extract_stage_results(stage_dir)
    
    def _load_or_create_status(self, status_file: str, n_stages: int, stage_keys: List[str]) -> Dict:
        """Load or create status tracking file."""
        if os.path.exists(status_file):
            try:
                with open(status_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Error loading status file: {e}")
        
        # Create new status
        return {
            'n_stages': n_stages,
            'stage_keys': stage_keys,
            'completed_stages': [],
            'current_stage': None,
            'failed_stage': None,
            'completed': False,
            'final_structure_file': None,
            'final_energy': None,
            'timestamp': datetime.now().isoformat()
        }
    
    def _save_status(self, status_file: str, status: Dict) -> None:
        """Save status to file."""
        status['timestamp'] = datetime.now().isoformat()
        with open(status_file, 'w') as f:
            json.dump(status, f, indent=2)
    
    def check_multistage_completion(self, base_dir: str, structure_name: str = "structure") -> Optional[Dict]:
        """
        Check if multi-stage relaxation has completed and extract results.
        
        Args:
            base_dir: Base directory for relaxation stages
            structure_name: Name used for directories
            
        Returns:
            Dict with final results or None if not completed
        """
        status_file = os.path.join(base_dir, "multistage_relax_status.json")
        
        if not os.path.exists(status_file):
            logger.warning(f"No status file found at {status_file}")
            return None
        
        # Load status
        try:
            with open(status_file, 'r') as f:
                status = json.load(f)
        except Exception as e:
            logger.error(f"Error reading status file: {e}")
            return None
        
        # Check if completed
        if not status.get('completed', False):
            logger.info(f"Multi-stage relaxation not yet completed")
            logger.info(f"Completed stages: {status.get('completed_stages', [])}")
            logger.info(f"Current stage: {status.get('current_stage', 'None')}")
            return None
        
        # Extract final results
        final_structure_file = status.get('final_structure_file')
        final_energy = status.get('final_energy')
        
        if final_structure_file and os.path.exists(final_structure_file):
            try:
                final_structure = Structure.from_file(final_structure_file)
                
                logger.info(f"Multi-stage relaxation completed successfully")
                logger.info(f"Final energy: {final_energy} eV")
                logger.info(f"Stages completed: {status['completed_stages']}")
                
                return {
                    'structure': final_structure,
                    'energy': final_energy,
                    'n_stages': status['n_stages'],
                    'stages_completed': status['completed_stages'],
                    'status': status
                }
            except Exception as e:
                logger.error(f"Error reading final structure: {e}")
                return None
        else:
            logger.error(f"Final structure file not found: {final_structure_file}")
            return None


# Convenience function for multi-stage relaxation
def run_multistage_relaxation(structure: Structure, base_dir: str,
                             structure_name: str = "structure",
                             n_stages: int = 2,
                             potcar_path: Optional[str] = None,
                             potcar_mapping: Optional[Dict[str, str]] = None,
                             ldau_settings: Optional[Dict] = None,
                             custom_kspacing: float = 0.3,
                             submit: bool = True, monitor: bool = True,
                             auto_cleanup: bool = True,
                             quiet_monitoring: bool = True) -> Union[Dict, str, None]:
    """
    Convenience function to run multi-stage relaxation.
    
    Args:
        structure: Structure to relax
        base_dir: Base directory for relaxation
        structure_name: Name for logging and directories
        n_stages: Number of stages (2 or 3)
        potcar_path: Path to POTCAR files
        potcar_mapping: Element to POTCAR mapping
        ldau_settings: LDAU parameters
        custom_kspacing: K-point spacing
        submit: Submit jobs to SLURM
        monitor: Monitor job progress
        auto_cleanup: Clean up intermediate files
        quiet_monitoring: Use quiet monitoring
        
    Returns:
        Dict with final structure and energy, or job ID if not monitoring
    """
    # Initialize components
    from file_manager import FileManager
    from vasp_inputs import VASPInputGenerator
    from slurm_manager import SLURMManager
    
    file_manager = FileManager(potcar_path, potcar_mapping)
    input_gen = VASPInputGenerator()
    slurm_manager = SLURMManager()
    
    # Create multi-stage relaxer
    ms_relaxer = MultiStageStructureRelaxer(file_manager, input_gen, slurm_manager, auto_cleanup)
    
    # Run multi-stage relaxation
    return ms_relaxer.run_multistage_relaxation(
        structure, base_dir, structure_name, n_stages,
        ldau_settings, custom_kspacing, submit, monitor, quiet_monitoring
    )
