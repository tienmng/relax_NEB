#!/usr/bin/env python
"""
Multi-stage NEB implementation for VASP calculations.
Provides flexible configuration with 2, 3, or 5 stages for optimal balance between speed and accuracy.
"""
import sys
import os
import shutil
import logging
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from pymatgen.core import Structure
from pymatgen.io.vasp.inputs import Poscar, Incar, Kpoints
from pymatgen.io.vasp.outputs import Outcar

module_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, module_dir)

# Import local modules
from relax_NEB.structure_analyzer import StructureAnalyzer
from relax_NEB.vasp_inputs import VASPInputGenerator
from relax_NEB.file_manager import FileManager
from relax_NEB.slurm_manager import SLURMManager
from relax_NEB.energy_analyzer import NEBEnergyAnalyzer

logger = logging.getLogger(__name__)

class MultiStageNEB:
    """
    Multi-stage NEB implementation with flexible stage configuration.
    Supports 2, 3, or 5 stages for different accuracy/speed tradeoffs.
    """
    
    def __init__(self, base_dir: str, potcar_path: str = None,
                 potcar_mapping: Dict[str, str] = None, auto_cleanup: bool = True):
        """
        Initialize multi-stage NEB.
        
        Args:
            base_dir: Base directory for calculations
            potcar_path: Path to VASP pseudopotentials
            potcar_mapping: Mapping of elements to POTCAR variants
            auto_cleanup: Automatically clean up large files after completion
        """
        self.base_dir = os.path.abspath(base_dir)
        self.auto_cleanup = auto_cleanup
        
        # Initialize component classes
        self.analyzer = StructureAnalyzer()
        self.input_generator = VASPInputGenerator()
        self.file_manager = FileManager(potcar_path, potcar_mapping)
        self.slurm_manager = SLURMManager()
        self.energy_analyzer = NEBEnergyAnalyzer()
        
        # Track calculation state
        self.initial_struct = None
        self.final_struct = None
        self.n_images = None
        self.moving_atom_idx = None
        self.ldau_settings = None
        
        # Define stage configurations
        self.stages = self._define_stages()
        self.current_stages = None
        
        # Create base directory
        os.makedirs(self.base_dir, exist_ok=True)
        logger.info(f"Initialized multi-stage NEB in {self.base_dir}")
    
    def _define_stages(self) -> Dict[str, Dict]:
        """Define the default stage configurations."""
        return {
            'stage1': {
                'name': 'Rough Optimization',
                'description': 'Initial path finding with loose convergence',
                'dir': 'neb_stage1',
                'incar_override': {
                    # NEB specific settings
                    'IMAGES': None,  # Will be set based on n_images
                    'LCLIMB': False,  # No climbing image in stage 1
                    'ICHAIN': 0,
                    'SPRING': 5,      # Positive spring constant for stage 1
                    'IBRION': 1,      # Use RMM-DIIS quasi-Newton
                    'POTIM': 0.2,
                    'IOPT': 0,
                    'EDIFFG': -0.5,   # Loose force convergence
                    'NSW': 20,        # Fewer steps for rough optimization
                    'ISIF': 2,        # Relax ions only
                    
                    # Electronic structure settings
                    'EDIFF': 1E-4,
                    'NELM': 20,
                    'ALGO': 'Fast',
                    'PREC': 'Normal',
                    'LSCALAPACK': True,
                    'LSCALU' : False,
                    'NCORE': 16,
                    'ISMEAR': 0,
                    'SIGMA': 0.05,
                    'LASPH': True,
                    'LREAL': 'Auto',
                    'LMAXMIX': 6,
                    'ENCUT': 520,
                    
                    # Large system parallelization
                    #'LPLANE': True,
                    #'NPAR': 1,        # Conservative for stage 1
                    #'NSIM': 4
                }
            },
            'stage2': {
                'name': 'Climbing Image',
                'description': 'Add climbing image with intermediate convergence',
                'dir': 'neb_stage2',
                'incar_override': {
                    # NEB specific settings
                    'IMAGES': None,   # Will be set based on n_images
                    'LCLIMB': True,   # Enable climbing image
                    'ICHAIN': 0,
                    'SPRING': -5,     # Negative spring for climbing image
                    'IBRION': 1,      # Use RMM-DIIS quasi-Newton
                    'POTIM': 0.1,     # Slightly larger step size
                    'IOPT': 0,
                    'EDIFFG': -0.05,  # Intermediate force convergence
                    'NSW': 100,       # More steps for climbing image
                    'ISIF': 2,        # Relax ions only
                    
                    # Electronic structure settings
                    'EDIFF': 1E-5,
                    'NELM': 100,
                    'ALGO': 'Fast',
                    'PREC': 'Accurate',  # Higher precision for stage 2
                    'LSCALAPACK': True,
                    'LSCALU' : False,
                    'NCORE': 16,
                    'ISMEAR': 0,
                    'SIGMA': 0.05,
                    'LASPH': True,
                    'LREAL': 'Auto',
                    'LMAXMIX': 6,
                    'ENCUT': 520,
                    
                    # Large system parallelization
                    #'LPLANE': True,
                    #'NPAR': 2,        # Can increase for stage 2
                    #'NSIM': 4
                }
            },
            'stage3': {
                'name': 'Final Convergence',
                'description': 'Tight convergence for final barrier',
                'dir': 'neb_stage3',
                'incar_override': {
                    # NEB specific settings
                    'IMAGES': None,   # Will be set based on n_images
                    'LCLIMB': True,   # Enable climbing image
                    'ICHAIN': 0,
                    'SPRING': -5,     # Negative spring for climbing image
                    'IBRION': 1,      # Use DIIS quasi-Newton
                    'POTIM': 0.05,    # Small step size for final convergence
                    'IOPT': 0,
                    'EDIFFG': -0.01,  # Tight force convergence
                    'NSW': 200,       # Maximum ionic steps
                    'ISIF': 2,        # Relax ions only
                    
                    # Electronic structure settings
                    'EDIFF': 1E-6,
                    'NELM': 199,
                    'ALGO': 'Fast',   # Fast algorithm for final stage
                    'PREC': 'Normal',
                    'LSCALAPACK': True,
                    'LSCALU' : False,
                    'NCORE': 16,
                    'ISMEAR': 0,
                    'SIGMA': 0.05,
                    'LASPH': True,
                    'LREAL': 'Auto',
                    'LMAXMIX': 6,
                    'ENCUT': 600,
                    
                    # Large system parallelization
                    #'LPLANE': True,
                    #'NPAR': 4,        # Most aggressive for final stage
                    #'NSIM': 4
                }
            },
            'stage4': {
                'name': 'Ultra-Fine Convergence',
                'description': 'Ultra-tight convergence for publication quality',
                'dir': 'neb_stage4',
                'incar_override': {
                    # NEB specific settings
                    'IMAGES': None,
                    'LCLIMB': True,
                    'ICHAIN': 0,
                    'SPRING': -5,
                    'IBRION': 1,
                    'POTIM': 0.02,    # Very small step size
                    'IOPT': 0,
                    'EDIFFG': -0.005, # Ultra-tight force convergence
                    'NSW': 300,
                    'ISIF': 2,
                    
                    # Electronic structure settings
                    'EDIFF': 1E-7,    # Tighter electronic convergence
                    'NELM': 199,
                    'ALGO': 'Normal',
                    'PREC': 'Accurate',
                    'LSCALAPACK': False,
                    'NCORE': 16,
                    'ISMEAR': 0,
                    'SIGMA': 0.05,
                    'LASPH': True,
                    'LREAL': 'Auto',
                    'LMAXMIX': 6,
                    'ENCUT': 600,
                    
                    # Large system parallelization
                    'LPLANE': True,
                    'NPAR': 4,
                    'NSIM': 4
                }
            },
            'stage5': {
                'name': 'Production Quality',
                'description': 'Production-level convergence with maximum accuracy',
                'dir': 'neb_stage5',
                'incar_override': {
                    # NEB specific settings
                    'IMAGES': None,
                    'LCLIMB': True,
                    'ICHAIN': 0,
                    'SPRING': -5,
                    'IBRION': 1,
                    'POTIM': 0.01,    # Extremely small step size
                    'IOPT': 0,
                    'EDIFFG': -0.002, # Production-level force convergence
                    'NSW': 400,
                    'ISIF': 2,
                    
                    # Electronic structure settings
                    'EDIFF': 1E-8,    # Very tight electronic convergence
                    'NELM': 199,
                    'ALGO': 'Normal',
                    'PREC': 'Accurate',
                    'LSCALAPACK': False,
                    'NCORE': 16,
                    'ISMEAR': 0,
                    'SIGMA': 0.05,
                    'LASPH': True,
                    'LREAL': 'Auto',
                    'LMAXMIX': 6,
                    'ENCUT': 700,     # Higher cutoff for final accuracy
                    
                    # Large system parallelization
                    'LPLANE': True,
                    'NPAR': 4,
                    'NSIM': 4
                }
            }
        }
    
    def _select_stages(self, n_stages: int) -> List[str]:
        """
        Select which stages to use based on n_stages.
        
        Args:
            n_stages: Number of stages (2, 3, or 5)
            
        Returns:
            List of stage keys to use
        """
        if n_stages == 2:
            return ['stage1', 'stage3']  # Rough → Final
        elif n_stages == 3:
            return ['stage1', 'stage2', 'stage3']  # Rough → Climbing → Final
        elif n_stages == 5:
            return ['stage1', 'stage2', 'stage3', 'stage4', 'stage5']  # All stages
        else:
            raise ValueError(f"Unsupported number of stages: {n_stages}. Use 2, 3, or 5.")
    
    def setup_multistage_neb(self, initial_struct: Structure, final_struct: Structure,
                            n_images: int = 5, n_stages: int = 3,
                            moving_atom_idx: int = None,
                            ldau_settings: Dict = None,
                            existing_neb_dir: str = None) -> bool:
        """
        Set up multi-stage NEB calculation.
        
        Args:
            initial_struct: Initial structure
            final_struct: Final structure
            n_images: Number of NEB images
            n_stages: Number of stages (2, 3, or 5)
            moving_atom_idx: Index of moving atom
            ldau_settings: LDAU parameters
            remove_constraints_after_ase: Remove ASE constraints after setup
            
        Returns:
            True if setup successful
        """
        logger.info(f"Setting up {n_stages}-stage NEB with {n_images} images")
        
        # Store parameters
        self.initial_struct = initial_struct
        self.final_struct = final_struct
        self.n_images = n_images
        self.moving_atom_idx = moving_atom_idx
        self.ldau_settings = ldau_settings
        
        # Store the existing NEB path directory
        self.neb_path_dir = existing_neb_dir or "./neb_calculation"

        # Check if existing NEB path exists
        if not os.path.exists(self.neb_path_dir):
            logger.error(f"NEB path directory {self.neb_path_dir} does not exist")
            return False

        # Verify that image directories exist
        missing_dirs = []
        for i in range(n_images + 2):
            img_dir = os.path.join(self.neb_path_dir, f"{i:02d}")
            if not os.path.exists(img_dir) or not os.path.exists(os.path.join(img_dir, "POSCAR")):
                missing_dirs.append(f"{i:02d}")

        if missing_dirs:
            logger.error(f"Missing NEB image directories: {missing_dirs}")
            return False

        logger.info(f"Found existing NEB path in {self.neb_path_dir}")

        # Select stages to use
        self.current_stages = self._select_stages(n_stages)
        
        # Set up each stage
        for stage_key in self.current_stages:
            success = self._setup_stage(stage_key)
            if not success:
                logger.error(f"Failed to set up {stage_key}")
                return False
        
        logger.info(f"Multi-stage NEB setup complete")
        return True
    
    
    def _setup_stage(self, stage_key: str) -> bool:
        """Set up a specific NEB stage."""
        stage_info = self.stages[stage_key]
        stage_dir = os.path.join(self.base_dir, stage_info['dir'])
        
        logger.info(f"Setting up {stage_info['name']}")
        
        # Create stage directory
        os.makedirs(stage_dir, exist_ok=True)
        
        # Get source for structure files
        # Get source for structure files
        if stage_key == self.current_stages[0]:
            # First stage uses existing NEB path
            source_dir = self.neb_path_dir
        else:
            # Subsequent stages use CONTCAR from previous stage
            prev_stage_idx = self.current_stages.index(stage_key) - 1
            prev_stage_key = self.current_stages[prev_stage_idx]
            prev_stage_dir = os.path.join(self.base_dir, self.stages[prev_stage_key]['dir'])
            source_dir = prev_stage_dir
        
        # Set up image directories
        for i in range(self.n_images + 2):
            img_dir = os.path.join(stage_dir, f"{i:02d}")
            os.makedirs(img_dir, exist_ok=True)
            
            # Copy/create POSCAR
            # Copy/create POSCAR
            if stage_key == self.current_stages[0]:
                # Use existing NEB path
                src_poscar = os.path.join(source_dir, f"{i:02d}", "POSCAR")
                if os.path.exists(src_poscar):
                    shutil.copy(src_poscar, os.path.join(img_dir, "POSCAR"))
                    logger.debug(f"Copied POSCAR for image {i:02d}")
                else:
                    logger.error(f"Missing POSCAR in source directory for image {i:02d}")
                    return False
            else:
                # Use CONTCAR from previous stage if available
                src_contcar = os.path.join(source_dir, f"{i:02d}", "CONTCAR")
                src_poscar = os.path.join(source_dir, f"{i:02d}", "POSCAR")
                
                if os.path.exists(src_contcar) and os.path.getsize(src_contcar) > 0:
                    # Clean CONTCAR and copy as POSCAR
                    temp_contcar = os.path.join(img_dir, "temp_CONTCAR")
                    shutil.copy(src_contcar, temp_contcar)
                    self.file_manager.clean_contcar_elements(temp_contcar)
                    shutil.move(temp_contcar, os.path.join(img_dir, "POSCAR"))
                elif os.path.exists(src_poscar):
                    shutil.copy(src_poscar, os.path.join(img_dir, "POSCAR"))
            
            # Create INCAR
            incar = self._create_stage_incar(stage_key, img_dir)
            incar.write_file(os.path.join(img_dir, "INCAR"))
            # Create main INCAR in stage directory with same settings as images
            main_incar = self._create_stage_incar(stage_key, stage_dir)
            main_incar.write_file(os.path.join(stage_dir, "INCAR"))
            # Create KPOINTS
            kpoints = self.input_generator.generate_kpoints(self.initial_struct)
            kpoints.write_file(os.path.join(img_dir, "KPOINTS"))
            
            # Create POTCAR
            elements = [str(site.specie) for site in self.initial_struct]
            unique_elements = list(dict.fromkeys(elements))
            self.file_manager.create_potcar(img_dir, unique_elements)
        
        # Create job script for the stage
        self._create_stage_job_script(stage_key)
        
        return True
    
    def _create_stage_incar(self, stage_key: str, img_dir: str) -> Incar:
        """Create INCAR for a specific stage and image."""
        # Get base NEB INCAR
        incar = self.input_generator.create_neb_incar(
            self.initial_struct,
            n_images=self.n_images,
            ldau_settings=self.ldau_settings
        )
        
        # Apply stage-specific overrides
        stage_overrides = self.stages[stage_key]['incar_override'].copy()
        stage_overrides['IMAGES'] = self.n_images
        
        for key, value in stage_overrides.items():
            if value is not None:
                incar[key] = value
        
        return incar
    
    def _create_stage_job_script(self, stage_key: str):
        """Create SLURM job script for a stage."""
        stage_info = self.stages[stage_key]
        stage_dir = os.path.join(self.base_dir, stage_info['dir'])
        
        # Adjust resources based on stage
        if stage_key == 'stage1':
            nodes, walltime = 5, "24:00:00"  # Fast stage
        elif stage_key in ['stage2', 'stage3']:
            nodes, walltime = 5, "48:00:00"  # Standard
        else:
            nodes, walltime = 5, "48:00:00"  # Long for fine stages
        
        script_path = self.slurm_manager.create_vasp_job_script(
            job_dir=stage_dir,
            job_name=f"neb_{stage_key}",
            nodes=nodes,
            walltime=walltime,
            auto_restart=True
        )
        
        return script_path
    
    def run_stage(self, stage_key: str, submit: bool = True, 
                  monitor: bool = True, quiet: bool = False) -> bool:
        """
        Run a specific NEB stage.
        
        Args:
            stage_key: Stage to run
            submit: Submit job to queue
            monitor: Monitor job completion
            quiet: Use quiet monitoring
            
        Returns:
            True if stage completed successfully
        """
        stage_info = self.stages[stage_key]
        stage_dir = os.path.join(self.base_dir, stage_info['dir'])
        
        logger.info(f"Running {stage_info['name']}")
        
        if not submit:
            logger.info(f"Job files prepared in {stage_dir}")
            return True
        
        # Submit job
        job_script = os.path.join(stage_dir, "job.sh")
        job_id = self.slurm_manager.submit_job(job_script, stage_dir)
        
        if not job_id:
            logger.error(f"Failed to submit {stage_key}")
            return False
        
        if monitor:
            # Monitor job completion
            success = self.slurm_manager.monitor_job(job_id, quiet=quiet)
            
            if success:
                logger.info(f"{stage_info['name']} completed successfully")
                
                # Cleanup if requested
                if self.auto_cleanup:
                    self.file_manager.cleanup_calculation(stage_dir)
            else:
                logger.error(f"{stage_info['name']} failed")
                return False
        
        return True
    
    def run_all_stages(self, submit: bool = True, monitor: bool = True,
                      quiet: bool = False) -> bool:
        """
        Run all configured stages in sequence.
        
        Args:
            submit: Submit jobs to queue
            monitor: Monitor job completion
            quiet: Use quiet monitoring
            
        Returns:
            True if all stages completed successfully
        """
        if not self.current_stages:
            logger.error("No stages configured. Run setup_multistage_neb first.")
            return False
        
        logger.info(f"Running {len(self.current_stages)}-stage NEB")
        
        for i, stage_key in enumerate(self.current_stages):
            logger.info(f"Starting stage {i+1}/{len(self.current_stages)}: {stage_key}")
            
            success = self.run_stage(stage_key, submit=submit, monitor=monitor, quiet=quiet)
            
            if not success:
                logger.error(f"Stage {stage_key} failed. Stopping workflow.")
                return False
        
        logger.info("All stages completed successfully")
        return True

    def analyze_final_results(self) -> Optional[Dict]:
        """
        Analyze final NEB results.
        
        Returns:
            Dictionary with analysis results
        """
        if not self.current_stages:
            logger.error("No stages configured")
            return None
        
        # Get final stage
        final_stage_key = self.current_stages[-1]
        final_stage_dir = os.path.join(self.base_dir, self.stages[final_stage_key]['dir'])
        
        logger.info("Analyzing final NEB results")
        
        # Extract energies from final stage
        energies, success = self.energy_analyzer.extract_vasp_energies(
            final_stage_dir, self.n_images
        )
        
        if not success:
            logger.warning("Could not extract all energies, attempting NEBEF.dat")
            distances, energies, success = self.energy_analyzer.extract_nebef_energies(final_stage_dir)
            if not success:
                logger.error("Failed to extract energies")
                return None
        
        # Calculate distances if not available
        if 'distances' not in locals():
            if self.moving_atom_idx is not None:
                distances = self.energy_analyzer.calculate_reaction_path_distances(
                    final_stage_dir, self.n_images, self.moving_atom_idx
                )
            else:
                distances = None
        
        # Analyze energy profile
        analysis = self.energy_analyzer.analyze_energy_profile(
            energies, distances, initial_energy=energies[0] if energies else None
        )
        
        # Create plots
        plot_path = self.energy_analyzer.plot_energy_profile(
            energies, distances,
            title=f"{len(self.current_stages)}-Stage NEB Energy Profile",
            output_path=os.path.join(self.base_dir, "final_energy_profile.png"),
            initial_energy=energies[0] if energies else None
        )
        
        # Save data
        data_path = self.energy_analyzer.save_energy_data(
            energies, distances,
            output_path=os.path.join(self.base_dir, "final_energy_profile.dat"),
            initial_energy=energies[0] if energies else None
        )
        
        analysis.update({
            'plot_path': plot_path,
            'data_path': data_path,
            'final_stage': final_stage_key,
            'n_stages': len(self.current_stages)
        })
        
        logger.info(f"Analysis complete. Barrier: {analysis.get('barrier', 'N/A'):.3f} eV")
        return analysis
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Example usage
    print("Multi-stage NEB examples:")
    print("1. 2-stage NEB (fastest)")
    print("2. 3-stage NEB (balanced)")
    print("3. 5-stage NEB (highest accuracy)")
    print("See function examples for usage details.")
            
