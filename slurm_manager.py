#!/usr/bin/env python
"""
Job submission and monitoring utilities for SLURM.
Handles job script creation, submission, and status monitoring.
"""

import os
import subprocess
import time
import logging
from datetime import datetime
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

class SLURMManager:
    """Manages SLURM job submission and monitoring."""
    
    def __init__(self):
        """Initialize SLURM manager."""
        pass
    
    def create_vasp_job_script(self, job_dir: str, job_name: str = "vasp_calc",
                              nodes: int = 5, ntasks_per_node: int = 128,
                              partition: str = "bigmem", qos: str = "normal",
                              walltime: str = "48:00:00", vasp_cmd: str = "vasp",
                              auto_restart: bool = False, 
                              buffer_minutes: int = 30) -> str:
        """
        Create a SLURM job submission script for VASP.
        
        Args:
            job_dir: Directory where job script will be created
            job_name: Name for the SLURM job
            nodes: Number of nodes to request
            ntasks_per_node: Number of tasks per node
            partition: SLURM partition
            qos: Quality of service
            walltime: Wall time in HH:MM:SS format
            vasp_cmd: VASP executable command
            auto_restart: Enable automatic restart before time limit
            buffer_minutes: Time buffer before walltime for restart
            
        Returns:
            Path to created job script
        """
        script_path = os.path.join(job_dir, "job.sh")
        total_tasks = nodes * ntasks_per_node
        
        # Create base script content
        script_content = f"""#!/bin/bash -l
#SBATCH --job-name={job_name}
#SBATCH --nodes={nodes}
#SBATCH --ntasks-per-node={ntasks_per_node}
#####SBATCH --ntasks={total_tasks}
#SBATCH --partition={partition}
#SBATCH --qos={qos}
#SBATCH --time={walltime}

# Load modules (modify according to your system)
module load vasp/6.4.2/standard_vtst199

"""
        
        # Add auto-restart functionality if requested
        if auto_restart:
            script_content += self._generate_auto_restart_code(job_dir, buffer_minutes)
        
        # Add the main VASP execution command
        script_content += f"""
# Run VASP
srun {vasp_cmd} > vasp.out
"""
        
        # Write script to file
        with open(script_path, "w") as f:
            f.write(script_content)
        
        # Make the script executable
        os.chmod(script_path, 0o755)
        
        logger.info(f"Created SLURM script at {script_path}")
        return script_path
    
    def _generate_auto_restart_code(self, job_dir: str, buffer_minutes: int) -> str:
        """
        Generate auto-restart code for job scripts.
        
        Args:
            job_dir: Job directory for restart script
            buffer_minutes: Time buffer before walltime
            
        Returns:
            Auto-restart code as string
        """
        return f"""
# Auto-restart functionality
# Get the job time limit in minutes
TIME_LIMIT_MIN=$(squeue -j $SLURM_JOB_ID -h --Format=TimeLimit | sed 's/\\-/ /g' | awk '{{split($1,a,":"); if (NF>1) print a[1]*60+a[2]; else print a[1]}}')

# Calculate when to stop (with {buffer_minutes} minute buffer)
STOP_TIME=$((TIME_LIMIT_MIN - {buffer_minutes}))

# Function to check progress and restart if needed
check_progress() {{
    # Get elapsed time in minutes
    ELAPSED=$(squeue -j $SLURM_JOB_ID -h --Format=TimeUsed | sed 's/\\-/ /g' | awk '{{split($1,a,":"); if (NF>1) print a[1]*60+a[2]; else print a[1]}}')
    
    # Check if we're approaching the time limit
    if [[ $ELAPSED -ge $STOP_TIME ]]; then
        echo "Approaching time limit. Preparing for restart..."
        
        # Check if calculation is finished by looking for convergence
        CONVERGED=false
        for outcar in $(find . -name "OUTCAR" -not -path "*/\\.*"); do
            if grep -q "reached required accuracy" $outcar; then
                CONVERGED=true
                break
            fi
        done
        
        if [[ "$CONVERGED" == "true" ]]; then
            echo "Calculation appears to have converged. No restart needed."
            exit 0
        fi
        
        # Submit restart job
        echo "Calculation not converged, submitting restart job..."
        cd {job_dir}
        
        # Create restart script (this would be customized for specific calculation type)
        python -c "
import sys
import os
sys.path.append('{os.path.dirname(os.path.abspath(__file__))}')
# Import appropriate restart function based on calculation type
# This would be customized in the actual implementation
print('Restart functionality would be called here')
"
        exit 0
    fi
}}

# Set up trap to check progress every 10 minutes
(while true; do sleep 600; check_progress; done) &
PROGRESS_CHECKER_PID=$!

# Clean up on exit
trap "kill $PROGRESS_CHECKER_PID" EXIT

"""
    
    def submit_job(self, script_path: str, job_dir: str = None) -> Optional[str]:
        """
        Submit a job to SLURM.
        
        Args:
            script_path: Path to job script
            job_dir: Directory to run job from (optional)
            
        Returns:
            Job ID if successful, None otherwise
        """
        try:
            # SIMPLE FIX: Use absolute paths
            script_path = os.path.abspath(script_path)
            if job_dir:
                job_dir = os.path.abspath(job_dir)
            
            # Make sure the script is executable
            os.chmod(script_path, 0o755)
            
            # Set working directory
            cwd = job_dir if job_dir else os.path.dirname(script_path)
            
            # Log the command with absolute paths
            logger.info(f"Running sbatch {script_path} in {cwd}")
            
            # Run sbatch command with absolute path
            result = subprocess.run(['sbatch', script_path], 
                                   capture_output=True, text=True, check=True,
                                   cwd=cwd)
            
            # Get output and error
            output = result.stdout.strip()
            error = result.stderr.strip()
            
            # Check for errors
            if result.returncode != 0 or error:
                logger.error(f"sbatch command failed with return code {result.returncode}")
                if error:
                    logger.error(f"STDERR: {error}")
                return None
            
            # Extract job ID from output
            try:
                job_id = output.split()[-1]
                logger.info(f"Job submitted with ID: {job_id}")
                return job_id
            except IndexError:
                logger.error(f"Failed to extract job ID from output: '{output}'")
                return None
                
        except Exception as e:
            logger.error(f"Error submitting job: {e}")
            return None
    def check_job_status(self, job_id: str) -> str:
        """
        Check status of a SLURM job.
        
        Args:
            job_id: SLURM job ID
            
        Returns:
            Job status string
        """
        try:
            # First try squeue for running jobs
            result = subprocess.run(['squeue', '-j', job_id, '-h', '-o', '%t'], 
                                   capture_output=True, text=True)
            
            output = result.stdout.strip()
            
            if output:
                status = output.upper()
                return status
            else:
                # Job not in queue, check if it completed
                sacct_result = subprocess.run(['sacct', '-j', job_id, '-n', '-o', 'State'], 
                                            capture_output=True, text=True)
                
                sacct_output = sacct_result.stdout.strip().split('\n')[0].strip()
                
                if 'COMPLETED' in sacct_output:
                    return "COMPLETED"
                elif 'FAILED' in sacct_output:
                    return "FAILED"
                elif 'CANCELLED' in sacct_output:
                    return "CANCELLED"
                else:
                    return "UNKNOWN"
        
        except Exception as e:
            logger.error(f"Error checking job status: {e}")
            return "ERROR"
    
    def monitor_job(self, job_id: str, check_interval: int = 600, 
                   max_runtime: int = 48*3600, quiet: bool = False) -> bool:
        """
        Monitor a job until completion or timeout.
        
        Args:
            job_id: SLURM job ID
            check_interval: Seconds between status checks
            max_runtime: Maximum runtime in seconds
            quiet: Use quiet monitoring (less console output)
            
        Returns:
            True if job completed successfully, False otherwise
        """
        if not job_id:
            logger.error("No job ID provided for monitoring")
            return False
        
        logger.info(f"Starting monitoring of job {job_id}")
        start_time = time.time()
        last_status = None
        status_count = 0
        
        while (time.time() - start_time) < max_runtime:
            status = self.check_job_status(job_id)
            
            if status in ["COMPLETED", "FAILED", "CANCELLED", "ERROR"]:
                logger.info(f"Job {job_id} final status: {status}")
                return status == "COMPLETED"
            
            # Log status periodically
            current_runtime = (time.time() - start_time) / 3600
            if status != last_status:
                # Status changed
                msg = f"Job {job_id} status changed: {last_status} -> {status}, runtime: {current_runtime:.1f} hours"
                logger.info(msg)
                last_status = status
                status_count = 0
            else:
                # Same status
                status_count += 1
                if not quiet or status_count % 10 == 0:  # Show update every 10 checks if quiet
                    logger.info(f"Job {job_id} status: {status}, runtime: {current_runtime:.1f} hours")
            
            time.sleep(check_interval)
        
        logger.warning(f"Job monitoring timed out after {max_runtime/3600} hours")
        return False
    
    def cancel_job(self, job_id: str) -> bool:
        """
        Cancel a SLURM job.
        
        Args:
            job_id: SLURM job ID
            
        Returns:
            True if successful, False otherwise
        """
        try:
            result = subprocess.run(['scancel', job_id], 
                                   capture_output=True, text=True, check=True)
            logger.info(f"Cancelled job {job_id}")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Error cancelling job {job_id}: {e}")
            return False
    
    def get_job_info(self, job_id: str) -> Dict[str, Any]:
        """
        Get detailed information about a job.
        
        Args:
            job_id: SLURM job ID
            
        Returns:
            Dictionary with job information
        """
        info = {}
        
        try:
            # Get job info from sacct
            fields = "JobID,State,Start,End,Elapsed,NodeList,NNodes,NCPUS"
            result = subprocess.run(['sacct', '-j', job_id, '-n', '-o', fields], 
                                   capture_output=True, text=True)
            
            if result.stdout:
                lines = result.stdout.strip().split('\n')
                if lines:
                    fields_list = fields.split(',')
                    values = lines[0].split()
                    for field, value in zip(fields_list, values):
                        info[field.lower()] = value
            
        except Exception as e:
            logger.error(f"Error getting job info: {e}")
        
        return info
