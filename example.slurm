#!/bin/bash
#SBATCH -N 1
#SBATCH -n 16
#SBATCH -J Pacman2D
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH --time=24:00:00
#SBATCH --exclusive
#SBATCH -p general
#SBATCH --mem=20000
work=${SLURM_SUBMIT_DIR}/${SLURM_JOB_ID}
mkdir ${work}
cp /home/mesgarnejad/Code/vfmfe/vfmfe-2fold1-opt ${work}/ 

cd $work
echo Running on host `hostname`
echo Time is `date`
echo Directory is `pwd`
echo ${LD_LIBRARY_PATH}
mpirun -np 16 $work/vfmfe-2fold1-opt prefix=${SLURM_JOB_ID} inputfile=${SLURM_JOB_ID}.in -log_view :${SLURM_JOB_ID}.log_summary -ksp_converged_reason   --use-petsc-dm  -elasticity_system_ksp_type cg -elasticity_system_ksp_converged_reason -elasticity_system_ksp_atol 1e-12 -elasticity_system_ksp_rtol 1e-15 -elasticity_system_ksp_max_it 100000  -elasticity_system_ksp_initial_guess_nonzero -elasticity_system_pc_type ml  -fracture_system_snes_type vinewtonrsls  -fracture_system_snes_converged_reason -fracture_system_snes_linesearch_type basic -fracture_system_ksp_type cg -fracture_system_snes_max_it 500 -fracture_system_ksp_converged_reason -fracture_system_pc_type bjacobi -fracture_system_snes_atol 1e-6 -fracture_system_ksp_atol 1e-12 -fracture_system_ksp_rtol 1e-15 -fracture_system_ksp_max_it 100000  -fracture_system_snes_lag_jacobian 20  --use_elastic_nullspace 
