#!/bin/bash
#SBATCH -N 1
#SBATCH -n 12
#SBATCH -J CompLimb
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH --time=15:00:00
#SBATCH -p short
#SBATCH --mem=15Gb
#SBATCH --constraint=E5-2680v4@2.40GHz

work=${SLURM_SUBMIT_DIR}/${SLURM_JOB_ID}

mkdir ${work}
cp parameters.prm ${work}
cd $work

echo Running on host `hostname`
echo Time is `date`
echo Directory is `pwd`
echo ${LD_LIBRARY_PATH}

module load gcc/7.3.0
module load openmpi/4.0.3-skylake-gcc7.3
module load cmake-3.17.3-gcc-7.3.0-ed5iorn
spack load dealii
mpirun -np 12 ../../complimb-biomech
