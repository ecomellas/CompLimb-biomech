#!/bin/bash
#SBATCH -N 1
#SBATCH -n 8
#SBATCH -J CompLimb
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH --time=6:00:00
#SBATCH -p short
#SBATCH --mem=4Gb
#SBATCH --constraint=broadwell

work=${SLURM_SUBMIT_DIR}/${SLURM_JOB_ID}

mkdir ${work}
cp parameters.prm ${work}
cp humerus_mesh.inp ${work}
cd $work

echo Running on host `hostname`
echo Time is `date`
echo Directory is `pwd`
echo ${LD_LIBRARY_PATH}

module load gcc/7.3.0
module load openmpi/4.0.3-skylake-gcc7.3
module load cmake-3.17.3-gcc-7.3.0-ed5iorn
spack load dealii
mpirun -np 8 ../../complimb-biomech
