
#-------------------------------------------------------------------------------
#
# This python script runs the deal.ii CompLimb problem
# on Northeastern's Discovery cluster. Python v3.0 or higher required.
#
# To execute, type in Terminal >>"python3 ./run-multi-calc-Discovery.py"
#
#-------------------------------------------------------------------------------
import fileinput
import sys
import os
import shutil
import select
import time


#-------------------------------------------------------------------------------
# Modify parameters to study sensitivity
#-------------------------------------------------------------------------------
gammas = ["1.0"]
kbios = ["5.0e-3"]
kmechs = ["1.5e-3","5.0e-3"]

for gamma in gammas:
    for kbio in kbios:
        for kmech in kmechs:
            # define jobname
            jobname = "kbio_"+kbio+"__kmech_"+kmech+"__gamma_"+gamma
            
            # create new folder, copy parameters and slurm files into it
            cwd = os.getcwd()
            os.mkdir(jobname)
            src_file_prm = cwd+"/parameters.prm"
            src_file_slurm = cwd+"/run-multi-dealii.slurm"
            dst_path = cwd+"/"+jobname
            shutil.copy2(src_file_prm, dst_path)
            shutil.copy2(src_file_slurm, dst_path)
            
            # access new folder and make changes in parameter file
            os.chdir(jobname)
            
            x = fileinput.input("parameters.prm", inplace=1)
            for line in x:
                if "set growth rate mech" in line:
                    line = "  set growth rate mech = "+kmech+"\n"
                if "set growth exponential mech" in line:
                    line = "  set growth exponential mech = "+gamma+"\n"
                if "set growth rate bio" in line:
                    line = "  set growth rate bio = "+kbio+"\n"
                print (line, end = ' '),
            x.close()

            # run problem
            os.system("sbatch run-multi-dealii.slurm")
            
            # move back to upper folder
            os.chdir(cwd)
