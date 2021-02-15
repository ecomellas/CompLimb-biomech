
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
#dts = ["0.1","0.05"]
loads = ["50e3","100e3"]
ks = [1,2,3,4]

#for dt in dts:
for load in loads:
    for k in ks:
    
        # define kbio and kmech combinations
        if k==1:
            kbio  = "1.0e-5"
            kmech = "1.0"
        
        elif k==2:
            kbio  = "1.0e-6"
            kmech = "1.0"
        
        elif k==3:
            kbio  = "1.0e-6"
            kmech = "0.1"
        
        elif k==4:
            kbio  = "1.0e-6"
            kmech = "10.0"

        # define jobname
        jobname = "p-bio-func-orig__dt_"+dt+"__kbio_"+kbio+"__kmech_"+kmech+"__load_"+load
        
        # create new folder, copy parameters and slurm files into it
        cwd = os.getcwd()
        os.mkdir(jobname)
        src_file_prm = cwd+"/parameters.prm"
        src_file_msh = cwd+"/humerus_mesh.inp"
        src_file_slurm = cwd+"/run-multi-dealii.slurm"
        dst_path = cwd+"/"+jobname
        shutil.copy2(src_file_prm, dst_path)
        shutil.copy2(src_file_msh, dst_path)
        shutil.copy2(src_file_slurm, dst_path)
        
        # access new folder and make changes in parameter file
        os.chdir(jobname)
        
        x = fileinput.input("parameters.prm", inplace=1)
        for line in x:
            if "set Load value" in line:
                line = "set Load value          = -"+load+"\n"
            if "set growth rate mech" in line:
                line = "set growth rate mech = "+kmech+"\n"
            if "set growth rate bio" in line:
                line = "set growth rate bio = "+kbio+"\n"
#            if "set Time step size" in line:
#                line = "set Time step size      = "+dt+"\n"
            print (line, end = ' '),
        x.close()

        # run problem
        os.system("sbatch run-multi-dealii.slurm")
        
        # move back to upper folder
        os.chdir(cwd)

