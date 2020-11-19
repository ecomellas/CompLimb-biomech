
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
dts = ["1.0e-4","0.5e-4"]
roms = [20,10]
ks = [1,2,3,4,5]

for dt in dts:
    for rom in roms:
        
        # compute angles for radius and ulna loads
        radius_phi_min = str(50 - rom)
        radius_phi_max = str(50 + rom)
        ulna_phi_min = str(23 + rom)
        ulna_phi_max = str(23 - rom)
        
        for k in ks:
        
            # define kbio and kmech combinations
            if k==1:
                kbio  = "1.0e-3"
                kmech = "1.0e-2"
                
            elif k==2:
                kbio  = "0"
                kmech = "1.0e-2"
                                
            elif k==3:
                kbio  = "1.0e-3"
                kmech = "5.0e-3"
                                
            elif k==4:
                kbio  = "1.0e-3"
                kmech = "1.5e-2"
                                
            else:
                kbio  = "0.1"
                kmech = "1.0"
    
            # define jobname
            jobname = "dt_"+dt+"__rom_"+str(rom)+"__kbio_"+kbio+"__kmech_"+kmech
            
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
                if "set Radius phi min" in line:
                    line = "set Radius phi min   = "+radius_phi_min+"\n"
                if "set Radius phi max" in line:
                    line = "set Radius phi max   = "+radius_phi_max+"\n"
                if "set Ulna phi min" in line:
                    line = "set Ulna phi min   = "+ulna_phi_min+"\n"
                if "set Ulna phi max" in line:
                    line = "set Ulna phi max   = "+ulna_phi_max+"\n"
                if "set growth rate mech" in line:
                    line = "set growth rate mech = "+kmech+"\n"
                if "set growth rate bio" in line:
                    line = "set growth rate bio = "+kbio+"\n"
                if "set Time step size" in line:
                    line = "set Time step size      = "+dt+"\n"
                print (line, end = ' '),
            x.close()

            # run problem
            os.system("sbatch run-multi-dealii.slurm")
            
            # move back to upper folder
            os.chdir(cwd)

