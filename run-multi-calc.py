
#-----------------------------------------------------------------------------------------
#
# This python script runs the deal.ii consolidation problem based on Franceschini 2006
# To execute, type in console >>"python ./run-multi-calc.py"
#
#   *** remember to load deal.ii and mpi before executing the script***
# >> "spack load dealii@develop"
# >> "spack load openmpi"
#
# NOTE: If changes are made to the file runPoro.sh, remember to define it as executable
#       by typing in the console >>"chmod +x runPoro.sh"
#
#-----------------------------------------------------------------------------------------
import fileinput
import sys
import subprocess
import os
import shutil
import select
import time

# Number of parallel processes, i.e. how many jobs are run in parallel.
# Since the deal.ii code is executed already in parallel (-np 8), it doesn't make sense
# to execute any jobs in parallel (unless on a server like kerberos). So , parallelProcesses = 1 should be used.
parallelProcesses = 1
processes = {}
jobnames = {}

#---------------------------------------------------------------------------------------
# Step loading problem with different kappa values
#---------------------------------------------------------------------------------------
gammas = ["1.0","5.0","0.5"]
kbios = ["5.0e-3","1.0e-3","5.0e-2"]
kps = ["1.5e-3","5.0e-3","2.5e-3"]

for gamma in gammas:
	for kbio in kbios:
		for kp in kps:
			# Jobname
			jobname = "phi10-80_1cycle_dt1e-4_kg"+kbio+"_kp"+kp+"_gamma"+gamma
			# make changes in parameter file
			x = fileinput.input("parameters.prm", inplace=1)
			for line in x:
				if "set growth rate pressure" in line:
					line = "  set growth rate pressure = "+kp+"\n"
				if "set growth exponential pressure" in line:
					line = "  set growth exponential pressure = "+gamma+"\n"
				if "set growth rate bio" in line:
					line = "  set growth rate bio = "+kbio+"\n"
				print (line),
			x.close()

			# start cp fem code without output
			#process = subprocess.Popen("./runPoro.sh " + jobname + "> /dev/null 2>&1", shell=True)
			process = subprocess.Popen("./runPoro.sh " + jobname, shell=True)

			# check if Input folder is copied
			# to make sure I look for the executable which is copied afterwards
			executable = "RESULTS/calcDir_" + jobname 
			results = "RESULTS/resultDir_" + jobname
			time.sleep(1)
			while not os.path.exists(executable) and not os.path.exists(results):
				time.sleep(1)
				print ("waiting for executable to be copied")
			
			# store process to wait for it later
			processes[process.pid] = process
			jobnames[process.pid] = jobname

			# if more than parallelProcesses running, wait for the first to finish
			while len(processes) >= parallelProcesses:
				pid, status = os.wait()
				if pid in processes:
					if status == 0:
						print ("Job %30s successful" % jobnames[pid])
					del processes[pid]
					del jobnames[pid]

# wait for the other processes
while processes:
	pid, status = os.wait()
	if pid in processes:
		if status == 0:
			print ("Job %30s successful" % jobnames[pid])
		del processes[pid]
		del jobnames[pid]

print ("   _   _ _     _     _                           _     _          _ _ ")
print ("  /_\ | | |   (_)___| |__ ___  __ ___ _ __  _ __| |___| |_ ___ __| | |")
print (" / _ \| | |   | / _ \ '_ (_-< / _/ _ \ '  \| '_ \ / -_)  _/ -_) _  |_|")
print ("/_/ \_\_|_|  _/ \___/_.__/__/ \__\___/_|_|_| .__/_\___|\__\___\__,_(_)")
print ("            |__/                           |_|                        ")
