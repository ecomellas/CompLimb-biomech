
#-----------------------------------------------------------------------------------------
#
# This python script runs the deal.ii consolidation problem based on Franceschini 2006
# To execute, type in console >>"python ./run-multi-calc-NUwin.py"
#
# NOTE: If changes are made to the file run-dealii.sh, remember to define it as executable
#       by typing in the console >>"chmod +x run-dealii.sh"
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
dts = ["0.1","0.05"]
ks = [1,2,3,4,5]

for dt in dts:
	for k in ks:
		# define kbio and kmech combinations
		if k==1:
			kbio  = "1.0e-3"
			kmech = "0.0"
			
		elif k==2:
			kbio  = "0"
			kmech = "1.0e-3"
							
		elif k==3:
			kbio  = "1.0e-3"
			kmech = "1.0e-3"
							
		elif k==4:
			kbio  = "1.0e-3"
			kmech = "1.0e-2"
							
		else:
			kbio  = "1.0e-2"
			kmech = "1.0e-3"

		# define jobname
		jobname = "dt_"+dt+"__kbio_"+kbio+"__kmech_"+kmech
			
		# make changes in parameter file
		x = fileinput.input("parameters.prm", inplace=1)
		for line in x:
			if "set growth rate mech" in line:
				line = "set growth rate mech = "+kmech+"\n"
			if "set growth rate bio" in line:
				line = "set growth rate bio = "+kbio+"\n"
			if "set Time step size" in line:
				line = "set Time step size      = "+dt+"\n"
			print (line, end = ''),
		x.close()

	# start cp fem code without output
	#process = subprocess.Popen("./runPoro.sh " + jobname + "> /dev/null 2>&1", shell=True)
	process = subprocess.Popen("./run-dealii.sh " + jobname, shell=True)

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
