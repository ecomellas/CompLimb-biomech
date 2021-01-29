 #!/bin/bash   
 # first parameter is the name for directory where the calculation is made
 
echo "                    _                _          _   _ _              "
echo " _ _ _  _ _ _  _ _ (_)_ _  __ _   __| |___ __ _| | (_|_)             "
echo "| '_| || | ' \| ' \| | ' \/ _  | / _  / -_) _  | |_| | |  _   _   _  "
echo "|_|  \_,_|_||_|_||_|_|_||_\__, | \__,_\___\__,_|_(_)_|_| (_) (_) (_) "
echo "                          |___/                                      "


echo "DEAL_II_DIR:               " $DEAL_II_DIR

# Define and print tmpdir, where calculations are made and resultsdir where results will be stored
maindir=$PWD
#maindir=/mnt/c/GitRepos/my-code
tmpdir=$maindir/RESULTS/calcDir_$1
resultdir=$maindir/RESULTS/$1
echo "Main directory            :" $maindir
echo "Directory for calculations:" $tmpdir
echo "Directory for results     :" $resultdir

# change to temporary job directory
mkdir -p $tmpdir
cd $tmpdir
# copy stuff from location where job was submitted to temporary directory
cp -r $maindir/parameters.prm . && echo "Copying parameters input file succeeded" || echo "Copying parameters input file failed"
cp -r $maindir/humerus_mesh.inp . && echo "Copying mesh input file succeeded" || echo "Copying mesh input file failed"

# run code
touch out.log
touch err.log
echo "Start run "
../../complimb-biomech >>out.log 2>err.log
if [ $? -eq 0 ]; then
	echo FEM Code OK
else
	echo $?
	echo FEM Code FAILED
fi

# create folder for output and copy parameter file and results into it
mkdir -p $resultdir
mkdir -p $resultdir/Paraview-files

cp parameters.prm $resultdir/
cp humerus_mesh.inp $resultdir/
cp *.sol $resultdir/
cp solution.* $resultdir/Paraview-files/
cp bcs.* $resultdir/Paraview-files/

# get rid of the temporary job dir
rm -rf $tmpdir
