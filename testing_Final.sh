#$ -l tmem=2G  
#$ -l h_vmem=2G
#$ -l h_rt=1:0:0
#$ -S /bin/bash
#$ -j y
#$ -N test_fast
#$ -o ~/artificial_selection
#$ -pe mpi 21
#$ -R y
cd ~/artificial_selection
export PATH="/share/apps/openmpi-4.0.0/bin:$PATH"
export PATH="/share/apps/openmpi-4.0.0/lib:$PATH"
source /share/apps/source_files/python/python-3.9.5.source 
hostname
date
mpiexec -n 20 python3 -m mpi4py.futures Final.py
date
