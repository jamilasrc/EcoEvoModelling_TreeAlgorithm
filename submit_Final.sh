# community selection test
#$ -S /bin/bash
#$ -l h_rt=48:0:0
#$ -l h_vmem=5.0G,tmem=5.0G
#$ -cwd
#$ -j y
#$ -N test
#$ -pe mpi 10
#$ -R y
# Commands
module load python/3.8.5
/share/apps/mpich-3.2.1/bin/mpiexec -n $NSLOTS python3 -m mpi4py.futures Final.py